"""
dkvmn_encoder.py -- key-value memory backbone for the deep_irt.

A lightweight subclass of ``BaseSeqEncoder`` (see encoder.py) implementing a
Dynamic Key-Value Memory Network (Zhang et al., 2017) as the sequence core.  It
inherits the entire interface -- the embedding tables, ``theta_proj``, the
single-shift alignment, and every public accessor -- unchanged, so the decoder
(including the decoupled alpha key in the base) composes with no change.  This
is the swappability contract.

Memory
------
A static key memory ``Mk`` (memory_size x key_dim) is shared across sequences; a
dynamic value memory ``Mv`` (memory_size x hidden_dim) starts from a learned
init and is updated step by step.  Addressing reads the current item's key and
softmaxes it against ``Mk`` to get attention weights over slots.  A write erases
and adds at the addressed slots; a read pools ``Mv`` by the same weights.

Per-step hidden
---------------
    key_t  = key_proj(item_emb(q_t))                         (key_dim)
    w_t    = softmax(key_t @ Mk.T)                           (memory_size)
    v_t    = value_proj([item_emb(q_t), resp_emb(r_t)])      (hidden_dim)
    Mv    <- erase/add update at slots weighted by w_t
    read_t = w_t @ Mv                                        (hidden_dim)
    h_t    = tanh(summary_proj([read_t, key_t]))             (hidden_dim)

``_direct_hidden`` writes (q_t, r_t) BEFORE reading, so h_t sees (q_t, r_t).  The
base single-shift then carries h_{t-1} into the step-t prediction state, so the
predicting state is item-blind for the current step by construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_irt.core.encoder import BaseSeqEncoder


class DKVMNEncoder(BaseSeqEncoder):
    """Key-value memory backbone mapping (item_ids, responses) to per-step theta.

    Parameters
    ----------
    num_items : int
        Total items in the embedding table.
    emb_dim : int
        Item / response embedding dimension.
    hidden_dim : int
        Value-memory width and per-step hidden width.  MUST be the hidden width
        so ``state_for_prediction`` returns (B, T, hidden_dim) and the decoder's
        state-conditioned alpha head matches.
    n_cats : int
        Response categories K (responses in {0, ..., K-1}).
    alpha_emb_dim : int or None
        When set, build the SEPARATE wide alpha key (``alpha_item_emb``) feeding
        only the decoder's discrimination head, never the memory.  Built LAST so
        it never perturbs the other inits.
    memory_size : int
        Number of memory slots.
    key_dim : int or None
        Key-memory / addressing width.  None (default) uses ``emb_dim``.
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int = 8,
        hidden_dim: int = 32,
        n_cats: int = 4,
        alpha_emb_dim: int | None = None,
        memory_size: int = 20,
        key_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_cats = n_cats
        self.alpha_emb_dim = alpha_emb_dim
        self.memory_size = memory_size
        self.key_dim = key_dim if key_dim is not None else emb_dim

        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.resp_emb = nn.Embedding(n_cats, emb_dim)

        # Static key memory and learned value-memory init.
        self.key_memory = nn.Parameter(torch.randn(memory_size, self.key_dim) * 0.1)
        self.value_init = nn.Parameter(torch.zeros(memory_size, hidden_dim))
        nn.init.normal_(self.value_init, std=0.1)

        # Addressing, write, erase/add, and read+key summary.
        self.key_proj = nn.Linear(emb_dim, self.key_dim)
        self.value_proj = nn.Linear(2 * emb_dim, hidden_dim)
        self.erase = nn.Linear(hidden_dim, hidden_dim)
        self.add = nn.Linear(hidden_dim, hidden_dim)
        self.summary_proj = nn.Linear(hidden_dim + self.key_dim, hidden_dim)

        self.theta_proj = nn.Linear(hidden_dim, 1)

        # Wide item key for the DECOUPLED alpha head.  Built last and only when
        # requested, so it never perturbs the other inits.  Feeds the decoder's
        # alpha head, never the memory.
        if alpha_emb_dim is not None:
            self.alpha_item_emb = nn.Embedding(num_items, alpha_emb_dim)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _address(self, key_t: torch.Tensor) -> torch.Tensor:
        """Softmax attention over memory slots: (B, memory_size)."""
        logits = key_t @ self.key_memory.t()      # (B, memory_size)
        return F.softmax(logits, dim=-1)

    def _write(
        self, mv: torch.Tensor, w_t: torch.Tensor, v_t: torch.Tensor
    ) -> torch.Tensor:
        """Erase/add update of the value memory at the addressed slots.

        mv : (B, memory_size, hidden) ; w_t : (B, memory_size) ; v_t : (B, hidden)
        """
        e = torch.sigmoid(self.erase(v_t)).unsqueeze(1)   # (B, 1, hidden)
        a = torch.tanh(self.add(v_t)).unsqueeze(1)        # (B, 1, hidden)
        w = w_t.unsqueeze(-1)                             # (B, memory_size, 1)
        mv = mv * (1.0 - w * e) + w * a
        return mv

    def _summary(self, w_t: torch.Tensor, mv: torch.Tensor,
                 key_t: torch.Tensor) -> torch.Tensor:
        """Read by the attention weights, fuse with the key: (B, hidden)."""
        read = torch.bmm(w_t.unsqueeze(1), mv).squeeze(1)   # (B, hidden)
        return torch.tanh(self.summary_proj(torch.cat([read, key_t], dim=-1)))

    def _run(
        self,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """Step through the sequence, returning per-step hidden (B, T, hidden).

        Write (q_t, r_t) then read, so h_t sees the current interaction.
        """
        B, T = item_ids.shape
        keys = self.key_proj(self.item_emb(item_ids))         # (B, T, key_dim)
        values = self.value_proj(
            torch.cat([self.item_emb(item_ids), self.resp_emb(responses)], dim=-1)
        )                                                     # (B, T, hidden)
        mv = self.value_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        out = []
        for t in range(T):
            key_t = keys[:, t]                                # (B, key_dim)
            v_t = values[:, t]                                # (B, hidden)
            w_t = self._address(key_t)                        # (B, memory_size)
            mv = self._write(mv, w_t, v_t)
            out.append(self._summary(w_t, mv, key_t))
        return torch.stack(out, dim=1)                        # (B, T, hidden)

    # ------------------------------------------------------------------
    # Per-step hidden producer (the raw responsive stream)
    # ------------------------------------------------------------------

    def _direct_hidden(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Raw responsive hidden: h_t is a function of (q_{<=t}, r_{<=t}).

        Writes (q_t, r_t) into the value memory BEFORE reading, so the read pools
        the current interaction.

        Returns
        -------
        h : (batch, seq_len, hidden_dim)
        """
        return self._run(item_ids, responses)
