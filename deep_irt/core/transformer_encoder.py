"""
transformer_encoder.py -- causal Transformer backbone for the deep_irt.

A lightweight subclass of ``BaseSeqEncoder`` (see encoder.py).  It swaps the
LSTM sequence core for a stack of causal self-attention layers and otherwise
inherits the entire interface -- the embedding tables, ``theta_proj``, the
single-shift alignment, and every public accessor -- unchanged.  The wide item
key (``item_key_emb``) lives in the base, so it composes here with no decoder
change.  This is the swappability contract.

The per-step token is ``x_t = input_proj([item_val_emb(q_t), resp_emb(r_t)])`` of
width ``hidden_dim``, plus a learned positional embedding.  A causal mask keeps
attention strictly to positions <= t, so ``_direct_hidden`` returns h_t seeing
(q_<=t, r_<=t).
"""

import torch
import torch.nn as nn

from deep_irt.core.encoder import BaseSeqEncoder


class TransformerEncoder(BaseSeqEncoder):
    """Causal Transformer backbone mapping (item_ids, responses) to per-step theta.

    Parameters
    ----------
    num_items : int
        Total items in the embedding table.
    emb_dim : int
        Thin item value / response embedding dimension.
    hidden_dim : int
        Token width and attention model dimension (d_model).  The per-step
        hidden width MUST equal this so ``state_for_prediction`` returns
        (B, T, hidden_dim) and the decoder's state-conditioned alpha head
        (Linear(hidden_dim + item_key_dim, 1)) matches.
    n_cats : int
        Response categories K (responses in {0, ..., K-1}).
    item_key_dim : int or None
        When set, build the SEPARATE wide item key (``item_key_emb``) that feeds
        only the decoder's static alpha and beta heads, never the token input.
        Built LAST so it never perturbs the other inits.
    n_heads : int
        Attention heads per layer.
    n_layers : int
        Number of ``nn.TransformerEncoderLayer`` blocks.
    max_seq_len : int
        Positional-embedding table size (sequences must be <= this length).
    dropout : float
        Dropout inside the attention layers.
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int = 8,
        hidden_dim: int = 32,
        n_cats: int = 4,
        item_key_dim: int | None = None,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_cats = n_cats
        self.item_key_dim = item_key_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.item_val_emb = nn.Embedding(num_items, emb_dim)
        self.resp_emb = nn.Embedding(n_cats, emb_dim)

        # Per-step token: [item_val_emb, resp_emb] -> hidden_dim, plus learned position.
        self.input_proj = nn.Linear(2 * emb_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.theta_proj = nn.Linear(hidden_dim, 1)

        # Wide item key for the DECOUPLED static readouts.  Built last and only
        # when requested, so it never perturbs the other inits.  Feeds the
        # decoder's alpha and beta heads, never the token input.
        if item_key_dim is not None:
            self.item_key_emb = nn.Embedding(num_items, item_key_dim)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tokens(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Per-step interaction tokens (B, T, hidden_dim), no positions added."""
        ie = self.item_val_emb(item_ids)          # (B, T, emb_dim)
        re = self.resp_emb(responses)             # (B, T, emb_dim)
        x = torch.cat([ie, re], dim=-1)           # (B, T, 2*emb_dim)
        return self.input_proj(x)                 # (B, T, hidden_dim)

    def _add_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Add the learned positional embedding for positions 0..T-1."""
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return x + self.pos_emb(pos).unsqueeze(0)  # (B, T, hidden_dim)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        """Causal self-attention over the token stream (mask positions > t)."""
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.transformer(x, mask=mask, is_causal=True)

    # ------------------------------------------------------------------
    # Per-step hidden producer (the raw responsive stream)
    # ------------------------------------------------------------------

    def _direct_hidden(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Raw responsive hidden: h_t is a function of (q_{<=t}, r_{<=t}).

        Tokens carry the current interaction (q_t, r_t); causal attention lets
        h_t attend to positions <= t only.

        Returns
        -------
        h : (batch, seq_len, hidden_dim)
        """
        x = self._add_positions(self._tokens(item_ids, responses))
        return self._attend(x)
