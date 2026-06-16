"""
dyn_model.py -- Dynamic-encoder x shared-multi-head decoder model.

Architecture
------------
Replaces jointfmt's static per-respondent nn.Parameter theta with an LSTM
sequence encoder that maps a (item, response) history to a TIME-VARYING
theta_t per learner.

Shared item embedding -> shared DifficultyExtractor -> scalar d_i, read by:
  GPCMHead  : (theta_t, d_i) -> GPCM NLL over ordinal responses
  BTHead    : (d_i, d_j)     -> BT NLL over pairwise outcomes (person-free)

Joint loss = GPCM_NLL + BT_NLL on the SAME item embedding.

Design for the richer pack
--------------------------
- theta interface is general: the encoder's hidden state (a KT-style mastery
  readout) is EXPOSED via get_kt_state(). Internally theta is scalar for v1,
  but nothing hard-wires this -- theta_proj is (hidden_dim -> theta_dim) and
  the heads read theta_dim-sized vectors once that is widened.
- Item embedding input is structured so content side-information could later
  be concatenated: item lookup feeds through item_emb (learned id-based for
  v1) with no hard coupling to the embedding dimension in the heads.
- get_kt_state() returns the raw (hidden, cell) pair from the LSTM so a KT
  consumer can read per-skill mastery readouts without re-running the encoder.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the proven shared mechanism from jointfmt
from deep_irt.jointfmt.joint_model import DifficultyExtractor, GPCMHead, BTHead


# ---------------------------------------------------------------------------
# LSTM Sequence Encoder (dynamic theta)
# ---------------------------------------------------------------------------

class SequenceEncoder(nn.Module):
    """
    LSTM encoder: (item_id, response) history -> per-step theta_t.

    Parameters
    ----------
    num_items  : int
    emb_dim    : int   -- item and response embedding size
    hidden_dim : int   -- LSTM hidden state size
    n_cats     : int   -- number of response categories K
    theta_dim  : int   -- output theta dimensionality (1 = scalar, expandable)

    Exposable KT state
    ------------------
    get_kt_state(item_ids, responses) -> Tuple[Tensor, Tensor]
        Returns the LSTM (hidden, cell) state at the LAST timestep for each
        learner in the batch: shapes (batch, hidden_dim) each.  This is the
        per-skill mastery readout that a KT consumer would inspect.
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int = 16,
        hidden_dim: int = 64,
        n_cats: int = 4,
        theta_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_cats = n_cats
        self.theta_dim = theta_dim

        # Item embeddings: id -> emb_dim vector.
        # Structured for future content concatenation: caller may replace or
        # augment item_emb, keeping emb_dim consistent.
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.resp_emb = nn.Embedding(n_cats, emb_dim)

        self.lstm = nn.LSTM(emb_dim + emb_dim, hidden_dim, batch_first=True)
        self.theta_proj = nn.Linear(hidden_dim, theta_dim)

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a batch of response sequences.

        Parameters
        ----------
        item_ids  : (batch, seq_len)  LongTensor
        responses : (batch, seq_len)  LongTensor in {0, ..., K-1}

        Returns
        -------
        theta_seq : (batch, seq_len, theta_dim)   per-step latent ability
        (h_n, c_n): each (1, batch, hidden_dim)   final LSTM state (the KT readout)
        """
        ie = self.item_emb(item_ids)        # (B, T, emb_dim)
        re = self.resp_emb(responses)       # (B, T, emb_dim)
        x = torch.cat([ie, re], dim=-1)     # (B, T, 2*emb_dim)
        h_seq, (h_n, c_n) = self.lstm(x)   # h_seq: (B, T, hidden)
        theta_seq = self.theta_proj(h_seq)  # (B, T, theta_dim)
        return theta_seq, (h_n, c_n)

    def shifted_theta(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """
        Return theta shifted one step (prepend zero prior, drop last).

        theta_in[t] is the ability estimate BEFORE answering step t, so the
        model predicts response[t] from (theta_in[t], item_id[t]).

        Returns
        -------
        theta_in : (batch, seq_len)   scalar ability per step (theta_dim squeezed)
        """
        theta_seq, _ = self.encode(item_ids, responses)   # (B, T, theta_dim)
        theta_scalar = theta_seq.squeeze(-1)               # (B, T)  for theta_dim=1
        batch = theta_scalar.size(0)
        zeros = torch.zeros(batch, 1, device=theta_scalar.device)
        return torch.cat([zeros, theta_scalar[:, :-1]], dim=1)  # (B, T)

    def get_final_theta(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """
        Return final-timestep scalar theta for each learner in the batch.

        Returns
        -------
        theta_final : (batch,)
        """
        theta_seq, _ = self.encode(item_ids, responses)
        return theta_seq[:, -1, :].squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    # Exposable KT state (the PI-mandated public interface)
    # ------------------------------------------------------------------

    def get_kt_state(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the LSTM (hidden, cell) state at the last timestep.

        This is the raw KT-style mastery readout: the LSTM cell carries
        cumulative "what has this learner seen" information; the hidden
        state is the gated output.  A downstream skill-level consumer
        can apply a skill-projection matrix W to h_n to get per-skill
        mastery scores.

        Parameters
        ----------
        item_ids  : (batch, seq_len)  LongTensor
        responses : (batch, seq_len)  LongTensor

        Returns
        -------
        h_n : (batch, hidden_dim)   LSTM hidden state at final step
        c_n : (batch, hidden_dim)   LSTM cell state at final step
        """
        _, (h_n, c_n) = self.encode(item_ids, responses)
        return h_n.squeeze(0), c_n.squeeze(0)  # remove num_layers dim


# ---------------------------------------------------------------------------
# Dynamic Joint Model
# ---------------------------------------------------------------------------

class DynJointModel(nn.Module):
    """
    Dynamic-encoder x shared-multi-head joint model.

    The LSTM encoder produces per-step theta_t from response history.
    The shared DifficultyExtractor maps item embeddings to a common scalar d_i.
    GPCMHead reads (theta_t, d_i); BTHead reads (d_i, d_j) person-free.

    Parameters
    ----------
    n_items   : int
    emb_dim   : int
    hidden_dim: int   -- LSTM hidden size
    K         : int   -- GPCM response categories
    theta_dim : int   -- encoder output dimension (1 for scalar; future: multidim)
    seed      : int
    """

    def __init__(
        self,
        n_items: int,
        emb_dim: int = 16,
        hidden_dim: int = 64,
        K: int = 4,
        theta_dim: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)

        # Sequence encoder with item embeddings
        self.encoder = SequenceEncoder(
            num_items=n_items,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_cats=K,
            theta_dim=theta_dim,
        )

        # Shared difficulty extractor: item_emb -> scalar d_i
        # Uses the SAME embedding table as the encoder via _get_d()
        self.difficulty_extractor = DifficultyExtractor(emb_dim=emb_dim, hidden_dim=8)

        # Two heads, both reading from shared d_i
        self.gpcm_head = GPCMHead(emb_dim=emb_dim, K=K)
        self.bt_head = BTHead()

        self.n_items = n_items
        self.emb_dim = emb_dim
        self.K = K
        self.theta_dim = theta_dim

    # ------------------------------------------------------------------
    # Item embedding access (shared between encoder and heads)
    # ------------------------------------------------------------------

    def _get_d(self, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (emb, d) for given item_ids using the ENCODER's item embedding.

        Both heads use the encoder's embedding table so that the GPCM and BT
        gradients flow through the same parameters.
        """
        emb = self.encoder.item_emb(item_ids)
        d = self.difficulty_extractor(emb)
        return emb, d

    # ------------------------------------------------------------------
    # GPCM loss (sequences -> per-step theta_t)
    # ------------------------------------------------------------------

    def gpcm_nll(
        self,
        seq_item_ids: torch.Tensor,
        seq_responses: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_responses: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        """
        GPCM NLL using DYNAMIC theta from the encoder.

        For each (learner, step), theta_in[t] is the ability estimate BEFORE
        answering step t (shifted by one).  This matches the validated convention
        in deep_irt/core/encoder.py.

        Parameters
        ----------
        seq_item_ids    : (N, T) full item sequences (for encoding theta)
        seq_responses   : (N, T) full response sequences
        target_item_ids : (M,)   flat item ids for the loss positions
        target_responses: (M,)   flat responses for the loss positions
        reg             : regularisation weight on theta L2

        The caller is responsible for extracting the flat (M,) theta values that
        correspond to the (learner_idx, step_idx) positions in target_item_ids.
        For the training loop we compute the full shifted theta and index into it.

        Returns
        -------
        loss : scalar
        """
        # Full shifted theta: (N, T)
        theta_in = self.encoder.shifted_theta(seq_item_ids, seq_responses)
        return theta_in  # signal back to caller for indexing

    def gpcm_nll_flat(
        self,
        theta_flat: torch.Tensor,
        item_ids_flat: torch.Tensor,
        responses_flat: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        """
        GPCM NLL on already-extracted flat (M,) theta values.

        Parameters
        ----------
        theta_flat     : (M,)  scalar ability at each observation
        item_ids_flat  : (M,)  item ids
        responses_flat : (M,)  ordinal responses in {0,..,K-1}
        reg            : L2 weight on theta

        Returns
        -------
        loss : scalar
        """
        emb, d = self._get_d(item_ids_flat)
        theta_reg = reg * theta_flat.pow(2).mean()
        return self.gpcm_head.nll(theta_flat, d, emb, responses_flat, reg=reg) + theta_reg

    # ------------------------------------------------------------------
    # BT loss (person-free)
    # ------------------------------------------------------------------

    def bt_nll(
        self,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        outcome: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        """
        BT NLL for pairwise comparisons.

        Parameters
        ----------
        item_i, item_j : (N,) item indices
        outcome        : (N,) long {0, 1}  (1 = i beats j)
        """
        emb_i, d_i = self._get_d(item_i)
        emb_j, d_j = self._get_d(item_j)
        nll = self.bt_head.nll(d_i, d_j, outcome)
        d_reg = reg * (d_i.pow(2).mean() + d_j.pow(2).mean()) * 0.5
        return nll + d_reg

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_item_difficulty(self) -> torch.Tensor:
        """
        Shared d_i for all items, mean-centred.

        This scalar is simultaneously the GPCM difficulty centroid and the BT
        strength, just as in jointfmt.

        Returns
        -------
        d : (n_items,)
        """
        all_idx = torch.arange(self.n_items)
        emb = self.encoder.item_emb(all_idx)
        d = self.difficulty_extractor(emb)
        return d - d.mean()

    def get_item_strengths(self) -> torch.Tensor:
        """Alias for get_item_difficulty()."""
        return self.get_item_difficulty()

    def get_gpcm_difficulty(self) -> torch.Tensor:
        """Alias for get_item_difficulty()."""
        return self.get_item_difficulty()

    # ------------------------------------------------------------------
    # Exposable KT state -- PI-mandated public method
    # ------------------------------------------------------------------

    def get_kt_state(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the LSTM (hidden, cell) state at the final timestep.

        This is the raw KT mastery readout.  A downstream skill-level
        consumer can apply a skill-projection matrix to h_n to recover
        per-skill mastery estimates without re-running the encoder.

        Parameters
        ----------
        item_ids  : (batch, seq_len)  LongTensor
        responses : (batch, seq_len)  LongTensor

        Returns
        -------
        h_n : (batch, hidden_dim)   LSTM hidden state at final step
        c_n : (batch, hidden_dim)   LSTM cell state at final step
        """
        return self.encoder.get_kt_state(item_ids, responses)
