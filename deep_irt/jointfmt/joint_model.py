"""
joint_model.py -- Shared-embedding joint model for cross-format transfer.

Architecture
------------
ONE shared item embedding table (n_items x emb_dim).
A SHARED difficulty extractor that maps embedding -> scalar d_i.
TWO heads reading the SAME scalar d_i:

  GPCMHead  : (theta, d_i) -> GPCM NLL over ordinal responses
              (also uses per-item discrimination a_i from embedding)
  BTHead    : (d_i, d_j)   -> BT NLL over pairwise outcomes

The SHARED difficulty scalar d_i is the "one scale" that both formats
write to and read from. This enforces genuine unification: BT and GPCM
both push and pull d_i in the same direction (higher d_i = harder item).

For cross-format transfer:
  - PAIRWISE-ONLY items: BT shapes d_i via pairwise comparisons.
    The GPCM head can then use d_i for ordinal placement even though
    it never saw this item directly.
  - DIRECT-ONLY items: GPCM shapes d_i via respondent responses.
    The BT head can then use d_i for pairwise strength even though
    it never saw this item in comparisons.

This is unambiguously "one scale" -- not two separate projections that
happen to share an embedding table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared difficulty extractor
# ---------------------------------------------------------------------------

class DifficultyExtractor(nn.Module):
    """
    Maps item embedding -> scalar difficulty d_i.

    This is the common currency shared by both formats.
    emb -> fc -> d  (linear, unconstrained sign)
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        # Small MLP: emb -> hidden -> scalar, to allow non-linear extraction
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        emb : (..., emb_dim)

        Returns
        -------
        d : (...,)  scalar difficulty
        """
        return self.net(emb).squeeze(-1)


# ---------------------------------------------------------------------------
# GPCM Head (uses shared d_i)
# ---------------------------------------------------------------------------

class GPCMHead(nn.Module):
    """
    Uses pre-extracted d_i (from DifficultyExtractor) as the difficulty centroid.
    Also extracts per-item discrimination a_i from embedding.

    GPCM step thresholds: beta_k = d_i + offset_k
    where offsets are free parameters per item (K-2 free values, first = 0).
    """

    def __init__(self, emb_dim: int, K: int) -> None:
        super().__init__()
        self.K = K
        # Discrimination (log scale for positivity)
        self.fc_a = nn.Linear(emb_dim, 1)
        # Step offsets: K-2 free values (first offset is 0 by convention)
        # So we have K-1 thresholds: [d, d+off1, d+off2]
        self.fc_offsets = nn.Linear(emb_dim, K - 2) if K > 2 else None

    def log_probs(
        self,
        theta: torch.Tensor,
        d: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        theta : (N,)
        d     : (N,)  shared difficulty scalar
        emb   : (N, emb_dim)

        Returns
        -------
        log_probs : (N, K)
        """
        a = F.softplus(self.fc_a(emb)).squeeze(-1)   # (N,) positive

        # Build K-1 thresholds from d + offsets
        # First threshold = d (offset 0), rest = d + raw_offsets
        if self.fc_offsets is not None:
            raw_off = self.fc_offsets(emb)            # (N, K-2)
            # thresholds: [d, d+off_0, d+off_1, ...]
            b = torch.cat([d.unsqueeze(-1), d.unsqueeze(-1) + raw_off], dim=-1)  # (N, K-1)
        else:
            b = d.unsqueeze(-1)                        # (N, 1) for K=2

        # GPCM unnormalised scores
        step = a.unsqueeze(-1) * (theta.unsqueeze(-1) - b)  # (N, K-1)
        cumsum = torch.cumsum(step, dim=-1)
        log_num = torch.cat(
            [torch.zeros(theta.size(0), 1, device=theta.device), cumsum], dim=-1
        )  # (N, K)
        return F.log_softmax(log_num, dim=-1)

    def nll(
        self,
        theta: torch.Tensor,
        d: torch.Tensor,
        emb: torch.Tensor,
        responses: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        """Mean GPCM NLL + small L2 reg on offsets."""
        log_p = self.log_probs(theta, d, emb)
        nll = F.nll_loss(log_p, responses)
        # Reg on offsets
        if self.fc_offsets is not None:
            off_reg = reg * self.fc_offsets(emb).pow(2).mean()
        else:
            off_reg = 0.0
        return nll + off_reg


# ---------------------------------------------------------------------------
# BT Head (uses shared d_i)
# ---------------------------------------------------------------------------

class BTHead(nn.Module):
    """
    Uses pre-extracted d_i directly as item strength.
    P(i beats j) = sigmoid(d_i - d_j)

    No additional parameters needed: the shared difficulty IS the BT strength.
    """

    def nll(
        self,
        d_i: torch.Tensor,
        d_j: torch.Tensor,
        outcome: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy NLL on pairwise outcomes.
        outcome == 1: i beats j.
        """
        delta = d_i - d_j
        bce = F.binary_cross_entropy_with_logits(delta, outcome.float(), reduction="mean")
        return bce


# ---------------------------------------------------------------------------
# Joint Model
# ---------------------------------------------------------------------------

class JointModel(nn.Module):
    """
    Shared-embedding, shared-difficulty joint GPCM + BT model.

    The key design: a single DifficultyExtractor maps embeddings to a scalar d_i.
    Both heads operate on this SAME d_i.
    - GPCM: d_i is the difficulty centroid for step thresholds.
    - BT  : d_i is the strength (P(i beats j) = sigmoid(d_i - d_j)).

    This is a true "one scale" -- any item's d_i can be read by either format.

    Parameters
    ----------
    n_items      : int
    n_respondents: int
    emb_dim      : int
    K            : int  -- GPCM categories
    seed         : int
    """

    def __init__(
        self,
        n_items: int,
        n_respondents: int,
        emb_dim: int = 16,
        K: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)

        # Shared item embedding table
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        # Shared difficulty extractor (the "one scale")
        self.difficulty_extractor = DifficultyExtractor(emb_dim=emb_dim, hidden_dim=8)

        # Per-respondent ability (GPCM only)
        self.theta = nn.Parameter(
            torch.randn(n_respondents, generator=g) * 0.1
        )

        # Heads (both read from the shared d_i)
        self.gpcm_head = GPCMHead(emb_dim=emb_dim, K=K)
        self.bt_head = BTHead()

        self.n_items = n_items
        self.n_respondents = n_respondents
        self.emb_dim = emb_dim
        self.K = K

    def _get_d(self, item_ids: torch.Tensor) -> tuple:
        """Returns (emb, d) for given item_ids."""
        emb = self.item_emb(item_ids)
        d = self.difficulty_extractor(emb)
        return emb, d

    def gpcm_nll(
        self,
        respondent_ids: torch.Tensor,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        """GPCM NLL for a batch of (respondent, item, response) triples."""
        theta = self.theta[respondent_ids]
        emb, d = self._get_d(item_ids)
        theta_reg = reg * self.theta.pow(2).mean()
        return self.gpcm_head.nll(theta, d, emb, responses, reg=reg) + theta_reg

    def bt_nll(
        self,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        outcome: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        """BT NLL for a batch of pairwise comparisons."""
        emb_i, d_i = self._get_d(item_i)
        emb_j, d_j = self._get_d(item_j)
        nll = self.bt_head.nll(d_i, d_j, outcome)
        # Soft L2 anchor on d to prevent mean drift
        d_reg = reg * (d_i.pow(2).mean() + d_j.pow(2).mean()) * 0.5
        return nll + d_reg

    @torch.no_grad()
    def get_item_difficulty(self) -> torch.Tensor:
        """
        Return the canonical shared difficulty d_i for all items.
        This scalar is SIMULTANEOUSLY the GPCM difficulty centroid
        and the BT strength. Mean-centered for identifiability.

        Returns
        -------
        d : (n_items,)
        """
        all_idx = torch.arange(self.n_items)
        emb = self.item_emb(all_idx)
        d = self.difficulty_extractor(emb)
        return d - d.mean()

    @torch.no_grad()
    def get_gpcm_difficulty(self) -> torch.Tensor:
        """Alias: same as get_item_difficulty() -- the shared d_i."""
        return self.get_item_difficulty()

    @torch.no_grad()
    def get_item_strengths(self) -> torch.Tensor:
        """Alias: same as get_item_difficulty() -- the shared d_i."""
        return self.get_item_difficulty()
