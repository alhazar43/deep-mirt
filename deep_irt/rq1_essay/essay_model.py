"""essay_model.py -- Shared content-extractor joint model for essay quality.

Architecture
------------
FROZEN content features x_i (n_essays x feat_dim) -- NOT learnable.
A SHARED content extractor maps x_i -> scalar quality d_i.
TWO heads read the SAME d_i:

  GradedHead : ordinal (GPCM/proportional-odds) NLL of the human holistic
               score given d_i. A single global "grader ability" theta plus a
               per-scale discrimination places d_i on the human-score scale.
  BTHead     : (d_i, d_j) -> Bradley-Terry NLL over LLM pairwise judgments.

The shared scalar d_i is the one quality scale both formats write to.

Why this is genuine content-mediated transfer (not the "both recover truth"
trap): d_i is a FUNCTION of frozen content features. The graded essays teach
the extractor a content->quality map; that map then positions a pairwise-only
essay from its content alone, so the pairwise comparisons refine a position
that content structure already informs. The contrast baseline (separate
extractors per format, SAME frozen features) isolates what the cross-format
sharing adds on top of content-only generalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared content extractor
# ---------------------------------------------------------------------------

class ContentExtractor(nn.Module):
    """Maps FROZEN content features -> scalar quality d_i.

    feat -> Linear -> Tanh -> Linear -> scalar. Sign unconstrained.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., feat_dim) -> d: (...,)"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Graded (ordinal) head -- single global grader
# ---------------------------------------------------------------------------

class GradedHead(nn.Module):
    """Ordinal head over human holistic scores using the shared d_i.

    QUALITY convention: d_i is ability-like, so HIGHER d_i -> HIGHER score.
    This MATCHES the BT head (higher d_i -> wins -> higher quality), which is
    essential: if the graded head treated d_i as difficulty (anti-correlated
    with score) while BT treats it as quality, the shared extractor would be
    torn between opposite sign conventions and transfer would break.

    GPCM/partial-credit form with GLOBAL increasing cutpoints c_k (a property
    of the score scale / rubric, shared across essays; essay quality IS d_i):
      step logit for category boundary k = a * (d_i - c_k),  c increasing.
    Higher d_i pushes mass toward higher categories. With a single grader the
    grader ability is absorbed into the cutpoint location (unidentifiable
    otherwise), so theta is not a separate parameter.
    """

    def __init__(self, K: int) -> None:
        super().__init__()
        self.K = K
        self.a = nn.Parameter(torch.tensor(0.0))            # log-discrimination
        # First cutpoint free; rest = first + cumulative positive increments,
        # so the K-1 cutpoints stay strictly increasing.
        self.c0 = nn.Parameter(torch.tensor(0.0))
        self.dc = nn.Parameter(torch.zeros(K - 2)) if K > 2 else None

    def _cutpoints(self) -> torch.Tensor:
        if self.dc is not None:
            inc = torch.cumsum(F.softplus(self.dc), dim=0)   # (K-2,) increasing
            return torch.cat([self.c0.unsqueeze(0), self.c0 + inc])  # (K-1,)
        return self.c0.unsqueeze(0)                          # (1,) for K=2

    def log_probs(self, d: torch.Tensor) -> torch.Tensor:
        """d: (N,) -> log_probs: (N, K). Higher d -> higher category."""
        a = F.softplus(self.a)
        c = self._cutpoints()                                # (K-1,)
        step = a * (d.unsqueeze(-1) - c.unsqueeze(0))        # (N, K-1)
        cumsum = torch.cumsum(step, dim=-1)
        log_num = torch.cat(
            [torch.zeros(d.size(0), 1, device=d.device), cumsum], dim=-1
        )  # (N, K)
        return F.log_softmax(log_num, dim=-1)

    def nll(self, d: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(self.log_probs(d), scores)


# ---------------------------------------------------------------------------
# BT (pairwise) head -- uses shared d_i directly
# ---------------------------------------------------------------------------

class BTHead(nn.Module):
    """P(i better than j) = sigmoid(d_i - d_j). No own parameters."""

    def nll(self, d_i: torch.Tensor, d_j: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(d_i - d_j, outcome.float(), reduction="mean")


# ---------------------------------------------------------------------------
# Joint model (shared extractor)
# ---------------------------------------------------------------------------

class JointEssayModel(nn.Module):
    """Shared-content-extractor joint Graded + BT model.

    features : (n_essays, feat_dim) FROZEN buffer.
    A single ContentExtractor produces d_i for every essay. Both heads read it.
    """

    def __init__(self, features: torch.Tensor, K: int,
                 hidden_dim: int = 64, dropout: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("features", features)           # frozen
        self.feat_dim = features.shape[1]
        self.K = K
        self.extractor = ContentExtractor(self.feat_dim, hidden_dim, dropout)
        self.graded_head = GradedHead(K)
        self.bt_head = BTHead()

    def d_all(self) -> torch.Tensor:
        return self.extractor(self.features)                 # (n_essays,)

    def graded_nll(self, idx: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        d = self.extractor(self.features[idx])
        return self.graded_head.nll(d, scores)

    def bt_nll(self, idx_i: torch.Tensor, idx_j: torch.Tensor,
               outcome: torch.Tensor) -> torch.Tensor:
        d_i = self.extractor(self.features[idx_i])
        d_j = self.extractor(self.features[idx_j])
        return self.bt_head.nll(d_i, d_j, outcome)

    @torch.no_grad()
    def get_quality(self) -> torch.Tensor:
        """Mean-centered shared quality d_i for ALL essays (eval=no dropout)."""
        self.eval()
        d = self.extractor(self.features)
        return d - d.mean()


# ---------------------------------------------------------------------------
# Independent baselines (separate extractors, SAME frozen features)
# ---------------------------------------------------------------------------

class IndepGradedModel(nn.Module):
    """Graded-only extractor. Fit on graded essays; read out elsewhere via
    content. This is content-based generalization, NOT noise."""

    def __init__(self, features: torch.Tensor, K: int,
                 hidden_dim: int = 64, dropout: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed + 11)
        self.register_buffer("features", features)
        self.extractor = ContentExtractor(features.shape[1], hidden_dim, dropout)
        self.graded_head = GradedHead(K)

    def nll(self, idx: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        d = self.extractor(self.features[idx])
        return self.graded_head.nll(d, scores)

    @torch.no_grad()
    def get_quality(self) -> torch.Tensor:
        self.eval()
        d = self.extractor(self.features)
        return d - d.mean()


class IndepBTModel(nn.Module):
    """Pairwise-only extractor. Fit on comparisons; read out elsewhere via
    content."""

    def __init__(self, features: torch.Tensor,
                 hidden_dim: int = 64, dropout: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed + 22)
        self.register_buffer("features", features)
        self.extractor = ContentExtractor(features.shape[1], hidden_dim, dropout)
        self.bt_head = BTHead()

    def nll(self, idx_i: torch.Tensor, idx_j: torch.Tensor,
            outcome: torch.Tensor) -> torch.Tensor:
        d_i = self.extractor(self.features[idx_i])
        d_j = self.extractor(self.features[idx_j])
        return self.bt_head.nll(d_i, d_j, outcome)

    @torch.no_grad()
    def get_quality(self) -> torch.Tensor:
        self.eval()
        d = self.extractor(self.features)
        return d - d.mean()
