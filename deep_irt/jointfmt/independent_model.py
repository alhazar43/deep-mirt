"""
independent_model.py -- Baseline: two completely independent models.

Architecture MATCHES the joint model: each uses a DifficultyExtractor
(emb -> d_i). The ONLY difference from the joint model is that the two
formats have SEPARATE embedding tables and SEPARATE difficulty extractors,
so the two models cannot share information.

GPCM fit (separate embedding + difficulty extractor):
  Fits only on direct-observable items (DIRECT-ONLY + BOTH).
  Has no knowledge of pairwise-only items.
  For pairwise-only items, the d_i output is uninformed random noise.

BT fit (separate embedding + difficulty extractor):
  Fits only on pairwise-observable items (PAIRWISE-ONLY + BOTH).
  Has no knowledge of direct-only items.
  For direct-only items, the d_i output is uninformed random noise.

KEY POINT:
  The architectural difference from the joint model is strictly the
  absence of a shared embedding table (and extractor). All other
  design choices are identical, so the metric gap directly measures
  the value of sharing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from joint_model import DifficultyExtractor


# ---------------------------------------------------------------------------
# Independent GPCM
# ---------------------------------------------------------------------------

class IndepGPCM(nn.Module):
    """
    GPCM with its own embedding table and difficulty extractor.
    Same architecture as the GPCM pathway in the joint model.

    Pairwise-only items: embedding never trained -> d_i is noise.
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
        g.manual_seed(seed + 10)

        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        self.difficulty_extractor = DifficultyExtractor(emb_dim=emb_dim, hidden_dim=8)

        self.theta = nn.Parameter(torch.randn(n_respondents, generator=g) * 0.1)

        self.fc_a = nn.Linear(emb_dim, 1)
        # Step offsets (K-2 free values, first threshold = d)
        self.fc_offsets = nn.Linear(emb_dim, K - 2) if K > 2 else None

        self.K = K
        self.n_items = n_items

    def nll(
        self,
        respondent_ids: torch.Tensor,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        theta = self.theta[respondent_ids]
        emb = self.item_emb(item_ids)
        d = self.difficulty_extractor(emb)

        a = F.softplus(self.fc_a(emb)).squeeze(-1)

        if self.fc_offsets is not None:
            raw_off = self.fc_offsets(emb)
            b = torch.cat([d.unsqueeze(-1), d.unsqueeze(-1) + raw_off], dim=-1)
        else:
            b = d.unsqueeze(-1)

        step = a.unsqueeze(-1) * (theta.unsqueeze(-1) - b)
        cumsum = torch.cumsum(step, dim=-1)
        log_num = torch.cat(
            [torch.zeros(theta.size(0), 1, device=theta.device), cumsum], dim=-1
        )
        log_p = F.log_softmax(log_num, dim=-1)
        nll = F.nll_loss(log_p, responses)

        theta_reg = reg * self.theta.pow(2).mean()
        if self.fc_offsets is not None:
            off_reg = reg * self.fc_offsets(emb).pow(2).mean()
        else:
            off_reg = 0.0
        return nll + theta_reg + off_reg

    @torch.no_grad()
    def get_difficulty(self) -> torch.Tensor:
        """
        Returns mean-centered d_i for ALL items.
        Pairwise-only items: d_i is noise (never trained).
        """
        all_idx = torch.arange(self.n_items)
        emb = self.item_emb(all_idx)
        d = self.difficulty_extractor(emb)
        return d - d.mean()


# ---------------------------------------------------------------------------
# Independent BT
# ---------------------------------------------------------------------------

class IndepBT(nn.Module):
    """
    BT model with its own embedding table and difficulty extractor.
    Same architecture as the BT pathway in the joint model.

    Direct-only items: embedding never trained -> d_i is noise.
    """

    def __init__(
        self,
        n_items: int,
        emb_dim: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed + 20)

        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        self.difficulty_extractor = DifficultyExtractor(emb_dim=emb_dim, hidden_dim=8)
        self.n_items = n_items

    def nll(
        self,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        outcome: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        emb_i = self.item_emb(item_i)
        emb_j = self.item_emb(item_j)
        d_i = self.difficulty_extractor(emb_i)
        d_j = self.difficulty_extractor(emb_j)
        delta = d_i - d_j
        bce = F.binary_cross_entropy_with_logits(delta, outcome.float(), reduction="mean")
        d_reg = reg * (d_i.pow(2).mean() + d_j.pow(2).mean()) * 0.5
        return bce + d_reg

    @torch.no_grad()
    def get_strength(self) -> torch.Tensor:
        """
        Returns mean-centered d_i for ALL items.
        Direct-only items: d_i is noise (never trained).
        """
        all_idx = torch.arange(self.n_items)
        emb = self.item_emb(all_idx)
        d = self.difficulty_extractor(emb)
        return d - d.mean()
