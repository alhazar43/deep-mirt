"""Static GPCM — pure IRT baseline with no sequential memory.

Theory
------
The Graded Partial Credit Model (GPCM, Muraki 1992) assigns each student s
a fixed latent ability θ_s ∈ R^D and each item j discrimination α_j ∈ R^D
and ordered thresholds β_{j,1} < … < β_{j,K-1}.

Cumulative logits (same formula as GPCMLogits in irt.py):

    η_j   = α_j · θ_s                    (dot-product interaction)
    ‖α_j‖ = sqrt(Σ_d α_{j,d}²)
    φ_{j,k} = Σ_{h=1}^{k} (η_j − ‖α_j‖ · β_{j,h})   k = 1, …, K-1
    φ_{j,0} = 0

    P(Y = k | s, j) = exp(φ_{j,k}) / Σ_{m=0}^{K-1} exp(φ_{j,m})

This model has no DKVMN memory and no learning dynamics — it is the
standard psychometric GPCM used as a non-sequential IRT baseline.

Parameterisation
----------------
- θ: nn.Embedding(n_students + 1, n_traits)  — index 0 = padding/unknown
- α: nn.Parameter(n_questions + 1, n_traits) — raw; mapped via exp(0.3 * raw)
- β: monotonic gap construction (same as IRTParameterExtractor):
      β_{j,0} = base_j
      β_{j,k} = β_{j,k-1} + softplus(gap_{j,k-1})   k ≥ 1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .components.irt import GPCMLogits
from .heads.gpcm import GPCMHead


class StaticGPCM(nn.Module):
    """Pure IRT GPCM baseline — no DKVMN, no sequential memory.

    Args:
        n_questions: Number of unique items Q (1-based IDs; 0 = padding).
        n_students:  Number of unique students N (1-based IDs; 0 = padding).
        n_categories: Number of ordinal response categories K ≥ 2.
        n_traits:    Latent trait dimension D.  D = 1 → standard GPCM.
        ability_scale: Scalar multiplied onto raw θ output.
    """

    def __init__(
        self,
        n_questions: int,
        n_students: int,
        n_categories: int = 5,
        n_traits: int = 1,
        ability_scale: float = 1.0,
        # Accept (and ignore) DeepGPCM-compatible kwargs so the same config
        # section can be reused without stripping irrelevant keys.
        **kwargs,
    ) -> None:
        super().__init__()

        self.n_questions = n_questions
        self.n_students = n_students
        self.n_categories = n_categories
        self.n_traits = n_traits
        self.ability_scale = ability_scale

        # ---- Student ability θ: (N+1, D) embedding --------------------------
        # Index 0 is reserved for padding / unknown students.
        self.theta_embed = nn.Embedding(n_students + 1, n_traits, padding_idx=0)

        # ---- Item discrimination α: raw parameter, mapped via exp(0.3 * raw)
        # Shape (Q+1, D); row 0 unused (padding item).
        self.alpha_raw = nn.Parameter(torch.zeros(n_questions + 1, n_traits))

        # ---- Item thresholds β: monotonic gap parameterisation --------------
        # β_base: (Q+1, 1) — unconstrained base threshold per item
        self.beta_base = nn.Parameter(torch.zeros(n_questions + 1, 1))
        # β_gaps: (Q+1, K-2) — strictly positive gaps via softplus
        if n_categories > 2:
            self.beta_gaps = nn.Parameter(
                torch.full((n_questions + 1, n_categories - 2), 0.3)
            )

        # ---- GPCM logit + probability layers --------------------------------
        self.gpcm_logits = GPCMLogits()
        self.gpcm_head = GPCMHead()

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.theta_embed.weight, std=1.0)
        # Zero out padding row
        with torch.no_grad():
            self.theta_embed.weight[0].zero_()
        nn.init.normal_(self.alpha_raw, std=0.1)
        nn.init.normal_(self.beta_base, std=0.05)
        if self.n_categories > 2:
            nn.init.constant_(self.beta_gaps, 0.3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        student_ids: Tensor,
        questions: Tensor,
        responses: Tensor,
        mask: Tensor | None = None,
    ) -> dict:
        """Static GPCM forward pass.

        Args:
            student_ids: Long tensor ``(B, S)`` — student IDs in [1, N].
                         Padding positions should use ID 0.
            questions:   Long tensor ``(B, S)`` — item IDs in [1, Q].
                         Padding positions should use ID 0.
            responses:   Long tensor ``(B, S)`` — ordinal responses in [0, K-1].
                         Not used in the forward pass (no memory), but kept
                         for API compatibility with DeepGPCM.
            mask:        Optional bool tensor ``(B, S)`` — True = valid position.

        Returns:
            Dict with keys:
                ``"theta"``   (B, S, D)  — student ability (same value repeated over S)
                ``"alpha"``   (B, S, D)  — item discrimination
                ``"beta"``    (B, S, K-1) — item thresholds
                ``"logits"``  (B, S, K)  — GPCM logits (for loss)
                ``"probs"``   (B, S, K)  — GPCM probabilities (for metrics)
        """
        B, S = questions.shape

        # ---- θ: look up per-student ability, broadcast over S ---------------
        # student_ids: (B, S) — take the first valid ID per sequence (all
        # positions in a sequence share the same student).  We use the full
        # (B, S) lookup so padding positions (ID=0) get zero vectors naturally.
        theta = self.theta_embed(student_ids) * self.ability_scale  # (B, S, D)

        # ---- α: look up per-item discrimination, apply exp(0.3 * raw) -------
        alpha = torch.exp(0.3 * self.alpha_raw[questions])  # (B, S, D)

        # ---- β: monotonic gap construction per item -------------------------
        beta_0 = self.beta_base[questions]  # (B, S, 1)
        if self.n_categories == 2:
            beta = beta_0  # (B, S, 1)
        else:
            gaps = F.softplus(self.beta_gaps[questions])  # (B, S, K-2)
            betas = [beta_0]
            for i in range(gaps.shape[-1]):
                betas.append(betas[-1] + gaps[:, :, i : i + 1])
            beta = torch.cat(betas, dim=-1)  # (B, S, K-1)

        # ---- GPCM logits + probabilities ------------------------------------
        logits = self.gpcm_logits(theta, alpha, beta)  # (B, S, K)
        probs = self.gpcm_head(logits)                 # (B, S, K)

        return {
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "logits": logits,
            "probs": probs,
        }
