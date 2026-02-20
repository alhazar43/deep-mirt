"""IRT parameter extraction and GPCM logit computation.

Two classes:

``IRTParameterExtractor``
    Maps neural summary vectors to psychometric parameters
    (θ, α, β) with multi-dimensional support.

``GPCMLogits``
    Converts (θ, α, β) triplets to K-category GPCM unnormalised logits.
    Kept as a separate module from ``GPCMHead`` so that the trainer can
    use logits directly for numerically stable loss computation.

Theory — GPCM (Muraki 1992)
----------------------------
Given student ability θ ∈ R^D, item discrimination α ∈ R^D, and
ordered thresholds β₁ < β₂ < … < β_{K-1}:

    Interaction: η = α · θ = Σ_d α_d θ_d    (scalar)
    Norm:        ‖α‖ = sqrt(Σ_d α_d²)

    Cumulative logit:
        φ_k = Σ_{h=1}^{k} (η − ‖α‖·β_h)   for k = 1, …, K-1
        φ_0 = 0                              (baseline)

    P(Y = k | θ, α, β) = exp(φ_k) / Σ_{j=0}^{K-1} exp(φ_j)

For D = 1: ‖α‖ = α (alpha > 0), so φ_k = Σ α·(θ − β_h), which is the
standard GPCM formula.  The ‖α‖ scaling of β gives identifiability:
scaling α by c scales the β contribution equally, so the data can
recover the correct α scale — unlike the M-GPCM formula (η − β_h)
which has a full α–θ rotational ambiguity.

Monotonic gap parameterisation for β (research-based)
------------------------------------------------------
To guarantee β₁ < β₂ < … < β_{K-1} without constrained optimisation:

    β₁ = linear(features)
    βₖ = β_{k-1} + softplus(gap_{k-1})    for k = 2, …, K-1

where ``softplus`` ensures strictly positive gaps.  The base threshold
and gaps are separate linear layers initialised with small weights and
a positive bias on the gaps (≈ 0.3) so the initial thresholds are
spread out.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IRTParameterExtractor(nn.Module):
    """Extract (θ, α, β) IRT parameters from DKVMN summary vectors.

    Outputs always have a trailing dimension D = ``n_traits``, even for
    D = 1.  This eliminates all special-casing downstream:

        theta: (B, S, D)
        alpha: (B, S, D)
        beta:  (B, S, K-1)

    Args:
        input_dim: Dimension of the summary vector fed to θ / α networks.
        n_questions: Number of items in the item bank (needed to size the
            question-embedding input dimension for β and α networks).
            Passed as ``question_dim = key_dim`` from the model.
        n_categories: Number of ordinal response categories K.
        n_traits: Latent trait dimension D.  D = 1 → standard GPCM.
        ability_scale: Scalar multiplied onto raw θ output.
        question_dim: Dimension of the item embedding (typically
            ``key_dim``).  Defaults to ``input_dim`` if ``None``.
    """

    def __init__(
        self,
        input_dim: int,
        n_questions: int,
        n_categories: int,
        n_traits: int = 1,
        ability_scale: float = 1.0,
        question_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_categories = n_categories
        self.n_traits = n_traits
        self.ability_scale = ability_scale
        self.question_dim = question_dim if question_dim is not None else input_dim

        # ---- Student ability (θ): (B, S, input_dim) → (B, S, D) --------
        self.ability_network = nn.Linear(input_dim, n_traits)

        # ---- Item discrimination (α): summary + question embedding → (B, S, D)
        # Concatenating the student summary with the question embedding gives the
        # network richer signal to learn item-level discrimination.  α becomes
        # student-state-dependent per timestep, but averaging estimates across
        # all students (as done in plot_recovery.py) recovers item-level α with
        # much higher correlation than using question embedding alone.
        # This matches deep-gpcm's architecture (discrim_input = summary + q_embed).
        self.discrimination_network = nn.Linear(input_dim + self.question_dim, n_traits)

        # ---- Item difficulty (β): monotonic gap parameterisation ---------
        # β₀: unconstrained base threshold per item
        self.threshold_base = nn.Linear(self.question_dim, 1)
        # Positive gaps for β₁, …, β_{K-2}  (K-2 extra gaps for K-1 total thresholds)
        if n_categories > 2:
            self.threshold_gaps = nn.Linear(self.question_dim, n_categories - 2)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise all sub-networks.

        Conventions (preserved from original deep-gpcm):
        - ability / discrimination: kaiming_normal weight, zero bias.
        - threshold_base: N(0, 0.05) weight, zero bias.
        - threshold_gaps: N(0, 0.05) weight, bias = 0.3
          (positive bias keeps initial thresholds spread out).
        - discrimination: N(0, 0.1) weight, zero bias so that
          exp(0.3 * 0) = 1.0 is the initial discrimination.
        """
        nn.init.kaiming_normal_(self.ability_network.weight)
        nn.init.constant_(self.ability_network.bias, 0.0)

        nn.init.normal_(self.discrimination_network.weight, std=0.1)
        nn.init.constant_(self.discrimination_network.bias, 0.0)

        nn.init.normal_(self.threshold_base.weight, std=0.05)
        nn.init.constant_(self.threshold_base.bias, 0.0)

        if self.n_categories > 2:
            nn.init.normal_(self.threshold_gaps.weight, std=0.05)
            nn.init.constant_(self.threshold_gaps.bias, 0.3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, summary: Tensor, question_embed: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract IRT parameters.

        Args:
            summary: ``(B, S, input_dim)`` — DKVMN summary vector.
            question_embed: ``(B, S, question_dim)`` — item embedding
                (typically the ``q_embed`` output at each timestep).

        Returns:
            Tuple ``(theta, alpha, beta)``:
                - ``theta``: ``(B, S, D)`` student ability.
                - ``alpha``: ``(B, S, D)`` item discrimination > 0.
                - ``beta``:  ``(B, S, K-1)`` ordered item thresholds.
        """
        # ---- theta: (B, S, D) -------------------------------------------
        theta = self.ability_network(summary) * self.ability_scale  # (B, S, D)

        # ---- alpha: lognormal mapping exp(0.3 * raw) -------------------
        # Input: [summary, question_embed] concatenated — richer signal for
        # learning item-level discrimination.  Averaging per-item estimates
        # across diverse student states recovers item-level α.
        raw_alpha = self.discrimination_network(
            torch.cat([summary, question_embed], dim=-1)
        )  # (B, S, D)
        alpha = torch.exp(0.3 * raw_alpha)                       # (B, S, D)

        # ---- beta: monotonic gap parameterisation -----------------------
        beta_0 = self.threshold_base(question_embed)  # (B, S, 1)
        if self.n_categories == 2:
            beta = beta_0  # (B, S, 1)
        else:
            # Strictly positive gaps via softplus
            gaps = F.softplus(self.threshold_gaps(question_embed))  # (B, S, K-2)
            betas = [beta_0]
            for i in range(gaps.shape[-1]):
                betas.append(betas[-1] + gaps[:, :, i : i + 1])
            beta = torch.cat(betas, dim=-1)  # (B, S, K-1)

        return theta, alpha, beta


# ---------------------------------------------------------------------------
# GPCM logit computation (Phase 5)
# ---------------------------------------------------------------------------


class GPCMLogits(nn.Module):
    """Compute unnormalised GPCM logits from IRT parameters.

    Takes ``(theta, alpha, beta)`` produced by ``IRTParameterExtractor``
    and returns logits of shape ``(B, S, K)`` where category 0 has
    logit value 0 (the cumulative-sum baseline).

    The split from ``GPCMHead`` (which applies softmax) lets the trainer
    pass logits directly to ``F.cross_entropy`` or ``FocalLoss``,
    avoiding the numerical instability of ``log(softmax(x) + ε)``.

    Formula — standard GPCM generalised to D dimensions
    -----------------------------------------------------
    For D = 1 (standard GPCM, Muraki 1992):

        η   = α · θ                 scalar interaction
        φ_k = Σ_{h=0}^{k-1} α·(θ − β_h)
            = Σ_{h=0}^{k-1} (η − α·β_h)

    For D > 1 (MIRT generalisation):

        η       = α · θ = Σ_d α_d θ_d   (dot product interaction)
        ‖α‖     = sqrt(Σ_d α_d²)         (Euclidean norm)
        φ_k     = Σ_{h=0}^{k-1} (η − ‖α‖·β_h)

    For D = 1 this reduces exactly to the scalar formula because ‖α‖ = α
    (alpha is positive via the exp mapping).  The ‖α‖ scaling of β gives
    the standard GPCM identifiability property: scaling α by c also
    scales the β contribution by c, so the data identify the ratio α/β
    rather than α and θ being interchangeable.
    """

    def forward(self, theta: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """Compute GPCM cumulative logits.

        Args:
            theta: ``(B, S, D)`` student ability.
            alpha: ``(B, S, D)`` item discrimination.
            beta:  ``(B, S, K-1)`` ordered item thresholds.

        Returns:
            Logits tensor ``(B, S, K)``.  logits[:, :, 0] == 0 always.
        """
        B, S, D = theta.shape
        K = beta.shape[-1] + 1

        # η = dot(alpha, theta) over trait dimension D  →  (B, S)
        interaction = (alpha * theta).sum(dim=-1)

        # ‖α‖ = Euclidean norm over trait dimension D  →  (B, S)
        # For D=1: ‖α‖ = α  (since alpha > 0 from exp mapping)
        alpha_norm = alpha.norm(dim=-1)  # (B, S)

        # Cumulative logits: φ_k = Σ_{h=0}^{k-1} (η − ‖α‖·β_h)
        # Equivalent to standard GPCM Σ α·(θ − β_h) for D = 1.
        logits = torch.zeros(B, S, K, device=theta.device, dtype=theta.dtype)
        for k in range(1, K):
            logits[:, :, k] = (
                interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta[:, :, :k]
            ).sum(dim=-1)

        return logits
