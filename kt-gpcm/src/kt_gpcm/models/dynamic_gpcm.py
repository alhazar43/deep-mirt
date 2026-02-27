"""Dynamic GPCM — sequential IRT baseline with learned gated ability updates.

Theory
------
Extends StaticGPCM by updating the student ability estimate θ_t at each step
using a gated recurrent update driven by the prediction error (surprise signal):

    surprise_t = r_t - E[Y_t | θ_t, α_j, β_j]          (scalar residual)
    h_t        = tanh(W_h · [θ_t, surprise_t, α_j, β_j] + b_h)
    gate_t     = σ(W_g · h_t + b_g)                      (D-dim gate)
    delta_t    = tanh(W_d · h_t + b_d)                   (D-dim update)
    θ_{t+1}   = θ_t + gate_t * delta_t                   (gated residual)

The update is causal: θ_t is used to predict r_t, then updated using r_t to
produce θ_{t+1} for predicting r_{t+1}.

This isolates the contribution of sequential ability updating (no DKVMN memory)
from the full DEEP-GPCM model, enabling a three-way ablation:

    Static GPCM → DynamicGPCM → DEEP-GPCM

Item parameters α_j and β_{j,k} are static lookup tables (same as StaticGPCM).
The only new component is the two-layer gated update network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .components.irt import GPCMLogits
from .heads.gpcm import GPCMHead


class DynamicGPCM(nn.Module):
    """Sequential IRT baseline with gated ability updates.

    Args:
        n_questions:   Number of unique items Q (1-based IDs; 0 = padding).
        n_students:    Number of unique students N (1-based IDs; 0 = padding).
        n_categories:  Number of ordinal response categories K >= 2.
        n_traits:      Latent trait dimension D.  D=1 -> standard GPCM.
        ability_scale: Scalar multiplied onto raw theta output.
        hidden_dim:    Hidden size of the gated update network.
    """

    def __init__(
        self,
        n_questions: int,
        n_students: int,
        n_categories: int = 5,
        n_traits: int = 1,
        ability_scale: float = 1.0,
        hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()

        self.n_questions = n_questions
        self.n_students = n_students
        self.n_categories = n_categories
        self.n_traits = n_traits
        self.ability_scale = ability_scale
        self.hidden_dim = hidden_dim

        # ---- Student ability θ_0: (N+1, D) embedding -----------------------
        self.theta_embed = nn.Embedding(n_students + 1, n_traits, padding_idx=0)

        # ---- Item discrimination α: raw parameter, mapped via exp(0.3 * raw)
        self.alpha_raw = nn.Parameter(torch.zeros(n_questions + 1, n_traits))

        # ---- Item thresholds β: monotonic gap parameterisation --------------
        self.beta_base = nn.Parameter(torch.zeros(n_questions + 1, 1))
        if n_categories > 2:
            self.beta_gaps = nn.Parameter(
                torch.full((n_questions + 1, n_categories - 2), 0.3)
            )

        # ---- Gated update network -------------------------------------------
        # Input: [theta(D), surprise(1), alpha(D), beta(K-1)]
        update_input_dim = n_traits + 1 + n_traits + (n_categories - 1)
        self.update_hidden = nn.Linear(update_input_dim, hidden_dim)
        self.update_gate   = nn.Linear(hidden_dim, n_traits)
        self.update_delta  = nn.Linear(hidden_dim, n_traits)

        # ---- GPCM logit + probability layers --------------------------------
        self.gpcm_logits = GPCMLogits()
        self.gpcm_head   = GPCMHead()

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.theta_embed.weight, std=1.0)
        with torch.no_grad():
            self.theta_embed.weight[0].zero_()
        nn.init.normal_(self.alpha_raw, std=0.1)
        nn.init.normal_(self.beta_base, std=0.05)
        if self.n_categories > 2:
            nn.init.constant_(self.beta_gaps, 0.3)
        # Update network: small weights, gate bias = -2 (starts nearly closed)
        nn.init.normal_(self.update_hidden.weight, std=0.01)
        nn.init.constant_(self.update_hidden.bias, 0.0)
        nn.init.normal_(self.update_gate.weight, std=0.01)
        nn.init.constant_(self.update_gate.bias, -2.0)
        nn.init.normal_(self.update_delta.weight, std=0.01)
        nn.init.constant_(self.update_delta.bias, 0.0)

    # ------------------------------------------------------------------
    # Item parameter helpers
    # ------------------------------------------------------------------

    def _get_item_params(self, q_ids: Tensor):
        """Look up alpha and beta for a batch of item IDs.

        Args:
            q_ids: (B,) long tensor of item IDs.

        Returns:
            alpha: (B, D)
            beta:  (B, K-1)
        """
        alpha = torch.exp(0.3 * self.alpha_raw[q_ids])  # (B, D)
        beta_0 = self.beta_base[q_ids]                  # (B, 1)
        if self.n_categories == 2:
            beta = beta_0
        else:
            gaps = F.softplus(self.beta_gaps[q_ids])    # (B, K-2)
            betas = [beta_0]
            for i in range(gaps.shape[-1]):
                betas.append(betas[-1] + gaps[:, i : i + 1])
            beta = torch.cat(betas, dim=-1)              # (B, K-1)
        return alpha, beta

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
        """Dynamic GPCM forward pass.

        Args:
            student_ids: Long tensor (B, S) — student IDs in [1, N].
            questions:   Long tensor (B, S) — item IDs in [1, Q].
            responses:   Long tensor (B, S) — ordinal responses in [0, K-1].
            mask:        Optional bool tensor (B, S) — True = valid position.

        Returns:
            Dict with keys:
                ``"theta"``   (B, S, D)   — ability trajectory
                ``"alpha"``   (B, S, D)   — item discrimination
                ``"beta"``    (B, S, K-1) — item thresholds
                ``"logits"``  (B, S, K)   — GPCM logits
                ``"probs"``   (B, S, K)   — GPCM probabilities
        """
        B, S = questions.shape
        K = self.n_categories
        D = self.n_traits
        device = questions.device

        # Category values for expected score: [0, 1, ..., K-1]
        cat_vals = torch.arange(K, dtype=torch.float32, device=device)  # (K,)

        # Initialise θ_0 from student embedding (use first column's student ID)
        theta = self.theta_embed(student_ids[:, 0]) * self.ability_scale  # (B, D)

        # Storage
        all_theta  = torch.zeros(B, S, D, device=device)
        all_alpha  = torch.zeros(B, S, D, device=device)
        all_beta   = torch.zeros(B, S, K - 1, device=device)
        all_logits = torch.zeros(B, S, K, device=device)
        all_probs  = torch.zeros(B, S, K, device=device)

        for t in range(S):
            q_t = questions[:, t]          # (B,)
            r_t = responses[:, t].float()  # (B,)

            alpha_t, beta_t = self._get_item_params(q_t)  # (B,D), (B,K-1)

            # Predict at step t using current theta
            theta_t_exp = theta.unsqueeze(1)      # (B, 1, D)
            alpha_t_exp = alpha_t.unsqueeze(1)    # (B, 1, D)
            beta_t_exp  = beta_t.unsqueeze(1)     # (B, 1, K-1)

            logits_t = self.gpcm_logits(theta_t_exp, alpha_t_exp, beta_t_exp)  # (B,1,K)
            probs_t  = self.gpcm_head(logits_t)                                 # (B,1,K)
            logits_t = logits_t.squeeze(1)  # (B, K)
            probs_t  = probs_t.squeeze(1)   # (B, K)

            # Store outputs
            all_theta[:, t, :]  = theta
            all_alpha[:, t, :]  = alpha_t
            all_beta[:, t, :]   = beta_t
            all_logits[:, t, :] = logits_t
            all_probs[:, t, :]  = probs_t

            # Gated update: use r_t to update theta -> theta_{t+1}
            expected_t = (probs_t * cat_vals).sum(dim=-1)  # (B,)
            surprise_t = r_t - expected_t                  # (B,)

            update_in = torch.cat([
                theta,                    # (B, D)
                surprise_t.unsqueeze(-1), # (B, 1)
                alpha_t,                  # (B, D)
                beta_t,                   # (B, K-1)
            ], dim=-1)  # (B, D+1+D+K-1)

            h     = torch.tanh(self.update_hidden(update_in))  # (B, H)
            gate  = torch.sigmoid(self.update_gate(h))         # (B, D)
            delta = torch.tanh(self.update_delta(h))           # (B, D)
            theta = theta + gate * delta                        # (B, D)

        return {
            "theta":  all_theta,
            "alpha":  all_alpha,
            "beta":   all_beta,
            "logits": all_logits,
            "probs":  all_probs,
        }
