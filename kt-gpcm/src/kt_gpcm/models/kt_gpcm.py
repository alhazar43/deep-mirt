"""DeepGPCM — DKVMN-backed neural GPCM model.

Design decisions (see TODO.md Architecture Decisions A1-A6):

A1. ``theta`` and ``alpha`` are always ``(B, S, D)`` tensors.  The
    interaction η = (alpha * theta).sum(-1) degenerates to scalar
    multiplication for D = 1 with no special-casing.

A2. ``forward()`` returns a **dict** with named keys so that adding
    diagnostic fields (e.g. attention weights) never breaks callers.

A3. Both ``logits`` and ``probs`` are returned, eliminating the
    ``return_logits`` flag and ``log(probs + 1e-8)`` workaround.

A4. The learned initial value memory is owned by ``DKVMN``
    (``learned_init=True``) rather than being a loose parameter on this class.

A5. ``tanh`` query transform inside ``DKVMN.attention()`` — preserved.

A6. Embedding vectorised over (B, S) — no Python loop over timesteps
    in ``create_embeddings``.  The DKVMN memory read/write still loops
    over timesteps because memory state is causally sequential.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .components.embeddings import LinearDecayEmbedding, StaticItemEmbedding
from .components.memory import DKVMN
from .components.irt import IRTParameterExtractor, GPCMLogits
from .heads.gpcm import GPCMHead


class DeepGPCM(nn.Module):
    """DKVMN-GPCM model for ordinal student response modelling.

    All constructor parameters are forwarded from ``ModelConfig``; the
    training script builds this with ``DeepGPCM(**vars(cfg.model))``.

    Args:
        n_questions: Number of unique items in the item bank (Q).
        n_categories: Number of ordinal response categories (K ≥ 2).
        n_traits: Latent trait dimension (D ≥ 1).  D = 1 → standard
            GPCM; D > 1 → MIRT with dot-product interaction.
        memory_size: Number of DKVMN memory slots (M).
        key_dim: Key / query vector dimension (d_k).
        value_dim: Value memory slot dimension (d_v).
        summary_dim: Hidden dimension of the summary FC network (d_s).
        ability_scale: Scale applied to raw θ output.
        dropout_rate: Dropout rate in the summary network.
        memory_add_activation: Activation for the DKVMN add gate.
            Only ``"tanh"`` is currently used.
        init_value_memory: If ``True``, the DKVMN uses a learned
            initial value memory (``nn.Parameter``).
    """

    def __init__(
        self,
        n_questions: int,
        n_categories: int = 5,
        n_traits: int = 1,
        memory_size: int = 50,
        key_dim: int = 64,
        value_dim: int = 128,
        summary_dim: int = 50,
        ability_scale: float = 1.0,
        dropout_rate: float = 0.0,
        memory_add_activation: str = "tanh",
        init_value_memory: bool = True,
        response_dim: int = 16,        # separable only
        use_separable_embed: bool = False,  # legacy; overridden by embedding_type
        embedding_type: str = "linear_decay",  # "linear_decay" | "separable" | "static_item"
        item_embed_dim: int = 0,       # static item embed dim H (static_item only); 0 = K*Q auto
    ) -> None:
        super().__init__()

        self.n_questions = n_questions
        self.n_categories = n_categories
        self.n_traits = n_traits
        self.memory_size = memory_size

        # ---- Embedding -------------------------------------------------------
        # Key queries — shared by DKVMN attention and IRT extraction.
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)

        self.embedding_type = embedding_type
        self.use_separable_embed = use_separable_embed  # kept for backward compat

        if embedding_type == "static_item":
            # StaticItemEmbedding: W_item(frozen_H) + W_resp(K) → value_dim directly.
            # No separate value_proj needed — absorbed into StaticItemEmbedding.
            # H = K*Q by default (auto-scaled); pass item_embed_dim to override.
            self.embedding = StaticItemEmbedding(
                n_questions, n_categories, value_dim, item_embed_dim
            )
            self.value_proj = None
        elif embedding_type == "separable" or use_separable_embed:
            # Separable: learned item_embed(key_dim) + triangular(K) → value_proj.
            self.item_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
            self.register_buffer(
                "k_indices",
                torch.arange(n_categories, dtype=torch.float32).view(1, 1, n_categories),
            )
            self.value_proj = nn.Linear(key_dim + n_categories, value_dim)
        else:
            # LinearDecay (default): K*Q ordinal encoding → value_proj.
            self.embedding = LinearDecayEmbedding(n_questions, n_categories)
            self.value_proj = nn.Linear(self.embedding.output_dim, value_dim)

        # ---- DKVMN memory ---------------------------------------------------
        self.memory = DKVMN(
            n_questions=n_questions,
            key_dim=key_dim,
            value_dim=value_dim,
            memory_size=memory_size,
            learned_init=init_value_memory,
        )

        # ---- Summary network ------------------------------------------------
        summary_input_dim = value_dim + key_dim
        self.summary = nn.Sequential(
            nn.Linear(summary_input_dim, summary_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )

        # ---- IRT parameter extractor ----------------------------------------
        self.irt = IRTParameterExtractor(
            input_dim=summary_dim,
            n_questions=n_questions,
            n_categories=n_categories,
            n_traits=n_traits,
            ability_scale=ability_scale,
            question_dim=key_dim,
        )

        # ---- GPCM logit / probability layers --------------------------------
        self.gpcm_logits = GPCMLogits()
        self.gpcm_head = GPCMHead()

        # Keep last attention weights for diagnostics
        self.last_attention: Optional[Tensor] = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming init for embedding and projection layers."""
        nn.init.kaiming_normal_(self.q_embed.weight)
        if self.embedding_type == "separable" or self.use_separable_embed:
            nn.init.kaiming_normal_(self.item_embed.weight)
        if self.value_proj is not None:
            nn.init.kaiming_normal_(self.value_proj.weight)
            nn.init.constant_(self.value_proj.bias, 0.0)
        for module in self.summary:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, questions: Tensor, responses: Tensor) -> dict:
        """Full model forward pass.

        Args:
            questions: Long tensor ``(B, S)`` — item IDs in [1, Q].
                       Padding positions should use ID 0.
            responses: Long tensor ``(B, S)`` — ordinal responses in
                       [0, K-1].

        Returns:
            Dict with keys:
                ``"theta"``   (B, S, D)  — student ability
                ``"alpha"``   (B, S, D)  — item discrimination
                ``"beta"``    (B, S, K-1) — item thresholds
                ``"logits"``  (B, S, K)  — GPCM logits (for loss)
                ``"probs"``   (B, S, K)  — GPCM probabilities (for metrics)
        """
        B, S = questions.shape

        # ---- 1. Key queries (whole sequence at once) -------------------------
        q_embed = self.q_embed(questions)    # (B, S, key_dim) — attention + IRT

        # ---- 2. Value encoding → value_dim ----------------------------------
        if self.embedding_type == "static_item":
            # StaticItemEmbedding outputs value_dim directly; no value_proj.
            value_embed = self.embedding(questions, responses)         # (B, S, value_dim)
        elif self.embedding_type == "separable" or self.use_separable_embed:
            item_v = self.item_embed(questions)                        # (B, S, key_dim)
            dist = torch.abs(self.k_indices - responses.float().unsqueeze(-1)) / (self.n_categories - 1)
            resp_feat = torch.clamp(1.0 - dist, min=0.0)              # (B, S, K)
            value_embed = self.value_proj(
                torch.cat([item_v, resp_feat], dim=-1)                 # (B, S, key_dim+K)
            )                                                          # (B, S, value_dim)
        else:
            embed = self.embedding(questions, responses)               # (B, S, K*Q)
            value_embed = self.value_proj(embed)                       # (B, S, value_dim)

        # ---- 3. Initialise value memory -------------------------------------
        value_mem = self.memory.init_value_memory(B)   # (B, M, value_dim)

        # ---- 4. Sequential DKVMN loop (causal: t reads before t writes) -----
        all_theta, all_alpha, all_beta, all_logits = [], [], [], []
        attn_list = []

        for t in range(S):
            q_t = q_embed[:, t, :]          # (B, key_dim)
            v_t = value_embed[:, t, :]      # (B, value_dim)

            # Attention + read
            attn_t = self.memory.attention(q_t)            # (B, M)
            read_t = self.memory.read(value_mem, attn_t)   # (B, value_dim)
            attn_list.append(attn_t)

            # Summary vector
            summary_input = torch.cat([read_t, q_t], dim=-1)  # (B, value_dim+key_dim)
            summary_t = self.summary(summary_input)             # (B, summary_dim)

            # IRT parameters (unsqueeze S dim, extract, leave as (B, 1, D/K-1))
            theta_t, alpha_t, beta_t = self.irt(
                summary_t.unsqueeze(1),   # (B, 1, summary_dim)
                q_t.unsqueeze(1),         # (B, 1, key_dim)
            )
            # theta_t: (B, 1, D), alpha_t: (B, 1, D), beta_t: (B, 1, K-1)

            # GPCM logits
            logits_t = self.gpcm_logits(theta_t, alpha_t, beta_t)  # (B, 1, K)

            all_theta.append(theta_t)
            all_alpha.append(alpha_t)
            all_beta.append(beta_t)
            all_logits.append(logits_t)

            # Write current timestep embedding into memory
            value_mem = self.memory.write(value_mem, attn_t, v_t)

        # ---- 5. Stack across time dimension ---------------------------------
        theta = torch.cat(all_theta, dim=1)   # (B, S, D)
        alpha = torch.cat(all_alpha, dim=1)   # (B, S, D)
        beta = torch.cat(all_beta, dim=1)     # (B, S, K-1)
        logits = torch.cat(all_logits, dim=1) # (B, S, K)
        probs = self.gpcm_head(logits)         # (B, S, K)

        # Store attention for diagnostics (detached to avoid grad retention)
        self.last_attention = torch.stack(attn_list, dim=1).detach()  # (B, S, M)

        return {
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "logits": logits,
            "probs": probs,
        }
