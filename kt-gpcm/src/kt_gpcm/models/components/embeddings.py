"""Ordinal response embedding for DKVMN-GPCM.

Only ``LinearDecayEmbedding`` is retained from the original
deep-gpcm codebase.  The ABC hierarchy, factory function, and the three
unused strategies (Ordered, Unordered, AdjacentWeighted) are dropped.

Theory
------
For a student at timestep *t* who answered question *q_t* with
ordinal response *r_t* in {0, …, K-1}, the embedding is:

    x_t^(k) = max(0, 1 − |k − r_t| / (K − 1)) · q_t    for k = 0, …, K-1

where *q_t* is the one-hot question vector of length Q.  The triangular
kernel assigns weight 1.0 at the exact category *r_t*, decreasing
linearly to 0.0 at the opposite extreme, so the ordinal structure of
the rating scale is encoded geometrically.

The full embedding is the concatenation of the K per-category weighted
question vectors, giving output dimension K * Q.  The computation is
fully vectorised over (B, S) — no Python loop over timesteps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearDecayEmbedding(nn.Module):
    """Triangular-kernel ordinal embedding (vectorised over batch and time).

    Args:
        n_questions: Number of distinct items in the item bank (Q).
        n_categories: Number of ordinal response categories (K).
        embed_dim: Unused by this class; kept for interface consistency.
            The actual output dimension is always K * Q.

    Shape:
        - Input  ``question_ids``: ``(B, S)`` — integer item IDs in [1, Q].
          Padding ID 0 is treated as question 0 (zero one-hot row).
        - Input  ``responses``: ``(B, S)`` — integer responses in [0, K-1].
        - Output: ``(B, S, K * Q)``
    """

    def __init__(self, n_questions: int, n_categories: int, embed_dim: int = 0) -> None:
        super().__init__()
        self.n_questions = n_questions
        self.n_categories = n_categories

    @property
    def output_dim(self) -> int:
        """Embedding output dimension: K * Q."""
        return self.n_categories * self.n_questions

    def forward(self, question_ids: Tensor, responses: Tensor) -> Tensor:
        """Compute triangular-decay ordinal embeddings.

        The computation is:
            1.  Build one-hot question matrix  (B, S, Q)  from question IDs.
            2.  Compute per-category weights   (B, S, K)  from responses.
            3.  Outer product + flatten →      (B, S, K*Q).

        Args:
            question_ids: Long tensor ``(B, S)`` with item IDs in [0, Q].
                          ID 0 is treated as a padding / unknown item.
            responses: Long tensor ``(B, S)`` with ordinal responses in
                       [0, K-1].

        Returns:
            Float tensor of shape ``(B, S, K * Q)``.
        """
        B, S = responses.shape
        device = responses.device

        # --- Build one-hot question matrix (B, S, Q) ----------------------
        # question_ids range: 1..Q (0 = padding).  We allocate Q+1 classes
        # then strip the padding column, giving a proper Q-dim one-hot.
        q_one_hot = F.one_hot(question_ids, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]   # (B, S, Q) — drop padding dim

        # --- Triangular weights over categories (B, S, K) -----------------
        # k_indices: (1, 1, K), r_expanded: (B, S, 1)
        k_indices = torch.arange(self.n_categories, device=device, dtype=torch.float32)
        k_indices = k_indices.unsqueeze(0).unsqueeze(0)      # (1, 1, K)
        r_expanded = responses.float().unsqueeze(-1)          # (B, S, 1)

        distance = torch.abs(k_indices - r_expanded) / (self.n_categories - 1)
        weights = torch.clamp(1.0 - distance, min=0.0)        # (B, S, K)

        # --- Outer product: (B, S, K, Q) → flatten → (B, S, K*Q) ---------
        # weights: (B, S, K, 1) * q_one_hot: (B, S, 1, Q) → (B, S, K, Q)
        weighted_q = weights.unsqueeze(-1) * q_one_hot.unsqueeze(2)
        embedded = weighted_q.view(B, S, self.n_categories * self.n_questions)
        return embedded


class StaticItemEmbedding(nn.Module):
    """Factored additive value embedding with static (frozen) random item projection.

    Replaces ``LinearDecayEmbedding + value_proj`` with two independent
    learned linear branches:

        v = W_item @ e_item  +  W_resp @ triangular_resp

    where ``e_item`` is a **frozen** (non-trained) random unit-norm vector
    of dimension H and ``triangular_resp`` is the same K-dim ordinal weight
    used inside ``LinearDecayEmbedding``.

    Why this helps alpha recovery vs separable embedding
    -----------------------------------------------------
    In the separable design, ``W_value_proj`` receives gradients from every
    item simultaneously — gradient interference corrupts item-specific
    discrimination signals in memory.

    Here the factored structure **separates** two gradient channels:

    * ``W_item``: updated only by the item-identity signal.  Cross-item
      interference is ``O(1/√H)`` — for ``H=K*Q`` this is ≈ 1/√(K*Q).
    * ``W_resp``: updated by the response-pattern signal, shared across all
      items.  This is correct — ordinal response patterns are item-agnostic.

    Freezing ``e_item`` prevents items from collapsing to similar dense
    representations during training (a failure mode of learned separable
    embeddings).

    H auto-scaling
    --------------
    By default H = next_power_of_2(Q // 2), clamped to [128, 1024]:
        Q ≤ 200  → H = 128
        Q ≈ 500  → H = 256
        Q ≈ 1000 → H = 512
        Q ≥ 1500 → H = 1024  (cap)
    Cross-item dot products are O(1/√H), so H grows with Q to prevent
    oversparse item representations.  Pass an explicit ``item_embed_dim``
    to override.

    Scalability
    -----------
    For any Q, the compute cost is ``O(H + K)`` per step (two small matmuls)
    rather than ``O(K * Q)`` for ``LinearDecayEmbedding`` at large Q.

    Shape
    -----
    - Input  ``question_ids``: ``(B, S)`` — integer item IDs in [1, Q].
    - Input  ``responses``:   ``(B, S)`` — integer responses in [0, K-1].
    - Output: ``(B, S, value_dim)`` — already projected, no external value_proj needed.

    Args:
        n_questions:    Item bank size Q.
        n_categories:   Ordinal response categories K.
        value_dim:      Output / value memory dimension.
        item_embed_dim: Frozen random item embedding dimension H.
                        Default 0 → auto-set to K * Q.
    """

    def __init__(
        self,
        n_questions: int,
        n_categories: int,
        value_dim: int,
        item_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.value_dim = value_dim
        # Auto H: next power of 2 of (Q // 2), clamped to [128, 1024].
        # Grows with Q (avoiding oversparse) but stays GPU-friendly:
        #   Q≤200 → 128,  Q≈500 → 256,  Q≈1000 → 512,  Q≥1500 → 1024 (cap).
        if item_embed_dim > 0:
            self.item_embed_dim = item_embed_dim
        else:
            raw = max(128, n_questions // 2)
            self.item_embed_dim = min(1024, 1 << math.ceil(math.log2(raw)))

        # Frozen random item embeddings — unit-norm rows, no gradient.
        # Shape: (Q+1, H); index 0 = padding (zero).
        item_vecs = torch.randn(n_questions + 1, self.item_embed_dim)
        item_vecs = F.normalize(item_vecs, dim=-1)
        item_vecs[0] = 0.0
        self.register_buffer("item_embed", item_vecs)   # not a Parameter

        # Triangular response weight index (1, 1, K) — reused every forward.
        k_idx = torch.arange(n_categories, dtype=torch.float32).view(1, 1, n_categories)
        self.register_buffer("k_indices", k_idx)

        # Factored learned projections — small, independent gradient channels.
        self.W_item = nn.Linear(self.item_embed_dim, value_dim, bias=False)
        self.W_resp = nn.Linear(n_categories, value_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # Scale so each branch contributes ~equal initial output variance.
        nn.init.normal_(self.W_item.weight, std=1.0 / math.sqrt(self.item_embed_dim))
        nn.init.normal_(self.W_resp.weight, std=1.0 / math.sqrt(self.n_categories))
        nn.init.zeros_(self.W_resp.bias)

    @property
    def output_dim(self) -> int:
        """Output is already value_dim — no external value_proj needed."""
        return self.value_dim

    def forward(self, question_ids: Tensor, responses: Tensor) -> Tensor:
        """
        Args:
            question_ids: ``(B, S)`` int64, 1-indexed item IDs.
            responses:    ``(B, S)`` int64, 0-indexed ordinal responses.

        Returns:
            ``(B, S, value_dim)``
        """
        # Item branch: frozen lookup → learned projection
        e_item = self.item_embed[question_ids]                     # (B, S, H)
        v_item = self.W_item(e_item)                               # (B, S, value_dim)

        # Response branch: triangular ordinal weights → learned projection
        dist = torch.abs(self.k_indices - responses.float().unsqueeze(-1)) / (self.n_categories - 1)
        w_resp = torch.clamp(1.0 - dist, min=0.0)                 # (B, S, K)
        v_resp = self.W_resp(w_resp)                               # (B, S, value_dim)

        return v_item + v_resp                                     # (B, S, value_dim)
