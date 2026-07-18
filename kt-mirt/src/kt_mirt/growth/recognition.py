"""Full-window amortized recognition network for ACT
(`_planning/design/a4_design.md` v1.1, section 2.3): emits per-learner
``u_i`` (ACT-P0, ACT-P1) and ``lambda_i`` (ACT-P1 only) from the learner's
FULL conditioning window, never a per-step causal read (u_i/lambda_i are
population-level per-learner summaries consumed ONCE to seed the
response-blind transition, not per-position predictions).

Interpretation note (recorded per the harness instructions): the vendor
report names two candidate attachment points for this net, "the
encoder's final state / `state_for_prediction`". `state_for_prediction`'s
LAST position is deliberately causal-shifted (it sees history strictly
before the final step, excluding the learner's very last observed
response, by the core's own one-shift alignment contract). Since the
design's own words are "full conditioning window" and the recognition
inputs are read ONCE per learner (not used to predict any individual
step, so there is no leakage concern to guard against), this module reads
the encoder's RAW (non-causal) final hidden state instead -- the same
`_direct_hidden` pass every encoder backbone already computes internally,
just without the extra shift -- so ``u_i``/``lambda_i`` are conditioned on
literally the FULL window, including the learner's last response. Battery
arm 5 (order-invariance stress, CG9-ACT) is the empirical check on
whether this choice is stable under interleaving reshuffles; nothing here
enforces invariance architecturally (section 2.3's own point: "instability
can enter only through them").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kt_mirt.core.encoder import BaseSeqEncoder, LSTMEncoder


@dataclass
class RecognitionConfig:
    emb_dim: int = 8
    hidden_dim: int = 32
    predict_lambda: bool = False  # True for ACT-P1


class RecognitionNetwork(nn.Module):
    """Reads a learner's full (item_ids, responses) window and emits a
    scalar ``u_i`` (and, if configured, a positive scalar ``lambda_i``)."""

    def __init__(self, num_items: int, cfg: RecognitionConfig, encoder: Optional[BaseSeqEncoder] = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else LSTMEncoder(
            num_items, emb_dim=cfg.emb_dim, hidden_dim=cfg.hidden_dim, n_cats=2
        )
        self.u_head = nn.Linear(cfg.hidden_dim, 1)
        self.predict_lambda = cfg.predict_lambda
        if cfg.predict_lambda:
            self.lambda_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self, item_ids: torch.Tensor, responses: torch.Tensor, seq_lens: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``item_ids``/``responses``: (B, T_max) padded. ``seq_lens``:
        (B,) each learner's true sequence length. Returns ``(u_i (B,),
        lambda_i (B,) or None)``."""
        h = self.encoder._direct_hidden(item_ids, responses)  # (B, T, H), raw, full window
        idx = (seq_lens - 1).clamp(min=0)
        final_h = h[torch.arange(h.shape[0], device=h.device), idx]  # (B, H)
        u_i = self.u_head(final_h).squeeze(-1)
        lam = F.softplus(self.lambda_head(final_h).squeeze(-1)) if self.predict_lambda else None
        return u_i, lam


__all__ = ["RecognitionConfig", "RecognitionNetwork"]
