"""A1 signed cross-KC influence model (`_planning/design/a1_design.md`
v1.1, section 2.2): the A4 ACTIVE per-KC growth model PLUS exactly one new
route, a fitted zero-diagonal signed matrix ``G[c, a]`` whose transfer term
is PRACTICE-GATED (driven by the source-practice indicator, never the
response) and SIGN-ASYMMETRIC-GATED (positive transfer ceiling-gated,
negative transfer floor-gated). Nothing else about the certified A4
substrate changes.

This module builds ONLY the machinery CT0 (section 2.1.1) needs: the
signed-influence model, its response-blind transfer transition, and a
convergence-gated two-stage fit. The full certification battery
(matched-null contrast, confound arms, external alignment, low-rank
factor) is out of CT0 scope and lives in later stages.

Transition (design section 2.2, the only state-moving law)::

    z_{i,c,t+1} = z_{i,c,t}
        + lambda_i * g_c * (M - z_{i,c,t})+ * 1[c practiced at t]      (own-gain, A4 ACT)
        + SUM_{a != c} 1[a practiced at t] * T(G[c,a], z_{i,c,t})       (cross-KC transfer, NEW)

with the sign-asymmetric ceiling/floor gate ("you can only lose what you
have built")::

    T(g, z) =  g * (M - z)+       if g >= 0   (facilitation, ceiling-gated)
               g * (z - floor)+   if g <  0   (interference, floor-gated)

Load-bearing identification choices, each tracing to a named design
constraint (section 2.2):

1. PRACTICE-GATED, RESPONSE-BLIND. The transfer term reads only the
   practice indicator ``1[a practiced at t]`` -- exactly the item tags
   already carried in ``kc_ids[:, t, :]`` (attachment point #2, design
   section 2.4), read structurally, never gated by ``y``. `run_transfer_
   transition` is a pure function of the practice schedule and the
   amortized seed ``(z0, lambda)``; given a fixed seed its output is
   invariant to permuting the responses (asserted in the tests).
2. G IS THE ONLY CROSS-KC ROUTE. With ``G = 0`` a non-practiced KC's ``z``
   never leaves ``z0`` (the qmirt isolation property; unit-tested here).
   The encoder is DEMOTED to the A4 `recognition.RecognitionNetwork`,
   which emits only ``(u_i, lambda_i)`` and is reused BYTE-IDENTICAL.
3. PER-LEARNER TRANSFER MULTIPLIER PINNED at 1 by construction. There is
   no free or amortized per-learner transfer trait. The optional
   `phantom_gamma` head (off by default) is the SECTION-4.6 sensitivity
   control for a later stage, NOT part of CT0; when enabled it adds a
   separate amortized head here and never edits `RecognitionNetwork`.
4. OWN-GAIN CEILING-GATED, ``g_c`` positive via softplus (a negative
   own-gain would reopen the unpracticed-decline channel). Negative
   transfer is the only downward channel and it is practice-gated and
   floor-bounded, so ``rho = 1`` / ``mu = 0`` are kept (design Lemma 2).
5. ZERO DIAGONAL is a HARD constraint: the diagonal of ``G`` is never a
   parameter (masked out), so no KC influences itself through the transfer
   route.
6. OFF-DIAGONAL L1. ``G`` carries an additive L1 penalty for sparsity,
   applied as an Adam subgradient (design section 2.3: "slow per-cell
   shrinkage, no exact zeros"), which is why downstream sign scoring
   thresholds ``G`` against a matched-null / permutation band rather than
   against a bare zero (`ct0.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kt_mirt.core.encoder import BaseSeqEncoder
from kt_mirt.growth.recognition import RecognitionConfig, RecognitionNetwork
from kt_mirt.growth.tracker import LearnerBatch


# ---------------------------------------------------------------------------
# Ceiling / floor initialization
# ---------------------------------------------------------------------------


def ceiling_init(b_hat: np.ndarray) -> float:
    """Positive-transfer ceiling ``M`` (reused from ACT, `active.ceiling_
    init`): the 95th percentile of calibrated difficulties plus 2."""
    b_hat = np.asarray(b_hat, dtype=float)
    return float(np.percentile(b_hat, 95) + 2.0)


def floor_init(b_hat: np.ndarray) -> float:
    """Negative-transfer floor (the mirror of `ceiling_init`, design
    section 2.2's floor-gate): the 5th percentile of calibrated
    difficulties minus 2. Kept as one global scalar, fixed (not fitted) so
    the interference gate's lower bound is a stated modeling choice, not a
    fitting artifact."""
    b_hat = np.asarray(b_hat, dtype=float)
    return float(np.percentile(b_hat, 5) - 2.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


#: Floor on the transfer trainer's epoch ceiling: the earliest the windowed
#: convergence gate can fire (``2*window_epochs + patience_epochs``), so a
#: caller cannot request FEWER epochs than the gate needs to even evaluate.
#: The gate (or ``cfg.n_epochs``) governs the actual stop. The frozen,
#: re-derived trainer numbers for the L1-penalized G objective are the
#: design's R0-A1 stationarity re-study, OUT of CT0 scope; CT0 is a
#: fail-fast power screen and uses the pragmatic ceiling ``cfg.n_epochs``,
#: verified generous enough for the D=3 fits to separate true edges from
#: zero cells (a best-case gate-fixed fit recovers the reference dose in a
#: few hundred epochs).
TRANSFER_MIN_EPOCHS_CEILING = 125


@dataclass
class TransferConfig:
    variant: str = "a1_p0"  # "a1_p0" (lambda_i pinned at 1) | "a1_p1" (amortized lambda_i)
    emb_dim: int = 8
    hidden_dim: int = 16
    lr: float = 5e-2
    n_epochs: int = 500  # CEILING (floored at TRANSFER_MIN_EPOCHS_CEILING), not a fixed count
    window_epochs: int = 50  # block size for the windowed-mean loss statistic (ACT note 8)
    patience_epochs: int = 25  # consecutive epochs the joint criterion must hold
    rel_tol: float = 1e-5  # windowed-mean relative loss change leg
    drift_tol: float = 5e-2  # macro guard on per-epoch [g_c, v_c, M, G] drift
    l1_weight: float = 1e-3  # off-diagonal L1 penalty on G (subgradient)
    floor: float = -6.0  # negative-transfer floor (set from floor_init at build time)
    m_fixed: bool = False  # fix M at m_init (isolates sign recovery from a drifting ceiling)
    phantom_gamma: bool = False  # section 4.6 sensitivity control; OFF for CT0
    seed: int = 0
    device: str = "cpu"


# ---------------------------------------------------------------------------
# The sign-asymmetric transfer gate
# ---------------------------------------------------------------------------


def sign_asymmetric_transfer(
    g: torch.Tensor, z: torch.Tensor, M: torch.Tensor, floor: float
) -> torch.Tensor:
    """``T(g, z)`` of design section 2.2, vectorized and differentiable:
    positive coefficients are ceiling-gated ``g*(M-z)+``, negative
    coefficients floor-gated ``g*(z-floor)+``. ``g`` and ``z`` broadcast."""
    g_pos = F.relu(g)
    g_neg = g - g_pos  # == min(g, 0)
    return g_pos * F.relu(M - z) + g_neg * F.relu(z - floor)


def _sign_asymmetric_np(g: np.ndarray, z: np.ndarray, M: float, floor: float) -> np.ndarray:
    """NumPy twin of `sign_asymmetric_transfer`, for the generator
    (`transfer/synth.py`) so generator and model SHARE the gate exactly
    (the qmirt discipline, design section 3.1)."""
    g = np.asarray(g, dtype=float)
    z = np.asarray(z, dtype=float)
    g_pos = np.maximum(g, 0.0)
    g_neg = g - g_pos
    return g_pos * np.maximum(M - z, 0.0) + g_neg * np.maximum(z - floor, 0.0)


# ---------------------------------------------------------------------------
# The response-blind transfer transition
# ---------------------------------------------------------------------------


def run_transfer_transition(
    z0: torch.Tensor,
    lam: torch.Tensor,
    g_c: torch.Tensor,
    M: torch.Tensor,
    G: torch.Tensor,
    floor: float,
    kc_ids: torch.Tensor,
    kc_mask: torch.Tensor,
    b_j: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extends `active.run_transition` with the cross-KC transfer route.

    ``z0`` (B,C), ``lam`` (B,), ``g_c`` (C,), ``M`` scalar, ``G`` (C,C)
    zero-diagonal, ``floor`` scalar, ``kc_ids``/``kc_mask`` (B,T,A),
    ``b_j`` (B,T), optional ``gamma`` (B,) per-learner transfer multiplier
    (pinned to 1 == ``None`` for CT0). Returns ``(logits (B,T), z_final
    (B,C))``.

    Response-blind (design constraint 1): reads no response tensor. Both
    the own-gain and the transfer increments are computed from the SAME
    pre-update ``Z`` (the design's simultaneous transition) and summed into
    one delta before the state advances, so ``logits[:, t]`` depends on
    history strictly before ``t`` (causal, multi-tag readout verbatim from
    `active.run_transition`)."""
    B, T_max, A = kc_ids.shape
    Z = z0
    logits = torch.zeros(B, T_max, device=Z.device, dtype=Z.dtype)
    G_t = G.t()  # (C_source, C_target) row-indexed by source for the column gather
    for t in range(T_max):
        tag_ids_t = kc_ids[:, t, :].clamp(min=0)  # (B, A)
        tag_mask_t = kc_mask[:, t, :].to(Z.dtype)  # (B, A)
        z_tagged = torch.gather(Z, 1, tag_ids_t)  # (B, A), PRE-update, causal
        denom = tag_mask_t.sum(dim=-1).clamp(min=1.0)
        mean_z = (z_tagged * tag_mask_t).sum(dim=-1) / denom
        logits[:, t] = mean_z - b_j[:, t]

        delta = torch.zeros_like(Z)  # (B, C), accumulated from the pre-update Z only

        # Own-gain (A4 ACT), scattered into the practiced KC columns.
        g_gathered = g_c[tag_ids_t]  # (B, A)
        own = lam.unsqueeze(-1) * g_gathered * F.relu(M - z_tagged) * tag_mask_t
        delta = delta.scatter_add(1, tag_ids_t, own)

        # Cross-KC transfer: every practiced source a pushes ALL targets c
        # by the sign-asymmetric-gated G[:, a], gated on each target's
        # CURRENT (pre-update) state. Looping the <= A practiced slots keeps
        # this O(B*C*A) per step (A tiny under practice-gating; A == 1 at
        # the KDD-shaped single-tag CT0 profile).
        for s in range(A):
            a_ids = tag_ids_t[:, s]  # (B,)
            a_m = tag_mask_t[:, s].unsqueeze(-1)  # (B, 1)
            G_cols = G_t[a_ids]  # (B, C): G_cols[b, c] = G[c, a_ids[b]]
            transfer = sign_asymmetric_transfer(G_cols, Z, M, floor)  # (B, C)
            if gamma is not None:
                transfer = transfer * gamma.unsqueeze(-1)
            delta = delta + transfer * a_m

        Z = Z + delta
    return logits, Z


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class TransferModel(nn.Module):
    """ACT own-gain + the signed cross-KC route. The ONLY learnable objects
    beyond A4's set are the ``(C, C)`` off-diagonal ``G`` table (design
    section 2.4). No free per-learner parameter exists (Gate B): the
    recognition network's weights, the population ``g_c``/``v_c``/``M``, and
    ``G`` are the entire learnable set. ``gamma`` is pinned to 1 unless the
    section-4.6 phantom head is explicitly enabled."""

    def __init__(
        self,
        num_items: int,
        n_kcs: int,
        cfg: TransferConfig,
        m_init: float,
        encoder: Optional[BaseSeqEncoder] = None,
    ):
        super().__init__()
        if cfg.variant not in ("a1_p0", "a1_p1"):
            raise ValueError(f"unknown A1 variant {cfg.variant!r}, expected 'a1_p0' or 'a1_p1'")
        self.variant = cfg.variant
        self.n_kcs = n_kcs
        self.floor = float(cfg.floor)
        predict_lambda = cfg.variant == "a1_p1"
        rec_cfg = RecognitionConfig(
            emb_dim=cfg.emb_dim, hidden_dim=cfg.hidden_dim, predict_lambda=predict_lambda
        )
        self.recognition = RecognitionNetwork(num_items, rec_cfg, encoder=encoder)
        self.g_raw = nn.Parameter(torch.zeros(n_kcs))  # g_c = softplus(g_raw), positive
        self.v_c = nn.Parameter(torch.zeros(n_kcs))
        self.G_off = nn.Parameter(torch.zeros(n_kcs, n_kcs))  # off-diagonal signed transfer
        self.register_buffer("_offdiag", 1.0 - torch.eye(n_kcs))  # hard zero-diagonal mask
        if cfg.m_fixed:
            self.register_buffer("_M", torch.tensor(float(m_init)))
        else:
            self._M = nn.Parameter(torch.tensor(float(m_init)))

        self.phantom_gamma = bool(cfg.phantom_gamma)
        if self.phantom_gamma:
            # Section 4.6 sensitivity control: a SEPARATE amortized head
            # built here, never an edit to RecognitionNetwork. Off for CT0.
            self.gamma_head = nn.Linear(cfg.hidden_dim, 1)

    @property
    def M(self) -> torch.Tensor:
        return self._M

    @property
    def g_c(self) -> torch.Tensor:
        return F.softplus(self.g_raw)

    @property
    def G(self) -> torch.Tensor:
        """The signed transfer matrix with its diagonal HARD-zeroed."""
        return self.G_off * self._offdiag

    def _maybe_gamma(self, batch: LearnerBatch, seq_lens: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.phantom_gamma:
            return None  # gamma pinned to 1 (CT0)
        h = self.recognition.encoder._direct_hidden(batch.item_ids, batch.responses)  # (B, T, H)
        idx = (seq_lens - 1).clamp(min=0)
        final_h = h[torch.arange(h.shape[0], device=h.device), idx]
        return F.softplus(self.gamma_head(final_h).squeeze(-1))

    def forward(
        self, batch: LearnerBatch, seq_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns ``(logits (B,T), z_final (B,C), u_i (B,), lambda_i (B,)
        or None)``. `recognition` reads ``batch.responses`` to amortize
        ``(u_i, lambda_i)`` (its entire job); `run_transfer_transition`
        then runs the response-blind recurrence given only that seed and
        the practice tags."""
        u_i, lam = self.recognition(batch.item_ids, batch.responses, seq_lens)
        lam_eff = lam if lam is not None else torch.ones_like(u_i)
        z0 = u_i.unsqueeze(-1) + self.v_c.unsqueeze(0).expand(u_i.shape[0], -1)  # (B, C)
        gamma = self._maybe_gamma(batch, seq_lens)
        logits, z_final = run_transfer_transition(
            z0, lam_eff, self.g_c, self.M, self.G, self.floor, batch.kc_ids, batch.kc_mask, batch.b_j, gamma
        )
        return logits, z_final, u_i, lam

    @torch.no_grad()
    def recovered_G(self) -> np.ndarray:
        """The fitted signed matrix as a NumPy array, diagonal zeroed --
        the object CT0 scores for sign recovery."""
        return self.G.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Loss, training, forecast NLL
# ---------------------------------------------------------------------------


def seq_lens_from_mask(seq_mask: torch.Tensor) -> torch.Tensor:
    return seq_mask.sum(dim=1).long()


def transfer_loss(
    model: TransferModel, batch: LearnerBatch, seq_lens: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(nll, l1)``: the Bernoulli forecast NLL against the frozen
    bank (design section 2.2's readout, identical to `active.active_loss`)
    and the off-diagonal L1(G) sparsity term (design section 2.3's new
    additive penalty), kept SEPARATE so training can weight them and the
    diagnostics can read the pure NLL. The diagonal is already zero in
    ``model.G``; ``.abs().sum()`` is therefore an off-diagonal L1."""
    if seq_lens is None:
        seq_lens = seq_lens_from_mask(batch.seq_mask)
    logits, _, _, _ = model(batch, seq_lens)
    y = batch.responses.to(logits.dtype)
    per_pos = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = batch.seq_mask.to(per_pos.dtype)
    nll = (per_pos * mask).sum() / mask.sum().clamp(min=1.0)
    l1 = model.G.abs().sum()
    return nll, l1


def _param_snapshot(model: TransferModel) -> np.ndarray:
    """Drift-tracked quantities: the population growth parameters AND the
    transfer matrix (design section 2.3 extends the ACT drift snapshot to
    track G, since G's true-zero cells are exactly what the sign gates
    read). Flat no-grad vector of ``[g_c, v_c, M, vec(G)]``."""
    with torch.no_grad():
        g_c = model.g_c.detach().cpu().numpy().reshape(-1)
        v_c = model.v_c.detach().cpu().numpy().reshape(-1)
        m = model.M.detach().cpu().numpy().reshape(-1)
        g = model.G.detach().cpu().numpy().reshape(-1)
    return np.concatenate([g_c, v_c, m, g]).astype(float)


def train_transfer(model: TransferModel, batch: LearnerBatch, cfg: TransferConfig) -> list[float]:
    """Stage 2 of the two-stage fit (stage 1, bank freeze, is
    `bank.calibrate_bank`/`freeze_bank`; on CT0's clean synthetic power
    screen the bank is the generator's true difficulties, fed through
    ``batch.b_j``). Fit ``{g_c, v_c, M, G, recognition}`` jointly by
    forecast NLL plus the L1(G) subgradient.

    Convergence-gated, mirroring `active.train_active`'s STRUCTURE (windowed-
    mean relative-loss leg, macro parameter-drift leg, epoch-ceiling floor)
    but with the drift snapshot extended to include ``G`` (design section
    2.3). The trainer NUMBERS here are pragmatic CT0 defaults; the frozen
    R0-A1 re-derivation for the certification matrix is a later stage. The
    returned trace is the pure forecast NLL per epoch (not the penalized
    objective), so it is comparable across L1 weights."""
    seq_lens = seq_lens_from_mask(batch.seq_mask)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    max_epochs = max(int(cfg.n_epochs), TRANSFER_MIN_EPOCHS_CEILING)
    W = int(cfg.window_epochs)

    trace: list[float] = []
    prev = _param_snapshot(model)
    good_epochs = 0
    for _ in range(max_epochs):
        optimizer.zero_grad()
        nll, l1 = transfer_loss(model, batch, seq_lens)
        loss = nll + cfg.l1_weight * l1
        loss.backward()
        optimizer.step()
        trace.append(float(nll.item()))

        cur = _param_snapshot(model)
        drift = float(np.max(np.abs(cur - prev))) if prev.size else 0.0
        prev = cur

        if len(trace) >= 2 * W:
            m1 = float(np.mean(trace[-W:]))
            m0 = float(np.mean(trace[-2 * W : -W]))
            rel_windowed = abs(m1 - m0) / max(abs(m0), 1e-8)
            if rel_windowed < cfg.rel_tol and drift < cfg.drift_tol:
                good_epochs += 1
            else:
                good_epochs = 0
            if good_epochs >= cfg.patience_epochs:
                break
    return trace


@torch.no_grad()
def forecast_nll(model: TransferModel, batch: LearnerBatch) -> float:
    """Causal next-step forecast NLL on a held-out batch (the pure NLL, no
    L1 term)."""
    nll, _ = transfer_loss(model, batch)
    return float(nll.item())


__all__ = [
    "ceiling_init",
    "floor_init",
    "TRANSFER_MIN_EPOCHS_CEILING",
    "TransferConfig",
    "sign_asymmetric_transfer",
    "run_transfer_transition",
    "TransferModel",
    "seq_lens_from_mask",
    "transfer_loss",
    "train_transfer",
    "forecast_nll",
]
