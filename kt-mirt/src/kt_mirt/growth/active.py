"""ACT, the ACTIVE posture's minimal gain-gated growth model
(`_planning/design/a4_design.md` v1.1, section 2.3), plus RB-A (section
5.2), the pre-registered real-bed firing definition that decides whether
an E-A1 read counts as ACT "firing".

Transition (verbatim, the only state-moving term):
``z_ic,t+1 = z_ic,t + lambda_i * g_c * (M - z_ic,t)+ * 1[item at t tagged c]``
-- practice-gated (moves only on a tagged interaction), ceiling-gated
(``(M - z)+`` diminishing returns), response-blind (never reads ``y``).
Pins: ``mu = 0`` (no mean-reversion target -- there is no such term in the
formula above, so nothing to implement for it beyond its absence) and
``rho = 1`` (no decay channel). Per-learner quantities are NEVER free
parameters (Gate B): ``z0_ic = u_i + v_c`` with ``u_i`` amortized by
`recognition.RecognitionNetwork` over the learner's full window and
``v_c`` a free POPULATION per-KC offset; ACT-P1 additionally amortizes a
per-learner ``lambda_i`` (ACT-P0 pins ``lambda_i = 1``). The ceiling ``M``
is one global scalar, fitted (default) or fixed at its initialization
(the pre-registered CG1 fallback, invoked via ``cfg.m_fixed``).

Interpretation notes (recorded per the harness instructions):

1. **Multi-tag readout, mirroring PAS-N1's stated pattern.** The design
   does not spell out how ACT should combine a multi-tag item's several
   per-KC states into one predicted logit, but section 2.5 gives exactly
   this rule for PAS-N1 ("per-KC gather averages over the item's tagged
   slots"). This module applies the SAME rule: the predicted logit at
   position ``t`` is ``mean_{c in tags(t)}(z_ic,t) - b_j``, using the
   PRE-update ``z`` (causal: it is a function of history strictly before
   ``t``, matching the vendored core's own causal-alignment convention).
   The transition then applies independently to EVERY tagged KC's own
   column, exactly as the formula specifies -- the shared-readout choice
   only affects what the loss predicts, never which columns move.
2. **``g_c`` is constrained positive via ``softplus``.** The design pins
   ``rho = 1`` specifically so that "the pin ... structurally bars ACT
   from ever representing an individual decline" (section 0; R1-I9's
   disposition, section 10). A free-signed ``g_c`` would reopen exactly
   that channel (a negative ``g_c`` times a positive ceiling gap is a
   decline), so positivity is enforced architecturally, not left to the
   optimizer to discover empirically.
3. **``z0_ic`` is defined for every ``(learner, KC)`` pair at "time 0"**
   (before the learner's first interaction of any kind), not only for
   KCs the learner goes on to practice -- this is what "initial state"
   means for a state that only moves on practice (an unpracticed KC's
   ``z`` simply never leaves ``z0``). Implemented as one dense ``(B, C)``
   tensor initialized to ``u_i + v_c`` (broadcast) before the recurrence.
4. **Population-level CG1/RB-A statistics are computed by a closed-form
   deterministic recurrence in latent ``z``-units** (`implied_z_
   trajectory`), not by replaying real interaction data. Section 2.3 states
   the transition, "as a function of its own opportunity index, [is]
   invariant to cross-KC interleaving by construction" -- so the model-
   implied rise over a KC's own opportunities 1..10 depends only on
   ``(z0_ic, lambda_i, g_c, M)``, never on the realized schedule. This
   also fixes the granularity of "population mean and p95 per-learner"
   (section 2.3): read literally against section 1's own usage ("Per-
   slice (per-learner) estimands"), "per-learner" here means the
   per-``(learner, KC)``-slice statistic -- population mean = the mean of
   that per-slice rise over every slice, p95 = its 95th percentile over
   the same population. This is the same slice granularity every other
   per-KC estimator in this package reports at.
5. **RB-A's magnitude/null-ratio/sign-consistency read is a pure function
   over already-computed per-seed rise arrays** (`rb_a_firing`), so a
   caller (this module's tests, or `battery.py`, a later stage) can feed
   it either a real fitted-model rise or a synthetic no-growth-twin rise
   without this module needing to know which twin or bed produced either
   number.
6. **Batch construction and the recognition network are reused, not
   duplicated.** ACT's forward pass consumes `tracker.LearnerBatch` /
   `tracker.build_learner_batch` (PAS-N1's own full-interleaved-sequence
   batch shape is exactly what ACT's global-time recurrence needs) and
   `recognition.RecognitionNetwork` (already built by this same stage) for
   ``(u_i, lambda_i)``. No second batch-construction or recognition
   implementation exists in this module.
7. **The saturation reporting rule** (section 2.3: "ACT latent-scale
   gains g_c are reported only on unsaturated KCs ... on saturation-
   flagged KCs only score-scale implied changes are reported") is exposed
   as a one-line masking helper (`report_gain`) here, since it is a
   trivial function of this module's own ``g_c`` parameter and the raw
   saturation flag `slices.saturation_stats` already computes; the report
   ASSEMBLY (deciding which KCs are saturation-flagged on a given bed,
   writing the verdict text) remains `report.py`'s job (a later stage).
8. **`train_active` is convergence-gated, mirroring `bank.calibrate_bank`'s
   dual criterion in structure (`bank.py:664-708`), not a bare fixed-epoch
   loop.** The ACT-P0 fabrication defect diagnosed in
   `_planning/research/act_p0_diagnosis.md` was a training-convergence
   pathology, not a design or transition-logic bug: the old bare
   ``for _ in range(cfg.n_epochs)`` loop (every real caller passing 20)
   stopped Adam one to two orders of magnitude before ``g_c`` could move
   from its ``softplus(0) ~ 0.69`` init down toward the near-zero region
   no-growth data demands, while the same estimator, given enough epochs
   from the SAME init, descends correctly on no-growth data and
   accurately recovers the true gain on known-growth data (diagnosis
   section 3.2). The fix touches only the OPTIMIZATION budget, never the
   model, the pins, or the transition. One deliberate refinement over a
   verbatim copy of `bank.calibrate_bank`'s rule, forced by a
   stationarity study on both twins (3 seeds x 3000 epochs, 2026-07-18):
   ACT trains FULL-batch, so at any genuine optimum Adam orbits a limit
   cycle whose per-epoch loss change and parameter drift measure the
   CYCLE amplitude (measured drift floor 1.5e-3 to 2.4e-2), not
   stationarity, while on no-growth data the descent is so flat that
   per-epoch changes are quiet long before convergence -- bank's exact
   per-epoch rule therefore stops no-growth fits mid-descent (~300
   epochs, with the implied rise still 2-3x its converged value) and can
   never satisfy a tight drift tolerance on known-growth fits. The
   relative-NLL leg is therefore computed on a WINDOWED MEAN
   (``cfg.window_epochs``-epoch blocks): averaging cancels the limit
   cycle but preserves a descent trend. ``cfg.rel_tol = 1e-5`` sits ~2x
   above the measured windowed noise floor (median ~5e-6 on the stable
   runs; late-phase cycling pushes it higher on some seeds, which only
   delays the stop); the parameter-drift leg (over ``g_c``, ``v_c``, ``M``,
   the exact quantities the closed-form CG1/RB-A readouts consume) is
   kept as a macro-movement guard with ``cfg.drift_tol`` deliberately
   ABOVE the measured cycle floor, since a tolerance below that floor is
   unreachable at any true optimum. ``cfg.n_epochs`` keeps its name and
   its role as an upper bound (backward-compatible signature), but is
   floored at ``ACT_MIN_EPOCHS_CEILING`` inside `train_active` so that a
   caller still passing the pre-fix value (20) is not silently
   undertrained again; see `train_active`'s own docstring.
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
# Ceiling initialization (section 2.3; judgment call 10)
# ---------------------------------------------------------------------------


def ceiling_init(b_hat: np.ndarray) -> float:
    """"Ceiling M: ... initialized at the 95th percentile of calibrated b_j
    plus 2 (an arbitrary anchoring constant ...)" (section 2.3 / judgment
    call 10)."""
    b_hat = np.asarray(b_hat, dtype=float)
    return float(np.percentile(b_hat, 95) + 2.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


#: Floor on `train_active`'s epoch ceiling (module docstring note 8;
#: `_planning/research/act_p0_diagnosis.md` section 3.2/5). The old bare
#: 20-epoch budget stopped training one to two orders of magnitude before
#: `g_c` converged; this constant guarantees every caller gets a ceiling
#: generous enough for convergence to be reachable, even one still passing
#: a pre-fix `n_epochs` value. Convergence (see `train_active`) typically
#: stops training long before this ceiling is hit on well-behaved fits --
#: it exists to bound the worst case, not as a target epoch count.
ACT_MIN_EPOCHS_CEILING = 3000


@dataclass
class ActiveConfig:
    variant: str = "act_p0"  # "act_p0" (lambda_i pinned at 1) | "act_p1" (amortized lambda_i)
    emb_dim: int = 8
    hidden_dim: int = 32
    lr: float = 1e-2
    n_epochs: int = 30  # CEILING (floored at ACT_MIN_EPOCHS_CEILING), not a fixed count -- see train_active
    window_epochs: int = 50  # block size for the windowed-mean loss statistic (note 8)
    patience_epochs: int = 25  # consecutive epochs the joint criterion must hold
    rel_tol: float = 1e-5  # windowed-mean relative loss change leg (~2x the measured noise floor)
    drift_tol: float = 5e-2  # macro guard on per-epoch [g_c, v_c, M] drift, above the cycle floor
    seed: int = 0
    device: str = "cpu"
    m_fixed: bool = False  # CG1 fallback (section 2.3): fix M at its init value


class ActiveModel(nn.Module):
    """The ACT transition plus its amortized recognition heads. No free
    per-learner parameter exists anywhere in this module (Gate B): only
    the shared population parameters ``g_c``, ``v_c``, ``M`` and the
    recognition network's (learner-generic) weights are learnable."""

    def __init__(
        self,
        num_items: int,
        n_kcs: int,
        cfg: ActiveConfig,
        m_init: float,
        encoder: Optional[BaseSeqEncoder] = None,
    ):
        super().__init__()
        if cfg.variant not in ("act_p0", "act_p1"):
            raise ValueError(f"unknown ACT variant {cfg.variant!r}, expected 'act_p0' or 'act_p1'")
        self.variant = cfg.variant
        self.n_kcs = n_kcs
        predict_lambda = cfg.variant == "act_p1"
        rec_cfg = RecognitionConfig(
            emb_dim=cfg.emb_dim, hidden_dim=cfg.hidden_dim, predict_lambda=predict_lambda
        )
        self.recognition = RecognitionNetwork(num_items, rec_cfg, encoder=encoder)
        self.g_raw = nn.Parameter(torch.zeros(n_kcs))  # g_c = softplus(g_raw), note 2
        self.v_c = nn.Parameter(torch.zeros(n_kcs))
        if cfg.m_fixed:
            self.register_buffer("_M", torch.tensor(float(m_init)))
        else:
            self._M = nn.Parameter(torch.tensor(float(m_init)))

    @property
    def M(self) -> torch.Tensor:
        return self._M

    @property
    def g_c(self) -> torch.Tensor:
        return F.softplus(self.g_raw)

    def forward(
        self, batch: LearnerBatch, seq_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns ``(logits (B,T), z_final (B,C), u_i (B,), lambda_i (B,)
        or None)``. Splits cleanly into the two concerns the design itself
        separates (section 2.3's own words: "order sensitivity can enter
        ONLY through the amortized recognition inputs"): `recognition`
        legitimately reads ``batch.responses`` to amortize ``(u_i,
        lambda_i)`` (that is its entire job), then `run_transition` runs
        the response-blind recurrence given only that seed and the item
        tag structure -- never touching ``batch.responses`` again."""
        u_i, lam = self.recognition(batch.item_ids, batch.responses, seq_lens)
        lam_eff = lam if lam is not None else torch.ones_like(u_i)
        z0 = u_i.unsqueeze(-1) + self.v_c.unsqueeze(0).expand(u_i.shape[0], -1)  # (B, C)
        logits, z_final = run_transition(z0, lam_eff, self.g_c, self.M, batch.kc_ids, batch.kc_mask, batch.b_j)
        return logits, z_final, u_i, lam


def run_transition(
    z0: torch.Tensor,
    lam: torch.Tensor,
    g_c: torch.Tensor,
    M: torch.Tensor,
    kc_ids: torch.Tensor,
    kc_mask: torch.Tensor,
    b_j: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The response-blind transition recurrence, standalone (module
    docstring note 1 for the multi-tag readout rule; note 3 for ``z0``).
    ``z0`` (B,C), ``lam`` (B,), ``g_c`` (C,), ``M`` scalar, ``kc_ids``/
    ``kc_mask`` (B,T,A), ``b_j`` (B,T). Never reads any response tensor --
    the caller (`ActiveModel.forward`) supplies only ``z0``/``lam``,
    already amortized upstream by `recognition.RecognitionNetwork`, which
    IS allowed to read responses. Returns ``(logits (B,T), z_final
    (B,C))``; ``logits[:, t]`` depends on history strictly before ``t``
    only (the pre-update ``Z`` is read, then updated)."""
    B, T_max, _ = kc_ids.shape
    Z = z0
    logits = torch.zeros(B, T_max, device=Z.device, dtype=Z.dtype)
    for t in range(T_max):
        tag_ids_t = kc_ids[:, t, :].clamp(min=0)  # (B, A)
        tag_mask_t = kc_mask[:, t, :].to(Z.dtype)  # (B, A)
        z_tagged = torch.gather(Z, 1, tag_ids_t)  # (B, A), PRE-update, causal
        denom = tag_mask_t.sum(dim=-1).clamp(min=1.0)
        mean_z = (z_tagged * tag_mask_t).sum(dim=-1) / denom
        logits[:, t] = mean_z - b_j[:, t]

        g_gathered = g_c[tag_ids_t]  # (B, A)
        gain = lam.unsqueeze(-1) * g_gathered * F.relu(M - z_tagged)
        gain = gain * tag_mask_t  # untagged slots contribute zero gain
        Z = Z.scatter_add(1, tag_ids_t, gain)
    return logits, Z


def seq_lens_from_mask(seq_mask: torch.Tensor) -> torch.Tensor:
    return seq_mask.sum(dim=1).long()


# ---------------------------------------------------------------------------
# Loss, training, forecast NLL
# ---------------------------------------------------------------------------


def active_loss(model: ActiveModel, batch: LearnerBatch, seq_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bernoulli forecast NLL (section 2.3: "frozen 1PL bank, Bernoulli
    forecast NLL"), masked to real (non-padding) positions."""
    if seq_lens is None:
        seq_lens = seq_lens_from_mask(batch.seq_mask)
    logits, _, _, _ = model(batch, seq_lens)
    y = batch.responses.to(logits.dtype)
    per_pos = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = batch.seq_mask.to(per_pos.dtype)
    return (per_pos * mask).sum() / mask.sum().clamp(min=1.0)


def _growth_param_snapshot(model: ActiveModel) -> np.ndarray:
    """The drift check's tracked quantity: the population growth
    parameters (``g_c``, ``v_c``, ``M``) concatenated into one flat
    vector, no-grad. These are exactly the quantities `implied_z_
    trajectory`/`implied_score_rise` read (module docstring note 4), so
    this mirrors `bank.calibrate_bank`'s choice to track its OWN frozen
    read-quantity (``b_hat``) for drift rather than raw nuisance
    parameters (module docstring note 8; `bank.py` note 3)."""
    with torch.no_grad():
        g_c = model.g_c.detach().cpu().numpy().reshape(-1)
        v_c = model.v_c.detach().cpu().numpy().reshape(-1)
        m = model.M.detach().cpu().numpy().reshape(-1)
    return np.concatenate([g_c, v_c, m]).astype(float)


def train_active(model: ActiveModel, batch: LearnerBatch, cfg: ActiveConfig) -> list[float]:
    """Stage 2 of the design's two-stage trainer (stage 1, bank
    calibration and freezing, is `bank.calibrate_bank`/`freeze_bank`, a
    prior stage's module): fit the dynamics parameters (``g_c``, ``v_c``,
    ``M`` unless fixed) and the recognition network jointly against the
    already-frozen bank's difficulties (carried in ``batch.b_j``).

    Convergence-gated (module docstring note 8), not a fixed epoch count.
    Dual criterion, `bank.calibrate_bank`'s structure adapted to ACT's
    single full-batch step per epoch: training stops once, for
    ``cfg.patience_epochs`` consecutive epochs, BOTH (i) the relative
    change between the mean loss of the last ``cfg.window_epochs`` epochs
    and the mean of the ``cfg.window_epochs`` before that is
    ``< cfg.rel_tol`` (the windowed mean cancels the Adam limit-cycle
    oscillation that a per-epoch change would measure, while a genuine
    descent trend survives the averaging -- note 8), and (ii) the
    population growth parameters (``g_c``, ``v_c``, ``M``, module
    function `_growth_param_snapshot`) moved ``< cfg.drift_tol`` (max
    absolute per-epoch change; a macro-movement guard, note 8). The
    earliest possible stop is therefore ``2*window_epochs +
    patience_epochs`` epochs.

    ``cfg.n_epochs`` is the epoch CEILING (kept under its old name for a
    backward-compatible signature), floored at ``ACT_MIN_EPOCHS_CEILING``:
    ``max(cfg.n_epochs, ACT_MIN_EPOCHS_CEILING)`` is the ceiling actually
    used, so a caller still passing the pre-fix value (20, e.g. `run.py`'s
    `RunConfig.act_epochs` default before this fix) is not silently
    undertrained the way ACT-P0 was
    (`_planning/research/act_p0_diagnosis.md`) -- raise ``cfg.n_epochs``
    to push the ceiling higher still, but there is no way to force a
    ceiling BELOW the floor, since the floor exists precisely to make
    silent undertraining unreachable through this argument. Fits stop at
    the convergence gate well before the ceiling in the measured cases
    (~400-1950 epochs across the note-8 stationarity study and a
    reduced-scale run of the real synth/bank pipeline, both twins, both
    variants); the ceiling bounds the worst case only."""
    seq_lens = seq_lens_from_mask(batch.seq_mask)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    max_epochs = max(int(cfg.n_epochs), ACT_MIN_EPOCHS_CEILING)
    W = int(cfg.window_epochs)

    trace: list[float] = []
    prev_growth = _growth_param_snapshot(model)
    good_epochs = 0
    for _ in range(max_epochs):
        optimizer.zero_grad()
        loss = active_loss(model, batch, seq_lens)
        loss.backward()
        optimizer.step()
        trace.append(float(loss.item()))

        cur_growth = _growth_param_snapshot(model)
        drift = float(np.max(np.abs(cur_growth - prev_growth))) if prev_growth.size else 0.0
        prev_growth = cur_growth

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
def forecast_nll(model: ActiveModel, batch: LearnerBatch) -> float:
    """Causal next-step forecast NLL on a held-out batch (the neural
    80/20-learner-split regime, section 2.2 / R1-I11)."""
    return float(active_loss(model, batch).item())


# ---------------------------------------------------------------------------
# CG1 / E-A1 / RB-A: closed-form implied trajectory and score-scale rise
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def implied_z_trajectory(
    z0: np.ndarray, lam: np.ndarray, g: np.ndarray, M: float, n_opportunities: int
) -> np.ndarray:
    """Closed-form iteration of the practice-gated, ceiling-gated
    transition treated as a function of a slice's OWN opportunity index
    (module docstring note 4): no data replay, since the design states
    this trajectory is interleaving-invariant by construction. ``z0``,
    ``lam``, ``g`` are ``(S,)`` (one slice per entry; ``lam``/``g`` may be
    scalars, broadcast). Returns ``(n_opportunities, S)``; row ``n`` (0-
    based) is the state used to predict opportunity ``n + 1`` (pre-update,
    causal)."""
    z0 = np.asarray(z0, dtype=float)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), z0.shape)
    g = np.broadcast_to(np.asarray(g, dtype=float), z0.shape)
    S = z0.shape[0]
    traj = np.zeros((max(n_opportunities, 0), S))
    z = z0.copy()
    for n in range(n_opportunities):
        traj[n] = z
        z = z + lam * g * np.maximum(M - z, 0.0)
    return traj


def implied_score_rise(
    z0: np.ndarray,
    lam: np.ndarray,
    g: np.ndarray,
    M: float,
    b_ref: np.ndarray,
    n_from: int = 1,
    n_to: int = 10,
) -> np.ndarray:
    """E-A1 / CG1's score-scale (proportion-of-max, since the response is
    binary) rise: ``sigmoid(z_{n_to} - b_ref) - sigmoid(z_{n_from} -
    b_ref)``, via the closed-form trajectory above. ``b_ref`` is the
    reference item difficulty (design: "on a median-difficulty item");
    scalar or ``(S,)``."""
    n_steps = max(int(n_from), int(n_to))
    traj = implied_z_trajectory(z0, lam, g, M, n_steps)
    z_from = traj[n_from - 1]
    z_to = traj[n_to - 1]
    b_ref = np.asarray(b_ref, dtype=float)
    return _sigmoid(z_to - b_ref) - _sigmoid(z_from - b_ref)


def implied_z_rise(z0: np.ndarray, lam: np.ndarray, g: np.ndarray, M: float, n_from: int = 1, n_to: int = 10) -> np.ndarray:
    """E-A1's state-units companion to `implied_score_rise` (design:
    "reported both in state units and as the model-implied ... rise ...
    (score scale)")."""
    n_steps = max(int(n_from), int(n_to))
    traj = implied_z_trajectory(z0, lam, g, M, n_steps)
    return traj[n_to - 1] - traj[n_from - 1]


# ---------------------------------------------------------------------------
# RB-A: the real-bed active-firing definition (section 5.2)
# ---------------------------------------------------------------------------


@dataclass
class FiringVerdict:
    bed_fires: bool
    kc_fires: np.ndarray  # (n_kcs,) bool
    bed_rise_mean_seed: float
    kc_rise_mean_seed: np.ndarray  # (n_kcs,)


def rb_a_firing(
    bed_rise_by_seed: np.ndarray,
    kc_rise_by_seed: np.ndarray,
    null_bed_rise: float,
    bar: float = 0.05,
    null_multiple: float = 5.0,
) -> FiringVerdict:
    """RB-A (section 5.2): ACT "fires" at bed level iff ALL of (i) the
    seed-mean population implied score rise over opportunities 1-10 is
    ``>= bar`` (default 0.05) proportion-of-max; (ii) that rise is
    ``>= null_multiple`` (default 5) times the same model's RB1-twin
    (in-situ bootstrap null) read; (iii) the rise is sign-consistent
    across every model seed. ACT fires on KC ``c`` iff the bed fired AND
    KC ``c``'s seed-mean rise is ``>= bar`` with all-seed sign
    consistency. ``null_bed_rise`` is the caller-supplied RB1 reference
    (this module computes only the firing logic, not the RB1 twin fit
    itself -- that is the same closed-form statistic run on a no-growth
    or bootstrap-twin fit, which is `battery.py`'s orchestration job, a
    later stage)."""
    bed_rise_by_seed = np.asarray(bed_rise_by_seed, dtype=float)
    kc_rise_by_seed = np.asarray(kc_rise_by_seed, dtype=float)

    bed_mean = float(bed_rise_by_seed.mean()) if bed_rise_by_seed.size else 0.0
    bed_sign_consistent = bool(
        bed_mean != 0.0 and np.all(np.sign(bed_rise_by_seed) == np.sign(bed_mean))
    )
    magnitude_ok = abs(bed_mean) >= bar
    null_ok = abs(bed_mean) >= null_multiple * abs(null_bed_rise)
    bed_fires = bool(magnitude_ok and null_ok and bed_sign_consistent)

    if kc_rise_by_seed.size == 0:
        kc_mean = np.zeros(0)
        kc_fires = np.zeros(0, dtype=bool)
    else:
        kc_mean = kc_rise_by_seed.mean(axis=0)
        n_kcs = kc_rise_by_seed.shape[1]
        kc_sign_consistent = np.zeros(n_kcs, dtype=bool)
        for c in range(n_kcs):
            m = kc_mean[c]
            kc_sign_consistent[c] = bool(m != 0.0 and np.all(np.sign(kc_rise_by_seed[:, c]) == np.sign(m)))
        kc_fires = bed_fires & (np.abs(kc_mean) >= bar) & kc_sign_consistent

    return FiringVerdict(
        bed_fires=bed_fires,
        kc_fires=kc_fires,
        bed_rise_mean_seed=bed_mean,
        kc_rise_mean_seed=kc_mean,
    )


# ---------------------------------------------------------------------------
# Saturation reporting rule (section 2.3)
# ---------------------------------------------------------------------------


def report_gain(g_c: np.ndarray, is_unsaturated: np.ndarray) -> np.ndarray:
    """"ACT latent-scale gains g_c are reported only on unsaturated KCs;
    on saturation-flagged KCs only score-scale implied changes are
    reported" (section 2.3): withholds (NaN) ``g_c`` on any KC the raw
    saturation flag (`slices.saturation_stats`) marks saturated."""
    g_c = np.asarray(g_c, dtype=float).copy()
    is_unsaturated = np.asarray(is_unsaturated, dtype=bool)
    g_c[~is_unsaturated] = np.nan
    return g_c


__all__ = [
    "ceiling_init",
    "ACT_MIN_EPOCHS_CEILING",
    "ActiveConfig",
    "ActiveModel",
    "run_transition",
    "seq_lens_from_mask",
    "active_loss",
    "train_active",
    "forecast_nll",
    "implied_z_trajectory",
    "implied_score_rise",
    "implied_z_rise",
    "FiringVerdict",
    "rb_a_firing",
    "report_gain",
]
