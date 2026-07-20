"""The A4 certification battery: all ten null/audit arms of
`_planning/design/a4_design.md` (v1.1, section 4), the CG1-CG10 synthetic
gates (section 5.1), the RB1-RB5 + RB-E1 real-bed licensing checks
(section 5.2, 5.3; RB0 wraps `bank.py`'s battery-arm-10 checks, RB-A is
`active.rb_a_firing`), and the K1-K7 kill-condition logic (section 5.6).
This module doubles as the A6 certification core (section 4's own
framing: "one module ... applied uniformly to synthetic and real beds").

Scope discipline (per the harness instructions): this module IMPLEMENTS
the machinery -- every arm's statistic, every gate's exact pre-registered
threshold, every kill condition's logic -- at whatever scale and replicate
count the caller supplies. It does not itself RUN the full pre-registered
replicate counts (B=199/999, 5 generator seeds, 3 model seeds) against
the true KDD/EdNet-scale beds; that is section 8's R2-R8 run sequence, a
later, compute-heavy campaign this build stage does not execute. Unit
tests here exercise every arm's mechanism and every gate's threshold
boundary at small scale, in seconds, on CPU.

Interpretation notes (recorded per the harness instructions):

1. **Arm 1's B=199 (per-KC) and B=999 (bed-level) replicate counts share
   one permutation run.** `run_permutation_battery` calls
   `gate.permutation_null` ONCE at ``max(n_replicates_bed,
   n_replicates_kc)`` replicates and reads the per-KC statistic off the
   first ``n_replicates_kc`` of those (a strict prefix of an i.i.d.
   replicate stream is itself a valid i.i.d. sample of that size), rather
   than re-running the whole permutation loop twice. This halves wall
   time at real-bed scale without changing either statistic's
   distribution; recorded as a documented implementation shortcut, not a
   threshold change.
2. **The M0-parametric-bootstrap twin's multi-tag redraw authority.**
   Section 4 arm 2 specifies the redraw per SLICE
   (``theta_hat_ic``), but a multi-tag ROW belongs to several slices at
   once and can carry only one realized response. Mirroring `synth.py`'s
   own precedent for exactly this problem ("the one Bernoulli draw for a
   multi-tag interaction is generated from the item's PRIMARY KC's
   trajectory only", that module's note 5), `m0_bootstrap_learners` reads
   each row's new response from its FIRST tagged KC's slot
   (``kc_ids[:, 0]``, the same primary-tag convention the rest of this
   package already uses) rather than inventing new multi-tag machinery.
3. **`m0_bootstrap_learners`'s row-reconstruction loop is O(rows) Python**,
   acceptable for this build stage's small-scale tests; a real-bed-scale
   run (millions of rows) is a later stage's vectorization concern, not
   this module's (the design's own section 7 costs bank calibration and
   the permutation battery separately from ordinary vectorized slice
   machinery for the same reason).
4. **Order-invariance's "KC group" for the cross-KC-interleaving
   reshuffle (arm 5) is each row's primary tag** (``tag_ids[:, 0]``),
   consistent with note 2 above; the reshuffle itself is a "riffle
   merge" of independently-random per-group sort keys that are
   monotonically increasing WITHIN a group (built from each group's
   already-time-ordered row indices), so a single global argsort
   randomizes cross-group interleaving while providing an exact,
   verifiable guarantee that within-group (within-KC) relative order is
   preserved.
5. **"Median per-KC/per-learner trajectory correlation" (arm 5, CG9,
   CG9-ACT) is read as: compute one correlation number per reshuffle
   (population-level, over learners or over concatenated per-KC
   positions), then take the median OVER RESHUFFLES; CG9's "per-KC"
   median additionally takes the median, across KCs, of each KC's own
   per-reshuffle median.** The design states "median ... across 5
   reshuffles" for CG9-ACT verbatim; CG9's parallel "median per-KC
   trajectory correlation" is read consistently. CG9-ACT's third clause
   (population per-KC implied-rise profile correlation >= 0.95) carries
   no "median" qualifier in the design text, so it is read as a MINIMUM
   over reshuffles (a stricter, not a laxer, reading -- permitted since
   thresholds may only tighten, never loosen).
6. **Arm 9's "seed-pooled analysis"** is implemented by pooling the
   observed statistic (its mean across seeds) against the CONCATENATION
   of every seed's own null replicates, reusing `gate.empirical_pvalue`'s
   existing convention (`seed_pooled_pvalue`), rather than introducing a
   new p-value-combination formula (e.g. Fisher's method): every other
   empirical p-value in this design already uses this exact convention,
   so extending it to the seed axis is the minimal, most consistent
   choice given the design does not specify a combination formula.
7. **K1-K7 are implemented as pure functions of FINAL gate verdicts**,
   not as stateful trackers of the two-revision-round budget (section
   5.6): "the budget" is a LEDGER-level bookkeeping concept (which
   fallback or revision round has been spent), external to this module;
   callers pass a ``revision_budget_exhausted`` flag (or simply call
   these once results are already final) rather than this module
   tracking round counts itself. This mirrors how `active.py`'s RB-A
   already separates "compute the statistic" from "the LEDGER decides
   when to interpret it."
8. **CG1c / CG6's "verdict is 'insufficient dynamic range'" clause** is a
   REPORT-TEXT condition (whether the pipeline's own generated verdict
   string matches the expected saturation-driven wording); this module's
   gate functions take it as a caller-supplied boolean
   (``verdict_is_insufficient_range``) rather than generating or parsing
   report text itself -- verdict-text assembly is `report.py`'s job
   (mirroring `active.py`'s own note 7 on the saturation reporting rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from kt_mirt.growth import active as active_mod
from kt_mirt.growth import bank as bank_mod
from kt_mirt.growth import gate as gate_mod
from kt_mirt.growth import rate as rate_mod
from kt_mirt.growth import synth as synth_mod
from kt_mirt.growth import tracker as tracker_mod
from kt_mirt.growth.bank import FrozenBank
from kt_mirt.growth.slices import Slice, build_slices
from kt_mirt.growth.synth import LearnerLog

B_KC_DEFAULT = 199
B_BED_DEFAULT = 999


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass
class GateVerdict:
    """Uniform machine-readable pass/fail record for every CG/RB/K check
    in this module; `report.py` assembles these into the final JSON."""

    name: str
    passed: bool
    detail: dict = field(default_factory=dict)


def is_sign_consistent(values) -> bool:
    """True iff every entry shares the sign of the mean (arm 9's
    sign-consistency reporting rule, reused by `active.rb_a_firing` and
    every seed-clustered gate here)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return False
    m = float(values.mean())
    return bool(m != 0.0 and np.all(np.sign(values) == np.sign(m)))


# =============================================================================
# Arm 1: permutation null
# =============================================================================


@dataclass
class PermutationBatteryResult:
    observed: gate_mod.GateResult
    bed_pvalue: float
    kc_pvalue: np.ndarray
    bh_reject: np.ndarray
    by_reject: np.ndarray
    bh_dependence_sensitive: np.ndarray  # BH-rejected but NOT BY-rejected
    n_replicates_bed: int
    n_replicates_kc: int


def run_permutation_battery(
    learners: Sequence[LearnerLog],
    n_learners: int,
    n_kcs: int,
    bank: Optional[FrozenBank],
    n_replicates_bed: int = B_BED_DEFAULT,
    n_replicates_kc: int = B_KC_DEFAULT,
    seed: int = 0,
    q: float = 0.05,
    device: str = "cpu",
    use_batched: bool = True,
    replicate_chunk_size: Optional[int] = None,
    safety_factor: float = 2.0,
    small_kc_threshold: int = 300,
) -> PermutationBatteryResult:
    """Arm 1: the primary reference null for PAS-C/PAS-G (and MIX-L's
    shared existence gate). See module docstring note 1 for the shared-
    run implementation of the two pre-registered replicate counts.

    ``use_batched`` (default True) selects `gate.permutation_null`'s
    replicate-batched Newton path (the A4 perf-surgery fix -- see
    `gate.permutation_null_batched`'s docstring); ``use_batched=False``
    reaches the original per-replicate loop (`gate.permutation_null_looped`),
    kept as the equivalence-gate reference and as an explicit fallback.
    ``replicate_chunk_size`` (default None) leaves M0's chunk sizing to
    `gate.permutation_null_batched`'s EMPIRICAL calibration (a real probe
    against real bed data and real free device memory, replacing an
    earlier analytic estimator that caused a production OOM by ignoring
    the KC-joint fit's dominant memory term); ``safety_factor`` (default
    2x) is that calibration's headroom multiplier. ``small_kc_threshold``
    (default 300) is the per-KC chunk-sizing cutoff (see
    `gate.permutation_null_batched`'s incident-context docstring, fix (4)):
    a real bed's KC sizes can be severely skewed, so chunk size is
    calibrated PER KC above this threshold, not once for the whole bed.
    """
    rows = bank_mod.build_calibration_rows(learners)
    slices = build_slices(rows)
    observed = gate_mod.compute_gate_result(slices, bank, n_kcs, device=device)

    n_rep = max(n_replicates_bed, n_replicates_kc)
    null = gate_mod.permutation_null(
        learners, n_learners, n_kcs, bank, n_rep, seed, device=device,
        use_batched=use_batched, replicate_chunk_size=replicate_chunk_size, safety_factor=safety_factor,
        small_kc_threshold=small_kc_threshold,
    )
    bed_null = null["bed"][:n_replicates_bed]
    kc_null = null["kc"][:n_replicates_kc]

    bed_p = gate_mod.empirical_pvalue(observed.bed_stat, bed_null)
    kc_p = np.array(
        [gate_mod.empirical_pvalue(observed.kc_stat[c], kc_null[:, c]) for c in range(n_kcs)]
    )
    bh = gate_mod.bh_fdr(kc_p, q=q)
    by = gate_mod.by_correction(kc_p, q=q)
    dependence_sensitive = bh & (~by)
    return PermutationBatteryResult(
        observed=observed,
        bed_pvalue=bed_p,
        kc_pvalue=kc_p,
        bh_reject=bh,
        by_reject=by,
        bh_dependence_sensitive=dependence_sensitive,
        n_replicates_bed=n_replicates_bed,
        n_replicates_kc=n_replicates_kc,
    )


# =============================================================================
# Arm 2: static twins (synthetic SYN-NG rerun; real-bed M0-parametric-bootstrap)
# =============================================================================


def static_twin_gate(
    twin_data: synth_mod.SyntheticTwin, bank: Optional[FrozenBank], device: str = "cpu"
) -> gate_mod.GateResult:
    """Synthetic leg of arm 2: rerun the identical gate pipeline on a
    twin's own learners (e.g. SYN-NG for CG2)."""
    rows = bank_mod.build_calibration_rows(twin_data.learners)
    slices = build_slices(rows)
    return gate_mod.compute_gate_result(slices, bank, twin_data.n_kcs, device=device)


def fit_full_slice_m0(slices: Sequence[Slice], bank: Optional[FrozenBank], device: str = "cpu") -> gate_mod.SliceFit:
    """Penalized constant-ability (M0) fit on a slice's FULL response
    history (no odd/even split -- this is generating a null replicate's
    ground truth, not evaluating a held-out statistic)."""
    return gate_mod.fit_batched(slices, bank, gate_mod._m0_design, P=1, time_filter=None, device=device)


def m0_bootstrap_slices(
    slices: dict, bank: Optional[FrozenBank], rng: np.random.Generator
) -> dict:
    """Real-bed leg of arm 2: redraw every slice's responses from its own
    penalized M0 fit against the frozen bank (design section 4 arm 2),
    preserving item sequence and opportunity structure; positions on
    items excluded from the likelihood (uncalibrated) are redrawn from
    the slice's own raw mean response rate instead of ``theta_hat - b_j``."""
    all_slices = list(slices.values())
    if not all_slices:
        return {}
    fit = fit_full_slice_m0(all_slices, bank, device="cpu")
    theta_hat = {k: float(fit.params[i, 0]) for i, k in enumerate(fit.keys)}

    new_slices: dict = {}
    for key, sl in slices.items():
        th = theta_hat[key]
        b = bank.difficulty(sl.item_id) if bank is not None else np.zeros(sl.T)
        p_theta = _sigmoid(th - b)
        valid = bank.is_calibrated(sl.item_id) if bank is not None else np.ones(sl.T, dtype=bool)
        slice_mean = float(sl.response.mean()) if sl.T else 0.5
        p_use = np.where(valid, p_theta, slice_mean)
        new_resp = (rng.random(sl.T) < p_use).astype(np.int8)
        new_slices[key] = Slice(
            learner=sl.learner, kc=sl.kc, item_id=sl.item_id.copy(), response=new_resp,
            opportunity=sl.opportunity.copy(), block_id=sl.block_id.copy(),
        )
    return new_slices


def m0_bootstrap_learners(
    learners: Sequence[LearnerLog], bank: Optional[FrozenBank], rng: np.random.Generator
) -> tuple[list[LearnerLog], dict]:
    """Reconstructs full per-learner logs (needed by trackers/ACT, design:
    "Trackers refit on it at 1 seed") with each row's response replaced by
    its primary-tagged slice's bootstrap redraw (module docstring note
    2). Returns ``(new_learners, new_slices)``."""
    rows = bank_mod.build_calibration_rows(learners)
    slices = build_slices(rows)
    new_slices = m0_bootstrap_slices(slices, bank, rng)

    R = len(rows)
    A = rows.kc_ids.shape[1] if rows.kc_ids.ndim == 2 else 0
    new_row_response = rows.response.copy()
    if R and A:
        has_primary = rows.kc_mask[:, 0]
        for r in np.where(has_primary)[0]:
            key = (int(rows.learner_idx[r]), int(rows.kc_ids[r, 0]))
            opp0 = int(rows.opportunity[r, 0])
            sl = new_slices.get(key)
            if sl is not None and 0 < opp0 <= sl.T:
                new_row_response[r] = sl.response[opp0 - 1]

    out: list[LearnerLog] = []
    ptr = 0
    for log in learners:
        T = len(log.item_ids)
        out.append(
            LearnerLog(
                learner=log.learner,
                item_ids=np.asarray(log.item_ids).copy(),
                responses=new_row_response[ptr : ptr + T].copy(),
                tag_ids=np.asarray(log.tag_ids).copy(),
                tag_mask=np.asarray(log.tag_mask).copy(),
            )
        )
        ptr += T
    return out, new_slices


# =============================================================================
# Arm 3: frozen/untrained-encoder null (CG7, RB2 bridge)
# =============================================================================


@dataclass
class Cg7Result:
    trained_rank_corr: float
    frozen_rank_corr: float
    margin: float
    passed: bool
    detail: dict = field(default_factory=dict)


def cg7_verdict(
    trained_profile: np.ndarray,
    frozen_profile: np.ndarray,
    true_rise: np.ndarray,
    detection_bar: float = 0.6,
    margin_bar: float = 0.2,
    decert_bar: float = 0.1,
) -> Cg7Result:
    """CG7 (design 5.1): trained KC-growth-profile rank recovery >=
    ``detection_bar``; trained-minus-frozen margin >= ``margin_bar``;
    frozen must not itself clear the detection bar; decertify if the
    margin is < ``decert_bar``."""
    trained_corr = bank_mod.spearman_rank_correlation(trained_profile, true_rise)
    frozen_corr = bank_mod.spearman_rank_correlation(frozen_profile, true_rise)
    margin = (
        trained_corr - frozen_corr if np.isfinite(trained_corr) and np.isfinite(frozen_corr) else float("nan")
    )
    frozen_clears_detection = np.isfinite(frozen_corr) and frozen_corr >= detection_bar
    decertified = (not np.isfinite(margin)) or margin < decert_bar
    passed = bool(
        np.isfinite(trained_corr)
        and trained_corr >= detection_bar
        and np.isfinite(margin)
        and margin >= margin_bar
        and not frozen_clears_detection
        and not decertified
    )
    return Cg7Result(
        trained_rank_corr=trained_corr,
        frozen_rank_corr=frozen_corr,
        margin=margin,
        passed=passed,
        detail={"frozen_clears_detection": bool(frozen_clears_detection), "decertified": bool(decertified)},
    )


def compute_rb2_bar(
    synthetic_trained_vs_frozen_profile_corr: np.ndarray, cap: float = 0.9, floor: float = 0.7, margin: float = 0.1
) -> float:
    """The CG7 bridge clause (design 5.1): "RB2's real-bed bar is set to
    min(0.9, synthetic p95 + 0.1), floored at 0.7, and frozen before any
    real run.\""""
    corr = np.asarray(synthetic_trained_vs_frozen_profile_corr, dtype=float)
    if corr.size == 0:
        return floor
    p95 = float(np.percentile(corr, 95))
    return float(np.clip(min(cap, p95 + margin), floor, cap))


def rb2_verdict(trained_nll: np.ndarray, frozen_nll: np.ndarray, profile_corr: float, bar: float) -> GateVerdict:
    """RB2 (design 5.2): trained must beat frozen on held-out forecast
    NLL, AND the tracker's growth claim is only attributable to learned
    state if the trained-vs-frozen profile correlation is BELOW the
    (bridge-computed) bar."""
    trained_mean = float(np.mean(trained_nll)) if len(trained_nll) else float("nan")
    frozen_mean = float(np.mean(frozen_nll)) if len(frozen_nll) else float("nan")
    beats = np.isfinite(trained_mean) and np.isfinite(frozen_mean) and trained_mean < frozen_mean
    attributable = np.isfinite(profile_corr) and profile_corr < bar
    passed = bool(beats and attributable)
    return GateVerdict(
        "RB2", passed,
        {"trained_nll_mean": trained_mean, "frozen_nll_mean": frozen_mean, "profile_corr": profile_corr, "bar": bar},
    )


# =============================================================================
# Arm 4: single-KC-drill contamination probe (CG8)
# =============================================================================


def build_drill_log(
    learner_id: int, kc_id: int, item_id: int, n_repeats: int = 100, correct: bool = True, max_arity: int = 1
) -> LearnerLog:
    """A single-KC drill sequence (design arm 4): one KC practiced
    ``n_repeats`` times, all-correct (oracle) or all-incorrect
    (anti-oracle), no other KC touched."""
    item_ids = np.full(n_repeats, item_id, dtype=np.int64)
    responses = np.full(n_repeats, 1 if correct else 0, dtype=np.int8)
    tag_ids = np.full((n_repeats, max_arity), -1, dtype=np.int64)
    tag_ids[:, 0] = kc_id
    tag_mask = np.zeros((n_repeats, max_arity), dtype=bool)
    tag_mask[:, 0] = True
    return LearnerLog(learner=learner_id, item_ids=item_ids, responses=responses, tag_ids=tag_ids, tag_mask=tag_mask)


def pas_n1_full_ability_trajectory(
    model: tracker_mod.PASN1Model, item_ids: torch.Tensor, responses: torch.Tensor
) -> np.ndarray:
    """The FULL (T, C) per-KC ability trajectory (pre-gather), needed to
    read off-target KC movement during a single-KC drill -- `PASN1Model.
    forward`'s own output collapses to only the current position's
    tagged KC(s), which cannot expose off-target channels."""
    with torch.no_grad():
        state = model.encoder.state_for_prediction(item_ids, responses)
        ability = model.ability_head(state)
    return ability[0].cpu().numpy()


@dataclass
class ContaminationResult:
    ratio: float
    on_target_displacement: dict
    off_target_max_abs: dict
    passed: bool


def contamination_probe(
    model: tracker_mod.PASN1Model,
    n_kcs: int,
    drilled_kcs: Sequence[int],
    item_for_kc: dict,
    n_repeats: int = 100,
    bar: float = 0.10,
    device: str = "cpu",
) -> ContaminationResult:
    """Arm 4 / CG8: contamination ratio = max over off-target KCs of
    |displacement| / on-target |displacement|, averaged over the drilled
    KCs and both directions (oracle/anti-oracle). Gate: ratio <= 0.10."""
    ratios: list[float] = []
    on_disp: dict = {}
    off_max: dict = {}
    for kc in drilled_kcs:
        for correct in (True, False):
            log = build_drill_log(0, kc, item_for_kc[kc], n_repeats=n_repeats, correct=correct)
            item_ids = torch.as_tensor(log.item_ids, dtype=torch.long, device=device).unsqueeze(0)
            responses = torch.as_tensor(log.responses, dtype=torch.long, device=device).unsqueeze(0)
            ability = pas_n1_full_ability_trajectory(model, item_ids, responses)
            on_d = tracker_mod.displacement(ability[:, kc])
            offs = [abs(tracker_mod.displacement(ability[:, c])) for c in range(n_kcs) if c != kc]
            off_m = max(offs) if offs else 0.0
            ratio = (off_m / abs(on_d)) if abs(on_d) > 1e-8 else float("inf")
            ratios.append(ratio)
            on_disp[(kc, correct)] = on_d
            off_max[(kc, correct)] = off_m
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    return ContaminationResult(
        ratio=mean_ratio, on_target_displacement=on_disp, off_target_max_abs=off_max,
        passed=bool(np.isfinite(mean_ratio) and mean_ratio <= bar),
    )


def contamination_diagnostic_band(off_target_values, syn_ng_displacement_p95: float) -> dict:
    """Arm 4's DEMOTED diagnostic (v1.1, R1-I3): off-target absolute
    movement compared against the length-matched SYN-NG p95 band --
    reported, never gated."""
    vals = np.asarray(list(off_target_values), dtype=float)
    frac_within = float(np.mean(np.abs(vals) <= syn_ng_displacement_p95)) if vals.size else float("nan")
    return {"frac_within_syn_ng_p95_band": frac_within, "syn_ng_p95": syn_ng_displacement_p95}


# =============================================================================
# Arm 5: order-invariance stress (CG9, CG9-ACT)
# =============================================================================


def permute_cross_kc_interleaving(learners: Sequence[LearnerLog], rng: np.random.Generator) -> list[LearnerLog]:
    """Hold each row's primary-tag group's internal order fixed; draw a
    fresh random cross-group interleaving (module docstring note 4): a
    riffle merge built from independently-sorted per-group uniform draws,
    assigned in original (ascending) intra-group row order so the GLOBAL
    argsort exactly reproduces every group's own relative order while
    randomizing which group each output position comes from."""
    out: list[LearnerLog] = []
    for log in learners:
        T = len(log.item_ids)
        if T == 0:
            out.append(log)
            continue
        primary = np.asarray(log.tag_ids)[:, 0]
        sort_key = np.empty(T)
        for c in np.unique(primary):
            idx = np.where(primary == c)[0]
            u = np.sort(rng.uniform(0.0, 1.0, size=len(idx)))
            sort_key[idx] = u
        order = np.argsort(sort_key, kind="stable")
        out.append(
            LearnerLog(
                learner=log.learner,
                item_ids=np.asarray(log.item_ids)[order].copy(),
                responses=np.asarray(log.responses)[order].copy(),
                tag_ids=np.asarray(log.tag_ids)[order].copy(),
                tag_mask=np.asarray(log.tag_mask)[order].copy(),
            )
        )
    return out


def full_ability_batch(
    model: tracker_mod.PASN1Model, learners: Sequence[LearnerLog], device: str = "cpu"
) -> list[np.ndarray]:
    """Per-learner (T, C) full per-KC PAS-N1 ability trajectories."""
    out = []
    for log in learners:
        item_ids = torch.as_tensor(log.item_ids, dtype=torch.long, device=device).unsqueeze(0)
        responses = torch.as_tensor(log.responses, dtype=torch.long, device=device).unsqueeze(0)
        out.append(pas_n1_full_ability_trajectory(model, item_ids, responses))
    return out


def collect_kc_trajectory(abilities: list[np.ndarray], learners: Sequence[LearnerLog], kc: int) -> np.ndarray:
    """Flatten, over learners (fixed order) and within-learner in row
    order, the KC-``kc`` ability reads at every position tagged with
    ``kc`` -- invariant in composition and intra-KC order to arm 5's
    cross-KC-only reshuffle, so directly comparable original-vs-reshuffle."""
    vals = []
    for ability, log in zip(abilities, learners):
        tagged = np.any((np.asarray(log.tag_ids) == kc) & np.asarray(log.tag_mask), axis=1)
        rows = np.where(tagged)[0]
        vals.append(ability[rows, kc])
    return np.concatenate(vals) if vals else np.zeros(0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(a) != len(b) or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class OrderInvarianceResult:
    kc_median_corr: float
    sign_flip_fraction: float
    passed: bool


def order_invariance_stress(
    model: tracker_mod.PASN1Model,
    learners: Sequence[LearnerLog],
    n_kcs: int,
    n_reshuffles: int = 5,
    seed: int = 0,
    device: str = "cpu",
    corr_bar: float = 0.8,
    flip_bar: float = 0.10,
) -> OrderInvarianceResult:
    """Arm 5 / CG9: trackers must be stable under cross-KC-only
    interleaving reshuffles (module docstring note 5's reading of
    "median")."""
    rng = np.random.default_rng(seed)
    original_abilities = full_ability_batch(model, learners, device=device)
    original_traj = {c: collect_kc_trajectory(original_abilities, learners, c) for c in range(n_kcs)}
    original_disp = {c: tracker_mod.displacement(original_traj[c]) for c in range(n_kcs)}

    corr_per_kc: dict = {c: [] for c in range(n_kcs)}
    flips = 0
    trials = 0
    for _ in range(n_reshuffles):
        reshuffled = permute_cross_kc_interleaving(learners, rng)
        reshuffled_abilities = full_ability_batch(model, reshuffled, device=device)
        for c in range(n_kcs):
            traj_r = collect_kc_trajectory(reshuffled_abilities, reshuffled, c)
            corr = _safe_corr(original_traj[c], traj_r)
            if np.isfinite(corr):
                corr_per_kc[c].append(corr)
            disp_r = tracker_mod.displacement(traj_r)
            d0 = original_disp[c]
            if np.isfinite(disp_r) and np.isfinite(d0) and d0 != 0 and disp_r != 0:
                trials += 1
                if np.sign(disp_r) != np.sign(d0):
                    flips += 1

    per_kc_medians = [float(np.median(v)) for v in corr_per_kc.values() if v]
    kc_median_corr = float(np.median(per_kc_medians)) if per_kc_medians else float("nan")
    flip_fraction = (flips / trials) if trials else float("nan")
    passed = bool(np.isfinite(kc_median_corr) and kc_median_corr >= corr_bar and np.isfinite(flip_fraction) and flip_fraction < flip_bar)
    return OrderInvarianceResult(kc_median_corr=kc_median_corr, sign_flip_fraction=flip_fraction, passed=passed)


@dataclass
class ActRecognitionStabilityResult:
    u_median_corr: float
    lambda_median_corr: Optional[float]
    rise_profile_min_corr: float
    passed: bool


def act_recognition_stability(
    model: active_mod.ActiveModel,
    learners: Sequence[LearnerLog],
    n_kcs: int,
    b_ref: float = 0.0,
    n_reshuffles: int = 5,
    seed: int = 0,
    device: str = "cpu",
    u_bar: float = 0.9,
    lambda_bar: float = 0.9,
    rise_bar: float = 0.95,
) -> ActRecognitionStabilityResult:
    """CG9-ACT: median per-learner corr(u_i) >= 0.9 (and lambda_i >= 0.9
    for ACT-P1) across 5 reshuffles; population per-KC implied-rise
    profile correlation >= 0.95 (read as a MINIMUM over reshuffles, module
    docstring note 5)."""
    rng = np.random.default_rng(seed)

    def _u_lambda_rise(logs: Sequence[LearnerLog]):
        item_ids_list = [torch.as_tensor(log.item_ids, dtype=torch.long) for log in logs]
        resp_list = [torch.as_tensor(log.responses, dtype=torch.long) for log in logs]
        T_max = max((len(t) for t in item_ids_list), default=0)
        B = len(logs)
        item_ids = torch.zeros(B, T_max, dtype=torch.long, device=device)
        responses = torch.zeros(B, T_max, dtype=torch.long, device=device)
        seq_lens = torch.zeros(B, dtype=torch.long, device=device)
        kc_ids = torch.zeros(B, T_max, 1, dtype=torch.long, device=device)
        kc_mask = torch.zeros(B, T_max, 1, dtype=torch.bool, device=device)
        for i, log in enumerate(logs):
            t = len(log.item_ids)
            item_ids[i, :t] = item_ids_list[i]
            responses[i, :t] = resp_list[i]
            seq_lens[i] = t
            kc_ids[i, :t, 0] = torch.as_tensor(np.asarray(log.tag_ids)[:, 0], dtype=torch.long)
            kc_mask[i, :t, 0] = torch.as_tensor(np.asarray(log.tag_mask)[:, 0], dtype=torch.bool)
        with torch.no_grad():
            u_i, lam = model.recognition(item_ids, responses, seq_lens)
        u_np = u_i.cpu().numpy()
        lam_np = lam.cpu().numpy() if lam is not None else None
        # Population per-KC implied rise, using this reshuffle's (u_i, lambda_i)
        # and the model's fixed g_c/v_c/M (design 2.3: the transition is
        # interleaving-invariant given the recognition outputs).
        g_c = model.g_c.detach().cpu().numpy()
        M = float(model.M.detach().cpu().item())
        v_c = model.v_c.detach().cpu().numpy()
        lam_eff = lam_np if lam_np is not None else np.ones_like(u_np)
        kc_rise = np.full(n_kcs, np.nan)
        for c in range(n_kcs):
            tagged_learners = [i for i, log in enumerate(logs) if (np.asarray(log.tag_ids)[:, 0] == c).any()]
            if not tagged_learners:
                continue
            z0 = u_np[tagged_learners] + v_c[c]
            rises = active_mod.implied_score_rise(z0, lam_eff[tagged_learners], g_c[c], M, b_ref)
            kc_rise[c] = float(np.mean(rises))
        return u_np, lam_np, kc_rise

    u0, lam0, rise0 = _u_lambda_rise(learners)
    u_corrs, lam_corrs, rise_corrs = [], [], []
    for _ in range(n_reshuffles):
        reshuffled = permute_cross_kc_interleaving(learners, rng)
        u_r, lam_r, rise_r = _u_lambda_rise(reshuffled)
        c = _safe_corr(u0, u_r)
        if np.isfinite(c):
            u_corrs.append(c)
        if lam0 is not None and lam_r is not None:
            cl = _safe_corr(lam0, lam_r)
            if np.isfinite(cl):
                lam_corrs.append(cl)
        finite = np.isfinite(rise0) & np.isfinite(rise_r)
        cr = _safe_corr(rise0[finite], rise_r[finite]) if finite.sum() >= 2 else float("nan")
        if np.isfinite(cr):
            rise_corrs.append(cr)

    u_median = float(np.median(u_corrs)) if u_corrs else float("nan")
    lam_median = float(np.median(lam_corrs)) if lam_corrs else (None if lam0 is None else float("nan"))
    rise_min = float(np.min(rise_corrs)) if rise_corrs else float("nan")

    u_ok = np.isfinite(u_median) and u_median >= u_bar
    lam_ok = True if lam_median is None else (np.isfinite(lam_median) and lam_median >= lambda_bar)
    rise_ok = np.isfinite(rise_min) and rise_min >= rise_bar
    passed = bool(u_ok and lam_ok and rise_ok)
    return ActRecognitionStabilityResult(
        u_median_corr=u_median, lambda_median_corr=lam_median, rise_profile_min_corr=rise_min, passed=passed
    )


# =============================================================================
# Arm 6: direction audit (CG10)
# =============================================================================


def direction_violation_fraction(theta_causal: np.ndarray, responses: np.ndarray, tol: float = 0.01) -> float:
    """CG10: a violation is d(theta) < -tol after y=1, or d(theta) > +tol
    after y=0, within one KC's own causally-aligned theta sequence."""
    theta_causal = np.asarray(theta_causal, dtype=float)
    responses = np.asarray(responses)
    T = len(theta_causal)
    if T < 2:
        return float("nan")
    dtheta = theta_causal[1:] - theta_causal[:-1]
    y = responses[:-1]
    violation = np.where(y == 1, dtheta < -tol, dtheta > tol)
    return float(violation.mean())


# =============================================================================
# Arm 7: split-half reliability
# =============================================================================


def spearman_brown(r_half: float, n_halves: int = 2) -> float:
    if not np.isfinite(r_half) or r_half <= -1.0:
        return float("nan")
    denom = 1.0 + (n_halves - 1.0) * r_half
    if denom == 0:
        return float("nan")
    return float(n_halves * r_half / denom)


def _rate_stat(response: np.ndarray) -> float:
    return float(response.mean()) if len(response) else float("nan")


def per_slice_statistic_odd_even(sl: Slice, statistic_fn: Callable[[np.ndarray], float]) -> tuple[float, float]:
    odd = gate_mod._time_filter_odd(sl.opportunity)
    even = gate_mod._time_filter_even(sl.opportunity)
    return statistic_fn(sl.response[odd]), statistic_fn(sl.response[even])


def per_learner_split_half(slices: Sequence[Slice], statistic_fn: Callable[[np.ndarray], float] = _rate_stat) -> float:
    """Odd/even split-half reliability within D2+ slices, Spearman-Brown
    corrected (arm 7's per-learner reliability)."""
    d2 = [sl for sl in slices if sl.is_d2_plus]
    odds, evens = [], []
    for sl in d2:
        o, e = per_slice_statistic_odd_even(sl, statistic_fn)
        if np.isfinite(o) and np.isfinite(e):
            odds.append(o)
            evens.append(e)
    if len(odds) < 2:
        return float("nan")
    r = bank_mod.spearman_rank_correlation(np.array(odds), np.array(evens))
    return spearman_brown(r)


def kc_level_split_half(kc_slices_by_kc: Sequence[Sequence[Slice]], bank: Optional[FrozenBank], seed: int = 0) -> float:
    """Split LEARNERS into two halves; fit r_c independently per half;
    correlate the resulting r_c estimates across KCs (arm 7's KC-level
    reliability)."""
    rng = np.random.default_rng(seed)
    r_a, r_b = [], []
    for kc_slices in kc_slices_by_kc:
        kc_slices = list(kc_slices)
        if len(kc_slices) < 4:
            continue
        idx = rng.permutation(len(kc_slices))
        half = len(idx) // 2
        fit_a = rate_mod.fit_kc_rate([kc_slices[i] for i in idx[:half]], bank)
        fit_b = rate_mod.fit_kc_rate([kc_slices[i] for i in idx[half:]], bank)
        if fit_a is not None and fit_b is not None and np.isfinite(fit_a.r_c) and np.isfinite(fit_b.r_c):
            r_a.append(fit_a.r_c)
            r_b.append(fit_b.r_c)
    if len(r_a) < 2:
        return float("nan")
    return bank_mod.spearman_rank_correlation(np.array(r_a), np.array(r_b))


def predicted_split_half_via_bootstrap(
    slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    theta_fn: Callable[[Slice], np.ndarray],
    statistic_fn: Callable[[np.ndarray], float] = _rate_stat,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> float:
    """The density-predicted split-half reliability (arm 7: "the
    qmirt no-fitting arithmetic, generalized by parametric bootstrap of
    the fitted curves"): redraw each D2+ slice's responses from its own
    ALREADY-FITTED theta(n) curve against the frozen bank, recompute the
    same reliability statistic on each bootstrap replicate, and average."""
    rng = np.random.default_rng(seed)
    d2 = [sl for sl in slices if sl.is_d2_plus]
    rels = []
    for _ in range(n_bootstrap):
        boot_slices = []
        for sl in d2:
            theta_n = np.asarray(theta_fn(sl), dtype=float)
            b = bank.difficulty(sl.item_id) if bank is not None else np.zeros(sl.T)
            p = _sigmoid(theta_n - b)
            resp = (rng.random(sl.T) < p).astype(np.int8)
            boot_slices.append(
                Slice(learner=sl.learner, kc=sl.kc, item_id=sl.item_id, response=resp,
                      opportunity=sl.opportunity, block_id=sl.block_id)
            )
        rels.append(per_learner_split_half(boot_slices, statistic_fn))
    rels = np.array([r for r in rels if np.isfinite(r)])
    return float(rels.mean()) if len(rels) else float("nan")


def split_half_gate(observed: float, predicted: float, tol: float = 0.10) -> GateVerdict:
    passed = np.isfinite(observed) and np.isfinite(predicted) and abs(observed - predicted) <= tol
    return GateVerdict("split_half_agreement", bool(passed), {"observed": observed, "predicted": predicted, "tol": tol})


# =============================================================================
# Arm 8: truncation stress
# =============================================================================

TRUNCATION_CUTOFFS: tuple[Optional[int], ...] = (5, 10, 20, None)  # None = full


def truncate_slice(sl: Slice, cutoff: Optional[int]) -> Slice:
    if cutoff is None:
        return sl
    keep = sl.opportunity <= cutoff
    return Slice(learner=sl.learner, kc=sl.kc, item_id=sl.item_id[keep], response=sl.response[keep],
                 opportunity=sl.opportunity[keep], block_id=sl.block_id[keep])


def truncation_rate_by_cutoff(
    kc_slices: Sequence[Slice], bank: Optional[FrozenBank], cutoffs=TRUNCATION_CUTOFFS, min_points: int = 3
) -> dict:
    out = {}
    for cutoff in cutoffs:
        truncated = [truncate_slice(sl, cutoff) for sl in kc_slices]
        truncated = [sl for sl in truncated if sl.T >= min_points]
        out[cutoff] = rate_mod.fit_kc_rate(truncated, bank) if truncated else None
    return out


@dataclass
class TruncationResult:
    rank_corr_10_full: float
    iqr_by_cutoff: dict
    consecutive_iqr_stable: bool
    cumulative_band_ok: bool
    passed: bool


def truncation_stress(
    per_kc_cutoff_fits: dict, cutoffs=TRUNCATION_CUTOFFS
) -> TruncationResult:
    """Arm 8: ``per_kc_cutoff_fits`` maps KC -> {cutoff: KCRateFit or
    None} (`truncation_rate_by_cutoff`'s output, collected over every
    claimed KC). Checks the rank-stability bar (cutoff 10 vs full),
    the consecutive-cutoff IQR-move bound, and the cumulative
    IQR(10)/IQR(full) band."""
    r_by_cutoff: dict = {c: [] for c in cutoffs}
    paired10, pairedfull = [], []
    for fits in per_kc_cutoff_fits.values():
        for c in cutoffs:
            fit = fits.get(c)
            if fit is not None and np.isfinite(fit.r_c):
                r_by_cutoff[c].append(fit.r_c)
        f10, ffull = fits.get(10), fits.get(None)
        if f10 is not None and ffull is not None and np.isfinite(f10.r_c) and np.isfinite(ffull.r_c):
            paired10.append(f10.r_c)
            pairedfull.append(ffull.r_c)

    rank_corr = (
        bank_mod.spearman_rank_correlation(np.array(paired10), np.array(pairedfull))
        if len(paired10) >= 2
        else float("nan")
    )

    iqr = {}
    for c in cutoffs:
        vals = np.array(r_by_cutoff[c])
        iqr[c] = float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) >= 4 else float("nan")

    order = [5, 10, 20, None]
    consecutive_ok = True
    for a, b in zip(order[:-1], order[1:]):
        ia, ib = iqr.get(a), iqr.get(b)
        if ia is None or ib is None or not np.isfinite(ia) or not np.isfinite(ib) or ia == 0:
            continue
        if abs(ib - ia) / abs(ia) >= 0.25:
            consecutive_ok = False

    iqr10, iqrfull = iqr.get(10), iqr.get(None)
    if iqr10 is not None and iqrfull is not None and np.isfinite(iqr10) and np.isfinite(iqrfull) and iqrfull != 0:
        cumulative_ok = 0.75 <= (iqr10 / iqrfull) <= 1.33
    else:
        cumulative_ok = False

    rank_ok = np.isfinite(rank_corr) and rank_corr >= 0.8
    passed = bool(rank_ok and consecutive_ok and cumulative_ok)
    return TruncationResult(
        rank_corr_10_full=rank_corr, iqr_by_cutoff=iqr,
        consecutive_iqr_stable=consecutive_ok, cumulative_band_ok=cumulative_ok, passed=passed,
    )


# =============================================================================
# Arm 9: seed clustering
# =============================================================================


def seed_pooled_pvalue(observed_by_seed: np.ndarray, null_by_seed: Sequence[np.ndarray]) -> float:
    """Arm 9's seed-pooled significance (module docstring note 6): pool
    the mean observed statistic against the concatenation of every seed's
    own null replicates, reusing `gate.empirical_pvalue`."""
    observed_by_seed = np.asarray(observed_by_seed, dtype=float)
    if observed_by_seed.size == 0:
        return float("nan")
    observed_mean = float(observed_by_seed.mean())
    pooled_null = np.concatenate([np.asarray(n) for n in null_by_seed]) if len(null_by_seed) else np.zeros(0)
    return gate_mod.empirical_pvalue(observed_mean, pooled_null)


@dataclass
class SeedClusterVerdict:
    sign_consistent: bool
    pooled_pvalue: Optional[float]
    passed: bool


def seed_cluster_verdict(values_by_seed, pooled_pvalue: Optional[float] = None, alpha: float = 0.05) -> SeedClusterVerdict:
    consistent = is_sign_consistent(values_by_seed)
    significant = True if pooled_pvalue is None else (np.isfinite(pooled_pvalue) and pooled_pvalue < alpha)
    return SeedClusterVerdict(sign_consistent=consistent, pooled_pvalue=pooled_pvalue, passed=bool(consistent and significant))


# =============================================================================
# Arm 10: bank robustness -- thin wrappers over `bank.py`'s own checks
# =============================================================================


def rb0_bank_robustness(pairwise_result: dict) -> GateVerdict:
    return GateVerdict("RB0", bool(pairwise_result["passed"]), dict(pairwise_result))


def bank_recovery_verdict(recovery_result: dict) -> GateVerdict:
    return GateVerdict("bank_recovery", bool(recovery_result["passed"]), dict(recovery_result))


# =============================================================================
# CG1-CG10: synthetic certification gates (design 5.1)
# =============================================================================

CG1_POP_MAX = 0.01
CG1_P95_MAX_KDD = 0.01
CG1_P95_MAX_EDNET = 0.02


def cg1_active_silence(pop_mean_rise: float, p95_rise: float, density: str) -> GateVerdict:
    p95_bar = CG1_P95_MAX_KDD if density == synth_mod.KDD_MATCHED.name else CG1_P95_MAX_EDNET
    passed = abs(pop_mean_rise) <= CG1_POP_MAX and abs(p95_rise) <= p95_bar
    return GateVerdict(
        "CG1", bool(passed),
        {"pop_mean_rise": pop_mean_rise, "p95_rise": p95_rise, "pop_bar": CG1_POP_MAX, "p95_bar": p95_bar, "density": density},
    )


def cg1a_active_positive_control(
    bed_fires: bool, pop_rise: float, true_pop_rise: float, kc_rank_corr: float, density: str
) -> GateVerdict:
    bar = 0.6 if density == synth_mod.KDD_MATCHED.name else 0.5
    ratio = (pop_rise / true_pop_rise) if true_pop_rise else float("nan")
    ratio_ok = np.isfinite(ratio) and 0.5 <= ratio <= 1.5
    corr_ok = np.isfinite(kc_rank_corr) and kc_rank_corr >= bar
    passed = bool(bed_fires and ratio_ok and corr_ok)
    return GateVerdict("CG1a", passed, {"bed_fires": bed_fires, "ratio": ratio, "kc_rank_corr": kc_rank_corr, "bar": bar})


def cg1b_active_misfit_robustness(
    silent_pop_mean: float, silent_p95: float, growing_rank_corr: float, overshoot_fraction: float
) -> GateVerdict:
    silent_ok = abs(silent_pop_mean) <= CG1_POP_MAX and abs(silent_p95) <= CG1_P95_MAX_KDD
    corr_ok = np.isfinite(growing_rank_corr) and growing_rank_corr >= 0.5
    overshoot_ok = np.isfinite(overshoot_fraction) and overshoot_fraction <= 0.10
    passed = bool(silent_ok and corr_ok and overshoot_ok)
    return GateVerdict(
        "CG1b", passed,
        {"silent_ok": bool(silent_ok), "growing_rank_corr": growing_rank_corr, "overshoot_fraction": overshoot_fraction},
    )


def cg1c_active_saturation_refusal(
    pop_rise: float, true_pop_rise: float, p95_rise: float, true_p95_rise: float, verdict_is_insufficient_range: bool
) -> GateVerdict:
    pop_ok = pop_rise <= true_pop_rise + 0.02
    p95_ok = p95_rise <= true_p95_rise + 0.03
    passed = bool(pop_ok and p95_ok and verdict_is_insufficient_range)
    return GateVerdict("CG1c", passed, {"pop_ok": bool(pop_ok), "p95_ok": bool(p95_ok), "verdict_ok": bool(verdict_is_insufficient_range)})


def cg2_gate_false_positives(bed_pvalue: float, bh_reject: np.ndarray) -> GateVerdict:
    n = len(bh_reject)
    frac = float(np.mean(bh_reject)) if n else 0.0
    passed = bool(bed_pvalue > 0.01 and frac <= 0.02)
    return GateVerdict("CG2", passed, {"bed_pvalue": bed_pvalue, "bh_discovery_fraction": frac})


def cg3_passive_detection_kdd(
    bed_pvalue: float, discovery_fraction_unsaturated: float, r_rank_corr: float, split_half_gap: float, bank_recovery_corr: float
) -> GateVerdict:
    passed = bool(
        bed_pvalue < 0.001
        and discovery_fraction_unsaturated >= 0.6
        and np.isfinite(r_rank_corr) and r_rank_corr >= 0.7
        and np.isfinite(split_half_gap) and abs(split_half_gap) <= 0.10
        and np.isfinite(bank_recovery_corr) and bank_recovery_corr >= 0.9
    )
    return GateVerdict(
        "CG3", passed,
        {"bed_pvalue": bed_pvalue, "discovery_fraction": discovery_fraction_unsaturated,
         "r_rank_corr": r_rank_corr, "split_half_gap": split_half_gap, "bank_recovery_corr": bank_recovery_corr},
    )


def cg4a_passive_existence_ednet(bed_pvalue: float) -> GateVerdict:
    return GateVerdict("CG4a", bool(bed_pvalue < 0.001), {"bed_pvalue": bed_pvalue})


def cg4b_rate_recovery_ednet(r_rank_corr: float) -> GateVerdict:
    passed = bool(np.isfinite(r_rank_corr) and r_rank_corr >= 0.6)
    return GateVerdict("CG4b", passed, {"r_rank_corr": r_rank_corr, "note": "failure is K7, not a kill"})


def cg5_non_standard_shape(
    bed_pvalue: float, misfit_fire_fraction: float, unflagged_fraction: float, bank_recovery_corr: float
) -> GateVerdict:
    passed = bool(
        bed_pvalue < 0.001
        and misfit_fire_fraction >= 0.8
        and unflagged_fraction <= 0.20
        and np.isfinite(bank_recovery_corr) and bank_recovery_corr >= 0.9
    )
    return GateVerdict(
        "CG5", passed,
        {"bed_pvalue": bed_pvalue, "misfit_fire_fraction": misfit_fire_fraction,
         "unflagged_fraction": unflagged_fraction, "bank_recovery_corr": bank_recovery_corr},
    )


def cg6_saturation_refusal(bed_pvalue: float, saturation_flag_fraction: float, verdict_is_insufficient_range: bool) -> GateVerdict:
    passed = bool(bed_pvalue > 0.05 and saturation_flag_fraction >= 0.95 and verdict_is_insufficient_range)
    return GateVerdict(
        "CG6", passed,
        {"bed_pvalue": bed_pvalue, "saturation_flag_fraction": saturation_flag_fraction, "verdict_ok": bool(verdict_is_insufficient_range)},
    )


def cg7_from_result(r: Cg7Result) -> GateVerdict:
    detail = dict(r.detail)
    detail.update({"trained_rank_corr": r.trained_rank_corr, "frozen_rank_corr": r.frozen_rank_corr, "margin": r.margin})
    return GateVerdict("CG7", r.passed, detail)


def cg8_from_result(r: ContaminationResult) -> GateVerdict:
    return GateVerdict("CG8", r.passed, {"ratio": r.ratio})


def cg9_from_result(r: OrderInvarianceResult) -> GateVerdict:
    return GateVerdict("CG9", r.passed, {"kc_median_corr": r.kc_median_corr, "sign_flip_fraction": r.sign_flip_fraction})


def cg9_act_from_result(r: ActRecognitionStabilityResult) -> GateVerdict:
    return GateVerdict(
        "CG9-ACT", r.passed,
        {"u_median_corr": r.u_median_corr, "lambda_median_corr": r.lambda_median_corr, "rise_profile_min_corr": r.rise_profile_min_corr},
    )


def cg10_direction_audit(violation_fraction: float) -> GateVerdict:
    return GateVerdict("CG10", bool(np.isfinite(violation_fraction) and violation_fraction <= 0.10), {"violation_fraction": violation_fraction})


# =============================================================================
# RB1-RB5, RB-E1: real-bed licensing (design 5.2, 5.3); RB-A is
# `active.rb_a_firing` (imported by `report.py`, not duplicated here)
# =============================================================================


def rb1_in_situ_fabrication(
    bootstrap_bed_pvalue: float, act_pop_rise: Optional[float] = None, act_p95_rise: Optional[float] = None
) -> GateVerdict:
    gate_ok = bootstrap_bed_pvalue > 0.05
    act_ok = True
    if act_pop_rise is not None:
        act_ok = act_ok and abs(act_pop_rise) <= 0.01
    if act_p95_rise is not None:
        act_ok = act_ok and abs(act_p95_rise) <= 0.02
    passed = bool(gate_ok and act_ok)
    return GateVerdict(
        "RB1", passed,
        {"bootstrap_bed_pvalue": bootstrap_bed_pvalue, "act_pop_rise": act_pop_rise, "act_p95_rise": act_p95_rise},
    )


def rb3_existence(unsaturated_bed_pvalue: float, rb0_passed: bool, rb1_passed: bool) -> GateVerdict:
    passed = bool(unsaturated_bed_pvalue < 0.01 and rb0_passed and rb1_passed)
    return GateVerdict("RB3", passed, {"unsaturated_bed_pvalue": unsaturated_bed_pvalue, "rb0_passed": rb0_passed, "rb1_passed": rb1_passed})


def rb4_kc_rates(split_half_kc: float, truncation_passed: bool, misfit_silent_on_claimed: bool) -> GateVerdict:
    passed = bool(np.isfinite(split_half_kc) and split_half_kc >= 0.8 and truncation_passed and misfit_silent_on_claimed)
    return GateVerdict(
        "RB4", passed,
        {"split_half_kc": split_half_kc, "truncation_passed": truncation_passed, "misfit_silent_on_claimed": misfit_silent_on_claimed},
    )


def rb5_learner_rates(split_half_per_stratum: dict, truncation_passed: bool) -> GateVerdict:
    any_stratum_passes = any(np.isfinite(v) and v >= 0.7 for v in split_half_per_stratum.values())
    passed = bool(any_stratum_passes and truncation_passed)
    return GateVerdict(
        "RB5", passed,
        {"split_half_per_stratum": dict(split_half_per_stratum), "truncation_passed": truncation_passed,
         "all_strata_failed": not any_stratum_passes},
    )


def rb_e1_population_corroboration(bed_pvalue: float, rb0_passed: bool, rb1_passed: bool) -> GateVerdict:
    passed = bool(bed_pvalue < 0.01 and rb0_passed and rb1_passed)
    return GateVerdict("RB-E1", passed, {"bed_pvalue": bed_pvalue, "rb0_passed": rb0_passed, "rb1_passed": rb1_passed})


# =============================================================================
# K1-K7: kill conditions (design 5.6). See module docstring note 7.
# =============================================================================


@dataclass
class KillVerdict:
    code: str
    triggered: bool
    detail: dict = field(default_factory=dict)


def k1_posture_machinery(
    cg1_family_passed: bool, cg2_passed: bool, cg5_passed: bool, cg6_passed: bool, cg3_passed: bool,
    revision_budget_exhausted: bool,
) -> KillVerdict:
    active_dead = bool(revision_budget_exhausted and not cg1_family_passed)
    passive_gate_dead = bool(revision_budget_exhausted and not (cg2_passed and cg5_passed and cg6_passed))
    density_floor_retreat = bool(not cg3_passed)
    triggered = bool(active_dead or passive_gate_dead or density_floor_retreat)
    return KillVerdict("K1", triggered, {"active_dead": active_dead, "passive_gate_dead": passive_gate_dead, "density_floor_retreat": density_floor_retreat})


def k2_kdd_existence(rb3_passed: bool) -> KillVerdict:
    return KillVerdict("K2", not rb3_passed, {"rb3_passed": rb3_passed})


def k3_kdd_magnitude(rb4_passed: bool, rb3_passed: bool) -> KillVerdict:
    return KillVerdict("K3", not rb4_passed, {"rb4_passed": rb4_passed, "tier1_stands": rb3_passed})


def k4_ednet(rb_e1_passed: bool) -> KillVerdict:
    return KillVerdict("K4", not rb_e1_passed, {"rb_e1_passed": rb_e1_passed})


def k5_tracker_leg(cg7_passed_any_config: bool, cg8_passed_any_config: bool) -> KillVerdict:
    triggered = bool((not cg7_passed_any_config) and (not cg8_passed_any_config))
    return KillVerdict("K5", triggered, {"cg7_passed_any_config": cg7_passed_any_config, "cg8_passed_any_config": cg8_passed_any_config})


def k6_in_situ_integrity(rb0_passed: bool, rb1_passed: bool) -> KillVerdict:
    triggered = bool(not (rb0_passed and rb1_passed))
    return KillVerdict("K6", triggered, {"rb0_passed": rb0_passed, "rb1_passed": rb1_passed})


def k7_density_floor(cg3_passed: bool, cg4b_passed: bool) -> KillVerdict:
    """K7 never kills; it is an informative, non-fatal finding when CG3
    passes but CG4b (EdNet-density rate recovery) fails."""
    informative_finding = bool(cg3_passed and not cg4b_passed)
    return KillVerdict("K7", False, {"informative_finding": informative_finding, "note": "K7 never kills; density-floor finding only"})


__all__ = [
    "B_KC_DEFAULT", "B_BED_DEFAULT", "GateVerdict", "is_sign_consistent",
    "PermutationBatteryResult", "run_permutation_battery",
    "static_twin_gate", "fit_full_slice_m0", "m0_bootstrap_slices", "m0_bootstrap_learners",
    "Cg7Result", "cg7_verdict", "compute_rb2_bar", "rb2_verdict",
    "build_drill_log", "pas_n1_full_ability_trajectory", "ContaminationResult",
    "contamination_probe", "contamination_diagnostic_band",
    "permute_cross_kc_interleaving", "full_ability_batch", "collect_kc_trajectory",
    "OrderInvarianceResult", "order_invariance_stress",
    "ActRecognitionStabilityResult", "act_recognition_stability",
    "direction_violation_fraction",
    "spearman_brown", "per_slice_statistic_odd_even", "per_learner_split_half",
    "kc_level_split_half", "predicted_split_half_via_bootstrap", "split_half_gate",
    "TRUNCATION_CUTOFFS", "truncate_slice", "truncation_rate_by_cutoff", "TruncationResult", "truncation_stress",
    "seed_pooled_pvalue", "SeedClusterVerdict", "seed_cluster_verdict",
    "rb0_bank_robustness", "bank_recovery_verdict",
    "CG1_POP_MAX", "CG1_P95_MAX_KDD", "CG1_P95_MAX_EDNET",
    "cg1_active_silence", "cg1a_active_positive_control", "cg1b_active_misfit_robustness", "cg1c_active_saturation_refusal",
    "cg2_gate_false_positives", "cg3_passive_detection_kdd", "cg4a_passive_existence_ednet", "cg4b_rate_recovery_ednet",
    "cg5_non_standard_shape", "cg6_saturation_refusal",
    "cg7_from_result", "cg8_from_result", "cg9_from_result", "cg9_act_from_result", "cg10_direction_audit",
    "rb1_in_situ_fabrication", "rb3_existence", "rb4_kc_rates", "rb5_learner_rates", "rb_e1_population_corroboration",
    "KillVerdict", "k1_posture_machinery", "k2_kdd_existence", "k3_kdd_magnitude", "k4_ednet", "k5_tracker_leg",
    "k6_in_situ_integrity", "k7_density_floor",
]
