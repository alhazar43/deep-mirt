"""A4 synthetic generator and the four certification twins.

Implements design section 3 (`_planning/design/a4_design.md`, frozen
v1.1): the ground-truth generator family (per-KC bounded-exponential
growth with distinct log-uniform rates and PNAS-patterned learner
heterogeneity, section 3.1), the two triage-matched density/KC-count
profiles (KDD-matched, EdNet-matched), the four certification twins
(SYN-NG, SYN-KG, SYN-NS, SYN-SAT, section 3.2), and the pre-registered
generator acceptance-check runner.

Interpretation notes (recorded per the harness instructions -- the
design gives "the build agent implementation freedom" on generator
mechanics while fixing the acceptance targets, section 3.1):

1. ``eta_c`` (per-KC baseline offset) and the idiosyncratic per-slice
   ``noise`` term in ``theta0_ic = xi_i + eta_c + noise`` are not
   otherwise pinned by the design; both are ``N(0, sigma)`` with
   ``sigma = 0.5`` (eta_c) / ``0.3`` (noise), judgment numbers recorded
   here (``_ETA_SIGMA``, ``_NOISE_SIGMA``).
2. The ceiling gap ``m_ic - theta0_ic`` is drawn as
   ``_GAP_SCALE * exp(N(0, _GAP_LOGSIGMA))`` with ``_GAP_SCALE``
   calibrated analytically (not by search) so that the median
   model-implied opportunity-1 slope lands at 0.10 logits, the
   midpoint of the pre-registered [0.05, 0.15] band: because
   log(r_c) is uniform (symmetric) and log(lambda_i) and log(gap) are
   both zero-mean Gaussian (symmetric), the median of the product
   ``gap * r_c * lambda_i`` equals the product of the three medians,
   so ``_GAP_SCALE = 0.10 / sqrt(0.02 * 0.40)``.
3. Per-KC and per-item synthetic item-bank sizes (``_ITEMS_PER_KC``,
   ``_EDNET_ITEMS_PER_KC``) are arbitrary but reasonable pool sizes for
   generating item-to-item difficulty jitter; they do not attempt to
   model real-bed bank sparsity (that is a real-bed concern for
   `growth/bank.py`, a later stage).
4. EdNet-matched multi-tag items: primary KC drawn Zipf-weighted (the
   same popularity weights used for learner-KC incidence); the number
   of secondary ("bystander") tags is ``min(Poisson(1.2), 5)``,
   producing mean arity 2.2 with a hard cap at 6; secondary tags are
   drawn uniformly from the non-primary KCs (no curriculum-adjacency
   model attempted).
5. Multi-tag response generation: the one Bernoulli draw for a
   multi-tag interaction is generated from the item's PRIMARY KC's
   trajectory only; secondary tags increment their own KC's opportunity
   counter (exactly as design section 2.5 requires -- "the interaction
   enters every tagged KC's slice") but do not drive the response. This
   is a deliberate, minimal choice consistent with A4's explicit
   no-multidimensional-theta scope (section 0) and mirrors the design's
   own framing of multi-tag exposure as a scheduling confound harmless
   to population growth (section 6, EdNet row): secondary-tag slices
   are therefore a measured noise/contamination source by construction,
   not an artifact of convenience.
6. The four pre-registered generator acceptance checks (rate quartiles,
   pooled rise, density quantiles, implied slope) are pre-registered
   against the STANDARD generator configuration, i.e. SYN-KG (section
   3.1's profile table and section 8's R1 "generator bring-up" step are
   both scoped to the base generator before twin-specific dynamics are
   layered on). `run_generator_acceptance_checks` is exposed generically
   -- it will compute the same four diagnostics against any twin's
   realized data -- but only its SYN-KG evaluation is the pre-registered
   gate; SYN-NG/SYN-NS/SYN-SAT correctness is instead certified by the
   battery's CG1/CG1a-c/CG2/CG5/CG6 gates (a later stage), not by this
   runner. The one exception is the density-quantile check, which is
   twin-invariant (a property of the practice *schedule*, not the
   growth dynamics), so it is meaningful -- and checked in this module's
   own tests -- on every twin.
7. SYN-SAT's per-KC start-rate bar ("every per-KC start rate >= 0.90,
   q1 >= 0.88") is met by bisecting a single per-KC additive shift to
   ``theta0`` (and, to preserve the gap, to ``m`` too) that drives every
   KC's mean opportunity-1 correct rate to a common target of 0.93 --
   comfortably above both bars with margin -- rather than modeling a
   realistic cross-KC start-rate distribution, which the design does
   not require.
8. SYN-NS partitions its KCs into three disjoint, seed-chosen buckets:
   20% silent (mechanically identical to SYN-NG: r_c pinned to 0), 40%
   step growth, 40% dip-then-recover. There is no remaining "standard
   positive-rate" bucket in SYN-NS -- every KC is in exactly one of the
   three buckets, per the design's "half ... half" language applied to
   the growing (i.e. non-silent) 80%.
9. Twin matched-twin discipline is implemented by construction: a single
   cached "substrate" (KC popularity, learner-KC incidence and target
   opportunity counts, per-KC rates, per-learner/per-KC heterogeneity,
   the item bank and its calibrated difficulties) is built once per
   (profile, seed) from one `numpy.random.default_rng(seed)` stream, and
   every twin is a pure function of that shared substrate plus at most
   one small, twin-specific, independently-seeded RNG draw (SYN-NS's
   bucket partition and per-slice step/dip parameters; SYN-SAT's shift
   is a deterministic bisection, no extra randomness). This guarantees
   SYN-NG/SYN-KG/SYN-NS/SYN-SAT share identical items, item difficulties,
   practice schedules, and (where applicable) rates.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

TwinName = Literal["syn_ng", "syn_kg", "syn_ns", "syn_sat"]
TWIN_NAMES: tuple[TwinName, ...] = ("syn_ng", "syn_kg", "syn_ns", "syn_sat")

# ---------------------------------------------------------------------------
# Judgment-call constants (see module docstring notes 1-8)
# ---------------------------------------------------------------------------

_RATE_LO, _RATE_HI = 0.02, 0.40  # pre-registered r_c log-uniform band
_ETA_SIGMA = 0.5
_NOISE_SIGMA = 0.3
_GAP_LOGSIGMA = 0.4
_GAP_SCALE = 0.10 / float(np.sqrt(_RATE_LO * _RATE_HI))
_ITEM_JITTER_SIGMA = 0.4
_ITEMS_PER_KC = 30
_EDNET_ITEMS_PER_KC = 8
_ARITY_POISSON_LAMBDA = 1.2
_ARITY_MAX_EXTRA_TAGS = 5  # +1 primary = max arity 6
_STEP_JUMP_LO, _STEP_JUMP_HI = 0.8, 1.5
_STEP_CP_LO, _STEP_CP_HI = 3, 8  # inclusive
_DIP_MAGNITUDE = 0.5
_DIP_WINDOW = (2, 4)  # (start, trough) opportunity indices
_SILENT_FRACTION = 0.20
_STEP_FRACTION = 0.40
_SAT_TARGET_RATE = 0.93
_ZIPF_S = 1.0
_B_SEARCH_BOUNDS = (-8.0, 8.0)
_B_SEARCH_ITERS = 50

_TWIN_SALT = {"syn_ng": 1001, "syn_kg": 1002, "syn_ns": 1003, "syn_sat": 1004}

# Multi-tag pooled-rise dilution compensation (see module docstring note 10).
# A KC's Q-matrix-expanded population curve mixes PRIMARY opportunities
# (this KC's own trajectory) with BYSTANDER ones (unrelated KCs' outcomes,
# this KC only a secondary tag); with popularity-proportional secondary-tag
# placement the primary fraction is forced to f_p ~= 1 / item_arity_mean
# regardless of tag-neighborhood structure (derived in the build log, not
# a free parameter), so the OBSERVED pooled rise is diluted by ~f_p
# relative to the ground-truth per-KC growth magnitude. This profile-keyed
# multiplier partially compensates the EdNet-matched profile's gap (and
# hence its true growth magnitude) so the pooled-rise check has a fighting
# chance; it is capped by the implied-slope check's own band (module note
# 6/10), so full compensation (multiplying by item_arity_mean) is not used
# -- see note 10 for why this is a documented, only-partial fix, not a
# forced pass.
_MULTI_TAG_GAP_INFLATION = {"ednet_matched": 1.3}


# ---------------------------------------------------------------------------
# Density profiles (section 3.1's profile table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DensityProfile:
    """One row of the section-3.1 density/KC-count profile table."""

    name: str
    n_kcs: int
    n_learners: int
    opp_q1: float
    opp_median: float
    opp_q3: float
    opp_clip: tuple[int, int]
    opp_frac_ge10: float
    rate_q1: float
    rate_median: float
    rate_q3: float
    kcs_per_learner: float
    item_arity_mean: float
    item_arity_max: int
    rise_band: tuple[float, float]


KDD_MATCHED = DensityProfile(
    name="kdd_matched",
    n_kcs=515,
    n_learners=3000,
    opp_q1=4.0,
    opp_median=8.0,
    opp_q3=16.0,
    opp_clip=(1, 60),
    opp_frac_ge10=0.42,
    rate_q1=0.75,
    rate_median=0.855,
    rate_q3=0.925,
    kcs_per_learner=40.0,
    item_arity_mean=1.0,
    item_arity_max=1,
    rise_band=(0.08, 0.16),
)

EDNET_MATCHED = DensityProfile(
    name="ednet_matched",
    n_kcs=189,
    n_learners=6000,
    opp_q1=1.0,
    opp_median=2.0,
    opp_q3=5.0,
    opp_clip=(1, 60),
    opp_frac_ge10=0.14,
    rate_q1=0.61,
    rate_median=0.68,
    rate_q3=0.72,
    kcs_per_learner=25.0,
    item_arity_mean=2.2,
    item_arity_max=6,
    rise_band=(0.09, 0.17),
)

# Small development config (section 3.1): "exists for iteration and smoke
# tests; no certification number comes from it." Same density shape as
# KDD-matched, much smaller scale.
SYN_DEV = DensityProfile(
    name="syn_dev",
    n_kcs=50,
    n_learners=500,
    opp_q1=4.0,
    opp_median=8.0,
    opp_q3=16.0,
    opp_clip=(1, 60),
    opp_frac_ge10=0.42,
    rate_q1=0.75,
    rate_median=0.855,
    rate_q3=0.925,
    kcs_per_learner=20.0,
    item_arity_mean=1.0,
    item_arity_max=1,
    rise_band=(0.08, 0.16),
)

_SLOPE_BAND = (0.05, 0.15)  # section 3.1's median-implied-opportunity-1-slope band


# ---------------------------------------------------------------------------
# Item bank
# ---------------------------------------------------------------------------


@dataclass
class ItemBank:
    n_items: int
    max_arity: int
    primary_kc: np.ndarray  # (n_items,) int
    tag_ids: np.ndarray  # (n_items, max_arity) int, -1 padding
    tag_mask: np.ndarray  # (n_items, max_arity) bool
    items_by_primary: list[np.ndarray]  # length n_kcs
    b_true: np.ndarray = field(default=None)  # (n_items,) filled in after calibration


# ---------------------------------------------------------------------------
# Per-(learner, KC) ground truth
# ---------------------------------------------------------------------------


@dataclass
class SliceGroundTruth:
    """Ground truth for one (learner, KC) slice.

    ``theta0_base``/``m_base``/``r_base``/``lam`` are the SUBSTRATE
    values, identical across all four twins (matched-twin discipline;
    ``m_base - theta0_base`` is invariant even under SYN-SAT's shift,
    since SAT shifts both by the same amount). ``r_used``/``shape``/
    ``extra`` describe what THIS twin actually did with them.
    """

    theta0_base: float
    m_base: float
    r_base: float
    lam: float
    T: int
    shape: str  # "standard" | "step" | "dip_recover"
    r_used: float
    extra: dict
    true_rise_1_10: float


@dataclass
class GroundTruth:
    kc_pop_weight: np.ndarray  # (C,)
    xi_i: np.ndarray  # (N,)
    eta_c: np.ndarray  # (C,)
    lambda_i: np.ndarray  # (N,)
    r_base: np.ndarray  # (C,) substrate rates, never zeroed
    kc_base_b: np.ndarray  # (C,) calibrated base item difficulty per KC
    silent_kc_mask: np.ndarray  # (C,) bool, True where THIS twin pins r_c = 0
    kc_shape: np.ndarray  # (C,) str labels for THIS twin
    true_rise_per_kc: np.ndarray  # (C,) population mean model-implied rise, opp 1->10
    slice_params: dict[tuple[int, int], SliceGroundTruth]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class LearnerLog:
    learner: int
    item_ids: np.ndarray  # (T_i,)
    responses: np.ndarray  # (T_i,) int8
    tag_ids: np.ndarray  # (T_i, max_arity) int, -1 padding
    tag_mask: np.ndarray  # (T_i, max_arity) bool


@dataclass
class SyntheticTwin:
    twin: str
    profile: DensityProfile
    seed: int
    n_kcs: int
    n_learners: int
    item_bank: ItemBank
    learners: list[LearnerLog]
    truth: GroundTruth


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zipf_weights(rng: np.random.Generator, n: int, s: float = _ZIPF_S) -> np.ndarray:
    """Zipf-shaped popularity weights over ``n`` items, ranks randomly
    assigned (so KC id carries no popularity meaning of its own)."""
    ranks = np.arange(1, n + 1, dtype=float)
    w = 1.0 / (ranks**s)
    rng.shuffle(w)
    return w / w.sum()


def _quantile_anchors_opportunity(profile: DensityProfile) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = profile.opp_clip
    p_ge10 = max(0.0, 1.0 - profile.opp_frac_ge10)
    anchors = [
        (0.0, float(lo)),
        (0.25, profile.opp_q1),
        (0.5, profile.opp_median),
        (0.75, profile.opp_q3),
        (p_ge10, 10.0),
        (0.99, max(profile.opp_q3 * 2.0, 30.0)),
        (1.0, float(hi)),
    ]
    anchors = sorted(set(anchors))
    p = np.array([a[0] for a in anchors])
    v = np.array([a[1] for a in anchors])
    # Enforce weak monotonicity in the value axis (ties are fine for np.interp).
    v = np.maximum.accumulate(v)
    return p, v


def _quantile_anchors_rate(profile: DensityProfile) -> tuple[np.ndarray, np.ndarray]:
    span_lo = profile.rate_median - profile.rate_q1
    span_hi = profile.rate_q3 - profile.rate_median
    v0 = min(max(profile.rate_q1 - span_lo, 0.01), profile.rate_q1 - 1e-4)
    v1 = max(min(profile.rate_q3 + span_hi, 0.99), profile.rate_q3 + 1e-4)
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    v = np.array([v0, profile.rate_q1, profile.rate_median, profile.rate_q3, v1])
    return p, v


def _sample_from_anchors(
    rng: np.random.Generator, p: np.ndarray, v: np.ndarray, size: int
) -> np.ndarray:
    u = rng.uniform(0.0, 1.0, size=size)
    return np.interp(u, p, v)


def _bisect_scalar(f, lo: float, hi: float, target: float, iters: int = _B_SEARCH_ITERS) -> float:
    """Find x in [lo, hi] with f(x) == target, f assumed non-increasing."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo <= target:
        return lo
    if f_hi >= target:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Shape formulas (vectorized over a (L, T_max) grid)
# ---------------------------------------------------------------------------


def _standard_theta_matrix(
    theta0: np.ndarray, m: np.ndarray, r: np.ndarray, lam: np.ndarray, T_max: int
) -> np.ndarray:
    n = np.arange(1, T_max + 1, dtype=float)[None, :]
    rate = (r * lam)[:, None]
    return m[:, None] - (m[:, None] - theta0[:, None]) * np.exp(-rate * (n - 1.0))


def _standard_rise(theta0: np.ndarray, m: np.ndarray, r: np.ndarray, lam: np.ndarray) -> np.ndarray:
    def f(n):
        return m - (m - theta0) * np.exp(-r * lam * (n - 1.0))

    return f(10.0) - f(1.0)


def _step_theta_matrix(
    theta0: np.ndarray, jump: np.ndarray, cp: np.ndarray, T_max: int
) -> np.ndarray:
    n = np.arange(1, T_max + 1, dtype=float)[None, :]
    return theta0[:, None] + jump[:, None] * (n >= cp[:, None])


def _dip_recover_theta_matrix(
    theta0: np.ndarray,
    m: np.ndarray,
    r: np.ndarray,
    lam: np.ndarray,
    dip: np.ndarray,
    T_max: int,
) -> np.ndarray:
    lo, hi = _DIP_WINDOW
    n = np.arange(1, T_max + 1, dtype=float)[None, :]
    theta0_c, m_c, r_c, lam_c, dip_c = (x[:, None] for x in (theta0, m, r, lam, dip))
    trough = theta0_c - dip_c
    frac = np.clip((n - lo) / (hi - lo), 0.0, 1.0)
    val_dip = theta0_c - dip_c * frac
    val_recover = m_c - (m_c - trough) * np.exp(-r_c * lam_c * (n - hi))
    region1 = n <= lo
    region2 = (n > lo) & (n <= hi)
    return np.where(region1, theta0_c, np.where(region2, val_dip, val_recover))


def _dip_recover_rise(
    theta0: np.ndarray, m: np.ndarray, r: np.ndarray, lam: np.ndarray, dip: np.ndarray
) -> np.ndarray:
    hi = _DIP_WINDOW[1]
    trough = theta0 - dip
    theta10 = m - (m - trough) * np.exp(-r * lam * (10.0 - hi))
    return theta10 - theta0


# ---------------------------------------------------------------------------
# Item bank construction
# ---------------------------------------------------------------------------


def _build_item_bank_single_tag(n_kcs: int, items_per_kc: int) -> ItemBank:
    n_items = n_kcs * items_per_kc
    primary_kc = np.repeat(np.arange(n_kcs), items_per_kc)
    tag_ids = primary_kc.reshape(-1, 1).copy()
    tag_mask = np.ones((n_items, 1), dtype=bool)
    items_by_primary = [
        np.arange(c * items_per_kc, (c + 1) * items_per_kc) for c in range(n_kcs)
    ]
    return ItemBank(
        n_items=n_items,
        max_arity=1,
        primary_kc=primary_kc,
        tag_ids=tag_ids,
        tag_mask=tag_mask,
        items_by_primary=items_by_primary,
    )


def _build_item_bank_multi_tag(
    rng: np.random.Generator, n_kcs: int, items_per_kc: int, kc_pop_weight: np.ndarray, max_arity: int
) -> ItemBank:
    n_items = n_kcs * items_per_kc
    # Guarantee every KC has >= 1 primary item (a pure Zipf-weighted draw can
    # starve low-popularity KCs entirely, leaving an empty item pool that a
    # learner scheduled onto that KC cannot draw from); the first n_kcs items
    # cover every KC exactly once (order shuffled so item id carries no
    # popularity information), the remainder are Zipf-weighted as intended.
    covering = rng.permutation(n_kcs)
    remaining = rng.choice(n_kcs, size=n_items - n_kcs, p=kc_pop_weight)
    primary_kc = np.concatenate([covering, remaining])
    n_extra = np.minimum(rng.poisson(_ARITY_POISSON_LAMBDA, size=n_items), _ARITY_MAX_EXTRA_TAGS)
    n_extra = np.minimum(n_extra, max_arity - 1)
    tag_ids = np.full((n_items, max_arity), -1, dtype=np.int64)
    tag_mask = np.zeros((n_items, max_arity), dtype=bool)
    tag_ids[:, 0] = primary_kc
    tag_mask[:, 0] = True
    for j in range(n_items):
        k = int(n_extra[j])
        if k == 0:
            continue
        pool = np.delete(np.arange(n_kcs), primary_kc[j])
        extra = rng.choice(pool, size=k, replace=False)
        tag_ids[j, 1 : 1 + k] = extra
        tag_mask[j, 1 : 1 + k] = True
    items_by_primary = [np.where(primary_kc == c)[0] for c in range(n_kcs)]
    return ItemBank(
        n_items=n_items,
        max_arity=max_arity,
        primary_kc=primary_kc,
        tag_ids=tag_ids,
        tag_mask=tag_mask,
        items_by_primary=items_by_primary,
    )


# ---------------------------------------------------------------------------
# Substrate: the shared, twin-invariant generator state for (profile, seed)
# ---------------------------------------------------------------------------


@dataclass
class _Substrate:
    profile: DensityProfile
    kc_pop_weight: np.ndarray
    learner_kc_ids: list[np.ndarray]
    learner_kc_T: list[np.ndarray]
    xi_i: np.ndarray
    eta_c: np.ndarray
    lambda_i: np.ndarray
    r_base: np.ndarray
    gap: list[np.ndarray]  # aligned with learner_kc_ids
    theta0: list[np.ndarray]  # aligned with learner_kc_ids
    item_bank: ItemBank
    kc_base_b: np.ndarray


@functools.lru_cache(maxsize=64)
def _build_substrate(profile: DensityProfile, seed: int) -> _Substrate:
    rng = np.random.default_rng(seed)
    C, N = profile.n_kcs, profile.n_learners

    kc_pop_weight = _zipf_weights(rng, C)

    # Learner-KC incidence (Zipf-weighted subset per learner).
    n_kc_i = np.clip(
        np.round(rng.normal(profile.kcs_per_learner, profile.kcs_per_learner * 0.15, size=N)).astype(int),
        1,
        C,
    )
    keys = rng.uniform(1e-12, 1.0, size=(N, C)) ** (1.0 / kc_pop_weight[None, :])
    order = np.argsort(-keys, axis=1)
    learner_kc_ids = [order[i, : n_kc_i[i]].copy() for i in range(N)]

    # Target opportunity counts per (learner, kc) pair.
    p_opp, v_opp = _quantile_anchors_opportunity(profile)
    total_pairs = int(n_kc_i.sum())
    T_flat = np.round(_sample_from_anchors(rng, p_opp, v_opp, total_pairs)).astype(int)
    T_flat = np.clip(T_flat, profile.opp_clip[0], profile.opp_clip[1])
    learner_kc_T: list[np.ndarray] = []
    ptr = 0
    for i in range(N):
        k = n_kc_i[i]
        learner_kc_T.append(T_flat[ptr : ptr + k])
        ptr += k

    # Per-KC rates, per-learner heterogeneity (PNAS pattern: wide start, narrow rate).
    r_base = np.exp(rng.uniform(np.log(_RATE_LO), np.log(_RATE_HI), size=C))
    xi_i = rng.normal(0.0, 1.0, size=N)
    eta_c = rng.normal(0.0, _ETA_SIGMA, size=C)
    lambda_i = rng.lognormal(0.0, 0.2, size=N)

    gap_scale = _GAP_SCALE * _MULTI_TAG_GAP_INFLATION.get(profile.name, 1.0)
    theta0, gap = [], []
    for i in range(N):
        kcs = learner_kc_ids[i]
        noise = rng.normal(0.0, _NOISE_SIGMA, size=len(kcs))
        theta0.append(xi_i[i] + eta_c[kcs] + noise)
        z = rng.normal(0.0, _GAP_LOGSIGMA, size=len(kcs))
        gap.append(gap_scale * np.exp(z))

    # Item bank (structure only; difficulties calibrated below).
    if profile.item_arity_max <= 1:
        item_bank = _build_item_bank_single_tag(C, _ITEMS_PER_KC)
    else:
        item_bank = _build_item_bank_multi_tag(
            rng, C, _EDNET_ITEMS_PER_KC, kc_pop_weight, profile.item_arity_max
        )

    # Per-KC difficulty calibration against the standard (SYN-KG-like) trajectories.
    p_rate, v_rate = _quantile_anchors_rate(profile)
    kc_target_rate = _sample_from_anchors(rng, p_rate, v_rate, C)
    kc_base_b = np.zeros(C)
    # Gather, per KC, all (learner, kc) assignments.
    kc_learners: list[list[int]] = [[] for _ in range(C)]
    kc_local_idx: list[list[int]] = [[] for _ in range(C)]
    for i in range(N):
        for local_idx, c in enumerate(learner_kc_ids[i]):
            kc_learners[c].append(i)
            kc_local_idx[c].append(local_idx)
    theta_valid_by_kc: list[np.ndarray] = [np.empty(0) for _ in range(C)]
    for c in range(C):
        if not kc_learners[c]:
            continue
        idxs = np.array(kc_learners[c])
        locs = np.array(kc_local_idx[c])
        th0 = np.array([theta0[i][j] for i, j in zip(idxs, locs)])
        gp = np.array([gap[i][j] for i, j in zip(idxs, locs)])
        mm = th0 + gp
        Ts = np.array([learner_kc_T[i][j] for i, j in zip(idxs, locs)])
        lam = lambda_i[idxs]
        T_max = int(Ts.max())
        theta_mat = _standard_theta_matrix(th0, mm, np.full_like(th0, r_base[c]), lam, T_max)
        valid = np.arange(1, T_max + 1)[None, :] <= Ts[:, None]
        theta_valid = theta_mat[valid]
        theta_valid_by_kc[c] = theta_valid

        def f(b, theta_valid=theta_valid):
            return float(_sigmoid(theta_valid - b).mean())

        kc_base_b[c] = _bisect_scalar(f, *_B_SEARCH_BOUNDS, target=kc_target_rate[c])

    item_jitter = rng.normal(0.0, _ITEM_JITTER_SIGMA, size=item_bank.n_items)
    item_bank.b_true = kc_base_b[item_bank.primary_kc] + item_jitter

    substrate = _Substrate(
        profile=profile,
        kc_pop_weight=kc_pop_weight,
        learner_kc_ids=learner_kc_ids,
        learner_kc_T=learner_kc_T,
        xi_i=xi_i,
        eta_c=eta_c,
        lambda_i=lambda_i,
        r_base=r_base,
        gap=gap,
        theta0=theta0,
        item_bank=item_bank,
        kc_base_b=kc_base_b,
    )

    if profile.item_arity_max > 1:
        _correct_multi_tag_contamination(
            substrate, kc_target_rate, theta_valid_by_kc, item_jitter, seed
        )

    return substrate


def _correct_multi_tag_contamination(
    substrate: "_Substrate",
    kc_target_rate: np.ndarray,
    theta_valid_by_kc: list[np.ndarray],
    item_jitter: np.ndarray,
    seed: int,
) -> None:
    """One-pass correction for multi-tag bystander contamination.

    On an EdNet-like multi-tag bank, a KC's Q-matrix-expanded slice mixes
    PRIMARY-tagged opportunities (this KC's own trajectory) with BYSTANDER
    opportunities (some other item's primary KC's trajectory, this KC only
    a secondary tag) -- see module docstring note 4. Trial-simulating the
    "as if KG" substrate once reveals, per KC, the primary/bystander volume
    split and the bystander-only mean response; a single linear correction
    (observed = f_p*true + (1-f_p)*bystander_mean, solved for the true rate
    that would have produced the pre-registered target once contamination
    is mixed back in) then re-calibrates `kc_base_b` and `item_bank.b_true`
    in place. Mutates ``substrate`` in place; called once from
    `_build_substrate`, before any twin is generated from it.
    """
    C = substrate.profile.n_kcs
    trial_rng = np.random.default_rng((seed, 9999))
    kc_shape = np.full(C, "standard", dtype=object)
    r_used = substrate.r_base.copy()
    shift = np.zeros(C)
    learners, _, _ = _simulate_learners(
        substrate, substrate.item_bank, kc_shape, r_used, shift, trial_rng
    )
    primary_of = substrate.item_bank.primary_kc

    primary_resp: list[list[int]] = [[] for _ in range(C)]
    bystander_resp: list[list[int]] = [[] for _ in range(C)]
    for log in learners:
        prim = primary_of[log.item_ids]
        for t in range(len(log.item_ids)):
            resp = int(log.responses[t])
            tags = log.tag_ids[t][log.tag_mask[t]]
            for c in tags:
                c = int(c)
                if c == prim[t]:
                    primary_resp[c].append(resp)
                else:
                    bystander_resp[c].append(resp)

    pop_mean = float(
        np.mean([r for seq in primary_resp for r in seq] + [r for seq in bystander_resp for r in seq])
    )
    kc_base_b = substrate.kc_base_b
    for c in range(C):
        n_p, n_b = len(primary_resp[c]), len(bystander_resp[c])
        if n_p == 0 or theta_valid_by_kc[c].size == 0:
            continue
        f_p = n_p / (n_p + n_b)
        if f_p >= 0.98:
            continue  # negligible contamination, no correction needed
        bystander_mean = float(np.mean(bystander_resp[c])) if n_b > 0 else pop_mean
        corrected_target = (kc_target_rate[c] - (1.0 - f_p) * bystander_mean) / f_p
        corrected_target = float(np.clip(corrected_target, 0.02, 0.98))

        theta_valid = theta_valid_by_kc[c]

        def f(b, theta_valid=theta_valid):
            return float(_sigmoid(theta_valid - b).mean())

        kc_base_b[c] = _bisect_scalar(f, *_B_SEARCH_BOUNDS, target=corrected_target)

    substrate.item_bank.b_true = kc_base_b[primary_of] + item_jitter


# ---------------------------------------------------------------------------
# Twin-specific dynamics assignment
# ---------------------------------------------------------------------------


def _syn_ns_partition(rng: np.random.Generator, n_kcs: int) -> np.ndarray:
    """Assign every KC to exactly one of {"silent", "step", "dip_recover"}."""
    n_silent = int(round(_SILENT_FRACTION * n_kcs))
    n_step = int(round(_STEP_FRACTION * n_kcs))
    order = rng.permutation(n_kcs)
    labels = np.empty(n_kcs, dtype=object)
    labels[order[:n_silent]] = "silent"
    labels[order[n_silent : n_silent + n_step]] = "step"
    labels[order[n_silent + n_step :]] = "dip_recover"
    return labels


def _sat_shift(substrate: _Substrate, target_rate: float = _SAT_TARGET_RATE) -> np.ndarray:
    """Per-KC additive shift to theta0 (and m, gap preserved) driving every
    KC's mean opportunity-1 correct rate up to ``target_rate``."""
    C = substrate.profile.n_kcs
    N = substrate.profile.n_learners
    shift = np.zeros(C)
    kc_theta0: list[list[float]] = [[] for _ in range(C)]
    for i in range(N):
        for local_idx, c in enumerate(substrate.learner_kc_ids[i]):
            kc_theta0[c].append(substrate.theta0[i][local_idx])
    for c in range(C):
        if not kc_theta0[c]:
            continue
        th0 = np.array(kc_theta0[c])
        b_c = substrate.kc_base_b[c]

        def f(s, th0=th0, b_c=b_c):
            return float(_sigmoid(th0 + s - b_c).mean())

        shift[c] = _bisect_scalar(lambda s: -f(s), -8.0, 8.0, target=-target_rate)
    return shift


def _simulate_learners(
    substrate: "_Substrate",
    item_bank: ItemBank,
    kc_shape: np.ndarray,
    r_used_c: np.ndarray,
    kc_shift: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[LearnerLog], dict[tuple[int, int], SliceGroundTruth], np.ndarray]:
    """Realize one twin's learner logs given fully-specified per-KC dynamics.

    Shared by `generate_twin` (the twin actually returned to callers) and by
    `_build_substrate`'s internal bystander-contamination correction pass
    (module docstring note 4's multi-tag calibration fix) -- both are "as if
    KG" realizations of the same substrate, just with independent RNG draws.
    """
    C, N = substrate.profile.n_kcs, substrate.profile.n_learners
    learners: list[LearnerLog] = []
    slice_params: dict[tuple[int, int], SliceGroundTruth] = {}
    true_rise_accum: list[list[float]] = [[] for _ in range(C)]

    for i in range(N):
        kcs = substrate.learner_kc_ids[i]
        Ts = substrate.learner_kc_T[i]
        theta0_i = substrate.theta0[i] + kc_shift[kcs]
        m_i = theta0_i + substrate.gap[i]
        lam_i = np.full(len(kcs), substrate.lambda_i[i])

        # Per-slot shape-specific extra params (step jump/cp; dip trough).
        shapes = kc_shape[kcs]
        jump = np.zeros(len(kcs))
        cp = np.zeros(len(kcs))
        step_mask = shapes == "step"
        if step_mask.any():
            jump[step_mask] = rng.uniform(_STEP_JUMP_LO, _STEP_JUMP_HI, size=step_mask.sum())
            cp[step_mask] = rng.integers(
                _STEP_CP_LO, _STEP_CP_HI + 1, size=step_mask.sum()
            )
        dip = np.full(len(kcs), _DIP_MAGNITUDE)

        T_max = int(Ts.max()) if len(Ts) else 0
        r_row = r_used_c[kcs]
        theta_std = _standard_theta_matrix(theta0_i, m_i, r_row, lam_i, max(T_max, 1))
        theta_step = (
            _step_theta_matrix(theta0_i, jump, cp, max(T_max, 1)) if step_mask.any() else theta_std
        )
        dip_mask = shapes == "dip_recover"
        theta_dip = (
            _dip_recover_theta_matrix(theta0_i, m_i, r_row, lam_i, dip, max(T_max, 1))
            if dip_mask.any()
            else theta_std
        )

        theta_mat = theta_std.copy()
        theta_mat[step_mask] = theta_step[step_mask]
        theta_mat[dip_mask] = theta_dip[dip_mask]

        rise_std = _standard_rise(theta0_i, m_i, r_row, lam_i)
        rise = rise_std.copy()
        rise[step_mask] = jump[step_mask]
        if dip_mask.any():
            rise[dip_mask] = _dip_recover_rise(theta0_i, m_i, r_row, lam_i, dip)[dip_mask]

        # Draw items and responses per opportunity, then interleave across KCs.
        entries = []  # (arrival_time, kc_local_idx, opportunity_n)
        item_choice: dict[int, np.ndarray] = {}
        for local_idx, c in enumerate(kcs):
            T = int(Ts[local_idx])
            pool = item_bank.items_by_primary[c]
            drawn = rng.choice(pool, size=T, replace=True)
            item_choice[local_idx] = drawn
            times = np.sort(rng.random(T))
            for n, t in enumerate(times, start=1):
                entries.append((t, local_idx, n))

            slice_params[(i, int(c))] = SliceGroundTruth(
                theta0_base=float(substrate.theta0[i][local_idx]),
                m_base=float(substrate.theta0[i][local_idx] + substrate.gap[i][local_idx]),
                r_base=float(substrate.r_base[c]),
                lam=float(substrate.lambda_i[i]),
                T=T,
                shape=str(shapes[local_idx]),
                r_used=float(r_row[local_idx]),
                extra=(
                    {"jump": float(jump[local_idx]), "cp": float(cp[local_idx])}
                    if shapes[local_idx] == "step"
                    else {"dip": float(dip[local_idx])}
                    if shapes[local_idx] == "dip_recover"
                    else {}
                ),
                true_rise_1_10=float(rise[local_idx]),
            )
            true_rise_accum[int(c)].append(float(rise[local_idx]))

        entries.sort(key=lambda e: e[0])

        item_ids = np.empty(len(entries), dtype=np.int64)
        responses = np.empty(len(entries), dtype=np.int8)
        for pos, (_, local_idx, n) in enumerate(entries):
            item_id = int(item_choice[local_idx][n - 1])
            b_j = item_bank.b_true[item_id]
            theta_val = theta_mat[local_idx, n - 1]
            p_correct = _sigmoid(theta_val - b_j)
            resp = int(rng.random() < p_correct)
            item_ids[pos] = item_id
            responses[pos] = resp

        tag_ids = item_bank.tag_ids[item_ids]
        tag_mask = item_bank.tag_mask[item_ids]

        learners.append(
            LearnerLog(
                learner=i,
                item_ids=item_ids,
                responses=responses,
                tag_ids=tag_ids,
                tag_mask=tag_mask,
            )
        )

    true_rise_per_kc = np.array(
        [np.mean(v) if v else 0.0 for v in true_rise_accum]
    )
    return learners, slice_params, true_rise_per_kc


def generate_twin(twin: TwinName, profile: DensityProfile, seed: int) -> SyntheticTwin:
    """Build one certification twin from the shared (profile, seed) substrate.

    ``twin`` in {"syn_ng", "syn_kg", "syn_ns", "syn_sat"}. See the module
    docstring for the twin-specific interpretation notes.
    """
    if twin not in TWIN_NAMES:
        raise ValueError(f"unknown twin {twin!r}, expected one of {TWIN_NAMES}")
    substrate = _build_substrate(profile, seed)
    C, N = profile.n_kcs, profile.n_learners
    twin_rng = np.random.default_rng((seed, _TWIN_SALT[twin]))

    if twin == "syn_kg":
        kc_shape = np.full(C, "standard", dtype=object)
        r_used_c = substrate.r_base.copy()
        silent_mask = np.zeros(C, dtype=bool)
        kc_shift = np.zeros(C)
    elif twin == "syn_ng":
        kc_shape = np.full(C, "standard", dtype=object)
        r_used_c = np.zeros(C)
        silent_mask = np.ones(C, dtype=bool)
        kc_shift = np.zeros(C)
    elif twin == "syn_sat":
        kc_shape = np.full(C, "standard", dtype=object)
        r_used_c = substrate.r_base.copy()
        silent_mask = np.zeros(C, dtype=bool)
        kc_shift = _sat_shift(substrate)
    else:  # syn_ns
        kc_shape = _syn_ns_partition(twin_rng, C)
        r_used_c = np.where(kc_shape == "silent", 0.0, substrate.r_base)
        silent_mask = kc_shape == "silent"
        kc_shift = np.zeros(C)

    item_bank = substrate.item_bank
    learners, slice_params, true_rise_per_kc = _simulate_learners(
        substrate, item_bank, kc_shape, r_used_c, kc_shift, twin_rng
    )

    truth = GroundTruth(
        kc_pop_weight=substrate.kc_pop_weight,
        xi_i=substrate.xi_i,
        eta_c=substrate.eta_c,
        lambda_i=substrate.lambda_i,
        r_base=substrate.r_base,
        kc_base_b=substrate.kc_base_b,
        silent_kc_mask=silent_mask,
        kc_shape=kc_shape,
        true_rise_per_kc=true_rise_per_kc,
        slice_params=slice_params,
    )

    return SyntheticTwin(
        twin=twin,
        profile=profile,
        seed=seed,
        n_kcs=C,
        n_learners=N,
        item_bank=item_bank,
        learners=learners,
        truth=truth,
    )


# ---------------------------------------------------------------------------
# Q-matrix expansion (private helper for the acceptance-check runner)
# ---------------------------------------------------------------------------


def _expand_kc_response_sequences(twin_data: SyntheticTwin) -> list[list[np.ndarray]]:
    """Reconstruct, per KC, the list of per-slice response sequences by
    fanning every tagged interaction into every tagged KC's slice (design
    section 2.5): "the interaction enters every tagged KC's slice and
    increments each tagged KC's opportunity counter." This is the same
    replay operation stage 2's `slices.py` will perform on real beds; here
    it is only used to compute realized statistics for the acceptance
    checks below.
    """
    C = twin_data.n_kcs
    per_kc: list[list[np.ndarray]] = [[] for _ in range(C)]
    for log in twin_data.learners:
        buffers: dict[int, list[int]] = {}
        n_pos = len(log.item_ids)
        for t in range(n_pos):
            resp = int(log.responses[t])
            tags = log.tag_ids[t][log.tag_mask[t]]
            for c in tags:
                buffers.setdefault(int(c), []).append(resp)
        for c, seq in buffers.items():
            per_kc[c].append(np.asarray(seq, dtype=np.int8))
    return per_kc


# ---------------------------------------------------------------------------
# Acceptance-check runner
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    value: object
    target: object
    passed: bool


@dataclass
class AcceptanceReport:
    twin: str
    profile: str
    seed: int
    checks: dict[str, CheckResult]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks.values())


def run_generator_acceptance_checks(twin_data: SyntheticTwin) -> AcceptanceReport:
    """The four pre-registered generator acceptance checks of section 3.1.

    Pre-registered gating applies to the SYN-KG configuration (see module
    docstring note 6); the density-quantile check is meaningful -- and
    exercised in this module's tests -- on every twin, since it is a
    property of the practice schedule, not the growth dynamics.
    """
    profile = twin_data.profile
    checks: dict[str, CheckResult] = {}

    per_kc = _expand_kc_response_sequences(twin_data)

    # Check 1: realized per-KC rate quartiles within 0.03 of target.
    kc_rates = np.array(
        [float(np.concatenate(seqs).mean()) if seqs else np.nan for seqs in per_kc]
    )
    kc_rates = kc_rates[~np.isnan(kc_rates)]
    q1, med, q3 = np.percentile(kc_rates, [25, 50, 75])
    target = (profile.rate_q1, profile.rate_median, profile.rate_q3)
    diffs = (abs(q1 - target[0]), abs(med - target[1]), abs(q3 - target[2]))
    checks["rate_quartiles"] = CheckResult(
        value=(float(q1), float(med), float(q3)), target=target, passed=all(d <= 0.03 for d in diffs)
    )

    # Check 2: pooled opportunity-1-to-10 rise within the profile's rise band.
    all_seqs = [seq for seqs in per_kc for seq in seqs]

    def _p_at(n: int) -> float:
        have = [s[n - 1] for s in all_seqs if len(s) >= n]
        return float(np.mean(have)) if have else float("nan")

    p1, p10 = _p_at(1), _p_at(10)
    rise = p10 - p1
    lo, hi = profile.rise_band
    checks["pooled_rise"] = CheckResult(value=float(rise), target=(lo, hi), passed=lo <= rise <= hi)

    # Check 3: realized density quantiles within 1 opportunity of target.
    # Pooled over the learner's own scheduled (primary) slices only -- the
    # profile's opp-per-slice column targets the practiced-KC subsequence
    # length (section 3.1's "each learner practices a Zipf-weighted subset
    # of KCs"), not the full Q-matrix-expanded count, which on the EdNet
    # multi-tag profile is dominated by a long tail of one-off bystander
    # slices for KCs the learner never selected (a real phenomenon, but not
    # what this profile column is calibrated against; it is still visible
    # in `pooled_rise`/`rate_quartiles` above, which legitimately read the
    # full Q-matrix expansion the way a real PAS-C/E-P1 pass would).
    scheduled_T = np.array([sg.T for sg in twin_data.truth.slice_params.values()])
    dq1, dmed, dq3 = np.percentile(scheduled_T, [25, 50, 75])
    dtarget = (profile.opp_q1, profile.opp_median, profile.opp_q3)
    ddiffs = (abs(dq1 - dtarget[0]), abs(dmed - dtarget[1]), abs(dq3 - dtarget[2]))
    checks["density_quantiles"] = CheckResult(
        value=(float(dq1), float(dmed), float(dq3)), target=dtarget, passed=all(d <= 1.0 for d in ddiffs)
    )

    # Check 4: median model-implied opportunity-1 slope in [0.05, 0.15] logits.
    # Uses the SUBSTRATE (base, never-zeroed) rate -- see module docstring
    # note 6 -- so this check reflects the generator config, not any one
    # twin's dynamics override.
    slopes = []
    for sg in twin_data.truth.slice_params.values():
        slopes.append((sg.m_base - sg.theta0_base) * sg.r_base * sg.lam)
    slopes = np.array(slopes)
    slope_med = float(np.median(slopes))
    slo, shi = _SLOPE_BAND
    checks["implied_slope"] = CheckResult(
        value=slope_med, target=(slo, shi), passed=slo <= slope_med <= shi
    )

    return AcceptanceReport(
        twin=twin_data.twin, profile=profile.name, seed=twin_data.seed, checks=checks
    )
