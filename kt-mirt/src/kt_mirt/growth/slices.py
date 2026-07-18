"""A4 slice layer: per-(learner, KC) opportunity streams, density strata,
the saturation flag, and the permutation-null hook
(`_planning/design/a4_design.md` v1.1, sections 1, 2.1, 2.2, battery arm 1).

Interpretation note (recorded per the harness instructions). `slices.py`
is named in the design's module map (section 2.5) as its own artifact, but
it was not built by either of the two stages that ran before this one
(stage 1: `synth.py`; stage 2: `bank.py`, `newton.py`). PAS-G (`gate.py`),
MIX-L (`rate.py`), and both neural arms (`tracker.py`, `active.py`) cannot
fit or train anything without slice construction and the permutation
hook, so this stage builds the MINIMAL slice layer those four estimators
need:

- Per-(learner, KC) ordered opportunity streams, built by REGROUPING
  `kt_mirt.growth.bank.build_calibration_rows`'s already-vectorized
  per-tag opportunity index (rather than re-deriving the running-count
  logic a second time -- one implementation of "opportunity index", reused
  by the bank calibrator and every downstream estimator).
- Density-stratum classification and the D2+ selection rule (section 1:
  "D0 = slices with 4-5 opportunities, D1 = 6-9, D2 = 10-19, D3 = 20+
  ... Per-slice (per-learner) estimands are defined only on D2+.").
- The raw, model-free per-KC saturation flag (section 2.1: "a RAW
  calibration-cohort per-KC correct rate ... it is not entangled with the
  bank fit").
- The permutation-null hook (battery arm 1: "permute each learner's full
  interaction order, then rebuild slices and opportunity indices ... This
  destroys within-slice trend while preserving slice membership, item
  multisets, marginal rates, and ... multi-tag co-occurrence"). Only the
  hook (one replicate) is implemented here; running B replicates and
  aggregating a null distribution at the pre-registered B=199/999 is
  `battery.py`'s job (a later stage) -- `gate.py`/`rate.py` call this hook
  directly to build small-B null distributions themselves where a
  posture's own machinery needs one (`gate.py`'s permutation p-values).

Explicitly OUT of scope for this module: `curves.py` (PAS-C), `qmatrix.py`
and `kc_data.py` (the REAL-BED raw-file ragged-collation bridge -- on
synthetic twins `synth.py`'s `LearnerLog` already carries the ragged
`tag_ids`/`tag_mask` shape section 2.5 specifies, so no bridge is needed
to build or test the estimators here), and the full `battery.py`
orchestration (all ten arms at pre-registered replicate counts, the
M0-bootstrap twin, contamination/order-invariance probes). Those remain
for whichever stage builds `prep_kdd.py`/`prep_ednet.py`/`battery.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from kt_mirt.growth.bank import CalibrationRows, N_BLOCKS
from kt_mirt.growth.synth import LearnerLog

# ---------------------------------------------------------------------------
# Density strata (section 1)
# ---------------------------------------------------------------------------

D2_PLUS_FLOOR = 10  # "Per-slice (per-learner) estimands are defined only on D2+"
POPULATION_FLOOR = 4  # "Slices with fewer than 4 opportunities enter population curves only"


def density_stratum(n_opportunities: int) -> str:
    """One slice's density stratum label: "sub" (< 4, population-curve-only),
    "D0" (4-5), "D1" (6-9), "D2" (10-19), "D3" (20+)."""
    T = int(n_opportunities)
    if T < POPULATION_FLOOR:
        return "sub"
    if T <= 5:
        return "D0"
    if T <= 9:
        return "D1"
    if T <= 19:
        return "D2"
    return "D3"


def is_d2_plus(n_opportunities) -> np.ndarray:
    """Vectorized D2+ selection rule (``T >= 10``)."""
    return np.asarray(n_opportunities) >= D2_PLUS_FLOOR


# ---------------------------------------------------------------------------
# Slice construction
# ---------------------------------------------------------------------------


@dataclass
class Slice:
    """One (learner, KC) slice: the ordered opportunity stream (design
    section 1's slice/opportunity definition), already regrouped from
    `CalibrationRows` into time order within the (learner, KC) pair."""

    learner: int
    kc: int
    item_id: np.ndarray  # (T,) int
    response: np.ndarray  # (T,) int8
    opportunity: np.ndarray  # (T,) int64, 1-based
    block_id: np.ndarray  # (T,) int64, `bank.opportunity_block(opportunity)`

    @property
    def T(self) -> int:
        return int(len(self.item_id))

    @property
    def stratum(self) -> str:
        return density_stratum(self.T)

    @property
    def is_d2_plus(self) -> bool:
        return self.T >= D2_PLUS_FLOOR


def build_slices(rows: CalibrationRows) -> dict[tuple[int, int], "Slice"]:
    """Regroup `CalibrationRows` (flattened, per-tag opportunity index
    already computed by `bank.build_calibration_rows`) into per-(learner,
    KC) ordered `Slice` objects. Vectorized: one sort over every valid
    (row, tag-slot) pair, then a single pass over group boundaries (no
    per-slice Python-level opportunity computation -- that running count
    is already correct in `rows.opportunity`, computed once)."""
    R, A = rows.kc_ids.shape
    if R == 0 or A == 0:
        return {}
    row_grid = np.repeat(np.arange(R), A)
    slot_grid = np.tile(np.arange(A), R)
    valid = rows.kc_mask.reshape(-1)
    if not valid.any():
        return {}
    rows_v = row_grid[valid]
    slots_v = slot_grid[valid]
    learner_v = rows.learner_idx[rows_v]
    kc_v = rows.kc_ids[rows_v, slots_v]
    opp_v = rows.opportunity[rows_v, slots_v]
    block_v = rows.block_id[rows_v, slots_v]
    item_v = rows.item_id[rows_v]
    resp_v = rows.response[rows_v]

    kc_max = int(kc_v.max()) + 1 if len(kc_v) else 1
    keys = learner_v.astype(np.int64) * kc_max + kc_v.astype(np.int64)
    # Sort by (key, opportunity): each group's rows land in opportunity
    # order (equivalently arrival order, since opportunity is a running
    # count over arrival order within the group).
    order = np.lexsort((opp_v, keys))
    keys_o = keys[order]
    change = np.empty(len(keys_o), dtype=bool)
    change[0] = True
    change[1:] = keys_o[1:] != keys_o[:-1]
    starts = np.where(change)[0]
    ends = np.append(starts[1:], len(keys_o))

    slices: dict[tuple[int, int], Slice] = {}
    for s, e in zip(starts, ends):
        idx = order[s:e]
        learner = int(learner_v[idx[0]])
        kc = int(kc_v[idx[0]])
        slices[(learner, kc)] = Slice(
            learner=learner,
            kc=kc,
            item_id=item_v[idx].copy(),
            response=resp_v[idx].copy(),
            opportunity=opp_v[idx].copy(),
            block_id=block_v[idx].copy(),
        )
    return slices


def slices_by_kc(slices: dict[tuple[int, int], Slice], n_kcs: int) -> list[list[Slice]]:
    """Group already-built slices by KC id (0-based, contiguous)."""
    out: list[list[Slice]] = [[] for _ in range(n_kcs)]
    for (_, kc), sl in slices.items():
        out[kc].append(sl)
    return out


def select_d2_plus(slices: dict[tuple[int, int], Slice]) -> dict[tuple[int, int], Slice]:
    return {k: v for k, v in slices.items() if v.is_d2_plus}


# ---------------------------------------------------------------------------
# Saturation flag (section 2.1: raw, model-free per-KC correct rate)
# ---------------------------------------------------------------------------


def saturation_stats(
    rows: CalibrationRows, n_kcs: int, threshold: float = 0.85
) -> tuple[np.ndarray, np.ndarray]:
    """Per-KC raw correct rate (pooled over every tagged occurrence in
    ``rows``, matching the Q-matrix-expanded granularity every other
    per-KC statistic in this design uses) and the unsaturated-subset flag
    (``rate <= threshold``, section 5.2: "Unsaturated subset = KCs with
    calibration-cohort correct rate <= 0.85"). Model-free: no fitted
    parameter enters this computation (section 2.1's stated rationale for
    why the permutation null's calibration is irrelevant to it).

    Returns ``(rate, is_unsaturated)``, both ``(n_kcs,)``; ``rate`` is
    ``nan`` for a KC with zero tagged occurrences in ``rows``.
    """
    sums = np.zeros(n_kcs)
    counts = np.zeros(n_kcs)
    R, A = rows.kc_ids.shape
    if R and A:
        valid = rows.kc_mask
        if valid.any():
            kc_v = rows.kc_ids[valid]
            resp_v = np.broadcast_to(rows.response[:, None], (R, A))[valid]
            np.add.at(sums, kc_v, resp_v)
            np.add.at(counts, kc_v, 1)
    rate = np.full(n_kcs, np.nan)
    have = counts > 0
    rate[have] = sums[have] / counts[have]
    is_unsaturated = np.where(have, rate <= threshold, False)
    return rate, is_unsaturated


# ---------------------------------------------------------------------------
# Permutation hook (battery arm 1)
# ---------------------------------------------------------------------------


def permute_learner_order(
    learners: Sequence[LearnerLog], rng: np.random.Generator
) -> list[LearnerLog]:
    """Battery arm 1's null-generating operation: "permute each learner's
    full interaction order, then rebuild slices and opportunity indices."
    Shuffling the ROW axis in lockstep (item id, response, and tags all
    move together) preserves slice membership, item multisets, marginal
    rates, and multi-tag co-occurrence while destroying within-slice
    trend -- exactly the design's stated invariants. Rebuilding slices and
    opportunity indices is the caller's job (feed the returned logs back
    through `kt_mirt.growth.bank.build_calibration_rows` then
    `build_slices`), since opportunity indices are a function of row
    order and must be recomputed, not carried over.
    """
    out: list[LearnerLog] = []
    for log in learners:
        T = len(log.item_ids)
        perm = rng.permutation(T)
        out.append(
            LearnerLog(
                learner=log.learner,
                item_ids=np.asarray(log.item_ids)[perm].copy(),
                responses=np.asarray(log.responses)[perm].copy(),
                tag_ids=np.asarray(log.tag_ids)[perm].copy(),
                tag_mask=np.asarray(log.tag_mask)[perm].copy(),
            )
        )
    return out


__all__ = [
    "D2_PLUS_FLOOR",
    "POPULATION_FLOOR",
    "N_BLOCKS",
    "density_stratum",
    "is_d2_plus",
    "Slice",
    "build_slices",
    "slices_by_kc",
    "select_d2_plus",
    "saturation_stats",
    "permute_learner_order",
]
