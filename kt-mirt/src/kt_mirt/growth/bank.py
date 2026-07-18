"""A4 shared measurement layer: the frozen item-difficulty bank every
posture reads (`_planning/design/a4_design.md` v1.1, section 2.1).

Implements: AFM-style calibration with a per-KC BLOCKWISE growth-
absorption profile (``logit P(y=1) = a_i + u_c(B(n)) - b_j``, section
2.1), hierarchical shrinkage of item difficulty under sparsity (steps
under problems under hierarchies, step offsets fit only at the
response-count floor, section 2.1's KDD identification fix), the Adam
MAP optimizer spec (minibatched over rows, relative-NLL-change plus
parameter-drift convergence), the ``FrozenBank`` object every posture
(PASSIVE/MIXED/ACTIVE) reads through, the cohort split, and battery arm
10's two inputs: synthetic b-recovery (rank corr vs generator truth) and
the tri-spec (no-growth / linear / blockwise) real-bed stability refit.

This module does NOT implement the permutation battery, the gate, or the
rate stage (`growth/gate.py`, `growth/rate.py`, `growth/battery.py`, later
stages); it exposes the calibration primitives and the two battery-arm-10
check functions those later modules call.

Interpretation notes (recorded per the harness instructions):

1. **Growth-term multi-tag extension.** Section 2.1 writes the
   calibration model for a single KC per interaction, ``u_c(B(n))``, but
   both real beds have multi-tag items (section 6). Consistent with the
   design's repeated multi-KC convention ("the interaction enters every
   tagged KC's slice and increments each tagged KC's opportunity
   counter", sections 1, 2.5, 6), the growth-absorption term used here is
   the SUM of ``u_c(B(n_c))`` (or, in the linear tri-spec arm,
   ``gamma_c*(n_c-1)``) over every KC tag of the item, each evaluated at
   that KC's own running opportunity index within the learner's slice.
   This is a minimal, order-consistent generalization: it purges the same
   curriculum-position confound the single-KC formula purges, additively
   over tags, and collapses to the single-tag formula exactly when
   ``item_arity_max == 1`` (KDD).
2. **Priors on the nuisance parameters ``a_i`` (learner intercept) and
   ``u_c``/``gamma_c`` (growth absorption).** Section 2.1 pins the item-
   hierarchy prior scales (1.5 / 1.0 / 0.5 logits) but not these. Both
   are nuisance parameters discarded after calibration, but Adam MAP
   still needs a proper prior on every parameter (the design's own
   stated rationale for the item-hierarchy priors: "no b diverges on
   all-correct or all-incorrect exposure"; the same separation risk
   exists for ``a_i`` on an all-correct/all-incorrect learner, and for
   ``u_c`` on a KC whose block is empty or saturated). ``prior_sigma_u =
   2.0`` reuses the exact scale section 2.2 pins for the structurally
   identical block-offset term in PAS-G's M1b (``N(0, 2.0^2)``);
   ``prior_sigma_a = 3.0`` is a new, wider judgment number (a learner-
   level ability nuisance plausibly spans a larger range than a single
   KC's within-block deviation).
3. **Adam convergence.** Section 2.1: "convergence = relative NLL change
   < 1e-4 over 3 consecutive epochs plus a parameter-drift check." The
   relative-NLL leg is read on the full calibration-cohort data NLL
   (excluding the prior term, evaluated at each epoch's end, no-grad),
   compared epoch-to-epoch. The parameter-drift leg is the max absolute
   change, epoch-to-epoch, in the FROZEN quantity itself (b_hat, over
   items with >=1 calibration-cohort exposure -- unexposed items never
   move and would trivially satisfy any drift bound) -- this is the
   quantity whose stability actually matters downstream, rather than a
   drift bound on the nuisance parameters. ``drift_tol = 1e-3`` logits is
   a new judgment number, picked well inside the smallest prior scale
   (0.5) so a "converged" bank cannot still be visibly moving on the
   scale every posture reads.
4. **Minibatch prior scaling.** Each minibatch's loss is (summed BCE over
   the batch's rows) + (batch_fraction * full prior penalty), with
   ``batch_fraction = batch_rows / n_calibration_rows``. This is the
   standard unbiased-stochastic-MAP scaling (equivalent to importance-
   weighting the prior by how much of the data each step "represents");
   the design specifies Adam-over-minibatches but not this detail.
5. **Hierarchy is a general N-level chain**, not hardcoded to 3 levels,
   so the identical code path serves both KDD (3 levels: hierarchy >
   problem > step, `build_kdd_hierarchy`) and EdNet/synthetic (1 level,
   `flat_hierarchy`) exactly as section 2.1 requires ("EdNet: ... a
   single-level b_j ~ N(0, 1.5^2) suffices, same code path"). Only the
   FINEST (leaf) level is subject to an exposure floor; coarser levels
   are always fit (their own proper prior is the only regularizer they
   need, matching "b_H ~ N(0, 1.5^2), e_P ~ N(0, 1.0^2)" having no stated
   floor, unlike ``d_j``). A flat, single-level hierarchy has no floor by
   construction (``FLAT_HIERARCHY_SPEC.leaf_floor = None``): with one
   level there is no coarser level to identify a sparse leaf against, so
   flooring would just zero out real items with no fallback, which is
   not what "EdNet: dense, single-level b_j suffices" describes.
6. **Analysis-cohort read rule / "is calibrated".** Per section 2.1,
   "Interactions on items whose problem (KDD) or item (EdNet) is
   entirely unseen in calibration are excluded from gate/rate
   likelihoods but still increment opportunity counters." ``FrozenBank.
   is_calibrated`` reports per-item whether the item's PARENT-OF-LEAF
   unit (the "problem" for a >=2-level hierarchy; the item itself for a
   flat one) had >=1 calibration-cohort response; excluding those rows
   from downstream likelihoods (while still counting opportunities) is
   the calling module's (`growth/slices.py`, a later stage)
   responsibility -- this module only exposes the boolean.
7. **Rank correlation is computed locally** (``spearman_rank_correlation``
   below), not via ``scipy.stats.spearmanr``, even though battery arm 10
   is fundamentally a Spearman-correlation statistic: the computation
   (rank + Pearson correlation of ranks, with average-rank tie handling)
   is a few lines of numpy and does not need a new runtime dependency
   (scipy stays a test-only dependency here, matching the vendor
   report's finding that `core/` needs no scipy either).
8. **Blockwise growth term is zero-mean-across-blocks per KC.** Section
   2.1 specifies ``u_c(B(n))`` as "a free per-KC piecewise-constant
   profile" without an explicit normalization. Taken completely
   unconstrained, ``u_c`` is exactly collinear with that KC's mean item
   difficulty whenever item draws are independent of opportunity (true
   of every twin in this design and, plausibly, close to true on real
   beds too outside genuine curriculum-position effects): shifting every
   block of ``u_c`` for KC ``c`` by a constant and every item difficulty
   in KC ``c`` by the same constant leaves every logit unchanged. Left
   unconstrained, this flat direction is resolved arbitrarily by the
   relative prior tightness of ``u_c`` vs. ``b_j`` rather than by data,
   which was measured (during build) to visibly corrupt ``b_hat`` recovery
   (a same-scale no-growth check dropped to rank corr ~0.33 instead of
   the ~0.82 a plain 1PL fit reaches) even with NO true opportunity-
   difficulty confound present. The fit therefore uses ``u_c(block) =
   u_raw_c(block) - mean_block(u_raw_c(.))`` (a zero-mean-across-blocks
   reparameterization, differentiable, no separate constraint needed):
   this removes exactly the one flat (collinear) direction while leaving
   the SHAPE across blocks -- early-vs-late-opportunity structure -- fully
   free, which is all "difficulty is not confounded with when-in-learning
   an item tends to appear" (2.1's own stated rationale) requires. The
   ``linear`` tri-spec arm needs no such fix: ``gamma_c*(n-1)`` is
   already zero at ``n=1`` by construction, so it carries no free level
   to confound with ``b_j``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Opportunity blocks (section 2.1/2.2's pre-registered edges, shared with
# PAS-G's M1b -- the SAME block edges, "the same block edges as M1b")
# ---------------------------------------------------------------------------

BLOCK_EDGES = (3, 7, 15)  # blocks: [1-3], [4-7], [8-15], [16+]
N_BLOCKS = 4


def opportunity_block(n: np.ndarray) -> np.ndarray:
    """Map a 1-based opportunity index (or array thereof) to its block id
    in ``{0, 1, 2, 3}`` for edges ``{1-3, 4-7, 8-15, 16+}``."""
    n = np.asarray(n)
    return np.where(n <= 3, 0, np.where(n <= 7, 1, np.where(n <= 15, 2, 3))).astype(np.int64)


# ---------------------------------------------------------------------------
# Item hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchySpec:
    """One growth-calibration bank's prior scales and leaf exposure floor.
    ``level_sigmas`` runs coarsest-first (matches ``ItemHierarchy.
    level_paths`` order); its length must equal the hierarchy's own
    number of levels."""

    level_sigmas: tuple[float, ...]
    leaf_floor: Optional[int]  # min calibration-cohort responses for d_j; None = always fit


KDD_HIERARCHY_SPEC = HierarchySpec(level_sigmas=(1.5, 1.0, 0.5), leaf_floor=20)
FLAT_HIERARCHY_SPEC = HierarchySpec(level_sigmas=(1.5,), leaf_floor=None)


@dataclass
class ItemHierarchy:
    """Maps each item id (0-based, contiguous, ``0..n_items-1``) to its
    path of level-local ids, one array per level, coarsest first, finest
    (leaf) last. A single-level hierarchy (``len(level_paths) == 1``) is
    the EdNet/synthetic flat code path."""

    n_items: int
    level_paths: tuple[np.ndarray, ...]  # each (n_items,) int64, 0-based within-level id
    level_sizes: tuple[int, ...]


def build_item_hierarchy(level_raw_ids: Sequence[np.ndarray]) -> ItemHierarchy:
    """Build an ``ItemHierarchy`` from already fully-QUALIFIED per-level id
    arrays (coarsest first), one array per level, each shape ``(n_items,)``.
    "Qualified" means: two items sharing a level-``k`` id are asserted to
    really share that ancestor (e.g. two steps with id 5 at the finest
    level really are the same step) -- this function only factorizes
    whatever ids it is given into contiguous 0-based indices; composing
    ids that are unique only WITHIN a parent (e.g. raw KDD problem/step
    names, which repeat across different hierarchies) is the caller's job
    (see ``build_kdd_hierarchy`` below for the KDD case)."""
    if not level_raw_ids:
        raise ValueError("build_item_hierarchy requires at least one level")
    n_items = len(level_raw_ids[0])
    for arr in level_raw_ids:
        if len(arr) != n_items:
            raise ValueError("all level id arrays must have the same length (n_items)")
    level_paths: list[np.ndarray] = []
    level_sizes: list[int] = []
    for arr in level_raw_ids:
        _, inv = np.unique(np.asarray(arr), return_inverse=True)
        inv = np.asarray(inv).reshape(-1).astype(np.int64)
        level_paths.append(inv)
        level_sizes.append(int(inv.max()) + 1 if n_items else 0)
    return ItemHierarchy(n_items=n_items, level_paths=tuple(level_paths), level_sizes=tuple(level_sizes))


def flat_hierarchy(n_items: int) -> ItemHierarchy:
    """Single-level hierarchy (EdNet / synthetic code path, section 2.1:
    "a single-level b_j ~ N(0, 1.5^2) suffices, same code path")."""
    return build_item_hierarchy([np.arange(n_items)])


def _factorize_composite(*columns: np.ndarray) -> np.ndarray:
    """Factorize rows of zipped columns into 0-based contiguous ids (a
    composite-key qualifier, e.g. (hierarchy, problem) -> one id, so that
    a raw problem name repeated under two different hierarchies gets two
    distinct ids)."""
    cols = [list(c) for c in columns]
    n = len(cols[0]) if cols else 0
    mapping: dict = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        key = tuple(col[i] for col in cols)
        idx = mapping.get(key)
        if idx is None:
            idx = len(mapping)
            mapping[key] = idx
        out[i] = idx
    return out


def build_kdd_hierarchy(
    hierarchy_id: np.ndarray, problem_name: np.ndarray, step_name: np.ndarray
) -> ItemHierarchy:
    """3-level hierarchy (design section 2.1): hierarchy > problem
    (Problem Hierarchy + Problem Name) > step (+ Step Name). Problem and
    step raw names are qualified by their ancestors BEFORE factorization,
    since the same problem/step name can recur under different
    hierarchies (they are not globally unique on their own)."""
    n_items = len(hierarchy_id)
    level0 = _factorize_composite(hierarchy_id)
    level1 = _factorize_composite(hierarchy_id, problem_name)
    level2 = _factorize_composite(hierarchy_id, problem_name, step_name)
    return ItemHierarchy(
        n_items=n_items,
        level_paths=(level0, level1, level2),
        level_sizes=(
            int(level0.max()) + 1 if n_items else 0,
            int(level1.max()) + 1 if n_items else 0,
            int(level2.max()) + 1 if n_items else 0,
        ),
    )


# ---------------------------------------------------------------------------
# Learner-cohort split
# ---------------------------------------------------------------------------


def split_learners(
    n_learners: int, seed: int, calib_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Random learner split into calibration/analysis cohorts (design
    section 2.1: "Split each real bed's learners at random into a
    calibration cohort (50%) and an analysis cohort (50%)."). Returns
    sorted 0-based index arrays, disjoint and covering ``0..n_learners-1``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_learners)
    n_calib = int(round(n_learners * calib_frac))
    return np.sort(perm[:n_calib]), np.sort(perm[n_calib:])


# ---------------------------------------------------------------------------
# Calibration rows: flattened interactions + per-tag opportunity indices
# ---------------------------------------------------------------------------


class LearnerRecord(Protocol):
    """Structural type any per-learner interaction log must satisfy to
    feed `build_calibration_rows` -- exactly the shape
    `kt_mirt.growth.synth.LearnerLog` already has (duck-typed, no import
    dependency on `synth.py` from this module)."""

    item_ids: np.ndarray  # (T,)
    responses: np.ndarray  # (T,)
    tag_ids: np.ndarray  # (T, A), -1 padding
    tag_mask: np.ndarray  # (T, A) bool


@dataclass
class CalibrationRows:
    """Flattened per-interaction rows across a set of learners, with
    per-tag opportunity indices already computed (design section 1's
    opportunity counter, replayed once here so the growth-absorption
    term has block ids to read without depending on a separate slice-
    construction module)."""

    learner_idx: np.ndarray  # (R,) int, into the learner list passed in
    item_id: np.ndarray  # (R,) int
    response: np.ndarray  # (R,) int8 in {0, 1}
    kc_ids: np.ndarray  # (R, A) int, -1 pad
    kc_mask: np.ndarray  # (R, A) bool
    opportunity: np.ndarray  # (R, A) int64, 1-based per-tag running count
    block_id: np.ndarray  # (R, A) int64, `opportunity_block(opportunity)`

    def __len__(self) -> int:
        return int(self.learner_idx.shape[0])


def build_calibration_rows(learners: Sequence[LearnerRecord]) -> CalibrationRows:
    """Flatten a list of per-learner logs into `CalibrationRows`, computing
    each row's per-tag opportunity index (the running count of prior
    interactions tagged with that KC, within that learner, in sequence
    order -- design section 1's slice/opportunity definition). Vectorized
    via a stable sort on a composite (learner, kc) key per tag slot,
    rather than a Python-level double loop, since real beds run to
    millions of rows (KDD: 8.9M)."""
    if not learners:
        return CalibrationRows(
            learner_idx=np.zeros(0, dtype=np.int64),
            item_id=np.zeros(0, dtype=np.int64),
            response=np.zeros(0, dtype=np.int8),
            kc_ids=np.zeros((0, 0), dtype=np.int64),
            kc_mask=np.zeros((0, 0), dtype=bool),
            opportunity=np.zeros((0, 0), dtype=np.int64),
            block_id=np.zeros((0, 0), dtype=np.int64),
        )

    learner_idx = np.concatenate(
        [np.full(len(rec.item_ids), i, dtype=np.int64) for i, rec in enumerate(learners)]
    )
    item_id = np.concatenate([np.asarray(rec.item_ids, dtype=np.int64) for rec in learners])
    response = np.concatenate([np.asarray(rec.responses, dtype=np.int8) for rec in learners])
    kc_ids = np.concatenate([np.asarray(rec.tag_ids, dtype=np.int64) for rec in learners], axis=0)
    kc_mask = np.concatenate([np.asarray(rec.tag_mask, dtype=bool) for rec in learners], axis=0)

    R, A = kc_ids.shape
    opportunity = np.zeros((R, A), dtype=np.int64)
    kc_max = int(kc_ids.max()) + 1 if R else 1
    if R and A and kc_mask.any():
        # Flatten every valid (row, slot) pair TOGETHER (not slot-by-slot):
        # the same KC can appear in different tag slots across different
        # rows (a multi-tag item's secondary tag this row can be another
        # item's primary tag next row), and the running opportunity count
        # must follow ROW (time) order within a (learner, kc) stream
        # regardless of which slot column it came from.
        row_grid = np.repeat(np.arange(R), A)
        slot_grid = np.tile(np.arange(A), R)
        valid = kc_mask.reshape(-1)
        rows_v = row_grid[valid]
        slots_v = slot_grid[valid]
        kc_v = kc_ids[rows_v, slots_v]
        learner_v = learner_idx[rows_v]
        # Stable sort by (learner, kc), tie-broken by row order, so each
        # group's entries stay in arrival sequence regardless of slot.
        order = np.lexsort((rows_v, kc_v, learner_v))
        keys = learner_v[order].astype(np.int64) * kc_max + kc_v[order].astype(np.int64)
        change = np.empty(len(keys), dtype=bool)
        change[0] = True
        change[1:] = keys[1:] != keys[:-1]
        group_start = np.where(change)[0]
        group_len = np.diff(np.append(group_start, len(keys)))
        cumcount = np.arange(len(keys)) - np.repeat(group_start, group_len)
        opportunity[rows_v[order], slots_v[order]] = cumcount + 1

    block_id = opportunity_block(opportunity)
    return CalibrationRows(
        learner_idx=learner_idx,
        item_id=item_id,
        response=response,
        kc_ids=kc_ids,
        kc_mask=kc_mask,
        opportunity=opportunity,
        block_id=block_id,
    )


def _subset_rows(rows: CalibrationRows, row_mask: np.ndarray) -> CalibrationRows:
    return CalibrationRows(
        learner_idx=rows.learner_idx[row_mask],
        item_id=rows.item_id[row_mask],
        response=rows.response[row_mask],
        kc_ids=rows.kc_ids[row_mask],
        kc_mask=rows.kc_mask[row_mask],
        opportunity=rows.opportunity[row_mask],
        block_id=rows.block_id[row_mask],
    )


# ---------------------------------------------------------------------------
# Leaf eligibility / parent-seen (analysis-cohort read rule inputs)
# ---------------------------------------------------------------------------


def compute_leaf_eligibility(
    calib_item_ids: np.ndarray, n_items: int, floor: Optional[int]
) -> np.ndarray:
    """Per-item bool: True iff eligible for a free finest-level (``d_j``)
    offset, i.e. ``floor is None`` (flat hierarchy, always fit) or the
    item has ``>= floor`` calibration-cohort responses."""
    if floor is None:
        return np.ones(n_items, dtype=bool)
    counts = np.zeros(n_items, dtype=np.int64)
    if len(calib_item_ids):
        np.add.at(counts, calib_item_ids, 1)
    return counts >= floor


def compute_calib_exposure_count(calib_item_ids: np.ndarray, n_items: int) -> np.ndarray:
    """Per-item raw calibration-cohort response count."""
    counts = np.zeros(n_items, dtype=np.int64)
    if len(calib_item_ids):
        np.add.at(counts, calib_item_ids, 1)
    return counts


def compute_parent_seen(hierarchy: ItemHierarchy, calib_item_ids: np.ndarray) -> np.ndarray:
    """Per-item bool: True iff the item's PARENT-OF-LEAF unit (its
    "problem" for a >=2-level hierarchy, or the item itself for a flat
    single-level hierarchy) had >=1 calibration-cohort response (design's
    analysis-cohort read rule, section 2.1: "A step unseen in calibration
    whose problem WAS calibrated reads b_H + e_P ... Interactions on items
    whose problem ... is entirely unseen in calibration are excluded.")."""
    n_levels = len(hierarchy.level_sizes)
    parent_level = max(n_levels - 2, 0)
    parent_ids_per_item = hierarchy.level_paths[parent_level]
    seen_parent_units = np.zeros(hierarchy.level_sizes[parent_level], dtype=bool)
    if len(calib_item_ids):
        seen_units = parent_ids_per_item[calib_item_ids]
        seen_parent_units[np.unique(seen_units)] = True
    return seen_parent_units[parent_ids_per_item]


# ---------------------------------------------------------------------------
# The Adam MAP calibration model
# ---------------------------------------------------------------------------

GrowthMode = str  # "none" | "linear" | "blockwise"


class _BankModel(nn.Module):
    """logit P(y=1) = a_i + growth_term - item_difficulty(item)."""

    def __init__(
        self,
        hierarchy: ItemHierarchy,
        n_calib_learners: int,
        n_kcs: int,
        growth_mode: GrowthMode,
        eligible_leaf: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if growth_mode not in ("none", "linear", "blockwise"):
            raise ValueError(f"unknown growth_mode {growth_mode!r}")
        self.growth_mode = growth_mode
        self.level_params = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(sz, dtype=dtype, device=device))
                for sz in hierarchy.level_sizes
            ]
        )
        level_paths = torch.stack(
            [torch.as_tensor(p, dtype=torch.long, device=device) for p in hierarchy.level_paths],
            dim=1,
        )  # (n_items, n_levels)
        self.register_buffer("level_paths", level_paths)
        self.register_buffer(
            "eligible_leaf", torch.as_tensor(eligible_leaf, dtype=dtype, device=device)
        )
        self.a = nn.Parameter(torch.zeros(max(n_calib_learners, 1), dtype=dtype, device=device))
        if growth_mode == "blockwise":
            self.growth: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(n_kcs, N_BLOCKS, dtype=dtype, device=device)
            )
        elif growth_mode == "linear":
            self.growth = nn.Parameter(torch.zeros(n_kcs, dtype=dtype, device=device))
        else:
            self.growth = None

    def item_difficulty(self, item_id: Tensor) -> Tensor:
        paths = self.level_paths[item_id]  # (R, n_levels)
        n_levels = paths.shape[1]
        b = torch.zeros(item_id.shape[0], device=item_id.device, dtype=self.a.dtype)
        for lvl in range(n_levels):
            contrib = self.level_params[lvl][paths[:, lvl]]
            if lvl == n_levels - 1:
                contrib = contrib * self.eligible_leaf[item_id]
            b = b + contrib
        return b

    def growth_term(
        self, kc_ids: Tensor, kc_mask: Tensor, opportunity: Tensor, block_id: Tensor
    ) -> Tensor:
        if self.growth_mode == "none" or self.growth is None:
            return torch.zeros(kc_ids.shape[0], device=kc_ids.device, dtype=self.a.dtype)
        kc_clamped = kc_ids.clamp(min=0)
        if self.growth_mode == "blockwise":
            # Zero-mean-across-blocks reparameterization (recorded
            # interpretation, module docstring note 8): an UNCONSTRAINED
            # per-(KC, block) offset is exactly collinear with that KC's
            # own mean item difficulty whenever items are drawn
            # independently of opportunity (no true curriculum-position
            # confound to correct) -- shifting u_c(*, all blocks) by any
            # constant k and every item difficulty in that KC by the same
            # k leaves every logit unchanged. Centering each KC's block
            # profile to zero mean removes exactly this one flat
            # direction while leaving the SHAPE across blocks (early vs.
            # late opportunity) fully free, which is the only thing
            # "difficulty is not confounded with when-in-learning an item
            # tends to appear" (section 2.1) actually needs: a shape
            # correction, not an absolute per-KC level competing with b_j.
            centered = self.growth - self.growth.mean(dim=1, keepdim=True)
            vals = centered[kc_clamped, block_id]
        else:  # linear
            vals = self.growth[kc_clamped] * (opportunity.to(self.a.dtype) - 1.0)
        vals = vals * kc_mask.to(vals.dtype)
        return vals.sum(dim=-1)

    def forward(
        self,
        item_id: Tensor,
        learner_local_idx: Tensor,
        kc_ids: Tensor,
        kc_mask: Tensor,
        opportunity: Tensor,
        block_id: Tensor,
    ) -> Tensor:
        b = self.item_difficulty(item_id)
        g = self.growth_term(kc_ids, kc_mask, opportunity, block_id)
        a = self.a[learner_local_idx]
        return a + g - b

    def prior_penalty(
        self, level_sigmas: Sequence[float], prior_sigma_a: float, prior_sigma_u: float
    ) -> Tensor:
        pen = torch.zeros((), device=self.a.device, dtype=self.a.dtype)
        for lvl_param, sigma in zip(self.level_params, level_sigmas):
            pen = pen + 0.5 * (lvl_param**2).sum() / (sigma**2)
        pen = pen + 0.5 * (self.a**2).sum() / (prior_sigma_a**2)
        if self.growth is not None:
            pen = pen + 0.5 * (self.growth**2).sum() / (prior_sigma_u**2)
        return pen


@dataclass
class BankModelConfig:
    prior_sigma_a: float = 3.0  # judgment call, see module docstring note 2
    prior_sigma_u: float = 2.0  # matches PAS-G's M1b block-offset prior, note 2
    n_epochs_max: int = 300
    lr: float = 1e-2
    batch_size: int = 4096
    patience_epochs: int = 3  # "3 consecutive epochs" (section 2.1)
    rel_tol: float = 1e-4
    drift_tol: float = 1e-3  # judgment call, see module docstring note 3
    seed: int = 0
    device: str = "cpu"


@dataclass
class BankFitResult:
    hierarchy: ItemHierarchy
    growth_mode: str
    b_hat: np.ndarray  # (n_items,) frozen difficulty, every item
    eligible_leaf: np.ndarray  # (n_items,) bool
    problem_seen: np.ndarray  # (n_items,) bool, analysis-cohort read rule input
    calib_exposure_count: np.ndarray  # (n_items,) int, raw calibration response count
    converged: bool
    n_epochs_run: int
    final_data_nll: float
    nll_trace: list = field(default_factory=list)


def calibrate_bank(
    rows: CalibrationRows,
    n_learners: int,
    n_kcs: int,
    calib_learners: np.ndarray,
    hierarchy: ItemHierarchy,
    hierarchy_spec: HierarchySpec,
    growth_mode: GrowthMode = "blockwise",
    config: Optional[BankModelConfig] = None,
) -> BankFitResult:
    """Fit the hierarchical, growth-absorbing 1PL bank by Adam MAP on the
    calibration cohort only (design section 2.1); return every item's
    frozen difficulty plus the diagnostics `FrozenBank`/battery arm 10
    need. ``growth_mode`` selects which of the tri-spec RB0 arms this
    call fits ("none" | "linear" | "blockwise", the last being the
    primary calibration spec)."""
    if len(hierarchy_spec.level_sigmas) != len(hierarchy.level_sizes):
        raise ValueError(
            "hierarchy_spec.level_sigmas must have one entry per hierarchy level "
            f"({len(hierarchy_spec.level_sigmas)} given, {len(hierarchy.level_sizes)} levels)"
        )
    config = config or BankModelConfig()
    device = torch.device(config.device)
    dtype = torch.float32

    is_calib = np.zeros(n_learners, dtype=bool)
    is_calib[calib_learners] = True
    row_mask = is_calib[rows.learner_idx] if len(rows) else np.zeros(0, dtype=bool)
    calib_rows = _subset_rows(rows, row_mask)

    learner_map = -np.ones(n_learners, dtype=np.int64)
    learner_map[calib_learners] = np.arange(len(calib_learners))
    learner_local = learner_map[calib_rows.learner_idx] if len(calib_rows) else np.zeros(0, dtype=np.int64)

    n_items = hierarchy.n_items
    eligible_leaf = compute_leaf_eligibility(calib_rows.item_id, n_items, hierarchy_spec.leaf_floor)
    problem_seen = compute_parent_seen(hierarchy, calib_rows.item_id)
    calib_exposure_count = compute_calib_exposure_count(calib_rows.item_id, n_items)

    model = _BankModel(
        hierarchy=hierarchy,
        n_calib_learners=len(calib_learners),
        n_kcs=n_kcs,
        growth_mode=growth_mode,
        eligible_leaf=eligible_leaf,
        device=device,
        dtype=dtype,
    ).to(device)

    n_rows = len(calib_rows)
    if n_rows == 0:
        b_hat = np.zeros(n_items)
        return BankFitResult(
            hierarchy=hierarchy,
            growth_mode=growth_mode,
            b_hat=b_hat,
            eligible_leaf=eligible_leaf,
            problem_seen=problem_seen,
            calib_exposure_count=calib_exposure_count,
            converged=True,
            n_epochs_run=0,
            final_data_nll=0.0,
            nll_trace=[],
        )

    t_item_id = torch.as_tensor(calib_rows.item_id, dtype=torch.long, device=device)
    t_learner = torch.as_tensor(learner_local, dtype=torch.long, device=device)
    t_kc_ids = torch.as_tensor(calib_rows.kc_ids, dtype=torch.long, device=device)
    t_kc_mask = torch.as_tensor(calib_rows.kc_mask, dtype=torch.bool, device=device)
    t_opportunity = torch.as_tensor(calib_rows.opportunity, dtype=torch.long, device=device)
    t_block_id = torch.as_tensor(calib_rows.block_id, dtype=torch.long, device=device)
    t_response = torch.as_tensor(calib_rows.response, dtype=dtype, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rng = torch.Generator(device="cpu").manual_seed(config.seed)

    def full_data_nll() -> float:
        with torch.no_grad():
            logits = model(t_item_id, t_learner, t_kc_ids, t_kc_mask, t_opportunity, t_block_id)
            nll = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, t_response, reduction="sum"
            )
        return float(nll.item())

    def frozen_b_hat() -> np.ndarray:
        with torch.no_grad():
            all_items = torch.arange(n_items, dtype=torch.long, device=device)
            return model.item_difficulty(all_items).cpu().numpy()

    nll_trace: list[float] = []
    prev_b: Optional[np.ndarray] = frozen_b_hat()
    good_epochs = 0
    converged = False
    n_epochs_run = 0

    for epoch in range(config.n_epochs_max):
        perm = torch.randperm(n_rows, generator=rng).to(device)
        for start in range(0, n_rows, config.batch_size):
            idx = perm[start : start + config.batch_size]
            optimizer.zero_grad()
            logits = model(
                t_item_id[idx],
                t_learner[idx],
                t_kc_ids[idx],
                t_kc_mask[idx],
                t_opportunity[idx],
                t_block_id[idx],
            )
            data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, t_response[idx], reduction="sum"
            )
            batch_fraction = idx.shape[0] / n_rows
            loss = data_loss + batch_fraction * model.prior_penalty(
                hierarchy_spec.level_sigmas, config.prior_sigma_a, config.prior_sigma_u
            )
            loss.backward()
            optimizer.step()

        n_epochs_run = epoch + 1
        cur_nll = full_data_nll()
        nll_trace.append(cur_nll)
        cur_b = frozen_b_hat()

        rel_change = abs(cur_nll - nll_trace[-2]) / max(abs(nll_trace[-2]), 1e-8) if len(nll_trace) > 1 else float("inf")
        drift = float(np.max(np.abs(cur_b - prev_b))) if prev_b is not None else float("inf")
        prev_b = cur_b

        if rel_change < config.rel_tol and drift < config.drift_tol:
            good_epochs += 1
        else:
            good_epochs = 0
        if good_epochs >= config.patience_epochs:
            converged = True
            break

    b_hat = frozen_b_hat()
    return BankFitResult(
        hierarchy=hierarchy,
        growth_mode=growth_mode,
        b_hat=b_hat,
        eligible_leaf=eligible_leaf,
        problem_seen=problem_seen,
        calib_exposure_count=calib_exposure_count,
        converged=converged,
        n_epochs_run=n_epochs_run,
        final_data_nll=nll_trace[-1] if nll_trace else 0.0,
        nll_trace=nll_trace,
    )


# ---------------------------------------------------------------------------
# The frozen-scale object every posture reads
# ---------------------------------------------------------------------------


@dataclass
class FrozenBank:
    """The single shared artifact every posture (PASSIVE, MIXED, ACTIVE)
    reads item difficulties through (design section 2.1). Immutable after
    construction; exposes only difficulty lookups plus the analysis-
    cohort read rule -- it never re-fits or absorbs analysis-cohort data."""

    hierarchy: ItemHierarchy
    b_hat: np.ndarray
    _is_calibrated: np.ndarray
    growth_mode: str
    n_epochs_run: int
    converged: bool

    def difficulty(self, item_id) -> np.ndarray:
        """Frozen ``b_j`` for the given item id(s) (scalar or array)."""
        return self.b_hat[np.asarray(item_id)]

    def is_calibrated(self, item_id) -> np.ndarray:
        """True iff the item's parent-of-leaf unit had >=1 calibration-
        cohort response (design's analysis-cohort exclusion rule, section
        2.1); rows failing this are excluded from downstream likelihoods
        by the calling module but still increment opportunity counters."""
        return self._is_calibrated[np.asarray(item_id)]


def freeze_bank(fit: BankFitResult) -> FrozenBank:
    """Freeze a `BankFitResult` into the read-only `FrozenBank` object
    (design section 2.1: "Freeze b_j; discard a_i and u_c.")."""
    return FrozenBank(
        hierarchy=fit.hierarchy,
        b_hat=fit.b_hat,
        _is_calibrated=fit.problem_seen,
        growth_mode=fit.growth_mode,
        n_epochs_run=fit.n_epochs_run,
        converged=fit.converged,
    )


# ---------------------------------------------------------------------------
# Battery arm 10: bank-robustness audit inputs
# ---------------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank ranking (ties share the mean rank of their span),
    the one ingredient Spearman correlation needs beyond Pearson."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sorted_a = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of ``x`` and ``y`` (equal length), local
    implementation (see module docstring note 7 for why this avoids a new
    scipy runtime dependency). Returns ``nan`` if either side is constant
    (rank correlation undefined)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0 or len(x) != len(y):
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def synthetic_bank_recovery_check(
    fit: BankFitResult, b_true: np.ndarray, min_calib_exposure: int = 1
) -> dict:
    """Battery arm 10(i): "calibrated b must rank-correlate >= 0.9 with
    the generator's true b" (SYN-KG-KDD, SYN-NS). Restricted to items
    with >= ``min_calib_exposure`` calibration-cohort responses: an item
    never seen in calibration carries no calibration signal at all (its
    frozen value is the untouched prior mean), so including it would only
    dilute the recoverability read the design is asking for; a floor of 1
    is the minimal restriction consistent with the design's unqualified
    "calibrated b" (recorded interpretation)."""
    mask = fit.calib_exposure_count >= min_calib_exposure
    corr = spearman_rank_correlation(fit.b_hat[mask], np.asarray(b_true)[mask])
    return {
        "rank_corr": corr,
        "n_items": int(mask.sum()),
        "passed": (not np.isnan(corr)) and corr >= 0.9,
    }


@dataclass
class TriSpecRefitResult:
    no_growth: BankFitResult
    linear: BankFitResult
    blockwise: BankFitResult


def tri_spec_refit(
    rows: CalibrationRows,
    n_learners: int,
    n_kcs: int,
    calib_learners: np.ndarray,
    hierarchy: ItemHierarchy,
    hierarchy_spec: HierarchySpec,
    config: Optional[BankModelConfig] = None,
) -> TriSpecRefitResult:
    """Battery arm 10 / RB0's tri-spec real-bed refit: no-growth / linear
    (gamma_c) / blockwise (u_c), same calibration cohort and hierarchy,
    varying only the growth-absorption term."""
    kwargs = dict(
        rows=rows,
        n_learners=n_learners,
        n_kcs=n_kcs,
        calib_learners=calib_learners,
        hierarchy=hierarchy,
        hierarchy_spec=hierarchy_spec,
        config=config,
    )
    return TriSpecRefitResult(
        no_growth=calibrate_bank(growth_mode="none", **kwargs),
        linear=calibrate_bank(growth_mode="linear", **kwargs),
        blockwise=calibrate_bank(growth_mode="blockwise", **kwargs),
    )


def bank_pairwise_stability(tri: TriSpecRefitResult, min_calib_responses: int = 50) -> dict:
    """RB0's pairwise rank-correlation matrix "at the finest fitted
    level, restricted to units with >= 50 calibration responses" (design
    section 2.1 / battery arm 10). The finest fitted level IS the item
    (step) level here (item == step by construction, section 6), so the
    restriction is exactly `calib_exposure_count >= min_calib_responses`,
    identical across the three specs (the growth-absorption term does not
    change which rows are calibration rows)."""
    fits = {"no_growth": tri.no_growth, "linear": tri.linear, "blockwise": tri.blockwise}
    counts = tri.blockwise.calib_exposure_count
    mask = counts >= min_calib_responses
    pairs = (("no_growth", "linear"), ("no_growth", "blockwise"), ("linear", "blockwise"))
    result: dict = {"n_items": int(mask.sum())}
    correlations = []
    for a, b in pairs:
        corr = spearman_rank_correlation(fits[a].b_hat[mask], fits[b].b_hat[mask])
        result[f"{a}_vs_{b}"] = corr
        correlations.append(corr)
    result["passed"] = all((not np.isnan(c)) and c >= 0.95 for c in correlations)
    return result
