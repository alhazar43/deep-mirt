"""PAS-G, the PASSIVE existence gate (`_planning/design/a4_design.md` v1.1,
section 2.2): M0 (constant-ability null) vs. the M1 two-member dynamic
family (M1a linear trend, M1b blockwise), fit under the shared penalized
bounded Newton primitive (`newton.py`), evaluated on the interpolative
odd/even split, with BH-FDR / Benjamini-Yekutieli discovery control and a
permutation-null hook. This module is also MIX-L's stage 1 verbatim
(design section 2.4: "Existence gate = PAS-G verbatim ... one
implementation, ONE result"); `rate.py` calls `fit_m1b_pooled` and
`fit_kc_joint` directly rather than re-implementing them.

Interpretation notes (recorded per the harness instructions):

1. **Two pooling parameterizations of M1a, two of M1b, matched by
   pooling level.** Section 2.2 gives M1a two named variants ("beta =
   beta_c shared across slices within KC (pooled gate) or beta = delta_ic
   per slice (per-slice gate, D2+ only)") but describes M1b with only the
   per-KC-shared form ("theta_ic + u_c(B(n)), a per-KC free profile").
   Read literally, M1b therefore has no per-slice-gate counterpart -- but
   the per-slice gate (D2+) is a real, pre-registered pooling level (5.1's
   CG2 lists "per-slice" reads is not explicit, but section 2.2 itself:
   "Pooling levels: per KC, per bed, per slice (D2+)" applies to "the gate
   statistic" generally, and "the M1a/M1b selection handled by taking the
   max" is stated for that same general gate statistic, not qualified to
   only the KC/bed levels). A per-slice M1b needs a per-slice-only
   parameterization; the natural, minimal one (symmetric with M1a's own
   per-slice delta_ic) is a set of 4 free PER-SLICE block intercepts (no
   separate free theta_ic column -- the 4 block intercepts already nest
   M0 exactly at the point where all four are equal, so no extra
   parameter is needed to recover the null). This is implemented as
   `fit_m1b_slice` (P=4, design = one-hot(block_id)) alongside the
   pooled `fit_m1b_pooled` (per-KC shared u_c, free per-slice theta_ic,
   via `fit_kc_joint`). Both are exercised: the per-slice gate takes
   max(M1a_slice, M1b_slice); the per-KC/per-bed gate takes max(M1a_pooled,
   M1b_pooled).
2. **KC/bed-pooled fits are one Newton call per KC (S=1 batch), not one
   call batched jointly across every KC.** A KC's pooled fit couples
   every one of its slices' free theta_ic to one shared parameter block
   (beta_c or the 4-vector u_c) -- a genuinely joint optimization, unlike
   M0/M1a-slice/M1b-slice which are independent per slice and hence fully
   vmapped across ALL slices in the bed in one call. Different KCs have
   different slice counts (a heterogeneous parameter-vector length P per
   KC), which does not fit the batched primitive's fixed-P-per-batch-row
   contract without padding every KC in the bed to the bed's max slice
   count (large, mostly-wasted memory on the real beds' skewed KC-size
   distributions). `fit_kc_joint` therefore loops over KCs in Python,
   reusing the SAME shared Newton primitive (`penalized_bounded_newton`,
   by default dispatched to `newton.py`'s arrow-structured fast path --
   see note 5 -- S=1 per call) -- functionally identical machinery, no
   duplicated calculus, just not vectorized across the KC axis. A fully
   KC-batched variant (grouping same-slice-count KCs, or padding) is a
   real-bed performance optimization left to whichever later stage tunes
   battery-scale runtime; it changes wall-clock, not any estimator's
   output.
3. **No explicit zero-mean-across-block centering on the gate's `u_c`**
   (unlike `bank.py`'s blockwise growth-absorption term, which needed it).
   The bank's centering fix exists because ``u_c`` there is nearly
   collinear with a WEAKLY-PENALIZED, effectively-free item difficulty
   ``b_j``, causing the flat direction to resolve almost arbitrarily and
   measurably corrupt ``b_hat`` recovery. Here ``b_j`` is FROZEN (not
   fit), and both ``theta_ic`` (one per slice) and ``u_c``/``beta_c``
   carry the SAME proper Gaussian prior (`N(0, 2.0^2)`, section 2.2's
   pre-registered scale for every slice-level parameter). The one flat
   direction (shift every slice's theta_ic by +k, shift u_c's mean by -k)
   is then a strictly convex function of k under the sum of the two
   independent quadratic penalties (positive curvature from both sides),
   so the penalized MAP is unique and finite with no divergence risk --
   the prior fully resolves the split, it just resolves it as a ridge
   partition rather than an exact zero-mean constraint. This is provable
   from the shared curvature argument, not merely assumed; a unit test
   below checks the fit stays finite and stable on a slice population
   with many slices per KC.
4. **The permutation null here is a REUSABLE hook at small B for
   `gate.py`'s own within-module tests and any posture that needs a
   direct p-value (`gate_pvalue_bed`, `gate_pvalue_per_kc`), not the
   pre-registered B=199/999 battery run.** `battery.py` (a later stage)
   is the module that runs the full replicate count, caches the
   rebuilt-slice tensors in memory per arm 1's binding implementation
   constraint, and reports BH/BY jointly across the whole certification
   matrix; this module exposes the per-replicate statistic computation
   and the BH/BY correction formulas it needs, at whatever B the caller
   supplies.
5. **`fit_kc_joint`/`fit_kc_joint_batched_replicates` default to
   `newton.py`'s arrow-structured fast path** (2026-07-20, 4th A4
   perf-surgery, `_planning/LEDGER.md`: the true root cause of the
   permutation battery's production timeouts). Note 2's per-KC Newton
   call couples every slice's free theta_ic to the KC's shared block
   (beta_c or u_c); on oversized real KCs (k up to ~3000 slices) the
   resulting P~k+shared_dim penalized Hessian was solved DENSELY, paying
   O(P^3) per Newton iteration for a matrix that is provably an ARROW
   (block-diagonal-bordered) structure -- each row's one-hot intercept
   column touches exactly one slice, so the intercept-intercept block is
   exactly diagonal, bordered by the small (shared_dim = 1 or 4) shared
   block. `binary_logit_nll_arrow` (`newton.py`) solves this via exact
   Schur-complement block elimination, O(k * shared_dim^2 + shared_dim^3)
   per iteration and O(k * shared_dim) memory instead of O(k^2) -- EXACT
   algebra, not an approximation, verified against the dense solve
   directly (`tests/test_growth_newton.py`) and end-to-end against this
   module's original dense one-hot-design path
   (`tests/test_growth_gate_perf_equivalence.py`). Both KC-joint fit
   functions take a ``use_arrow`` kwarg (default True); ``False`` reaches
   the original dense path (`_build_kc_joint_arrays` +
   `binary_logit_nll`), kept as that equivalence reference and a manual
   escape hatch.
6. **Saturation-aware bed statistic (`bed_kc_mask`).** The passive
   existence gate FALSE-FIRES on near-ceiling (saturated) data: the
   constant-ability null M0 approximates a saturating success curve worse
   than the dynamic M1, so M1 wins a held-out NLL edge with no real
   growth present (the synthetic certification's CG6 inversion on the
   syn_sat twin -- gate fires p=0.001 where it must stay silent;
   `_planning/verdict_synthetic_complete.md` CHECK 2 / section 2). A
   saturated KC carries no observable growth information, so it must not
   vote in the BED-LEVEL existence decision. `compute_gate_result` and
   every permutation-null path therefore accept an optional ``(n_kcs,)``
   boolean ``bed_kc_mask``: only KCs with ``bed_kc_mask[c] == True`` are
   summed into ``bed_stat`` (and into each null replicate's bed
   statistic). ``None`` (default) sums every KC -- the frozen behavior,
   preserved for all existing callers and tests. The mask this fix relies
   on is the RAW, model-free calibration-cohort saturation flag
   (`slices.saturation_stats`'s ``is_unsaturated``: per-KC calib correct
   rate <= 0.85, design section 2.1/5.2's "unsaturated subset"), which is
   permutation-invariant (a calibration-cohort property, computed once
   and disjoint from the permuted analysis cohort), so the SAME mask is
   passed to the observed gate and to the null and both bed statistics are
   summed over an identical KC set -- the permutation calibration is
   preserved exactly. Two properties the fix guarantees: (i) the per-KC
   ``kc_stat``/``kc_selected_family`` are NEVER altered for any KC,
   saturated or not -- only bed-level inclusion changes, so the frozen
   per-KC statistic and its BH/BY discovery control are untouched; (ii)
   when EVERY KC is saturated (the syn_sat limit) the bed statistic sums
   to 0 for observed and null alike, and `empirical_pvalue` returns 1.0
   (correctly NULL). `run.py`'s `run_slice_cell` sources the mask from
   the calibration-cohort ``calib_unsaturated`` flag and passes it to both
   `compute_gate_result` and `permutation_null`. THIS FIX HAS A HARD
   DEPENDENCY ON THE SATURATION FLAG: without a trustworthy per-KC
   saturation flag the bed statistic reverts (mask=None) to the
   false-firing frozen behavior on near-ceiling data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from kt_mirt.growth.bank import (
    CalibrationRows,
    FrozenBank,
    N_BLOCKS,
    build_calibration_rows,
    opportunity_block,
)
from kt_mirt.growth.newton import (
    NewtonResult,
    binary_logit_nll,
    binary_logit_nll_arrow,
    penalized_bounded_newton,
)
from kt_mirt.growth.slices import Slice, build_slices, permute_learner_order, select_d2_plus, slices_by_kc

PRIOR_VAR = 4.0  # N(0, 2.0^2) logits, section 2.2's pre-registered slice-level prior


# ---------------------------------------------------------------------------
# Design-matrix builders (per model)
# ---------------------------------------------------------------------------


def _m0_design(opp: np.ndarray, block: np.ndarray) -> np.ndarray:
    return np.ones((len(opp), 1), dtype=np.float32)


def _m1a_slice_design(opp: np.ndarray, block: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(opp, dtype=np.float32), (opp - 1).astype(np.float32)], axis=1)


def _m1b_slice_design(opp: np.ndarray, block: np.ndarray) -> np.ndarray:
    d = np.zeros((len(opp), N_BLOCKS), dtype=np.float32)
    d[np.arange(len(opp)), block] = 1.0
    return d


def _m1a_shared_design(opp: np.ndarray, block: np.ndarray) -> np.ndarray:
    return (opp - 1).astype(np.float32).reshape(-1, 1)


def _m1b_shared_design(opp: np.ndarray, block: np.ndarray) -> np.ndarray:
    d = np.zeros((len(opp), N_BLOCKS), dtype=np.float32)
    d[np.arange(len(opp)), block] = 1.0
    return d


def _time_filter_odd(opp: np.ndarray) -> np.ndarray:
    return (opp % 2) == 1


def _time_filter_even(opp: np.ndarray) -> np.ndarray:
    return (opp % 2) == 0


def _validity_mask(sl: Slice, bank: Optional[FrozenBank]) -> np.ndarray:
    if bank is None:
        return np.ones(sl.T, dtype=bool)
    return bank.is_calibrated(sl.item_id)


# ---------------------------------------------------------------------------
# Per-slice batched fits (M0, M1a-slice, M1b-slice)
# ---------------------------------------------------------------------------


@dataclass
class SliceFit:
    keys: list[tuple[int, int]]
    params: torch.Tensor  # (S, P)
    P: int


def _pad_design(
    slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    P: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    S = len(slices)
    T_max = max((sl.T for sl in slices), default=0)
    y = torch.zeros(S, T_max)
    mask = torch.zeros(S, T_max, dtype=torch.bool)
    logit_no_theta = torch.zeros(S, T_max)
    design = torch.zeros(S, T_max, P)
    for i, sl in enumerate(slices):
        t = sl.T
        if t == 0:
            continue
        valid = _validity_mask(sl, bank)
        if time_filter is not None:
            valid = valid & time_filter(sl.opportunity)
        y[i, :t] = torch.as_tensor(sl.response, dtype=torch.float32)
        mask[i, :t] = torch.as_tensor(valid, dtype=torch.bool)
        b = bank.difficulty(sl.item_id) if bank is not None else np.zeros(t)
        logit_no_theta[i, :t] = torch.as_tensor(-b, dtype=torch.float32)
        design[i, :t] = torch.as_tensor(design_fn(sl.opportunity, sl.block_id), dtype=torch.float32)
    return y.to(device), mask.to(device), logit_no_theta.to(device), design.to(device)


def fit_batched(
    slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    P: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    prior_var: float = PRIOR_VAR,
    device: str = "cpu",
) -> SliceFit:
    """Fit a per-slice-independent GLM model (M0, M1a-slice, or M1b-slice)
    batched across every slice in one Newton call."""
    keys = [(sl.learner, sl.kc) for sl in slices]
    if not slices:
        return SliceFit(keys=[], params=torch.zeros(0, P), P=P)
    y, mask, logit_no_theta, design = _pad_design(slices, bank, design_fn, P, time_filter, device)
    x0 = torch.zeros(len(slices), P, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll, x0, data_args=(y, mask, logit_no_theta, design), prior_var=prior_var
    )
    return SliceFit(keys=keys, params=result.params, P=P)


def held_out_nll(
    slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    params: torch.Tensor,
    time_filter: Callable[[np.ndarray], np.ndarray],
    device: str = "cpu",
) -> torch.Tensor:
    """Evaluate the (unpenalized) Bernoulli NLL of a FIXED per-slice
    parameter vector on the positions ``time_filter`` selects (the
    interpolative held-out read; no fitting happens here)."""
    P = params.shape[1] if len(slices) else 0
    if not slices:
        return torch.zeros(0)
    y, mask, logit_no_theta, design = _pad_design(slices, bank, design_fn, P, time_filter, device)
    logits = (design * params.unsqueeze(1)).sum(dim=-1) + logit_no_theta
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per_pos * mask.to(per_pos.dtype)).sum(dim=-1)


def fit_batched_replicates(
    slices_by_replicate: Sequence[Sequence[Slice]],
    bank: Optional[FrozenBank],
    design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    P: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    prior_var: float = PRIOR_VAR,
    device: str = "cpu",
) -> torch.Tensor:
    """The A4 perf-surgery fix's M0 primitive: batches `fit_batched`'s
    per-slice-independent model across BOTH the slice axis (as
    `fit_batched` already does) AND the permutation-replicate axis, in
    ONE Newton call (batch = B * S), instead of one `fit_batched` call
    per replicate repeated B times in a Python loop.

    ``slices_by_replicate[b]`` must list the SAME S slices, in the SAME
    key order, for every replicate ``b`` -- callers build this from one
    canonical key list, since slice membership and per-slice length T are
    permutation-invariant (`slices.permute_learner_order` only reorders
    each learner's own rows; see `permutation_null_batched`). Returns
    ``(B, S, P)`` fitted params.
    """
    B = len(slices_by_replicate)
    S = len(slices_by_replicate[0]) if B else 0
    if B == 0 or S == 0:
        return torch.zeros(B, S, P)

    ys, masks, logits, designs = [], [], [], []
    for b in range(B):
        y, mask, logit_no_theta, design = _pad_design(slices_by_replicate[b], bank, design_fn, P, time_filter, device)
        ys.append(y)
        masks.append(mask)
        logits.append(logit_no_theta)
        designs.append(design)
    T_max = max(y.shape[1] for y in ys)
    if any(y.shape[1] != T_max for y in ys):
        # Defensive only: T is permutation-invariant in this design, so
        # every replicate's T_max should already match.
        ys = [torch.nn.functional.pad(y, (0, T_max - y.shape[1])) for y in ys]
        masks = [torch.nn.functional.pad(m, (0, T_max - m.shape[1])) for m in masks]
        logits = [torch.nn.functional.pad(l, (0, T_max - l.shape[1])) for l in logits]
        designs = [torch.nn.functional.pad(d, (0, 0, 0, T_max - d.shape[1])) for d in designs]

    y_all = torch.stack(ys, dim=0).reshape(B * S, T_max)
    mask_all = torch.stack(masks, dim=0).reshape(B * S, T_max)
    logit_all = torch.stack(logits, dim=0).reshape(B * S, T_max)
    design_all = torch.stack(designs, dim=0).reshape(B * S, T_max, P)
    del ys, masks, logits, designs
    x0 = torch.zeros(B * S, P, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll, x0, data_args=(y_all, mask_all, logit_all, design_all), prior_var=prior_var
    )
    # `params` is a reshaped VIEW of `result.params` (same storage), so it
    # survives `del result` below; every OTHER field of `result`
    # (converged/n_iter/data_nll/penalized_nll) and the input tensors are
    # explicitly freed here rather than left to eventual scope-exit GC --
    # this is the per-chunk-cleanup half of the OOM fix (see
    # `permutation_null_batched`'s incident-context docstring note).
    params = result.params.reshape(B, S, P)
    del y_all, mask_all, logit_all, design_all, x0, result
    return params


def fit_m0(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m0_design, P=1, time_filter=time_filter, device=device)


def fit_m1a_slice(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m1a_slice_design, P=2, time_filter=time_filter, device=device)


def fit_m1b_slice(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m1b_slice_design, P=N_BLOCKS, time_filter=time_filter, device=device)


# ---------------------------------------------------------------------------
# KC-pooled joint fits (M1a-pooled, M1b-pooled): one Newton call per KC
# ---------------------------------------------------------------------------


def _kc_joint_parts(
    kc_slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared per-slice assembly for one KC's joint fit: concatenated flat
    ``(L,)``/``(L, shared_dim)`` numpy arrays (``L`` the number of valid,
    time-filtered positions across all ``k = len(kc_slices)`` slices) --
    pure data assembly, no fitting. Factored out of `_build_kc_joint_arrays`
    (the dense one-hot-bordered design, kept as the equivalence reference)
    and `_build_kc_joint_arrow_arrays` (the compact arrow representation,
    the production default -- 2026-07-20 4th A4 perf-surgery,
    `_planning/LEDGER.md`) so the SAME per-slice loop feeds both
    representations of the identical model: they can never independently
    drift apart. Returns ``(slice_idx (L,) int64, y (L,), mask (L,) bool,
    logit_no_theta (L,), shared_cols (L, shared_dim))``.
    """
    k = len(kc_slices)
    slice_idx_parts, y_parts, mask_parts, logit_parts, shared_parts = [], [], [], [], []
    for local_idx, sl in enumerate(kc_slices):
        t = sl.T
        if t == 0:
            continue
        valid = _validity_mask(sl, bank)
        if time_filter is not None:
            valid = valid & time_filter(sl.opportunity)
        slice_idx_parts.append(np.full(t, local_idx, dtype=np.int64))
        y_parts.append(sl.response.astype(np.float32))
        mask_parts.append(valid)
        b = bank.difficulty(sl.item_id) if bank is not None else np.zeros(t)
        logit_parts.append(-b)
        shared_parts.append(shared_design_fn(sl.opportunity, sl.block_id))
    if not slice_idx_parts:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, shared_dim), dtype=np.float32),
        )
    return (
        np.concatenate(slice_idx_parts),
        np.concatenate(y_parts),
        np.concatenate(mask_parts),
        np.concatenate(logit_parts),
        np.concatenate(shared_parts, axis=0).astype(np.float32),
    )


def _build_kc_joint_arrays(
    kc_slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assembles one KC's joint (per-slice one-hot theta + shared block)
    design as flat ``(L,)``/``(L, P)`` numpy arrays (``P = k +
    shared_dim``, ``k = len(kc_slices)``). A thin one-hot materialization
    over `_kc_joint_parts`'s shared per-slice assembly, kept as the DENSE
    equivalence reference for `fit_kc_joint`/`fit_kc_joint_batched_replicates`'s
    ``use_arrow=False`` path (2026-07-20 4th A4 perf-surgery): production
    fits no longer build this by default (see `_build_kc_joint_arrow_arrays`,
    which skips ever materializing the ``(L, k)`` one-hot block -- real
    memory/compute cost at oversized KCs, ``k`` up to ~3000), but any
    caller can still request it directly.
    """
    k = len(kc_slices)
    slice_idx, y, mask, logit_no_theta, shared_cols = _kc_joint_parts(
        kc_slices, bank, shared_design_fn, shared_dim, time_filter
    )
    L = len(y)
    if L == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, k + shared_dim), dtype=np.float32),
        )
    onehot = np.zeros((L, k), dtype=np.float32)
    onehot[np.arange(L), slice_idx] = 1.0
    design = np.concatenate([onehot, shared_cols], axis=1)
    return y, mask, logit_no_theta, design


def _build_kc_joint_arrow_arrays(
    kc_slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compact arrow-structured representation of one KC's joint fit
    (2026-07-20 4th A4 perf-surgery, `_planning/LEDGER.md`): instead of
    materializing `_build_kc_joint_arrays`'s dense ``(L, k)`` one-hot
    block, returns the per-row SLICE INDEX directly (``slice_idx``, ``(L,)``
    int64) alongside the shared design columns (``shared_cols``, ``(L,
    shared_dim)``) -- mathematically identical to the dense design (same
    rows, same values: ``design[:, :k] == one_hot(slice_idx, k)``,
    ``design[:, k:] == shared_cols``), consumed by `newton.py`'s
    `binary_logit_nll_arrow` fast path, which exploits the resulting
    Hessian's exact block-diagonal-bordered (arrow) structure via Schur-
    complement elimination instead of a dense ``O(k^3)`` solve. Returns
    ``(y, mask, logit_no_theta, slice_idx, shared_cols)``.
    """
    slice_idx, y, mask, logit_no_theta, shared_cols = _kc_joint_parts(
        kc_slices, bank, shared_design_fn, shared_dim, time_filter
    )
    # Reordered to (y, mask, logit_no_theta, slice_idx, shared_cols) --
    # matching `_build_kc_joint_arrays`'s own (y, mask, logit_no_theta,
    # design) return order, with `design` split into its two
    # arrow-structured pieces.
    return y, mask, logit_no_theta, slice_idx, shared_cols


def fit_kc_joint(
    kc_slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    prior_var: float = PRIOR_VAR,
    device: str = "cpu",
    use_arrow: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint MAP fit of one KC's slices: free per-slice intercept theta_ic
    (one column each, one-hot-selected) plus a SHARED ``shared_dim``-length
    parameter block (beta_c for M1a-pooled, the 4-vector u_c for
    M1b-pooled), via one call to the shared Newton primitive (S=1: a
    single joint optimization whose parameter vector concatenates every
    slice's own theta with the shared block; note 2 in the module
    docstring explains why this is a Python loop over KCs rather than a
    KC-batched vmap). Returns ``(theta_ic array (k,), shared array
    (shared_dim,))``.

    ``use_arrow`` (default True, 2026-07-20 4th A4 perf-surgery,
    `_planning/LEDGER.md`) selects `newton.py`'s arrow-structured Schur-
    complement fast path (`_build_kc_joint_arrow_arrays` +
    `binary_logit_nll_arrow`): oversized real KCs (``k`` up to ~3000)
    previously forced a dense ``O(k^3)`` Hessian solve per Newton
    iteration for no reason -- the true cost is ``O(k)``, exploiting the
    provably block-diagonal-bordered Hessian this model always has. This
    is EXACT block elimination, not an approximation, so ``use_arrow=True``
    reproduces ``use_arrow=False``'s dense one-hot-design path
    (`_build_kc_joint_arrays` + `binary_logit_nll`) to float tolerance
    (`tests/test_growth_newton.py`'s exactness suite,
    `tests/test_growth_gate_perf_equivalence.py`'s KC-joint equivalence
    tests); ``use_arrow=False`` is kept as that equivalence reference and
    a manual escape hatch.
    """
    k = len(kc_slices)
    if k == 0:
        return np.zeros(0), np.zeros(shared_dim)
    P = k + shared_dim

    if use_arrow:
        y, mask, logit_no_theta, slice_idx, shared_cols = _build_kc_joint_arrow_arrays(
            kc_slices, bank, shared_design_fn, shared_dim, time_filter
        )
        if len(y) == 0:
            return np.zeros(k), np.zeros(shared_dim)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logit_t = torch.as_tensor(logit_no_theta, dtype=torch.float32, device=device).unsqueeze(0)
        slice_idx_t = torch.as_tensor(slice_idx, dtype=torch.long, device=device).unsqueeze(0)
        shared_cols_t = torch.as_tensor(shared_cols, dtype=torch.float32, device=device).unsqueeze(0)
        x0 = torch.zeros(1, P, device=device)
        result = penalized_bounded_newton(
            binary_logit_nll_arrow, x0,
            data_args=(y_t, mask_t, logit_t, slice_idx_t, shared_cols_t, k),
            prior_var=prior_var,
        )
        params = result.params[0].cpu().numpy()
        return params[:k], params[k:]

    y, mask, logit_no_theta, design = _build_kc_joint_arrays(
        kc_slices, bank, shared_design_fn, shared_dim, time_filter
    )
    if len(y) == 0:
        return np.zeros(k), np.zeros(shared_dim)

    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    logit_t = torch.as_tensor(logit_no_theta, dtype=torch.float32, device=device).unsqueeze(0)
    design_t = torch.as_tensor(design, dtype=torch.float32, device=device).unsqueeze(0)
    x0 = torch.zeros(1, P, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll, x0, data_args=(y_t, mask_t, logit_t, design_t), prior_var=prior_var
    )
    params = result.params[0].cpu().numpy()
    return params[:k], params[k:]


def fit_kc_joint_batched_replicates(
    kc_slices_by_replicate: Sequence[Sequence[Slice]],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    prior_var: float = PRIOR_VAR,
    device: str = "cpu",
    use_arrow: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The A4 perf-surgery fix's core primitive: fits the SAME KC's pooled
    model (M1a-pooled or M1b-pooled) across ``B`` permutation replicates
    in ONE Newton call (batch = B) instead of `fit_kc_joint`'s S=1-per-
    replicate call repeated B times in a Python loop (`newton.py`'s
    module docstring notes the dominant cost is eager `torch.func`
    dispatch overhead PER CALL, not per batch element, so widening the
    batch to include the replicate axis removes ~B-fold redundant
    dispatch for exactly this reason).

    Valid because permutation (`slices.permute_learner_order`) only
    reorders each learner's OWN rows -- every row keeps its own (learner,
    KC) tag and item id, so which (learner, KC) slices exist is
    permutation-invariant. Hence every replicate's ``kc_slices_by_replicate[b]``
    has the SAME slice count ``k`` (just different response/order
    content within each slice), so the design width ``P = k +
    shared_dim`` is identical across replicates -- only the number of
    valid (time-filtered) positions ``L`` can differ slightly per
    replicate (different items land on odd/even opportunity parity each
    time), so replicates are padded to a shared ``L_max`` with a mask,
    exactly `_pad_design`'s existing padding convention elsewhere in this
    module. Returns ``(theta_ic (B, k), shared (B, shared_dim))``.

    ``use_arrow`` (default True, 2026-07-20 4th A4 perf-surgery,
    `_planning/LEDGER.md`): see `fit_kc_joint`'s docstring -- the SAME
    arrow-vs-dense choice, applied per-replicate-chunk here. This is the
    call site the production incident actually traced to (oversized real
    KCs' dense ``O(k^3)`` Hessian solve inside the permutation battery, k
    up to ~3000): the arrow path drops memory from ``O(B * k^2)`` to
    ``O(B * k)`` in addition to the compute fix, which is why
    `_calibrate_kc_chunk_size_empirically`'s empirical probe now measures
    a much smaller footprint and recovers large chunks on the same KCs
    that used to collapse to chunk=1.
    """
    B = len(kc_slices_by_replicate)
    k = len(kc_slices_by_replicate[0]) if B else 0
    if B == 0 or k == 0:
        return np.zeros((B, k)), np.zeros((B, shared_dim))
    P = k + shared_dim

    if use_arrow:
        arrays = [
            _build_kc_joint_arrow_arrays(kc_slices_by_replicate[b], bank, shared_design_fn, shared_dim, time_filter)
            for b in range(B)
        ]
        L_max = max((a[0].shape[0] for a in arrays), default=0)
        if L_max == 0:
            return np.zeros((B, k)), np.zeros((B, shared_dim))

        y = np.zeros((B, L_max), dtype=np.float32)
        mask = np.zeros((B, L_max), dtype=bool)
        logit_no_theta = np.zeros((B, L_max), dtype=np.float32)
        slice_idx = np.zeros((B, L_max), dtype=np.int64)
        shared_cols = np.zeros((B, L_max, shared_dim), dtype=np.float32)
        for b, (y_b, mask_b, logit_b, idx_b, shared_b) in enumerate(arrays):
            L_b = y_b.shape[0]
            if L_b == 0:
                continue
            y[b, :L_b] = y_b
            mask[b, :L_b] = mask_b
            logit_no_theta[b, :L_b] = logit_b
            slice_idx[b, :L_b] = idx_b
            shared_cols[b, :L_b] = shared_b

        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
        logit_t = torch.as_tensor(logit_no_theta, dtype=torch.float32, device=device)
        slice_idx_t = torch.as_tensor(slice_idx, dtype=torch.long, device=device)
        shared_cols_t = torch.as_tensor(shared_cols, dtype=torch.float32, device=device)
        del y, mask, logit_no_theta, slice_idx, shared_cols, arrays
        x0 = torch.zeros(B, P, device=device)
        result = penalized_bounded_newton(
            binary_logit_nll_arrow, x0,
            data_args=(y_t, mask_t, logit_t, slice_idx_t, shared_cols_t, k),
            prior_var=prior_var,
        )
        # `.cpu().numpy()` makes an independent CPU copy, so every GPU
        # tensor involved (inputs and `result`, unlike `fit_batched_replicates`
        # whose returned params must stay on-device) can be freed
        # immediately -- the per-chunk-cleanup half of the OOM fix.
        params = result.params.cpu().numpy()
        del y_t, mask_t, logit_t, slice_idx_t, shared_cols_t, x0, result
        return params[:, :k], params[:, k:]

    # `use_arrow=False`: original dense one-hot-design path (equivalence
    # reference / manual escape hatch).
    arrays = [
        _build_kc_joint_arrays(kc_slices_by_replicate[b], bank, shared_design_fn, shared_dim, time_filter)
        for b in range(B)
    ]
    L_max = max((a[0].shape[0] for a in arrays), default=0)
    if L_max == 0:
        return np.zeros((B, k)), np.zeros((B, shared_dim))

    y = np.zeros((B, L_max), dtype=np.float32)
    mask = np.zeros((B, L_max), dtype=bool)
    logit_no_theta = np.zeros((B, L_max), dtype=np.float32)
    design = np.zeros((B, L_max, P), dtype=np.float32)
    for b, (y_b, mask_b, logit_b, design_b) in enumerate(arrays):
        L_b = y_b.shape[0]
        if L_b == 0:
            continue
        y[b, :L_b] = y_b
        mask[b, :L_b] = mask_b
        logit_no_theta[b, :L_b] = logit_b
        design[b, :L_b] = design_b

    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    logit_t = torch.as_tensor(logit_no_theta, dtype=torch.float32, device=device)
    design_t = torch.as_tensor(design, dtype=torch.float32, device=device)
    del y, mask, logit_no_theta, design, arrays
    x0 = torch.zeros(B, P, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll, x0, data_args=(y_t, mask_t, logit_t, design_t), prior_var=prior_var
    )
    # `.cpu().numpy()` makes an independent CPU copy, so every GPU tensor
    # involved (inputs and `result`, unlike `fit_batched_replicates` whose
    # returned params must stay on-device) can be freed immediately --
    # the per-chunk-cleanup half of the OOM fix.
    params = result.params.cpu().numpy()
    del y_t, mask_t, logit_t, design_t, x0, result
    return params[:, :k], params[:, k:]


def held_out_nll_kc_joint(
    kc_slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_ic: np.ndarray,
    shared: np.ndarray,
    time_filter: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Per-slice held-out NLL contribution under a KC-pooled fit's FIXED
    parameters (used both for the KC/bed-pooled statistic and to report a
    per-slice share of that pooled fit)."""
    out = np.zeros(len(kc_slices))
    for i, sl in enumerate(kc_slices):
        t = sl.T
        if t == 0:
            continue
        valid = _validity_mask(sl, bank) & time_filter(sl.opportunity)
        if not valid.any():
            continue
        b = bank.difficulty(sl.item_id) if bank is not None else np.zeros(t)
        shared_cols = shared_design_fn(sl.opportunity, sl.block_id)
        logits = theta_ic[i] + shared_cols @ shared - b
        y = sl.response.astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        nll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        out[i] = float(nll[valid].sum())
    return out


def held_out_total_nll_kc_joint_batched_replicates(
    kc_slices_by_replicate: Sequence[Sequence[Slice]],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: np.ndarray,
    shared: np.ndarray,
    time_filter: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Per-replicate TOTAL held-out NLL of a KC-pooled fit (summed over
    the KC's own slices), at FIXED per-replicate parameters ``theta`` (B,
    k) / ``shared`` (B, shared_dim) from `fit_kc_joint_batched_replicates`.
    Only the scalar-per-replicate total is needed by the permutation
    null's `kc_stat`/`bed_stat` (never the per-slice breakdown), so this
    reuses `held_out_nll_kc_joint` UNCHANGED, once per replicate, and
    sums: that function is a plain, already-vectorized-per-slice numpy
    evaluation, not a Newton/autodiff call, so looping it B times carries
    none of the eager-`torch.func` dispatch cost this module's batching
    targets, while guaranteeing bit-for-bit the same held-out formula as
    the reference (looped) path.
    """
    B = len(kc_slices_by_replicate)
    out = np.zeros(B)
    for b in range(B):
        per_slice = held_out_nll_kc_joint(
            kc_slices_by_replicate[b], bank, shared_design_fn, theta[b], shared[b], time_filter
        )
        out[b] = float(per_slice.sum())
    return out


# ---------------------------------------------------------------------------
# Gate statistic assembly
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Held-out improvement of M1 over M0 (max of the M1a/M1b family), at
    every pooling level the design names (section 2.2's "per KC, per bed,
    per slice (D2+)")."""

    slice_keys: list[tuple[int, int]]
    slice_stat: np.ndarray  # (n_d2plus_slices,) M1-vs-M0 held-out improvement, per slice, D2+ only
    kc_stat: np.ndarray  # (n_kcs,)
    bed_stat: float
    kc_selected_family: np.ndarray  # (n_kcs,) "m1a" | "m1b" | "none" (no slices)


def compute_gate_result(
    slices: dict[tuple[int, int], Slice],
    bank: Optional[FrozenBank],
    n_kcs: int,
    device: str = "cpu",
    bed_kc_mask: Optional[np.ndarray] = None,
) -> GateResult:
    """Fit M0 once (batched over every slice in the bed) and both M1
    families at both pooling levels, on the interpolative odd/even split,
    and assemble the max-selected gate statistic at every pooling level
    (module docstring note 1).

    ``bed_kc_mask`` (module docstring note 6, the saturation-aware null
    fix): an optional ``(n_kcs,)`` boolean array selecting which KCs
    CONTRIBUTE to the bed-level pooled statistic. ``None`` (default) sums
    every KC, the frozen behavior. A KC with ``bed_kc_mask[c] == False``
    (e.g. a saturation-flagged near-ceiling KC) is dropped from
    ``bed_stat`` but keeps its own UNCHANGED ``kc_stat[c]`` /
    ``kc_selected_family[c]`` -- the per-KC statistic is never altered,
    only its inclusion in the bed decision. Callers MUST pass the identical
    mask to the permutation null so observed and null bed statistics are
    computed over the same KC set (the mask is a calibration-cohort
    property, permutation-invariant; see note 6)."""
    all_slices = list(slices.values())
    d2_slices = [sl for sl in all_slices if sl.is_d2_plus]

    # --- M0, the shared null at every pooling level ---
    m0_fit = fit_batched(all_slices, bank, _m0_design, P=1, time_filter=_time_filter_odd, device=device)
    m0_nll_even = held_out_nll(all_slices, bank, _m0_design, m0_fit.params, _time_filter_even, device=device)
    m0_by_key = {k: float(v) for k, v in zip(m0_fit.keys, m0_nll_even)}

    # --- Per-slice gate (D2+ only): max(M1a-slice, M1b-slice) ---
    slice_keys = [(sl.learner, sl.kc) for sl in d2_slices]
    if d2_slices:
        m1a_s = fit_batched(d2_slices, bank, _m1a_slice_design, P=2, time_filter=_time_filter_odd, device=device)
        m1a_s_nll = held_out_nll(d2_slices, bank, _m1a_slice_design, m1a_s.params, _time_filter_even, device=device)
        m1b_s = fit_batched(d2_slices, bank, _m1b_slice_design, P=N_BLOCKS, time_filter=_time_filter_odd, device=device)
        m1b_s_nll = held_out_nll(d2_slices, bank, _m1b_slice_design, m1b_s.params, _time_filter_even, device=device)
        m0_d2_nll = np.array([m0_by_key[k] for k in slice_keys])
        stat_a = m0_d2_nll - m1a_s_nll.cpu().numpy()
        stat_b = m0_d2_nll - m1b_s_nll.cpu().numpy()
        slice_stat = np.maximum(stat_a, stat_b)
    else:
        slice_stat = np.zeros(0)

    # --- KC-pooled and bed-pooled gate: max(M1a-pooled, M1b-pooled) ---
    by_kc = slices_by_kc(slices, n_kcs)
    kc_stat = np.zeros(n_kcs)
    kc_family = np.full(n_kcs, "none", dtype=object)
    total_m1a = 0.0
    total_m1b = 0.0
    total_m0 = float(sum(m0_by_key.values()))
    for c in range(n_kcs):
        kc_slices = by_kc[c]
        if not kc_slices:
            continue
        theta_a, beta_c = fit_kc_joint(kc_slices, bank, _m1a_shared_design, 1, time_filter=_time_filter_odd, device=device)
        nll_a = held_out_nll_kc_joint(kc_slices, bank, _m1a_shared_design, theta_a, beta_c, _time_filter_even)
        theta_b, u_c = fit_kc_joint(kc_slices, bank, _m1b_shared_design, N_BLOCKS, time_filter=_time_filter_odd, device=device)
        nll_b = held_out_nll_kc_joint(kc_slices, bank, _m1b_shared_design, theta_b, u_c, _time_filter_even)
        m0_kc_nll = np.array([m0_by_key[(sl.learner, sl.kc)] for sl in kc_slices])
        stat_kc_a = float((m0_kc_nll - nll_a).sum())
        stat_kc_b = float((m0_kc_nll - nll_b).sum())
        if bed_kc_mask is None or bed_kc_mask[c]:
            total_m1a += stat_kc_a
            total_m1b += stat_kc_b
        if stat_kc_a >= stat_kc_b:
            kc_stat[c] = stat_kc_a
            kc_family[c] = "m1a"
        else:
            kc_stat[c] = stat_kc_b
            kc_family[c] = "m1b"

    bed_stat = max(total_m0 - (total_m0 - total_m1a), total_m0 - (total_m0 - total_m1b))
    # (equivalently bed_stat = max(total_m1a, total_m1b); written out for clarity)
    bed_stat = max(total_m1a, total_m1b)

    return GateResult(
        slice_keys=slice_keys,
        slice_stat=slice_stat,
        kc_stat=kc_stat,
        bed_stat=bed_stat,
        kc_selected_family=kc_family,
    )


# ---------------------------------------------------------------------------
# Permutation-null hook (battery arm 1)
# ---------------------------------------------------------------------------


def gate_statistic_on_learners(
    learners: Sequence, n_learners: int, n_kcs: int, bank: Optional[FrozenBank], device: str = "cpu",
    bed_kc_mask: Optional[np.ndarray] = None,
) -> GateResult:
    """Rebuild rows and slices from a (possibly permuted) learner log list
    and compute the gate result -- the one operation the permutation null
    repeats B times. ``bed_kc_mask`` is forwarded to `compute_gate_result`
    (the saturation-aware bed-statistic restriction, module docstring note
    6)."""
    rows = build_calibration_rows(learners)
    slices = build_slices(rows)
    return compute_gate_result(slices, bank, n_kcs, device=device, bed_kc_mask=bed_kc_mask)


def permutation_null_looped(
    learners: Sequence,
    n_learners: int,
    n_kcs: int,
    bank: Optional[FrozenBank],
    n_replicates: int,
    seed: int,
    device: str = "cpu",
    bed_kc_mask: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Battery arm 1's hook, REFERENCE implementation: ``n_replicates``
    permutation draws of the bed-level and per-KC gate statistic (design:
    "permute each learner's full interaction order, then rebuild slices
    and opportunity indices"), one full `compute_gate_result`-equivalent
    Newton-fit pass PER REPLICATE, in a Python loop. This is the
    numerically-authoritative path (every Newton call is a fresh S=1 or
    S=n_slices call, exactly as `compute_gate_result` itself does) kept
    as the A4 perf-surgery equivalence-gate reference and as an explicit
    fallback (`permutation_null(..., use_batched=False)`); `newton.py`'s
    module docstring records why this loop's per-replicate eager
    `torch.func` dispatch is the dominant cost `permutation_null_batched`
    removes. ``bed_kc_mask`` restricts the bed statistic to the same KC
    subset as the observed read (module docstring note 6).
    """
    rng = np.random.default_rng(seed)
    bed_null = np.zeros(n_replicates)
    kc_null = np.zeros((n_replicates, n_kcs))
    for r in range(n_replicates):
        perm_learners = permute_learner_order(learners, rng)
        result = gate_statistic_on_learners(
            perm_learners, n_learners, n_kcs, bank, device=device, bed_kc_mask=bed_kc_mask
        )
        bed_null[r] = result.bed_stat
        kc_null[r] = result.kc_stat
    return {"bed": bed_null, "kc": kc_null}


# ---------------------------------------------------------------------------
# Memory budget precedence (2026-07-22 real-data permutation-null OOM fix):
# a chunk-size heuristic that reads the whole NODE's total/free RAM sizes
# chunks toward that number even when the running job is capped far below
# it by a SLURM `--mem` cgroup (a shared 250G node running a 96G-capped job
# gets OOM-killed the moment usage crosses 96G, long before "free node RAM"
# would suggest trouble). The functions below give every chunk-size
# calibration in this module ONE authoritative memory budget, in strict
# precedence order:
#   1. `KT_MIRT_MEM_BUDGET_GB` env var -- an operator's explicit,
#      authoritative override (e.g. set to a job's `--mem` cap, minus
#      headroom for the run's fixed baseline outside the chunked-fit math:
#      data loading, the frozen bank, OS/page cache).
#   2. This process's cgroup memory limit -- v2's `memory.max`, then v1's
#      `memory.limit_in_bytes` -- i.e. the job's REAL allocation, as
#      opposed to the physical node's full RAM.
#   3. Unavailable: neither an override nor a readable cgroup limit exists
#      (e.g. a bare, non-containerized workstation). Callers fall back to
#      their OWN pre-existing, budget-agnostic sizing in this case --
#      deliberately NOT to a total/free node-RAM read, since that read is
#      exactly the bug this precedence exists to close.
# Every use of this budget (CPU or CUDA) logs one line naming the budget,
# its source, and the resulting chunk size, so a bad size is never silently
# wrong again.
# ---------------------------------------------------------------------------

_MEM_BUDGET_ENV_VAR = "KT_MIRT_MEM_BUDGET_GB"
_CGROUP_V2_MEM_MAX_PATH = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_MEM_LIMIT_PATH = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
# A v1 "no limit" cgroup reads back as a huge near-INT64_MAX byte count
# (implementation detail of the kernel's default), not a real allocation;
# treated the same as v2's literal "max" sentinel string.
_CGROUP_UNLIMITED_SENTINEL_BYTES = 10 * (1024 ** 4)  # 10 TB
# No cross-platform RSS-based per-replicate probe exists for the CPU path
# (unlike CUDA's real `torch.cuda.max_memory_allocated` measurement below);
# this is a deliberately conservative, UNMEASURED placeholder used ONLY to
# translate a KNOWN budget (tiers 1-2 above) into a chunk count, and only
# ever applied as a ceiling via `min()` against the pre-existing
# budget-agnostic default (`_cpu_chunk_from_budget`) -- so an imprecise
# assumption here can only shrink a chunk (safer), never grow one past
# what today's behavior would already have picked.
_CPU_FALLBACK_BYTES_PER_REPLICATE = 512 * 1024 * 1024  # 512 MiB


def _env_memory_budget_bytes() -> Optional[int]:
    """Tier 1: `KT_MIRT_MEM_BUDGET_GB`, parsed as float GiB (matching
    SLURM's own `--mem=` convention). None if unset, blank, or unparseable
    as a positive number."""
    raw = os.environ.get(_MEM_BUDGET_ENV_VAR)
    if not raw:
        return None
    try:
        gb = float(raw)
    except ValueError:
        return None
    if not (gb > 0):
        return None
    return int(gb * (1024 ** 3))


def _cgroup_memory_budget_bytes() -> Optional[int]:
    """Tier 2: this process's cgroup memory cap, v2 first then v1 (see the
    module-level precedence note above). None if neither file is readable
    or parseable, or the value found is an "unlimited" sentinel."""
    for path in (_CGROUP_V2_MEM_MAX_PATH, _CGROUP_V1_MEM_LIMIT_PATH):
        try:
            with open(path, "r") as f:
                raw = f.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _CGROUP_UNLIMITED_SENTINEL_BYTES:
            continue
        return value
    return None


def _host_memory_budget_bytes() -> tuple[Optional[int], str]:
    """The authoritative memory budget for THIS job, in bytes, with the
    source that produced it -- `_env_memory_budget_bytes` (tier 1), else
    `_cgroup_memory_budget_bytes` (tier 2), else `(None, "unavailable")`.
    See the module-level precedence note above for why tier 3 is a bare
    "unavailable" rather than a total/free node-RAM read."""
    env_bytes = _env_memory_budget_bytes()
    if env_bytes is not None:
        return env_bytes, "env"
    cgroup_bytes = _cgroup_memory_budget_bytes()
    if cgroup_bytes is not None:
        return cgroup_bytes, "cgroup"
    return None, "unavailable"


def _cpu_chunk_from_budget(default_chunk: int, max_chunk: int, safety_factor: float) -> int:
    """Caps `default_chunk` (the pre-existing, budget-agnostic CPU
    heuristic) by this job's real memory budget (`_host_memory_budget_bytes`),
    translated to a replicate count via `_CPU_FALLBACK_BYTES_PER_REPLICATE`.
    With no budget signal (tier 3), returns `min(default_chunk, max_chunk)`
    unchanged -- exactly today's behavior. Always logs one line naming the
    resolved budget/source and the resulting chunk."""
    budget_bytes, source = _host_memory_budget_bytes()
    if budget_bytes is None:
        chunk = max(1, min(default_chunk, max_chunk))
        print(
            f"[permutation_null_batched] memory budget: unavailable (no {_MEM_BUDGET_ENV_VAR}, "
            f"no readable cgroup limit) -- CPU chunk={chunk} (default, budget-agnostic)",
            flush=True,
        )
        return chunk
    budget_chunk = max(1, int(budget_bytes / (safety_factor * _CPU_FALLBACK_BYTES_PER_REPLICATE)))
    chunk = max(1, min(default_chunk, budget_chunk, max_chunk))
    print(
        f"[permutation_null_batched] memory budget: {budget_bytes / (1024 ** 3):.2f} GiB "
        f"(source={source}) safety_factor={safety_factor} -> CPU chunk={chunk}",
        flush=True,
    )
    return chunk


def _budget_capped_free_bytes(free_bytes: int) -> int:
    """Caps a device's own reported free bytes (e.g.
    `torch.cuda.mem_get_info`) by this job's real memory budget
    (`_host_memory_budget_bytes`), applying the SAME env-var/cgroup
    precedence used for the CPU path on top of -- never instead of -- the
    device's own accurate reading (tier 3 there IS this reading, per the
    module-level precedence note above). Logs only when the budget
    actually binds (is smaller than what the device itself reports), so
    the ordinary case (no override, or a generous one) stays silent."""
    budget_bytes, source = _host_memory_budget_bytes()
    if budget_bytes is None or budget_bytes >= free_bytes:
        return free_bytes
    print(
        f"[permutation_null_batched] memory budget: {budget_bytes / (1024 ** 3):.2f} GiB "
        f"(source={source}) caps device free_bytes from {free_bytes} to {budget_bytes}",
        flush=True,
    )
    return budget_bytes


def _calibrate_m0_chunk_size_empirically(
    perm_learners_list: Sequence[Sequence],
    all_keys: Sequence[tuple[int, int]],
    bank: Optional[FrozenBank],
    device: str,
    safety_factor: float = 2.0,
    probe_replicates: int = 2,
    max_chunk: int = 200,
) -> int:
    """Empirically measures the M0 stage's ACTUAL per-replicate GPU memory
    footprint (batch = chunk * n_slices_bed, P=1) by running a small real
    probe, instead of an analytic guess. M0's cost does not depend on any
    KC's own size (P=1 always), so ONE bed-wide chunk size is appropriate
    here -- unlike the KC-pooled fits (`_calibrate_kc_chunk_size_empirically`),
    which must be calibrated PER KC (see that function's docstring for why
    a single shared chunk size across every KC was the root cause of a
    production non-completion: a chunk sized safe for the bed's largest
    KC starves every smaller KC of any batching benefit at all).

    On CPU (or no CUDA), no per-replicate memory probe is attempted (no
    cross-platform RSS-based mechanism is implemented); the chunk is a
    small, fixed, conservative default, further capped by this job's real
    memory budget when one is known (`_cpu_chunk_from_budget` -- see the
    module-level precedence note above). On CUDA, the empirically-measured
    free bytes are themselves capped by the same budget precedence
    (`_budget_capped_free_bytes`) before being divided by the measured
    per-replicate footprint.
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    n_replicates = len(perm_learners_list)
    if not is_cuda:
        return _cpu_chunk_from_budget(max(1, min(20, n_replicates)), max_chunk, safety_factor)

    probe_n = min(probe_replicates, n_replicates)
    if probe_n == 0 or not all_keys:
        return 1

    probe_learners = perm_learners_list[:probe_n]
    probe_slices = [build_slices(build_calibration_rows(lg)) for lg in probe_learners]

    m0_peak: Optional[int] = None
    try:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        slices_by_replicate_all = [[sd[k] for k in all_keys] for sd in probe_slices]
        m0_params = fit_batched_replicates(
            slices_by_replicate_all, bank, _m0_design, P=1, time_filter=_time_filter_odd, device=device
        )
        del m0_params, slices_by_replicate_all
        torch.cuda.synchronize(device)
        m0_peak = torch.cuda.max_memory_allocated(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    if m0_peak is None:
        return 1

    bytes_per_replicate = max(m0_peak, 1) / probe_n
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_bytes = _budget_capped_free_bytes(free_bytes)
    safe_chunk = max(1, int(free_bytes / (safety_factor * bytes_per_replicate)))
    return int(min(safe_chunk, max_chunk))


def _calibrate_kc_chunk_size_empirically(
    kc_slices_by_replicate_probe: Sequence[Sequence[Slice]],
    bank: Optional[FrozenBank],
    device: str,
    safety_factor: float = 2.0,
    max_chunk: int = 200,
) -> int:
    """Empirically measures ONE KC's ACTUAL per-replicate GPU memory
    footprint (the KC-pooled M1b-shape Hessian, ``P = k + N_BLOCKS`` for
    THIS KC's own ``k``) and returns a chunk size safe for THIS KC alone.

    INCIDENT CONTEXT (the actual root cause, found only after a live
    verification run still failed to complete within 2h despite the OOM
    fixes holding): a real bed's KC sizes are severely skewed -- e.g. one
    KDD_MATCHED bed had KCs ranging from a median of ~103 slices up to
    several KCs with ~1700-3000 slices (nearly every learner tagged to
    them). A single GLOBAL chunk size, calibrated against the bed's
    single largest KC, collapses to 1 for safety on that KC -- but that
    same chunk=1 then applies to EVERY other KC too, including the
    hundreds of small/medium ones that could safely batch at 100-200.
    At chunk=1 the "batched" call degenerates to batch-size-1, i.e. NO
    dispatch-count reduction at all: the fix avoided the OOM but
    silently gave back the entire speedup for the whole unit, which is
    why a live run still failed to complete (SLURM TIMEOUT, not OOM).
    Calibrating PER KC (this function, called once per KC -- or per KC
    size-bucket by the caller, to bound overhead) lets ~500 normally-
    sized KCs batch at full speed while only the handful of genuinely
    oversized KCs fall back to a small chunk (or, via
    `_fit_kc_joint_batched_replicates_safe`, to per-replicate looping).

    ``kc_slices_by_replicate_probe`` should be 1-2 real replicates' worth
    of THIS KC's own slices (reusing already-drawn permutations, so this
    does not perturb determinism or consume extra RNG draws). Wrapped in
    an OOM catch: even a 1-2-replicate probe of a sufficiently wide KC can
    itself attempt a huge allocation (confirmed: 287+ GiB on the worst KC
    seen in production) -- caught here and treated as "calibration
    couldn't measure this," never a fatal error, since
    `_fit_kc_joint_batched_replicates_safe`'s OOM fallback is the actual
    safety net regardless of what chunk size calibration returns.

    Like `_calibrate_m0_chunk_size_empirically`, the CPU fallback and the
    CUDA free-bytes reading are both capped by this job's real memory
    budget when one is known (see the module-level precedence note above
    `_calibrate_m0_chunk_size_empirically`).
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    probe_n = len(kc_slices_by_replicate_probe)
    if probe_n == 0:
        return 1
    if not is_cuda:
        return _cpu_chunk_from_budget(max(1, probe_n), max_chunk, safety_factor)

    kc_peak: Optional[int] = None
    try:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        theta_b, u_c = fit_kc_joint_batched_replicates(
            kc_slices_by_replicate_probe, bank, _m1b_shared_design, N_BLOCKS, time_filter=_time_filter_odd, device=device
        )
        del theta_b, u_c
        torch.cuda.synchronize(device)
        kc_peak = torch.cuda.max_memory_allocated(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    if kc_peak is None:
        return 1

    bytes_per_replicate = max(kc_peak, 1) / probe_n
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_bytes = _budget_capped_free_bytes(free_bytes)
    safe_chunk = max(1, int(free_bytes / (safety_factor * bytes_per_replicate)))
    return int(min(safe_chunk, max_chunk))


def _assert_no_memory_growth_across_chunks(
    post_chunk_allocated_bytes: Sequence[int],
    growth_factor: float = 1.5,
    min_absolute_bytes: int = 50_000_000,
) -> None:
    """Guards against the OTHER production failure signature (allocations
    climbing chunk-to-chunk until a later, modest allocation fails --
    retained autograd graphs/intermediates/result tensors, not freed
    between chunks). ``post_chunk_allocated_bytes`` is
    ``torch.cuda.memory_allocated()`` sampled right after each chunk's own
    cleanup (``del`` + ``empty_cache()``); if cleanup is working, this
    should stay roughly flat across chunks (varying only with each
    chunk's own data, not accumulating). Raises if the LAST reading
    exceeds the FIRST by more than ``growth_factor`` (with an absolute
    floor so a near-zero baseline doesn't trigger on ordinary noise).
    """
    if len(post_chunk_allocated_bytes) < 2:
        return
    first = post_chunk_allocated_bytes[0]
    last = post_chunk_allocated_bytes[-1]
    bound = max(first * growth_factor, min_absolute_bytes)
    if last > bound:
        raise RuntimeError(
            "permutation_null_batched: GPU memory allocated grew from "
            f"{first} to {last} bytes across chunks (bound {bound:.0f}) -- "
            "this is the chunk-to-chunk accumulation signature (retained "
            "autograd graphs/intermediates/result tensors); per-chunk "
            "cleanup (del + torch.cuda.empty_cache()) is not releasing "
            "memory as expected."
        )


def _fit_batched_replicates_safe(
    slices_by_replicate: Sequence[Sequence[Slice]],
    bank: Optional[FrozenBank],
    design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    P: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
    device: str,
) -> torch.Tensor:
    """OOM-safe wrapper around `fit_batched_replicates`: on a CUDA
    out-of-memory error, falls back to `fit_batched`'s ORIGINAL per-
    replicate S=1 loop (proven safe -- this is what the pre-batching
    implementation always did), so a bed whose M0 batch is too large for
    the calibrated chunk size cannot crash the whole unit. Calibration
    (`_calibrate_m0_chunk_size_empirically`) picks a good STARTING chunk
    size but is explicitly NOT relied on for correctness/safety -- this
    fallback is the actual guarantee.
    """
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return fit_batched_replicates(slices_by_replicate, bank, design_fn, P, time_filter=time_filter, device=device)
    try:
        return fit_batched_replicates(slices_by_replicate, bank, design_fn, P, time_filter=time_filter, device=device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        B = len(slices_by_replicate)
        S = len(slices_by_replicate[0]) if B else 0
        out = torch.zeros(B, S, P, device=device)
        for b in range(B):
            fit = fit_batched(slices_by_replicate[b], bank, design_fn, P, time_filter=time_filter, device=device)
            out[b] = fit.params
        return out


def _fit_kc_joint_batched_replicates_safe(
    kc_slices_by_replicate: Sequence[Sequence[Slice]],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """OOM-safe wrapper around `fit_kc_joint_batched_replicates`: the
    PRIMARY fix for the production incident's root cause -- a single KC
    whose design width ``P = k + shared_dim`` is large enough that even a
    tiny (2-replicate) batched Hessian attempt tries to allocate 100s of
    GiB, REGARDLESS of chunk size (confirmed: the calibration probe
    itself OOM'd on this exact call, at ``B=2``). No amount of smarter
    chunk-size arithmetic fixes this -- a pathologically wide KC is
    simply not safe to batch over the replicate axis at all. On a CUDA
    OOM, falls back to `fit_kc_joint`'s ORIGINAL per-replicate S=1 loop
    (proven safe: it is what the pre-batching implementation always did,
    and what the production units that eventually succeeded, pre-batching,
    ran for hours), so ONE pathological KC degrades to slow-but-correct
    for itself alone -- every other (normally-sized) KC in the same bed
    still gets the batched speedup.
    """
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return fit_kc_joint_batched_replicates(
            kc_slices_by_replicate, bank, shared_design_fn, shared_dim, time_filter=time_filter, device=device
        )
    try:
        return fit_kc_joint_batched_replicates(
            kc_slices_by_replicate, bank, shared_design_fn, shared_dim, time_filter=time_filter, device=device
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        B = len(kc_slices_by_replicate)
        k = len(kc_slices_by_replicate[0]) if B else 0
        theta = np.zeros((B, k))
        shared = np.zeros((B, shared_dim))
        for b in range(B):
            th, sh = fit_kc_joint(
                kc_slices_by_replicate[b], bank, shared_design_fn, shared_dim, time_filter=time_filter, device=device
            )
            theta[b] = th
            shared[b] = sh
        return theta, shared


def _compute_per_kc_chunk_sizes(
    perm_learners_list: Sequence[Sequence],
    by_kc_keys: Sequence[Sequence[tuple[int, int]]],
    bank: Optional[FrozenBank],
    device: str,
    safety_factor: float = 2.0,
    probe_replicates: int = 2,
    max_chunk: int = 200,
    small_kc_threshold: int = 300,
) -> dict[int, int]:
    """Chunk size PER KC (see `_calibrate_kc_chunk_size_empirically`'s
    incident-context docstring: a single bed-wide chunk size is wrong --
    it collapses to 1 for the whole bed the moment ONE KC is large,
    starving every smaller KC of any batching benefit). KCs with at most
    ``small_kc_threshold`` slices are assigned ``max_chunk`` directly,
    with no probe: their KC-joint design width ``P = k + N_BLOCKS`` is
    trivially small (e.g. k=300 -> P=304, a Hessian of ~370KB) regardless
    of batch size, so probing them would only add overhead without
    changing the answer. Probing (the real, potentially-costly empirical
    measurement) is paid ONLY for the handful of KCs actually large
    enough to matter -- on the KDD_MATCHED bed that motivated this fix,
    that was single digits out of 515 KCs (median slice count ~103, but a
    few KCs had ~1700-3000).

    On CPU, every KC shares one budget-capped default chunk (resolved
    ONCE, not per KC, so a bed with hundreds of KCs logs its memory
    budget one time, not hundreds -- see `_cpu_chunk_from_budget` and the
    module-level precedence note above `_calibrate_m0_chunk_size_empirically`).
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    n_replicates = len(perm_learners_list)
    probe_n = min(probe_replicates, n_replicates)
    probe_learners = perm_learners_list[:probe_n] if probe_n else []
    probe_slices = [build_slices(build_calibration_rows(lg)) for lg in probe_learners] if probe_n else []
    cpu_chunk = (
        None if is_cuda else _cpu_chunk_from_budget(max(1, min(20, n_replicates)), max_chunk, safety_factor)
    )

    sizes: dict[int, int] = {}
    for c, keys_c in enumerate(by_kc_keys):
        if not keys_c:
            continue
        if not is_cuda:
            sizes[c] = cpu_chunk
            continue
        if len(keys_c) <= small_kc_threshold:
            sizes[c] = max_chunk
            continue
        kc_probe = [[sd[key] for key in keys_c] for sd in probe_slices]
        sizes[c] = _calibrate_kc_chunk_size_empirically(
            kc_probe, bank, device, safety_factor=safety_factor, max_chunk=max_chunk,
        )
        # Diagnostic (2026-07-20 3rd A4 perf-surgery incident, cost model
        # updated after the 4th surgery's arrow fix): a collapsed chunk
        # size on a wide KC used to be the LEADING indicator of the
        # dense-Hessian slow path this module's docstrings document (O(L *
        # P^2) per Newton iteration, dominating wall time). Since
        # `fit_kc_joint_batched_replicates` now defaults to the arrow-
        # structured Schur-complement solve (module docstring note 5),
        # whose memory is O(k) instead of O(k^2), a genuine collapse here
        # should be rare even on the largest real KCs -- if this still
        # fires, it means EITHER `use_arrow=False` is in effect somewhere
        # upstream, OR the KC is wide enough that even the arrow path's
        # O(B * k) footprint is large at this chunk's B. Printed once per
        # probed KC (never per replicate/chunk) so this is never silent
        # again without being spammy.
        if sizes[c] <= 8:
            print(
                f"[permutation_null_batched] wide-KC chunk collapse: kc={c} "
                f"n_slices={len(keys_c)} calibrated_chunk={sizes[c]} "
                f"(post-arrow-fix, this chunk's own probe still measured a "
                f"small safe batch -- see newton.py's arrow fast path and "
                f"this function's docstring)",
                flush=True,
            )
    return sizes


# ---------------------------------------------------------------------------
# Vectorized replicate assembly (2026-07-20 4th A4 perf-surgery, phase 2):
# node-level profiling of a running production proof unit found the
# permutation battery GPU-idle and single-CPU-core-bound the ENTIRE time --
# the arrow-structured Newton fix above (phase 1) made the SOLVE fast, but
# every replicate still paid FOUR independent O(n_slices) PYTHON-LEVEL loops
# to rebuild its data from scratch: `slices.build_slices`'s per-group
# `Slice` construction, `_pad_design`'s per-slice torch-tensor conversion
# (used both for fitting AND, redundantly, again inside `held_out_nll`),
# and `_build_kc_joint_arrow_arrays`'s per-slice loop (via `_kc_joint_parts`,
# also paid twice per KC for M1a-pooled and M1b-pooled). Direct local
# profiling at KDD_MATCHED production scale (515 KCs, 3000 learners,
# 119,648 slices) measured, PER REPLICATE: build_slices 0.69s,
# _build_kc_joint_arrow_arrays (all KCs, M1a+M1b) 1.55s, held_out_nll_kc_joint
# (all KCs, M1a+M1b) 3.34s, _pad_design (bed-wide M0) 3.82s, held_out_nll
# (bed-wide M0, which itself re-runs _pad_design) 3.94s -- roughly 13-14s of
# PURE PYTHON LOOP overhead per replicate, which at B=999 alone accounts for
# 3-4 HOURS of the incident's wall time, dwarfing anything the Newton solve
# itself could cost regardless of how it is computed.
#
# The fix below is a from-scratch vectorized assembly path used ONLY by
# `permutation_null_batched` (this module's B=199/999 hot loop); every OTHER
# caller (`compute_gate_result`, `permutation_null_looped`, `rate.py`, the
# unit tests) keeps using `build_slices`/`_pad_design`/`_build_kc_joint_arrow_arrays`
# completely UNCHANGED -- both because those call sites run far too rarely
# for this cost to matter, and because they remain the EQUIVALENCE
# REFERENCE this fast path is tested against
# (`tests/test_growth_gate_perf_equivalence.py`).
#
# Key facts that make the vectorization exact (not approximate):
# 1. Slice membership (which (learner, kc) pairs exist, and each one's
#    length T) is PERMUTATION-INVARIANT (`slices.permute_learner_order`
#    only reorders each learner's OWN rows; `bank.CalibrationRows.learner_idx`
#    is the learner's POSITIONAL index in the learners list, itself
#    unchanged by permutation) -- so the (learner, kc) -> column mapping
#    and each slice's T are computed ONCE per bed, reused for every
#    replicate and chunk.
# 2. `bank.build_calibration_rows`'s ``opportunity`` field is ALREADY the
#    correct 1-based within-slice rank (its own vectorized cumcount, kept
#    completely unchanged here) -- so ``opportunity - 1`` is directly the
#    padded column position for that row within its slice, with no
#    re-derivation of the opportunity/grouping logic at all: the fast path
#    below reuses `build_calibration_rows`'s output verbatim and replaces
#    ONLY `build_slices`'s/`_pad_design`'s/`_build_kc_joint_arrow_arrays`'s
#    final per-slice PYTHON materialization with one vectorized
#    scatter-assign per replicate (`_vectorized_replicate_padded_arrays`).
# 3. Values written to PADDED (non-real) positions are never read: every
#    downstream Newton/held-out computation masks them out before they
#    reach a sum (see `_binary_logit_grad_hess_batch`'s masking, unchanged),
#    so the scatter never needs to explicitly zero padding -- it is simply
#    never touched by the scatter-assign for those positions, and
#    pre-allocated zero-filled arrays leave it that way.
# ---------------------------------------------------------------------------


def _bed_key_scaffold(all_keys: Sequence[tuple[int, int]], n_kcs: int, n_learners: int) -> np.ndarray:
    """Precomputes the ``(learner, kc) -> column-in-all_keys`` lookup as a
    flat array (``key_to_col[learner * n_kcs + kc] = column, or -1``),
    ONCE per bed -- reused for every replicate and chunk by the vectorized
    assembly functions below (see module-level note above: this mapping is
    permutation-invariant). ``learner`` is the POSITIONAL index into the
    learners list (`bank.CalibrationRows.learner_idx`'s own convention,
    which `slices.Slice.learner` inherits unchanged from it).
    """
    key_to_col = np.full(n_learners * n_kcs, -1, dtype=np.int64)
    for col, (learner, kc) in enumerate(all_keys):
        key_to_col[learner * n_kcs + kc] = col
    return key_to_col


def _vectorized_replicate_padded_arrays(
    rows: CalibrationRows, key_to_col: np.ndarray, n_kcs: int, S: int, T_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The core vectorized replacement for ``build_slices(rows)`` followed
    by per-slice consumption (`_pad_design`'s own per-slice loop): scatters
    ONE replicate's `CalibrationRows` directly into padded ``(S, T_max)``
    arrays -- ``(y, valid_pos, opportunity, block_id, item_id)`` -- via a
    single vectorized numpy scatter-assign, with NO per-slice/per-group
    Python-level loop and no `Slice` object construction (module-level
    note above). ``valid_pos`` marks REAL (non-padding) positions only --
    it does not yet include bank-calibration eligibility or the odd/even
    time filter (`_mask_and_logit_from_padded` adds those, separately,
    since they differ between the fitting and held-out passes but the
    underlying data does not).

    Reuses the SAME (row, slot)-flattening and composite ``learner *
    n_kcs + kc`` grouping key as `build_slices`, with `key_to_col`
    (`_bed_key_scaffold`) taking the place of that function's dict-based
    per-group lookup. ``opportunity - 1`` is used directly as each row's
    column position (module-level note 2).
    """
    y = np.zeros((S, T_max), dtype=np.float32)
    valid_pos = np.zeros((S, T_max), dtype=bool)
    opp_padded = np.zeros((S, T_max), dtype=np.int32)
    block_padded = np.zeros((S, T_max), dtype=np.int32)
    item_padded = np.zeros((S, T_max), dtype=np.int32)

    R, A = rows.kc_ids.shape
    if R == 0 or A == 0 or S == 0 or T_max == 0:
        return y, valid_pos, opp_padded, block_padded, item_padded

    row_grid = np.repeat(np.arange(R), A)
    slot_grid = np.tile(np.arange(A), R)
    valid_grid = rows.kc_mask.reshape(-1)
    rows_v = row_grid[valid_grid]
    if len(rows_v) == 0:
        return y, valid_pos, opp_padded, block_padded, item_padded
    slots_v = slot_grid[valid_grid]
    learner_v = rows.learner_idx[rows_v]
    kc_v = rows.kc_ids[rows_v, slots_v]
    opp_v = rows.opportunity[rows_v, slots_v]
    block_v = rows.block_id[rows_v, slots_v]
    item_v = rows.item_id[rows_v]
    resp_v = rows.response[rows_v]

    key_id = learner_v.astype(np.int64) * n_kcs + kc_v.astype(np.int64)
    col_v = key_to_col[key_id]
    keep = col_v >= 0
    col_v = col_v[keep]
    pos_v = (opp_v[keep] - 1).astype(np.int64)
    # Defensive (should not trigger under normal operation: T_max is
    # computed from the SAME slices these rows belong to): drop any
    # position that would overflow the padded row.
    in_range = pos_v < T_max
    col_v = col_v[in_range]
    pos_v = pos_v[in_range]
    item_v = item_v[keep][in_range]
    resp_v = resp_v[keep][in_range]
    block_v = block_v[keep][in_range]
    opp_v = opp_v[keep][in_range]

    y[col_v, pos_v] = resp_v
    valid_pos[col_v, pos_v] = True
    opp_padded[col_v, pos_v] = opp_v
    block_padded[col_v, pos_v] = block_v
    item_padded[col_v, pos_v] = item_v
    return y, valid_pos, opp_padded, block_padded, item_padded


def _stack_chunk_padded_arrays(
    chunk_rows: Sequence[CalibrationRows], key_to_col: np.ndarray, n_kcs: int, S: int, T_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stacks `_vectorized_replicate_padded_arrays` over a CHUNK of ``Bc``
    replicates into ``(Bc, S, T_max)`` arrays. Still one Python-level call
    per replicate, but each call is now a handful of vectorized numpy ops
    instead of an ``O(S)`` Python loop -- the entire fix (module-level
    note above)."""
    Bc = len(chunk_rows)
    y = np.zeros((Bc, S, T_max), dtype=np.float32)
    valid_pos = np.zeros((Bc, S, T_max), dtype=bool)
    opp = np.zeros((Bc, S, T_max), dtype=np.int32)
    block = np.zeros((Bc, S, T_max), dtype=np.int32)
    item = np.zeros((Bc, S, T_max), dtype=np.int32)
    for b, rows in enumerate(chunk_rows):
        y[b], valid_pos[b], opp[b], block[b], item[b] = _vectorized_replicate_padded_arrays(
            rows, key_to_col, n_kcs, S, T_max
        )
    return y, valid_pos, opp, block, item


def _mask_and_logit_from_padded(
    item_padded: np.ndarray,
    valid_pos: np.ndarray,
    opp_padded: np.ndarray,
    bank: Optional[FrozenBank],
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``(mask, logit_no_theta)`` from padded arrays, matching
    `_validity_mask` + `_pad_design`'s semantics exactly: ``mask`` ANDs
    "real (non-pad) position" with the bank's calibration-eligibility check
    and the odd/even ``time_filter``. `FrozenBank.is_calibrated`/`.difficulty`
    and `_time_filter_odd`/`_time_filter_even` are plain fancy-indexing /
    elementwise-modulo operations that already work unchanged on ANY array
    shape, so no flatten/reshape is needed here -- this is the same
    property that makes note 3 above hold generally."""
    mask = valid_pos
    if bank is not None:
        mask = mask & bank.is_calibrated(item_padded)
    if time_filter is not None:
        mask = mask & time_filter(opp_padded)
    if bank is not None:
        logit_no_theta = (-bank.difficulty(item_padded)).astype(np.float32)
    else:
        logit_no_theta = np.zeros(item_padded.shape, dtype=np.float32)
    return mask, logit_no_theta


def _design_from_padded(
    opp_padded: np.ndarray,
    block_padded: np.ndarray,
    design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    P: int,
) -> np.ndarray:
    """Vectorized ``design`` array from padded ``opportunity``/``block_id``:
    flattens to 1D (the shape every ``design_fn`` in this module expects,
    since they use ``len(opp)``), calls the EXACT SAME design_fn used
    everywhere else (zero risk of behavioral drift), and reshapes back.
    Padding positions get whatever value ``design_fn`` produces on their
    zero-filled opp/block entries -- harmless, since ``mask`` (computed
    separately) always zeroes a padded position's contribution before it
    reaches the Newton objective (module-level note 3 above)."""
    shape = opp_padded.shape
    flat = design_fn(opp_padded.reshape(-1), block_padded.reshape(-1))
    return flat.reshape(*shape, P).astype(np.float32)


def _fit_m0_and_held_out_vectorized(
    y: np.ndarray,
    opp: np.ndarray,
    valid_pos: np.ndarray,
    item: np.ndarray,
    bank: Optional[FrozenBank],
    device: str,
) -> np.ndarray:
    """M0 (constant-ability null), fit on the odd split and evaluated
    (held-out NLL) on the even split, batched over the WHOLE ``(Bc, S,
    T_max)`` chunk in one Newton call -- the vectorized-assembly
    replacement for `_fit_batched_replicates_safe` + `held_out_nll`.
    Returns ``(Bc, S)`` held-out NLL, matching `held_out_nll`'s per-slice
    output shape and formula (``binary_cross_entropy_with_logits``)
    exactly. M0's design is the constant column ``1.0`` (`_m0_design`), so
    ``logits = theta + logit_no_theta`` directly (broadcasting ``theta``
    ``(N, 1)`` against ``logit_no_theta`` ``(N, T_max)``) without ever
    materializing a design tensor.
    """
    Bc, S, T_max = y.shape
    N = Bc * S
    mask_odd, logit_no_theta = _mask_and_logit_from_padded(item, valid_pos, opp, bank, _time_filter_odd)

    y_t = torch.as_tensor(y.reshape(N, T_max), dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_odd.reshape(N, T_max), dtype=torch.bool, device=device)
    logit_t = torch.as_tensor(logit_no_theta.reshape(N, T_max), dtype=torch.float32, device=device)
    design_t = torch.ones(N, T_max, 1, dtype=torch.float32, device=device)
    x0 = torch.zeros(N, 1, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll, x0, data_args=(y_t, mask_t, logit_t, design_t), prior_var=PRIOR_VAR
    )
    theta = result.params  # (N, 1)
    del y_t, mask_t, design_t, x0, result

    mask_even, _ = _mask_and_logit_from_padded(item, valid_pos, opp, bank, _time_filter_even)
    y_t2 = torch.as_tensor(y.reshape(N, T_max), dtype=torch.float32, device=device)
    mask_t2 = torch.as_tensor(mask_even.reshape(N, T_max), dtype=torch.bool, device=device)
    logits = theta + logit_t  # (N,1) + (N,T_max) -> (N,T_max), matches design@params+logit_no_theta for design==1
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t2, reduction="none")
    nll_even = (per_pos * mask_t2.to(per_pos.dtype)).sum(dim=-1)
    out = nll_even.reshape(Bc, S).cpu().numpy()
    del y_t2, mask_t2, logit_t, logits, per_pos, nll_even, theta
    return out


def _fit_kc_pooled_and_held_out_vectorized(
    y: np.ndarray,
    opp: np.ndarray,
    block: np.ndarray,
    valid_pos: np.ndarray,
    item: np.ndarray,
    cols: np.ndarray,
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    bank: Optional[FrozenBank],
    device: str,
) -> np.ndarray:
    """One KC's pooled fit (M1a-pooled ``shared_dim=1`` or M1b-pooled
    ``shared_dim=N_BLOCKS``), fit on the odd split and evaluated on the
    even split, over the chunk's ``Bc`` replicates via the arrow-structured
    Newton fast path (`binary_logit_nll_arrow`) -- the vectorized-assembly
    replacement for `_fit_kc_joint_batched_replicates_safe` +
    `held_out_total_nll_kc_joint_batched_replicates`. ``cols`` are THIS
    KC's own column indices into the bed-wide ``S`` axis (fixed,
    permutation-invariant, computed once outside the replicate loop).
    Returns ``(Bc,)`` total held-out NLL summed over this KC's own slices,
    matching `held_out_total_nll_kc_joint_batched_replicates`'s output and
    `held_out_nll_kc_joint`'s exact formula (manual sigmoid + eps-clipped
    log, in float64) -- deliberately NOT `binary_cross_entropy_with_logits`,
    since that reference function uses the manual form.
    """
    Bc, S, T_max = y.shape
    k = len(cols)
    y_kc = y[:, cols, :].reshape(Bc, k * T_max)
    opp_kc = opp[:, cols, :]
    block_kc = block[:, cols, :]
    valid_kc = valid_pos[:, cols, :]
    item_kc = item[:, cols, :]

    mask_odd, logit_no_theta = _mask_and_logit_from_padded(item_kc, valid_kc, opp_kc, bank, _time_filter_odd)
    mask_odd = mask_odd.reshape(Bc, k * T_max)
    logit_no_theta = logit_no_theta.reshape(Bc, k * T_max)
    shared_cols = _design_from_padded(opp_kc, block_kc, shared_design_fn, shared_dim).reshape(Bc, k * T_max, shared_dim)
    slice_idx = np.repeat(np.arange(k), T_max)[None, :].repeat(Bc, axis=0)

    y_t = torch.as_tensor(y_kc, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_odd, dtype=torch.bool, device=device)
    logit_t = torch.as_tensor(logit_no_theta, dtype=torch.float32, device=device)
    slice_idx_t = torch.as_tensor(slice_idx, dtype=torch.long, device=device)
    shared_cols_t = torch.as_tensor(shared_cols, dtype=torch.float32, device=device)
    x0 = torch.zeros(Bc, k + shared_dim, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll_arrow, x0,
        data_args=(y_t, mask_t, logit_t, slice_idx_t, shared_cols_t, k),
        prior_var=PRIOR_VAR,
    )
    theta = result.params[:, :k]
    shared = result.params[:, k:]
    del y_t, mask_t, x0, result

    mask_even, _ = _mask_and_logit_from_padded(item_kc, valid_kc, opp_kc, bank, _time_filter_even)
    mask_even = mask_even.reshape(Bc, k * T_max)
    y_t2 = torch.as_tensor(y_kc, dtype=torch.float64, device=device)
    mask_t2 = torch.as_tensor(mask_even, dtype=torch.bool, device=device)
    logit_t2 = torch.as_tensor(logit_no_theta, dtype=torch.float64, device=device)

    theta_gather = torch.gather(theta.double(), 1, slice_idx_t)
    shared_term = torch.matmul(shared_cols_t.double(), shared.double().unsqueeze(-1)).squeeze(-1)
    logits = theta_gather + shared_term + logit_t2
    p = torch.sigmoid(logits)
    eps = 1e-12
    nll = -(y_t2 * torch.log(p + eps) + (1.0 - y_t2) * torch.log(1.0 - p + eps))
    nll_even_total = (nll * mask_t2.to(nll.dtype)).sum(dim=-1)
    out = nll_even_total.cpu().numpy()
    del y_t2, mask_t2, logit_t2, shared_cols_t, slice_idx_t, theta_gather, shared_term, logits, p, nll, nll_even_total
    del theta, shared, logit_no_theta
    return out


def _fit_kc_bucket_pooled_and_held_out_vectorized(
    y: np.ndarray,
    opp: np.ndarray,
    block: np.ndarray,
    valid_pos: np.ndarray,
    item: np.ndarray,
    kc_cols_list: Sequence[np.ndarray],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    bank: Optional[FrozenBank],
    device: str,
) -> list[np.ndarray]:
    """Batches MULTIPLE KCs' pooled fits (same model, i.e. same
    ``shared_dim``) into ONE Newton call by padding every KC in the
    bucket to a common ``k_max`` (the bucket's own largest member) --
    module-level note above `_bed_key_scaffold`'s "phase 3" fix: 515
    separate per-KC Newton loops per model per chunk (`_fit_kc_pooled_and_held_out_vectorized`,
    called once per KC) each pay ~25 iterations of several small GPU
    kernel launches, and dispatch overhead from having over a thousand
    such SEPARATE calls per chunk dominates wall time once data assembly
    (phase 2) and per-call contraction-path search (phase 3's other fix,
    `torch.einsum` -> `torch.matmul`) are removed -- confirmed directly at
    production scale.

    Padding rows (a KC with ``k_c < k_max``) are all-zero AND masked out
    of both the odd (fit) and even (held-out) splits via ``real_k``, so
    they contribute EXACTLY zero to that KC's own gradient/Hessian/NLL --
    the same masking argument that already makes ``T_max`` padding within
    one slice harmless (module-level note 3), just one level up (padding
    whole SLICES within a KC's own arrow system, not just positions
    within one slice). Every KC still gets its OWN free intercept
    theta_ic (`slice_idx` ranges ``[0, k_max)`` per KC, independently) and
    its OWN shared block (``shared`` is NOT shared ACROSS KCs -- each
    KC's k_max-wide slot occupies its OWN row of the batch, and
    `penalized_bounded_newton`'s batch axis fits every row completely
    independently, exactly as it already does for different replicates).

    Returns one ``(Bc,)`` held-out-NLL-total array PER KC in
    ``kc_cols_list``, in the same order, matching
    `_fit_kc_pooled_and_held_out_vectorized`'s per-KC output exactly.
    """
    Bc, S, T_max = y.shape
    m = len(kc_cols_list)
    k_max = max(len(c) for c in kc_cols_list)

    y_b = np.zeros((Bc, m, k_max, T_max), dtype=np.float32)
    opp_b = np.zeros((Bc, m, k_max, T_max), dtype=np.int32)
    block_b = np.zeros((Bc, m, k_max, T_max), dtype=np.int32)
    valid_b = np.zeros((Bc, m, k_max, T_max), dtype=bool)
    item_b = np.zeros((Bc, m, k_max, T_max), dtype=np.int32)
    real_k = np.zeros((m, k_max), dtype=bool)
    for j, cols in enumerate(kc_cols_list):
        kc = len(cols)
        y_b[:, j, :kc, :] = y[:, cols, :]
        opp_b[:, j, :kc, :] = opp[:, cols, :]
        block_b[:, j, :kc, :] = block[:, cols, :]
        valid_b[:, j, :kc, :] = valid_pos[:, cols, :]
        item_b[:, j, :kc, :] = item[:, cols, :]
        real_k[j, :kc] = True

    N = Bc * m
    L = k_max * T_max
    y_flat = y_b.reshape(N, L)
    opp_flat = opp_b.reshape(N, L)
    block_flat = block_b.reshape(N, L)
    valid_flat = valid_b.reshape(N, L)
    item_flat = item_b.reshape(N, L)
    real_k_positions = np.repeat(np.tile(real_k, (Bc, 1)), T_max, axis=1)  # (N, L)
    del y_b, opp_b, block_b, valid_b, item_b

    mask_odd, logit_no_theta = _mask_and_logit_from_padded(item_flat, valid_flat, opp_flat, bank, _time_filter_odd)
    mask_odd = mask_odd & real_k_positions
    shared_cols = _design_from_padded(opp_flat, block_flat, shared_design_fn, shared_dim)
    slice_idx = np.tile(np.repeat(np.arange(k_max), T_max), (N, 1))

    y_t = torch.as_tensor(y_flat, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_odd, dtype=torch.bool, device=device)
    logit_t = torch.as_tensor(logit_no_theta, dtype=torch.float32, device=device)
    slice_idx_t = torch.as_tensor(slice_idx, dtype=torch.long, device=device)
    shared_cols_t = torch.as_tensor(shared_cols, dtype=torch.float32, device=device)
    x0 = torch.zeros(N, k_max + shared_dim, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll_arrow, x0,
        data_args=(y_t, mask_t, logit_t, slice_idx_t, shared_cols_t, k_max),
        prior_var=PRIOR_VAR,
    )
    theta = result.params[:, :k_max]
    shared = result.params[:, k_max:]
    del y_t, mask_t, x0, result

    mask_even, _ = _mask_and_logit_from_padded(item_flat, valid_flat, opp_flat, bank, _time_filter_even)
    mask_even = mask_even & real_k_positions
    y_t2 = torch.as_tensor(y_flat, dtype=torch.float64, device=device)
    mask_t2 = torch.as_tensor(mask_even, dtype=torch.bool, device=device)
    logit_t2 = torch.as_tensor(logit_no_theta, dtype=torch.float64, device=device)

    theta_gather = torch.gather(theta.double(), 1, slice_idx_t)
    shared_term = torch.matmul(shared_cols_t.double(), shared.double().unsqueeze(-1)).squeeze(-1)
    logits = theta_gather + shared_term + logit_t2
    p = torch.sigmoid(logits)
    eps = 1e-12
    nll = -(y_t2 * torch.log(p + eps) + (1.0 - y_t2) * torch.log(1.0 - p + eps))
    nll_total = (nll * mask_t2.to(nll.dtype)).sum(dim=-1)  # (N,)
    out = nll_total.reshape(Bc, m).cpu().numpy()
    del y_t2, mask_t2, logit_t2, shared_cols_t, slice_idx_t, theta_gather, shared_term, logits, p, nll, nll_total
    del theta, shared, logit_no_theta

    return [out[:, j] for j in range(m)]


def _bucket_kcs_by_size(
    by_kc_keys: Sequence[Sequence[tuple[int, int]]], small_kc_threshold: int = 300, bucket_size: int = 25,
) -> list[list[int]]:
    """Groups KC ids into buckets for `_fit_kc_bucket_pooled_and_held_out_vectorized`:
    KCs at or below ``small_kc_threshold`` slices (the vast majority on a
    real bed -- ~493 of 515 on KDD_MATCHED) are sorted by slice count
    ascending, then split into buckets of (at most) ``bucket_size`` KCs
    each, so each bucket's members are similarly sized (bounding the
    padding waste smaller members pay for a shared ``k_max``). KCs ABOVE
    the threshold each get their OWN singleton bucket -- the caller
    recognizes a length-1 bucket and routes it to the ORIGINAL, proven
    per-KC path (`_fit_kc_pooled_and_held_out_vectorized`) instead, exactly
    mirroring `_compute_per_kc_chunk_sizes`'s own small/large distinction
    (the reason it exists there applies here identically: a single
    oversized KC must never force a huge ``k_max`` onto a bucket of
    otherwise-small KCs). Empty KCs (no slices) are dropped entirely."""
    small = [c for c, keys in enumerate(by_kc_keys) if keys and len(keys) <= small_kc_threshold]
    large = [c for c, keys in enumerate(by_kc_keys) if keys and len(keys) > small_kc_threshold]
    small.sort(key=lambda c: len(by_kc_keys[c]))
    buckets = [small[i : i + bucket_size] for i in range(0, len(small), bucket_size)]
    buckets.extend([c] for c in large)
    return buckets


def _permutation_null_batched_chunk_vectorized(
    chunk_learners: Sequence,
    key_to_col: np.ndarray,
    by_kc_keys: Sequence[Sequence[tuple[int, int]]],
    cols_by_kc: Sequence[np.ndarray],
    n_kcs: int,
    S: int,
    T_max: int,
    bank: Optional[FrozenBank],
    device: str,
    kc_chunk_sizes: dict[int, int],
    kc_buckets: Sequence[Sequence[int]],
    bed_kc_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One chunk's worth of `permutation_null_batched`'s vectorized-assembly
    fast path (module-level note above `_bed_key_scaffold`): builds the
    chunk's padded arrays ONCE (`_stack_chunk_padded_arrays`), fits M0
    across the whole chunk, then processes KCs in `kc_buckets` groups
    (`_bucket_kcs_by_size`) slicing OUT of that SAME padded array (no
    rebuild per KC) instead of `_build_kc_joint_arrow_arrays`. A
    multi-member bucket is fit in ONE Newton call via
    `_fit_kc_bucket_pooled_and_held_out_vectorized` (phase 3's fix for
    per-KC GPU-dispatch overhead); a singleton bucket (an oversized KC,
    per `_bucket_kcs_by_size`'s small/large split) uses the ORIGINAL
    per-KC path (`_fit_kc_pooled_and_held_out_vectorized`) unchanged.
    Returns ``(total_m1a, total_m1b, kc_null_chunk)`` -- ``(Bc,)``, ``(Bc,)``,
    ``(Bc, n_kcs)`` -- matching the original per-chunk loop's own local
    variables exactly, so the caller's bed_null/kc_null assembly is
    unchanged regardless of which path produced these.
    """
    Bc = len(chunk_learners)
    chunk_rows = [build_calibration_rows(lg) for lg in chunk_learners]
    y, valid_pos, opp, block, item = _stack_chunk_padded_arrays(chunk_rows, key_to_col, n_kcs, S, T_max)
    del chunk_rows

    m0_nll_even = _fit_m0_and_held_out_vectorized(y, opp, valid_pos, item, bank, device)  # (Bc, S)

    total_m1a = np.zeros(Bc)
    total_m1b = np.zeros(Bc)
    kc_null_chunk = np.zeros((Bc, n_kcs))
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for bucket in kc_buckets:
        if len(bucket) == 1:
            c = bucket[0]
            cols = cols_by_kc[c]
            m0_kc_total = m0_nll_even[:, cols].sum(axis=1)  # (Bc,)

            # Sub-chunk THIS KC's own replicate batch by ITS OWN chunk
            # size (see `permutation_null_batched`'s incident-context
            # docstring): even with the arrow fast path, a sufficiently
            # wide KC's own (Bc, k*T_max, shared_dim) tensors are sized
            # by its OWN k, not the outer chunk's M0 sizing.
            kc_chunk = kc_chunk_sizes.get(c, 1)
            nll_a_total = np.zeros(Bc)
            nll_b_total = np.zeros(Bc)
            for sub_start in range(0, Bc, kc_chunk):
                sub_end = min(sub_start + kc_chunk, Bc)
                sl_ = slice(sub_start, sub_end)
                nll_a_total[sl_] = _fit_kc_pooled_and_held_out_vectorized(
                    y[sl_], opp[sl_], block[sl_], valid_pos[sl_], item[sl_], cols, _m1a_shared_design, 1, bank, device
                )
                nll_b_total[sl_] = _fit_kc_pooled_and_held_out_vectorized(
                    y[sl_], opp[sl_], block[sl_], valid_pos[sl_], item[sl_], cols, _m1b_shared_design, N_BLOCKS,
                    bank, device,
                )
            stat_kc_a = m0_kc_total - nll_a_total
            stat_kc_b = m0_kc_total - nll_b_total
            if bed_kc_mask is None or bed_kc_mask[c]:
                total_m1a += stat_kc_a
                total_m1b += stat_kc_b
            kc_null_chunk[:, c] = np.maximum(stat_kc_a, stat_kc_b)
        else:
            kc_cols_list = [cols_by_kc[c] for c in bucket]
            # Bound the replicate sub-chunk by the bucket's own most
            # conservative member (a shared sub-chunk size across the
            # whole bucket, never rebuilt per-KC).
            kc_chunk = min((kc_chunk_sizes.get(c, 1) for c in bucket), default=1)
            nll_a_by_kc = [np.zeros(Bc) for _ in bucket]
            nll_b_by_kc = [np.zeros(Bc) for _ in bucket]
            for sub_start in range(0, Bc, kc_chunk):
                sub_end = min(sub_start + kc_chunk, Bc)
                sl_ = slice(sub_start, sub_end)
                a_results = _fit_kc_bucket_pooled_and_held_out_vectorized(
                    y[sl_], opp[sl_], block[sl_], valid_pos[sl_], item[sl_], kc_cols_list,
                    _m1a_shared_design, 1, bank, device,
                )
                b_results = _fit_kc_bucket_pooled_and_held_out_vectorized(
                    y[sl_], opp[sl_], block[sl_], valid_pos[sl_], item[sl_], kc_cols_list,
                    _m1b_shared_design, N_BLOCKS, bank, device,
                )
                for j in range(len(bucket)):
                    nll_a_by_kc[j][sl_] = a_results[j]
                    nll_b_by_kc[j][sl_] = b_results[j]
            for j, c in enumerate(bucket):
                m0_kc_total = m0_nll_even[:, cols_by_kc[c]].sum(axis=1)
                stat_kc_a = m0_kc_total - nll_a_by_kc[j]
                stat_kc_b = m0_kc_total - nll_b_by_kc[j]
                if bed_kc_mask is None or bed_kc_mask[c]:
                    total_m1a += stat_kc_a
                    total_m1b += stat_kc_b
                kc_null_chunk[:, c] = np.maximum(stat_kc_a, stat_kc_b)

        if is_cuda:
            torch.cuda.empty_cache()

    return total_m1a, total_m1b, kc_null_chunk


def permutation_null_batched(
    learners: Sequence,
    n_learners: int,
    n_kcs: int,
    bank: Optional[FrozenBank],
    n_replicates: int,
    seed: int,
    device: str = "cpu",
    replicate_chunk_size: Optional[int] = None,
    safety_factor: float = 2.0,
    small_kc_threshold: int = 300,
    use_vectorized_assembly: bool = True,
    bed_kc_mask: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Replicate-batched equivalent of `permutation_null_looped`: the SAME
    permutation draws, in the SAME order (identical RNG consumption --
    ``rng = np.random.default_rng(seed)`` then ``permute_learner_order``
    called ``n_replicates`` times, exactly as the looped path does), and
    the SAME ``bed``/``kc`` null distributions to float tolerance -- but
    fits M0 and every KC-pooled model (M1a-pooled, M1b-pooled) in ONE
    Newton call per chunk of replicates (batch = chunk_size * n_slices
    for M0; batch = THAT KC's OWN chunk size for each KC's pooled fit --
    see ``small_kc_threshold`` and the note below on why this must be
    per-KC, not one bed-wide size) instead of one call per replicate.
    This is the A4 perf-surgery fix `newton.py`'s module docstring flags:
    `battery.run_permutation_battery`'s B=999/199 replicate counts
    otherwise re-pay `penalized_bounded_newton`'s eager `torch.func`
    nested-vmap dispatch overhead on every one of B replicates, dominated
    in call COUNT by the per-KC pooled fits (2 Newton calls per KC per
    replicate).

    Only ``bed_stat``/``kc_stat`` are computed here (matching what
    `permutation_null` itself has always returned) -- NOT the per-slice
    M1a-slice/M1b-slice models, since those feed `GateResult.slice_stat`
    alone, which no caller of the permutation null
    (`battery.run_permutation_battery`, `run.py`'s campaign cells) reads
    off the null distributions.

    M0's outer chunk is sized by `_calibrate_m0_chunk_size_empirically`
    (an EMPIRICAL probe against real bed data and real free device
    memory, with a ``safety_factor`` headroom multiplier -- default 2x)
    unless ``replicate_chunk_size`` is given explicitly. Each KC's OWN
    chunk size is sized SEPARATELY by `_compute_per_kc_chunk_sizes`
    (KCs at or below ``small_kc_threshold`` slices get ``max_chunk``
    directly with no probe; only larger KCs are individually calibrated).
    Both are best-effort STARTING sizes only -- correctness/safety never
    depend on them being accurate (see the per-KC OOM fallback below).

    INCIDENT CONTEXT (production A40 cluster non-completions; four fixes,
    in the order the incident actually unfolded): (1) ``unit_29`` hit a
    SINGLE huge allocation because an earlier ANALYTIC chunk estimator
    sized the chunk from M0's cheap footprint alone, ignoring the far
    larger per-row cost of the KC-joint fit on a real bed's largest KC --
    addressed by empirical calibration. (2) two other units died by
    ACCUMULATION -- allocations climbing chunk-to-chunk until a later,
    modest allocation failed -- fixed by explicit per-chunk cleanup below
    (``del`` every GPU-resident intermediate as soon as its CPU-side
    result is extracted, plus ``torch.cuda.empty_cache()`` at chunk
    boundaries and periodically within a chunk's KC loop) and a runtime
    assertion (`_assert_no_memory_growth_across_chunks`) that the
    allocator's reported allocated bytes do NOT grow chunk-over-chunk once
    cleanup has run. (3) a KC whose design width ``P = k + shared_dim`` is
    large enough that even a 2-replicate batched Hessian attempt tried to
    allocate 287+ GiB -- no chunk-size arithmetic, however conservative,
    avoids this alone, because the cost is combinatorial in ``P`` via
    `torch.func`'s nested vmap/jvp/vjp Hessian construction, not linear in
    the replicate batch size; every batched Newton call in the hot path
    (M0 via `_fit_batched_replicates_safe`, each KC's pooled fit via
    `_fit_kc_joint_batched_replicates_safe`) therefore catches
    ``torch.cuda.OutOfMemoryError`` and falls back to the ORIGINAL
    per-replicate S=1 loop (proven safe -- literally what the pre-batching
    implementation always did) for whichever chunk/KC triggered it. (4)
    THE ACTUAL ROOT CAUSE OF NON-COMPLETION, found only after (1)-(3) still
    failed a live 2-hour verification run (SLURM TIMEOUT, not OOM -- the
    process was confirmed alive and CPU-bound the whole time): a single
    GLOBAL chunk size, calibrated against the bed's single largest KC,
    collapses to 1 for safety -- but that SAME chunk=1 then applied to
    EVERY other KC too (a real KDD_MATCHED bed had a median KC size of
    ~103 slices but several KCs with ~1700-3000), so batch=1 degenerates
    every KC-joint call to no better than the original per-replicate loop:
    the fix avoided the OOM but silently gave back the ENTIRE speedup for
    the whole unit. Fixed by calibrating and chunking PER KC (see
    ``small_kc_threshold``, `_compute_per_kc_chunk_sizes`) so ~500 normal-
    sized KCs batch at full speed while only the genuinely oversized ones
    fall back to a small chunk or per-replicate looping -- for themselves
    alone, not for the whole bed.

    ``use_vectorized_assembly`` (default True, 2026-07-20 4th A4 perf-
    surgery phase 2, `_planning/LEDGER.md`): the incident-context notes
    above describe the SOLVE-side fixes; a SEPARATE root cause, found
    only after those held, is that every replicate paid several O(n_slices)
    PYTHON-LEVEL loops (`build_slices`'s per-group `Slice` construction,
    `_pad_design`'s per-slice torch conversion -- run twice, once for
    fitting and once inside `held_out_nll` -- and `_build_kc_joint_arrow_arrays`'s
    per-slice loop, paid twice per KC) to rebuild its data from scratch --
    node-level profiling of a live production unit found the permutation
    battery GPU-IDLE and single-CPU-core-bound the ENTIRE time, and local
    production-scale profiling measured ~13-14s of PURE PYTHON LOOP
    overhead per replicate this way (`_bed_key_scaffold` and friends'
    module-level docstring above has the exact breakdown), which at
    B=999 alone accounts for 3-4 hours of the incident's wall time --
    dwarfing anything the Newton solve itself could cost. The default
    path (`_permutation_null_batched_chunk_vectorized`) replaces ALL of
    that with vectorized numpy scatter-assigns straight from
    `bank.build_calibration_rows`'s own (unchanged, already-vectorized)
    output into padded ``(Bc, S, T_max)`` arrays, reused directly by
    every KC's pooled fit (sliced out of the SAME array, never rebuilt) --
    no `Slice` object, and no per-slice/per-group Python loop, anywhere in
    the hot path. ``use_vectorized_assembly=False`` reaches the ORIGINAL
    per-replicate Slice-based path, kept as the equivalence reference
    (`tests/test_growth_gate_perf_equivalence.py`) and as the automatic
    per-chunk fallback on `torch.cuda.OutOfMemoryError`/`MemoryError`.
    """
    rng = np.random.default_rng(seed)
    perm_learners_list = [permute_learner_order(learners, rng) for _ in range(n_replicates)]

    # Slice membership, per-slice length T, and hence D2+/stratum status
    # are permutation-invariant (`permute_learner_order` only reorders
    # each learner's OWN rows; every row keeps its own (learner, KC) tags
    # and item id, so which (learner, KC) slices exist, and how long each
    # one is, cannot change) -- computed ONCE from the unpermuted learners.
    base_rows = build_calibration_rows(learners)
    base_slices = build_slices(base_rows)
    all_keys = list(base_slices.keys())
    by_kc_keys: list[list[tuple[int, int]]] = [[] for _ in range(n_kcs)]
    for key, sl in base_slices.items():
        by_kc_keys[sl.kc].append(key)

    bed_null = np.zeros(n_replicates)
    kc_null = np.zeros((n_replicates, n_kcs))
    if not all_keys:
        return {"bed": bed_null, "kc": kc_null}

    if replicate_chunk_size is None:
        replicate_chunk_size = _calibrate_m0_chunk_size_empirically(
            perm_learners_list, all_keys, bank, device, safety_factor=safety_factor,
        )
    kc_chunk_sizes = _compute_per_kc_chunk_sizes(
        perm_learners_list, by_kc_keys, bank, device,
        safety_factor=safety_factor, small_kc_threshold=small_kc_threshold,
    )

    key_to_col = {k: i for i, k in enumerate(all_keys)}
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    post_chunk_allocated: list[int] = []

    S = len(all_keys)
    T_max = max((sl.T for sl in base_slices.values()), default=0)
    key_to_col_arr = _bed_key_scaffold(all_keys, n_kcs, n_learners) if use_vectorized_assembly else None
    cols_by_kc = (
        [np.array([key_to_col[k] for k in keys_c], dtype=np.int64) for keys_c in by_kc_keys]
        if use_vectorized_assembly
        else None
    )
    kc_buckets = (
        _bucket_kcs_by_size(by_kc_keys, small_kc_threshold=small_kc_threshold)
        if use_vectorized_assembly
        else None
    )

    for chunk_start in range(0, n_replicates, replicate_chunk_size):
        chunk_end = min(chunk_start + replicate_chunk_size, n_replicates)
        chunk_learners = perm_learners_list[chunk_start:chunk_end]
        Bc = len(chunk_learners)

        if use_vectorized_assembly:
            try:
                total_m1a, total_m1b, kc_null_chunk = _permutation_null_batched_chunk_vectorized(
                    chunk_learners, key_to_col_arr, by_kc_keys, cols_by_kc, n_kcs, S, T_max,
                    bank, device, kc_chunk_sizes, kc_buckets, bed_kc_mask=bed_kc_mask,
                )
                kc_null[chunk_start:chunk_end, :] = kc_null_chunk
                bed_null[chunk_start:chunk_end] = np.maximum(total_m1a, total_m1b)
                if is_cuda:
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                    post_chunk_allocated.append(int(torch.cuda.memory_allocated(device)))
                continue
            except (torch.cuda.OutOfMemoryError, MemoryError):
                # Fall through to the original per-replicate Slice-based
                # path for THIS chunk only (module-level note above): the
                # vectorized path's (Bc, S, T_max) arrays are a different
                # memory shape than the original per-slice construction,
                # so a chunk that OOMs under one may still fit under the
                # other. Every OTHER chunk keeps trying the fast path.
                if is_cuda:
                    torch.cuda.empty_cache()

        chunk_slices = []
        for lg in chunk_learners:
            rows = build_calibration_rows(lg)
            chunk_slices.append(build_slices(rows))
        for sd in chunk_slices:
            if set(sd.keys()) != set(all_keys):
                raise ValueError(
                    "permutation_null_batched: a permuted replicate's slice "
                    "membership does not match the unpermuted bed; the "
                    "batched path's permutation-invariance assumption is violated."
                )

        slices_by_replicate_all = [[sd[k] for k in all_keys] for sd in chunk_slices]
        m0_params = _fit_batched_replicates_safe(
            slices_by_replicate_all, bank, _m0_design, P=1, time_filter=_time_filter_odd, device=device
        )
        m0_nll_even = np.stack(
            [
                held_out_nll(
                    slices_by_replicate_all[b], bank, _m0_design, m0_params[b], _time_filter_even, device=device
                )
                .cpu()
                .numpy()
                for b in range(Bc)
            ]
        )  # (Bc, S)
        del m0_params, slices_by_replicate_all

        total_m1a = np.zeros(Bc)
        total_m1b = np.zeros(Bc)
        for c in range(n_kcs):
            keys_c = by_kc_keys[c]
            if not keys_c:
                continue
            kc_slices_by_replicate = [[sd[k] for k in keys_c] for sd in chunk_slices]
            cols = [key_to_col[k] for k in keys_c]
            m0_kc_total = m0_nll_even[:, cols].sum(axis=1)  # (Bc,)

            # Sub-chunk THIS KC's own replicate batch by ITS OWN chunk
            # size (never the outer M0 chunk size, which the incident-
            # context docstring above explains is the wrong scope for
            # this decision): a KC's own P = k + shared_dim is what
            # determines whether it is safe to batch at 200 replicates
            # at once, at 1, or anywhere between.
            k_local = len(keys_c)
            kc_chunk = kc_chunk_sizes.get(c, 1)
            theta_a = np.zeros((Bc, k_local))
            beta_c = np.zeros((Bc, 1))
            theta_b = np.zeros((Bc, k_local))
            u_c = np.zeros((Bc, N_BLOCKS))
            for sub_start in range(0, Bc, kc_chunk):
                sub_end = min(sub_start + kc_chunk, Bc)
                sub_slices = kc_slices_by_replicate[sub_start:sub_end]
                ta, bc_ = _fit_kc_joint_batched_replicates_safe(
                    sub_slices, bank, _m1a_shared_design, 1, time_filter=_time_filter_odd, device=device
                )
                theta_a[sub_start:sub_end] = ta
                beta_c[sub_start:sub_end] = bc_
                tb, uc_ = _fit_kc_joint_batched_replicates_safe(
                    sub_slices, bank, _m1b_shared_design, N_BLOCKS, time_filter=_time_filter_odd, device=device
                )
                theta_b[sub_start:sub_end] = tb
                u_c[sub_start:sub_end] = uc_

            nll_a_total = held_out_total_nll_kc_joint_batched_replicates(
                kc_slices_by_replicate, bank, _m1a_shared_design, theta_a, beta_c, _time_filter_even
            )
            nll_b_total = held_out_total_nll_kc_joint_batched_replicates(
                kc_slices_by_replicate, bank, _m1b_shared_design, theta_b, u_c, _time_filter_even
            )

            stat_kc_a = m0_kc_total - nll_a_total
            stat_kc_b = m0_kc_total - nll_b_total
            if bed_kc_mask is None or bed_kc_mask[c]:
                total_m1a += stat_kc_a
                total_m1b += stat_kc_b
            kc_null[chunk_start:chunk_end, c] = np.maximum(stat_kc_a, stat_kc_b)
            del kc_slices_by_replicate, theta_a, beta_c, theta_b, u_c

            # Periodic (not per-KC) cache release: a real bed can have
            # hundreds of KCs, each a DIFFERENT (L_max, P) tensor shape, so
            # PyTorch's caching allocator can accumulate many differently-
            # sized cached-but-unused blocks within a single chunk even
            # though no Python reference leaks -- this bounds that growth
            # without paying a synchronize+cache-scan on every KC.
            if is_cuda and (c % 50 == 49):
                torch.cuda.empty_cache()

        bed_null[chunk_start:chunk_end] = np.maximum(total_m1a, total_m1b)
        del chunk_slices, m0_nll_even

        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            post_chunk_allocated.append(int(torch.cuda.memory_allocated(device)))

    if is_cuda:
        _assert_no_memory_growth_across_chunks(post_chunk_allocated)

    return {"bed": bed_null, "kc": kc_null}


def permutation_null(
    learners: Sequence,
    n_learners: int,
    n_kcs: int,
    bank: Optional[FrozenBank],
    n_replicates: int,
    seed: int,
    device: str = "cpu",
    use_batched: bool = True,
    replicate_chunk_size: Optional[int] = None,
    safety_factor: float = 2.0,
    small_kc_threshold: int = 300,
    bed_kc_mask: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Battery arm 1's hook: ``n_replicates`` permutation draws of the
    bed-level and per-KC gate statistic (design: "permute each learner's
    full interaction order, then rebuild slices and opportunity indices").
    Returns the null distributions; p-value/BH-FDR assembly is a separate
    step (`gate_pvalue_bed`, `bh_fdr`) so callers can mix a small-B
    exploratory null (this module's own tests) with the pre-registered
    B=199/999 battery run (`battery.py`, a later stage) through the same
    interface.

    Dispatches to the replicate-batched path (`permutation_null_batched`,
    the A4 perf-surgery fix) by default; ``use_batched=False`` reaches the
    original per-replicate loop (`permutation_null_looped`), kept as the
    equivalence-gate reference and as an explicit fallback. ``bed_kc_mask``
    (the saturation-aware bed-statistic restriction, module docstring note
    6) is forwarded to whichever path runs, so the null's bed statistic is
    summed over exactly the KC subset the observed read used.
    """
    if use_batched:
        return permutation_null_batched(
            learners, n_learners, n_kcs, bank, n_replicates, seed,
            device=device, replicate_chunk_size=replicate_chunk_size, safety_factor=safety_factor,
            small_kc_threshold=small_kc_threshold, bed_kc_mask=bed_kc_mask,
        )
    return permutation_null_looped(
        learners, n_learners, n_kcs, bank, n_replicates, seed, device=device, bed_kc_mask=bed_kc_mask
    )


def empirical_pvalue(observed: float, null: np.ndarray) -> float:
    """Right-tailed permutation p-value (design's empirical-p floor
    convention: ``(1 + #{null >= observed}) / (B + 1)``)."""
    B = len(null)
    if B == 0:
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (B + 1))


# ---------------------------------------------------------------------------
# BH-FDR and Benjamini-Yekutieli (section 2.2's dual discovery control)
# ---------------------------------------------------------------------------


def bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR control (primary per-KC discovery rule).
    Returns a boolean rejection array in the ORIGINAL order of
    ``pvalues``."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = sorted_p <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    k_max = np.max(np.where(passed)[0])
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[: k_max + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def by_correction(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Yekutieli FDR control, valid under arbitrary dependence
    (section 2.2's cross-KC dependence protection: "every per-KC discovery
    list is additionally reported under Benjamini-Yekutieli q = 0.05")."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = q * (np.arange(1, m + 1) / (m * c_m))
    passed = sorted_p <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    k_max = np.max(np.where(passed)[0])
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[: k_max + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


__all__ = [
    "PRIOR_VAR",
    "SliceFit",
    "fit_batched",
    "fit_batched_replicates",
    "held_out_nll",
    "fit_m0",
    "fit_m1a_slice",
    "fit_m1b_slice",
    "fit_kc_joint",
    "fit_kc_joint_batched_replicates",
    "held_out_nll_kc_joint",
    "held_out_total_nll_kc_joint_batched_replicates",
    "GateResult",
    "compute_gate_result",
    "gate_statistic_on_learners",
    "permutation_null",
    "permutation_null_looped",
    "permutation_null_batched",
    "empirical_pvalue",
    "bh_fdr",
    "by_correction",
    "_m1a_shared_design",
    "_m1b_shared_design",
    "_time_filter_odd",
    "_time_filter_even",
    "_env_memory_budget_bytes",
    "_cgroup_memory_budget_bytes",
    "_host_memory_budget_bytes",
    "_cpu_chunk_from_budget",
    "_budget_capped_free_bytes",
    "_calibrate_m0_chunk_size_empirically",
    "_calibrate_kc_chunk_size_empirically",
    "_compute_per_kc_chunk_sizes",
    "_assert_no_memory_growth_across_chunks",
    "_fit_batched_replicates_safe",
    "_fit_kc_joint_batched_replicates_safe",
]
