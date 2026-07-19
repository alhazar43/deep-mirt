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
   reusing the SAME shared Newton primitive (`penalized_bounded_newton`
   with `binary_logit_nll`, S=1 per call) -- functionally identical
   machinery, no duplicated calculus, just not vectorized across the KC
   axis. A fully KC-batched variant (grouping same-slice-count KCs, or
   padding) is a real-bed performance optimization left to whichever
   later stage tunes battery-scale runtime; it changes wall-clock, not
   any estimator's output.
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
"""

from __future__ import annotations

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
from kt_mirt.growth.newton import NewtonResult, binary_logit_nll, penalized_bounded_newton
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
    x0 = torch.zeros(B * S, P, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll, x0, data_args=(y_all, mask_all, logit_all, design_all), prior_var=prior_var
    )
    return result.params.reshape(B, S, P)


def fit_m0(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m0_design, P=1, time_filter=time_filter, device=device)


def fit_m1a_slice(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m1a_slice_design, P=2, time_filter=time_filter, device=device)


def fit_m1b_slice(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m1b_slice_design, P=N_BLOCKS, time_filter=time_filter, device=device)


# ---------------------------------------------------------------------------
# KC-pooled joint fits (M1a-pooled, M1b-pooled): one Newton call per KC
# ---------------------------------------------------------------------------


def _build_kc_joint_arrays(
    kc_slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assembles one KC's joint (per-slice one-hot theta + shared block)
    design as flat ``(L,)``/``(L, P)`` numpy arrays (``P = k +
    shared_dim``, ``k = len(kc_slices)``, ``L`` the number of valid,
    time-filtered positions across all ``k`` slices) -- pure data
    assembly, no fitting. Factored out of `fit_kc_joint` (unchanged
    arithmetic, a pure refactor) so the identical construction is shared
    by the S=1 per-replicate call AND `fit_kc_joint_batched_replicates`'s
    S=B call: the only thing that differs between the looped and batched
    paths is how many of these per-replicate arrays go into one Newton
    call, never how one replicate's own array is built.
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
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, k + shared_dim), dtype=np.float32),
        )
    slice_idx = np.concatenate(slice_idx_parts)
    y = np.concatenate(y_parts)
    mask = np.concatenate(mask_parts)
    logit_no_theta = np.concatenate(logit_parts)
    shared_cols = np.concatenate(shared_parts, axis=0)
    L = len(y)
    onehot = np.zeros((L, k), dtype=np.float32)
    onehot[np.arange(L), slice_idx] = 1.0
    design = np.concatenate([onehot, shared_cols.astype(np.float32)], axis=1)
    return y, mask, logit_no_theta, design


def fit_kc_joint(
    kc_slices: Sequence[Slice],
    bank: Optional[FrozenBank],
    shared_design_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shared_dim: int,
    time_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    prior_var: float = PRIOR_VAR,
    device: str = "cpu",
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
    """
    k = len(kc_slices)
    if k == 0:
        return np.zeros(0), np.zeros(shared_dim)
    y, mask, logit_no_theta, design = _build_kc_joint_arrays(
        kc_slices, bank, shared_design_fn, shared_dim, time_filter
    )
    if len(y) == 0:
        return np.zeros(k), np.zeros(shared_dim)

    P = k + shared_dim
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
    """
    B = len(kc_slices_by_replicate)
    k = len(kc_slices_by_replicate[0]) if B else 0
    if B == 0 or k == 0:
        return np.zeros((B, k)), np.zeros((B, shared_dim))

    arrays = [
        _build_kc_joint_arrays(kc_slices_by_replicate[b], bank, shared_design_fn, shared_dim, time_filter)
        for b in range(B)
    ]
    P = k + shared_dim
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
    x0 = torch.zeros(B, P, device=device)
    result = penalized_bounded_newton(
        binary_logit_nll, x0, data_args=(y_t, mask_t, logit_t, design_t), prior_var=prior_var
    )
    params = result.params.cpu().numpy()
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
    slices: dict[tuple[int, int], Slice], bank: Optional[FrozenBank], n_kcs: int, device: str = "cpu"
) -> GateResult:
    """Fit M0 once (batched over every slice in the bed) and both M1
    families at both pooling levels, on the interpolative odd/even split,
    and assemble the max-selected gate statistic at every pooling level
    (module docstring note 1)."""
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
    learners: Sequence, n_learners: int, n_kcs: int, bank: Optional[FrozenBank], device: str = "cpu"
) -> GateResult:
    """Rebuild rows and slices from a (possibly permuted) learner log list
    and compute the gate result -- the one operation the permutation null
    repeats B times."""
    rows = build_calibration_rows(learners)
    slices = build_slices(rows)
    return compute_gate_result(slices, bank, n_kcs, device=device)


def permutation_null_looped(
    learners: Sequence,
    n_learners: int,
    n_kcs: int,
    bank: Optional[FrozenBank],
    n_replicates: int,
    seed: int,
    device: str = "cpu",
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
    removes.
    """
    rng = np.random.default_rng(seed)
    bed_null = np.zeros(n_replicates)
    kc_null = np.zeros((n_replicates, n_kcs))
    for r in range(n_replicates):
        perm_learners = permute_learner_order(learners, rng)
        result = gate_statistic_on_learners(perm_learners, n_learners, n_kcs, bank, device=device)
        bed_null[r] = result.bed_stat
        kc_null[r] = result.kc_stat
    return {"bed": bed_null, "kc": kc_null}


def _estimate_replicate_chunk_size(
    n_slices_bed: int,
    t_max_bed: int,
    device: str,
    target_bytes: Optional[int] = None,
    max_chunk: int = 200,
) -> int:
    """Memory-aware default chunk size for `permutation_null_batched`'s
    outer replicate loop. M0's ``(chunk, S_bed, T_max)`` data tensors
    dominate per-chunk memory (the KC-pooled fits are far smaller per
    replicate, since one KC's own slice count is a small fraction of the
    bed), so the estimate is based on M0 alone: ~4 float32 data tensors
    per (replicate, slice, position) triple, times a conservative x6
    safety margin for the Newton solver's own gradient/Hessian
    intermediates during the backward pass. On CUDA, targets a quarter of
    currently-FREE device memory (leaving headroom for a co-resident
    process, e.g. a running campaign unit sharing the same GPU); on CPU,
    a fixed conservative default. Never exceeds ``max_chunk`` (the
    100-200 range the perf-surgery task itself suggests) regardless of
    how much memory is technically available, since dispatch-count
    reduction has strongly diminishing returns well before that size.
    """
    if target_bytes is None:
        if device.startswith("cuda") and torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            target_bytes = int(0.25 * free_bytes)
        else:
            target_bytes = 1_500_000_000  # ~1.5 GB, conservative default for CPU RAM
    bytes_per_replicate = max(1, n_slices_bed) * max(1, t_max_bed) * 4 * 6
    chunk = max(1, target_bytes // bytes_per_replicate)
    return int(min(chunk, max_chunk))


def permutation_null_batched(
    learners: Sequence,
    n_learners: int,
    n_kcs: int,
    bank: Optional[FrozenBank],
    n_replicates: int,
    seed: int,
    device: str = "cpu",
    replicate_chunk_size: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Replicate-batched equivalent of `permutation_null_looped`: the SAME
    permutation draws, in the SAME order (identical RNG consumption --
    ``rng = np.random.default_rng(seed)`` then ``permute_learner_order``
    called ``n_replicates`` times, exactly as the looped path does), and
    the SAME ``bed``/``kc`` null distributions to float tolerance -- but
    fits M0 and every KC-pooled model (M1a-pooled, M1b-pooled) in ONE
    Newton call per chunk of replicates (batch = chunk_size * n_slices
    for M0; batch = chunk_size for each KC's pooled fit) instead of one
    call per replicate. This is the A4 perf-surgery fix `newton.py`'s
    module docstring flags: `battery.run_permutation_battery`'s B=999/199
    replicate counts otherwise re-pay `penalized_bounded_newton`'s eager
    `torch.func` nested-vmap dispatch overhead on every one of B
    replicates, dominated in call COUNT by the per-KC pooled fits (2
    Newton calls per KC per replicate).

    Only ``bed_stat``/``kc_stat`` are computed here (matching what
    `permutation_null` itself has always returned) -- NOT the per-slice
    M1a-slice/M1b-slice models, since those feed `GateResult.slice_stat`
    alone, which no caller of the permutation null
    (`battery.run_permutation_battery`, `run.py`'s campaign cells) reads
    off the null distributions.

    Chunked over the replicate axis (`replicate_chunk_size`, memory-aware
    default via `_estimate_replicate_chunk_size`) to bound GPU/CPU memory
    at real-bed scale (e.g. KDD_MATCHED: 515 KCs, ~tens of thousands of
    slices per bed).
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
    t_max_bed = max((sl.T for sl in base_slices.values()), default=0)
    by_kc_keys: list[list[tuple[int, int]]] = [[] for _ in range(n_kcs)]
    for key, sl in base_slices.items():
        by_kc_keys[sl.kc].append(key)

    bed_null = np.zeros(n_replicates)
    kc_null = np.zeros((n_replicates, n_kcs))
    if not all_keys:
        return {"bed": bed_null, "kc": kc_null}

    if replicate_chunk_size is None:
        replicate_chunk_size = _estimate_replicate_chunk_size(len(all_keys), t_max_bed, device)

    key_to_col = {k: i for i, k in enumerate(all_keys)}

    for chunk_start in range(0, n_replicates, replicate_chunk_size):
        chunk_end = min(chunk_start + replicate_chunk_size, n_replicates)
        chunk_learners = perm_learners_list[chunk_start:chunk_end]
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

        Bc = len(chunk_learners)
        slices_by_replicate_all = [[sd[k] for k in all_keys] for sd in chunk_slices]
        m0_params = fit_batched_replicates(
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

        total_m1a = np.zeros(Bc)
        total_m1b = np.zeros(Bc)
        for c in range(n_kcs):
            keys_c = by_kc_keys[c]
            if not keys_c:
                continue
            kc_slices_by_replicate = [[sd[k] for k in keys_c] for sd in chunk_slices]
            cols = [key_to_col[k] for k in keys_c]
            m0_kc_total = m0_nll_even[:, cols].sum(axis=1)  # (Bc,)

            theta_a, beta_c = fit_kc_joint_batched_replicates(
                kc_slices_by_replicate, bank, _m1a_shared_design, 1, time_filter=_time_filter_odd, device=device
            )
            nll_a_total = held_out_total_nll_kc_joint_batched_replicates(
                kc_slices_by_replicate, bank, _m1a_shared_design, theta_a, beta_c, _time_filter_even
            )
            theta_b, u_c = fit_kc_joint_batched_replicates(
                kc_slices_by_replicate, bank, _m1b_shared_design, N_BLOCKS, time_filter=_time_filter_odd, device=device
            )
            nll_b_total = held_out_total_nll_kc_joint_batched_replicates(
                kc_slices_by_replicate, bank, _m1b_shared_design, theta_b, u_c, _time_filter_even
            )

            stat_kc_a = m0_kc_total - nll_a_total
            stat_kc_b = m0_kc_total - nll_b_total
            total_m1a += stat_kc_a
            total_m1b += stat_kc_b
            kc_null[chunk_start:chunk_end, c] = np.maximum(stat_kc_a, stat_kc_b)

        bed_null[chunk_start:chunk_end] = np.maximum(total_m1a, total_m1b)

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
    equivalence-gate reference and as an explicit fallback.
    """
    if use_batched:
        return permutation_null_batched(
            learners, n_learners, n_kcs, bank, n_replicates, seed,
            device=device, replicate_chunk_size=replicate_chunk_size,
        )
    return permutation_null_looped(learners, n_learners, n_kcs, bank, n_replicates, seed, device=device)


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
]
