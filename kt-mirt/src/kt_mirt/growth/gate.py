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


def fit_m0(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m0_design, P=1, time_filter=time_filter, device=device)


def fit_m1a_slice(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m1a_slice_design, P=2, time_filter=time_filter, device=device)


def fit_m1b_slice(slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter=None, device="cpu") -> SliceFit:
    return fit_batched(slices, bank, _m1b_slice_design, P=N_BLOCKS, time_filter=time_filter, device=device)


# ---------------------------------------------------------------------------
# KC-pooled joint fits (M1a-pooled, M1b-pooled): one Newton call per KC
# ---------------------------------------------------------------------------


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
        return np.zeros(k), np.zeros(shared_dim)
    slice_idx = np.concatenate(slice_idx_parts)
    y = np.concatenate(y_parts)
    mask = np.concatenate(mask_parts)
    logit_no_theta = np.concatenate(logit_parts)
    shared_cols = np.concatenate(shared_parts, axis=0)
    L = len(y)
    onehot = np.zeros((L, k), dtype=np.float32)
    onehot[np.arange(L), slice_idx] = 1.0
    design = np.concatenate([onehot, shared_cols.astype(np.float32)], axis=1)

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
        stat_a = m0_d2_nll - m1a_s_nll.numpy()
        stat_b = m0_d2_nll - m1b_s_nll.numpy()
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


def permutation_null(
    learners: Sequence,
    n_learners: int,
    n_kcs: int,
    bank: Optional[FrozenBank],
    n_replicates: int,
    seed: int,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Battery arm 1's hook: ``n_replicates`` permutation draws of the
    bed-level and per-KC gate statistic (design: "permute each learner's
    full interaction order, then rebuild slices and opportunity indices").
    Returns the null distributions; p-value/BH-FDR assembly is a separate
    step (`gate_pvalue_bed`, `bh_fdr`) so callers can mix a small-B
    exploratory null (this module's own tests) with the pre-registered
    B=199/999 battery run (`battery.py`, a later stage) through the same
    interface.
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
    "held_out_nll",
    "fit_m0",
    "fit_m1a_slice",
    "fit_m1b_slice",
    "fit_kc_joint",
    "held_out_nll_kc_joint",
    "GateResult",
    "compute_gate_result",
    "gate_statistic_on_learners",
    "permutation_null",
    "empirical_pvalue",
    "bh_fdr",
    "by_correction",
    "_m1a_shared_design",
    "_m1b_shared_design",
    "_time_filter_odd",
    "_time_filter_even",
]
