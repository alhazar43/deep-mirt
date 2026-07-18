"""MIX-L, the MIXED posture's gate-then-rate ladder
(`_planning/design/a4_design.md` v1.1, section 2.4): existence gate =
`gate.py` verbatim (stage 1 of the ladder, one implementation, ONE
result), then the bounded-exponential rate stage on gate-passing KCs
(E-M1 pooled KC rate, E-M2 per-learner rate), and the family-misfit flag
E-M3 that withholds ``r`` wherever the shape-agnostic blockwise model
beats the bounded-exponential on held-out data.

Interpretation notes (recorded per the harness instructions):

1. **The bounded-exponential fit is nonlinear in its own parameters**
   (``theta(n) = m - (m - theta0) * exp(-r*(n-1))``), unlike M0/M1a/M1b's
   GLM-linear-in-parameters shape, so it cannot reuse `newton.py`'s
   `binary_logit_nll` convenience wrapper; it supplies its OWN `nll_fn` to
   the same shared `penalized_bounded_newton` primitive (no duplicated
   Newton machinery, only a different per-slice objective, exactly what
   the primitive is designed to accept).
2. **E-M1 (pooled per-KC rate) is a joint fit** (shared ``m_c, r_c``, free
   ``theta0_ic`` per slice) -- structurally identical to `gate.py`'s
   `fit_kc_joint` (one Newton call per KC, S=1, looped over KCs; see that
   module's note 2 for why). **E-M2 (per-learner rate)** fixes ``m_c`` at
   its E-M1 value (external constant, not re-fit) and frees ``(theta0_ic,
   r_ic)`` per slice -- with ``m_c`` now a plain per-slice DATA input
   rather than a shared free parameter, E-M2 is fully independent across
   slices and is therefore batched across every gate-passing D2+ slice in
   ONE Newton call (S = that many slices), which is both faithful to the
   design ("same family with per-slice (theta0, r) and shared m_c") and
   the cheaper of the two fits.
3. **Exponent clamping.** ``exp(-r*(n-1))`` can transiently overflow
   during a bad (pre-backtrack) Newton trial step; `newton.py`'s own
   backtracking/accept logic already guarantees a non-finite trial step is
   never accepted (see that module's note 4), so this cannot corrupt the
   MAP estimate, but an unclamped overflow still raises a noisy
   `RuntimeWarning` and wastes a step. The exponent is clamped to
   ``[-30, 30]`` (comfortably outside the pre-registered rate range
   ``[0.02, 0.40]`` times any realistic opportunity count times the
   Newton step clamp) purely as numerical hygiene, matching the ridge
   safety margin `newton.py` already applies to its Hessian.
4. **E-M3's paired bootstrap** resamples KC SLICES with replacement (not
   individual responses -- "paired bootstrap over slices" per section
   2.4), recomputing each replicate's mean(bounded-exponential held-out
   NLL) minus mean(blockwise held-out NLL) under the ALREADY-FITTED
   (odd-index) parameters; the one-sided p-value is the bootstrap fraction
   where blockwise does NOT beat bounded-exponential (paired difference
   <= 0). This reuses `gate.py`'s `fit_kc_joint`/`held_out_nll_kc_joint`
   for the blockwise side (M1b, one implementation, section 2.4's own
   "Existence gate = PAS-G verbatim" principle extended to the shared
   blockwise machinery) rather than re-deriving it.
5. **The r=0 within-family test** (section 2.4 point 4: "the rate stage
   carries its own within-family test in addition to the gate") is
   implemented as a Wald test: the penalized objective's Hessian at the
   converged joint MAP (recomputed once via `torch.func.hessian`, the same
   autodiff machinery `newton.py` already uses internally) gives an
   approximate posterior covariance (inverse Hessian); ``z = r_c /
   se(r_c)`` against the standard normal is reported alongside the point
   estimate. This is a standard MAP-Laplace approximation, not a new
   statistical assumption beyond what the shared Newton primitive already
   commits to (a Gaussian-MAP fit whose curvature is exactly this
   Hessian).
6. **Data-driven initialization for E-M1's joint fit, not a zero start**
   (build-time regression, found while iterating tests to green): a
   literal all-zeros ``x0`` lands ``(theta0, m_c, r_c)`` exactly at the
   point where ``e = exp(-r*(n-1)) = 1`` for every opportunity, which
   makes ``d(theta_n)/dm = 1 - e = 0`` identically -- a genuine
   stationary saddle in the ``m`` direction, not merely a slow start. On
   synthetic data generated at ``r_true = 0.2`` this converged (per the
   solver's own step-norm criterion) to ``r_hat ~ 0.0001`` with a
   penalized NLL of 556 against 379 at the true parameters -- a real bad
   local optimum, confirmed by an essentially-zero gradient AT the
   converged point (this is not a Newton-solver bug: `newton.py`'s
   convergence check is correct, the objective genuinely has this basin).
   The fix seeds ``theta0_ic`` from each slice's own Laplace-smoothed
   early-quarter logit rate and ``m_c``/``r_c`` from the population's
   late-quarter logit rate (plus a margin) and a mid-band rate constant
   respectively (`_quarter_logit`, mirroring the SAME early/late-quarter
   vocabulary `tracker.displacement` and E-P1's ``Delta_c`` already use
   elsewhere in this package) -- entirely a smarter STARTING POINT, never
   a change to the pre-registered step-clamp (1.0) or iteration cap (25).
   With this init the same synthetic case recovers ``r_hat ~ 0.09`` (true
   0.2, correct sign and order of magnitude) within the pre-registered
   25-iteration budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from kt_mirt.growth.bank import FrozenBank, N_BLOCKS
from kt_mirt.growth.gate import (
    PRIOR_VAR,
    _m1b_shared_design,
    _time_filter_even,
    _time_filter_odd,
    _validity_mask,
    fit_kc_joint,
    held_out_nll_kc_joint,
)
from kt_mirt.growth.newton import penalized_bounded_newton
from kt_mirt.growth.slices import Slice

_EXP_CLAMP = 30.0
_R_INIT = 0.1  # nonzero init, roughly the pre-registered band's geometric mean (module docstring note 6)
_M_INIT_MARGIN = 0.5


def _quarter_logit(response: np.ndarray) -> tuple[float, float]:
    """Laplace-smoothed logit success rate over a slice's first and last
    quarter (module docstring note 6) -- reuses the same early/late-
    quarter split `tracker.displacement` and E-P1's ``Delta_c`` already
    apply elsewhere, here purely to seed the nonlinear Newton fit away
    from the ``r=0`` degenerate stationary point a zero start lands on."""
    T = len(response)
    if T == 0:
        return 0.0, 0.0
    q = max(1, T // 4)
    early = response[:q].astype(float)
    late = response[-q:].astype(float)
    p_early = (early.sum() + 0.5) / (len(early) + 1.0)
    p_late = (late.sum() + 0.5) / (len(late) + 1.0)
    return float(np.log(p_early / (1 - p_early))), float(np.log(p_late / (1 - p_late)))


# ---------------------------------------------------------------------------
# E-M1: pooled per-KC bounded-exponential rate (shared m_c, r_c; free theta0_ic)
# ---------------------------------------------------------------------------


def _bounded_exp_kc_nll(
    params: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, b: torch.Tensor,
    opp: torch.Tensor, onehot: torch.Tensor,
) -> torch.Tensor:
    """``params`` = [theta0_1..theta0_k, m_c, r_c]; ``onehot`` (L, k)
    selects each position's slice-local theta0."""
    k = onehot.shape[-1]
    theta0 = onehot @ params[:k]
    m_c = params[k]
    r_c = params[k + 1]
    exponent = torch.clamp(-r_c * (opp - 1.0), min=-_EXP_CLAMP, max=_EXP_CLAMP)
    theta_n = m_c - (m_c - theta0) * torch.exp(exponent)
    logits = theta_n - b
    per_pos = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per_pos * mask.to(per_pos.dtype)).sum()


@dataclass
class KCRateFit:
    theta0: np.ndarray  # (k,) per-slice
    m_c: float
    r_c: float
    r_c_se: float
    r_c_z: float


def _build_kc_arrays(
    kc_slices: Sequence[Slice], bank: Optional[FrozenBank], time_filter
) -> tuple[np.ndarray, ...]:
    slice_idx_parts, y_parts, mask_parts, b_parts, opp_parts = [], [], [], [], []
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
        b_parts.append(b.astype(np.float32))
        opp_parts.append(sl.opportunity.astype(np.float32))
    if not slice_idx_parts:
        return (np.zeros(0, dtype=np.int64),) * 5
    return (
        np.concatenate(slice_idx_parts),
        np.concatenate(y_parts),
        np.concatenate(mask_parts),
        np.concatenate(b_parts),
        np.concatenate(opp_parts),
    )


def fit_kc_rate(
    kc_slices: Sequence[Slice], bank: Optional[FrozenBank], device: str = "cpu", prior_var: float = PRIOR_VAR
) -> Optional[KCRateFit]:
    """E-M1: fit the bounded-exponential family jointly for one KC (shared
    ``m_c, r_c``, free per-slice ``theta0_ic``), on the odd-index split
    (design section 2.2's evaluation regime, shared by the rate stage)."""
    k = len(kc_slices)
    if k == 0:
        return None
    slice_idx, y, mask, b, opp = _build_kc_arrays(kc_slices, bank, _time_filter_odd)
    if len(y) == 0 or not mask.any():
        return None
    L = len(y)
    onehot = np.zeros((L, k), dtype=np.float32)
    onehot[np.arange(L), slice_idx] = 1.0
    P = k + 2

    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=device).unsqueeze(0)
    opp_t = torch.as_tensor(opp, dtype=torch.float32, device=device).unsqueeze(0)
    onehot_t = torch.as_tensor(onehot, dtype=torch.float32, device=device).unsqueeze(0)

    theta0_init = np.zeros(k)
    late_vals: list[float] = []
    for i, sl in enumerate(kc_slices):
        early_logit, late_logit = _quarter_logit(sl.response)
        theta0_init[i] = early_logit
        late_vals.append(late_logit)
    m_init = (max(late_vals + [float(theta0_init.max())]) + _M_INIT_MARGIN) if late_vals else 0.0
    x0 = torch.zeros(1, P, device=device)
    x0[0, :k] = torch.as_tensor(theta0_init, dtype=torch.float32, device=device)
    x0[0, k] = m_init
    x0[0, k + 1] = _R_INIT

    result = penalized_bounded_newton(
        _bounded_exp_kc_nll, x0, data_args=(y_t, mask_t, b_t, opp_t, onehot_t), prior_var=prior_var
    )
    params = result.params[0]

    # Wald test on r_c (module docstring note 5): Hessian of the penalized
    # objective at the converged MAP, inverted for an approximate covariance.
    def objective(p: torch.Tensor) -> torch.Tensor:
        return _bounded_exp_kc_nll(p, y_t[0], mask_t[0], b_t[0], opp_t[0], onehot_t[0]) + 0.5 * (
            p**2
        ).sum() / prior_var

    hess = torch.func.hessian(objective)(params)
    r_idx = P - 1
    try:
        cov = torch.linalg.inv(hess + 1e-6 * torch.eye(P))
        se = float(torch.sqrt(cov[r_idx, r_idx].clamp(min=1e-12)))
    except Exception:
        se = float("nan")
    r_c = float(params[r_idx])
    z = r_c / se if se and np.isfinite(se) and se > 0 else float("nan")

    return KCRateFit(
        theta0=params[:k].detach().cpu().numpy(),
        m_c=float(params[k]),
        r_c=r_c,
        r_c_se=se,
        r_c_z=z,
    )


# ---------------------------------------------------------------------------
# E-M2: per-slice rate (shared, fixed m_c; free theta0_ic, r_ic), batched
# ---------------------------------------------------------------------------


def _rate_slice_nll(
    params: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, b: torch.Tensor,
    opp: torch.Tensor, m_c: torch.Tensor,
) -> torch.Tensor:
    theta0, r = params[0], params[1]
    exponent = torch.clamp(-r * (opp - 1.0), min=-_EXP_CLAMP, max=_EXP_CLAMP)
    theta_n = m_c - (m_c - theta0) * torch.exp(exponent)
    logits = theta_n - b
    per_pos = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per_pos * mask.to(per_pos.dtype)).sum()


@dataclass
class SliceRateFit:
    keys: list[tuple[int, int]]
    theta0: np.ndarray  # (S,)
    r_ic: np.ndarray  # (S,)


def fit_slice_rate(
    slices: Sequence[Slice], m_c_by_kc: dict[int, float], bank: Optional[FrozenBank],
    device: str = "cpu", prior_var: float = PRIOR_VAR,
) -> SliceRateFit:
    """E-M2: per-slice ``(theta0_ic, r_ic)`` with ``m_c`` fixed at the
    E-M1 value for that slice's KC (module docstring note 2) -- batched
    across every slice in one Newton call, since ``m_c`` is now a fixed
    per-slice data input rather than a shared free parameter."""
    slices = [sl for sl in slices if sl.kc in m_c_by_kc]
    keys = [(sl.learner, sl.kc) for sl in slices]
    if not slices:
        return SliceRateFit(keys=[], theta0=np.zeros(0), r_ic=np.zeros(0))
    T_max = max(sl.T for sl in slices)
    S = len(slices)
    y = torch.zeros(S, T_max)
    mask = torch.zeros(S, T_max, dtype=torch.bool)
    b_t = torch.zeros(S, T_max)
    opp_t = torch.zeros(S, T_max)
    m_c_t = torch.zeros(S)
    for i, sl in enumerate(slices):
        t = sl.T
        valid = _validity_mask(sl, bank) & _time_filter_odd(sl.opportunity)
        y[i, :t] = torch.as_tensor(sl.response, dtype=torch.float32)
        mask[i, :t] = torch.as_tensor(valid, dtype=torch.bool)
        b = bank.difficulty(sl.item_id) if bank is not None else np.zeros(t)
        b_t[i, :t] = torch.as_tensor(b, dtype=torch.float32)
        opp_t[i, :t] = torch.as_tensor(sl.opportunity, dtype=torch.float32)
        m_c_t[i] = m_c_by_kc[sl.kc]
    x0 = torch.zeros(S, 2, device=device)
    for i, sl in enumerate(slices):
        early_logit, _ = _quarter_logit(sl.response)
        x0[i, 0] = early_logit
        x0[i, 1] = _R_INIT
    result = penalized_bounded_newton(
        _rate_slice_nll, x0,
        data_args=(y.to(device), mask.to(device), b_t.to(device), opp_t.to(device), m_c_t.to(device)),
        prior_var=prior_var,
    )
    params = result.params.cpu().numpy()
    return SliceRateFit(keys=keys, theta0=params[:, 0], r_ic=params[:, 1])


# ---------------------------------------------------------------------------
# E-M3: family-misfit flag (blockwise beats bounded-exponential -> withhold r)
# ---------------------------------------------------------------------------


def _bounded_exp_held_out_nll(
    kc_slices: Sequence[Slice], bank: Optional[FrozenBank], fit: KCRateFit
) -> np.ndarray:
    """Per-slice held-out (even-index) NLL under the E-M1 fit's fixed
    parameters."""
    out = np.zeros(len(kc_slices))
    for i, sl in enumerate(kc_slices):
        t = sl.T
        if t == 0:
            continue
        valid = _validity_mask(sl, bank) & _time_filter_even(sl.opportunity)
        if not valid.any():
            continue
        b = bank.difficulty(sl.item_id) if bank is not None else np.zeros(t)
        exponent = np.clip(-fit.r_c * (sl.opportunity - 1.0), -_EXP_CLAMP, _EXP_CLAMP)
        theta_n = fit.m_c - (fit.m_c - fit.theta0[i]) * np.exp(exponent)
        logits = theta_n - b
        y = sl.response.astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        nll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        out[i] = float(nll[valid].sum())
    return out


@dataclass
class MisfitFlagResult:
    fired: bool
    p_value: float
    mean_diff: float  # mean(bounded_exp_nll) - mean(blockwise_nll); >0 favors blockwise


def misfit_flag(
    kc_slices: Sequence[Slice], bank: Optional[FrozenBank], rate_fit: KCRateFit,
    n_bootstrap: int = 200, seed: int = 0, alpha: float = 0.05,
) -> MisfitFlagResult:
    """E-M3: does the shape-agnostic blockwise model (M1b, `gate.py`'s
    pooled fit) beat the bounded-exponential on held-out data? Paired
    bootstrap over slices (module docstring note 4)."""
    if len(kc_slices) == 0:
        return MisfitFlagResult(fired=False, p_value=float("nan"), mean_diff=0.0)
    theta_b, u_c = fit_kc_joint(kc_slices, bank, _m1b_shared_design, N_BLOCKS, time_filter=_time_filter_odd)
    blockwise_nll = held_out_nll_kc_joint(kc_slices, bank, _m1b_shared_design, theta_b, u_c, _time_filter_even)
    bounded_exp_nll = _bounded_exp_held_out_nll(kc_slices, bank, rate_fit)

    diff = bounded_exp_nll - blockwise_nll  # >0: blockwise has lower (better) held-out NLL
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diff[idx].mean()
    p_value = float(np.mean(boot_means <= 0.0))
    return MisfitFlagResult(fired=p_value < alpha, p_value=p_value, mean_diff=float(diff.mean()))


__all__ = [
    "KCRateFit",
    "fit_kc_rate",
    "SliceRateFit",
    "fit_slice_rate",
    "MisfitFlagResult",
    "misfit_flag",
]
