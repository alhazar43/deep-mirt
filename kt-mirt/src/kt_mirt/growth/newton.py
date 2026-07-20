"""Penalized bounded Newton fitting: the shared numeric primitive behind
every slice-level fit in the A4 design (`_planning/design/a4_design.md`
v1.1, section 2.2, review item R2-B3): PAS-G's M0/M1a/M1b (`growth/gate.py`,
a later stage), the MIX-L bounded-exponential rate stage (`growth/rate.py`,
a later stage), and the permutation null, all of which "run under the
identical penalized machinery" so that "the gate statistic's calibration
is preserved." This module owns none of those models' likelihoods -- it
is a generic, batched, Gaussian-MAP Newton-Raphson solver: a caller
supplies a per-slice scalar data-NLL callable (any functional form,
linear or nonlinear in its own parameters -- both the GLM-shaped M0/M1
family and the nonlinear bounded-exponential rate stage are ordinary
Python callables here) plus batched per-slice data tensors, and gets back
finite MAP estimates even under quasi-complete separation.

Design quotes (section 2.2): "all slice-level logistic fits (M0, M1a,
M1b, and the MIX-L rate stage's deviations) are PENALIZED, Gaussian prior
N(0, 2.0^2 logits) ... fit by damped, bounded Newton (step-norm clamp
1.0, max 25 iterations, backtracking on any NLL increase) ... The penalty
guarantees finite MAP estimates and finite held-out log-likelihoods on
separated slices; it is shared by M0 and M1 (no asymmetric advantage),
and the permutation null runs under the identical penalized machinery."
A unit test in this module asserts finite parameters and finite held-out
NLL on an all-correct and an all-incorrect slice (the design's explicit
pre-registered unit test, section 2.2).

Interpretation notes (recorded per the harness instructions -- the design
fixes the prior scale, step clamp, iteration cap, and backtracking
requirement, but not the solver's internals beyond those four numbers):

1. Gradients and Hessians of the (data NLL + Gaussian penalty) objective
   are computed by automatic differentiation (`torch.func.grad` /
   `torch.func.hessian`, batched over slices with `torch.func.vmap`), not
   hand-derived per model, EXCEPT for the `binary_logit_nll` family (every
   M0/M1a/M1b fit in `gate.py`, both the plain and replicate-batched
   forms), which is linear-in-params penalized logistic regression: its
   gradient and Hessian have closed forms (``grad = X^T(sigma(X beta) -
   y) + beta/prior_var``, ``Hessian = X^T diag(p(1-p)) X + I/prior_var``),
   computed below as batched einsums with no `torch.func` involved. A
   production perf incident (`_planning/LEDGER.md`, 2026-07-20 morning)
   found that eager `torch.func` vmap/jvp/vjp pays Python dispatch
   overhead PER NEWTON ITERATION (not just per outer call), which
   dominates real-scale slice-fit wall time; the closed form removes that
   dispatch entirely for the one objective family that is by far the
   hottest path (every gate/battery Newton call in `gate.py` uses
   `binary_logit_nll`). Dispatch is automatic and by identity
   (``nll_fn is binary_logit_nll``, see `penalized_bounded_newton`'s
   `analytic` parameter) -- any OTHER objective, notably `rate.py`'s
   nonlinear bounded-exponential rate stage, is unaffected and keeps
   running the generic `torch.func` path, which also remains callable
   directly (`analytic=False`) as the equivalence reference the analytic
   path is tested against (`tests/test_growth_newton.py`). This keeps the
   primitive model-agnostic for any OTHER caller while making the one
   linear-in-params GLM family that dominates real usage dispatch-free;
   gate.py's M0/M1a/M1b and rate.py's bounded-exponential still share one
   Newton loop with zero duplicated damping/backtracking/convergence
   logic, which is exactly what "shared by M0 and M1 (no asymmetric
   advantage)" requires.
6. **KC-joint pooled fits get a SECOND, structured fast path**
   (`binary_logit_nll_arrow`, 4th A4 perf-surgery, `_planning/LEDGER.md`
   2026-07-20): `gate.py`'s `fit_kc_joint`/`fit_kc_joint_batched_replicates`
   give each of a KC's ``k`` slices a free intercept theta_ic plus a small
   shared parameter block (``shared_dim`` = 1 or 4), so the per-KC design
   is a ``k``-column one-hot block bordered by ``shared_dim`` dense
   columns. Since each row's one-hot column is disjoint from every other
   row's (a row belongs to exactly one slice), the penalized Hessian's
   intercept-intercept block is EXACTLY diagonal -- an ARROW matrix ``[[D,
   B], [B^T, C]]`` with ``D`` (k,) diagonal, ``B`` (k, shared_dim) the
   border, ``C`` (shared_dim, shared_dim) the shared-shared block -- and
   oversized real KCs (k up to ~3000) previously paid a dense ``O(k^3)``
   `torch.linalg.solve` per Newton iteration for no reason: the true
   solve is an ``O(k * shared_dim^2 + shared_dim^3)`` block-elimination
   (Schur complement) of that structure (`_arrow_schur_step`), PROVABLY
   IDENTICAL to the dense solve (exact linear algebra, not an
   approximation -- `tests/test_growth_newton.py`'s exactness suite checks
   this directly against `torch.linalg.solve` on random arrow systems).
   `binary_logit_nll_arrow` is the identity-dispatched marker for this
   path (mirroring `binary_logit_nll`'s role for note 1's dense fast
   path), consuming the compact ``(slice_idx, shared_cols)``
   representation of the same design (`gate._build_kc_joint_arrow_arrays`)
   instead of a materialized ``(L, k+shared_dim)`` one-hot-bordered
   matrix, which is itself the other half of the memory fix (``O(k)``
   instead of ``O(k^2)``). This path has no generic-`torch.func`
   counterpart (forcing ``analytic=False`` on it raises `ValueError`); its
   equivalence reference is the EXISTING dense `binary_logit_nll` path on
   the literal one-hot design, reachable via `gate.py`'s
   ``use_arrow=False`` escape hatch.
2. Each slice's own data (its response sequence, mask, frozen item
   difficulties, any per-slice design features) is threaded through as
   extra batched tensor arguments to ``nll_fn`` (leading dimension ``S``,
   the slice axis), vmapped alongside the parameters. This matches the
   design's arm-1 framing of slice-based machinery as "vectorized ...
   over hundreds of thousands of slices, torch tensors."
3. The Hessian is ridge-regularized (``hessian_ridge``, default 1e-6)
   before solving, purely for numerical conditioning; the Gaussian prior
   alone already guarantees a strictly positive-definite Hessian in exact
   arithmetic (the prior curvature ``1/sigma^2`` lower-bounds the total
   curvature since the data NLL of a logistic/exponential-family model is
   convex), so the ridge is a safety margin, not load-bearing.
4. "Backtracking on any NLL increase" is read as the TOTAL penalized
   objective (data NLL + prior), since that is the quantity the Newton
   step is derived to decrease; a step that raises the penalized
   objective for a given slice is halved (up to ``max_backtrack`` times,
   independently per slice) before being accepted, matching "damped."
   Slices that still fail to improve after ``max_backtrack`` halvings
   simply keep their previous iterate for that outer iteration (never
   accept a worsening step); this cannot itself produce a non-finite
   result since the previous iterate was finite.
5. Per-slice convergence (``step norm < tol``) is tracked independently,
   so faster-converging slices in a batch stop updating while slower
   ones continue for up to ``max_iter`` outer iterations; the returned
   ``n_iter`` records how many updates each slice actually received.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor


# Dispatch-path visibility (2026-07-20 3rd A4 perf-surgery, per the
# "never silent again" review note): a lightweight per-process counter,
# NOT a per-call print (penalized_bounded_newton runs thousands of times
# per slice cell; per-call logging would be pure spam). Callers that want
# to confirm the analytic fast path is actually engaging in their own hot
# loop (e.g. `run.py`'s slice-cell driver, after the permutation-battery
# stage) read `dispatch_counts()` and reset with `reset_dispatch_counts()`.
_dispatch_counts = {"analytic": 0, "generic": 0, "arrow": 0}


def dispatch_counts() -> dict[str, int]:
    """Returns a COPY of the process-wide {"analytic": n, "generic": n,
    "arrow": n} call counts accumulated by `penalized_bounded_newton` since
    the last `reset_dispatch_counts()` (or process start)."""
    return dict(_dispatch_counts)


def reset_dispatch_counts() -> None:
    _dispatch_counts["analytic"] = 0
    _dispatch_counts["generic"] = 0
    _dispatch_counts["arrow"] = 0


@dataclass
class NewtonResult:
    """Batched penalized-Newton output; every field has leading dim ``S``
    (or ``(S, P)`` for ``params``), one entry per slice."""

    params: Tensor  # (S, P) MAP estimates
    converged: Tensor  # (S,) bool
    n_iter: Tensor  # (S,) long, iterations actually taken
    data_nll: Tensor  # (S,) unpenalized data NLL at the final params
    penalized_nll: Tensor  # (S,) data NLL + Gaussian prior penalty


def _prior_penalty(params: Tensor, prior_var: Tensor) -> Tensor:
    """``params`` (..., P), ``prior_var`` (P,) broadcastable -> (...,)."""
    return 0.5 * (params**2 / prior_var).sum(dim=-1)


def _run_newton_loop(
    step_fn: Callable[[Tensor], Tensor],
    obj_fn: Callable[[Tensor], Tensor],
    data_nll_fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    step_clip: float,
    max_iter: int,
    max_backtrack: int,
    tol: float,
) -> NewtonResult:
    """The damped, bounded, per-slice-independent Newton loop itself
    (design section 2.2), factored out of `penalized_bounded_newton` so
    EVERY dispatch path -- the generic `torch.func` path, the
    `binary_logit_nll` dense analytic fast path, and the KC-joint
    `binary_logit_nll_arrow` arrow-structured fast path (module docstring
    note 6) -- runs through the exact same damping/backtracking/step-
    clamp/convergence code below; the paths differ ONLY in how
    ``step_fn``/``obj_fn``/``data_nll_fn`` compute their values, never in
    the control flow that consumes them. ``step_fn(x)`` returns the RAW
    (pre-step-clip) Newton step for the penalized objective at ``x`` --
    gradient, Hessian (or its arrow-structured equivalent), ridge, and the
    linear solve are all its own responsibility, so this loop never
    constructs a dense Hessian itself; ``obj_fn``/``data_nll_fn`` return
    the penalized objective / unpenalized data NLL respectively, each
    ``(S,)``.
    """
    device = x0.device
    dtype = x0.dtype
    S = x0.shape[0]

    x = x0.clone()
    converged = torch.zeros(S, dtype=torch.bool, device=device)
    n_iter = torch.zeros(S, dtype=torch.long, device=device)
    obj_val = obj_fn(x)

    for _ in range(max_iter):
        active = ~converged
        if not bool(active.any()):
            break

        step = step_fn(x)
        step_norm = step.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (step_clip / step_norm).clamp(max=1.0)
        step = step * scale

        x_new = x - step
        obj_new = obj_fn(x_new)

        # Backtracking (damped Newton): halve the step, per slice,
        # independently, until the penalized objective does not increase
        # or the backtrack budget is exhausted.
        bt_scale = torch.ones(S, device=device, dtype=dtype)
        need_bt = (obj_new > obj_val) & active
        bt_count = 0
        while bool(need_bt.any()) and bt_count < max_backtrack:
            bt_scale = torch.where(need_bt, bt_scale * 0.5, bt_scale)
            x_try = x - step * bt_scale.unsqueeze(-1)
            obj_try = obj_fn(x_try)
            improved = obj_try <= obj_val
            take = need_bt & improved
            x_new = torch.where(take.unsqueeze(-1), x_try, x_new)
            obj_new = torch.where(take, obj_try, obj_new)
            need_bt = need_bt & (~improved)
            bt_count += 1

        # Accept only non-worsening steps; a slice whose backtracking
        # never found an improving point simply holds its previous
        # (finite) iterate for this outer iteration.
        accept = active & (obj_new <= obj_val + 1e-12)
        step_taken_norm = torch.where(accept, step.norm(dim=-1), torch.zeros_like(obj_val))
        x = torch.where(accept.unsqueeze(-1), x_new, x)
        obj_val = torch.where(accept, obj_new, obj_val)
        n_iter = torch.where(active, n_iter + 1, n_iter)

        newly_converged = active & (step_taken_norm < tol)
        converged = converged | newly_converged

    final_data_nll = data_nll_fn(x)
    final_penalized = obj_fn(x)
    return NewtonResult(
        params=x,
        converged=converged,
        n_iter=n_iter,
        data_nll=final_data_nll,
        penalized_nll=final_penalized,
    )


def _binary_logit_logits(params: Tensor, logit_no_theta: Tensor, design: Tensor) -> Tensor:
    """``params`` (S, P), ``logit_no_theta`` (S, T), ``design`` (S, T, P)
    -> (S, T) logits, batched (no `torch.func`, plain einsum)."""
    return torch.einsum("stp,sp->st", design, params) + logit_no_theta


def _binary_logit_data_nll_batch(
    params: Tensor, y: Tensor, mask: Tensor, logit_no_theta: Tensor, design: Tensor
) -> Tensor:
    """Batched, `torch.func`-free equivalent of ``vmap(binary_logit_nll)``:
    same masked Bernoulli NLL sum, evaluated directly over the leading
    slice axis instead of one slice at a time under `vmap`."""
    logits = _binary_logit_logits(params, logit_no_theta, design)
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per_pos * mask.to(per_pos.dtype)).sum(dim=-1)


def _binary_logit_grad_hess_batch(
    params: Tensor,
    y: Tensor,
    mask: Tensor,
    logit_no_theta: Tensor,
    design: Tensor,
    prior_var: Tensor,
) -> tuple[Tensor, Tensor]:
    """Closed-form gradient and Hessian of ``binary_logit_nll``'s penalized
    objective (data NLL + Gaussian prior), batched over the leading slice
    axis as plain einsums -- the analytic replacement for
    ``vmap(grad(objective))`` / ``vmap(hessian(objective))``.

    Since ``logit = design @ params + logit_no_theta`` is LINEAR in
    ``params``, the per-position Bernoulli NLL has the standard GLM closed
    form: writing ``p = sigmoid(logit)`` and ``r = mask * (p - y)``,
    ``grad_data = design^T @ r`` (sum over the T axis); writing
    ``w = mask * p * (1 - p)``, ``hess_data = design^T @ diag(w) @
    design``. Masked-out positions (``mask`` False) contribute exactly
    zero to both, matching `binary_logit_nll`'s own masking (it zeroes the
    same positions' contribution to the NLL sum before differentiating).
    The Gaussian prior adds ``params / prior_var`` to the gradient and
    ``diag(1 / prior_var)`` to the Hessian, matching `_prior_penalty`.
    """
    logits = _binary_logit_logits(params, logit_no_theta, design)
    p = torch.sigmoid(logits)
    mask_f = mask.to(p.dtype)
    r = mask_f * (p - y)
    w = mask_f * p * (1.0 - p)

    grad_data = torch.einsum("stp,st->sp", design, r)
    hess_data = torch.einsum("stp,st,stq->spq", design, w, design)

    grad = grad_data + params / prior_var
    hess = hess_data + torch.diag(1.0 / prior_var)
    return grad, hess


# ---------------------------------------------------------------------------
# KC-joint arrow-structured fast path (module docstring note 6): the
# compact (slice_idx, shared_cols) representation of a one-hot-bordered
# design, its analytic grad/Hessian PIECES (never a dense (P, P) Hessian),
# and the exact Schur-complement solve of the resulting arrow system.
# ---------------------------------------------------------------------------


def _kc_joint_arrow_logits_batch(
    theta: Tensor, shared: Tensor, logit_no_theta: Tensor, slice_idx: Tensor, shared_cols: Tensor
) -> Tensor:
    """``theta`` (Bn, k), ``shared`` (Bn, shared_dim), ``logit_no_theta``
    (Bn, L), ``slice_idx`` (Bn, L) long in ``[0, k)``, ``shared_cols`` (Bn,
    L, shared_dim) -> (Bn, L) logits. Mathematically identical to
    ``design @ params + logit_no_theta`` with ``design = concat(one_hot(
    slice_idx, k), shared_cols)`` and ``params = concat(theta, shared)``,
    but a per-row gather instead of a one-hot matmul -- the compact
    representation never materializes the (Bn, L, k) one-hot block.

    Uses `torch.matmul` (batched BLAS, no contraction-path search) rather
    than `torch.einsum` for the ``shared_cols @ shared`` term -- see the
    module-level perf note above `_arrow_schur_step`: for these small
    ``shared_dim`` (1 or 4) tensors, called thousands of times per
    permutation-battery chunk, `torch.einsum`'s `opt_einsum.contract_path`
    search cost EXCEEDS the actual contraction cost by an order of
    magnitude (confirmed by direct cProfile at production scale, 2026-07-20
    4th A4 perf-surgery phase 3)."""
    theta_gather = torch.gather(theta, 1, slice_idx)
    shared_term = torch.matmul(shared_cols, shared.unsqueeze(-1)).squeeze(-1)
    return theta_gather + shared_term + logit_no_theta


def _kc_joint_arrow_data_nll_batch(
    params: Tensor,
    y: Tensor,
    mask: Tensor,
    logit_no_theta: Tensor,
    slice_idx: Tensor,
    shared_cols: Tensor,
    k: int,
) -> Tensor:
    """Batched data NLL for the arrow-structured KC-joint model, same
    masked Bernoulli NLL sum as `_binary_logit_data_nll_batch` (used for
    backtracking's objective comparisons and the final data-NLL report,
    never for the Newton step itself)."""
    theta, shared = params[:, :k], params[:, k:]
    logits = _kc_joint_arrow_logits_batch(theta, shared, logit_no_theta, slice_idx, shared_cols)
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per_pos * mask.to(per_pos.dtype)).sum(dim=-1)


def _kc_joint_arrow_grad_hess_batch(
    params: Tensor,
    y: Tensor,
    mask: Tensor,
    logit_no_theta: Tensor,
    slice_idx: Tensor,
    shared_cols: Tensor,
    k: int,
    prior_var: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Closed-form gradient and ARROW-STRUCTURED Hessian pieces of the
    penalized KC-joint objective, batched over the leading (replicate)
    axis -- the structured analogue of `_binary_logit_grad_hess_batch`,
    but returning ``(g_theta, g_shared, D, B, C)`` instead of one dense
    ``(P, P)`` Hessian, since the intercept-intercept block is diagonal by
    construction (every row's one-hot column touches exactly one slice, so
    cross terms between two different slices' intercepts are always zero
    -- see `newton.py`'s module docstring note 6 and gate.py's
    `_build_kc_joint_arrow_arrays`).

    ``D`` (Bn, k): diagonal intercept curvature, computed by segment-
    summing (``scatter_add``) each row's curvature ``w`` into its own
    slice's column -- ``O(Bn * L)``, never ``O(Bn * L * k)``. ``B`` (Bn,
    k, shared_dim): the border, ``sum_{t in slice i} w_t * shared_cols_t``,
    same scatter-add trick. ``C`` (Bn, shared_dim, shared_dim): the dense
    shared-shared block, ``shared_cols^T diag(w) shared_cols`` -- cheap
    since ``shared_dim`` is 1 or 4. The Gaussian prior (same scale on
    every parameter, `gate.PRIOR_VAR`) adds ``theta/prior_var_theta`` /
    ``shared/prior_var_shared`` to the two gradient pieces and
    ``1/prior_var_theta`` / ``diag(1/prior_var_shared)`` to ``D``/``C``
    respectively -- never touching ``B`` (a Gaussian prior is diagonal in
    the FULL parameter vector, and no diagonal entry of the ``P x P``
    identity falls in the theta/shared CROSS block), matching
    `_prior_penalty`/`_binary_logit_grad_hess_batch`'s convention exactly.
    """
    theta, shared = params[:, :k], params[:, k:]
    shared_dim = shared.shape[-1]
    logits = _kc_joint_arrow_logits_batch(theta, shared, logit_no_theta, slice_idx, shared_cols)
    p = torch.sigmoid(logits)
    mask_f = mask.to(p.dtype)
    r = mask_f * (p - y)  # (Bn, L)
    w = mask_f * p * (1.0 - p)  # (Bn, L)

    Bn = theta.shape[0]
    g_theta_data = torch.zeros_like(theta).scatter_add_(1, slice_idx, r)
    # `torch.matmul`, not `torch.einsum`: see `_kc_joint_arrow_logits_batch`'s
    # perf note (opt_einsum's path search dominates over the actual
    # contraction for these small shared_dim tensors).
    g_shared_data = torch.matmul(shared_cols.transpose(-2, -1), r.unsqueeze(-1)).squeeze(-1)

    D_data = torch.zeros_like(theta).scatter_add_(1, slice_idx, w)
    Bmat = torch.zeros(Bn, k, shared_dim, device=theta.device, dtype=theta.dtype)
    idx_exp = slice_idx.unsqueeze(-1).expand(-1, -1, shared_dim)
    Bmat.scatter_add_(1, idx_exp, w.unsqueeze(-1) * shared_cols)
    weighted_shared = shared_cols * w.unsqueeze(-1)
    C_data = torch.matmul(weighted_shared.transpose(-2, -1), shared_cols)

    prior_theta = prior_var[:k]
    prior_shared = prior_var[k:]
    g_theta = g_theta_data + theta / prior_theta
    g_shared = g_shared_data + shared / prior_shared
    D = D_data + 1.0 / prior_theta
    C = C_data + torch.diag(1.0 / prior_shared)

    return g_theta, g_shared, D, Bmat, C


def _arrow_schur_step(
    g_theta: Tensor,
    g_shared: Tensor,
    D: Tensor,
    Bmat: Tensor,
    C: Tensor,
    hessian_ridge: float,
) -> Tensor:
    """Exact block-elimination (Schur complement) solve of the arrow
    Hessian system ``[[diag(D), B], [B^T, C]] @ step = [g_theta,
    g_shared]`` for the Newton step, batched over the leading axis --
    PROVABLY IDENTICAL to assembling the dense ``(k + shared_dim, k +
    shared_dim)`` matrix and calling `torch.linalg.solve` (exact linear
    algebra: block elimination never approximates), verified directly in
    `tests/test_growth_newton.py`'s exactness suite. Cost is ``O(k *
    shared_dim^2 + shared_dim^3)`` -- linear in ``k`` -- versus the dense
    solve's ``O((k + shared_dim)^3)``, the actual fix for the 4th A4
    perf-surgery incident (oversized real KCs, k up to ~3000).

    ``hessian_ridge`` is added to BOTH diagonal blocks -- ``D``
    (elementwise) and ``C``'s own diagonal -- exactly matching ``H +
    hessian_ridge * eye`` restricted to an arrow-structured ``H``: the
    ``P x P`` identity has zero cross terms between the theta index range
    and the shared index range, so ridge never touches the border ``B``.

    Standard block elimination: with ``Dr = D + ridge`` (diagonal, so
    ``Dr^-1`` is elementwise) and ``Cr = C + ridge * I``, the Schur
    complement of the ``D`` block is ``Cr - B^T Dr^-1 B``; solving that
    small ``shared_dim x shared_dim`` system gives ``step_shared``, and
    back-substitution gives ``step_theta = Dr^-1 (g_theta - B
    step_shared)``. Returns the FULL step ``(Bn, k + shared_dim)`` in the
    same ``[theta, shared]`` column order as the dense design (module
    docstring note 6), ready to hand to `_run_newton_loop`'s shared
    damping/backtracking/convergence code unchanged.
    """
    # `torch.matmul`, not `torch.einsum`, throughout this function: see
    # `_kc_joint_arrow_logits_batch`'s perf note. This solve runs once per
    # Newton iteration per (KC, model, replicate-chunk) -- tens of
    # thousands of calls per permutation-battery chunk at production
    # scale -- and `opt_einsum`'s per-call contraction-path search
    # (triggered by any 3-operand `torch.einsum`) measured as the single
    # largest cost in the whole battery once the O(n_slices) Python-loop
    # assembly bottleneck (module docstring note 6's sibling fix in
    # gate.py) was removed: 12+ seconds of pure path-search bookkeeping
    # for tensors this small, confirmed by direct cProfile at production
    # scale (2026-07-20 4th A4 perf-surgery phase 3).
    shared_dim = g_shared.shape[-1]
    D_r = D + hessian_ridge  # (Bn, k)
    Dinv = 1.0 / D_r
    eye_shared = torch.eye(shared_dim, device=D.device, dtype=D.dtype)
    C_r = C + hessian_ridge * eye_shared  # (Bn, shared_dim, shared_dim)

    Bmat_scaled = Bmat * Dinv.unsqueeze(-1)  # (Bn, k, shared_dim)
    schur = C_r - torch.matmul(Bmat_scaled.transpose(-2, -1), Bmat)  # (Bn, shared_dim, shared_dim)
    weighted_theta = Dinv * g_theta  # (Bn, k)
    rhs_term = torch.matmul(Bmat.transpose(-2, -1), weighted_theta.unsqueeze(-1)).squeeze(-1)  # (Bn, shared_dim)
    rhs_shared = g_shared - rhs_term

    if shared_dim == 1:
        # M1a-pooled's shared block (beta_c) is a SCALAR: the "schur
        # system" is a 1x1 matrix, i.e. an ordinary division -- solving it
        # via `torch.linalg.solve` is mathematically correct but pays a
        # full batched-cuSOLVER kernel launch for what is exactly
        # `rhs_shared / schur`. `torch.linalg.solve` is confirmed (direct
        # cProfile at production scale, 2026-07-20 4th A4 perf-surgery
        # phase 3) as the single largest remaining cost once per-KC
        # dispatch overhead was fixed by KC-bucketing (gate.py's
        # `_fit_kc_bucket_pooled_and_held_out_vectorized`) -- HALF of
        # every arrow fit in the permutation battery is M1a-pooled
        # (shared_dim=1), so this shortcut is exact (elementwise division
        # IS the unique solution of a 1x1 linear system) and removes half
        # of all `torch.linalg.solve` calls outright.
        step_shared = rhs_shared / schur.squeeze(-1)  # (Bn, 1)
    else:
        step_shared = torch.linalg.solve(schur, rhs_shared.unsqueeze(-1)).squeeze(-1)  # (Bn, shared_dim)
    border_term = torch.matmul(Bmat, step_shared.unsqueeze(-1)).squeeze(-1)  # (Bn, k)
    step_theta = Dinv * (g_theta - border_term)

    return torch.cat([step_theta, step_shared], dim=-1)


def penalized_bounded_newton(
    nll_fn: Callable[..., Tensor],
    x0: Tensor,
    data_args: Sequence[Tensor] = (),
    prior_var: Tensor | float = 4.0,
    step_clip: float = 1.0,
    max_iter: int = 25,
    max_backtrack: int = 10,
    tol: float = 1e-6,
    hessian_ridge: float = 1e-6,
    analytic: bool | None = None,
) -> NewtonResult:
    """Batched penalized Newton-Raphson (design section 2.2's numerical
    safeguard, shared by every slice-level fit in the A4 pipeline).

    ``nll_fn(params, *data_args_i)`` maps ONE slice's parameter vector
    ``(P,)`` plus that slice's own data tensors (each with leading dim
    stripped by ``vmap``, e.g. a ``(T_max,)`` response sequence out of a
    batched ``(S, T_max)`` tensor) to that slice's scalar data NLL --
    excluding the Gaussian prior, which this function adds. ``x0`` is
    ``(S, P)``: the per-slice initial parameters (e.g. zeros). Returns a
    ``NewtonResult`` batched over ``S``.

    ``prior_var`` is the Gaussian prior VARIANCE (design's ``N(0, 2.0^2)``
    prior on theta_ic/beta/block offsets/rate-stage deviations means
    ``prior_var=4.0``, the default), either a scalar or a ``(P,)`` tensor
    if different parameters carry different prior scales (e.g. a mean-
    reversion asymptote deviation prior differing from a rate prior).

    ``analytic`` selects the closed-form fast path (module docstring note
    1): ``None`` (default) auto-dispatches to it iff ``nll_fn is
    binary_logit_nll`` (identity, not duck-typing -- any wrapper around
    `binary_logit_nll`, e.g. a closure that rearranges its arguments,
    falls through to the generic path, which stays correct for it
    regardless). ``True``/``False`` force the analytic/generic path and
    raise ``ValueError`` if ``analytic=True`` is forced for any other
    ``nll_fn`` (the closed form is only valid for `binary_logit_nll`'s own
    ``data_args`` signature ``(y, mask, logit_no_theta, design)``). The
    generic path remains fully general and is this module's equivalence
    reference for the analytic path (`tests/test_growth_newton.py`).

    A THIRD path dispatches by identity on ``nll_fn is binary_logit_nll_arrow``
    (module docstring note 6): the KC-joint arrow-structured fast path,
    which consumes ``data_args = (y, mask, logit_no_theta, slice_idx,
    shared_cols, k)`` (the compact representation, not a materialized
    one-hot design) and solves each Newton step via exact block
    elimination (`_arrow_schur_step`) instead of a dense ``torch.linalg.
    solve``. This path has no generic-`torch.func` counterpart -- its
    equivalence reference is the dense `binary_logit_nll` path on the
    literal one-hot design (`gate.py`'s ``use_arrow=False``) -- so
    ``analytic`` is not meaningful for it; passing ``analytic=False``
    alongside the arrow marker raises ``ValueError``.
    """
    device = x0.device
    dtype = x0.dtype
    S, P = x0.shape
    if not torch.is_tensor(prior_var):
        prior_var_t = torch.full((P,), float(prior_var), device=device, dtype=dtype)
    else:
        prior_var_t = prior_var.to(device=device, dtype=dtype)
        if prior_var_t.ndim == 0:
            prior_var_t = prior_var_t.expand(P)

    if nll_fn is binary_logit_nll_arrow:
        if analytic is False:
            raise ValueError(
                "penalized_bounded_newton(analytic=False) has no generic-path "
                "counterpart for binary_logit_nll_arrow -- the arrow fast path "
                "IS the only implementation for its compact (slice_idx, "
                "shared_cols) data_args signature. Use the dense "
                "binary_logit_nll marker with an explicit one-hot design "
                "(analytic=False) as the generic-path equivalence reference "
                "instead (see gate.py's use_arrow=False)."
            )
        _dispatch_counts["arrow"] += 1
        y, mask, logit_no_theta, slice_idx, shared_cols, k = data_args

        def step_fn(params: Tensor) -> Tensor:
            g_theta, g_shared, D, Bmat, C = _kc_joint_arrow_grad_hess_batch(
                params, y, mask, logit_no_theta, slice_idx, shared_cols, k, prior_var_t
            )
            return _arrow_schur_step(g_theta, g_shared, D, Bmat, C, hessian_ridge)

        def obj_fn(params: Tensor) -> Tensor:
            return _kc_joint_arrow_data_nll_batch(
                params, y, mask, logit_no_theta, slice_idx, shared_cols, k
            ) + _prior_penalty(params, prior_var_t)

        def data_nll_fn(params: Tensor) -> Tensor:
            return _kc_joint_arrow_data_nll_batch(params, y, mask, logit_no_theta, slice_idx, shared_cols, k)

        return _run_newton_loop(step_fn, obj_fn, data_nll_fn, x0, step_clip, max_iter, max_backtrack, tol)

    use_analytic = (nll_fn is binary_logit_nll) if analytic is None else analytic
    if use_analytic and nll_fn is not binary_logit_nll:
        raise ValueError(
            "penalized_bounded_newton(analytic=True) requires nll_fn is "
            "binary_logit_nll -- the closed-form fast path only matches "
            "that objective's data_args signature (y, mask, logit_no_theta, "
            "design). Pass analytic=False (or omit analytic) for any other "
            "nll_fn."
        )
    _dispatch_counts["analytic" if use_analytic else "generic"] += 1

    eye = torch.eye(P, device=device, dtype=dtype)

    if use_analytic:
        y, mask, logit_no_theta, design = data_args

        def step_fn(params: Tensor) -> Tensor:
            g, H = _binary_logit_grad_hess_batch(params, y, mask, logit_no_theta, design, prior_var_t)
            H = H + hessian_ridge * eye
            return torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)

        def obj_fn(params: Tensor) -> Tensor:
            return _binary_logit_data_nll_batch(params, y, mask, logit_no_theta, design) + _prior_penalty(
                params, prior_var_t
            )

        def data_nll_fn(params: Tensor) -> Tensor:
            return _binary_logit_data_nll_batch(params, y, mask, logit_no_theta, design)

    else:
        n_data_args = len(data_args)
        in_dims = (0,) + (0,) * n_data_args

        def objective(params: Tensor, *data: Tensor) -> Tensor:
            return nll_fn(params, *data) + _prior_penalty(params, prior_var_t)

        grad_fn_v = torch.func.vmap(torch.func.grad(objective), in_dims=in_dims)
        hess_fn_v = torch.func.vmap(torch.func.hessian(objective), in_dims=in_dims)
        obj_fn_v = torch.func.vmap(objective, in_dims=in_dims)
        data_nll_fn_v = torch.func.vmap(nll_fn, in_dims=in_dims)

        def step_fn(params: Tensor) -> Tensor:
            g = grad_fn_v(params, *data_args)
            H = hess_fn_v(params, *data_args)
            H = H + hessian_ridge * eye
            return torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)

        def obj_fn(params: Tensor) -> Tensor:
            return obj_fn_v(params, *data_args)

        def data_nll_fn(params: Tensor) -> Tensor:
            return data_nll_fn_v(params, *data_args)

    return _run_newton_loop(step_fn, obj_fn, data_nll_fn, x0, step_clip, max_iter, max_backtrack, tol)


def binary_logit_nll(params: Tensor, y: Tensor, mask: Tensor, logit_no_theta: Tensor, design: Tensor) -> Tensor:
    """Reference per-slice NLL for a linear-in-parameters binary model,
    provided as a ready-made ``nll_fn`` for the common GLM case (M0/M1a/
    M1b are all instances of this shape): ``logit = design @ params +
    logit_no_theta``, Bernoulli NLL summed over the slice's (masked)
    positions.

    ``params`` (P,), ``y``/``mask``/``logit_no_theta`` (T,), ``design``
    (T, P). ``logit_no_theta`` carries any fixed per-position offset that
    does not multiply a free parameter (e.g. ``-b_j``, the frozen item
    difficulty). Masked-out positions (``mask`` False, e.g. padding or
    excluded uncalibrated items) contribute zero.
    """
    logits = design @ params + logit_no_theta
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    )
    return (per_pos * mask.to(per_pos.dtype)).sum()


def binary_logit_nll_arrow(
    params: Tensor,
    y: Tensor,
    mask: Tensor,
    logit_no_theta: Tensor,
    slice_idx: Tensor,
    shared_cols: Tensor,
    k: int,
) -> Tensor:
    """Reference per-KC NLL for the arrow-structured KC-joint model
    (module docstring note 6): mathematically identical to
    `binary_logit_nll` with ``design = concat(one_hot(slice_idx, k),
    shared_cols)`` and ``params = concat(theta, shared)``, but taking the
    compact ``(slice_idx, shared_cols)`` representation instead of a
    materialized ``(T, k + shared_dim)`` one-hot-bordered design matrix.

    ``params`` (k + shared_dim,), ``y``/``mask``/``logit_no_theta`` (T,),
    ``slice_idx`` (T,) long in ``[0, k)`` (which of the KC's ``k`` slices
    each position belongs to), ``shared_cols`` (T, shared_dim). Used as
    the identity-dispatch marker for `penalized_bounded_newton`'s arrow
    fast path (``nll_fn is binary_logit_nll_arrow``); also directly
    callable (as here) for a plain, non-batched sanity check or as a
    single-element reference.
    """
    theta, shared = params[:k], params[k:]
    logits = theta[slice_idx] + shared_cols @ shared + logit_no_theta
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    )
    return (per_pos * mask.to(per_pos.dtype)).sum()
