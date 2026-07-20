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
_dispatch_counts = {"analytic": 0, "generic": 0}


def dispatch_counts() -> dict[str, int]:
    """Returns a COPY of the process-wide {"analytic": n, "generic": n}
    call counts accumulated by `penalized_bounded_newton` since the last
    `reset_dispatch_counts()` (or process start)."""
    return dict(_dispatch_counts)


def reset_dispatch_counts() -> None:
    _dispatch_counts["analytic"] = 0
    _dispatch_counts["generic"] = 0


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
    grad_hess_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    obj_fn: Callable[[Tensor], Tensor],
    data_nll_fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    step_clip: float,
    max_iter: int,
    max_backtrack: int,
    tol: float,
    hessian_ridge: float,
) -> NewtonResult:
    """The damped, bounded, per-slice-independent Newton loop itself
    (design section 2.2), factored out of `penalized_bounded_newton` so
    BOTH the generic `torch.func` path and the `binary_logit_nll` analytic
    fast path run through the exact same damping/backtracking/step-clamp/
    convergence code -- the two paths differ ONLY in how ``grad_hess_fn``/
    ``obj_fn``/``data_nll_fn`` compute their values, never in the control
    flow that consumes them. ``grad_hess_fn(x)`` returns the penalized
    objective's ``(grad, hessian)`` (ridge NOT yet added -- added once,
    here, uniformly for both paths); ``obj_fn``/``data_nll_fn`` return the
    penalized objective / unpenalized data NLL respectively, each ``(S,)``.
    """
    device = x0.device
    dtype = x0.dtype
    S, P = x0.shape

    x = x0.clone()
    converged = torch.zeros(S, dtype=torch.bool, device=device)
    n_iter = torch.zeros(S, dtype=torch.long, device=device)
    obj_val = obj_fn(x)
    eye = torch.eye(P, device=device, dtype=dtype)

    for _ in range(max_iter):
        active = ~converged
        if not bool(active.any()):
            break

        g, H = grad_hess_fn(x)
        H = H + hessian_ridge * eye
        step = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
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

    if use_analytic:
        y, mask, logit_no_theta, design = data_args

        def grad_hess_fn(params: Tensor) -> tuple[Tensor, Tensor]:
            return _binary_logit_grad_hess_batch(params, y, mask, logit_no_theta, design, prior_var_t)

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

        def grad_hess_fn(params: Tensor) -> tuple[Tensor, Tensor]:
            return grad_fn_v(params, *data_args), hess_fn_v(params, *data_args)

        def obj_fn(params: Tensor) -> Tensor:
            return obj_fn_v(params, *data_args)

        def data_nll_fn(params: Tensor) -> Tensor:
            return data_nll_fn_v(params, *data_args)

    return _run_newton_loop(
        grad_hess_fn, obj_fn, data_nll_fn, x0, step_clip, max_iter, max_backtrack, tol, hessian_ridge
    )


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
