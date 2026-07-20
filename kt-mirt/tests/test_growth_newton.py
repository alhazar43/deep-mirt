"""Tests for `kt_mirt.growth.newton`: the shared penalized bounded Newton
primitive (design section 2.2, `_planning/design/a4_design.md` v1.1).

`test_separation_all_correct_and_all_incorrect_slices_stay_finite` is the
design's own explicitly pre-registered unit test ("A unit test asserts
finite parameters and finite held-out NLL on an all-correct and an
all-incorrect slice").
"""

import pytest
import torch

from kt_mirt.growth import newton as newton_mod
from kt_mirt.growth.newton import binary_logit_nll, penalized_bounded_newton


def _make_constant_slice_data(y_rows, b_rows=None):
    """Build (y, mask, design, b) tensors for a batch of constant-model
    (M0-shaped, P=1) slices, each row a full-length response sequence
    (all rows same length here for simplicity; mask marks real positions)."""
    T = max(len(y) for y in y_rows)
    S = len(y_rows)
    y = torch.zeros(S, T)
    mask = torch.zeros(S, T)
    b = torch.zeros(S, T)
    design = torch.ones(S, T, 1)
    for i, seq in enumerate(y_rows):
        y[i, : len(seq)] = torch.tensor(seq, dtype=torch.float32)
        mask[i, : len(seq)] = 1.0
        if b_rows is not None:
            b[i, : len(seq)] = torch.tensor(b_rows[i][: len(seq)], dtype=torch.float32)
    return y, mask, design, b


def _m0_nll_fn(params, y_i, mask_i, design_i, b_i):
    return binary_logit_nll(params, y_i, mask_i, -b_i, design_i)


def test_separation_all_correct_and_all_incorrect_slices_stay_finite():
    """Design's pre-registered unit test (section 2.2, R2-B3): finite MAP
    parameters and finite held-out NLL on an all-correct and an
    all-incorrect slice."""
    y_rows = [[1.0] * 10, [0.0] * 10, [1.0, 0.0] * 5]
    y, mask, design, b = _make_constant_slice_data(y_rows)
    x0 = torch.zeros(3, 1)

    res = penalized_bounded_newton(
        _m0_nll_fn, x0, data_args=(y, mask, design, b), prior_var=4.0
    )

    assert torch.isfinite(res.params).all()
    assert torch.isfinite(res.data_nll).all()
    assert torch.isfinite(res.penalized_nll).all()
    # All-correct slice: MAP theta pulled positive (finite, not diverging to +inf).
    assert res.params[0, 0] > 0
    # All-incorrect slice: MAP theta pulled negative.
    assert res.params[1, 0] < 0
    # Mixed, balanced slice: MAP theta near 0.
    assert abs(float(res.params[2, 0])) < 0.5

    # Held-out NLL on a fresh (interpolative) all-correct sequence at the
    # fitted theta must also be finite.
    held_out_y = torch.ones(3, 5)
    held_out_mask = torch.ones(3, 5)
    held_out_b = torch.zeros(3, 5)
    held_out_design = torch.ones(3, 5, 1)
    held_out_nll = binary_logit_nll(
        res.params[0], held_out_y[0], held_out_mask[0], -held_out_b[0], held_out_design[0]
    )
    assert torch.isfinite(held_out_nll)


def test_converges_within_iteration_budget():
    y_rows = [[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]] * 4
    y, mask, design, b = _make_constant_slice_data(y_rows)
    x0 = torch.zeros(4, 1)
    res = penalized_bounded_newton(
        _m0_nll_fn, x0, data_args=(y, mask, design, b), prior_var=4.0, max_iter=25
    )
    assert bool(res.converged.all())
    assert bool((res.n_iter <= 25).all())


def test_recovers_known_intercept_close_to_unpenalized_mle():
    """A slice with plenty of balanced data and a real signal should
    recover close to the (very lightly penalized) MLE."""
    torch.manual_seed(0)
    T = 200
    true_theta = 1.3
    b_const = 0.0
    p = torch.sigmoid(torch.tensor(true_theta - b_const))
    y_seq = (torch.rand(T) < p).float()
    y, mask, design, b = _make_constant_slice_data([y_seq.tolist()])
    x0 = torch.zeros(1, 1)
    # Wide prior (prior_var large) so the MAP estimate is close to the MLE.
    res = penalized_bounded_newton(
        _m0_nll_fn, x0, data_args=(y, mask, design, b), prior_var=100.0
    )
    assert torch.isfinite(res.params).all()
    assert abs(float(res.params[0, 0]) - true_theta) < 0.35


def test_multi_parameter_linear_trend_slice():
    """A P=2 design (M1a-shaped: intercept + linear trend in opportunity)
    exercises the generic multi-parameter path, not just the P=1 case."""
    torch.manual_seed(1)
    T = 40
    n = torch.arange(1, T + 1, dtype=torch.float32)
    true_theta0, true_beta = 0.5, 0.05
    logits = true_theta0 + true_beta * (n - 1)
    y_seq = (torch.rand(T) < torch.sigmoid(logits)).float()

    y = y_seq.unsqueeze(0)
    mask = torch.ones(1, T)
    b = torch.zeros(1, T)
    design = torch.stack([torch.ones(T), n - 1.0], dim=-1).unsqueeze(0)  # (1, T, 2)
    x0 = torch.zeros(1, 2)

    res = penalized_bounded_newton(
        _m0_nll_fn, x0, data_args=(y, mask, design, b), prior_var=4.0
    )
    assert torch.isfinite(res.params).all()
    assert torch.isfinite(res.data_nll).all()


def test_backtracking_handles_a_bad_initial_point():
    """Starting far from the optimum (where a raw Newton step would badly
    overshoot) must still land at a finite, sensible MAP point via
    damping/backtracking."""
    y_rows = [[1.0] * 8]
    y, mask, design, b = _make_constant_slice_data(y_rows)
    x0 = torch.tensor([[8.0]])  # deliberately far from the optimum
    res = penalized_bounded_newton(
        _m0_nll_fn, x0, data_args=(y, mask, design, b), prior_var=4.0, step_clip=1.0
    )
    assert torch.isfinite(res.params).all()
    assert torch.isfinite(res.penalized_nll).all()


# =============================================================================
# Analytic fast path (A4 perf-surgery, third pass): closed-form grad/Hessian
# for `binary_logit_nll`, verified against the `torch.func` generic path it
# replaces in the hot loop. `binary_logit_nll`'s own data_args order is
# (y, mask, logit_no_theta, design), matching `gate.py`'s call sites exactly
# -- this is the shape the analytic fast path dispatches on.
# =============================================================================


def _make_random_binary_problem(S, T, P, seed, mask_frac=0.0):
    """A random-but-consistent (well-posed) penalized logistic problem:
    `design`/`logit_no_theta` generate `y` from a real Bernoulli model (not
    pure noise), so the Hessian is well-conditioned near the truth, and
    `mask` marks a `mask_frac` fraction of positions as excluded (never the
    first position of any row, so no slice degenerates to zero valid data)."""
    g = torch.Generator().manual_seed(seed)
    design = torch.randn(S, T, P, generator=g)
    true_params = torch.randn(S, P, generator=g) * 0.5
    logit_no_theta = torch.randn(S, T, generator=g) * 0.3
    logits = torch.einsum("stp,sp->st", design, true_params) + logit_no_theta
    probs = torch.sigmoid(logits)
    y = (torch.rand(S, T, generator=g) < probs).float()
    if mask_frac > 0:
        mask = torch.rand(S, T, generator=g) >= mask_frac
        mask[:, 0] = True  # never let a whole row go empty
    else:
        mask = torch.ones(S, T, dtype=torch.bool)
    return design, y, mask, logit_no_theta


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("P", [1, 2, 4, 7])
def test_analytic_grad_hessian_matches_torch_func_reference(P, masked):
    """Ladder rung 1: the closed-form batched grad/Hessian
    (`newton._binary_logit_grad_hess_batch`) must match `torch.func`'s
    autodiff (`vmap(grad)` / `vmap(hessian)`) of the SAME penalized
    objective, on random small problems, at multiple P, with and without
    masking."""
    S, T = 6, 15
    design, y, mask, logit_no_theta = _make_random_binary_problem(
        S, T, P, seed=1000 + 10 * P + int(masked), mask_frac=0.3 if masked else 0.0
    )
    g = torch.Generator().manual_seed(2000 + P)
    params = torch.randn(S, P, generator=g) * 0.7
    prior_var_t = torch.full((P,), 4.0)

    def objective(p, y_i, mask_i, logit_i, design_i):
        return binary_logit_nll(p, y_i, mask_i, logit_i, design_i) + newton_mod._prior_penalty(p, prior_var_t)

    ref_grad = torch.func.vmap(torch.func.grad(objective))(params, y, mask, logit_no_theta, design)
    ref_hess = torch.func.vmap(torch.func.hessian(objective))(params, y, mask, logit_no_theta, design)

    ana_grad, ana_hess = newton_mod._binary_logit_grad_hess_batch(
        params, y, mask, logit_no_theta, design, prior_var_t
    )

    assert torch.allclose(ana_grad, ref_grad, rtol=1e-5, atol=1e-6), \
        f"grad mismatch P={P} masked={masked}: max abs diff {(ana_grad - ref_grad).abs().max().item():.3e}"
    assert torch.allclose(ana_hess, ref_hess, rtol=1e-5, atol=1e-6), \
        f"hessian mismatch P={P} masked={masked}: max abs diff {(ana_hess - ref_hess).abs().max().item():.3e}"


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("P", [1, 2, 5])
def test_analytic_path_reproduces_generic_path_iterates(P, masked):
    """Ladder rung 2 (trajectory equivalence): `penalized_bounded_newton`'s
    analytic fast path must reproduce the generic `torch.func` path's
    PARAMETER ITERATES at every outer iteration -- checked by replaying
    both paths from the SAME x0 at every partial iteration budget (1..8)
    and comparing the resulting params directly (not just the final
    answer), which is what actually exercises "same damping decisions": a
    genuinely WRONG backtracking/acceptance branch would show up as an
    order-of-magnitude jump here, not a small numeric wobble.

    What is NOT asserted, and why: exact equality of `n_iter`/`converged`
    at every partial budget. Empirically (see the diagnostic run this test
    is derived from), the two paths' iterates agree to ~1e-7 early and
    drift to at most ~1e-4 (absolute, on O(0.1-2) params) by iteration 8-25
    -- float32 noise from two DIFFERENT numerical mechanisms (batched
    einsum vs. autodiff) computing the same closed-form quantity, exactly
    analogous to the batched-vs-looped linalg noise this module's sibling
    `test_growth_gate_perf_equivalence.py` documents and tolerates for the
    same reason. That noise occasionally lands within `tol=1e-6` of a
    slice's OWN convergence boundary and flips which side of the boundary
    it lands on, changing `converged`/`n_iter` for that slice by a step or
    two without the underlying iterate having diverged at all -- asserting
    boolean-flag equality would make the test flaky on exactly the cases
    where the numerics are working as intended (both paths landing at
    essentially the same fixed point). The params comparison below is the
    one that actually detects a logic bug.
    """
    S, T = 5, 12
    design, y, mask, logit_no_theta = _make_random_binary_problem(
        S, T, P, seed=3000 + 10 * P + int(masked), mask_frac=0.2 if masked else 0.0
    )
    g = torch.Generator().manual_seed(4000 + P)
    x0 = torch.randn(S, P, generator=g) * 2.0  # off-center start: exercises backtracking

    for max_iter in range(1, 9):
        analytic_res = penalized_bounded_newton(
            binary_logit_nll, x0.clone(), data_args=(y, mask, logit_no_theta, design),
            prior_var=4.0, max_iter=max_iter, analytic=True,
        )
        generic_res = penalized_bounded_newton(
            binary_logit_nll, x0.clone(), data_args=(y, mask, logit_no_theta, design),
            prior_var=4.0, max_iter=max_iter, analytic=False,
        )
        assert torch.allclose(analytic_res.params, generic_res.params, rtol=2e-4, atol=5e-4), (
            f"P={P} masked={masked} max_iter={max_iter}: params diverged far beyond the float32 "
            f"noise floor (max abs diff {(analytic_res.params - generic_res.params).abs().max().item():.3e})"
        )


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("P", [1, 2, 5])
def test_analytic_path_matches_generic_path_at_default_convergence(P, masked):
    """At the production default (`max_iter=25`, the design's own cap),
    both paths have had ample budget to settle near the SAME convex
    objective's unique MAP optimum, so -- unlike the partial-budget replay
    above, which can catch a slice mid-iteration on either side of a
    knife-edge convergence tie -- the `converged` flags themselves should
    agree here, and params should agree tightly."""
    S, T = 5, 12
    design, y, mask, logit_no_theta = _make_random_binary_problem(
        S, T, P, seed=5000 + 10 * P + int(masked), mask_frac=0.2 if masked else 0.0
    )
    g = torch.Generator().manual_seed(6000 + P)
    x0 = torch.randn(S, P, generator=g) * 2.0

    analytic_res = penalized_bounded_newton(
        binary_logit_nll, x0.clone(), data_args=(y, mask, logit_no_theta, design), prior_var=4.0, analytic=True
    )
    generic_res = penalized_bounded_newton(
        binary_logit_nll, x0.clone(), data_args=(y, mask, logit_no_theta, design), prior_var=4.0, analytic=False
    )
    assert torch.equal(analytic_res.converged, generic_res.converged), (
        f"P={P} masked={masked}: converged flags disagree at the default 25-iteration budget "
        f"(analytic={analytic_res.converged.tolist()} generic={generic_res.converged.tolist()})"
    )
    assert torch.allclose(analytic_res.params, generic_res.params, rtol=2e-4, atol=5e-4)


def test_default_analytic_dispatch_matches_explicit_for_binary_logit_nll():
    """`analytic=None` (the default) must auto-dispatch to the analytic
    path -- bit-identically -- whenever `nll_fn is binary_logit_nll`."""
    S, T, P = 4, 10, 2
    design, y, mask, logit_no_theta = _make_random_binary_problem(S, T, P, seed=42)
    x0 = torch.zeros(S, P)

    default_res = penalized_bounded_newton(
        binary_logit_nll, x0.clone(), data_args=(y, mask, logit_no_theta, design), prior_var=4.0
    )
    explicit_res = penalized_bounded_newton(
        binary_logit_nll, x0.clone(), data_args=(y, mask, logit_no_theta, design), prior_var=4.0, analytic=True
    )
    assert torch.equal(default_res.params, explicit_res.params)
    assert torch.equal(default_res.n_iter, explicit_res.n_iter)
    assert torch.equal(default_res.converged, explicit_res.converged)


def test_default_dispatch_stays_generic_for_a_binary_logit_nll_wrapper():
    """A wrapper closure around `binary_logit_nll` (e.g. one that
    rearranges argument order, like this module's own `_m0_nll_fn`) is NOT
    `binary_logit_nll` by identity, so it must keep using the generic path
    automatically -- the analytic path is never silently applied to an
    objective whose data_args shape it hasn't verified."""
    y_rows = [[1.0, 0.0] * 5]
    y, mask, design, b = _make_constant_slice_data(y_rows)
    x0 = torch.zeros(1, 1)
    # Must not raise, and must reach the generic path with no error despite
    # `_m0_nll_fn`'s data_args not matching binary_logit_nll's own signature.
    res = penalized_bounded_newton(_m0_nll_fn, x0, data_args=(y, mask, design, b), prior_var=4.0)
    assert torch.isfinite(res.params).all()


def test_analytic_true_forced_on_non_binary_logit_nll_raises():
    """Forcing `analytic=True` on any `nll_fn` other than `binary_logit_nll`
    itself must fail loudly (the closed form is only valid for that one
    objective's data_args signature) rather than silently compute the
    wrong gradient/Hessian for a mismatched signature."""
    y_rows = [[1.0, 0.0] * 5]
    y, mask, design, b = _make_constant_slice_data(y_rows)
    x0 = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        penalized_bounded_newton(_m0_nll_fn, x0, data_args=(y, mask, design, b), prior_var=4.0, analytic=True)
