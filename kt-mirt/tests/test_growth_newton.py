"""Tests for `kt_mirt.growth.newton`: the shared penalized bounded Newton
primitive (design section 2.2, `_planning/design/a4_design.md` v1.1).

`test_separation_all_correct_and_all_incorrect_slices_stay_finite` is the
design's own explicitly pre-registered unit test ("A unit test asserts
finite parameters and finite held-out NLL on an all-correct and an
all-incorrect slice").
"""

import torch

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
