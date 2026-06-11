"""``FrozenMAGPCM`` tests, verifying the freeze contract.

The tests exercise the wrapper against a tiny MA-GPCM instance from
``ma-irt``. They do not require a trained checkpoint, the freeze
contract is structural rather than weights-dependent.
"""

from __future__ import annotations

import pytest
import torch

from ordrec.envs.frozen_magpcm import FrozenMAGPCM, MAIRTOutput, freeze_magpcm


pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


def _build_tiny_magpcm():
    """Build a tiny MAGPCM instance from ma-irt for unit tests."""
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    return MAGPCM(
        n_questions=16,
        n_categories=4,
        n_traits=1,
        memory_size=8,
        key_dim=8,
        value_dim=8,
        summary_dim=8,
        embedding_type="learned",
        dropout_rate=0.1,  # non-zero so eval-vs-train would diverge
        ability_scale=1.0,
        separate_theta=True,
        init_value_memory=False,
    )


def _rand_inputs(seed: int = 0, B: int = 2, S: int = 6):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    q = torch.randint(1, 17, size=(B, S), generator=g, dtype=torch.long)
    r = torch.randint(0, 4, size=(B, S), generator=g, dtype=torch.long)
    return q, r


def test_freeze_magpcm_helper_sets_eval_and_no_grad() -> None:
    m = _build_tiny_magpcm()
    # By default a fresh nn.Module is in training mode and parameters
    # require grad.
    assert m.training is True
    assert any(p.requires_grad for p in m.parameters())

    freeze_magpcm(m)
    assert m.training is False
    assert all(not p.requires_grad for p in m.parameters())


def test_forward_no_grad_returns_five_key_dict() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    q, r = _rand_inputs()
    out = frozen.forward_no_grad(q, r)
    assert set(out.keys()) == {"logits", "probs", "theta", "alpha", "beta"}
    B, S = q.shape
    K = frozen.n_categories
    assert out["logits"].shape == (B, S, K)
    assert out["probs"].shape == (B, S, K)


def test_forward_alias_dispatches_to_no_grad() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    q, r = _rand_inputs()
    out1 = frozen.forward_no_grad(q, r)
    out2 = frozen(q, r)
    assert torch.equal(out1["logits"], out2["logits"])
    # And neither path should carry grad.
    assert not out1["logits"].requires_grad
    assert not out2["logits"].requires_grad


def test_train_keeps_underlying_model_in_eval() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    # Caller flips the wrapper to train(); the underlying model must
    # stay in eval to honour the freeze invariant.
    frozen.train(True)
    assert frozen.model.training is False
    frozen.train(False)
    assert frozen.model.training is False


def test_no_grad_invariance_across_repeated_calls() -> None:
    """Identical (q, r) yield identical logits across repeated forwards."""
    from ordrec.envs.bench_forward import no_grad_invariance_check

    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    q, r = _rand_inputs()
    report = no_grad_invariance_check(frozen, q, r, n_calls=4, atol=0.0)
    assert report["passed"] is True
    assert report["max_abs_diff"] == 0.0
    assert report["n_pairs"] == 3


def test_theta_invariant_to_future_positions_under_eval_no_grad() -> None:
    """``theta[:, t, :]`` for ``t < S - 1`` must not change when we
    extend the sequence past S.

    This is the v2 invariance claim that replaces the M1-era parity
    test. See ``docs/exrec_ordinal_plan.md`` Section 4.
    """
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    q, r = _rand_inputs(seed=1, B=1, S=6)
    out_short = frozen.forward_no_grad(q, r)
    theta_short = out_short["theta"]  # (1, 6, D)

    # Extend by two extra positions and confirm the first six positions'
    # theta is unchanged.
    q_ext = torch.cat([q, torch.tensor([[3, 11]])], dim=1)
    r_ext = torch.cat([r, torch.tensor([[1, 2]])], dim=1)
    out_ext = frozen.forward_no_grad(q_ext, r_ext)
    theta_ext_prefix = out_ext["theta"][:, :6, :]
    assert torch.allclose(theta_short, theta_ext_prefix, atol=1e-6, rtol=0.0)


def test_grad_does_not_flow_into_frozen_model() -> None:
    """No autograd tape ever spans the frozen wrapper."""
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    q, r = _rand_inputs()
    out = frozen.forward_no_grad(q, r)
    # The output tensor must not require grad: requires_grad_(False)
    # plus no_grad together guarantee this.
    assert not out["logits"].requires_grad
    # And attempting to backprop would fail; we verify that no leaf
    # parameter has a grad after the forward.
    for p in frozen.model.parameters():
        assert p.grad is None


def test_wrapper_exposes_device_property() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    # The model was built on cpu by default; the wrapper should report cpu.
    assert frozen.device.type == "cpu"


def test_wrapper_rejects_non_module() -> None:
    with pytest.raises(TypeError):
        FrozenMAGPCM("not-a-module")  # type: ignore[arg-type]


def test_n_questions_and_categories_propagate_from_model() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    assert frozen.n_questions == 16
    assert frozen.n_categories == 4


# ---------------------------------------------------------------------------
# B3 MAIRTOutput TypedDict tests
# ---------------------------------------------------------------------------


def test_forward_no_grad_returns_mairt_output_type() -> None:
    """forward_no_grad returns a MAIRTOutput TypedDict with all five keys."""
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    q = torch.randint(1, 17, (2, 5))
    r = torch.randint(0, 4, (2, 5))
    out = frozen.forward_no_grad(q, r)
    for key in ("logits", "probs", "theta", "alpha", "beta"):
        assert key in out, f"Missing key '{key}' in MAIRTOutput"
    # probs must sum to 1 along last dim
    assert torch.allclose(out["probs"].sum(-1), torch.ones(2, 5), atol=1e-5)


def test_mairt_output_missing_key_raises_assertion() -> None:
    """forward_no_grad raises AssertionError when world model omits a key."""
    import torch.nn as nn

    class _BrokenModel(nn.Module):
        """Returns a dict without 'beta'."""
        def forward(self, q, r):
            B, S = q.shape
            return {
                "logits": torch.zeros(B, S, 4),
                "probs": torch.zeros(B, S, 4),
                "theta": torch.zeros(B, S, 1),
                "alpha": torch.zeros(B, S, 1),
                # beta is missing
            }

    broken = _BrokenModel()
    frozen = FrozenMAGPCM(broken)
    q = torch.randint(1, 5, (2, 3))
    r = torch.randint(0, 4, (2, 3))
    with pytest.raises(AssertionError, match="missing required keys"):
        frozen.forward_no_grad(q, r)
