"""Test forward-pass output shapes for D=1 (GPCM) and D=3 (MIRT).

These tests verify Architecture Decision A1: theta and alpha are always
(B, S, D) regardless of D, and the multi-dimensional path is activated
purely by setting n_traits in the config / constructor.
"""

from __future__ import annotations

import pytest
import torch

from kt_gpcm.models.kt_gpcm import DeepGPCM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_model(n_traits: int, n_cats: int = 4) -> DeepGPCM:
    return DeepGPCM(
        n_questions=20,
        n_categories=n_cats,
        n_traits=n_traits,
        memory_size=8,
        key_dim=16,
        value_dim=32,
        summary_dim=16,
        dropout_rate=0.0,
        init_value_memory=True,
    )


def make_batch(B: int = 3, S: int = 10, Q: int = 20, K: int = 4):
    """Random (questions, responses) batch."""
    questions = torch.randint(1, Q + 1, (B, S))
    responses = torch.randint(0, K, (B, S))
    return questions, responses


# ---------------------------------------------------------------------------
# D = 1 (standard GPCM)
# ---------------------------------------------------------------------------


class TestShapesSingleTrait:
    """Output shapes for n_traits = 1."""

    B, S, K, D = 3, 10, 4, 1

    def setup_method(self):
        self.model = make_model(n_traits=self.D, n_cats=self.K)
        self.questions, self.responses = make_batch(self.B, self.S, Q=20, K=self.K)

    def test_output_is_dict(self):
        out = self.model(self.questions, self.responses)
        assert isinstance(out, dict)
        for key in ("theta", "alpha", "beta", "logits", "probs"):
            assert key in out, f"Missing key: {key}"

    def test_theta_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["theta"].shape == (self.B, self.S, self.D)

    def test_alpha_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["alpha"].shape == (self.B, self.S, self.D)

    def test_beta_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["beta"].shape == (self.B, self.S, self.K - 1)

    def test_logits_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["logits"].shape == (self.B, self.S, self.K)

    def test_probs_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["probs"].shape == (self.B, self.S, self.K)

    def test_probs_sum_to_one(self):
        out = self.model(self.questions, self.responses)
        sums = out["probs"].sum(dim=-1)  # (B, S)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_logit_baseline_zero(self):
        """Category-0 logit must always be 0 (GPCM cumulative baseline)."""
        out = self.model(self.questions, self.responses)
        assert torch.allclose(
            out["logits"][:, :, 0],
            torch.zeros(self.B, self.S),
            atol=1e-6,
        )

    def test_alpha_positive(self):
        out = self.model(self.questions, self.responses)
        assert (out["alpha"] > 0).all()

    def test_last_attention_stored(self):
        self.model(self.questions, self.responses)
        assert self.model.last_attention is not None
        assert self.model.last_attention.shape == (self.B, self.S, self.model.memory_size)

    def test_attention_sums_to_one(self):
        self.model(self.questions, self.responses)
        attn_sums = self.model.last_attention.sum(dim=-1)
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# D = 3 (MIRT)
# ---------------------------------------------------------------------------


class TestShapesMultiTrait:
    """Output shapes for n_traits = 3."""

    B, S, K, D = 4, 8, 5, 3

    def setup_method(self):
        self.model = make_model(n_traits=self.D, n_cats=self.K)
        self.questions, self.responses = make_batch(self.B, self.S, Q=20, K=self.K)

    def test_theta_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["theta"].shape == (self.B, self.S, self.D)

    def test_alpha_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["alpha"].shape == (self.B, self.S, self.D)

    def test_beta_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["beta"].shape == (self.B, self.S, self.K - 1)

    def test_logits_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["logits"].shape == (self.B, self.S, self.K)

    def test_probs_sum_to_one(self):
        out = self.model(self.questions, self.responses)
        sums = out["probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_logit_baseline_zero(self):
        out = self.model(self.questions, self.responses)
        assert torch.allclose(
            out["logits"][:, :, 0],
            torch.zeros(self.B, self.S),
            atol=1e-6,
        )

    def test_alpha_positive(self):
        out = self.model(self.questions, self.responses)
        assert (out["alpha"] > 0).all()


# ---------------------------------------------------------------------------
# Binary case (K = 2)
# ---------------------------------------------------------------------------


class TestShapesBinary:
    """K = 2 is the minimal ordinal case — one threshold."""

    B, S, K, D = 2, 6, 2, 1

    def setup_method(self):
        self.model = make_model(n_traits=self.D, n_cats=self.K)
        self.questions, self.responses = make_batch(self.B, self.S, Q=20, K=self.K)

    def test_beta_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["beta"].shape == (self.B, self.S, 1)

    def test_logits_shape(self):
        out = self.model(self.questions, self.responses)
        assert out["logits"].shape == (self.B, self.S, 2)
