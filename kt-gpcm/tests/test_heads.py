"""Tests for GPCMLogits and GPCMHead.

Verifies:
- Category-0 logit is always 0 (cumulative baseline)
- Probabilities sum to 1 over the category dimension
- GPCMLogits + GPCMHead are inverse-consistent with log-softmax
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kt_gpcm.models.components.irt import GPCMLogits
from kt_gpcm.models.heads.gpcm import GPCMHead


B, S, K, D = 4, 7, 5, 2


def make_params(D=D, K=K):
    """Random (theta, alpha, beta) in valid ranges."""
    theta = torch.randn(B, S, D)
    alpha = torch.exp(torch.randn(B, S, D) * 0.3)   # positive
    # Monotonically increasing beta
    beta_base = torch.randn(B, S, 1)
    gaps = F.softplus(torch.randn(B, S, K - 2))
    betas = [beta_base]
    for i in range(K - 2):
        betas.append(betas[-1] + gaps[:, :, i : i + 1])
    beta = torch.cat(betas, dim=-1)  # (B, S, K-1)
    return theta, alpha, beta


class TestGPCMLogits:
    def setup_method(self):
        self.logit_fn = GPCMLogits()
        self.theta, self.alpha, self.beta = make_params()

    def test_output_shape(self):
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        assert logits.shape == (B, S, K)

    def test_category0_is_zero(self):
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        assert torch.allclose(
            logits[:, :, 0], torch.zeros(B, S), atol=1e-6
        ), "Category-0 logit must be 0 (cumulative baseline)"

    def test_finite_output(self):
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        assert torch.isfinite(logits).all()

    def test_scalar_equivalence_d1(self):
        """For D=1, dot product == scalar multiplication."""
        theta_1d = torch.randn(B, S, 1)
        alpha_1d = torch.exp(torch.randn(B, S, 1) * 0.3)
        beta_1d = torch.randn(B, S, K - 1)

        logits_nd = self.logit_fn(theta_1d, alpha_1d, beta_1d)

        # Manual scalar: eta = alpha * theta (both squeezed)
        eta = (alpha_1d * theta_1d).sum(-1)  # (B, S)
        logits_manual = torch.zeros(B, S, K)
        for k in range(1, K):
            logits_manual[:, :, k] = (eta.unsqueeze(-1) - beta_1d[:, :, :k]).sum(-1)

        assert torch.allclose(logits_nd, logits_manual, atol=1e-5)


class TestGPCMHead:
    def setup_method(self):
        self.logit_fn = GPCMLogits()
        self.head = GPCMHead()
        self.theta, self.alpha, self.beta = make_params()

    def test_output_shape(self):
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        probs = self.head(logits)
        assert probs.shape == (B, S, K)

    def test_probs_sum_to_one(self):
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        probs = self.head(logits)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B, S), atol=1e-5)

    def test_probs_non_negative(self):
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        probs = self.head(logits)
        assert (probs >= 0).all()

    def test_probs_at_most_one(self):
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        probs = self.head(logits)
        assert (probs <= 1.0 + 1e-6).all()

    def test_logits_probs_consistent(self):
        """softmax(logits) == probs — consistency check."""
        logits = self.logit_fn(self.theta, self.alpha, self.beta)
        probs_head = self.head(logits)
        probs_manual = F.softmax(logits, dim=-1)
        assert torch.allclose(probs_head, probs_manual, atol=1e-6)
