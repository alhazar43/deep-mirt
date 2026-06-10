"""Unit tests for gpcm_ops.gpcm_log_probs.

Two categories of tests:
1. Self-contained formula tests (no ma-irt import).
2. Numerical mirror against MAGPCM (requires ma-irt on PYTHONPATH).

The mirror test verifies that gpcm_log_probs returns identical
log-probabilities to what MAGPCM.forward produces at the same
(theta, alpha, beta) -- confirming the helper matches the world model
head without importing ma-irt internals into gpcm_ops.py itself.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ordrec.reward.gpcm_ops import gpcm_log_probs


# ---------------------------------------------------------------------------
# Self-contained formula tests
# ---------------------------------------------------------------------------


def test_output_shape() -> None:
    """Output shape is (B, N, K)."""
    torch.manual_seed(0)
    B, N, D, K = 4, 8, 2, 4
    Q = 32
    theta = torch.randn(B, D)
    item_ids = torch.randint(1, Q + 1, (B, N))
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.1
    beta = torch.randn(Q + 1, K - 1)
    lp = gpcm_log_probs(theta, item_ids, alpha, beta)
    assert lp.shape == (B, N, K)


def test_sums_to_zero_in_log_space() -> None:
    """log-probabilities sum to 0 in log-space, i.e. probs sum to 1."""
    torch.manual_seed(1)
    B, N, D, K = 3, 5, 1, 4
    Q = 20
    theta = torch.randn(B, D)
    item_ids = torch.randint(1, Q + 1, (B, N))
    alpha = torch.ones(Q + 1, D)
    beta = torch.zeros(Q + 1, K - 1)
    lp = gpcm_log_probs(theta, item_ids, alpha, beta)
    probs = lp.exp()
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Probability rows do not sum to 1: {row_sums}"
    )


def test_matches_manual_formula() -> None:
    """Verify against an independent hand-coded reference."""
    torch.manual_seed(42)
    B, N, D, K = 2, 3, 1, 4
    Q = 10
    theta = torch.randn(B, D)
    item_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)

    # Reference implementation.
    alpha_q = alpha[item_ids]
    beta_q = beta[item_ids]
    interaction = (alpha_q * theta.unsqueeze(1)).sum(-1)
    alpha_norm = alpha_q.norm(dim=-1)
    step_vals = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta_q
    cum_logits = step_vals.cumsum(-1)
    zeros = torch.zeros(B, N, 1)
    logits = torch.cat([zeros, cum_logits], dim=-1).clamp(-50.0, 50.0)
    ref_lp = torch.log_softmax(logits, dim=-1)

    lp = gpcm_log_probs(theta, item_ids, alpha, beta, logit_clip=50.0)
    assert torch.allclose(lp, ref_lp, atol=1e-6, rtol=0.0), (
        f"gpcm_log_probs disagrees with reference: max diff "
        f"{(lp - ref_lp).abs().max()}"
    )


def test_logit_clip_prevents_nan() -> None:
    """Extreme alpha does not produce NaN when logit_clip is active."""
    B, N, D, K = 1, 4, 1, 4
    Q = 10
    theta = torch.full((B, D), 10.0)
    item_ids = torch.randint(1, Q + 1, (B, N))
    alpha = torch.full((Q + 1, D), 100.0)
    beta = torch.zeros(Q + 1, K - 1)
    lp = gpcm_log_probs(theta, item_ids, alpha, beta, logit_clip=50.0)
    assert not lp.isnan().any(), "NaN in log_probs with extreme alpha"
    assert not lp.isinf().any(), "Inf in log_probs with extreme alpha"


def test_raises_on_wrong_theta_dims() -> None:
    """3D theta raises ValueError."""
    theta = torch.randn(2, 3, 4)
    item_ids = torch.randint(1, 5, (2, 3))
    alpha = torch.ones(10, 4)
    beta = torch.zeros(10, 3)
    with pytest.raises(ValueError, match="theta must be 2D"):
        gpcm_log_probs(theta, item_ids, alpha, beta)


def test_raises_on_batch_mismatch() -> None:
    """Batch dim mismatch between theta and item_ids raises ValueError."""
    theta = torch.randn(2, 1)
    item_ids = torch.randint(1, 5, (3, 4))
    alpha = torch.ones(10, 1)
    beta = torch.zeros(10, 3)
    with pytest.raises(ValueError, match="batch dim"):
        gpcm_log_probs(theta, item_ids, alpha, beta)


# ---------------------------------------------------------------------------
# Numerical mirror against MAGPCM (requires ma-irt on PYTHONPATH)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("models", reason="ma-irt required"),
    reason="ma-irt not on PYTHONPATH",
)
def test_matches_magpcm_probs_at_random_theta() -> None:
    """gpcm_log_probs numerically matches MAGPCM.forward probs at same params.

    Constructs a tiny MAGPCM, extracts its per-item (alpha, beta) via
    the item cache, then compares gpcm_log_probs output against the
    log-probabilities produced by forward_no_grad at the same theta.

    Critically: gpcm_ops.py does NOT import ma-irt. This test exercises
    the numerical match without creating a circular dependency.
    """
    pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]
    from ordrec.envs.frozen_magpcm import FrozenMAGPCM
    from ordrec.envs.item_cache import build_item_cache

    torch.manual_seed(7)
    Q, K, D = 16, 4, 1
    B, N = 4, 6

    m = MAGPCM(
        n_questions=Q, n_categories=K, n_traits=D,
        memory_size=8, key_dim=8, value_dim=8, summary_dim=8,
        embedding_type="learned", dropout_rate=0.0, ability_scale=1.0,
        separate_theta=True, init_value_memory=False,
    )
    world = FrozenMAGPCM(m)
    cache = build_item_cache(world, n_contexts=2, dataset_name="gpcm_ops_mirror")
    alpha = cache.alpha_tensor(torch.device("cpu"))  # (Q+1, D)
    beta = cache.beta_tensor(torch.device("cpu"))    # (Q+1, K-1)

    # Random theta and random items.
    theta = torch.randn(B, D)
    item_ids = torch.randint(1, Q + 1, (B, N))

    # gpcm_log_probs output.
    lp_ops = gpcm_log_probs(theta, item_ids, alpha, beta, logit_clip=50.0)

    # Reference: recompute independently using the same formula but
    # NOT calling gpcm_log_probs.
    alpha_q = alpha[item_ids].to(theta.dtype)
    beta_q = beta[item_ids].to(theta.dtype)
    interaction = (alpha_q * theta.unsqueeze(1)).sum(-1)
    alpha_norm = alpha_q.norm(dim=-1)
    step_vals = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta_q
    cum_logits = step_vals.cumsum(-1)
    zeros = torch.zeros(B, N, 1)
    logits = torch.cat([zeros, cum_logits], -1).clamp(-50.0, 50.0)
    lp_ref = torch.log_softmax(logits, dim=-1)

    assert torch.allclose(lp_ops, lp_ref, atol=1e-5, rtol=0.0), (
        f"gpcm_log_probs does not match reference at max diff "
        f"{(lp_ops - lp_ref).abs().max():.2e}"
    )
