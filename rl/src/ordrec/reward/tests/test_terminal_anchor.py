"""Terminal NLL anchor fires only at the episode horizon.

The composer adds ``r_voi = w_voi * (nll_prior - nll_terminal)`` only
when ``info["step_index"] == cfg.T // cfg.K_B``. Mid-episode steps
must contribute exactly zero.
"""

from __future__ import annotations

import torch

from ordrec.reward.config import RewardConfig
from ordrec.reward.nll_anchor import gpcm_nll, terminal_anchor
from ordrec.reward.ordinal_reward import OrdinalRewardCompute


def _basic_info(
    cfg: RewardConfig, B: int, D: int, Q: int, K: int, step_index: int
):
    probe_C = torch.randint(1, Q + 1, (B, cfg.probe_M))
    probe_H = torch.randint(1, Q + 1, (B, cfg.probe_H))
    probe_H_resp = torch.randint(0, K, (B, cfg.probe_H))
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)
    fleet = torch.zeros(Q + 1)
    return {
        "probe_C_ids": probe_C,
        "probe_H_ids": probe_H,
        "probe_H_resp": probe_H_resp,
        "alpha_table": alpha,
        "beta_table": beta,
        "fleet_expo": fleet,
        "step_index": int(step_index),
        "theta_0": torch.randn(B, D),
        "horizon_steps": cfg.T // cfg.K_B,
    }


def test_voi_zero_mid_episode() -> None:
    torch.manual_seed(0)
    cfg = RewardConfig()
    compute = OrdinalRewardCompute(cfg, n_categories=4)
    B, D, Q, K = 2, 1, 64, 4
    info = _basic_info(cfg, B, D, Q, K, step_index=1)  # mid-episode (T/K_B = 2)
    st_prev = {"theta": torch.randn(B, D)}
    st_next = {"theta": torch.randn(B, D)}
    action = torch.randint(1, Q + 1, (B, cfg.K_B))
    r, br = compute(st_prev, action, st_next, info)
    assert torch.allclose(br["r_voi"], torch.zeros(B), atol=0.0)


def test_voi_nonzero_at_horizon() -> None:
    torch.manual_seed(1)
    cfg = RewardConfig()
    compute = OrdinalRewardCompute(cfg, n_categories=4)
    B, D, Q, K = 2, 1, 64, 4
    info = _basic_info(cfg, B, D, Q, K, step_index=cfg.T // cfg.K_B)
    st_prev = {"theta": torch.randn(B, D)}
    st_next = {"theta": torch.randn(B, D)}
    action = torch.randint(1, Q + 1, (B, cfg.K_B))
    r, br = compute(st_prev, action, st_next, info)
    # Non-zero with very high probability under random theta movement.
    assert (br["r_voi"].abs() > 1e-6).any()


def test_terminal_anchor_sign_convention() -> None:
    """``terminal_anchor`` is positive when theta_t fits H_probe better."""
    torch.manual_seed(2)
    B, D, Q, K, H = 1, 1, 16, 4, 5
    # Construct a deterministic ground-truth theta and pull theta_t
    # toward it; theta_0 stays at the origin.
    theta_truth = torch.full((B, D), 2.0)
    theta_0 = torch.zeros(B, D)
    alpha = torch.full((Q + 1, D), 1.0)
    beta = torch.full((Q + 1, K - 1), 0.0)
    probe_H_ids = torch.randint(1, Q + 1, (B, H))
    # "Observed" responses are samples from the truth model. Use the
    # GPCM head at theta_truth directly.
    interaction = (alpha[probe_H_ids] * theta_truth.unsqueeze(1)).sum(dim=-1)
    alpha_norm = alpha[probe_H_ids].norm(dim=-1)
    step_vals = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta[probe_H_ids]
    cum_logits = step_vals.cumsum(dim=-1)
    zeros = torch.zeros(B, H, 1)
    logits = torch.cat([zeros, cum_logits], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    # Take the argmax as the observed label.
    probe_H_resp = probs.argmax(dim=-1)
    # NLL at theta_0 should be greater than NLL at theta_truth.
    diff = terminal_anchor(
        theta_truth, theta_0, probe_H_ids, probe_H_resp, alpha, beta,
    )
    assert (diff > 0.0).all(), (
        f"terminal_anchor sign is wrong, got {diff}; theta_truth should "
        "fit H_probe better than theta_0"
    )


def test_gpcm_nll_matches_cross_entropy() -> None:
    """``gpcm_nll`` matches ``F.cross_entropy`` averaged across H."""
    import torch.nn.functional as F  # noqa: WPS433

    torch.manual_seed(3)
    B, D, Q, K, H = 2, 1, 8, 4, 5
    theta = torch.randn(B, D)
    probe = torch.randint(1, Q + 1, (B, H))
    resp = torch.randint(0, K, (B, H))
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)

    nll = gpcm_nll(theta, probe, resp, alpha, beta)

    # Independent re-derivation of the logits via the documented formula.
    alpha_q = alpha[probe]
    beta_q = beta[probe]
    interaction = (alpha_q * theta.unsqueeze(1)).sum(dim=-1)
    alpha_norm = alpha_q.norm(dim=-1)
    step_vals = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta_q
    cum_logits = step_vals.cumsum(dim=-1)
    zeros = torch.zeros(B, H, 1)
    logits = torch.cat([zeros, cum_logits], dim=-1).clamp(-50.0, 50.0)
    # cross_entropy expects (N, K), so flatten and re-average.
    flat_logits = logits.reshape(-1, K)
    flat_resp = resp.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_resp, reduction="none")
    ce = ce.view(B, H).mean(dim=-1)
    assert torch.allclose(nll, ce, atol=1e-5, rtol=0.0)
