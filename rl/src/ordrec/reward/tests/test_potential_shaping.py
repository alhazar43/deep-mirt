"""Ng-Harada-Russell (1999) potential-based shaping telescopes.

When the per-batch info reward is ``r_info_t = w_info * (phi_prev -
phi_t)``, the sum over a closed trajectory must telescope to
``w_info * (phi(theta_0) - phi(theta_T))``. This is what makes the
shaping policy-invariant on the optimal-policy set.
"""

from __future__ import annotations

import torch

from ordrec.reward.config import RewardConfig
from ordrec.reward.ordinal_reward import OrdinalRewardCompute


def _info_only_reward(theta_a, theta_b, probe_C, alpha_table, beta_table, cfg):
    """Run the composer with no exposure penalty and no terminal anchor."""
    compute = OrdinalRewardCompute(cfg, n_categories=int(beta_table.shape[-1] + 1))
    B = theta_a.shape[0]
    info = {
        "probe_C_ids": probe_C,
        "alpha_table": alpha_table,
        "beta_table": beta_table,
        "fleet_expo": torch.zeros(alpha_table.shape[0]),
        "step_index": 0,  # non-terminal, so r_voi is zero
        "horizon_steps": cfg.T // cfg.K_B,
    }
    # Zero action so r_cost and r_expo are 0.
    zero_act = torch.zeros(B, cfg.K_B, dtype=torch.long)
    reward, breakdown = compute(
        {"theta": theta_a}, zero_act, {"theta": theta_b}, info
    )
    return reward, breakdown


def test_two_transitions_telescope() -> None:
    """``r_info(0->1) + r_info(1->2)`` equals ``w_info * (phi_0 - phi_2)``."""
    torch.manual_seed(0)
    B, D, Q, K, M = 4, 1, 12, 4, 6
    theta_0 = torch.randn(B, D)
    theta_1 = torch.randn(B, D)
    theta_2 = torch.randn(B, D)
    probe_C = torch.randint(1, Q + 1, (B, M))
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)
    cfg = RewardConfig(w_cost=0.0, w_expo=0.0, w_voi=0.0)

    _, b01 = _info_only_reward(theta_0, theta_1, probe_C, alpha, beta, cfg)
    _, b12 = _info_only_reward(theta_1, theta_2, probe_C, alpha, beta, cfg)

    summed = b01["r_info"] + b12["r_info"]
    expected = cfg.w_info * (b01["phi_prev"] - b12["phi_t"])
    assert torch.allclose(summed, expected, atol=1e-6, rtol=0.0)


def test_phi_prev_phi_t_chain_at_boundary() -> None:
    """``phi_t`` at boundary b equals ``phi_prev`` at boundary b + 1."""
    torch.manual_seed(1)
    B, D, Q, K, M = 2, 1, 8, 3, 4
    theta_0 = torch.randn(B, D)
    theta_1 = torch.randn(B, D)
    theta_2 = torch.randn(B, D)
    probe_C = torch.randint(1, Q + 1, (B, M))
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)
    cfg = RewardConfig(w_cost=0.0, w_expo=0.0, w_voi=0.0)
    _, b01 = _info_only_reward(theta_0, theta_1, probe_C, alpha, beta, cfg)
    _, b12 = _info_only_reward(theta_1, theta_2, probe_C, alpha, beta, cfg)
    assert torch.allclose(b01["phi_t"], b12["phi_prev"], atol=1e-6, rtol=0.0)
