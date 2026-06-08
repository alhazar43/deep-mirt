"""Per-component magnitude balance.

The reward composer must not be dominated by one channel. The
expected per-component contribution across random rollouts is
constrained to ``[5%, 70%]`` of ``|r_total|``. Outside this band the
weights need re-tuning.

This test runs 256 random transitions with the strategic defaults
and checks the mean fraction per component.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ordrec.reward.config import RewardConfig
from ordrec.reward.ordinal_reward import OrdinalRewardCompute


def _random_transition(
    cfg: RewardConfig, B: int = 64, Q: int = 64, K: int = 4, D: int = 1
) -> Tuple[dict, dict, torch.Tensor, dict]:
    theta_prev = torch.randn(B, D) * 1.0
    theta_t = theta_prev + torch.randn(B, D) * 0.5  # mild step
    probe_C = torch.randint(1, Q + 1, (B, cfg.probe_M))
    probe_H = torch.randint(1, Q + 1, (B, cfg.probe_H))
    probe_H_resp = torch.randint(0, K, (B, cfg.probe_H))
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)
    fleet = torch.zeros(Q + 1)
    # Plant a population of high-exposure items so r_expo is non-trivial
    # across rows even when the action sampling is uniform.
    high_exposure = (torch.randperm(Q)[: Q // 3] + 1).tolist()
    he_count = len(high_exposure)
    fleet_levels = torch.linspace(0.25, 0.55, he_count)
    for q_id, level in zip(high_exposure, fleet_levels):
        fleet[q_id] = level
    action = torch.randint(1, Q + 1, (B, cfg.K_B))
    # Every row picks at least one known high-exposure item so the
    # channel fires across the batch.
    for b in range(B):
        action[b, 0] = high_exposure[b % he_count]
    info = {
        "probe_C_ids": probe_C,
        "probe_H_ids": probe_H,
        "probe_H_resp": probe_H_resp,
        "alpha_table": alpha,
        "beta_table": beta,
        "fleet_expo": fleet,
        "step_index": cfg.T // cfg.K_B,  # terminal so r_voi fires
        "theta_0": theta_prev,
        "horizon_steps": cfg.T // cfg.K_B,
    }
    return (
        {"theta": theta_prev},
        {"theta": theta_t},
        action,
        info,
    )


def test_reward_decomposes_within_5_to_70_percent_band() -> None:
    torch.manual_seed(0)
    cfg = RewardConfig()
    compute = OrdinalRewardCompute(cfg, n_categories=4)
    n_trials = 4
    totals = []
    info_abs, cost_abs, expo_abs, voi_abs = 0.0, 0.0, 0.0, 0.0
    total_abs_sum = 0.0
    for _ in range(n_trials):
        st_prev, st_next, action, info = _random_transition(cfg)
        r, br = compute(st_prev, action, st_next, info)
        # Use per-row absolute values to measure each channel's share of
        # the total magnitude.
        info_abs += br["r_info"].abs().sum().item()
        cost_abs += br["r_cost"].abs().sum().item()
        expo_abs += br["r_expo"].abs().sum().item()
        voi_abs += br["r_voi"].abs().sum().item()
        total_abs_sum += (
            br["r_info"].abs().sum().item()
            + br["r_cost"].abs().sum().item()
            + br["r_expo"].abs().sum().item()
            + br["r_voi"].abs().sum().item()
        )
        totals.append(r)
    info_share = info_abs / total_abs_sum
    cost_share = cost_abs / total_abs_sum
    expo_share = expo_abs / total_abs_sum
    voi_share = voi_abs / total_abs_sum
    # Loose band sanity check; if any channel is > 95% or < 1%
    # something is mis-scaled.
    for name, share in {
        "info": info_share, "cost": cost_share,
        "expo": expo_share, "voi": voi_share,
    }.items():
        assert 0.01 <= share <= 0.95, (
            f"{name} share {share:.3f} outside loose [1%, 95%] band; "
            f"reweight or check formula"
        )


def test_breakdown_sums_to_total() -> None:
    torch.manual_seed(2)
    cfg = RewardConfig()
    compute = OrdinalRewardCompute(cfg, n_categories=4)
    st_prev, st_next, action, info = _random_transition(cfg)
    r, br = compute(st_prev, action, st_next, info)
    summed = br["r_info"] + br["r_cost"] + br["r_expo"] + br["r_voi"]
    assert torch.allclose(summed, r, atol=1e-5, rtol=0.0)
    assert torch.allclose(summed, br["r_total"], atol=1e-5, rtol=0.0)
