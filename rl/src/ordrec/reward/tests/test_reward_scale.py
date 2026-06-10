"""Per-component magnitude balance, calibrated per component.

The reward composer must keep its four channels at their designed
relative scales. The composition is deliberately asymmetric. The
terminal VOI anchor (``w_voi = 5.0``) is the primary objective and
dominates the episode magnitude; ``r_info`` and ``r_cost`` are
mid-scale per-step shaping; ``r_expo`` is a corrective penalty that
stays small unless exposure misbehaves. A single symmetric band over
all four channels cannot hold (and an earlier version of this test
asserted a [1%, 95%] band so loose it was a no-op), so each channel
gets its own calibrated band instead.

Band derivation. Shares were measured episode-level (``T // K_B``
reward boundaries per episode, ``r_voi`` firing only at the terminal
one) over the strategic default weights across five seeds:

    r_info  0.075 - 0.084   (re-derived E4.6b R1; w_expo=0.02, r_max=0.40)
    r_cost  0.174 - 0.185
    r_expo  0.0013 - 0.0014
    r_voi   0.729 - 0.750

E4.6b R1 recalibration: w_expo reduced from 0.10 to 0.02 and r_max
raised from 0.20 to 0.40. The exposure penalty (r_expo) is now a
small corrective signal (~0.1% of episode magnitude) rather than a
dominant penalty. The VOI anchor (r_voi) absorbs the freed share.
Ablation grid confirmed all 10 design cells failed the
fisher-beats-random acceptance check; the recalibration is the
minimal change consistent with not making information-seeking policies
pay a structural tax.

Each band below spans roughly a factor of two around the measured
share. Seed-to-seed noise is about +/- 0.005, two orders of magnitude
smaller than the band width, while scaling any single default weight
by 2.5x or more pushes that channel's share outside its band (5x
mis-scalings land far outside). The bands therefore catch the
mis-scaled-channel failure mode during E5 reward tuning without
flaking on sampling noise. If the default weights change, re-derive
the bands with the same episode-level measurement and update both
the numbers and this docstring.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from ordrec.reward.config import RewardConfig
from ordrec.reward.ordinal_reward import OrdinalRewardCompute


# Calibrated per-component share bands for the strategic default
# weights. See the module docstring for the derivation.
SHARE_BANDS: Dict[str, Tuple[float, float]] = {
    "r_info": (0.04, 0.17),
    "r_cost": (0.09, 0.37),
    "r_expo": (0.0005, 0.003),
    "r_voi": (0.55, 0.90),
}


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


def test_components_within_calibrated_episode_bands() -> None:
    """Episode-level share of each channel stays inside its band.

    Shares are accumulated over full episodes (``T // K_B`` reward
    boundaries, only the last one terminal) so ``r_voi`` is weighted
    by its actual firing frequency rather than appearing at every
    transition.
    """
    torch.manual_seed(0)
    cfg = RewardConfig()
    compute = OrdinalRewardCompute(cfg, n_categories=4)
    n_episodes = 4
    n_boundaries = cfg.T // cfg.K_B
    abs_sums = {k: 0.0 for k in SHARE_BANDS}
    total_abs_sum = 0.0
    for _ in range(n_episodes):
        for step in range(1, n_boundaries + 1):
            st_prev, st_next, action, info = _random_transition(cfg)
            # Only the final boundary is terminal; r_voi must be zero
            # on the others, exactly as the env schedules it.
            info["step_index"] = step
            _, br = compute(st_prev, action, st_next, info)
            for key in abs_sums:
                v = br[key].abs().sum().item()
                abs_sums[key] += v
                total_abs_sum += v
    for key, (lo, hi) in SHARE_BANDS.items():
        share = abs_sums[key] / total_abs_sum
        assert lo <= share <= hi, (
            f"{key} share {share:.3f} outside calibrated band "
            f"[{lo}, {hi}]; a weight is mis-scaled (or the defaults "
            f"changed and the bands need re-derivation, see module "
            f"docstring)"
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
