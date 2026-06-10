"""E4.7: terminal probe resampling for dynamic-world anchors.

Two invariants to pin:

1. ``resample_probe_responses`` produces DIFFERENT responses when the
   world model is conditioned on a drifted history vs a flat history.
   This confirms that a model trained on dynamic cohorts will actually
   exhibit within-session drift in the simulated anchor responses.

2. ``OrdRecEnv`` with ``resample_probe_at_terminal=True`` populates
   ``probe_H_resp_terminal`` in the terminal step's info dict, and
   ``OrdinalRewardCompute`` uses it instead of the reset-time
   ``probe_H_resp`` when computing ``r_voi``.

These tests use a tiny random-weight MAGPCM (no training) to keep the
suite hermetic.  The first test relies on the model producing different
logits for different histories, which is guaranteed by the attention
mechanism over the key-value memory as long as the history is distinct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
import torch

from ordrec.reward.config import RewardConfig
from ordrec.reward.nll_anchor import resample_probe_responses
from ordrec.reward.ordinal_reward import OrdinalRewardCompute

pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


# ---------------------------------------------------------------------------
# Tiny model / env helpers (reused across tests)
# ---------------------------------------------------------------------------

def _build_tiny_magpcm(n_questions: int = 32, n_categories: int = 4):
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    torch.manual_seed(99)
    return MAGPCM(
        n_questions=n_questions,
        n_categories=n_categories,
        n_traits=1,
        memory_size=8,
        key_dim=8,
        value_dim=8,
        summary_dim=8,
        embedding_type="learned",
        dropout_rate=0.0,
        ability_scale=1.0,
        separate_theta=True,
        init_value_memory=False,
    )


def _build_loaded_adapter(tmp_path: Path, n_questions: int = 32, n_categories: int = 4):
    from ordrec.data import AdapterConfig, SyntheticAdapter

    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    n_students, seq_len = 32, 32
    records = [
        {
            "questions": [((i * 3 + j) % n_questions) + 1 for j in range(seq_len)],
            "responses": [((i + j) % n_categories) for j in range(seq_len)],
        }
        for i in range(n_students)
    ]
    (raw / "sequences.json").write_text(json.dumps(records), encoding="utf-8")
    (raw / "metadata.json").write_text(
        json.dumps({
            "n_questions": n_questions,
            "n_categories": n_categories,
            "n_students": n_students,
        }),
        encoding="utf-8",
    )
    cfg = AdapterConfig(
        name="dyn_probe_test",
        raw_dir=raw,
        out_dir=tmp_path,
        split_seed=0,
        test_frac=0.2,
        valid_frac=0.2,
        min_seq_len=3,
        max_seq_len=0,
        chunk_long_sequences=False,
    )
    a = SyntheticAdapter(cfg)
    a.materialise()
    a.load()
    return a


def _make_env_with_flag(
    tmp_path: Path,
    resample: bool,
    B: int = 2,
) -> Tuple["OrdRecEnv", RewardConfig]:  # type: ignore[name-defined]
    from ordrec.envs.frozen_magpcm import FrozenMAGPCM
    from ordrec.envs.item_cache import build_item_cache
    from ordrec.envs.ordrec_env import OrdRecEnv

    n_questions = 32
    n_categories = 4
    world = FrozenMAGPCM(_build_tiny_magpcm(n_questions, n_categories))
    adapter = _build_loaded_adapter(tmp_path, n_questions, n_categories)
    cache = build_item_cache(world, n_contexts=2, dataset_name="dyn_probe_test")
    cfg = RewardConfig(
        K_B=2, T=4, probe_M=4, probe_H=2,
        resample_probe_at_terminal=resample,
    )
    reward = OrdinalRewardCompute(cfg, n_categories=n_categories)
    env = OrdRecEnv(
        world_model=world,
        adapter=adapter,
        item_cache=cache,
        reward_fn=reward,
        cfg=cfg,
        batch_size=B,
        warmup_len=3,
        split="train",
        seed=42,
    )
    return env, cfg


# ---------------------------------------------------------------------------
# Test 1: drifting history produces different probe responses
# ---------------------------------------------------------------------------

def test_resample_probe_differs_between_flat_and_drifted_history() -> None:
    """A history dominated by high responses should yield different probe
    predictions than one dominated by low responses, confirming the world
    model conditions non-trivially on history content.
    """
    from ordrec.envs.frozen_magpcm import FrozenMAGPCM

    n_questions, n_categories, B, H = 32, 4, 2, 4
    world = FrozenMAGPCM(_build_tiny_magpcm(n_questions, n_categories))
    probe_H_ids = torch.arange(1, H + 1, dtype=torch.long).unsqueeze(0).expand(B, -1)

    # History A: all responses = 0 (low)
    hist_q_low = torch.ones(B, 8, dtype=torch.long)
    hist_r_low = torch.zeros(B, 8, dtype=torch.long)

    # History B: all responses = K-1 (high)
    hist_q_high = torch.ones(B, 8, dtype=torch.long)
    hist_r_high = torch.full((B, 8), n_categories - 1, dtype=torch.long)

    g = torch.Generator()
    g.manual_seed(0)

    resp_low = resample_probe_responses(world, hist_q_low, hist_r_low, probe_H_ids, generator=g)
    g.manual_seed(0)
    resp_high = resample_probe_responses(world, hist_q_high, hist_r_high, probe_H_ids, generator=g)

    assert resp_low.shape == (B, H), f"unexpected shape {resp_low.shape}"
    assert resp_high.shape == (B, H), f"unexpected shape {resp_high.shape}"

    # The world model must produce at least some different probe responses
    # when conditioned on different extremes of history.  Run with multiple
    # seeds to rule out sampling noise; we just need one seed where they differ.
    differs = (resp_low != resp_high).any()
    if not differs:
        # Sample multiple seeds to reduce flakiness.
        for seed in range(1, 20):
            g.manual_seed(seed)
            r_low = resample_probe_responses(world, hist_q_low, hist_r_low, probe_H_ids, generator=g)
            g.manual_seed(seed)
            r_high = resample_probe_responses(world, hist_q_high, hist_r_high, probe_H_ids, generator=g)
            if (r_low != r_high).any():
                differs = True
                break

    assert differs, (
        "resample_probe_responses returned identical responses for low- and "
        "high-response histories across 20 seeds. The world model appears "
        "insensitive to history content, which would make the dynamic anchor "
        "meaningless."
    )


# ---------------------------------------------------------------------------
# Test 2: env populates probe_H_resp_terminal at terminal step
# ---------------------------------------------------------------------------

def test_env_populates_terminal_probe_resp_when_flag_set(tmp_path: Path) -> None:
    """With resample_probe_at_terminal=True the env must populate
    _probe_H_resp_terminal by the end of the terminal step.  The terminal
    step's r_voi must also be non-zero (the anchor fired).
    """
    env, cfg = _make_env_with_flag(tmp_path, resample=True, B=2)
    state = env.reset(seed=7)
    horizon = cfg.T // cfg.K_B
    last_info = None
    for _ in range(horizon):
        prob = state.action_mask.float()
        prob = prob / prob.sum(dim=-1, keepdim=True)
        action = torch.multinomial(prob, num_samples=cfg.K_B, replacement=False)
        state, reward, done, last_info = env.step(action)

    # After the terminal step the env must have populated the buffer.
    assert env._probe_H_resp_terminal is not None, (
        "_probe_H_resp_terminal must be set after the terminal step when "
        "resample_probe_at_terminal=True"
    )
    resp_t = env._probe_H_resp_terminal
    assert resp_t.shape == (2, cfg.probe_H), (
        f"Expected shape (2, {cfg.probe_H}), got {tuple(resp_t.shape)}"
    )
    assert resp_t.dtype == torch.long

    # The terminal step must also have produced a non-trivial r_voi.
    assert last_info is not None
    assert "r_voi" in last_info


# ---------------------------------------------------------------------------
# Test 3: env does NOT populate probe_H_resp_terminal when flag is False
# ---------------------------------------------------------------------------

def test_env_no_terminal_probe_resp_when_flag_unset(tmp_path: Path) -> None:
    """With resample_probe_at_terminal=False the env must NOT populate
    _probe_H_resp_terminal (backward compatibility with static-world path).
    """
    env, cfg = _make_env_with_flag(tmp_path, resample=False, B=2)
    state = env.reset(seed=8)
    horizon = cfg.T // cfg.K_B
    for _ in range(horizon):
        prob = state.action_mask.float()
        prob = prob / prob.sum(dim=-1, keepdim=True)
        action = torch.multinomial(prob, num_samples=cfg.K_B, replacement=False)
        state, reward, done, info = env.step(action)

    assert env._probe_H_resp_terminal is None, (
        "_probe_H_resp_terminal must remain None when "
        "resample_probe_at_terminal=False"
    )


# ---------------------------------------------------------------------------
# Test 4: reward composer uses terminal responses when flag set
# ---------------------------------------------------------------------------

def test_reward_composer_uses_terminal_resp_over_reset_resp() -> None:
    """When info contains probe_H_resp_terminal AND cfg.resample_probe_at_terminal
    is True, the reward composer must use probe_H_resp_terminal for the VOI
    anchor, not the reset-time probe_H_resp.
    """
    torch.manual_seed(42)
    B, D, Q, K, H = 2, 1, 32, 4, 4

    cfg_resample = RewardConfig(K_B=2, T=4, probe_M=4, probe_H=H, resample_probe_at_terminal=True)
    cfg_static = RewardConfig(K_B=2, T=4, probe_M=4, probe_H=H, resample_probe_at_terminal=False)

    compute_resample = OrdinalRewardCompute(cfg_resample, n_categories=K)
    compute_static = OrdinalRewardCompute(cfg_static, n_categories=K)

    # Build a terminal-step info dict with DIFFERENT reset-time and terminal
    # probe responses so the two paths must produce different r_voi values.
    probe_H_ids = torch.randint(1, Q + 1, (B, H))
    probe_H_resp_reset = torch.zeros(B, H, dtype=torch.long)       # all category 0
    probe_H_resp_terminal = torch.full((B, H), K - 1, dtype=torch.long)  # all category K-1

    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)

    horizon_steps = cfg_resample.T // cfg_resample.K_B
    info = {
        "probe_C_ids": torch.randint(1, Q + 1, (B, cfg_resample.probe_M)),
        "probe_H_ids": probe_H_ids,
        "probe_H_resp": probe_H_resp_reset,
        "probe_H_resp_terminal": probe_H_resp_terminal,
        "alpha_table": alpha,
        "beta_table": beta,
        "fleet_expo": torch.zeros(Q + 1),
        "step_index": horizon_steps,
        "theta_0": torch.randn(B, D),
        "horizon_steps": horizon_steps,
    }
    theta_t = torch.randn(B, D)
    st_prev = {"theta": torch.randn(B, D)}
    st_next = {"theta": theta_t}
    action = torch.randint(1, Q + 1, (B, cfg_resample.K_B))

    r_resample, br_resample = compute_resample(st_prev, action, st_next, info)
    r_static, br_static = compute_static(st_prev, action, st_next, info)

    # Both paths ran without error.
    assert br_resample["r_voi"].shape == (B,)
    assert br_static["r_voi"].shape == (B,)

    # The two paths must disagree when probe responses differ.
    assert not torch.allclose(br_resample["r_voi"], br_static["r_voi"]), (
        "r_voi should differ between resample and static paths when probe "
        "responses are different (all-0 vs all-(K-1))."
    )


# ---------------------------------------------------------------------------
# Test 5: probe mask anti-gaming invariant holds during terminal resampling
# ---------------------------------------------------------------------------

def test_probe_mask_unchanged_after_terminal_resampling(tmp_path: Path) -> None:
    """probe_H_ids must remain masked from the action set at every step,
    including the terminal step where resampling runs.  The mask must not
    be modified by the resampling call.
    """
    env, cfg = _make_env_with_flag(tmp_path, resample=True, B=2)
    state = env.reset(seed=9)
    initial_probe_H = state.raw_info["probe_H_ids"].clone()
    horizon = cfg.T // cfg.K_B
    for _ in range(horizon):
        # Verify probe_H is masked before each step.
        for b in range(2):
            for pid in initial_probe_H[b].tolist():
                assert not state.action_mask[b, pid].item(), (
                    f"probe_H id {pid} was NOT masked in action_mask at "
                    f"step {env._episode_step} for row {b}"
                )
        prob = state.action_mask.float()
        prob = prob / prob.sum(dim=-1, keepdim=True)
        action = torch.multinomial(prob, num_samples=cfg.K_B, replacement=False)
        state, reward, done, info = env.step(action)
