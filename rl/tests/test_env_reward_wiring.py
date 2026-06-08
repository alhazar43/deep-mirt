"""Cross-package wiring test for env <-> reward integration.

This test confirms two things that no single-package test can.

1. One ``env.step(action)`` returns a reward whose four-component
   breakdown sums to the total. This validates the env-to-reward
   data path, including info population and the boundary between the
   composer's per-batch outputs and the env's scalar/tensor surface.
2. The composed action mask blocks every probe id end-to-end.
   Sampling a thousand uniform actions under the mask must never
   collide with any probe id, exercising the admin + probe + no-repeat
   AND through the env's mask buffer rather than through a synthetic
   mask construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ordrec.data import AdapterConfig, SyntheticAdapter
from ordrec.envs.frozen_magpcm import FrozenMAGPCM
from ordrec.envs.item_cache import build_item_cache
from ordrec.envs.ordrec_env import OrdRecEnv
from ordrec.reward.config import RewardConfig
from ordrec.reward.ordinal_reward import OrdinalRewardCompute


pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


def _make_env(tmp_path: Path, B: int = 4) -> OrdRecEnv:
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    n_questions, n_categories = 64, 4
    m = MAGPCM(
        n_questions=n_questions, n_categories=n_categories, n_traits=1,
        memory_size=8, key_dim=8, value_dim=8, summary_dim=8,
        embedding_type="learned", dropout_rate=0.0, ability_scale=1.0,
        separate_theta=True, init_value_memory=False,
    )
    world = FrozenMAGPCM(m)

    raw = tmp_path / "raw"
    raw.mkdir()
    n_students, seq_len = 24, 24
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
            "n_questions": n_questions, "n_categories": n_categories,
            "n_students": n_students,
        }),
        encoding="utf-8",
    )
    cfg_adapter = AdapterConfig(
        name="wire_synth", raw_dir=raw, out_dir=tmp_path,
        split_seed=0, test_frac=0.2, valid_frac=0.2, min_seq_len=3,
        max_seq_len=0, chunk_long_sequences=False,
    )
    adapter = SyntheticAdapter(cfg_adapter)
    adapter.materialise()
    adapter.load()

    cache = build_item_cache(world, n_contexts=2, dataset_name="wire_synth")
    cfg_reward = RewardConfig(K_B=2, T=4, probe_M=6, probe_H=3)
    reward = OrdinalRewardCompute(cfg_reward, n_categories=n_categories)
    env = OrdRecEnv(
        world_model=world, adapter=adapter, item_cache=cache,
        reward_fn=reward, cfg=cfg_reward, batch_size=B, warmup_len=3,
        split="train", seed=0,
    )
    return env


def test_one_step_breakdown_sums_to_total(tmp_path: Path) -> None:
    env = _make_env(tmp_path, B=4)
    state = env.reset(seed=10)
    prob = state.action_mask.to(torch.float32)
    prob = prob / prob.sum(dim=-1, keepdim=True)
    action = torch.multinomial(prob, num_samples=env.K_B, replacement=False)
    _, reward, _, info = env.step(action)
    summed = (
        info["r_info"] + info["r_cost"] + info["r_expo"] + info["r_voi"]
    )
    assert torch.allclose(summed, reward, atol=1e-5, rtol=0.0)
    # And ``r_total`` is the same scalar surface.
    assert torch.allclose(summed, info["r_total"], atol=1e-5, rtol=0.0)


def test_mask_blocks_every_probe_id_through_env(tmp_path: Path) -> None:
    env = _make_env(tmp_path, B=4)
    state = env.reset(seed=11)
    probe_C = state.raw_info["probe_C_ids"]
    probe_H = state.raw_info["probe_H_ids"]
    forbidden = []
    for b in range(env.B):
        forbidden.append(set(probe_C[b].tolist()) | set(probe_H[b].tolist()))

    mask = state.action_mask
    # Direct structural check, no probe id is True in the mask.
    for b in range(env.B):
        for pid in forbidden[b]:
            assert mask[b, pid].item() is False
        # Pad slot also forbidden.
        assert mask[b, 0].item() is False

    # 1000 uniform samples under the mask, none hit any probe id.
    prob = mask.to(torch.float32)
    prob = prob / prob.sum(dim=-1, keepdim=True)
    for _ in range(1000):
        a = torch.multinomial(prob, num_samples=1).squeeze(-1)  # (B,)
        for b in range(env.B):
            assert int(a[b].item()) not in forbidden[b]


def test_fleet_exposure_updates_after_episode(tmp_path: Path) -> None:
    """Episode termination must update the fleet EMA toward
    session-level admin frequencies."""
    env = _make_env(tmp_path, B=2)
    state = env.reset(seed=12)
    # The fleet buffer is zeros at construction.
    assert torch.allclose(env.fleet_expo, torch.zeros_like(env.fleet_expo))
    horizon = env.horizon_steps
    for _ in range(horizon):
        prob = state.action_mask.to(torch.float32)
        prob = prob / prob.sum(dim=-1, keepdim=True)
        action = torch.multinomial(prob, num_samples=env.K_B, replacement=False)
        state, _, done, _ = env.step(action)
    assert done is True
    # At least one item should have a non-zero EMA after one episode.
    assert (env.fleet_expo > 0.0).any()
