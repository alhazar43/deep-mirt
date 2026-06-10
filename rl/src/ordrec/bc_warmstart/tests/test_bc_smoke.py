"""BC warm-start smoke against the max-Fisher teacher.

The student PPO actor is warm-started against the max-Fisher teacher
on a synthetic adapter; after BC the student's argmax-policy match
rate against the teacher on a held-out slice must exceed 85%. This
test exercises the full warm-start path (env reset + teacher
selection + BC gradient step + match-rate evaluation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
import torch

from ordrec.bc_warmstart import (
    bc_warmstart,
    max_fisher_actions,
)
from ordrec.data import AdapterConfig, SyntheticAdapter
from ordrec.envs import FrozenMAGPCM, OrdRecEnv, build_item_cache
from ordrec.reward import OrdinalRewardCompute, RewardConfig
from ordrec.training import PPO, set_seed


pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


def _make_env(tmp_path: Path, B: int = 4) -> Tuple[OrdRecEnv, FrozenMAGPCM]:
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    n_questions, n_categories = 32, 4
    m = MAGPCM(
        n_questions=n_questions, n_categories=n_categories, n_traits=1,
        memory_size=8, key_dim=8, value_dim=8, summary_dim=8,
        embedding_type="learned", dropout_rate=0.0, ability_scale=1.0,
        separate_theta=True, init_value_memory=False,
    )
    world = FrozenMAGPCM(m)

    raw = tmp_path / "raw"
    raw.mkdir()
    n_students, seq_len = 32, 24
    records = [
        {
            "questions": [((i * 5 + j) % n_questions) + 1 for j in range(seq_len)],
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
    adapter = SyntheticAdapter(AdapterConfig(
        name="bc_synth", raw_dir=raw, out_dir=tmp_path,
        split_seed=0, test_frac=0.2, valid_frac=0.2, min_seq_len=3,
        max_seq_len=0, chunk_long_sequences=False,
    ))
    adapter.materialise(); adapter.load()
    cache = build_item_cache(world, n_contexts=2, dataset_name="bc_synth")
    cfg = RewardConfig(K_B=2, T=4, probe_M=4, probe_H=2)
    reward = OrdinalRewardCompute(cfg, n_categories=n_categories)
    env = OrdRecEnv(
        world_model=world, adapter=adapter, item_cache=cache,
        reward_fn=reward, cfg=cfg, batch_size=B, warmup_len=3,
        split="train", seed=0,
    )
    return env, world


def test_bc_matches_max_fisher_teacher(tmp_path: Path) -> None:
    # Seed globally before any construction so the MAGPCM world model,
    # the ActorCritic init and the BC loop are deterministic regardless
    # of test order. PPO's constructor no longer reseeds global state,
    # so without this line the world-model weights would depend on
    # whatever ambient RNG state earlier tests left behind (the old
    # order-sensitive flake).
    set_seed(0)
    env, _ = _make_env(tmp_path)
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        seed=0,
        hidden_dim=64, n_hidden_layers=2,
        n_episodes_per_update=4, max_steps_per_episode=env.horizon_steps,
        minibatch_size=16, n_epochs=4, total_updates=30,
        learning_rate=3e-3,
    )

    # Pre-warmstart, the student should be roughly random against the
    # teacher (around 1 / n_actions match rate).
    history = bc_warmstart(
        ppo, env, n_updates=30, n_episodes_per_update=2, seed=0,
    )
    assert len(history) > 0
    final_match = history[-1].teacher_match_rate
    assert final_match >= 0.85, (
        f"BC warm-start failed to match teacher; got {final_match:.3f}"
    )


def test_max_fisher_teacher_respects_mask(tmp_path: Path) -> None:
    set_seed(0)
    env, _ = _make_env(tmp_path)
    state = env.reset(seed=99)
    theta = state.theta_t
    mask = state.action_mask
    teacher = max_fisher_actions(
        theta, env.alpha_table, env.beta_table, mask,
    )
    # Every teacher action must be allowed by the mask.
    for b in range(env.B):
        assert bool(mask[b, int(teacher[b].item())].item()) is True
    # And never the pad slot.
    assert not (teacher == 0).any()
