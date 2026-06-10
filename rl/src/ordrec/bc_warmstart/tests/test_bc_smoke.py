"""BC warm-start smoke against the top-k Fisher soft teacher.

The student PPO actor is warm-started against the soft teacher that
assigns probability proportional to Fisher information over the top-5
items. After BC the student's top-5 overlap rate against the teacher
must exceed 80%. This test exercises the full warm-start path (env
reset + soft-teacher construction + BC gradient step + top-5-overlap
evaluation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
import torch

from ordrec.bc_warmstart import (
    BCStats,
    bc_warmstart,
    max_fisher_actions,
    top_k_fisher_soft_target,
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


def test_bc_top5_overlap_improves(tmp_path: Path) -> None:
    """BC warm-start must improve top-5 overlap against the soft teacher.

    After 30 updates the student's argmax should land in the teacher's
    top-5 Fisher items for at least 80% of examples. This is a weaker
    target than the old argmax match-rate because the soft target
    distributes probability across 5 items.
    """
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

    history = bc_warmstart(
        ppo, env, n_updates=30, n_episodes_per_update=2, seed=0,
        teacher_top_k=5,
    )
    assert len(history) > 0
    final_overlap = history[-1].teacher_top5_overlap
    # teacher_match_rate and teacher_top5_overlap are aliases here.
    assert final_overlap == history[-1].teacher_match_rate
    assert final_overlap >= 0.80, (
        f"BC warm-start failed; top-5 overlap={final_overlap:.3f} < 0.80"
    )


def test_bc_stats_has_top5_overlap_field(tmp_path: Path) -> None:
    """BCStats must expose ``teacher_top5_overlap`` as a named field."""
    stats = BCStats(
        bc_loss=0.5, teacher_match_rate=0.9, teacher_top5_overlap=0.9,
        entropy=1.2, n_examples=16,
    )
    assert stats.teacher_top5_overlap == 0.9
    assert stats.teacher_match_rate == stats.teacher_top5_overlap


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


def test_top_k_soft_target_sums_to_1_and_respects_mask(tmp_path: Path) -> None:
    """top_k_fisher_soft_target rows sum to 1 and zero out forbidden items."""
    set_seed(0)
    env, _ = _make_env(tmp_path)
    state = env.reset(seed=77)
    theta = state.theta_t
    mask = state.action_mask
    soft = top_k_fisher_soft_target(
        theta, env.alpha_table, env.beta_table, mask, k=5,
    )
    # Each row must sum to 1.0.
    row_sums = soft.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Soft target rows do not sum to 1: {row_sums.tolist()}"
    )
    # Forbidden items (mask=False) must have probability 0.
    assert (soft[~mask] == 0.0).all(), "Soft target assigns prob to forbidden item."
    # At most k items per row have non-zero probability.
    nonzero_per_row = (soft > 0.0).sum(dim=-1)
    assert (nonzero_per_row <= 5).all(), (
        f"Soft target has more than 5 non-zero entries: {nonzero_per_row.tolist()}"
    )
