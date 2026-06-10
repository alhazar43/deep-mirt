"""Tests for B6 PPOConfig and RewardConfig.from_dict.

Verifies that both dataclasses:
- accept only known keys from an arbitrary dict (unknown keys dropped)
- fall back to defaults for missing keys
- are constructable from an empty dict
- are frozen (immutable)
- round-trip through from_dict -> asdict
"""

from __future__ import annotations

import dataclasses

import pytest

from ordrec.reward.config import RewardConfig
from ordrec.training.config import PPOConfig


# ---------------------------------------------------------------------------
# RewardConfig.from_dict
# ---------------------------------------------------------------------------


def test_reward_config_from_empty_dict_gives_defaults() -> None:
    cfg = RewardConfig.from_dict({})
    default = RewardConfig()
    assert cfg == default


def test_reward_config_from_dict_sets_known_keys() -> None:
    cfg = RewardConfig.from_dict({"w_info": 2.5, "w_voi": 0.0, "K_B": 10})
    assert cfg.w_info == 2.5
    assert cfg.w_voi == 0.0
    assert cfg.K_B == 10
    # Unset keys keep their defaults.
    assert cfg.w_cost == RewardConfig().w_cost


def test_reward_config_from_dict_ignores_unknown_keys() -> None:
    """Extra keys from a broader YAML section must be silently dropped."""
    cfg = RewardConfig.from_dict({"w_info": 1.5, "_unknown_key": "boom"})
    assert cfg.w_info == 1.5


def test_reward_config_is_frozen() -> None:
    cfg = RewardConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.w_info = 99.0  # type: ignore[misc]


def test_reward_config_round_trip() -> None:
    """from_dict(asdict(cfg)) must reproduce the original."""
    original = RewardConfig(w_info=3.0, K_B=8, probe_M=16)
    d = dataclasses.asdict(original)
    reconstructed = RewardConfig.from_dict(d)
    assert reconstructed == original


# ---------------------------------------------------------------------------
# PPOConfig
# ---------------------------------------------------------------------------


def test_ppo_config_from_empty_dict_gives_defaults() -> None:
    cfg = PPOConfig.from_dict({})
    default = PPOConfig()
    assert cfg == default


def test_ppo_config_from_dict_sets_known_keys() -> None:
    cfg = PPOConfig.from_dict({
        "learning_rate": 1e-3,
        "n_epochs": 8,
        "clip_eps": 0.1,
    })
    assert cfg.learning_rate == 1e-3
    assert cfg.n_epochs == 8
    assert cfg.clip_eps == 0.1
    assert cfg.gamma == PPOConfig().gamma  # default kept


def test_ppo_config_from_dict_ignores_unknown_keys() -> None:
    cfg = PPOConfig.from_dict({"learning_rate": 1e-4, "_extra": "noise"})
    assert cfg.learning_rate == 1e-4


def test_ppo_config_is_frozen() -> None:
    cfg = PPOConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.learning_rate = 0.1  # type: ignore[misc]


def test_ppo_config_round_trip() -> None:
    original = PPOConfig(hidden_dim=256, n_epochs=8, seed=42)
    d = dataclasses.asdict(original)
    reconstructed = PPOConfig.from_dict(d)
    assert reconstructed == original


def test_ppo_config_all_fields_have_defaults() -> None:
    """PPOConfig must be constructable with zero arguments."""
    cfg = PPOConfig()
    for f in dataclasses.fields(cfg):
        assert f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING, (  # type: ignore[misc]
            f"Field '{f.name}' has no default and no default_factory"
        )
