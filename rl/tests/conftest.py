"""Shared fixtures for the cross-package script smoke tests.

The E5-facing scripts (``rl/scripts/train_ppo.py`` and
``rl/scripts/eval_policy.py``) are plain files rather than package
modules, so they are loaded here via ``importlib`` under their plain
names. ``eval_policy`` imports ``from train_ppo import ...`` at module
import time, which resolves through ``sys.modules`` once the
``train_ppo_module`` fixture has run.

``smoke_cfg`` builds a fully hermetic tiny config, raw synthetic data,
adapter artefacts and training outputs all live under ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "rl" / "scripts"


def _load_script(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    script = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="session")
def train_ppo_module() -> ModuleType:
    """``rl/scripts/train_ppo.py`` loaded as module ``train_ppo``."""
    return _load_script("train_ppo")


@pytest.fixture(scope="session")
def eval_policy_module(train_ppo_module: ModuleType) -> ModuleType:
    """``rl/scripts/eval_policy.py``; needs ``train_ppo`` pre-loaded."""
    return _load_script("eval_policy")


@pytest.fixture()
def smoke_cfg(tmp_path: Path) -> Dict[str, Any]:
    """Tiny hermetic train_ppo config dict (2 updates, K_B=2, T=4).

    Generates deterministic synthetic raw data under ``tmp_path`` and
    points every output (adapter artefacts, checkpoints, metrics) at
    ``tmp_path`` so the smoke run leaves no repo-level artifacts.
    """
    n_questions, n_categories = 32, 4
    n_students, seq_len = 32, 24
    raw = tmp_path / "raw"
    raw.mkdir()
    records = [
        {
            "questions": [
                ((i * 5 + j) % n_questions) + 1 for j in range(seq_len)
            ],
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
    return {
        "experiment_name": "smoke_script_test",
        "output_dir": str(tmp_path / "outputs"),
        "device": "cpu",
        "seed": 0,
        "adapter": {
            "type": "synthetic",
            "name": "smoke_script_synth",
            "raw_dir": str(raw),
            "out_dir": str(tmp_path / "artifacts"),
            "split_seed": 0,
            "test_frac": 0.2,
            "valid_frac": 0.2,
            "min_seq_len": 3,
            "max_seq_len": 0,
            "chunk_long_sequences": False,
        },
        "world_model": {
            "n_traits": 1,
            "memory_size": 8,
            "key_dim": 8,
            "value_dim": 8,
            "summary_dim": 8,
            "embedding_type": "learned",
            "dropout_rate": 0.0,
            "ability_scale": 1.0,
            "separate_theta": True,
            "init_value_memory": False,
        },
        "env": {
            "batch_size": 4,
            "warmup_len": 3,
            "split": "train",
            "K_B": 2,
            "T": 4,
            "cache_n_contexts": 2,
            "cache_context_seq_len": 1,
        },
        "reward": {
            "probe_M": 4,
            "probe_H": 2,
        },
        "warmstart": {"enabled": False},
        "ppo": {
            "total_updates": 2,
            "n_episodes_per_update": 2,
            "max_steps_per_episode": 2,
            "hyperparameters": {
                "hidden_dim": 16,
                "n_hidden_layers": 1,
                "minibatch_size": 8,
                "n_epochs": 2,
            },
        },
    }
