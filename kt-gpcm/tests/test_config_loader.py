"""Tests for the YAML config loader.

Verifies:
- smoke.yaml loads without error
- Missing sections receive dataclass defaults
- Invalid values raise ValueError
- Fields are correctly overridden from YAML
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kt_gpcm.config import load_config, Config, BaseConfig, ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(content: str) -> Path:
    """Write a temporary YAML file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSmokeYaml:
    """Verify smoke.yaml (in the project configs/ directory) loads cleanly."""

    def test_loads_without_error(self):
        # Locate configs/ relative to this file:  tests/ -> memirt/ -> configs/
        configs_dir = Path(__file__).parent.parent / "configs"
        smoke_path = configs_dir / "smoke.yaml"
        if not smoke_path.exists():
            pytest.skip(f"smoke.yaml not found at {smoke_path}")
        cfg = load_config(smoke_path)
        assert isinstance(cfg, Config)

    def test_smoke_experiment_name(self):
        configs_dir = Path(__file__).parent.parent / "configs"
        smoke_path = configs_dir / "smoke.yaml"
        if not smoke_path.exists():
            pytest.skip("smoke.yaml not found")
        cfg = load_config(smoke_path)
        assert cfg.base.experiment_name == "smoke"

    def test_smoke_device_cpu(self):
        configs_dir = Path(__file__).parent.parent / "configs"
        smoke_path = configs_dir / "smoke.yaml"
        if not smoke_path.exists():
            pytest.skip("smoke.yaml not found")
        cfg = load_config(smoke_path)
        assert cfg.base.device == "cpu"

    def test_smoke_small_model(self):
        configs_dir = Path(__file__).parent.parent / "configs"
        smoke_path = configs_dir / "smoke.yaml"
        if not smoke_path.exists():
            pytest.skip("smoke.yaml not found")
        cfg = load_config(smoke_path)
        assert cfg.model.n_questions == 20
        assert cfg.model.n_categories == 4
        assert cfg.model.memory_size == 10


class TestDefaultFallbacks:
    """Missing YAML sections should use dataclass defaults."""

    def test_empty_yaml_uses_defaults(self):
        path = write_yaml("")
        cfg = load_config(path)
        defaults = Config()
        assert cfg.model.n_questions == defaults.model.n_questions
        assert cfg.model.n_traits == defaults.model.n_traits
        assert cfg.training.epochs == defaults.training.epochs
        assert cfg.data.train_split == defaults.data.train_split

    def test_partial_model_section(self):
        """Specifying one model field should not reset other fields."""
        path = write_yaml("model:\n  n_traits: 3\n")
        cfg = load_config(path)
        assert cfg.model.n_traits == 3
        assert cfg.model.n_questions == ModelConfig().n_questions  # default

    def test_base_section_only(self):
        path = write_yaml("base:\n  experiment_name: my_exp\n  seed: 7\n")
        cfg = load_config(path)
        assert cfg.base.experiment_name == "my_exp"
        assert cfg.base.seed == 7
        assert cfg.model.n_traits == ModelConfig().n_traits  # untouched


class TestValidation:
    """Invalid configs must raise ValueError."""

    def test_n_categories_less_than_2_raises(self):
        path = write_yaml("model:\n  n_categories: 1\n")
        with pytest.raises(ValueError, match="n_categories"):
            load_config(path)

    def test_n_traits_zero_raises(self):
        path = write_yaml("model:\n  n_traits: 0\n")
        with pytest.raises(ValueError, match="n_traits"):
            load_config(path)

    def test_train_split_out_of_range_raises(self):
        path = write_yaml("data:\n  train_split: 1.5\n")
        with pytest.raises(ValueError, match="train_split"):
            load_config(path)

    def test_train_split_zero_raises(self):
        path = write_yaml("data:\n  train_split: 0.0\n")
        with pytest.raises(ValueError, match="train_split"):
            load_config(path)


class TestFileNotFound:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/to/config.yaml")


class TestFieldOverride:
    """Verify that YAML values override defaults correctly."""

    def test_lr_override(self):
        path = write_yaml("training:\n  lr: 0.005\n")
        cfg = load_config(path)
        assert abs(cfg.training.lr - 0.005) < 1e-9

    def test_device_override(self):
        path = write_yaml("base:\n  device: cpu\n")
        cfg = load_config(path)
        assert cfg.base.device == "cpu"

    def test_multi_trait_override(self):
        path = write_yaml("model:\n  n_traits: 5\n  n_categories: 3\n")
        cfg = load_config(path)
        assert cfg.model.n_traits == 5
        assert cfg.model.n_categories == 3

    def test_loss_weights_override(self):
        path = write_yaml(
            "training:\n  focal_weight: 0.2\n  weighted_ordinal_weight: 0.8\n"
        )
        cfg = load_config(path)
        assert abs(cfg.training.focal_weight - 0.2) < 1e-9
        assert abs(cfg.training.weighted_ordinal_weight - 0.8) < 1e-9
