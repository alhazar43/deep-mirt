"""YAML config loader.

Usage::

    from kt_gpcm.config import load_config
    cfg = load_config("configs/smoke.yaml")
    print(cfg.model.n_traits)   # 1

The loader merges each YAML section over the corresponding dataclass
defaults, so partial YAML files (smoke configs) work without listing
every field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import Config, BaseConfig, DataConfig, ModelConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge(dataclass_instance: Any, overrides: dict) -> Any:
    """Apply *overrides* dict onto *dataclass_instance* in-place.

    Unknown keys are silently ignored so that typos in YAML produce a
    clear AttributeError at training time rather than a cryptic failure.
    """
    for key, value in overrides.items():
        if hasattr(dataclass_instance, key):
            setattr(dataclass_instance, key, value)
    return dataclass_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path) -> Config:
    """Load an :class:`Config` from a YAML file.

    Sections missing from the YAML keep their dataclass defaults.
    Only the four recognised top-level keys (``base``, ``model``,
    ``training``, ``data``) are processed; anything else is ignored.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Fully populated :class:`Config`.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        ValueError: If semantic validation fails (e.g. n_categories < 2).
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    cfg = Config()

    if "base" in raw:
        _merge(cfg.base, raw["base"])
    if "model" in raw:
        _merge(cfg.model, raw["model"])
    if "training" in raw:
        _merge(cfg.training, raw["training"])
    if "data" in raw:
        _merge(cfg.data, raw["data"])

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Raise :exc:`ValueError` for obviously invalid configurations."""
    if cfg.model.n_categories < 2:
        raise ValueError(
            f"model.n_categories must be >= 2, got {cfg.model.n_categories}"
        )
    if cfg.model.n_traits < 1:
        raise ValueError(
            f"model.n_traits must be >= 1, got {cfg.model.n_traits}"
        )
    if not (0.0 < cfg.data.train_split < 1.0):
        raise ValueError(
            f"data.train_split must be in (0, 1), got {cfg.data.train_split}"
        )
