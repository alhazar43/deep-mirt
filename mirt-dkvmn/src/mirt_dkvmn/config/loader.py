"""Config loader utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml

from .base import BaseConfig
from .data import DataConfig
from .model import ModelConfig
from .training import TrainingConfig
from .types import AppConfig


def load_config(path: str | Path) -> AppConfig:
    payload = _load_yaml(path)

    base_cfg = BaseConfig(**payload.get("base", {}))
    model_cfg = ModelConfig(**payload["model"])
    training_cfg = TrainingConfig(**payload.get("training", {}))
    data_cfg = DataConfig(**payload["data"])

    return AppConfig(base=base_cfg, model=model_cfg, training=training_cfg, data=data_cfg)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
