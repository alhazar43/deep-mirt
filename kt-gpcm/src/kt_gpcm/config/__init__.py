"""Configuration module: dataclass types and YAML loader."""

from .types import Config, BaseConfig, ModelConfig, TrainingConfig, DataConfig
from .loader import load_config

__all__ = [
    "Config",
    "BaseConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "load_config",
]
