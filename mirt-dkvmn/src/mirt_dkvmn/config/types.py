"""Config dataclasses."""

from dataclasses import dataclass

from .base import BaseConfig
from .model import ModelConfig
from .training import TrainingConfig
from .data import DataConfig


@dataclass
class AppConfig:
    base: BaseConfig
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
