"""Training configuration definitions."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    output_dir: str = "artifacts"
