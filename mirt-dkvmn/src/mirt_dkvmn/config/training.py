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
    qwk_weight: float = 0.5
    ordinal_weight: float = 0.2
    attention_entropy_weight: float = 0.0
    theta_norm_weight: float = 0.0
    alpha_prior_weight: float = 0.0
    beta_prior_weight: float = 0.0
    alpha_norm_weight: float = 0.0
    alpha_norm_target: float = 1.0
    alpha_ortho_weight: float = 0.0
