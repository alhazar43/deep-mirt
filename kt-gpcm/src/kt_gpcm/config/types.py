"""Configuration dataclasses for memirt / kt_gpcm.

Five flat dataclasses form a nested Config that is populated from
YAML via :func:`kt_gpcm.config.loader.load_config`.  All fields carry
defaults that match the validated Deep-GPCM training recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseConfig:
    """Global experiment settings."""

    experiment_name: str = "base"
    device: str = "cuda"   # Gracefully falls back to CPU when CUDA is absent
    seed: int = 42


@dataclass
class ModelConfig:
    """DKVMN-GPCM model hyperparameters.

    n_traits controls the latent trait dimension D.  Setting it to 1
    recovers the scalar IRT model; any D > 1 gives a MIRT model with
    no code changes — the dot-product interaction
        sum(theta * alpha, dim=-1)
    degenerates correctly for D = 1.
    """

    n_questions: int = 100
    n_categories: int = 5          # K — number of ordinal response categories
    n_traits: int = 1              # D — multi-dimensionality is a config toggle
    memory_size: int = 50          # M — number of DKVMN memory slots
    key_dim: int = 64              # d_k — key / query dimension
    value_dim: int = 128           # d_v — value memory dimension
    summary_dim: int = 50          # d_s — summary (FC) hidden dimension
    response_dim: int = 16         # d_r — response encoding dimension (separable only)
    use_separable_embed: bool = False  # legacy separable toggle
    embedding_type: str = "linear_decay"  # "linear_decay" | "separable" | "static_item"
    item_embed_dim: int = 0        # static item embedding dim H (static_item only); 0 = K*Q auto
    ability_scale: float = 1.0     # Global scale applied to raw theta output
    dropout_rate: float = 0.0
    memory_add_activation: str = "tanh"   # Activation for DKVMN add gate
    init_value_memory: bool = True        # Learned initial value memory


@dataclass
class TrainingConfig:
    """Training loop and loss configuration.

    Loss recipe (validated Deep-GPCM defaults):
        L = focal_weight * FocalLoss
          + weighted_ordinal_weight * WeightedOrdinalLoss(ordinal_penalty)

    Regularization penalties are off by default (weight = 0).
    """

    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0

    # Loss weights
    focal_weight: float = 0.5
    weighted_ordinal_weight: float = 0.5
    ordinal_penalty: float = 0.5   # Internal penalty inside WeightedOrdinalLoss

    # LR scheduler (ReduceLROnPlateau on QWK, mode='max')
    lr_patience: int = 3
    lr_factor: float = 0.8

    # Optional regularization penalties (all off by default)
    attention_entropy_weight: float = 0.0
    theta_norm_weight: float = 0.0
    alpha_prior_weight: float = 0.0
    beta_prior_weight: float = 0.0


@dataclass
class DataConfig:
    """Dataset location and split parameters."""

    data_dir: str = "data"
    dataset_name: str = "synthetic"
    train_split: float = 0.8
    min_seq_len: int = 10


@dataclass
class Config:
    """Root configuration object.  Constructed by :func:`load_config`."""

    base: BaseConfig = field(default_factory=BaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
