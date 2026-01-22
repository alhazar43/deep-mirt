"""Model configuration definitions."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_questions: int
    n_cats: int = 4
    n_traits: int = 4
    memory_size: int = 50
    key_dim: int = 64
    value_dim: int = 64
    summary_dim: int = 64
    embedding_strategy: str = "linear_decay"
    concept_aligned_memory: bool = False
    theta_projection: bool = False
    memory_add_activation: str = "tanh"
    theta_source: str = "summary"
    gpcm_mode: str = "k_minus_1"
