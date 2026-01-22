"""Base configuration definitions."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """Shared config fields."""
    seed: int = 42
    device: str = "cuda"
    log_dir: Optional[str] = None
