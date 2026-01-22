"""Data configuration definitions."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    dataset_name: str
    data_root: str = "data"
    max_seq_len: Optional[int] = None
