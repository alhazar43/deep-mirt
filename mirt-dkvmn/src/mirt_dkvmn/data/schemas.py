"""Dataset schemas and validation helpers."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SequenceBatch:
    questions: List[List[int]]
    responses: List[List[int]]


@dataclass
class DatasetBundle:
    """Loaded dataset bundle."""

    questions: List[List[int]]
    responses: List[List[int]]
    n_questions: int
    n_cats: int
