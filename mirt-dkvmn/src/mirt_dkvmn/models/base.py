"""Base model interfaces."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn


class BaseKTModel(nn.Module, ABC):
    """Common interface for MIRT-KT models."""

    @abstractmethod
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError
