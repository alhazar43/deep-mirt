"""DKVMN memory skeleton."""

import torch
import torch.nn as nn


class DKVMN(nn.Module):
    """Placeholder DKVMN with key/value memories."""

    def __init__(self, memory_size: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.key_memory = nn.Parameter(torch.randn(memory_size, key_dim))
        self.value_memory = None

        self.query_key = nn.Linear(key_dim, key_dim)
        self.erase = nn.Linear(value_dim, value_dim)
        self.add = nn.Linear(value_dim, value_dim)

    def init_value_memory(self, batch_size: int) -> None:
        device = self.key_memory.device
        self.value_memory = torch.zeros(batch_size, self.memory_size, self.value_dim, device=device)

    def attention(self, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.query_key(query), self.key_memory.t())
        return torch.softmax(scores, dim=-1)

    def read(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.matmul(weights.unsqueeze(1), self.value_memory).squeeze(1)

    def write(self, weights: torch.Tensor, content: torch.Tensor) -> None:
        erase = torch.sigmoid(self.erase(content))
        add = torch.tanh(self.add(content))
        erase = erase.unsqueeze(1)
        add = add.unsqueeze(1)
        weights = weights.unsqueeze(2)
        self.value_memory = self.value_memory * (1 - weights * erase) + weights * add
