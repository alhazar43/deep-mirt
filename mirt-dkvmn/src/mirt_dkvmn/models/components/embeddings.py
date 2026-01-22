"""Embedding strategies for (question, response) pairs."""

import torch
import torch.nn as nn


class LinearDecayEmbedding(nn.Module):
    """Triangular-weight embedding for ordinal responses."""

    def __init__(self, n_questions: int, n_cats: int) -> None:
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats

    @property
    def output_dim(self) -> int:
        return self.n_questions * self.n_cats

    def forward(self, q_onehot: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        k = torch.arange(self.n_cats, device=responses.device).float()
        r = responses.unsqueeze(-1).float()
        dist = torch.abs(k - r) / max(self.n_cats - 1, 1)
        weights = torch.clamp(1.0 - dist, min=0.0)
        weighted_q = weights.unsqueeze(-1) * q_onehot.unsqueeze(2)
        return weighted_q.flatten(start_dim=2)
