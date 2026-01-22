"""Nominal Response Model (NRM) head."""

import torch
import torch.nn as nn


class NRMHead(nn.Module):
    """Compute NRM probabilities with category-specific discriminations."""

    def forward(self, theta: torch.Tensor, alpha_cat: torch.Tensor, beta_cat: torch.Tensor) -> torch.Tensor:
        # alpha_cat: (batch, seq, n_cats, n_traits)
        # beta_cat: (batch, seq, n_cats)
        logits = torch.sum(alpha_cat * theta.unsqueeze(2), dim=-1) + beta_cat
        return torch.softmax(logits, dim=-1)
