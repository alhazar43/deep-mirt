"""MIRT parameter extraction and logits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIRTParameterExtractor(nn.Module):
    """Extract theta, alpha, beta for MIRT heads."""

    def __init__(
        self,
        input_dim: int,
        n_traits: int,
        n_cats: int,
        question_dim: int,
        gpcm_mode: str = "k_minus_1",
    ) -> None:
        super().__init__()
        self.n_traits = n_traits
        self.n_cats = n_cats
        self.gpcm_mode = gpcm_mode

        self.theta_net = nn.Linear(input_dim, n_traits)
        self.alpha_net = nn.Linear(input_dim + question_dim, n_traits)
        self.beta_base = nn.Linear(question_dim, 1)
        gap_size = n_cats - 1 if gpcm_mode == "k_full" else max(n_cats - 2, 1)
        self.beta_gaps = nn.Linear(question_dim, gap_size)

    def forward(self, features: torch.Tensor, question_features: torch.Tensor) -> tuple:
        theta = self.theta_net(features)
        alpha_input = torch.cat([features, question_features], dim=-1)
        raw_alpha = self.alpha_net(alpha_input)
        alpha = torch.exp(0.3 * raw_alpha)
        beta_0 = self.beta_base(question_features)

        if self.gpcm_mode == "k_full":
            gaps = F.softplus(self.beta_gaps(question_features))
            betas = [beta_0]
            for idx in range(gaps.shape[-1]):
                betas.append(betas[-1] + gaps[..., idx:idx + 1])
            beta = torch.cat(betas, dim=-1)
        else:
            if self.n_cats <= 2:
                beta = beta_0
            else:
                gaps = F.softplus(self.beta_gaps(question_features))
                betas = [beta_0]
                for idx in range(gaps.shape[-1]):
                    betas.append(betas[-1] + gaps[..., idx:idx + 1])
                beta = torch.cat(betas, dim=-1)

        return theta, alpha, beta


class MIRTGPCMLogits(nn.Module):
    """Compute GPCM logits for multi-dimensional traits."""

    def __init__(self, n_cats: int, gpcm_mode: str = "k_minus_1") -> None:
        super().__init__()
        self.n_cats = n_cats
        self.gpcm_mode = gpcm_mode

    def forward(self, theta: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = theta.shape
        n_cats = self.n_cats
        dot = torch.sum(theta * alpha, dim=-1)

        logits = torch.zeros(batch, seq, n_cats, device=theta.device)
        if self.gpcm_mode == "k_full":
            for k in range(n_cats):
                logits[:, :, k] = torch.sum(dot.unsqueeze(-1) - beta[:, :, : k + 1], dim=-1)
        else:
            for k in range(1, n_cats):
                logits[:, :, k] = torch.sum(dot.unsqueeze(-1) - beta[:, :, :k], dim=-1)
        return logits
