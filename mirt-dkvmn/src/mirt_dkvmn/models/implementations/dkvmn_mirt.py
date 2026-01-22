"""DKVMN + MIRT-GPCM skeleton implementation."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseKTModel
from ..components.memory import DKVMN
from ..components.embeddings import LinearDecayEmbedding
from ..components.irt import MIRTParameterExtractor, MIRTGPCMLogits
from ..heads.gpcm import GPCMHead


class DKVMNMIRT(BaseKTModel):
    """Minimal MIRT-DKVMN model stub."""

    def __init__(self, n_questions: int, n_cats: int, n_traits: int,
                 memory_size: int = 50, key_dim: int = 64, value_dim: int = 64,
                 summary_dim: int = 64,
                 concept_aligned_memory: bool = False,
                 theta_projection: bool = False,
                 memory_add_activation: str = "tanh") -> None:
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.n_traits = n_traits
        self.concept_aligned_memory = concept_aligned_memory
        self.theta_projection = theta_projection

        if self.concept_aligned_memory and value_dim != n_traits:
            raise ValueError("concept_aligned_memory requires value_dim == n_traits")

        self.embedding = LinearDecayEmbedding(n_questions, n_cats)
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        self.value_proj = nn.Linear(self.embedding.output_dim, value_dim)

        self.memory = DKVMN(memory_size, key_dim, value_dim, add_activation=memory_add_activation)
        self.summary = nn.Sequential(
            nn.Linear(key_dim + value_dim, summary_dim),
            nn.Tanh(),
        )

        self.irt = MIRTParameterExtractor(summary_dim, n_traits, n_cats, question_dim=key_dim)
        self.theta_from_memory = nn.Linear(value_dim, n_traits) if theta_projection else None
        self.gpcm_logits = MIRTGPCMLogits()
        self.gpcm_head = GPCMHead()

    def forward(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch, seq = questions.shape
        self.memory.init_value_memory(batch)

        q_onehot = F.one_hot(questions, num_classes=self.n_questions + 1).float()
        q_onehot = q_onehot[:, :, 1:]

        embeds = self.embedding(q_onehot, responses)
        q_embed = self.q_embed(questions)

        thetas = []
        alphas = []
        betas = []
        probs = []
        attn_weights = []

        for t in range(seq):
            q_t = q_embed[:, t, :]
            v_t = self.value_proj(embeds[:, t, :])

            weights = self.memory.attention(q_t)
            read = self.memory.read(weights)
            attn_weights.append(weights)

            summary = self.summary(torch.cat([read, q_t], dim=-1))
            theta, alpha, beta = self.irt(summary.unsqueeze(1), q_t.unsqueeze(1))
            if self.concept_aligned_memory:
                if self.theta_from_memory is not None:
                    theta = self.theta_from_memory(read).unsqueeze(1)
                else:
                    theta = read.unsqueeze(1)
            logits = self.gpcm_logits(theta, alpha, beta)
            prob = self.gpcm_head(logits)

            thetas.append(theta.squeeze(1))
            alphas.append(alpha.squeeze(1))
            betas.append(beta.squeeze(1))
            probs.append(prob.squeeze(1))

            if t < seq - 1:
                self.memory.write(weights, v_t)

        self.last_attention = torch.stack(attn_weights, dim=1) if attn_weights else None
        return (
            torch.stack(thetas, dim=1),
            torch.stack(betas, dim=1),
            torch.stack(alphas, dim=1),
            torch.stack(probs, dim=1),
        )
