"""DKVMN + Softmax baseline — no IRT structure.

Identical DKVMN backbone to DeepGPCM but replaces the IRT parameter
extractor and GPCM logit layer with a direct linear classifier:

    logits = W_out @ summary + b    (B, S, K)

Used as RQ1 baseline to isolate the contribution of the GPCM head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .components.embeddings import LinearDecayEmbedding, StaticItemEmbedding
from .components.memory import DKVMN


class DKVMNSoftmax(nn.Module):
    """DKVMN backbone with a plain K-way softmax head.

    Constructor signature is compatible with DeepGPCM so the same
    train.py can be used with a model_type config switch.
    """

    def __init__(
        self,
        n_questions: int,
        n_categories: int = 5,
        memory_size: int = 50,
        key_dim: int = 64,
        value_dim: int = 64,
        summary_dim: int = 50,
        dropout_rate: float = 0.0,
        memory_add_activation: str = "tanh",
        init_value_memory: bool = False,
        embedding_type: str = "static_item",
        item_embed_dim: int = 0,
        # unused — kept for config compatibility with DeepGPCM
        n_traits: int = 1,
        ability_scale: float = 1.0,
        response_dim: int = 16,
        use_separable_embed: bool = False,
        model_type: str = "dkvmn_softmax",
    ) -> None:
        super().__init__()
        self.n_categories = n_categories

        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)

        self.embedding_type = embedding_type
        if embedding_type == "static_item":
            self.embedding = StaticItemEmbedding(
                n_questions, n_categories, value_dim, item_embed_dim
            )
            self.value_proj = None
        elif embedding_type == "separable" or use_separable_embed:
            self.item_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
            self.register_buffer(
                "k_indices",
                torch.arange(n_categories, dtype=torch.float32).view(1, 1, n_categories),
            )
            self.value_proj = nn.Linear(key_dim + n_categories, value_dim)
        else:
            self.embedding = LinearDecayEmbedding(n_questions, n_categories)
            self.value_proj = nn.Linear(self.embedding.output_dim, value_dim)

        self.memory = DKVMN(
            n_questions=n_questions,
            key_dim=key_dim,
            value_dim=value_dim,
            memory_size=memory_size,
            learned_init=init_value_memory,
        )

        summary_input_dim = value_dim + key_dim
        self.summary = nn.Sequential(
            nn.Linear(summary_input_dim, summary_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(summary_dim, n_categories)

    def forward(self, questions: Tensor, responses: Tensor) -> dict:
        B, S = questions.shape
        q_embed = self.q_embed(questions)  # (B, S, key_dim)

        if self.embedding_type == "static_item":
            value_embed = self.embedding(questions, responses)
        elif self.embedding_type == "separable" or hasattr(self, "item_embed"):
            item_v = self.item_embed(questions)
            dist = torch.abs(self.k_indices - responses.float().unsqueeze(-1)) / (self.n_categories - 1)
            resp_feat = torch.clamp(1.0 - dist, min=0.0)
            value_embed = self.value_proj(torch.cat([item_v, resp_feat], dim=-1))
        else:
            embed = self.embedding(questions, responses)
            value_embed = self.value_proj(embed)

        value_mem = self.memory.init_value_memory(B)

        all_logits = []
        for t in range(S):
            q_t = q_embed[:, t, :]
            v_t = value_embed[:, t, :]
            attn_t = self.memory.attention(q_t)
            read_t = self.memory.read(value_mem, attn_t)
            summary_t = self.summary(torch.cat([read_t, q_t], dim=-1))
            logits_t = self.classifier(summary_t).unsqueeze(1)  # (B, 1, K)
            all_logits.append(logits_t)
            value_mem = self.memory.write(value_mem, attn_t, v_t)

        logits = torch.cat(all_logits, dim=1)   # (B, S, K)
        probs = torch.softmax(logits, dim=-1)

        # Dummy IRT fields so trainer/plot_recovery don't crash
        return {
            "logits": logits,
            "probs": probs,
            "theta": torch.zeros(B, S, 1, device=logits.device),
            "alpha": torch.ones(B, S, 1, device=logits.device),
            "beta": torch.zeros(B, S, self.n_categories - 1, device=logits.device),
        }
