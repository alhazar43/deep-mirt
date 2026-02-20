"""Dynamic Key-Value Memory Network (DKVMN).

Collapses the three-class hierarchy from the original deep-gpcm
(``MemoryNetwork`` ABC → ``MemoryHeadGroup`` → ``DKVMN``) into a single
``nn.Module``.  The ``learned_init`` parameter moves the initial value
memory parameter inside this class (was a loose ``nn.Parameter`` on
``DeepGPCM`` before).

Theory
------
DKVMN [Zhang et al., 2017] augments a key memory M^k ∈ R^{M×d_k} (static,
learned) with a dynamic value memory M^v_t ∈ R^{M×d_v} (updated each
timestep).

Attention (correlation weight):
    q̃  = tanh(W_q · q + b_q)          (query transform, d_k → d_k)
    w_t = softmax(M^k · q̃)            (M,)

Read:
    r_t = w_t^⊤ M^v_t                  (d_v,)

Write (erase-add):
    e_t = sigmoid(W_e · v_t + b_e)     (d_v,)  erase signal
    a_t = tanh(W_a · v_t + b_a)        (d_v,)  add signal
    M^v_{t+1,i} = M^v_{t,i} · (1 − w_{t,i} · e_t) + w_{t,i} · a_t

where v_t is the projected value embedding for timestep t.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DKVMN(nn.Module):
    """Flattened Dynamic Key-Value Memory Network.

    Args:
        n_questions: Item-bank size (Q).  Kept for future use / logging.
        key_dim: Dimension of query / key vectors (d_k).
        value_dim: Dimension of value memory slots (d_v).
        memory_size: Number of memory slots (M).
        learned_init: If ``True``, the initial value memory is an
            ``nn.Parameter`` of shape ``(M, d_v)`` owned by this module.
            If ``False``, the value memory is zero-initialised each call.
    """

    def __init__(
        self,
        n_questions: int,
        key_dim: int,
        value_dim: int,
        memory_size: int = 50,
        learned_init: bool = True,
    ) -> None:
        super().__init__()
        self.n_questions = n_questions
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.memory_size = memory_size
        self.learned_init = learned_init

        # ---- Key memory (static, learned) --------------------------------
        self.key_memory = nn.Parameter(torch.empty(memory_size, key_dim))
        nn.init.kaiming_normal_(self.key_memory)

        # ---- Query transform (tanh, preserved from original) -------------
        self.query_linear = nn.Linear(key_dim, key_dim, bias=True)
        nn.init.kaiming_normal_(self.query_linear.weight)
        nn.init.constant_(self.query_linear.bias, 0.0)

        # ---- Write head (erase + add) ------------------------------------
        self.erase_linear = nn.Linear(value_dim, value_dim, bias=True)
        self.add_linear = nn.Linear(value_dim, value_dim, bias=True)
        nn.init.kaiming_normal_(self.erase_linear.weight)
        nn.init.kaiming_normal_(self.add_linear.weight)
        nn.init.constant_(self.erase_linear.bias, 0.0)
        nn.init.constant_(self.add_linear.bias, 0.0)

        # ---- Learned initial value memory --------------------------------
        if learned_init:
            self.init_memory_param = nn.Parameter(
                torch.empty(memory_size, value_dim)
            )
            nn.init.kaiming_normal_(self.init_memory_param)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def init_value_memory(self, batch_size: int) -> Tensor:
        """Return a fresh value memory for a new batch.

        Args:
            batch_size: B — number of sequences in the batch.

        Returns:
            Tensor of shape ``(B, M, d_v)``.  When ``learned_init=True``
            the shared parameter is expanded (not cloned) to save memory;
            the first ``write()`` call produces a new contiguous tensor.
        """
        device = self.key_memory.device
        if self.learned_init:
            # Expand shared param across batch dimension: (1, M, dv) → (B, M, dv)
            return self.init_memory_param.unsqueeze(0).expand(
                batch_size, self.memory_size, self.value_dim
            ).contiguous()
        else:
            return torch.zeros(
                batch_size, self.memory_size, self.value_dim, device=device
            )

    def attention(self, query: Tensor) -> Tensor:
        """Compute softmax attention weights over memory slots.

        Applies a learned linear+tanh transform to the query before
        computing the dot product with the key memory.  This is the
        architectural choice carried over from the original deep-gpcm.

        Args:
            query: ``(B, d_k)`` query vector (typically the question
                   embedding at the current timestep).

        Returns:
            ``(B, M)`` attention weight vector (sums to 1 over M).
        """
        # q̃ = tanh(W_q · q)   shape: (B, d_k)
        query_transformed = torch.tanh(self.query_linear(query))
        # Dot product with key memory: (B, d_k) × (d_k, M)ᵀ → (B, M)
        scores = torch.matmul(query_transformed, self.key_memory.t())
        return F.softmax(scores, dim=-1)

    def read(self, value_memory: Tensor, attention_weights: Tensor) -> Tensor:
        """Weighted read from value memory.

        Args:
            value_memory: ``(B, M, d_v)`` current value memory.
            attention_weights: ``(B, M)`` softmax weights.

        Returns:
            Read vector ``(B, d_v)``.
        """
        # (B, 1, M) × (B, M, d_v) → (B, 1, d_v) → (B, d_v)
        return torch.bmm(attention_weights.unsqueeze(1), value_memory).squeeze(1)

    def write(
        self,
        value_memory: Tensor,
        attention_weights: Tensor,
        write_value: Tensor,
    ) -> Tensor:
        """Erase-add write to value memory.

        Args:
            value_memory: ``(B, M, d_v)`` current value memory.
            attention_weights: ``(B, M)`` softmax weights.
            write_value: ``(B, d_v)`` content to write.

        Returns:
            Updated value memory ``(B, M, d_v)``.
        """
        # Erase / add signals — each (B, d_v)
        erase = torch.sigmoid(self.erase_linear(write_value))   # (B, d_v)
        add = torch.tanh(self.add_linear(write_value))           # (B, d_v)

        # Reshape for broadcast over memory slots
        # w: (B, M, 1),  erase/add: (B, 1, d_v)
        w = attention_weights.unsqueeze(-1)      # (B, M, 1)
        erase_mat = torch.bmm(w, erase.unsqueeze(1))   # (B, M, d_v)
        add_mat = torch.bmm(w, add.unsqueeze(1))       # (B, M, d_v)

        new_memory = value_memory * (1.0 - erase_mat) + add_mat
        return new_memory
