"""Training utilities: seeding, schedules, polyak averaging.

Small helpers used by every algorithm. The aim is one place per
concept rather than scattering ``random.seed``/``np.random.seed``/
``torch.manual_seed`` triples across the code base.

See ``docs/ordrec_impl_guide.md`` Section 4.1 (Module 3 utilities).
"""

from __future__ import annotations

import math
import os
import random
from typing import Optional

import numpy as np
import torch
from torch import nn


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, numpy, torch CPU and CUDA in one call.

    Args:
        seed: Integer seed; coerced via ``int(seed)``.
        deterministic: When ``True`` also forces ``torch.use_deterministic_algorithms``
            and clears the ``CUBLAS_WORKSPACE_CONFIG`` env var. Off by
            default because some torch ops fall back to slower kernels
            in deterministic mode.
    """
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def linear_anneal(
    step: int, *, anneal_steps: int, start: float, end: float,
) -> float:
    """Linear schedule from ``start`` to ``end`` over ``anneal_steps``.

    After ``step >= anneal_steps`` the value clamps to ``end``.

    Args:
        step: Current training step (e.g. update index).
        anneal_steps: Number of steps to span ``start -> end``.
        start: Initial value.
        end: Final value.

    Returns:
        Linearly interpolated value in ``[min(start, end), max(start, end)]``.
    """
    if anneal_steps <= 0:
        return float(end)
    frac = min(1.0, max(0.0, float(step) / float(anneal_steps)))
    return float(start) + frac * (float(end) - float(start))


def cosine_anneal(
    step: int, *, anneal_steps: int, start: float, end: float,
) -> float:
    """Half-cosine schedule from ``start`` to ``end`` over ``anneal_steps``.

    Cosine schedules are smoother than linear at both endpoints and are
    a common LR schedule in PPO/Spinning Up follow-ups. Included here
    so configs can switch without code changes.
    """
    if anneal_steps <= 0:
        return float(end)
    frac = min(1.0, max(0.0, float(step) / float(anneal_steps)))
    cos = 0.5 * (1.0 - math.cos(math.pi * frac))
    return float(start) + cos * (float(end) - float(start))


def polyak_update(
    online: nn.Module, target: nn.Module, *, tau: float
) -> None:
    """In-place polyak average of ``target`` toward ``online``.

    ``target <- (1 - tau) * target + tau * online``. ``tau = 0.005`` is
    the SAC/DQN default. Both modules must share state-dict keys.
    """
    if tau < 0.0 or tau > 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}.")
    with torch.no_grad():
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


def pick_device(device: Optional[str] = None) -> torch.device:
    """Resolve a device string to ``torch.device`` with CUDA fallback.

    Args:
        device: ``"cuda"``, ``"cpu"`` or ``None`` (auto-detect).

    Returns:
        ``torch.device``. ``cuda`` is honoured only when CUDA is
        actually available.
    """
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


__all__ = [
    "cosine_anneal",
    "linear_anneal",
    "pick_device",
    "polyak_update",
    "set_seed",
]
