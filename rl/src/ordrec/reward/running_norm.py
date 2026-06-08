"""Welford running mean/variance with a freeze buffer.

The reward composer applies an optional running-mean/std normaliser
to ``r_total`` for the first ``running_norm_freeze_after`` updates,
then freezes the statistics. Freezing keeps the policy gradient
signal stationary in the second half of training where the running
estimates would otherwise still drift slightly under bootstrap noise.

The Welford recurrence is the standard online algorithm (Welford
1962, Technometrics). We keep it numerically stable by using the
sum-of-squared-differences accumulator ``M2`` rather than ``E[X^2]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


@dataclass
class RunningStats:
    mean: float
    var: float
    count: int
    frozen: bool


class RunningMeanStd:
    """Welford running mean/variance with an explicit freeze flag.

    The state is scalar; the composer normalises ``r_total`` rather
    than each component to keep the per-component breakdown
    interpretable.

    Args:
        freeze_after: Number of ``update`` calls after which the
            buffers freeze. Subsequent ``update`` calls are no-ops.
            ``0`` disables freezing.
        eps: Lower bound on the variance used in ``normalise`` to
            avoid division by zero before any samples arrive.
    """

    def __init__(self, freeze_after: int = 1000, eps: float = 1e-8) -> None:
        if freeze_after < 0:
            raise ValueError(
                f"freeze_after must be non-negative, got {freeze_after}."
            )
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._count: int = 0
        self._freeze_after: int = int(freeze_after)
        self._frozen: bool = False
        self._eps: float = float(eps)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def mean(self) -> float:
        return float(self._mean)

    @property
    def var(self) -> float:
        if self._count < 2:
            return 0.0
        return float(self._m2 / max(1, self._count - 1))

    @property
    def std(self) -> float:
        return float(max(self._eps, self.var) ** 0.5)

    @property
    def count(self) -> int:
        return int(self._count)

    @property
    def frozen(self) -> bool:
        return bool(self._frozen)

    def stats(self) -> RunningStats:
        return RunningStats(
            mean=self.mean, var=self.var, count=self.count, frozen=self.frozen,
        )

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(self, x: Union[float, Tensor]) -> None:
        """Fold ``x`` (scalar or batched) into the running statistics.

        Args:
            x: A python scalar or a 1D / scalar ``Tensor``. Tensors are
                detached and converted to float64 internally for the
                Welford step.

        Notes:
            One ``update(x_batch_of_B)`` call counts as ``B`` samples,
            not one. The freeze cutoff therefore measures samples seen,
            not the number of ``update`` calls.
        """
        if self._frozen:
            return
        if isinstance(x, Tensor):
            values = x.detach().reshape(-1).to(dtype=torch.float64).tolist()
        else:
            values = [float(x)]

        for v in values:
            self._count += 1
            delta = v - self._mean
            self._mean += delta / self._count
            delta2 = v - self._mean
            self._m2 += delta * delta2
            if (
                self._freeze_after > 0
                and self._count >= self._freeze_after
            ):
                self._frozen = True
                break

    def normalise(self, x: Tensor) -> Tensor:
        """Return ``(x - mean) / max(std, sqrt(eps))``.

        Detaches scaling constants so they do not bleed gradients into
        the reward target.

        Args:
            x: ``Tensor`` of any shape.

        Returns:
            ``Tensor`` of the same shape, centred and scaled by the
            current statistics. If no samples have been seen yet the
            result is ``x`` unchanged.
        """
        if self._count < 2:
            return x
        return (x - float(self._mean)) / float(self.std)

    # ------------------------------------------------------------------
    # Persistence (for save/load round-trip)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "mean": float(self._mean),
            "m2": float(self._m2),
            "count": int(self._count),
            "freeze_after": int(self._freeze_after),
            "frozen": bool(self._frozen),
            "eps": float(self._eps),
        }

    def load_state_dict(self, state: dict) -> None:
        self._mean = float(state["mean"])
        self._m2 = float(state["m2"])
        self._count = int(state["count"])
        self._freeze_after = int(state["freeze_after"])
        self._frozen = bool(state["frozen"])
        self._eps = float(state["eps"])


__all__ = ["RunningMeanStd", "RunningStats"]
