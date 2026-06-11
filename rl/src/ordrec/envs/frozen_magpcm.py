"""``FrozenMAGPCM``, the two-line world-model freeze wrapper.

The E2 plan replaces the v1 M1-era ``freeze_irt`` helper (state-dict
gymnastics, atol=1e-5 parity tests, ``StepState``) with a thin wrapper
that does the two things actually required of a frozen world model.

1. ``model.eval()``, so dropout and any future train-only normalisation
   layers are inactive.
2. ``model.requires_grad_(False)``, so the autograd graph does not
   build through the world model when the policy forwards through it.

See ``docs/ordrec_impl_guide.md`` Section 7 (E2 deliverables) and
``docs/exrec_ordinal_plan.md`` Section 4 ("What is gone") for the
narrative around this replacement.

The wrapper exposes a single stable callable
``forward_no_grad(questions, responses) -> dict`` that calls the
underlying ``model.forward`` inside ``torch.no_grad()`` and returns the
same five-key dict the native MA-IRT forward returns
(``logits``, ``probs``, ``theta``, ``alpha``, ``beta``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

try:
    from typing import TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict  # type: ignore[assignment]


class MAIRTOutput(TypedDict):
    """Typed return value of :meth:`FrozenMAGPCM.forward_no_grad`.

    All tensors share the batch dimension ``B`` and sequence dimension
    ``S`` from the input ``(questions, responses)`` pair.

    Attributes:
        logits: ``Tensor (B, S, K)`` un-normalised GPCM logits.
        probs: ``Tensor (B, S, K)`` category probabilities (softmax of
            logits).
        theta: ``Tensor (B, S, D)`` latent ability at each step.
        alpha: ``Tensor (B, S, D)`` item discrimination at each step.
        beta: ``Tensor (B, S, K-1)`` step thresholds at each step.
    """

    logits: Tensor
    probs: Tensor
    theta: Tensor
    alpha: Tensor
    beta: Tensor


def freeze_magpcm(model: nn.Module) -> nn.Module:
    """Apply the two-line freeze contract to a MA-IRT model in-place.

    Args:
        model: A torch ``nn.Module``, typically an ``MAGPCM`` instance.

    Returns:
        The same module, with ``eval()`` and ``requires_grad_(False)``
        applied. Returned for ergonomic chaining
        (``model = freeze_magpcm(MAGPCM(...))``).
    """
    model.eval()
    model.requires_grad_(False)
    return model


class FrozenMAGPCM(nn.Module):
    """Frozen-world-model wrapper around an MA-IRT model.

    The wrapper does not own learnable parameters of its own. It owns
    the underlying model as a submodule, enforces ``eval()`` and
    ``requires_grad_(False)`` at construction time, and provides
    ``forward_no_grad`` as the canonical inference entry point.

    Construction contract.

    - ``__init__`` calls :func:`freeze_magpcm` on ``model``.
    - ``train(mode)`` is overridden to keep the model in ``eval`` mode
      so calling ``ppo.train()`` upstream does not silently re-enable
      dropout in the world model.
    - ``forward_no_grad(questions, responses)`` is the public callable.
      The plain ``forward`` method dispatches to ``forward_no_grad`` so
      ``frozen(q, r)`` keeps working when called as a regular module.

    Args:
        model: The underlying MA-IRT model (an ``EncoderDecoderModel``
            in practice, but the wrapper does not depend on that type
            hierarchy; anything with ``forward(questions, responses) ->
            dict`` works).
        n_questions: Item bank size Q. Optional; if provided it is
            stored as a property for downstream consumers (e.g. the
            per-item cache builder). If omitted the wrapper attempts to
            read ``model.n_questions``.
        n_categories: K. Same fallback rule as ``n_questions``.
    """

    def __init__(
        self,
        model: nn.Module,
        n_questions: Optional[int] = None,
        n_categories: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"FrozenMAGPCM expects an nn.Module, got {type(model).__name__}"
            )
        self.model = freeze_magpcm(model)
        self._n_questions = (
            int(n_questions) if n_questions is not None
            else int(getattr(model, "n_questions", 0))
        )
        self._n_categories = (
            int(n_categories) if n_categories is not None
            else int(getattr(model, "n_categories", 0))
        )

    # ------------------------------------------------------------------
    # nn.Module overrides
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "FrozenMAGPCM":
        """Override; keep the underlying model in ``eval`` regardless of mode.

        This is the freeze invariant. Calling ``ppo.train()`` upstream
        would otherwise propagate to submodules and silently re-enable
        any train-only behaviour in the world model.
        """
        # Keep this wrapper's flag in sync with the requested mode so
        # external introspection (e.g. ``frozen.training``) reflects
        # caller intent, but the underlying world model stays frozen.
        super().train(mode)
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def forward_no_grad(self, questions: Tensor, responses: Tensor) -> MAIRTOutput:
        """Run a single frozen forward pass and return the five-key dict.

        Args:
            questions: ``LongTensor`` of shape ``(B, S)`` with item ids
                in ``[1, n_questions]`` (``0`` reserved for padding).
            responses: ``LongTensor`` of shape ``(B, S)`` with ordinal
                labels in ``[0, n_categories - 1]``.

        Returns:
            An :class:`MAIRTOutput` TypedDict with keys
            ``logits, probs, theta, alpha, beta``. All tensors have
            leading dimensions ``(B, S, ...)`` matching the input.

        Raises:
            AssertionError: When the returned dict is missing one of the
                five expected keys. Catches silent forward-contract
                breakage early.

        Notes:
            ``torch.no_grad()`` is used explicitly (rather than relying
            on ``requires_grad_(False)``) because some downstream tensors
            can still autograd through buffers that retain grad
            connections. ``no_grad`` is the belt-and-braces guarantee.
        """
        with torch.no_grad():
            out: Dict[str, Tensor] = self.model(questions, responses)

        # Shape contract assertion: all five keys must be present.
        _EXPECTED_KEYS = {"logits", "probs", "theta", "alpha", "beta"}
        missing = _EXPECTED_KEYS - set(out.keys())
        if missing:
            raise AssertionError(
                f"FrozenMAGPCM.forward_no_grad: world model returned a dict "
                f"missing required keys {missing}. "
                "Check that the underlying model implements the MA-IRT "
                "forward contract."
            )

        return MAIRTOutput(  # type: ignore[misc]
            logits=out["logits"],
            probs=out["probs"],
            theta=out["theta"],
            alpha=out["alpha"],
            beta=out["beta"],
        )

    def forward(self, questions: Tensor, responses: Tensor) -> MAIRTOutput:
        """Alias for :meth:`forward_no_grad`.

        Lets the wrapper be called as a regular ``nn.Module`` and keeps
        the type signature compatible with the unwrapped model.
        """
        return self.forward_no_grad(questions, responses)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_questions(self) -> int:
        return self._n_questions

    @property
    def n_categories(self) -> int:
        return self._n_categories

    @property
    def device(self) -> torch.device:
        """Device of the first underlying parameter or buffer."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            try:
                return next(self.model.buffers()).device
            except StopIteration:
                return torch.device("cpu")


__all__ = ["FrozenMAGPCM", "MAIRTOutput", "freeze_magpcm"]
