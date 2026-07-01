"""_p2_gpcm_alpha_key.py -- Option A: STATIC discrimination reads the item key.

The routing catch (blueprint ground-truth 4, decoders.py:374-420): the core
``GPCMDecoder`` static-decouple toggle routes ONLY beta (the step thresholds) to
the wide item KEY.  The slope alpha rides the item key ONLY through the dynamic
head (``state_alpha``); its STATIC readout ``fc_a`` always reads the thin item
VALUE.  So the naive {decoupled} x {static, dynamic} 2x2 does not cleanly cross
"slope has its own channel" with "slope is dynamic": the static-decoupled corner
gives beta a channel but leaves alpha on the value.

Option A (LOCKED decision 2) fixes this.  ``GPCMAlphaKeyDecoder`` is a thin
``GPCMDecoder`` subclass whose STATIC ``fc_a`` reads the wide item key when
``item_key_dim`` is set, exactly mirroring ``fc_b``.  With it, the static-
decoupled arm genuinely puts the SLOPE on its own item channel, so the 2x2
cleanly crosses {slope-channel} x {slope-dynamic}.

Bit-compatibility.  With ``item_key_dim=None`` (the shared arm) the subclass is
identical to the core decoder -- ``fc_a`` is untouched and reads the thin value.
So a single class serves all four GPCM configs (shared/decoupled x static/dynamic):
only the presence of ``item_key_dim`` (and whether a state is passed) selects the
corner.

Scratch file: ``_``-prefixed, gitignored.  Imports the read-only core decoder
(reuse, not edit).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from deep_irt.core.decoders import GPCMDecoder


class GPCMAlphaKeyDecoder(GPCMDecoder):
    """GPCM decoder with the STATIC discrimination head on the wide item key.

    Overrides exactly two things relative to :class:`GPCMDecoder`:

      1. ``__init__`` -- when ``item_key_dim`` is set, rebuild ``fc_a`` so its
         input width is ``item_key_dim`` (the wide key) instead of ``emb_dim``
         (the thin value).  ``fc_b`` (parent) already reads the key; now alpha
         does too, so the SLOPE has no thin-value channel in the decoupled arm.
      2. ``item_params`` -- the STATIC alpha branch reads ``item_key`` instead of
         ``emb`` when ``item_key_dim`` is set.  Everything else (the dynamic
         ``fc_a_state`` head, the three-mode beta routing, the positivity map)
         is inherited unchanged.

    Parameter count: replacing ``fc_a`` (``emb_dim -> 1``) with (``item_key_dim
    -> 1``) is the ONLY count change vs the core decoder, and it is the honest
    capacity the decoupled slope needs.  No dead heads.
    """

    def __init__(
        self,
        emb_dim: int,
        n_cats: int,
        state_dim: int | None = None,
        item_key_dim: int | None = None,
        alpha_log_scale: float | None = None,
        alpha_pos_map: str | None = None,
        alpha_pos_kwargs: dict[str, Any] | None = None,
        state_alpha: bool = True,
        state_beta: bool = False,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            n_cats=n_cats,
            state_dim=state_dim,
            item_key_dim=item_key_dim,
            alpha_log_scale=alpha_log_scale,
            alpha_pos_map=alpha_pos_map,
            alpha_pos_kwargs=alpha_pos_kwargs,
            state_alpha=state_alpha,
            state_beta=state_beta,
        )
        # Option A: rebuild the STATIC discrimination readout to ride the wide
        # item key.  Only when decoupled; the shared arm keeps parent's fc_a.
        # Re-init draws fresh RNG for fc_a only -- fc_b / fc_a_state / fc_b_state
        # were already built inside super().__init__ and are untouched.
        if self.item_key_dim is not None:
            self.fc_a = nn.Linear(self.item_key_dim, 1)

    def item_params(
        self,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> dict:
        """Mirror of :meth:`GPCMDecoder.item_params` with static alpha on the key.

        Only the STATIC alpha branch differs from the parent: when
        ``item_key_dim`` is set it reads ``item_key`` (Option A) rather than the
        thin ``emb``.  The dynamic-alpha branch and all three beta modes are
        reproduced faithfully.
        """
        # --- Discrimination ---
        if self.state_alpha and state is not None:
            # Dynamic alpha: unchanged from parent (already reads the item key
            # when decoupled).  A state passed only for the beta head does NOT
            # pull alpha off its static map (state_alpha gates this).
            if self.item_key_dim is not None:
                if item_key is None:
                    raise RuntimeError(
                        "GPCMAlphaKeyDecoder was built with item_key_dim "
                        "(decoupled wide item key); pass item_key to item_params."
                    )
                key = item_key
            else:
                key = emb
            a = self._alpha_pos(self.fc_a_state(torch.cat([state, key], dim=-1)))
        else:
            # STATIC alpha -- Option A: read the wide item key when decoupled,
            # else the thin value (parent's legacy behaviour, bit-identical).
            if self.item_key_dim is not None:
                if item_key is None:
                    raise RuntimeError(
                        "GPCMAlphaKeyDecoder was built with item_key_dim "
                        "(decoupled wide item key); pass item_key so the STATIC "
                        "discrimination head (Option A) can read it."
                    )
                a = self._alpha_pos(self.fc_a(item_key))
            else:
                a = self._alpha_pos(self.fc_a(emb))

        # --- Step thresholds (identical to parent's three-mode routing) ---
        if self.state_beta:
            if state is None:
                raise RuntimeError(
                    "GPCMAlphaKeyDecoder was built with state_beta "
                    "(state-conditioned step thresholds); pass state."
                )
            if item_key is None:
                raise RuntimeError(
                    "GPCMAlphaKeyDecoder was built with state_beta (which requires "
                    "the wide item key); pass item_key so beta reads it."
                )
            b = self.fc_b_state(torch.cat([state, item_key], dim=-1))
        elif self.item_key_dim is not None:
            if item_key is None:
                raise RuntimeError(
                    "GPCMAlphaKeyDecoder was built with item_key_dim (decoupled "
                    "wide item key); pass item_key so beta reads it."
                )
            b = self.fc_b(item_key)
        else:
            b = self.fc_b(emb)
        return {"alpha": a, "beta": b}


class Binary2PLAlphaKeyDecoder(nn.Module):
    """2PL (GPCM K=2) with the STATIC discrimination head on the wide item key.

    Thin wrapper mirroring ``core.decoders.Binary2PLDecoder`` but wrapping
    :class:`GPCMAlphaKeyDecoder`, so the 2PL static-decoupled arm also routes the
    single slope to its own item channel (Option A).  Exposes the same
    ``item_params`` / ``item_params_sorted`` / ``log_probs`` / ``binary_logit`` /
    ``nll`` surface the training and recovery paths use.
    """

    def __init__(
        self,
        emb_dim: int,
        state_dim: int | None = None,
        item_key_dim: int | None = None,
        alpha_log_scale: float | None = None,
        alpha_pos_map: str | None = None,
        alpha_pos_kwargs: dict[str, Any] | None = None,
        state_alpha: bool = True,
        state_beta: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cats = 2
        self.state_dim = state_dim
        self.item_key_dim = item_key_dim
        self.state_beta = state_beta
        self._gpcm = GPCMAlphaKeyDecoder(
            emb_dim=emb_dim, n_cats=2, state_dim=state_dim,
            item_key_dim=item_key_dim, alpha_log_scale=alpha_log_scale,
            alpha_pos_map=alpha_pos_map, alpha_pos_kwargs=alpha_pos_kwargs,
            state_alpha=state_alpha, state_beta=state_beta,
        )

    def item_params(self, item_val, state=None, item_key=None) -> dict:
        return self._gpcm.item_params(item_val, state=state, item_key=item_key)

    def item_params_sorted(self, item_val, state=None, item_key=None) -> dict:
        return self._gpcm.item_params_sorted(item_val, state=state, item_key=item_key)

    def log_probs(self, theta, alpha, beta) -> torch.Tensor:
        return self._gpcm.log_probs(theta, alpha, beta)

    def binary_logit(self, theta, item_val, state=None, item_key=None) -> torch.Tensor:
        lg = self._gpcm.logits_from_emb(theta, item_val, state=state, item_key=item_key)
        return lg[..., 1] - lg[..., 0]

    def nll(self, theta, item_val, responses, state=None, item_key=None) -> torch.Tensor:
        return self._gpcm.nll(theta, item_val, responses, state=state, item_key=item_key)


def build_alpha_key_decoder(
    decoder: str,
    emb_dim: int,
    n_cats: int,
    **kwargs: Any,
):
    """Factory: return the Option-A decoder for ``decoder in {gpcm, binary}``.

    Mirrors ``DeepIRTModel._make_decoder`` for the two slope-bearing decoders so
    step 7 (toggle wiring) can swap the alpha-key variant in without touching
    core.  ``kwargs`` forwards ``state_dim`` / ``item_key_dim`` / ``state_alpha``
    / ``state_beta`` / ``alpha_*`` unchanged.
    """
    if decoder == "gpcm":
        return GPCMAlphaKeyDecoder(emb_dim=emb_dim, n_cats=n_cats, **kwargs)
    if decoder == "binary":
        kwargs.pop("n_cats", None)
        return Binary2PLAlphaKeyDecoder(emb_dim=emb_dim, **kwargs)
    raise ValueError(
        f"build_alpha_key_decoder handles gpcm|binary (the slope-bearing "
        f"decoders); got {decoder!r}. NRM routing lives in _p2_nrm_channels."
    )
