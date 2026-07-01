"""_p2_model.py -- _P2Model: DeepIRTModel with the proper GPCM loss + halting fit.

Phase-2 INTEGRATION build (blueprint build order step 1).  Two things the base
``DeepIRTModel`` cannot do on its own, added here as a thin subclass (reuse, not
rewrite -- the encoder, the decoders, the forward/recovery paths are all
inherited unchanged):

1. GPCM trains on the STRICTLY-PROPER cumulative-link ordinal CE
   (``_p2_ordinal_ce.OrdinalCELoss``) instead of the improper WeightedOrdinalLoss
   the user bans.  Only ``_build_loss_fn`` changes: 2PL keeps its inline BCE and
   NRM keeps its softmax CE (both already the proper scoring rule for their
   response type), and ``train.gpcm_loss == "wol"`` restores the legacy loss for
   the ablation contrast.  ``_predict_loss`` is untouched -- ``OrdinalCELoss`` and
   ``CombinedLoss`` share the ``(logits, responses) -> scalar`` call surface.

2. A fit loop that can actually HALT on an early-stopping rule.  The base
   ``fit``'s per-epoch callback only NOTIFIES (a raised ``StopIteration`` is
   swallowed, so it cannot stop training).  ``fit_early_stop`` re-implements the
   base training step verbatim (same seeding, grad clip, mask, mini-batch logic)
   and adds: monitor a prediction loss on an inner-val slice, restore the
   best-so-far weights, and STOP after ``patience`` epochs without improvement.
   The stopping rule is identical across every cell (metric = the same
   ``_compute_loss`` on the inner-val slice; patience; restore-best), the honesty
   requirement in the blueprint.

Option A (LOCKED decision 2) is realized here too: ``_make_decoder`` routes the
GPCM/2PL DECOUPLED arms (``item_key_dim`` set) through
``_p2_gpcm_alpha_key.build_alpha_key_decoder`` so the STATIC discrimination reads
the wide item key.  With ``item_key_dim=None`` (shared/benchmark arms) the parent
decoder is used unchanged, so the fixed benchmark stays byte-identical.

Scratch file: ``_``-prefixed, gitignored.  Subclasses the read-only core model;
does not edit it.
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from deep_irt.core.model import DeepIRTModel
from deep_irt.bench._p2_ordinal_ce import OrdinalCELoss
from deep_irt.bench._p2_gpcm_alpha_key import build_alpha_key_decoder


class _P2Model(DeepIRTModel):
    """DeepIRTModel with the proper GPCM ordinal CE and a halting early-stop fit.

    Parameters
    ----------
    gpcm_loss : "ordinal_ce" (default) | "wol".  Selects the GPCM training loss;
        inert for the 2PL/NRM decoders.  Set BEFORE ``super().__init__`` so the
        loss built during construction already reflects it.
    All other parameters are forwarded verbatim to :class:`DeepIRTModel`.
    """

    def __init__(self, *args, gpcm_loss: str = "ordinal_ce", **kwargs) -> None:
        if gpcm_loss not in ("ordinal_ce", "wol"):
            raise ValueError(
                f"gpcm_loss must be 'ordinal_ce' or 'wol', got {gpcm_loss!r}"
            )
        # Set before super().__init__: the base constructor calls _build_loss_fn().
        self.gpcm_loss = gpcm_loss
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Loss: proper GPCM ordinal CE (2PL BCE and NRM softmax CE unchanged)
    # ------------------------------------------------------------------

    def _build_loss_fn(
        self,
        responses: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Optional[nn.Module]:
        """Swap the cumulative-link ordinal CE in for GPCM; defer otherwise.

        ``gpcm_loss == "ordinal_ce"`` (default) + GPCM decoder -> the strictly-
        proper :class:`OrdinalCELoss` (no class weights, no ordinal penalty; a
        proper scoring rule is what the recovery claim needs).  ``"wol"`` or any
        non-GPCM decoder -> the parent's format-keyed loss (WOL / plain CE / inline
        BCE), byte-identical to the base model.
        """
        gpcm_loss = getattr(self, "gpcm_loss", "ordinal_ce")
        if self.decoder_name == "gpcm" and gpcm_loss == "ordinal_ce":
            return OrdinalCELoss(reduction="mean").to(self.device)
        return super()._build_loss_fn(responses, mask)

    # ------------------------------------------------------------------
    # Decoder: Option A alpha-key routing for the decoupled GPCM/2PL arms
    # ------------------------------------------------------------------

    @staticmethod
    def _make_decoder(
        name: str,
        emb_dim: int,
        n_cats: int,
        correct_option: int = 0,
        state_dim: Optional[int] = None,
        item_key_dim: Optional[int] = None,
        alpha_log_scale: Optional[float] = None,
        alpha_pos_map: Optional[str] = None,
        alpha_pos_kwargs: Optional[dict] = None,
        state_alpha: bool = True,
        state_beta: bool = False,
    ) -> nn.Module:
        """GPCM/2PL STATIC-DECOUPLED (item_key_dim set, no dynamic head) ->
        Option-A alpha-key decoder.

        The alpha-key decoder puts the STATIC slope on the wide item key (Option A,
        decision 2), so the decoupled-static arm genuinely gives the slope its own
        channel.  The DYNAMIC arm (``state_alpha``/``state_beta``) is left on the
        CORE decoder, whose ``fc_a_state`` already reads ``[state, item_key]`` (the
        slope already rides the key when dynamic) AND which passes the core
        recovery's isinstance checks.  For ``item_key_dim=None`` (shared /
        benchmark arms) and the bt/nrm decoders, defer to the parent factory --
        byte-identical to the base.  This makes ONE decoder per corner of the clean
        {slope-channel} x {slope-dynamic} 2x2.
        """
        if (name in ("gpcm", "binary") and item_key_dim is not None
                and not state_alpha and not state_beta):
            return build_alpha_key_decoder(
                name,
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
        return DeepIRTModel._make_decoder(
            name, emb_dim, n_cats, correct_option, state_dim, item_key_dim,
            alpha_log_scale, alpha_pos_map, alpha_pos_kwargs, state_alpha, state_beta,
        )

    # ------------------------------------------------------------------
    # Recovery: static alpha-key decoders bypass the base isinstance guard
    # ------------------------------------------------------------------

    def recover_item_params(
        self,
        item_ids: Optional[Tensor] = None,
        responses: Optional[Tensor] = None,
        use_extended: bool = False,
    ) -> dict:
        """Recover item params; handle the static Option-A alpha-key decoders.

        The base static recovery asserts ``isinstance(decoder, (GPCMDecoder,
        Binary2PLDecoder))``.  ``Binary2PLAlphaKeyDecoder`` is neither (it WRAPS a
        GPCMAlphaKeyDecoder), so for the static-decoupled arm we recover directly
        -- read the item value + wide key and call the decoder's
        ``item_params_sorted`` (duck-typed, identical numbers to the base path for
        the GPCM alpha-key which the base would have accepted).  Every other case
        (shared static, any dynamic head, bt/nrm) defers to the parent unchanged.
        """
        if (not self._uses_state and self.item_key_dim is not None
                and self.decoder_name in ("gpcm", "binary")):
            enc = self._get_encoder(use_extended)
            if item_ids is None:
                ids = torch.arange(enc.num_items, device=self.device)
            else:
                ids = item_ids.to(self.device)
            enc.eval()
            self.decoder.eval()
            with torch.no_grad():
                embs = enc.item_val_emb(ids)
                item_key = enc.item_key_emb(ids)
                params = self.decoder.item_params_sorted(embs, item_key=item_key)
            return {
                "alpha": params["alpha"].squeeze(-1).cpu().numpy(),
                "beta": params["beta"].cpu().numpy(),
            }
        return super().recover_item_params(item_ids, responses, use_extended)

    # ------------------------------------------------------------------
    # Halting early-stop fit (the base callback only notifies, cannot stop)
    # ------------------------------------------------------------------

    def fit_early_stop(
        self,
        item_ids: Tensor,
        responses: Tensor,
        val_item_ids: Tensor,
        val_responses: Tensor,
        n_epochs: int = 150,
        lr: float = 1e-2,
        mask: Optional[Tensor] = None,
        val_mask: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        grad_clip_norm: Optional[float] = None,
        patience: Optional[int] = 10,
        min_delta: float = 0.0,
        verbose: bool = False,
    ) -> dict:
        """Train with a halting early-stopping rule and restore the best weights.

        The training step is byte-identical to :meth:`DeepIRTModel.fit` (same seed,
        same CPU shuffle generator, same grad-clip, same full-/mini-batch logic).
        After each epoch the monitored metric is ``_compute_loss`` on the inner-val
        slice (``val_item_ids``/``val_responses``, masked by ``val_mask``); the
        best-so-far encoder+decoder state is cached and restored at the end.

        Stopping: STOP after ``patience`` epochs with no improvement over the best
        val loss by more than ``min_delta``.  ``patience=None`` runs the full
        ``n_epochs`` but still restores the best weights.

        Returns a dict: ``best_epoch``, ``best_val_loss``, ``stopped_epoch``,
        ``final_train_loss``, ``train_time``.
        """
        if self.decoder_name == "bt":
            raise RuntimeError("BT decoder requires pairwise data; use fit_pairs().")
        if grad_clip_norm is not None and grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided.")

        torch.manual_seed(self.seed)
        item_ids = item_ids.to(self.device)
        responses = responses.to(self.device)
        val_item_ids = val_item_ids.to(self.device)
        val_responses = val_responses.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if val_mask is not None:
            val_mask = val_mask.to(self.device)

        # Loss built from the TRAINING responses (weights match the data for WOL;
        # a no-op for the weight-free ordinal CE).
        self.loss_fn = self._build_loss_fn(responses, mask)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + 1)

        N = item_ids.size(0)
        full_batch = batch_size is None or batch_size >= N

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        bad = 0
        stopped_epoch = n_epochs
        final_train_loss = float("nan")
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            self.encoder.train()
            self.decoder.train()

            if full_batch:
                optimizer.zero_grad()
                loss = self._compute_loss(item_ids, responses, mask=mask)
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip_norm)
                optimizer.step()
                final_train_loss = loss.item()
            else:
                perm = torch.randperm(N, generator=rng)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    idx = perm[start : start + batch_size]
                    optimizer.zero_grad()
                    loss = self._compute_loss(
                        item_ids[idx], responses[idx],
                        mask=mask[idx] if mask is not None else None,
                    )
                    loss.backward()
                    if grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(params, grad_clip_norm)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                final_train_loss = epoch_loss / n_batches

            # --- monitor: prediction loss on the inner-val slice ---
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                val_loss = self._compute_loss(
                    val_item_ids, val_responses, mask=val_mask
                ).item()

            improved = val_loss < best_val - min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                bad = 0
                best_state = {
                    "enc": copy.deepcopy(self.encoder.state_dict()),
                    "dec": copy.deepcopy(self.decoder.state_dict()),
                }
            else:
                bad += 1

            if verbose and (epoch % 25 == 0 or epoch == 1):
                print(f"  [ES] epoch {epoch:4d}  train={final_train_loss:.4f} "
                      f"val={val_loss:.4f}  best={best_val:.4f}@{best_epoch}")

            if patience is not None and bad >= patience:
                stopped_epoch = epoch
                break

        # Restore the best-so-far weights (restore-best is part of the rule).
        if best_state is not None:
            self.encoder.load_state_dict(best_state["enc"])
            self.decoder.load_state_dict(best_state["dec"])

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "stopped_epoch": stopped_epoch,
            "final_train_loss": final_train_loss,
            "train_time": time.time() - t0,
        }
