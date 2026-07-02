"""_p2_engine.py -- _P2Engine: DeepIRTEngine that builds _P2Model + masks the tail.

Phase-2 INTEGRATION build (blueprint build order step 1).  A thin
``DeepIRTEngine`` subclass (reuse, not rewrite -- the tensor plumbing, the
recovery, the cost accounting are all inherited):

1. Builds a :class:`_P2Model` (proper GPCM ordinal CE + a halting early-stop fit)
   instead of the base ``DeepIRTModel``.  The base engine hardcodes the model
   class, so we let it construct once and then rebuild the model as ``_P2Model``
   with the SAME resolved args + ``gpcm_loss``.  Because ``_P2Model.__init__``
   re-seeds ``torch.manual_seed(seed)`` before building its encoder/decoder, the
   rebuilt weights are deterministic and identical to a fresh ``_P2Model``.

2. ``fit`` threads a per-sequence validity MASK to the model (the variable-length
   leg) and, when a patience is given, carves an inner-val slice from the TRAINING
   learners and drives ``_P2Model.fit_early_stop`` (a real halt).  The inner-val
   slice is disjoint from the CV held-out fold, so early stopping never peeks at
   the scored data.

3. ``predict_heldout`` scores the last ``n_holdout`` VALID positions per learner
   (not the last ``h`` array positions), so tail padding on variable-length
   sequences is skipped.  With no mask (the rectangular clean bed) it is
   byte-identical to the base method.

Scratch file: ``_``-prefixed, gitignored.  Subclasses the read-only core engine;
does not edit it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench._p2_model import _P2Model


class _P2Engine(DeepIRTEngine):
    """DeepIRTEngine variant that constructs a :class:`_P2Model`.

    Parameters
    ----------
    gpcm_loss : forwarded to :class:`_P2Model` ("ordinal_ce" | "wol").
    inner_val_frac : fraction of TRAIN learners carved as the early-stop
        inner-val slice (disjoint from the CV held-out fold).
    Other kwargs match :class:`DeepIRTEngine.__init__`.
    """

    def __init__(self, ds, gpcm_loss: str = "ordinal_ce",
                 inner_val_frac: float = 0.15,
                 nrm_channel: str = "shared", **kwargs) -> None:
        self.gpcm_loss = gpcm_loss
        self.inner_val_frac = float(inner_val_frac)

        # NRM channel routing.  For decoder == "nrm" the x-mode (static vs
        # dynamic option readout) rides the config's ``state_alpha``, but the core
        # DeepIRTModel FORBIDS state_alpha on NRM (it is a gpcm/binary head).  So
        # capture the x-mode here, neutralize ``state_alpha`` before the base
        # build, and hand the channel decoder its OWN wide key (item_key_dim=None
        # to the core, so the encoder builds no dead key table -- the channel
        # decoder owns the wide-key tables).  The real key width is threaded to
        # the channel-decoder factory via ``nrm_item_key_dim``.
        is_nrm = kwargs.get("decoder", "gpcm") == "nrm"
        nrm_dynamic = False
        nrm_item_key_dim = kwargs.get("item_key_dim", None)
        if is_nrm:
            nrm_dynamic = bool(kwargs.get("state_alpha", False))
            kwargs["state_alpha"] = False
            kwargs["item_key_dim"] = None

        # Let the base engine set up labels/attrs and build a DeepIRTModel...
        super().__init__(ds=ds, **kwargs)
        # ...then rebuild the model as _P2Model with the SAME resolved config.
        # _P2Model re-seeds internally, so this is deterministic (no RNG drift).
        old = self.model
        self.model = _P2Model(
            num_items=old.num_items,
            emb_dim=old.emb_dim,
            hidden_dim=old.hidden_dim,
            n_cats=old.n_cats,
            decoder=old.decoder_name,
            correct_option=old.correct_option,
            state_alpha=old.state_alpha,
            item_key_dim=old.item_key_dim,
            alpha_log_scale=old.alpha_log_scale,
            alpha_pos_map=old.alpha_pos_map,
            alpha_pos_kwargs=old.alpha_pos_kwargs,
            state_beta=old.state_beta,
            decouple=False,
            encoder=old.encoder_name,
            encoder_kwargs=old.encoder_kwargs,
            device=self.device,
            seed=self.seed,
            gpcm_loss=self.gpcm_loss,
            nrm_channel=(nrm_channel if is_nrm else None),
            nrm_dynamic=nrm_dynamic,
            nrm_item_key_dim=nrm_item_key_dim,
        )
        self.label = self.label.replace("deep_irt-", "p2-", 1)
        # Channel-mode markers for predict/recover routing on the engine.
        self._nrm_channel = nrm_channel if is_nrm else None
        self._nrm_dynamic = nrm_dynamic

    # ------------------------------------------------------------------
    # Per-sequence validity mask (variable-length leg; None on the clean bed)
    # ------------------------------------------------------------------

    def _seq_mask(self) -> Optional[torch.Tensor]:
        """Return the full (N, T) bool validity mask, or None if the bed is clean.

        Looked up on ``ds.mask`` first, then ``ds.aux['mask']`` -- the two places
        the step-6a realistic generator (_p2_datagen_realistic, a separate build)
        may emit it.  True = valid position, False = tail padding.
        """
        m = getattr(self.ds, "mask", None)
        if m is None:
            aux = getattr(self.ds, "aux", None)
            if isinstance(aux, dict):
                m = aux.get("mask")
        if m is None:
            return None
        return torch.as_tensor(np.asarray(m), dtype=torch.bool)

    # ------------------------------------------------------------------
    # Fit: mask-aware + optional halting early stop
    # ------------------------------------------------------------------

    def fit(
        self,
        n_epochs: int = 150,
        lr: float = 1e-2,
        batch_size: Optional[int] = None,
        callback=None,
        grad_clip_norm: Optional[float] = None,
        patience: Optional[int] = None,
        verbose: bool = False,
    ):
        """Train on the train learners.

        ``patience is None`` -> the base masked ``fit`` (runs all epochs).
        ``patience`` set    -> carve an inner-val slice from the train learners and
                               drive ``_P2Model.fit_early_stop`` (a real halt with
                               restore-best).  The inner-val slice is disjoint from
                               the CV held-out fold, so no scored data leaks in.
        """
        import time

        it, rp = self._tensor(self.ds.train_idx)
        seq_mask = self._seq_mask()
        train_mask = None if seq_mask is None else seq_mask[self.ds.train_idx]

        t0 = time.time()
        if patience is None:
            # Base path + mask (callback still honored for parity).
            self.model.fit(
                it, rp, n_epochs=n_epochs, lr=lr, verbose=verbose,
                mask=train_mask, batch_size=batch_size, callback=callback,
                grad_clip_norm=grad_clip_norm,
            )
        else:
            n = it.size(0)
            g = np.random.default_rng(self.seed + 777)
            perm = g.permutation(n)
            n_val = max(1, int(round(n * self.inner_val_frac)))
            n_val = min(n_val, n - 1)  # keep at least one training learner
            val_rows = np.sort(perm[:n_val])
            tr_rows = np.sort(perm[n_val:])
            m_tr = None if train_mask is None else train_mask[tr_rows]
            m_va = None if train_mask is None else train_mask[val_rows]
            self.model.fit_early_stop(
                it[tr_rows], rp[tr_rows], it[val_rows], rp[val_rows],
                n_epochs=n_epochs, lr=lr, mask=m_tr, val_mask=m_va,
                batch_size=batch_size, grad_clip_norm=grad_clip_norm,
                patience=patience, verbose=verbose,
            )
        self._train_time = time.time() - t0
        return self

    # ------------------------------------------------------------------
    # Predict: last n_holdout VALID positions per learner
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_heldout(self):
        """probs (M, K), targets (M,) on the val learners' last h VALID positions.

        Mirrors :meth:`DeepIRTEngine.predict_heldout` to build the full per-step
        probability tensor, then selects the last ``n_holdout`` VALID positions per
        learner (ragged when lengths differ).  On the rectangular clean bed (no
        mask) every position is valid, so the selection is exactly the base
        ``[:, T-h:]`` tail.
        """
        if self._nrm_channel is not None:
            return self._predict_heldout_nrm_channel()

        K = self.ds.cfg.n_cats
        T = self.ds.cfg.seq_len
        h = self.ds.cfg.n_holdout
        it, rp = self._tensor(self.ds.val_idx)
        it = it.to(self.device); rp = rp.to(self.device)
        embs = self.model.encoder.item_val_emb(it)                 # (B,T,emb)

        if self._uses_state:
            theta_in, state_in = self.model.encoder.aligned_theta_and_state(it, rp)
            item_keys = (self.model.encoder.item_key_emb(it)
                         if self.item_key_dim is not None else None)
        else:
            theta_in = self.model.encoder.theta_for_prediction(it, rp)
            state_in = None
            item_keys = (self.model.encoder.item_key_emb(it)
                         if self.item_key_dim is not None else None)

        is_nrm = self.decoder_name == "nrm"
        probs_full = []
        for t in range(T):
            state_t = state_in[:, t, :] if state_in is not None else None
            key_t = item_keys[:, t, :] if item_keys is not None else None
            if is_nrm:
                # NRM item_params has no state kwarg (options are never
                # state-conditioned in the core decoder) and returns intercept.
                params = self.model.decoder.item_params(embs[:, t, :], item_key=key_t)
            else:
                params = self.model.decoder.item_params(
                    embs[:, t, :], state=state_t, item_key=key_t)
            lp = self.model.decoder.log_probs(theta_in[:, t], **params)  # (B,K)
            probs_full.append(lp.exp().unsqueeze(1))
        probs_full = torch.cat(probs_full, dim=1)                  # (B,T,K)
        self._infer_probs = probs_full

        # Score mask: the last h VALID positions per learner.
        seq_mask = self._seq_mask()
        if seq_mask is None:
            valid = torch.ones(it.shape, dtype=torch.bool, device=self.device)
        else:
            valid = seq_mask[self.ds.val_idx].to(self.device)      # (B,T)
        csum = torch.cumsum(valid.long(), dim=1)                   # 1-based valid rank
        L = csum[:, -1:]                                           # (B,1) valid length
        score = valid & (csum > (L - h))                          # (B,T) bool

        probs_tail = probs_full[score].reshape(-1, K).cpu().numpy()
        targ_tail = rp[score].reshape(-1).cpu().numpy()
        return probs_tail, targ_tail

    # ------------------------------------------------------------------
    # Predict for the NRM channel decoder (its own (B, T) forward surface)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_heldout_nrm_channel(self):
        """probs (M, K), targets (M,) via the channel decoder's (B, T) forward.

        The channel decoder reads a_k/c_k from the item ids and returns the full
        (B, T, K) option logits in one call, so there is no per-step loop.  Static
        couplings pass ``state=None``; dynamic couplings pass the aligned state.
        The same last-``n_holdout``-VALID selection as the GPCM path is applied.
        """
        K = self.ds.cfg.n_cats
        h = self.ds.cfg.n_holdout
        it, rp = self._tensor(self.ds.val_idx)
        it = it.to(self.device); rp = rp.to(self.device)

        item_val = self.model.encoder.item_val_emb(it)             # (B,T,emb)
        if self._nrm_dynamic:
            theta_in, state_in = self.model.encoder.aligned_theta_and_state(it, rp)
        else:
            theta_in = self.model.encoder.theta_for_prediction(it, rp)
            state_in = None
        logits = self.model.decoder(theta_in, item_val, it, state=state_in)  # (B,T,K)
        probs_full = torch.softmax(logits, dim=-1)                 # (B,T,K)
        self._infer_probs = probs_full

        seq_mask = self._seq_mask()
        if seq_mask is None:
            valid = torch.ones(it.shape, dtype=torch.bool, device=self.device)
        else:
            valid = seq_mask[self.ds.val_idx].to(self.device)      # (B,T)
        csum = torch.cumsum(valid.long(), dim=1)
        L = csum[:, -1:]
        score = valid & (csum > (L - h))

        probs_tail = probs_full[score].reshape(-1, K).cpu().numpy()
        targ_tail = rp[score].reshape(-1).cpu().numpy()
        return probs_tail, targ_tail
