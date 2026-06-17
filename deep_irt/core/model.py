"""deep_irt.py -- Top-level API tying encoder, decoder, and anchoring together.

The DeepIRTModel class provides a clean, high-level interface:

    model = DeepIRTModel(decoder="gpcm", ...)
    model.fit(item_ids, responses, n_epochs=300)
    theta = model.track(item_ids, responses)        # (N, T) trajectory
    params = model.recover_item_params(item_ids)    # dict of item parameters
    model.extend(new_items_count, anchor_theta, ext_item_ids, ext_responses)

Selectable decoders (pass decoder="gpcm"|"binary"|"bt"|"nrm" to __init__):

    "gpcm"   -- ordinal K-category GPCM  (default)
    "binary" -- binary 2PL (GPCM at K=2)
    "bt"     -- Bradley-Terry pairwise (item-only; fit with fit_pairs())
    "nrm"    -- Bock nominal response model (unordered K options; multiple-choice)

The organic single-pass engine
------------------------------
The encoder is run ONCE per forward (no double pass).  Theta comes from the
direct responsive stream (h_t sees the current interaction; the reported dynamic
track).  The SINGLE-SHIFT alignment contract (encoder.theta_for_prediction /
state_for_prediction) means the theta used to predict step t is always a
function of history strictly before t.

  state_alpha : a clean optional DECODER feature (gpcm/binary only).  When True
      the GPCM/binary decoder reads discrimination from
      ``[state, item_val_emb]`` -- where ``state`` is the SAME prediction-aligned
      hidden the theta is read from (encoder.state_for_prediction) -- in the
      spirit of ma-irt's IRTParameterExtractor.  Discrimination is then
      occurrence-averaged per item at recovery.  Default False keeps the static
      item-only alpha map and is bit-for-bit identical to the static decoder.
      Capacity-tunable through emb_dim / hidden_dim.  No second encoder stream:
      one LSTM, one pass, one shift.

Anchored extension (extend()) works with "gpcm" and "binary" decoders.
The "bt" decoder can be calibrated separately via fit_pairs().
"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from deep_irt.core.encoder import LSTMEncoder
from deep_irt.core.decoders import (
    GPCMDecoder,
    Binary2PLDecoder,
    BradleyTerryDecoder,
    NRMDecoder,
)
from deep_irt.core.anchor import anchored_extend, build_extended_encoder
from deep_irt.core.losses import CombinedLoss, compute_class_weights


# ---------------------------------------------------------------------------
# DeepIRTModel
# ---------------------------------------------------------------------------

class DeepIRTModel:
    """
    Dynamic assessment deep_irt: deep encoder + swappable decoder.

    Parameters
    ----------
    num_items   : int -- initial (base) item bank size B
    emb_dim     : int -- item / response embedding dimension
    hidden_dim  : int -- LSTM hidden size
    n_cats      : int -- response categories K (ordered for gpcm/binary,
                         unordered options for "nrm"; ignored for "bt")
    decoder     : str -- "gpcm" | "binary" | "bt" | "nrm"
    correct_option : int -- for "nrm", index of the designated-correct option
                            (default 0); ignored by other decoders
    state_alpha : bool -- when True (gpcm/binary only), discrimination is read
                            from the prediction-aligned encoder state plus the
                            item embedding (ma-irt's IRTParameterExtractor
                            recipe) and occurrence-averaged per item at recovery.
                            Default False keeps the static item-only alpha map
                            and is bit-for-bit identical.
    item_key_dim : int or None -- the DECOUPLED variant (requires state_alpha;
                            gpcm/binary only).  When set, the encoder gets a
                            SEPARATE wide item KEY table of this width that feeds
                            ONLY the decoder's static alpha and beta readouts, NOT
                            the LSTM input.  This decouples the static item
                            params' capacity from theta's: keep emb_dim/hidden_dim
                            cheap (theta unchanged) while alpha and beta read a
                            wide item key (so beta recovers far better, ma-irt's
                            shared-key path).  None (default) keeps alpha and beta
                            on the thin item value and is bit-for-bit identical to
                            plain state_alpha.
    alpha_log_scale : float or None -- discrimination positivity transform for
                            the GPCM/binary decoder.  None (default) uses softplus
                            and is bit-for-bit identical.  A positive float ``s``
                            uses ``exp(s * raw)``, ma-irt's exponential map
                            (``s = 1.0`` matches ma-irt; the MLP head absorbs the
                            scale constant).
    state_beta  : bool -- the dynamic-beta arm (gpcm/binary only; requires
                            state_alpha=True AND item_key_dim).  When True the step
                            thresholds are read from a state-conditioned head
                            ``fc_b_state([state, item_key])`` -- the EXACT mirror
                            of the state_alpha discrimination head -- and
                            occurrence-averaged per item at recovery (instead of
                            the static item-key read).  This makes difficulty
                            per-occurrence so the alpha-dynamic vs beta-dynamic
                            comparison is clean.  Default False keeps the static
                            beta and is bit-for-bit identical.
    decouple    : bool -- when True (DEFAULT) the GPCM/binary model uses the
                            decoupled architecture (state_alpha + a separate
                            item_key_dim=64 wide item key + exp link), the
                            validated deep-irt s_0.  Applied ONLY when none of
                            state_alpha / item_key_dim / alpha_log_scale is set
                            explicitly (passing any one defers to manual config);
                            a no-op for the bt/nrm decoders.  Pass decouple=False
                            for the plain (legacy) GPCM path.
    device      : torch.device
    seed        : int
    """

    _DECODER_CHOICES = ("gpcm", "binary", "bt", "nrm")
    _ENCODER_CHOICES = ("lstm", "transformer", "dkvmn")

    def __init__(
        self,
        num_items: int,
        emb_dim: int = 8,
        hidden_dim: int = 32,
        n_cats: int = 4,
        decoder: str = "gpcm",
        correct_option: int = 0,
        state_alpha: Optional[bool] = None,
        item_key_dim: Optional[int] = None,
        alpha_log_scale: Optional[float] = None,
        state_beta: bool = False,
        decouple: bool = True,
        encoder: str = "lstm",
        encoder_kwargs: Optional[dict] = None,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        ordinal_penalty: float = 0.5,
        class_weight_strategy: str = "sqrt_balanced",
    ) -> None:
        if decoder not in self._DECODER_CHOICES:
            raise ValueError(
                f"decoder must be one of {self._DECODER_CHOICES}, got '{decoder}'"
            )
        if encoder not in self._ENCODER_CHOICES:
            raise ValueError(
                f"encoder must be one of {self._ENCODER_CHOICES}, got '{encoder}'"
            )
        # Decoupling is the default deep-irt architecture.  ``decouple=True``
        # (default) turns on state-conditioned alpha + a separate wide item key
        # table (item_key_dim=64) + the exp link, but ONLY for the alpha-capable
        # GPCM/binary decoders and ONLY when the user has set NONE of the alpha
        # knobs explicitly (passing any one is a manual config, and decouple
        # defers).  For bt/nrm it is a no-op.  Pass decouple=False for the plain
        # (legacy) path.
        if state_alpha is None:
            if (decouple and decoder in ("gpcm", "binary")
                    and item_key_dim is None and alpha_log_scale is None):
                state_alpha = True
                item_key_dim = 64
                alpha_log_scale = 1.0
            else:
                state_alpha = False
        if state_alpha and decoder not in ("gpcm", "binary"):
            raise ValueError(
                "state_alpha is defined for the GPCM/binary decoders (it routes "
                f"a state-conditioned discrimination), not decoder='{decoder}'."
            )
        # ``item_key_dim`` (the wide item key) feeds the static step-threshold
        # head and, when a state is present, the state-conditioned alpha/beta
        # heads.  It is therefore valid on its own (a capacity-matched all-static
        # decoupled config: static alpha on the thin value, static beta on the
        # wide key), so no state head is required.
        if alpha_log_scale is not None and decoder not in ("gpcm", "binary"):
            raise ValueError(
                "alpha_log_scale (the exp discrimination transform) is defined for "
                f"the GPCM/binary decoders, not decoder='{decoder}'."
            )
        if state_beta:
            if decoder not in ("gpcm", "binary"):
                raise ValueError(
                    "state_beta (state-conditioned step thresholds) is defined "
                    f"for the GPCM/binary decoders, not decoder='{decoder}'."
                )
            if item_key_dim is None:
                raise ValueError(
                    "state_beta requires item_key_dim (the decoupled wide item "
                    "key); the head reads [state, item_key], the exact mirror of "
                    "the state_alpha head."
                )
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_cats = n_cats
        self.decoder_name = decoder
        self.correct_option = correct_option
        self.state_alpha = state_alpha
        self.item_key_dim = item_key_dim
        self.alpha_log_scale = alpha_log_scale
        self.state_beta = state_beta
        self.decouple = decouple
        self.encoder_name = encoder
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.device = device
        self.seed = seed
        # Prediction-loss config (IRT is the readout flavor; the loss scores
        # the decoder's logits).  gpcm -> WeightedOrdinalLoss(ordinal_penalty,
        # sqrt-balanced weights); nrm -> plain CE; binary -> BCE.  Built per fit.
        self.ordinal_penalty = ordinal_penalty
        self.class_weight_strategy = class_weight_strategy
        # Built now (uniform class weights) so _compute_loss works pre-fit;
        # fit() rebuilds it with data-driven sqrt-balanced weights.
        self.loss_fn: Optional[nn.Module] = self._build_loss_fn()

        torch.manual_seed(seed)
        self.encoder = self._make_encoder(
            encoder, num_items, emb_dim, hidden_dim, n_cats,
            item_key_dim, self.encoder_kwargs,
        ).to(device)

        # In state_alpha mode the GPCM/binary decoder gets a state-conditioned
        # alpha head whose state input is the encoder's prediction-aligned state.
        # In the decoupled variant its item key is the wide item_key_emb.
        # The decoder gets a state input whenever EITHER head is state-
        # conditioned (alpha via state_alpha, beta via state_beta).  The explicit
        # ``state_alpha`` flag gates the discrimination head independently, so
        # beta can be dynamic while alpha stays static (and vice versa).
        self._uses_state = bool(state_alpha) or bool(state_beta)
        decoder_state_dim = hidden_dim if self._uses_state else None
        self.decoder: nn.Module = self._make_decoder(
            decoder, emb_dim, n_cats, correct_option, state_dim=decoder_state_dim,
            item_key_dim=item_key_dim, alpha_log_scale=alpha_log_scale,
            state_alpha=bool(state_alpha), state_beta=state_beta,
        )
        self.decoder = self.decoder.to(device)

        # Extended encoder (populated after extend())
        self._extended_encoder: Optional[LSTMEncoder] = None
        self._n_ext: int = 0

    # ------------------------------------------------------------------
    # Fit (base calibration)
    # ------------------------------------------------------------------

    def fit(
        self,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        n_epochs: int = 300,
        lr: float = 1e-2,
        verbose: bool = True,
        mask: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        callback=None,
    ) -> dict:
        """
        Train encoder + decoder on base-item sequences.

        Parameters
        ----------
        item_ids   : (N, T) long -- base item indices {0,..,B-1}
        responses  : (N, T) long -- responses {0,..,K-1}
        n_epochs   : training epochs
        lr         : Adam learning rate
        verbose    : print loss every 50 epochs
        mask       : (N, T) bool, optional -- True for valid positions,
                     False for padding. When None the full rectangular
                     batch is used (bit-for-bit identical to the old path).
        batch_size : int, optional -- mini-batch size over the learner
                     dimension. None (default) means full-batch, which
                     preserves the exact existing behaviour when mask is
                     also None.
        callback   : optional callable, called once per epoch as
                     ``callback(epoch, loss_value, self)`` AFTER the optimizer
                     step (epoch is 1-based, loss_value is the epoch's final/mean
                     loss).  It is fire-and-forget: a raised ``StopIteration`` is
                     swallowed (the loop continues), and any return value is
                     ignored.  This lets a runner recover the model every k epochs
                     WITHOUT re-initializing the optimizer (chunked fit calls
                     would reset Adam).  None (default) = no behavior change,
                     bit-for-bit identical.

        Returns
        -------
        dict with final_nll and train_time
        """
        if self.decoder_name == "bt":
            raise RuntimeError(
                "BT decoder requires pairwise data; use fit_pairs() instead."
            )

        torch.manual_seed(self.seed)
        item_ids = item_ids.to(self.device)
        responses = responses.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Format-keyed prediction loss, built from the training data (the
        # ordinal WOL's class weights come from the response histogram, exactly
        # as ma-irt computes them at train time).
        self.loss_fn = self._build_loss_fn(responses, mask)

        N = item_ids.size(0)
        full_batch = batch_size is None or batch_size >= N

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = optim.Adam(params, lr=lr)

        # Seeded shuffle generator -- separate from the global seed so that
        # the no-batch-size path is numerically unaffected.
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + 1)

        t0 = time.time()
        final_loss = float("nan")

        for epoch in range(1, n_epochs + 1):
            self.encoder.train()
            self.decoder.train()

            if full_batch:
                optimizer.zero_grad()
                loss = self._compute_loss(item_ids, responses, mask=mask)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
            else:
                # Shuffle learner indices each epoch.
                perm = torch.randperm(N, generator=rng)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    idx = perm[start : start + batch_size]
                    b_ids = item_ids[idx]
                    b_resp = responses[idx]
                    b_mask = mask[idx] if mask is not None else None

                    optimizer.zero_grad()
                    loss = self._compute_loss(b_ids, b_resp, mask=b_mask)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                final_loss = epoch_loss / n_batches

            if verbose and (epoch % 50 == 0 or epoch == 1):
                print(f"  [Fit] epoch {epoch:4d}  loss={final_loss:.4f}")

            # Fire-and-forget per-epoch hook (e.g. a runner recovering the model
            # every k epochs).  A raised StopIteration is swallowed so the hook
            # cannot abort training; any return value is ignored.
            if callback is not None:
                try:
                    callback(epoch, final_loss, self)
                except StopIteration:
                    pass

        # "final_nll" kept as a back-compat alias; under the prediction loss it
        # holds the final WOL/CE value, not an IRT-likelihood NLL.
        return {
            "final_loss": final_loss,
            "final_nll": final_loss,
            "train_time": time.time() - t0,
        }

    # ------------------------------------------------------------------
    # Fit pairs (BT decoder)
    # ------------------------------------------------------------------

    def fit_pairs(
        self,
        item_emb_i: torch.Tensor,
        item_emb_j: torch.Tensor,
        outcome: torch.Tensor,
        n_epochs: int = 500,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> dict:
        """
        Train the BT decoder on pairwise comparison data.

        item_emb_i / item_emb_j are precomputed embeddings (frozen encoder output).
        The BT decoder fc_s layer is the only thing trained here.

        Parameters
        ----------
        item_emb_i : (N, emb_dim) precomputed embeddings for "winner" item
        item_emb_j : (N, emb_dim) precomputed embeddings for "loser" candidate
        outcome    : (N,) long {0, 1} -- 1 if i beats j
        n_epochs, lr, verbose: training controls

        Returns
        -------
        dict with final_nll and train_time
        """
        if self.decoder_name != "bt":
            raise RuntimeError("fit_pairs() only works with decoder='bt'.")

        assert isinstance(self.decoder, BradleyTerryDecoder)

        torch.manual_seed(self.seed)
        item_emb_i = item_emb_i.to(self.device)
        item_emb_j = item_emb_j.to(self.device)
        outcome = outcome.to(self.device)

        optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        t0 = time.time()
        final_nll = float("nan")

        for epoch in range(1, n_epochs + 1):
            self.decoder.train()
            optimizer.zero_grad()
            loss = self.decoder.nll_pairs(item_emb_i, item_emb_j, outcome)
            loss.backward()
            optimizer.step()
            final_nll = loss.item()
            if verbose and (epoch % 100 == 0 or epoch == 1):
                print(f"  [BT fit] epoch {epoch:4d}  BCE={final_nll:.4f}")

        return {"final_nll": final_nll, "train_time": time.time() - t0}

    # ------------------------------------------------------------------
    # Track (per-step theta trajectory)
    # ------------------------------------------------------------------

    def track(
        self,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        use_extended: bool = False,
    ) -> torch.Tensor:
        """
        Return per-step latent theta trajectory (the RAW responsive theta).

        Parameters
        ----------
        item_ids      : (N, T) long
        responses     : (N, T) long
        use_extended  : if True, use the extended encoder (after extend() call)

        Returns
        -------
        theta : (N, T) float -- per-step ability estimates
        """
        enc = self._get_encoder(use_extended)
        enc.eval()
        item_ids = item_ids.to(self.device)
        responses = responses.to(self.device)
        with torch.no_grad():
            return enc.encode(item_ids, responses)

    # ------------------------------------------------------------------
    # Recover item parameters
    # ------------------------------------------------------------------

    def recover_item_params(
        self,
        item_ids: Optional[torch.Tensor] = None,
        responses: Optional[torch.Tensor] = None,
        use_extended: bool = False,
    ) -> dict:
        """
        Recover item parameters from the decoder for a set of item indices.

        Parameters
        ----------
        item_ids      : item indices.  For the static item-only read (default),
                        a 1-D (M,) tensor of item ids (None -> all B/B+E items).
                        In ``state_alpha`` mode this must be the (N, T) sequence
                        of item ids the model was trained on, paired with
                        ``responses``, so the state-conditioned discrimination
                        can be occurrence-averaged.
        responses     : (N, T) long -- required in ``state_alpha`` mode; ignored
                        otherwise.
        use_extended  : use extended encoder's embedding table

        Returns
        -------
        For "gpcm" / "binary": dict with 'a' (Q,) and 'b' (Q, K-1) arrays.  In
            ``state_alpha`` mode also returns 'seen' (Q,) bool for items that
            appeared in the sequences.
        For "bt":              dict with 'strength' (M,) array
        For "nrm":             dict with 'a' (M, K), 'c' (M, K) per-option params
                               and 'beta' (M,) correct-option difficulty
        All values are numpy arrays on cpu, sorted b for GPCM/binary.
        """
        if self._uses_state:
            return self._recover_item_params_state_alpha(item_ids, responses)

        enc = self._get_encoder(use_extended)
        total_items = enc.num_items
        if item_ids is None:
            item_ids = torch.arange(total_items, device=self.device)
        else:
            item_ids = item_ids.to(self.device)

        enc.eval()
        self.decoder.eval()
        with torch.no_grad():
            embs = enc.item_val_emb(item_ids)       # (M, emb_dim)
            if self.decoder_name in ("gpcm", "binary"):
                assert isinstance(self.decoder, (GPCMDecoder, Binary2PLDecoder))
                # All-static decoupled config: the static beta head reads the wide
                # item key, so supply it when it exists.
                item_key = (enc.item_key_emb(item_ids)
                            if self.item_key_dim is not None else None)
                params = self.decoder.item_params_sorted(embs, item_key=item_key)
                return {
                    "alpha": params["alpha"].squeeze(-1).cpu().numpy(),
                    "beta": params["beta"].cpu().numpy(),
                }
            elif self.decoder_name == "nrm":
                assert isinstance(self.decoder, NRMDecoder)
                params = self.decoder.item_params(embs)
                beta = self.decoder.item_difficulty(embs)
                return {
                    "alpha": params["alpha"].cpu().numpy(),
                    "intercept": params["intercept"].cpu().numpy(),
                    "beta": beta.cpu().numpy(),
                }
            else:
                assert isinstance(self.decoder, BradleyTerryDecoder)
                strength = self.decoder.item_strength(embs).squeeze(-1)
                return {"strength": strength.cpu().numpy()}

    def _recover_item_params_state_alpha(
        self,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
    ) -> dict:
        """Recover GPCM item parameters under state-conditioned discrimination.

        Discrimination is read from the state-conditioned head at every
        (learner, step) occurrence -- using the prediction-aligned state
        ``encoder.state_for_prediction`` so each alpha conditions on the history
        before answering that item, the same alignment training uses -- then
        averaged per item over all occurrences, the ma-irt IRTParameterExtractor
        recovery recipe.  Step thresholds (beta) are an item-only read recovered
        directly from each item's embedding.

        Parameters
        ----------
        item_ids  : (N, T) long -- item indices the model was trained on
        responses : (N, T) long -- responses {0,..,K-1}

        Returns
        -------
        dict with:
            a    : (Q,)      per-item discrimination, occurrence-averaged
            b    : (Q, K-1)  per-item sorted step thresholds
            seen : (Q,) bool -- items that actually appeared in the sequences
        """
        if item_ids is None or responses is None:
            raise ValueError(
                "state_alpha recovery needs the (N, T) item_ids AND responses so "
                "the state-conditioned discrimination can be occurrence-averaged. "
                "Pass recover_item_params(item_ids, responses)."
            )
        assert isinstance(self.decoder, (GPCMDecoder, Binary2PLDecoder))

        enc = self.encoder
        enc.eval()
        self.decoder.eval()
        item_ids = item_ids.to(self.device)
        responses = responses.to(self.device)
        N, T = item_ids.shape
        Q = enc.num_items

        with torch.no_grad():
            state_in = enc.state_for_prediction(item_ids, responses)  # (N,T,hidden)
            embs = enc.item_val_emb(item_ids)                        # (N,T,emb)
            if self.item_key_dim is not None:
                item_keys = enc.item_key_emb(item_ids)              # (N,T,item_key)
                key_flat = item_keys.reshape(N * T, self.item_key_dim)
            else:
                key_flat = None
            params = self.decoder.item_params(
                embs.reshape(N * T, self.emb_dim),
                state=state_in.reshape(N * T, self.hidden_dim),
                item_key=key_flat,
            )
            alpha = params["alpha"].squeeze(-1).reshape(N, T).cpu().numpy()  # (N, T)

            if self.state_beta:
                # Dynamic beta: the per-occurrence state-conditioned step
                # thresholds come straight from the same item_params call (the
                # exact mirror of alpha) and are occurrence-averaged below.  Sort
                # AFTER averaging so the recovered thresholds are ascending.
                beta_occ = params["beta"].reshape(N, T, self.n_cats - 1)
                beta_occ = beta_occ.cpu().numpy()                   # (N, T, K-1)
            else:
                # Static beta, occurrence-invariant (an item-only read).  Beta
                # reads the wide item key whenever it exists (decoupled, ma-irt's
                # shared-key path), else the thin item value.
                all_ids = torch.arange(Q, device=self.device)
                if self.item_key_dim is not None:
                    b_all = self.decoder.item_params_sorted(
                        enc.item_val_emb(all_ids),
                        item_key=enc.item_key_emb(all_ids),
                    )["beta"]
                else:
                    b_all = self.decoder.item_params_sorted(
                        enc.item_val_emb(all_ids))["beta"]
                b_hat = b_all.cpu().numpy()                          # (Q, K-1)

        items = item_ids.cpu().numpy()                              # (N, T)
        a_sum = np.zeros(Q)
        a_cnt = np.zeros(Q)
        np.add.at(a_sum, items.ravel(), alpha.ravel())
        np.add.at(a_cnt, items.ravel(), 1.0)
        a_hat = a_sum / np.maximum(a_cnt, 1.0)
        seen = a_cnt > 0

        if self.state_beta:
            # Occurrence-average the state-conditioned thresholds per item, then
            # sort ascending -- the same recovery recipe alpha uses, applied to
            # the K-1 threshold vector.
            K1 = self.n_cats - 1
            b_sum = np.zeros((Q, K1))
            np.add.at(b_sum, items.ravel(), beta_occ.reshape(N * T, K1))
            b_hat = b_sum / np.maximum(a_cnt[:, None], 1.0)         # (Q, K-1)
            b_hat = np.sort(b_hat, axis=1)
        return {"alpha": a_hat, "beta": b_hat, "seen": seen}

    # ------------------------------------------------------------------
    # Extend item bank (anchored)
    # ------------------------------------------------------------------

    def extend(
        self,
        n_ext: int,
        anchor_theta: torch.Tensor,
        ext_item_ids: torch.Tensor,
        ext_responses: torch.Tensor,
        n_epochs: int = 300,
        lr: float = 1e-2,
        verbose: bool = True,
    ) -> dict:
        """
        Anchored extension: add E new items without disturbing the base model.

        Freezes encoder and decoder, fits E new embedding rows to minimise
        decoder NLL against anchor learners' E-item responses at their
        pinned theta values.

        Parameters
        ----------
        n_ext          : E -- number of new items
        anchor_theta   : (n_anchor,) fixed abilities for anchor learners (cpu or device)
        ext_item_ids   : (n_anchor, E) local indices {0,..,E-1}
        ext_responses  : (n_anchor, E) responses
        n_epochs, lr, verbose : training controls

        Returns
        -------
        dict with ext_emb_weight, n_params, train_time, final_nll,
             and a_ext / b_ext (GPCM/binary) or strength_ext (bt, from embedding)
        """
        if self.decoder_name == "bt":
            raise NotImplementedError(
                "Anchored extension is defined for GPCM/binary decoders. "
                "For BT, call fit_pairs() with new-item embeddings after "
                "prepending them to the embedding table manually."
            )
        if self.encoder_name != "lstm":
            raise NotImplementedError(
                "Anchored extension currently supports only the LSTM backbone "
                f"(build_extended_encoder is LSTM-specific), not "
                f"encoder='{self.encoder_name}'."
            )

        result = anchored_extend(
            encoder=self.encoder,
            decoder=self.decoder,
            anchor_theta=anchor_theta.to(self.device),
            ext_item_ids=ext_item_ids,
            ext_responses=ext_responses,
            n_epochs=n_epochs,
            lr=lr,
            device=self.device,
            verbose=verbose,
        )

        self._n_ext = n_ext
        self._extended_encoder = build_extended_encoder(
            base_encoder=self.encoder,
            ext_emb_weight=result["ext_emb_weight"],
            device=self.device,
        )

        # Recover E-item params from the fitted embeddings via the frozen decoder
        with torch.no_grad():
            ext_embs = result["ext_emb_weight"].to(self.device)
            assert isinstance(self.decoder, (GPCMDecoder, Binary2PLDecoder))
            params = self.decoder.item_params_sorted(ext_embs)
            result["a_ext"] = params["alpha"].squeeze(-1).cpu().numpy()
            result["b_ext"] = params["beta"].cpu().numpy()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_encoder(self, use_extended: bool) -> LSTMEncoder:
        if use_extended:
            if self._extended_encoder is None:
                raise RuntimeError("Call extend() before using use_extended=True.")
            return self._extended_encoder
        return self.encoder

    def _build_loss_fn(
        self,
        responses: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Optional[nn.Module]:
        """Build the format-keyed prediction loss (ma-irt's recipe).

        gpcm (ordinal) -> ``CombinedLoss`` = ``WeightedOrdinalLoss`` with
            ``ordinal_penalty`` and sqrt-balanced class weights.  ``fit`` passes
            the training responses so the weights match the data, exactly as
            ma-irt computes them at train time; with ``responses=None`` (the
            __init__ default) the weights are uniform until ``fit`` rebuilds.
        nrm (nominal)  -> ``CombinedLoss(weighted_ordinal_weight=0)`` = plain CE
            (no ordinal penalty: the options carry no order).
        binary         -> None; BCE is applied inline in ``_predict_loss``.
        """
        if self.decoder_name == "binary":
            return None
        if self.decoder_name == "nrm":
            return CombinedLoss(
                self.n_cats, weighted_ordinal_weight=0.0
            ).to(self.device)
        # gpcm: WeightedOrdinalLoss; class weights from the data when available.
        if responses is None:
            class_weights = None
        else:
            valid = (responses[mask.bool()] if mask is not None
                     else responses.reshape(-1))
            class_weights = compute_class_weights(
                valid, self.n_cats, strategy=self.class_weight_strategy,
                device=self.device,
            )
        return CombinedLoss(
            self.n_cats, class_weights=class_weights,
            weighted_ordinal_weight=1.0, ordinal_penalty=self.ordinal_penalty,
        ).to(self.device)

    def _compute_loss(
        self,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        mask: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: encoder -> aligned theta (+ state) -> prediction loss.

        IRT is the readout FLAVOR.  The decoder emits per-category logits and a
        separate, format-keyed prediction loss (``self.loss_fn``: ma-irt's
        WeightedOrdinalLoss for gpcm, plain CE for nrm; BCE inline for binary)
        scores them.  No IRT-likelihood NLL is optimized in training.

        Parameters
        ----------
        item_ids  : (B, T) long
        responses : (B, T) long
        mask      : (B, T) bool, optional -- True for valid positions.  When
                    None, every position is included.

        The theta used to predict step t is the single-shift aligned theta (a
        function of history strictly before t), for both the direct and
        item-blind paths.  When ``state_alpha`` is on, discrimination is read
        from the SAME aligned hidden, so both come from ONE encoder pass.
        """
        batch, T = item_ids.shape
        embs = self.encoder.item_val_emb(item_ids)                  # (B, T, emb_dim)
        embs_flat = embs.view(batch * T, self.emb_dim)
        resp_flat = responses.reshape(batch * T)

        if self._uses_state:
            # One aligned hidden -> theta and the alpha/beta state, no second
            # pass.  Used whenever EITHER head is state-conditioned (state_alpha
            # for discrimination, state_beta for the step thresholds).
            theta_in, state_in = self.encoder.aligned_theta_and_state(
                item_ids, responses
            )
            theta_flat = theta_in.reshape(batch * T)
            state_flat = state_in.reshape(batch * T, self.hidden_dim)
            if self.item_key_dim is not None:
                # Decoupled wide item key: a separate item table, NOT the LSTM
                # input.  Gathered at every position to feed the static readouts.
                item_keys = self.encoder.item_key_emb(item_ids)
                key_flat = item_keys.reshape(batch * T, self.item_key_dim)
            else:
                key_flat = None
        else:
            theta_in = self.encoder.theta_for_prediction(item_ids, responses)
            theta_flat = theta_in.reshape(batch * T)
            state_flat = None
            # A wide item key with NO state head still feeds the static beta head
            # (the all-static decoupled config), so thread it when it exists.
            if self.item_key_dim is not None:
                item_keys = self.encoder.item_key_emb(item_ids)
                key_flat = item_keys.reshape(batch * T, self.item_key_dim)
            else:
                key_flat = None

        # Select valid positions (no-op when mask is None).
        if mask is not None:
            mask_flat = mask.reshape(batch * T).bool()
            theta_flat = theta_flat[mask_flat]
            embs_flat = embs_flat[mask_flat]
            resp_flat = resp_flat[mask_flat]
            if state_flat is not None:
                state_flat = state_flat[mask_flat]
            if key_flat is not None:
                key_flat = key_flat[mask_flat]

        return self._predict_loss(theta_flat, embs_flat, resp_flat,
                                  state_flat, key_flat)

    def _predict_loss(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        responses: torch.Tensor,
        state: Optional[Tensor],
        item_key: Optional[Tensor],
    ) -> torch.Tensor:
        """Format-keyed prediction loss on the decoder's logits.

        binary -> BCE on the single logit;  nrm -> plain CE;  gpcm -> WOL.
        The decoder produces the logits (the IRT flavor); the loss scores them.
        """
        if self.decoder_name == "binary":
            z = self.decoder.binary_logit(
                theta, emb, state=state, item_key=item_key
            )
            return F.binary_cross_entropy_with_logits(z, responses.float())
        if self.decoder_name == "nrm":
            logits = self.decoder.logits_from_emb(theta, emb)
            return self.loss_fn(logits, responses)
        # gpcm (ordinal): WeightedOrdinalLoss on the GPCM logits.
        logits = self.decoder.logits_from_emb(
            theta, emb, state=state, item_key=item_key
        )
        return self.loss_fn(logits, responses)

    @staticmethod
    def _make_encoder(
        kind: str,
        num_items: int,
        emb_dim: int,
        hidden_dim: int,
        n_cats: int,
        item_key_dim: Optional[int],
        kw: dict,
    ) -> nn.Module:
        """Build the requested sequence backbone.

        Every backbone subclasses ``BaseSeqEncoder`` and exposes the identical
        public interface (aligned_theta_and_state / state_for_prediction /
        theta_for_prediction / encode + the item_val / resp / item_key_emb
        tables), so the decoder is unchanged across backbones -- this is the
        swappability contract.  ``kind="lstm"`` is the default and is bit-for-bit
        identical to the original encoder.  Backbone-specific knobs come through
        ``kw``.
        """
        common = dict(
            num_items=num_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
            n_cats=n_cats, item_key_dim=item_key_dim,
        )
        if kind == "lstm":
            return LSTMEncoder(**common)
        if kind == "transformer":
            from deep_irt.core.transformer_encoder import TransformerEncoder
            return TransformerEncoder(
                **common,
                n_heads=kw.get("n_heads", 4),
                n_layers=kw.get("n_layers", 2),
                max_seq_len=kw.get("max_seq_len", 512),
                dropout=kw.get("dropout", 0.0),
            )
        if kind == "dkvmn":
            from deep_irt.core.dkvmn_encoder import DKVMNEncoder
            return DKVMNEncoder(
                **common,
                memory_size=kw.get("memory_size", 20),
                key_dim=kw.get("key_dim", emb_dim),
            )
        raise ValueError(f"Unknown encoder backbone: {kind}")

    @staticmethod
    def _make_decoder(
        name: str,
        emb_dim: int,
        n_cats: int,
        correct_option: int = 0,
        state_dim: Optional[int] = None,
        item_key_dim: Optional[int] = None,
        alpha_log_scale: Optional[float] = None,
        state_alpha: bool = True,
        state_beta: bool = False,
    ) -> nn.Module:
        if name == "gpcm":
            return GPCMDecoder(emb_dim=emb_dim, n_cats=n_cats, state_dim=state_dim,
                               item_key_dim=item_key_dim,
                               alpha_log_scale=alpha_log_scale,
                               state_alpha=state_alpha, state_beta=state_beta)
        if name == "binary":
            return Binary2PLDecoder(emb_dim=emb_dim, state_dim=state_dim,
                                    item_key_dim=item_key_dim,
                                    alpha_log_scale=alpha_log_scale,
                                    state_alpha=state_alpha, state_beta=state_beta)
        if name == "bt":
            return BradleyTerryDecoder(emb_dim=emb_dim)
        if name == "nrm":
            return NRMDecoder(
                emb_dim=emb_dim,
                n_options=n_cats,
                correct_option=correct_option,
            )
        raise ValueError(f"Unknown decoder: {name}")
