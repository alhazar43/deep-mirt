"""
decoders.py -- Three decoders sharing a common item-embedding interface.

All decoders accept the thin item VALUE embedding vector (emb_dim,) produced by
the encoder's ``item_val_emb`` table and map it to item parameters.  In the
DECOUPLED variant the static alpha/beta readouts instead ride the wide item KEY
(``item_key_emb`` of width ``item_key_dim``), passed in as ``item_key``; width
follows inverse Fisher (capacity-hungry alpha wants a wide key, well-determined
theta wants a thin value).  The theta signal (from the encoder) is then combined
with the item parameters to compute a response distribution.

Decoder contract
----------------
Each decoder exposes:

    item_params(emb) -> named dict of item parameter tensors
        emb: (..., emb_dim)

    log_probs(theta, **item_params) -> (batch, n_outcomes)
        Compute log-probabilities over the response space.

    nll(theta, emb, responses) -> scalar
        Convenience: map emb -> params -> log_probs -> NLL.

The four decoders:

    GPCMDecoder   -- ordinal K-category GPCM (the primary decoder used in training).
                     K=2 special case recovers 2PL binary IRT.

    Binary2PLDecoder -- thin wrapper: GPCM at K=2.  Accepts binary responses {0,1}.

    BradleyTerryDecoder -- pairwise decoder.  Each item has a learned strength
                           scalar.  Does not use theta from the encoder (BT is
                           a purely inter-item model).  Exposes fit_pairs() for
                           training on pairwise comparison data.

    NRMDecoder    -- Bock (1972) nominal response model for UNORDERED categorical
                     responses (e.g. multiple-choice option a/b/c/d).  Each option
                     k has its own slope a_k and intercept c_k; the response
                     distribution is softmax_k(a_k * theta + c_k).  Consumes option
                     indices {0,..,K-1} like GPCM consumes ordinal levels, but the
                     categories carry no order.  One option is designated "correct"
                     for data generation and accuracy scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GPCM Decoder (ordinal, K categories)
# ---------------------------------------------------------------------------

class GPCMDecoder(nn.Module):
    """
    Item embedding -> (a, b) GPCM parameters -> ordered categorical log-probs.

    Discrimination (a) has two readout modes:

      static map (default)  -- ``a = softplus(fc_a(item_val_emb))``.  Identical
                               for every occurrence of an item; the deep_irt's
                               original behaviour.
      state-conditioned     -- when ``state_dim`` is set, an extra head
                               ``a = softplus(fc_a_state([state, item_key]))``
                               reads discrimination from an encoder state PLUS
                               the item key, in the spirit of ma-irt's
                               IRTParameterExtractor.  Passing ``state`` to
                               ``item_params`` / ``log_probs`` / ``nll`` selects
                               this head; omitting it falls back to the static
                               map, so models that never pass a state are
                               bit-for-bit identical to the static decoder.

    The item KEY vs VALUE split (width follows inverse Fisher)
    ----------------------------------------------------------
    The decoder's STATIC readouts (alpha and beta) ride the wide item KEY when it
    exists; the thin item VALUE feeds the encoder/theta.  ``item_key_dim`` set
    (the DECOUPLED variant) means the encoder supplies a separate wide
    ``item_key`` of that width: the state-conditioned alpha head becomes
    ``fc_a_state : state_dim + item_key_dim -> 1`` and the step-threshold head
    becomes ``fc_b : item_key_dim -> n_cats-1``, so BOTH static item parameters
    read the wide key while theta stays on the cheap encoder.  ``item_key_dim``
    None (legacy non-decoupled path) keeps alpha and beta on the thin
    ``emb_dim`` value and is bit-for-bit identical to the original decoder.
    ``item_key_dim`` requires ``state_dim``.

    Discrimination positivity transform (``alpha_log_scale``)
    ---------------------------------------------------------
    The raw discrimination head output is pushed through a positive map.  GPCM
    needs ``a > 0``, so the raw scalar gets one of two positivity transforms
    (nothing is being "linked" in the scale-equating sense):

      softplus (default, ``alpha_log_scale=None``) -- ``a = softplus(raw)``, the
          deep_irt's original transform.  Kept as the default so every model
          that does not pass ``alpha_log_scale`` is bit-for-bit identical.
      exponential (``alpha_log_scale=s`` for s>0) -- ``a = exp(s * raw)``,
          ma-irt's transform (reimplemented locally, not imported, to respect
          the frozen-Chapter-0 boundary).  ``s`` is a gradient-damping scale the
          head absorbs, not a parameterisation choice, so ``s = 1.0`` (ma-irt's
          setting, plain ``exp(raw)``) is the apples-to-apples value.

    Parameters
    ----------
    emb_dim       : int  -- thin item value width
    n_cats        : int  -- K, number of ordered categories (responses in
                    {0,..,K-1})
    state_dim     : int or None -- when set, build the state-conditioned alpha
                    head.  Its item input is the wide ``item_key`` when
                    ``item_key_dim`` is set, else the thin ``emb`` value.  None
                    (default) builds only the static ``fc_a`` map and leaves the
                    parameter count unchanged.
    item_key_dim  : int or None -- when set (requires ``state_dim``), both static
                    readouts ride the encoder's wide item key: the alpha head
                    reads ``[state, item_key]`` of width
                    ``state_dim + item_key_dim`` and ``fc_b`` reads
                    ``item_key`` of width ``item_key_dim``.  None (default) keeps
                    alpha and beta on the thin ``emb_dim`` value.
    alpha_log_scale : float or None -- discrimination positivity transform.  None
                    (default) uses ``softplus`` and is bit-for-bit identical to
                    before.  A positive float ``s`` uses ``exp(s * raw)``,
                    ma-irt's exponential map (``s = 1.0`` matches ma-irt; the
                    head absorbs the scale constant).
    state_beta    : bool -- when True (requires both ``state_dim`` and
                    ``item_key_dim``) the step thresholds are read from a
                    state-conditioned head ``fc_b_state([state, item_key])``
                    instead of the static ``fc_b(item_key)``, the EXACT mirror of
                    the state-conditioned alpha head.  This is the dynamic-beta
                    arm: difficulty becomes per-occurrence, conditioned on the
                    same prediction-aligned state alpha uses.  Default False keeps
                    the static (item-only) beta read and is bit-for-bit identical
                    (``fc_b_state`` is built last, so the state_alpha init RNG is
                    untouched).
    """

    def __init__(
        self,
        emb_dim: int,
        n_cats: int,
        state_dim: int | None = None,
        item_key_dim: int | None = None,
        alpha_log_scale: float | None = None,
        state_alpha: bool = True,
        state_beta: bool = False,
    ) -> None:
        super().__init__()
        if n_cats < 2:
            raise ValueError("n_cats must be >= 2")
        # ``item_key_dim`` (the wide item key) feeds the static step-threshold
        # head ``fc_b`` and, when a state is present, the state-conditioned alpha
        # head.  It is therefore meaningful WITHOUT a state too (a static
        # wide-key beta -- the capacity-matched all-static decoupled baseline), so
        # no state_dim is required here.
        if alpha_log_scale is not None and alpha_log_scale <= 0.0:
            raise ValueError(
                f"alpha_log_scale must be > 0 (the exp scale), got "
                f"{alpha_log_scale}.  Use None for softplus."
            )
        if state_beta and state_dim is None:
            raise ValueError(
                "state_beta (the state-conditioned step-threshold head) requires "
                "state_dim; the head reads [state, item_key]."
            )
        if state_beta and item_key_dim is None:
            raise ValueError(
                "state_beta requires item_key_dim (the decoupled wide item key); "
                "the head reads [state, item_key], mirroring the state_alpha head."
            )
        # ``state_alpha`` gates whether the discrimination head is state-
        # conditioned.  It is meaningful only when a state is available
        # (``state_dim`` set); with no state the decoder is the legacy static map
        # and ``state_alpha`` is moot.  Decoupling alpha from beta (so beta can be
        # dynamic while alpha stays static) needs this explicit gate.
        self.state_alpha = bool(state_alpha) and state_dim is not None
        self.emb_dim = emb_dim
        self.n_cats = n_cats
        self.state_dim = state_dim
        self.item_key_dim = item_key_dim
        self.alpha_log_scale = alpha_log_scale
        self.state_beta = state_beta

        self.fc_a = nn.Linear(emb_dim, 1)          # -> raw discrimination
        # Step thresholds: the wide item key whenever it exists (decoupled), else
        # the thin item value (legacy non-decoupled path).
        beta_in = item_key_dim if item_key_dim is not None else emb_dim
        self.fc_b = nn.Linear(beta_in, n_cats - 1) # -> step thresholds (raw)

        # State-conditioned discrimination head, built only when the alpha head is
        # state-conditioned (``state_alpha`` and a state available), so the
        # default (state-free) decoder is bit-for-bit identical to before.  In
        # the decoupled variant its item input is the wide item key.
        if self.state_alpha:
            alpha_in = item_key_dim if item_key_dim is not None else emb_dim
            self.fc_a_state = nn.Linear(state_dim + alpha_in, 1)

        # State-conditioned step-threshold head, the EXACT mirror of fc_a_state
        # for beta.  Built only when requested (requires state_dim AND
        # item_key_dim) and ONLY after fc_a_state above so the state_alpha
        # default-path init RNG is untouched (default state_beta=False is
        # bit-for-bit identical).  Reads [state, item_key] of width
        # state_dim + item_key_dim -> n_cats-1 raw step thresholds.
        if state_beta:
            self.fc_b_state = nn.Linear(state_dim + item_key_dim, n_cats - 1)

    # --- positivity transform ---

    def _alpha_pos(self, raw: torch.Tensor) -> torch.Tensor:
        """Push the raw discrimination output through a positive map.

        ``alpha_log_scale is None`` -> ``softplus(raw)`` (the original transform,
        bit-for-bit identical).  ``alpha_log_scale = s`` -> ``exp(s * raw)``,
        ma-irt's exponential map.  ``s`` is a gradient-damping scale the head
        absorbs, so ``s = 1.0`` (plain ``exp(raw)``) is the apples-to-apples
        value.  Reimplemented here (not imported from ma-irt) to respect the
        frozen-Chapter-0 boundary.
        """
        if self.alpha_log_scale is None:
            return F.softplus(raw)
        return torch.exp(self.alpha_log_scale * raw)

    # --- item_params ---

    def item_params(
        self,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        emb      : (..., emb_dim) -- thin item value
        state    : (..., state_dim) or None -- the prediction-aligned state.  When
                   the alpha head is state-conditioned (``state_alpha``)
                   discrimination reads it; when the beta head is state-
                   conditioned (``state_beta``) the step thresholds read it.  A
                   state may be supplied for beta alone, in which case alpha still
                   uses the static map (alpha and beta are gated independently).
        item_key : (..., item_key_dim) or None -- the wide item key for the
                   decoupled static readouts.  Required when the decoder was built
                   with ``item_key_dim``; ignored otherwise.

        Returns
        -------
        dict with:
            a : (..., 1)      discrimination (softplus, always > 0)
            b : (..., K-1)    step thresholds (raw; sort before evaluation)
        """
        # Discrimination.  State-conditioned ONLY when the alpha head was built
        # state-conditioned (``state_alpha``) AND a state is supplied; a state
        # passed solely for the beta head does NOT pull alpha off its static map,
        # and omitting the state falls back to the static map (the legacy
        # contract -- item-only recovery reads still work).
        if self.state_alpha and state is not None:
            if self.item_key_dim is not None:
                if item_key is None:
                    raise RuntimeError(
                        "GPCMDecoder was built with item_key_dim (decoupled "
                        "wide item key); pass item_key to item_params."
                    )
                key = item_key
            else:
                key = emb
            a = self._alpha_pos(self.fc_a_state(torch.cat([state, key], dim=-1)))  # > 0
        else:
            a = self._alpha_pos(self.fc_a(emb))   # > 0
        # Step thresholds.  Three modes, in priority order:
        #   state_beta on  -> b = fc_b_state([state, item_key]), the per-occurrence
        #                     state-conditioned threshold (the dynamic-beta arm,
        #                     the exact mirror of the state-conditioned alpha).
        #   item_key_dim   -> b = fc_b(item_key), wide static item-key threshold
        #                     (ma-irt's shared-key path).
        #   otherwise      -> b = fc_b(emb), thin item-value threshold (legacy).
        if self.state_beta:
            if state is None:
                raise RuntimeError(
                    "GPCMDecoder was built with state_beta (state-conditioned "
                    "step thresholds); pass state to item_params so beta reads it."
                )
            if item_key is None:
                raise RuntimeError(
                    "GPCMDecoder was built with state_beta (which requires the "
                    "wide item key); pass item_key to item_params so beta reads it."
                )
            b = self.fc_b_state(torch.cat([state, item_key], dim=-1))
        elif self.item_key_dim is not None:
            if item_key is None:
                raise RuntimeError(
                    "GPCMDecoder was built with item_key_dim (decoupled wide "
                    "item key); pass item_key to item_params so beta reads it."
                )
            b = self.fc_b(item_key)      # wide-key beta (ma-irt's shared-key path)
        else:
            b = self.fc_b(emb)           # unconstrained (sorted only at eval time)
        return {"alpha": a, "beta": b}

    def item_params_sorted(
        self,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> dict:
        """Same as item_params but with beta sorted (use at evaluation time)."""
        params = self.item_params(emb, state=state, item_key=item_key)
        params["beta"] = torch.sort(params["beta"], dim=-1).values
        return params

    # --- logits (pre-softmax category scores; the prediction-loss target) ---

    def logits(
        self,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """GPCM per-category logits psi (the readout the prediction loss scores).

        psi_0 = 0;  psi_k = sum_{c=1..k} alpha * (theta - beta_{c-1}),  k=1..K-1.
        ``log_softmax(psi)`` recovers the GPCM log-prob; the prediction loss
        (WeightedOrdinalLoss) scores ``psi`` directly, exactly as ma-irt feeds
        its GPCM logits to ``CombinedLoss``.

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        alpha : (batch, 1)
        beta  : (batch, K-1)

        Returns
        -------
        logits : (batch, K)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)            # (batch, 1)
        diff = alpha * (theta - beta)             # (batch, K-1)
        psi_pos = torch.cumsum(diff, dim=1)       # (batch, K-1)
        zeros = torch.zeros(theta.size(0), 1, device=theta.device)
        return torch.cat([zeros, psi_pos], dim=1)  # (batch, K)

    def logits_from_emb(
        self,
        theta: torch.Tensor,
        item_val: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-item GPCM logits from (theta, item value, optional state/item key).

        Mirrors ``nll``'s signature but returns the ``(N, K)`` logits for the
        prediction loss instead of computing a likelihood.
        """
        params = self.item_params(item_val, state=state, item_key=item_key)
        return self.logits(theta, params["alpha"], params["beta"])

    # --- log_probs ---

    def log_probs(
        self,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GPCM log-probabilities.

        psi_0 = 0
        psi_k = sum_{c=1..k}  alpha * (theta - beta_{c-1})   for k=1..K-1

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        alpha : (batch, 1)
        beta  : (batch, K-1)

        Returns
        -------
        log_probs : (batch, K)
        """
        return F.log_softmax(self.logits(theta, alpha, beta), dim=1)

    # --- nll convenience ---

    def nll(
        self,
        theta: torch.Tensor,
        item_val: torch.Tensor,
        responses: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute mean NLL for a flat batch.

        Parameters
        ----------
        theta     : (N,)
        item_val  : (N, emb_dim)  -- thin item value
        responses : (N,)  long, in {0,..,K-1}
        state     : (N, state_dim) or None -- when given, discrimination is read
                    from the state-conditioned head (dual-channel mode).  None
                    keeps the static item-only alpha and is bit-for-bit
                    identical to the pre-dual-channel decoder.
        item_key  : (N, item_key_dim) or None -- the wide item key for the
                    decoupled static readouts (required when built with
                    ``item_key_dim``).

        Returns
        -------
        loss : scalar
        """
        params = self.item_params(item_val, state=state, item_key=item_key)
        log_p = self.log_probs(theta, params["alpha"], params["beta"])
        return F.nll_loss(log_p, responses)


# ---------------------------------------------------------------------------
# Binary 2PL Decoder (thin GPCM K=2 wrapper)
# ---------------------------------------------------------------------------

class Binary2PLDecoder(nn.Module):
    """
    Binary 2PL IRT decoder: GPCM restricted to K=2.

    Binary responses {0, 1}.  The single step threshold b plays the role of item
    difficulty; discrimination a is the same as in GPCM.

    This is a thin wrapper around GPCMDecoder(n_cats=2).  It exposes the same
    item_params / log_probs / nll interface, including the optional
    state-conditioned discrimination head (dual-channel mode) when built with
    ``state_dim``.
    """

    def __init__(
        self,
        emb_dim: int,
        state_dim: int | None = None,
        item_key_dim: int | None = None,
        alpha_log_scale: float | None = None,
        state_alpha: bool = True,
        state_beta: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cats = 2
        self.state_dim = state_dim
        self.item_key_dim = item_key_dim
        self.alpha_log_scale = alpha_log_scale
        self.state_beta = state_beta
        self._gpcm = GPCMDecoder(
            emb_dim=emb_dim, n_cats=2, state_dim=state_dim,
            item_key_dim=item_key_dim, alpha_log_scale=alpha_log_scale,
            state_alpha=state_alpha, state_beta=state_beta,
        )

    def item_params(
        self,
        item_val: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> dict:
        """Returns alpha and beta (single threshold) from embedding."""
        return self._gpcm.item_params(item_val, state=state, item_key=item_key)

    def item_params_sorted(
        self,
        item_val: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> dict:
        return self._gpcm.item_params_sorted(item_val, state=state, item_key=item_key)

    def log_probs(
        self,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns log-probs for {0, 1}.

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        alpha : (batch, 1)
        beta  : (batch, 1)   single threshold

        Returns
        -------
        log_probs : (batch, 2)
        """
        return self._gpcm.log_probs(theta, alpha, beta)

    def binary_logit(
        self,
        theta: torch.Tensor,
        item_val: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single binary logit z = alpha * (theta - beta); P(y=1) = sigmoid(z).

        ``binary_cross_entropy_with_logits(z, y.float())`` is the binary
        prediction loss.  Equals the log-odds of the K=2 GPCM logits.
        """
        lg = self._gpcm.logits_from_emb(theta, item_val, state=state, item_key=item_key)
        return lg[..., 1] - lg[..., 0]            # (N,)

    def nll(
        self,
        theta: torch.Tensor,
        item_val: torch.Tensor,
        responses: torch.Tensor,
        state: torch.Tensor | None = None,
        item_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean NLL for a flat batch of binary responses."""
        return self._gpcm.nll(theta, item_val, responses, state=state,
                              item_key=item_key)


# ---------------------------------------------------------------------------
# Bradley-Terry Decoder (pairwise; item-only, no theta)
# ---------------------------------------------------------------------------

class BradleyTerryDecoder(nn.Module):
    """
    Bradley-Terry pairwise decoder.

    Each item has a learned scalar strength s_i derived from its embedding:
        s_i = linear(emb_i)   (single output, no activation)

    P(item_i beats item_j) = sigmoid(s_i - s_j)

    The BT model is purely inter-item: it does not use theta from the encoder.
    The strength scores can be compared against the encoder's item-difficulty
    signal to verify cross-format scale agreement.

    Parameters
    ----------
    emb_dim : int
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.fc_s = nn.Linear(emb_dim, 1)  # emb -> item strength scalar

    def item_strength(self, item_val: torch.Tensor) -> torch.Tensor:
        """
        Compute item strength from embedding.

        Parameters
        ----------
        item_val : (..., emb_dim)

        Returns
        -------
        strength : (..., 1)
        """
        return self.fc_s(item_val)

    def item_params(self, item_val: torch.Tensor) -> dict:
        """Unified interface: returns {'strength': (..., 1)}."""
        return {"strength": self.item_strength(item_val)}

    def nll_pairs(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        outcome: torch.Tensor,
        reg_strength: float = 0.01,
    ) -> torch.Tensor:
        """
        Binary cross-entropy NLL on pairwise comparison outcomes.

        outcome == 1: item i beats j; outcome == 0: item j beats i.

        Parameters
        ----------
        emb_i      : (N, emb_dim)
        emb_j      : (N, emb_dim)
        outcome    : (N,)  long {0, 1}
        reg_strength : L2 regulariser weight on strengths (soft identifiability anchor)

        Returns
        -------
        loss : scalar
        """
        s_i = self.fc_s(emb_i).squeeze(-1)   # (N,)
        s_j = self.fc_s(emb_j).squeeze(-1)   # (N,)
        delta = s_i - s_j
        bce = F.binary_cross_entropy_with_logits(
            delta, outcome.float(), reduction="mean"
        )
        # Soft L2 anchor so the mean does not drift
        reg = reg_strength * (s_i.pow(2).mean() + s_j.pow(2).mean()) * 0.5
        return bce + reg


# ---------------------------------------------------------------------------
# Nominal Response Model Decoder (Bock 1972; unordered K options)
# ---------------------------------------------------------------------------

class NRMDecoder(nn.Module):
    """
    Item embedding -> per-option (a_k, c_k) -> nominal-categorical log-probs.

    Bock's (1972) nominal response model.  For an item with K unordered options,

        P(option k | theta) = softmax_k( a_k * theta + c_k )

    with per-option slope a_k and intercept c_k.  Unlike GPCM the categories
    carry no order, so each option gets its own free slope rather than a single
    shared discrimination.

    Identifiability
    ---------------
    The softmax is invariant to adding a constant to every a_k or every c_k:
    softmax_k(a_k * theta + c_k) is unchanged under a_k -> a_k + u,
    c_k -> c_k + v.  The standard Bock constraints remove this redundancy:

        sum_k a_k = 0,   sum_k c_k = 0.

    We enforce them by mean-centering the raw network outputs across the option
    axis.  Centering is the exact sum-to-zero projection and, because the
    softmax is shift-invariant, it does not change the likelihood -- it only
    pins down a unique representative of each equivalence class so the recovered
    a_k / c_k are comparable across fits.

    Parameters
    ----------
    emb_dim       : int
    n_options     : int   -- K, number of options (responses in {0,..,K-1})
    correct_option: int   -- index of the designated-correct option (default 0)
    """

    def __init__(
        self,
        emb_dim: int,
        n_options: int,
        correct_option: int = 0,
        item_key_dim: int | None = None,
    ) -> None:
        super().__init__()
        if n_options < 2:
            raise ValueError("n_options must be >= 2")
        if not (0 <= correct_option < n_options):
            raise ValueError(
                f"correct_option must be in [0, {n_options}), got {correct_option}"
            )
        self.emb_dim = emb_dim
        self.n_options = n_options
        self.correct_option = correct_option
        self.item_key_dim = item_key_dim

        # Per-option slopes a_k and intercepts c_k.  In the DECOUPLED variant
        # (item_key_dim set) BOTH read the wide item KEY -- the NRM item
        # parameters are exactly a_k and c_k, so both go on the fat key while
        # theta stays on the thin item value, mirroring GPCM's alpha+beta split.
        # None (default) reads the thin item value and is bit-for-bit identical
        # to the original item-only NRM.
        readout_in = item_key_dim if item_key_dim is not None else emb_dim
        self.fc_a = nn.Linear(readout_in, n_options)   # -> raw slopes a_k
        self.fc_c = nn.Linear(readout_in, n_options)   # -> raw intercepts c_k

    # --- item_params ---

    def item_params(
        self,
        item_val: torch.Tensor,
        item_key: torch.Tensor | None = None,
    ) -> dict:
        """
        Map the item embedding to centered per-option NRM parameters.

        Parameters
        ----------
        item_val : (..., emb_dim)  thin item value (read when not decoupled)
        item_key : (..., item_key_dim) or None -- the wide item key.  Required
                   when built with ``item_key_dim`` (the decoupled variant), in
                   which case BOTH a_k and c_k read it; ignored otherwise.

        Returns
        -------
        dict with:
            alpha     : (..., K)   per-option slopes, sum_k alpha_k = 0
            intercept : (..., K)   per-option intercepts, sum_k intercept_k = 0
        """
        if self.item_key_dim is not None:
            if item_key is None:
                raise RuntimeError(
                    "NRMDecoder was built with item_key_dim (decoupled wide "
                    "item key); pass item_key to item_params."
                )
            readout = item_key
        else:
            readout = item_val
        a_raw = self.fc_a(readout)                             # (..., K)
        c_raw = self.fc_c(readout)                             # (..., K)
        alpha = a_raw - a_raw.mean(dim=-1, keepdim=True)        # sum_k alpha_k = 0
        intercept = c_raw - c_raw.mean(dim=-1, keepdim=True)   # sum_k intercept_k = 0
        return {"alpha": alpha, "intercept": intercept}

    # --- logits (pre-softmax option scores; the prediction-loss target) ---

    def logits(
        self,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        intercept: torch.Tensor,
    ) -> torch.Tensor:
        """NRM per-option logits ``alpha_k * theta + intercept_k`` (unordered).

        The nominal prediction loss is plain cross-entropy on these logits,
        with NO ordinal-distance penalty: the options carry no order.

        Parameters
        ----------
        theta     : (batch,) or (batch, 1)
        alpha     : (batch, K)  per-option slopes (already centered)
        intercept : (batch, K)  per-option intercepts (already centered)

        Returns
        -------
        logits : (batch, K)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)               # (batch, 1)
        return alpha * theta + intercept             # (batch, K)

    def logits_from_emb(
        self,
        theta: torch.Tensor,
        item_val: torch.Tensor,
        item_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-item NRM logits from (theta, item value, optional wide item key).

        Mirrors ``nll``'s signature but returns the ``(N, K)`` logits for the
        prediction loss instead of computing a likelihood.
        """
        params = self.item_params(item_val, item_key=item_key)
        return self.logits(theta, params["alpha"], params["intercept"])

    # --- log_probs ---

    def log_probs(
        self,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        intercept: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NRM log-probabilities over the K options.

        Parameters
        ----------
        theta     : (batch,) or (batch, 1)
        alpha     : (batch, K)  per-option slopes (already centered)
        intercept : (batch, K)  per-option intercepts (already centered)

        Returns
        -------
        log_probs : (batch, K)
        """
        return F.log_softmax(self.logits(theta, alpha, intercept), dim=1)

    # --- nll convenience ---

    def nll(
        self,
        theta: torch.Tensor,
        item_val: torch.Tensor,
        responses: torch.Tensor,
        item_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Mean NLL for a flat batch of option choices.

        Parameters
        ----------
        theta     : (N,)
        item_val  : (N, emb_dim)
        responses : (N,)  long, option indices in {0,..,K-1}
        item_key  : (N, item_key_dim) or None -- the wide item key (decoupled)

        Returns
        -------
        loss : scalar
        """
        params = self.item_params(item_val, item_key=item_key)
        log_p = self.log_probs(theta, params["alpha"], params["intercept"])
        return F.nll_loss(log_p, responses)
