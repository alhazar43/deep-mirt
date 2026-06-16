"""
decoders.py -- Three decoders sharing a common item-embedding interface.

All decoders accept the item embedding vector (emb_dim,) produced by the encoder's
item_emb table and map it to item parameters.  The theta signal (from the encoder)
is then combined with those parameters to compute a response distribution.

Decoder contract
----------------
Each decoder exposes:

    item_params(emb) -> named dict of item parameter tensors
        emb: (..., emb_dim)

    log_probs(theta, **item_params) -> (batch, n_outcomes)
        Compute log-probabilities over the response space.

    nll(theta, item_emb, responses) -> scalar
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
                     so a 2PL-comparable difficulty scale can be recovered.
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

      static map (default)  -- ``a = softplus(fc_a(item_emb))``.  Identical for
                               every occurrence of an item; the deep_irt's
                               original behaviour.
      state-conditioned     -- when ``state_dim`` is set, an extra head
                               ``a = softplus(fc_a_state([state, item_emb]))``
                               reads discrimination from an encoder state PLUS
                               the item embedding, in the spirit of ma-irt's
                               IRTParameterExtractor.  Passing ``state`` to
                               ``item_params`` / ``log_probs`` / ``nll`` selects
                               this head; omitting it falls back to the static
                               map, so models that never pass a state are
                               bit-for-bit identical to the static decoder.

    Step thresholds (b) are always an item-only read (``fc_b(item_emb)``).

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

    Decoupled alpha key (the DECOUPLED variant)
    -------------------------------------------
    When ``alpha_emb_dim`` is set the state-conditioned alpha head reads
    ``[state, alpha_emb]`` where ``alpha_emb`` is a SEPARATE, wider item key
    (from the encoder's ``alpha_item_emb`` table) of width ``alpha_emb_dim``,
    rather than the standard ``item_emb``.  This decouples alpha's item capacity
    from theta's: the head is ``fc_a_state : state_dim + alpha_emb_dim -> 1``.
    Step thresholds and the static fallback alpha still read the standard
    ``item_emb``.  ``alpha_emb_dim`` requires ``state_dim``.

    Parameters
    ----------
    emb_dim       : int
    n_cats        : int  -- K, number of ordered categories (responses in
                    {0,..,K-1})
    state_dim     : int or None -- when set, build the state-conditioned alpha
                    head whose input is ``[state, item_emb]`` of width
                    ``state_dim + emb_dim``.  None (default) builds only the
                    static ``fc_a`` map and leaves the parameter count unchanged.
    alpha_emb_dim : int or None -- when set (requires ``state_dim``), the
                    state-conditioned alpha head instead reads
                    ``[state, alpha_emb]`` of width ``state_dim + alpha_emb_dim``,
                    where ``alpha_emb`` is the encoder's separate wide alpha key.
                    None (default) keeps the head on the standard ``item_emb``.
    alpha_log_scale : float or None -- discrimination positivity transform.  None
                    (default) uses ``softplus`` and is bit-for-bit identical to
                    before.  A positive float ``s`` uses ``exp(s * raw)``,
                    ma-irt's exponential map (``s = 1.0`` matches ma-irt; the
                    head absorbs the scale constant).
    """

    def __init__(
        self,
        emb_dim: int,
        n_cats: int,
        state_dim: int | None = None,
        alpha_emb_dim: int | None = None,
        alpha_log_scale: float | None = None,
    ) -> None:
        super().__init__()
        if n_cats < 2:
            raise ValueError("n_cats must be >= 2")
        if alpha_emb_dim is not None and state_dim is None:
            raise ValueError(
                "alpha_emb_dim (decoupled wide alpha key) requires state_dim; "
                "the wide key only feeds the state-conditioned alpha head."
            )
        if alpha_log_scale is not None and alpha_log_scale <= 0.0:
            raise ValueError(
                f"alpha_log_scale must be > 0 (the exp scale), got "
                f"{alpha_log_scale}.  Use None for softplus."
            )
        self.emb_dim = emb_dim
        self.n_cats = n_cats
        self.state_dim = state_dim
        self.alpha_emb_dim = alpha_emb_dim
        self.alpha_log_scale = alpha_log_scale

        self.fc_a = nn.Linear(emb_dim, 1)          # -> raw discrimination
        self.fc_b = nn.Linear(emb_dim, n_cats - 1) # -> step thresholds (raw)

        # State-conditioned discrimination head, built only when requested so the
        # default (state-free) decoder is bit-for-bit identical to before.  In
        # the decoupled variant its item input is the wide alpha key.
        if state_dim is not None:
            alpha_in = alpha_emb_dim if alpha_emb_dim is not None else emb_dim
            self.fc_a_state = nn.Linear(state_dim + alpha_in, 1)

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
        alpha_emb: torch.Tensor | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        emb       : (..., emb_dim)
        state     : (..., state_dim) or None -- when given (and the decoder was
                    built with ``state_dim``), discrimination is read from the
                    state-conditioned head; otherwise from the static item-only
                    map.
        alpha_emb : (..., alpha_emb_dim) or None -- the wide alpha key for the
                    decoupled head.  Required when the decoder was built with
                    ``alpha_emb_dim`` and ``state`` is given; ignored otherwise.

        Returns
        -------
        dict with:
            a : (..., 1)      discrimination (softplus, always > 0)
            b : (..., K-1)    step thresholds (raw; sort before evaluation)
        """
        if state is not None:
            if self.state_dim is None:
                raise RuntimeError(
                    "GPCMDecoder was built without state_dim; cannot read a "
                    "state-conditioned discrimination.  Pass state_dim to the "
                    "constructor (dual-channel mode)."
                )
            if self.alpha_emb_dim is not None:
                if alpha_emb is None:
                    raise RuntimeError(
                        "GPCMDecoder was built with alpha_emb_dim (decoupled "
                        "wide alpha key); pass alpha_emb to item_params."
                    )
                key = alpha_emb
            else:
                key = emb
            a = self._alpha_pos(self.fc_a_state(torch.cat([state, key], dim=-1)))  # > 0
        else:
            a = self._alpha_pos(self.fc_a(emb))   # > 0
        b = self.fc_b(emb)               # unconstrained (sorted only at eval time)
        return {"a": a, "b": b}

    def item_params_sorted(
        self,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        alpha_emb: torch.Tensor | None = None,
    ) -> dict:
        """Same as item_params but with b sorted (use at evaluation time)."""
        params = self.item_params(emb, state=state, alpha_emb=alpha_emb)
        params["b"] = torch.sort(params["b"], dim=-1).values
        return params

    # --- logits (pre-softmax category scores; the prediction-loss target) ---

    def logits(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """GPCM per-category logits psi (the readout the prediction loss scores).

        psi_0 = 0;  psi_k = sum_{c=1..k} a * (theta - b_{c-1}),  k=1..K-1.
        ``log_softmax(psi)`` recovers the GPCM log-prob; the prediction loss
        (WeightedOrdinalLoss) scores ``psi`` directly, exactly as ma-irt feeds
        its GPCM logits to ``CombinedLoss``.

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        a     : (batch, 1)
        b     : (batch, K-1)

        Returns
        -------
        logits : (batch, K)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)            # (batch, 1)
        diff = a * (theta - b)                    # (batch, K-1)
        psi_pos = torch.cumsum(diff, dim=1)       # (batch, K-1)
        zeros = torch.zeros(theta.size(0), 1, device=theta.device)
        return torch.cat([zeros, psi_pos], dim=1)  # (batch, K)

    def category_logits(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        alpha_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-item GPCM logits from (theta, item emb, optional state/alpha key).

        Mirrors ``nll``'s signature but returns the ``(N, K)`` logits for the
        prediction loss instead of computing a likelihood.
        """
        params = self.item_params(emb, state=state, alpha_emb=alpha_emb)
        return self.logits(theta, params["a"], params["b"])

    # --- log_probs ---

    def log_probs(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GPCM log-probabilities.

        psi_0 = 0
        psi_k = sum_{c=1..k}  a * (theta - b_{c-1})   for k=1..K-1

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        a     : (batch, 1)
        b     : (batch, K-1)

        Returns
        -------
        log_probs : (batch, K)
        """
        return F.log_softmax(self.logits(theta, a, b), dim=1)

    # --- nll convenience ---

    def nll(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        responses: torch.Tensor,
        state: torch.Tensor | None = None,
        alpha_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute mean NLL for a flat batch.

        Parameters
        ----------
        theta     : (N,)
        emb       : (N, emb_dim)
        responses : (N,)  long, in {0,..,K-1}
        state     : (N, state_dim) or None -- when given, discrimination is read
                    from the state-conditioned head (dual-channel mode).  None
                    keeps the static item-only alpha and is bit-for-bit
                    identical to the pre-dual-channel decoder.
        alpha_emb : (N, alpha_emb_dim) or None -- the wide alpha key for the
                    decoupled head (required when built with ``alpha_emb_dim``).

        Returns
        -------
        loss : scalar
        """
        params = self.item_params(emb, state=state, alpha_emb=alpha_emb)
        log_p = self.log_probs(theta, params["a"], params["b"])
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
        alpha_emb_dim: int | None = None,
        alpha_log_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cats = 2
        self.state_dim = state_dim
        self.alpha_emb_dim = alpha_emb_dim
        self.alpha_log_scale = alpha_log_scale
        self._gpcm = GPCMDecoder(
            emb_dim=emb_dim, n_cats=2, state_dim=state_dim,
            alpha_emb_dim=alpha_emb_dim, alpha_log_scale=alpha_log_scale,
        )

    def item_params(
        self,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        alpha_emb: torch.Tensor | None = None,
    ) -> dict:
        """Returns a and b (single threshold) from embedding."""
        return self._gpcm.item_params(emb, state=state, alpha_emb=alpha_emb)

    def item_params_sorted(
        self,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        alpha_emb: torch.Tensor | None = None,
    ) -> dict:
        return self._gpcm.item_params_sorted(emb, state=state, alpha_emb=alpha_emb)

    def log_probs(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns log-probs for {0, 1}.

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        a     : (batch, 1)
        b     : (batch, 1)   single threshold

        Returns
        -------
        log_probs : (batch, 2)
        """
        return self._gpcm.log_probs(theta, a, b)

    def binary_logit(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        state: torch.Tensor | None = None,
        alpha_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single binary logit z = a * (theta - b); P(y=1) = sigmoid(z).

        ``binary_cross_entropy_with_logits(z, y.float())`` is the binary
        prediction loss.  Equals the log-odds of the K=2 GPCM logits.
        """
        lg = self._gpcm.category_logits(theta, emb, state=state, alpha_emb=alpha_emb)
        return lg[..., 1] - lg[..., 0]            # (N,)

    def nll(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        responses: torch.Tensor,
        state: torch.Tensor | None = None,
        alpha_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean NLL for a flat batch of binary responses."""
        return self._gpcm.nll(theta, emb, responses, state=state,
                              alpha_emb=alpha_emb)


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

    def item_strength(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Compute item strength from embedding.

        Parameters
        ----------
        emb : (..., emb_dim)

        Returns
        -------
        strength : (..., 1)
        """
        return self.fc_s(emb)

    def item_params(self, emb: torch.Tensor) -> dict:
        """Unified interface: returns {'strength': (..., 1)}."""
        return {"strength": self.item_strength(emb)}

    def log_prob_i_beats_j(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log P(item_i beats item_j) = log sigmoid(s_i - s_j).

        Parameters
        ----------
        emb_i : (N, emb_dim)
        emb_j : (N, emb_dim)

        Returns
        -------
        log_prob : (N,)
        """
        s_i = self.fc_s(emb_i).squeeze(-1)   # (N,)
        s_j = self.fc_s(emb_j).squeeze(-1)   # (N,)
        return F.logsigmoid(s_i - s_j)

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

    Recoverable difficulty scale
    ----------------------------
    For cross-format comparison with 2PL/GPCM we expose a scalar item location
    derived from the designated-correct option.  Define the correct option's
    linear predictor f_corr(theta) = a_corr * theta + c_corr.  Its zero-crossing

        beta_i = - c_corr / a_corr

    is the NRM analogue of 2PL difficulty: the ability at which the correct
    option's utility crosses zero (higher beta_i -> harder item).  This is the
    standard NRM location for the keyed option and is directly comparable to the
    2PL b parameter and the GPCM mean threshold.  A small floor on |a_corr|
    keeps the division stable.

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

        # emb -> raw per-option slopes and intercepts (centered in item_params)
        self.fc_a = nn.Linear(emb_dim, n_options)   # -> raw slopes a_k
        self.fc_c = nn.Linear(emb_dim, n_options)   # -> raw intercepts c_k

    # --- item_params ---

    def item_params(self, emb: torch.Tensor) -> dict:
        """
        Map embedding to centered per-option NRM parameters.

        Parameters
        ----------
        emb : (..., emb_dim)

        Returns
        -------
        dict with:
            a : (..., K)   per-option slopes, sum_k a_k = 0
            c : (..., K)   per-option intercepts, sum_k c_k = 0
        """
        a_raw = self.fc_a(emb)                       # (..., K)
        c_raw = self.fc_c(emb)                       # (..., K)
        a = a_raw - a_raw.mean(dim=-1, keepdim=True)  # sum_k a_k = 0
        c = c_raw - c_raw.mean(dim=-1, keepdim=True)  # sum_k c_k = 0
        return {"a": a, "c": c}

    # --- logits (pre-softmax option scores; the prediction-loss target) ---

    def logits(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """NRM per-option logits ``a_k * theta + c_k`` (unordered).

        The nominal prediction loss is plain cross-entropy on these logits,
        with NO ordinal-distance penalty: the options carry no order.

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        a     : (batch, K)  per-option slopes (already centered)
        c     : (batch, K)  per-option intercepts (already centered)

        Returns
        -------
        logits : (batch, K)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)               # (batch, 1)
        return a * theta + c                         # (batch, K)

    def category_logits(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """Per-item NRM logits from (theta, item emb).

        Mirrors ``nll``'s signature but returns the ``(N, K)`` logits for the
        prediction loss instead of computing a likelihood.
        """
        params = self.item_params(emb)
        return self.logits(theta, params["a"], params["c"])

    # --- log_probs ---

    def log_probs(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NRM log-probabilities over the K options.

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        a     : (batch, K)  per-option slopes (already centered)
        c     : (batch, K)  per-option intercepts (already centered)

        Returns
        -------
        log_probs : (batch, K)
        """
        return F.log_softmax(self.logits(theta, a, c), dim=1)

    # --- difficulty scale (correct-option location) ---

    def item_difficulty(
        self,
        emb: torch.Tensor,
        a_floor: float = 0.1,
    ) -> torch.Tensor:
        """
        2PL-comparable item location from the correct option.

        beta_i = - c_corr / a_corr, the ability at which the correct option's
        linear predictor crosses zero.  Higher beta -> harder.

        Parameters
        ----------
        emb     : (..., emb_dim)
        a_floor : minimum |a_corr| used in the denominator for stability

        Returns
        -------
        beta : (...,)  scalar difficulty per item
        """
        params = self.item_params(emb)
        a_corr = params["a"][..., self.correct_option]   # (...,)
        c_corr = params["c"][..., self.correct_option]   # (...,)
        # Stabilise the denominator without changing its sign.
        denom = torch.sign(a_corr) * a_corr.abs().clamp_min(a_floor)
        # sign(0) == 0 would zero the denom; guard that degenerate case.
        denom = torch.where(
            denom == 0, torch.full_like(denom, a_floor), denom
        )
        return -c_corr / denom

    # --- nll convenience ---

    def nll(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean NLL for a flat batch of option choices.

        Parameters
        ----------
        theta     : (N,)
        emb       : (N, emb_dim)
        responses : (N,)  long, option indices in {0,..,K-1}

        Returns
        -------
        loss : scalar
        """
        params = self.item_params(emb)
        log_p = self.log_probs(theta, params["a"], params["c"])
        return F.nll_loss(log_p, responses)
