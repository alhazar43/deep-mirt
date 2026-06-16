"""
decoders_ext.py -- Three additional response-format decoders for the wider-format
RQ1 exploration.  They extend the pack in ``deep_irt/core/decoders.py`` without
touching it: same item-embedding interface, same item_params / log_probs / nll
contract, so each plugs into DeepIRTModel / dynjoint / the ma-irt pack the same
way the GPCM / Binary2PL / NRM decoders do.

A note on the latent argument
-----------------------------
The categorical decoders in ``decoders.py`` name their person latent ``theta``
(an ability).  Two of the decoders here also key off an ability theta (Beta,
Poisson).  The response-time decoder keys off a person SPEED latent instead; we
still pass it through the ``theta`` argument slot for a uniform call signature,
but name it ``tau`` in the docstrings to keep the speed interpretation explicit.

The three decoders
------------------

    LogNormalRTDecoder  -- van der Linden (2007) hierarchical log-normal speed
                           model for response times.  log(RT) ~ Normal(tau - beta_i,
                           sigma).  tau is person speed, beta_i the item's
                           time-intensity (a difficulty-like item parameter).
                           This is the priority decoder: it feeds the 3-format
                           RQ1 (accuracy + choice + latency on EdNet).

    BetaResponseDecoder -- Beta continuous-response decoder for bounded scores in
                           (0,1).  mean mu_i = sigmoid(a_i*(theta - b_i)),
                           precision phi.  a_i, b_i from the embedding.

    PoissonCountDecoder -- Poisson decoder for non-negative counts.
                           rate lambda = exp(a_i*(theta - b_i)) (or exp(theta - b_i)
                           when discrimination is fixed).  Item param(s) from the
                           embedding.

All three expose:

    item_params(emb) -> named dict of item parameter tensors
    log_probs / log_density(latent, **item_params) -> per-observation log-likelihood
    nll(latent, emb, responses) -> scalar mean NLL

For the continuous decoders "log_probs" is a log-DENSITY rather than a
log-probability over a finite outcome set, so the method is named
``log_density``; ``nll`` is provided under the shared name for a uniform contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Log-normal response-time decoder (van der Linden 2007)
# ---------------------------------------------------------------------------

class LogNormalRTDecoder(nn.Module):
    r"""
    Item embedding -> time-intensity beta_i -> log-normal response-time density.

    van der Linden's (2007) hierarchical speed model.  For person speed latent
    :math:`\tau` and item time-intensity :math:`\beta_i`,

        log(RT) ~ Normal( tau - beta_i , sigma^2 )

    so a faster person (larger tau) gives smaller log-RT, and a more
    time-intensive item (larger beta_i) gives larger log-RT.  ``sigma`` is a
    shared (or per-item) residual scale on the log-time scale.

    Item parameters from the embedding
    ----------------------------------
    The embedding maps to the time-intensity ``beta`` (a real scalar) and, when
    ``per_item_sigma`` is set, to a per-item log-residual-scale; otherwise a
    single global ``log_sigma`` parameter is shared across items.

    Identifiability
    ---------------
    The mean log-RT depends on tau and beta only through ``tau - beta_i``, so the
    pair is confounded by an additive constant: shifting every tau by +c and
    every beta_i by +c leaves the likelihood unchanged.  One location must be
    pinned.  The canonical van der Linden anchor fixes the person-speed
    distribution to mean zero (E[tau] = 0); equivalently one can sum-to-zero the
    item time-intensities.  This decoder does not own the person latent, so it
    cannot enforce the anchor internally -- the convention is that whoever
    supplies ``tau`` centers it (the synthetic recovery check below centers the
    fitted speeds before scoring, exactly as one would standardise the speed
    posterior).  ``sigma`` is identified directly by the residual spread and
    needs no constraint.

    Parameters
    ----------
    emb_dim        : int
    per_item_sigma : bool
        If True, predict a per-item log residual scale from the embedding.
        If False (default), use a single shared ``log_sigma`` parameter.
    """

    def __init__(self, emb_dim: int, per_item_sigma: bool = False) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.per_item_sigma = per_item_sigma

        self.fc_beta = nn.Linear(emb_dim, 1)        # -> time-intensity beta_i
        if per_item_sigma:
            self.fc_log_sigma = nn.Linear(emb_dim, 1)
        else:
            # single shared residual scale on log-time; init log(1)=0
            self.log_sigma = nn.Parameter(torch.zeros(1))

    # --- item_params ---

    def item_params(self, emb: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        emb : (..., emb_dim)

        Returns
        -------
        dict with:
            beta      : (..., 1)   time-intensity (real-valued)
            log_sigma : (..., 1)   log residual scale on log-time
        """
        beta = self.fc_beta(emb)                       # (..., 1)
        if self.per_item_sigma:
            log_sigma = self.fc_log_sigma(emb)         # (..., 1)
        else:
            shape = beta.shape
            log_sigma = self.log_sigma.expand(shape)   # broadcast shared scale
        return {"beta": beta, "log_sigma": log_sigma}

    # --- log density ---

    def log_density(
        self,
        tau: torch.Tensor,
        beta: torch.Tensor,
        log_sigma: torch.Tensor,
        log_rt: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Log-density of the observed log-RT under Normal(tau - beta, sigma^2).

        log N(y; m, sigma^2) = -0.5*log(2 pi) - log_sigma - 0.5*((y-m)/sigma)^2

        Parameters
        ----------
        tau       : (batch,) or (batch, 1)   person speed latent
        beta      : (batch, 1)               item time-intensity
        log_sigma : (batch, 1)               log residual scale
        log_rt    : (batch,) or (batch, 1)   observed log response time

        Returns
        -------
        log_density : (batch,)
        """
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        if log_rt.dim() == 1:
            log_rt = log_rt.unsqueeze(1)
        mean = tau - beta                              # (batch, 1)
        sigma = log_sigma.exp()
        z = (log_rt - mean) / sigma                    # (batch, 1)
        ld = -0.5 * _LOG_2PI - log_sigma - 0.5 * z * z
        return ld.squeeze(1)                           # (batch,)

    # --- nll convenience ---

    def nll(
        self,
        tau: torch.Tensor,
        emb: torch.Tensor,
        log_rt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean NLL for a flat batch of log response times.

        Parameters
        ----------
        tau    : (N,)            person speed latents
        emb    : (N, emb_dim)
        log_rt : (N,)            observed log response times

        Returns
        -------
        loss : scalar
        """
        params = self.item_params(emb)
        ld = self.log_density(tau, params["beta"], params["log_sigma"], log_rt)
        return -ld.mean()


# ---------------------------------------------------------------------------
# Beta continuous-response decoder (bounded scores in (0,1))
# ---------------------------------------------------------------------------

class BetaResponseDecoder(nn.Module):
    r"""
    Item embedding -> (a_i, b_i) -> Beta-distributed continuous response density.

    For bounded continuous scores y in (0,1) (e.g. fractional credit, normalised
    rubric scores).  The conditional mean is an IRT-style sigmoid,

        mu_i = sigmoid( a_i * (theta - b_i) )

    and the response is Beta-distributed with that mean and a precision phi.  The
    Beta is taken in its mean-precision parameterisation,

        alpha = mu * phi,   beta_param = (1 - mu) * phi,
        y ~ Beta(alpha, beta_param),   E[y] = mu,   Var[y] = mu(1-mu)/(phi+1).

    Larger phi -> tighter responses around the mean.

    Item parameters from the embedding
    ----------------------------------
    a_i (discrimination, softplus -> positive) and b_i (difficulty, real).  The
    precision phi is a single shared positive parameter (softplus of a free
    scalar), which is the usual choice; a per-item precision head is available
    via ``per_item_phi``.

    Identifiability
    ---------------
    This is a 2PL mean structure, so the latent metric is fixed exactly as for
    binary 2PL: theta has an arbitrary location and scale, pinned by the standard
    N(0,1) ability convention, and the sign is pinned by forcing a_i > 0 (so
    larger theta -> larger mean, larger b_i -> smaller mean).  phi is identified
    by the response spread.

    Parameters
    ----------
    emb_dim      : int
    per_item_phi : bool
        If True, predict a per-item log-precision from the embedding.
        If False (default), share one global precision across items.
    """

    def __init__(self, emb_dim: int, per_item_phi: bool = False) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.per_item_phi = per_item_phi

        self.fc_a = nn.Linear(emb_dim, 1)              # -> discrimination (softplus)
        self.fc_b = nn.Linear(emb_dim, 1)              # -> difficulty (real)
        if per_item_phi:
            self.fc_log_phi = nn.Linear(emb_dim, 1)
        else:
            # shared log-precision; init so phi = softplus(0) ~ 0.69 is gentle,
            # use a positive init giving phi ~ 5 for a reasonable starting spread.
            self.log_phi_raw = nn.Parameter(torch.tensor([2.0]))

    # --- item_params ---

    def item_params(self, emb: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        emb : (..., emb_dim)

        Returns
        -------
        dict with:
            a   : (..., 1)   discrimination (softplus, > 0)
            b   : (..., 1)   difficulty (real)
            phi : (..., 1)   precision (> 0)
        """
        a = F.softplus(self.fc_a(emb))                 # > 0
        b = self.fc_b(emb)
        if self.per_item_phi:
            phi = F.softplus(self.fc_log_phi(emb)) + 1e-3
        else:
            phi = (F.softplus(self.log_phi_raw) + 1e-3).expand(a.shape)
        return {"a": a, "b": b, "phi": phi}

    # --- log density ---

    def log_density(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        phi: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        r"""
        Beta log-density in the mean-precision parameterisation.

        log f(y) = lgamma(phi) - lgamma(mu*phi) - lgamma((1-mu)*phi)
                   + (mu*phi - 1) log y + ((1-mu)*phi - 1) log(1-y)

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        a     : (batch, 1)
        b     : (batch, 1)
        phi   : (batch, 1)
        y     : (batch,) or (batch, 1)   responses in (0,1)
        eps   : clamp to keep y and the shape params strictly positive

        Returns
        -------
        log_density : (batch,)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        mu = torch.sigmoid(a * (theta - b))            # (batch, 1) in (0,1)
        # Keep shape params away from 0; clamp mu strictly inside (0,1).
        mu = mu.clamp(eps, 1.0 - eps)
        y = y.clamp(eps, 1.0 - eps)
        alpha = mu * phi
        beta_p = (1.0 - mu) * phi
        log_b = (
            torch.lgamma(alpha + beta_p)
            - torch.lgamma(alpha)
            - torch.lgamma(beta_p)
        )
        ld = (
            log_b
            + (alpha - 1.0) * torch.log(y)
            + (beta_p - 1.0) * torch.log1p(-y)
        )
        return ld.squeeze(1)                           # (batch,)

    # --- nll convenience ---

    def nll(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean NLL for a flat batch of continuous responses in (0,1).

        Parameters
        ----------
        theta     : (N,)
        emb       : (N, emb_dim)
        responses : (N,)   floats in (0,1)

        Returns
        -------
        loss : scalar
        """
        params = self.item_params(emb)
        ld = self.log_density(
            theta, params["a"], params["b"], params["phi"], responses
        )
        return -ld.mean()


# ---------------------------------------------------------------------------
# Poisson count decoder
# ---------------------------------------------------------------------------

class PoissonCountDecoder(nn.Module):
    r"""
    Item embedding -> (a_i, b_i) -> Poisson-distributed count log-pmf.

    For non-negative integer counts y (e.g. number of attempts, hint requests,
    items completed in a window).  The conditional rate is an exponential IRT
    link,

        lambda_i = exp( a_i * (theta - b_i) )        (when discrimination=True)
        lambda_i = exp( theta - b_i )                (when discrimination=False)

    and y ~ Poisson(lambda_i).  Larger theta -> larger rate; larger b_i ->
    smaller rate.  The exponential link guarantees lambda > 0.

    Item parameters from the embedding
    ----------------------------------
    b_i (difficulty / log-rate offset, real) always; a_i (discrimination,
    softplus -> positive) when ``discrimination=True``.

    Identifiability
    ---------------
    The log-rate is an affine 2PL form in theta, so the latent metric carries the
    usual 2PL indeterminacy: theta has an arbitrary location/scale fixed by the
    N(0,1) convention.  With discrimination=True the sign of theta is pinned by
    a_i > 0; the b_i location is then identified relative to the centred theta.
    With discrimination=False (a_i == 1) the model is a one-parameter Rasch-style
    count model and theta/b share a single additive offset that the N(0,1)
    convention removes.

    Parameters
    ----------
    emb_dim        : int
    discrimination : bool
        If True (default), learn a per-item discrimination a_i.
        If False, fix a_i == 1 (Rasch-style count model).
    """

    def __init__(self, emb_dim: int, discrimination: bool = True) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.discrimination = discrimination

        self.fc_b = nn.Linear(emb_dim, 1)              # -> difficulty (real)
        if discrimination:
            self.fc_a = nn.Linear(emb_dim, 1)          # -> discrimination (softplus)

    # --- item_params ---

    def item_params(self, emb: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        emb : (..., emb_dim)

        Returns
        -------
        dict with:
            b : (..., 1)   difficulty / log-rate offset (real)
            a : (..., 1)   discrimination (> 0); only when discrimination=True
        """
        b = self.fc_b(emb)
        if self.discrimination:
            a = F.softplus(self.fc_a(emb))             # > 0
            return {"a": a, "b": b}
        return {"b": b}

    # --- log pmf ---

    def log_probs(
        self,
        theta: torch.Tensor,
        b: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Poisson log-pmf: log P(y) = y*log(lambda) - lambda - lgamma(y+1).

        Parameters
        ----------
        theta : (batch,) or (batch, 1)
        b     : (batch, 1)
        y     : (batch,) or (batch, 1)   non-negative counts (float or long)
        a     : (batch, 1) or None       discrimination; if None, treated as 1

        Returns
        -------
        log_probs : (batch,)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y = y.to(theta.dtype)
        if a is not None:
            log_lambda = a * (theta - b)               # (batch, 1)
        else:
            log_lambda = theta - b
        lam = log_lambda.exp()
        lp = y * log_lambda - lam - torch.lgamma(y + 1.0)
        return lp.squeeze(1)                           # (batch,)

    # --- nll convenience ---

    def nll(
        self,
        theta: torch.Tensor,
        emb: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean NLL for a flat batch of counts.

        Parameters
        ----------
        theta     : (N,)
        emb       : (N, emb_dim)
        responses : (N,)   non-negative counts

        Returns
        -------
        loss : scalar
        """
        params = self.item_params(emb)
        a = params.get("a", None)
        lp = self.log_probs(theta, params["b"], responses, a=a)
        return -lp.mean()
