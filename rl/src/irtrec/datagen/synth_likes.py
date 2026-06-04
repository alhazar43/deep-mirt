"""Synthetic like / dislike side of the v1 + v2 generator.

v1 (M3) implements the Option A binary preference model from Section
5.3 of the v1 plan. Engaged users like a job ``j`` with probability

.. math::

    P(\\text{IsLiked} = 1 \\mid u, j) = \\sigma(\\lambda (\\theta_u - \\delta_j) - b)

Rejecter users always emit IsLiked = 0. lambda and bias are calibrated
jointly to hit a target overall like rate.

v2 (M4-RL) replaces the binary sigmoid with a K=5 GPCM ordinal
response. Each user has a positive discrimination scalar
``lambda_u`` (drawn upstream in :mod:`synth_users`) and emits a
category ``y in {0, 1, 2, 3, 4}`` per candidate job under

.. math::

    y_{uj} \\sim \\text{GPCM}(\\lambda_u \\theta_u - \\delta_j,
                              \\beta = (-1.5, -0.5, 0.5, 1.5))

A backward-compatible ``IsLiked = 1[y >= 3]`` is derived for v1 columns.
The oracle :func:`oracle_like_prob` returns ``P(y >= 3)`` for use
inside StudentEnv in M5-RL only; it must never be exposed as state.

The candidate-set size ``n_u`` for each user is drawn from a clipped
negative binomial. The NegBinom(r=2, p=0.05, clip=[1, 200]) preset
matches the empirically observed shape of real recommender feedback.

This module exposes :func:`build_likes_table` (v1) and
:func:`build_likes_table_v2` (v2) returning long-form like records
ready for ``likes.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class LikesGenSpec:
    """Configuration for :func:`build_likes_table`.

    Attributes
    ----------
    target_like_rate
        Target marginal like rate over the long-form table (across all
        engaged-user candidate pairs after rejecter pairs are folded in
        with IsLiked = 0). Calibrated to ``+/- 0.005`` by bisection.
    rejecter_share
        Share of users with z_u = 0 (rejecter class). Their candidate
        pairs are still emitted with IsLiked = 0 so the table has full
        coverage.
    engaged_share
        Share of users with z_u = 1 (engaged class). Must satisfy
        ``rejecter_share + engaged_share == 1``.
    lambda_range
        Inclusive ``(lo, hi)`` search range for ``lambda``.
    bias_range
        Inclusive ``(lo, hi)`` search range for ``bias``.
    negbinom_r, negbinom_p
        Negative binomial parameters for per-user candidate set size.
        scipy / numpy parameterise NegBinom by (r, p). Mean is
        ``r * (1 - p) / p``.
    negbinom_clip
        Inclusive clamp on ``n_u``. Defaults to [1, 200].
    """

    target_like_rate: float = 0.20
    rejecter_share: float = 0.40
    engaged_share: float = 0.60
    lambda_range: Tuple[float, float] = (0.5, 3.0)
    bias_range: Tuple[float, float] = (-2.0, 2.0)
    negbinom_r: float = 2.0
    negbinom_p: float = 0.05
    negbinom_clip: Tuple[int, int] = (1, 200)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid that avoids overflow on large negative inputs.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg = ~pos
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


def assign_engagement(
    *,
    n_users: int,
    rejecter_share: float,
    seed: int = 0,
) -> np.ndarray:
    """Assign engagement class per user via stratified sampling.

    To keep the realised share within tight tolerance of the target
    even at small N (the dev preset uses N=500), we draw exact counts
    deterministically and permute the assignment. Binomial sampling at
    N=500 has SE ~ 0.022 which routinely produces realised shares
    outside the +/- 0.03 sanity-check band.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_users,)`` where 0 = rejecter and
        1 = engaged.
    """
    rng = np.random.default_rng(seed)
    n_rejecter = int(round(rejecter_share * n_users))
    n_rejecter = max(0, min(n_users, n_rejecter))
    z = np.ones(n_users, dtype=np.int64)
    z[:n_rejecter] = 0
    rng.shuffle(z)
    return z


def sample_candidate_set_sizes(
    *,
    n_users: int,
    spec: LikesGenSpec,
    seed: int = 0,
) -> np.ndarray:
    """Draw per-user candidate set sizes from clipped NegBinom.

    NumPy parameterises NegBinom as ``(n, p)`` where ``n`` plays the
    role of ``r`` (number of successes to terminate). Mean is
    ``n * (1 - p) / p``.
    """
    rng = np.random.default_rng(seed)
    draws = rng.negative_binomial(spec.negbinom_r, spec.negbinom_p, size=n_users)
    lo, hi = spec.negbinom_clip
    draws = np.clip(draws, lo, hi).astype(np.int64)
    return draws


def sample_candidate_jobs(
    *,
    n_jobs: int,
    candidate_sizes: np.ndarray,
    seed: int = 0,
) -> List[np.ndarray]:
    """Sample, for each user, an n_u-sized set of candidate job IDs.

    Sampling is uniform without replacement so a job appears at most
    once per user. When ``n_u > n_jobs`` (which the NegBinom clip is
    set to prevent), we fall back to sampling with replacement.

    Returns
    -------
    list of np.ndarray
        One array of job IDs per user.
    """
    rng = np.random.default_rng(seed)
    sets: List[np.ndarray] = []
    all_jobs = np.arange(n_jobs, dtype=np.int64)
    for n_u in candidate_sizes.tolist():
        if n_u >= n_jobs:
            # Should not happen at default clip but defined behaviour.
            sample = rng.choice(all_jobs, size=n_jobs, replace=False)
        else:
            sample = rng.choice(all_jobs, size=int(n_u), replace=False)
        sample = np.sort(sample)
        sets.append(sample.astype(np.int64))
    return sets


def _like_probability(
    theta_engaged: np.ndarray,
    delta: np.ndarray,
    lam: float,
    bias: float,
) -> np.ndarray:
    """Return the engaged-user like probabilities for all (u, j) pairs.

    ``theta_engaged`` and ``delta`` are 1D arrays of equal length, one
    entry per (user, candidate-job) pair. The output is the elementwise
    sigmoid of the cumulative logit.
    """
    logits = lam * (theta_engaged - delta) - bias
    return _sigmoid(logits)


def calibrate_lambda_bias(
    *,
    theta_engaged: np.ndarray,
    delta_engaged: np.ndarray,
    target_like_rate: float,
    lambda_range: Tuple[float, float],
    bias_range: Tuple[float, float],
    max_iter: int = 60,
    rate_tol: float = 0.005,
) -> Tuple[float, float]:
    """Calibrate ``(lambda, bias)`` to hit the target like rate.

    Bisection happens only over ``bias`` since the marginal like rate is
    monotone decreasing in ``bias`` (larger bias subtracts more from the
    logit). ``lambda`` is chosen as the midpoint of its range, which is
    deterministic given the spec and reproducible.

    Parameters
    ----------
    theta_engaged
        1D array of engaged-user theta values, repeated to match the
        long-form pair table (one entry per (engaged_user, candidate)
        pair).
    delta_engaged
        1D array of the matching delta_j values, same length as
        ``theta_engaged``.
    target_like_rate
        Target marginal like rate on the engaged-user pair table.
    lambda_range
        Search range for ``lambda``. The chosen value is the midpoint.
    bias_range
        Bisection range for ``bias``.

    Returns
    -------
    (float, float)
        Calibrated ``(lambda, bias)``.

    Notes
    -----
    The target rate here is the engaged-only like rate. The overall
    table-level rate (with rejecters folded in) is::

        overall = engaged_share * engaged_rate

    so the caller passes ``target / engaged_share`` as the engaged
    target. :func:`build_likes_table` handles this conversion.
    """
    lam = 0.5 * (lambda_range[0] + lambda_range[1])

    def rate_at(bias_val: float) -> float:
        probs = _like_probability(theta_engaged, delta_engaged, lam, bias_val)
        return float(probs.mean())

    lo, hi = bias_range
    rate_lo = rate_at(lo)  # bias low -> high like rate
    rate_hi = rate_at(hi)  # bias high -> low like rate
    if rate_lo < target_like_rate or rate_hi > target_like_rate:
        # Target outside the achievable range. Fall back to the closer
        # boundary, which is still the best feasible answer.
        if abs(rate_lo - target_like_rate) <= abs(rate_hi - target_like_rate):
            return lam, lo
        return lam, hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate_mid = rate_at(mid)
        if abs(rate_mid - target_like_rate) <= rate_tol:
            return lam, float(mid)
        if rate_mid > target_like_rate:
            # Need lower like rate, raise bias floor.
            lo = mid
        else:
            hi = mid
    return lam, float(0.5 * (lo + hi))


def build_likes_table(
    *,
    theta: np.ndarray,
    engagement: np.ndarray,
    candidate_sets: List[np.ndarray],
    delta_j: np.ndarray,
    spec: LikesGenSpec,
    seed: int = 0,
) -> Tuple[List[dict], float, float]:
    """Assemble the long-form likes table.

    Steps

    1. Build the long-form (user_id, job_id) pair table over engaged
       users only.
    2. Calibrate ``(lambda, bias)`` so the engaged-only like rate hits
       ``target_like_rate / engaged_share``. This makes the
       overall (engaged + rejecter) rate match the spec target.
    3. Sample Bernoulli IsLiked per engaged pair.
    4. Append rejecter pairs with IsLiked = 0 and ``prob = 0.0``.
    5. Sort the table by ``(user_id, job_id)`` for byte-identical
       reproducibility.

    Parameters
    ----------
    theta
        Per-user 1D theta, shape ``(N,)``.
    engagement
        Per-user engagement class, shape ``(N,)`` with 1 = engaged.
    candidate_sets
        Per-user candidate job ID arrays, length N.
    delta_j
        Per-job delta scalar, shape ``(n_jobs,)``.
    spec
        Spec controlling calibration ranges and target rate.
    seed
        Seed for the Bernoulli sampling RNG.

    Returns
    -------
    (list, float, float)
        ``(records, lambda, bias)`` where ``records`` is the long-form
        likes table sorted by ``(user_id, job_id)``.
    """
    n_users = int(theta.shape[0])
    assert engagement.shape[0] == n_users
    assert len(candidate_sets) == n_users

    engaged_mask = engagement == 1
    engaged_uids = np.where(engaged_mask)[0]
    rejecter_uids = np.where(~engaged_mask)[0]

    # Build engaged-pair table.
    engaged_user_repeat: List[int] = []
    engaged_job_repeat: List[int] = []
    for uid in engaged_uids.tolist():
        cand = candidate_sets[uid]
        if cand.size == 0:
            continue
        engaged_user_repeat.extend([uid] * int(cand.size))
        engaged_job_repeat.extend(cand.tolist())
    eu = np.asarray(engaged_user_repeat, dtype=np.int64)
    ej = np.asarray(engaged_job_repeat, dtype=np.int64)
    if eu.size == 0:
        # No engaged users; calibration is undefined. Return rejecter-
        # only table with all-zero likes.
        lam = 0.5 * (spec.lambda_range[0] + spec.lambda_range[1])
        bias = 0.5 * (spec.bias_range[0] + spec.bias_range[1])
        records: List[dict] = []
        for uid in rejecter_uids.tolist():
            for j in candidate_sets[uid].tolist():
                records.append(
                    {
                        "user_id": int(uid),
                        "job_id": int(j),
                        "IsLiked": 0,
                        "prob": 0.0,
                    }
                )
        records.sort(key=lambda r: (r["user_id"], r["job_id"]))
        return records, float(lam), float(bias)

    theta_engaged = theta[eu]
    delta_engaged = delta_j[ej]

    # Convert overall target to engaged-only target. Rejecter pairs are
    # all zeros so engaged_target = target / engaged_share.
    engaged_share = float(spec.engaged_share)
    engaged_target = float(spec.target_like_rate) / max(engaged_share, 1e-9)

    lam, bias = calibrate_lambda_bias(
        theta_engaged=theta_engaged,
        delta_engaged=delta_engaged,
        target_like_rate=engaged_target,
        lambda_range=spec.lambda_range,
        bias_range=spec.bias_range,
    )

    probs = _like_probability(theta_engaged, delta_engaged, lam, bias)
    rng = np.random.default_rng(seed)
    likes = (rng.random(probs.size) < probs).astype(np.int64)

    records = []
    for k in range(eu.size):
        records.append(
            {
                "user_id": int(eu[k]),
                "job_id": int(ej[k]),
                "IsLiked": int(likes[k]),
                "prob": float(probs[k]),
            }
        )
    for uid in rejecter_uids.tolist():
        for j in candidate_sets[uid].tolist():
            records.append(
                {
                    "user_id": int(uid),
                    "job_id": int(j),
                    "IsLiked": 0,
                    "prob": 0.0,
                }
            )
    records.sort(key=lambda r: (r["user_id"], r["job_id"]))
    return records, float(lam), float(bias)


# ---------------------------------------------------------------------
# v2 helpers (M4-RL): K=5 GPCM ordinal response.
# ---------------------------------------------------------------------


# v2 step thresholds. K=5 categories means K-1=4 thresholds.
GPCM_K: int = 5
GPCM_BETA: tuple[float, float, float, float] = (-1.5, -0.5, 0.5, 1.5)
GPCM_LIKED_CATEGORY: int = 3  # y >= GPCM_LIKED_CATEGORY counts as IsLiked = 1.


def gpcm_category_probs(
    *,
    theta: float,
    lam: float,
    delta_j: np.ndarray,
    beta: tuple[float, ...] = GPCM_BETA,
) -> np.ndarray:
    """Vectorised GPCM category probabilities for one user vs. many jobs.

    Cumulative-logit form for K categories with discrimination
    ``lambda_u`` and step thresholds shifted by per-job ``delta_j``,

    .. math::

        \\phi_{j,0} &= 0 \\\\
        \\phi_{j,k} &= \\sum_{h=0}^{k-1}
            [\\lambda_u \\theta_u - (\\beta_h + \\delta_j)],
            \\quad k = 1, \\dots, K - 1 \\\\
        P(Y_{uj} = k) &= \\frac{\\exp(\\phi_{j,k})}
                              {\\sum_{l=0}^{K-1} \\exp(\\phi_{j,l})}.

    Parameters
    ----------
    theta
        Scalar ability for one user.
    lam
        Positive discrimination scalar for the user.
    delta_j
        Per-job difficulty array, shape ``(n_jobs,)``.
    beta
        K-1 step thresholds, ascending. Defaults to :data:`GPCM_BETA`.

    Returns
    -------
    np.ndarray
        Probability matrix of shape ``(n_jobs, K)``, rows sum to 1.
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    K = beta_arr.shape[0] + 1
    n_jobs = delta_j.shape[0]
    psi = np.zeros((n_jobs, K), dtype=np.float64)
    cumsum = np.zeros(n_jobs, dtype=np.float64)
    for k in range(1, K):
        cumsum = cumsum + lam * theta - (beta_arr[k - 1] + delta_j)
        psi[:, k] = cumsum
    # Stable softmax.
    psi -= psi.max(axis=1, keepdims=True)
    p = np.exp(psi)
    p /= p.sum(axis=1, keepdims=True)
    return p


def oracle_like_prob(
    *,
    theta: float,
    lam: float,
    delta_j: np.ndarray,
    beta: tuple[float, ...] = GPCM_BETA,
    liked_category: int = GPCM_LIKED_CATEGORY,
) -> np.ndarray:
    """Oracle ``P(y >= liked_category | theta, j)`` under the v2 GPCM.

    Returns the per-job probability mass at the "liked" tail of the
    response distribution. Used inside :class:`StudentEnv` (M5-RL) for
    Bayes-ceiling diagnostics and reward shaping. **Must never be
    exposed to the policy as state input**, otherwise the policy could
    inflate its own reward by reading the oracle.

    Returns
    -------
    np.ndarray
        Shape ``(n_jobs,)``, values in ``[0, 1]``.
    """
    if liked_category < 1 or liked_category >= GPCM_K:
        raise ValueError(
            f"liked_category must be in [1, {GPCM_K - 1}], got {liked_category}"
        )
    p = gpcm_category_probs(theta=theta, lam=lam, delta_j=delta_j, beta=beta)
    return p[:, liked_category:].sum(axis=1)


def sample_gpcm_response(
    *,
    theta: float,
    lam: float,
    delta_j: np.ndarray,
    rng: np.random.Generator,
    beta: tuple[float, ...] = GPCM_BETA,
) -> np.ndarray:
    """Sample one K=5 ordinal response per job for a single user.

    Parameters
    ----------
    theta
        Scalar ability for the user.
    lam
        Positive discrimination scalar for the user.
    delta_j
        Per-job difficulty, shape ``(n_jobs,)``.
    rng
        Generator to draw uniforms from. The caller controls seeding.
    beta
        K-1 step thresholds.

    Returns
    -------
    np.ndarray
        Integer category array of shape ``(n_jobs,)`` in ``{0, ..., K-1}``.
    """
    p = gpcm_category_probs(theta=theta, lam=lam, delta_j=delta_j, beta=beta)
    cum = np.cumsum(p, axis=1)
    u = rng.uniform(size=delta_j.shape[0])
    y = (u[:, None] > cum).sum(axis=1).astype(np.int64)
    # The expression above can produce K for u == 1.0 exactly; clip to K-1.
    np.clip(y, 0, GPCM_K - 1, out=y)
    return y


def build_likes_table_v2(
    *,
    theta: np.ndarray,
    lambda_u: np.ndarray,
    candidate_sets: List[np.ndarray],
    delta_j: np.ndarray,
    beta: tuple[float, ...] = GPCM_BETA,
    seed: int = 0,
) -> List[dict]:
    """Assemble the long-form v2 likes table with K=5 ordinal responses.

    Each (user, candidate-job) pair gets a sampled ``y in {0, ..., 4}``
    under the per-user GPCM and a backward-compatible ``IsLiked = 1[y
    >= 3]`` field. The simulator's oracle like-probability ``prob =
    P(y >= 3)`` is also recorded for downstream diagnostics; the
    policy must not consume it as state.

    Parameters
    ----------
    theta
        Per-user 1D theta, shape ``(N,)``.
    lambda_u
        Per-user positive discrimination, shape ``(N,)``.
    candidate_sets
        Per-user candidate job ID arrays, length N.
    delta_j
        Per-job difficulty, shape ``(n_jobs,)``.
    beta
        K-1 step thresholds. Defaults to :data:`GPCM_BETA`.
    seed
        Seed for the response RNG.

    Returns
    -------
    list of dict
        Long-form table sorted by ``(user_id, job_id)`` with keys
        ``user_id, job_id, y, IsLiked, prob``.
    """
    n_users = int(theta.shape[0])
    if lambda_u.shape[0] != n_users:
        raise ValueError("theta and lambda_u must have matching length")
    if len(candidate_sets) != n_users:
        raise ValueError("candidate_sets length must match theta length")

    rng = np.random.default_rng(seed)
    records: List[dict] = []
    for uid in range(n_users):
        cand = candidate_sets[uid]
        if cand.size == 0:
            continue
        cand_deltas = delta_j[cand]
        probs = oracle_like_prob(
            theta=float(theta[uid]),
            lam=float(lambda_u[uid]),
            delta_j=cand_deltas,
            beta=beta,
        )
        y = sample_gpcm_response(
            theta=float(theta[uid]),
            lam=float(lambda_u[uid]),
            delta_j=cand_deltas,
            rng=rng,
            beta=beta,
        )
        is_liked = (y >= GPCM_LIKED_CATEGORY).astype(np.int64)
        for k in range(cand.size):
            records.append(
                {
                    "user_id": int(uid),
                    "job_id": int(cand[k]),
                    "y": int(y[k]),
                    "IsLiked": int(is_liked[k]),
                    "prob": float(probs[k]),
                }
            )
    records.sort(key=lambda r: (r["user_id"], r["job_id"]))
    return records
