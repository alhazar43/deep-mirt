"""Synthetic user side of the v1 + v2 generator.

v1 (M3) draws ``N`` synthetic users with 1D ``theta`` ~ N(0, 1) and
generates a full-questionnaire response per user against a mixed 2PL +
GPCM bank matching Section 5.1 of the v1 plan.

v2 (M4-RL) adds per-user discrimination heterogeneity by sampling
``lambda_u`` ~ LogNormal(log 1.5, 0.4) alongside ``theta`` and exposes
:func:`sample_users_v2` and :func:`stratified_split` helpers. The
engagement-mixture concept from v1 (rejecter / engaged) is dropped;
all v2 users are treated as engaged and emit K=5 GPCM ordinal
responses inside :mod:`synth_likes`.

Item bank composition

  - 25 binary 2PL items (K = 2)
  - 15 GPCM items at K = 3
  -  8 GPCM items at K = 5
  -  2 GPCM items at K = 6

Each user answers every item in the bank (50 responses), so the
per-user response count satisfies the >= 30 sanity check by
construction.

Item parameters

  - alpha ~ LogNormal(mean=0, sigma=0.4)
  - GPCM thresholds, per item, are K-1 values drawn N(b_mean, 0.5) then
    sorted ascending. ``b_mean`` ~ N(0, 0.7).
  - 2PL difficulty is the single sorted threshold (b_mean itself, no
    extra jitter from the N(b_mean, 0.5) draw because K-1 = 1 and the
    single draw plus sort collapses to one value).

Output schema notes

  ``sequences.json`` records use 1-based question IDs to match the
  existing ma-irt loader contract (0 is reserved for padding). Question
  IDs index into the flat bank in the order returned by
  :func:`build_item_bank` (25 binary first, then K=3, K=5, K=6).

  ``true_irt_parameters.json`` follows the existing ma-irt schema, with
  ``alpha`` as a list of scalars and ``beta`` as a list of variable-
  length lists (each of length K_q - 1). Consumers that assume a square
  ``beta`` matrix must handle the variable K case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

# Default bank composition from Section 5.1.
DEFAULT_BANK_COMPOSITION: Sequence[tuple[int, int]] = (
    (2, 25),
    (3, 15),
    (5, 8),
    (6, 2),
)
# Total items in the default bank.
DEFAULT_BANK_SIZE: int = sum(count for _, count in DEFAULT_BANK_COMPOSITION)


@dataclass
class ItemBank:
    """Container for the questionnaire item bank.

    Attributes
    ----------
    n_categories
        Per-item number of categories (length n_items).
    alpha
        Discrimination per item (length n_items, 1D).
    beta
        Per-item sorted thresholds, list of length n_items. Each entry
        has length ``n_categories[i] - 1``.
    """

    n_categories: np.ndarray
    alpha: np.ndarray
    beta: List[np.ndarray] = field(default_factory=list)

    @property
    def n_items(self) -> int:
        return int(self.n_categories.shape[0])

    @property
    def k_distribution(self) -> dict:
        """Mapping K -> count of items at that K in the bank."""
        unique, counts = np.unique(self.n_categories, return_counts=True)
        return {int(k): int(c) for k, c in zip(unique, counts)}

    def beta_as_list(self) -> List[List[float]]:
        """JSON-friendly nested list of thresholds."""
        return [arr.tolist() for arr in self.beta]


def build_item_bank(
    *,
    composition: Sequence[tuple[int, int]] = DEFAULT_BANK_COMPOSITION,
    alpha_sigma: float = 0.4,
    bmean_sigma: float = 0.7,
    threshold_sigma: float = 0.5,
    seed: int = 0,
) -> ItemBank:
    """Construct the mixed 2PL + GPCM item bank.

    Parameters
    ----------
    composition
        Sequence of ``(K, count)`` pairs. Items are laid out in the
        order provided; question IDs are 0-based positions inside the
        flat bank.
    alpha_sigma
        Sigma for the LogNormal(0, sigma) alpha prior.
    bmean_sigma
        Sigma for the per-item N(0, sigma) base difficulty prior.
    threshold_sigma
        Sigma for the per-step N(b_mean, sigma) threshold draws (sorted
        per item to enforce monotonicity).
    seed
        Seed for this generator. Use a dedicated RNG so the bank can be
        reproduced independent of other draws.

    Returns
    -------
    ItemBank
        A populated bank ready for response generation.
    """
    rng = np.random.default_rng(seed)
    n_categories_list: List[int] = []
    for K, count in composition:
        n_categories_list.extend([int(K)] * int(count))
    n_categories = np.asarray(n_categories_list, dtype=np.int64)
    n_items = n_categories.shape[0]

    alpha = rng.lognormal(mean=0.0, sigma=alpha_sigma, size=n_items).astype(np.float64)
    beta: List[np.ndarray] = []
    for i in range(n_items):
        K = int(n_categories[i])
        b_mean = float(rng.normal(loc=0.0, scale=bmean_sigma))
        raw = rng.normal(loc=b_mean, scale=threshold_sigma, size=K - 1)
        beta.append(np.sort(raw).astype(np.float64))
    return ItemBank(n_categories=n_categories, alpha=alpha, beta=beta)


def gpcm_probabilities(
    theta: float,
    alpha: float,
    betas: np.ndarray,
) -> np.ndarray:
    """Standard scalar GPCM category probabilities (Muraki 1992).

    Cumulative logit form

    .. math::

        \\phi_0 &= 0 \\\\
        \\phi_k &= \\sum_{h=0}^{k-1} \\alpha (\\theta - \\beta_h),
            \\quad k = 1, \\dots, K - 1 \\\\
        P(Y = k \\mid \\theta) &= \\frac{\\exp(\\phi_k)}{\\sum_j \\exp(\\phi_j)}.

    Parameters
    ----------
    theta
        Scalar ability.
    alpha
        Scalar discrimination, must be positive.
    betas
        Sorted thresholds, shape ``(K - 1,)``. For 2PL (K = 2) this is
        a length-1 array and the formula reduces to the standard 2PL
        logit ``alpha * (theta - beta_0)``.

    Returns
    -------
    np.ndarray
        Probability vector of shape ``(K,)`` that sums to 1.
    """
    K = int(betas.shape[0]) + 1
    cum = np.zeros(K, dtype=np.float64)
    for k in range(1, K):
        cum[k] = float(np.sum(alpha * (theta - betas[:k])))
    # Numerically stable softmax.
    cum -= cum.max()
    exp_logits = np.exp(cum)
    return exp_logits / exp_logits.sum()


def sample_responses(
    *,
    theta: np.ndarray,
    bank: ItemBank,
    seed: int = 0,
) -> List[dict]:
    """Generate a full-questionnaire response per user.

    Each user receives every item in the bank in a fixed order (item
    index 0..n_items-1). The questionnaire is identical across users so
    ma-irt downstream can learn the item parameters cleanly.

    Parameters
    ----------
    theta
        1D true theta per user, shape ``(N,)``.
    bank
        Item bank used to score.
    seed
        Seed for the response RNG (independent of theta and bank).

    Returns
    -------
    list of dict
        Each entry has the schema expected by ma-irt loaders, namely
        ``{"questions": [int...], "responses": [int...]}``, with 1-based
        question IDs.
    """
    rng = np.random.default_rng(seed)
    n_users = int(theta.shape[0])
    n_items = bank.n_items
    # Fixed presentation order, 0-based item indices.
    item_order = np.arange(n_items, dtype=np.int64)
    questions_1based = (item_order + 1).tolist()

    sequences: List[dict] = []
    for u in range(n_users):
        theta_u = float(theta[u])
        responses: List[int] = []
        for i in range(n_items):
            K_i = int(bank.n_categories[i])
            probs = gpcm_probabilities(theta_u, float(bank.alpha[i]), bank.beta[i])
            r = int(rng.choice(K_i, p=probs))
            responses.append(r)
        sequences.append(
            {
                "questions": list(questions_1based),
                "responses": responses,
            }
        )
    return sequences


def sample_users(
    *,
    n_users: int,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n_users`` 1D theta values from N(0, 1).

    Uses a dedicated RNG so user draws are reproducible independent of
    item bank or response draws.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(n_users)).astype(np.float64)


# ---------------------------------------------------------------------
# v2 helpers (M4-RL).
# ---------------------------------------------------------------------


# v2 lambda_u prior. LogNormal(log 1.5, 0.4) has median 1.5, mean ~1.62,
# and a long right tail to capture high-discrimination users.
LAMBDA_U_LOG_MEAN: float = float(np.log(1.5))
LAMBDA_U_LOG_SIGMA: float = 0.4


def sample_users_v2(
    *,
    n_users: int,
    theta_seed: int = 0,
    lambda_seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw v2 user-level latent variables.

    Returns ``(theta, lambda_u)`` where

    - ``theta`` ~ N(0, 1) per user (1D ability scalar).
    - ``lambda_u`` ~ LogNormal(log 1.5, 0.4) per user
      (positive discrimination scalar).

    The two RNG streams are seeded independently so swapping one prior
    does not perturb the other set of draws.

    Parameters
    ----------
    n_users
        Number of users to sample.
    theta_seed
        Seed for the ``theta`` RNG.
    lambda_seed
        Seed for the ``lambda_u`` RNG. Must differ from ``theta_seed``
        so the two draws are independent.

    Returns
    -------
    theta, lambda_u
        Both shape ``(n_users,)`` and dtype float64.
    """
    if theta_seed == lambda_seed:
        raise ValueError(
            "theta_seed and lambda_seed must differ for independent draws"
        )
    theta_rng = np.random.default_rng(theta_seed)
    lam_rng = np.random.default_rng(lambda_seed)
    theta = theta_rng.standard_normal(int(n_users)).astype(np.float64)
    lambda_u = lam_rng.lognormal(
        mean=LAMBDA_U_LOG_MEAN,
        sigma=LAMBDA_U_LOG_SIGMA,
        size=int(n_users),
    ).astype(np.float64)
    return theta, lambda_u


def stratified_split(
    *,
    n_users: int,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Stratified 80 / 10 / 10 split of user IDs.

    The split is exact-count: we round ``train_frac`` and ``val_frac``
    to integer counts and assign the remainder to ``test`` so the union
    covers all users exactly once. The order of user IDs is shuffled
    with the provided seed so future splits can be reproduced from the
    same seed.

    Parameters
    ----------
    n_users
        Total number of users to split.
    train_frac, val_frac, test_frac
        Target proportions. Must sum to 1.0 within 1e-6.
    seed
        Seed for the user-ID shuffle.

    Returns
    -------
    dict
        Mapping ``"train" / "val" / "test"`` to int64 arrays of user IDs.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"split fractions must sum to 1.0, got {total:.6f}"
        )
    rng = np.random.default_rng(seed)
    ids = np.arange(int(n_users), dtype=np.int64)
    rng.shuffle(ids)
    n_train = int(round(train_frac * n_users))
    n_val = int(round(val_frac * n_users))
    # Clip and let the test partition take the remainder.
    n_train = max(0, min(n_users, n_train))
    n_val = max(0, min(n_users - n_train, n_val))
    train_ids = np.sort(ids[:n_train])
    val_ids = np.sort(ids[n_train : n_train + n_val])
    test_ids = np.sort(ids[n_train + n_val :])
    return {
        "train": train_ids.astype(np.int64),
        "val": val_ids.astype(np.int64),
        "test": test_ids.astype(np.int64),
    }
