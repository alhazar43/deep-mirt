"""Synthetic user side of the v1 generator.

Draws ``N`` synthetic users with 1D ``theta`` ~ N(0, 1) and generates a
full-questionnaire response per user against a mixed 2PL + GPCM bank
matching Section 5.1 of the v1 plan.

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
