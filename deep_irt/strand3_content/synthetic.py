"""synthetic.py -- mechanism proof for the content channel (pos / neg control).

We manufacture a world where CONTENT DETERMINES DIFFICULTY and ask whether the
content channel can place a held-out item on the ability scale from its content
alone (cold-start), recovering the true difficulty.

Generative model
----------------
- Each item i has a content vector ``c_i in R^{text_dim}`` (the text-embedding
  analogue), drawn iid standard normal.
- TRUE item difficulty is a deterministic linear function of the content:
      b_true_i = c_i . w_b              (w_b a fixed random direction)
  standardised to unit scale.  Discrimination is content-driven too:
      a_true_i = softplus(c_i . w_a + 0.5)   (kept positive, mild spread)
- Learner abilities theta_n ~ N(0, 1).
- Responses are sampled from the GPCM with K categories using
  ``a_true_i``, equally-spaced step thresholds centred at ``b_true_i``, and
  ``theta_n`` (a static-ability simulation -- the cleanest test of item
  placement; the dynamic engine still trains fine on it).

Positive vs negative control
----------------------------
- POSITIVE: the content channel sees the TRUE content vectors ``c_i``.  Content
  genuinely predicts difficulty, so cold-start from content should recover
  ``b_true`` on held-out items.
- NEGATIVE: the content vectors handed to the model are SHUFFLED across items,
  breaking the content<->difficulty link while keeping the SAME responses and
  the SAME marginal content distribution.  Content now carries no information
  about difficulty, so cold-start must collapse to ~0.  This rules out leakage:
  if cold-start still worked, it would not be using content.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SyntheticContentData:
    content: np.ndarray        # (n_items, text_dim) content vectors (the "text")
    b_true: np.ndarray         # (n_items,) true difficulty
    a_true: np.ndarray         # (n_items,) true discrimination
    item_ids: np.ndarray       # (n_learners, seq_len) 0-based item indices
    responses: np.ndarray      # (n_learners, seq_len) in {0..K-1}
    theta_true: np.ndarray     # (n_learners,) true ability
    n_items: int
    text_dim: int
    K: int


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _gpcm_probs(theta: float, a: float, b_steps: np.ndarray) -> np.ndarray:
    """GPCM category probabilities for one (theta, a, b_steps).

    b_steps : (K-1,) step thresholds.  psi_0 = 0; psi_k = sum_{c<=k} a*(theta-b_c).
    """
    diffs = a * (theta - b_steps)              # (K-1,)
    psi = np.concatenate([[0.0], np.cumsum(diffs)])   # (K,)
    psi = psi - psi.max()
    p = np.exp(psi)
    return p / p.sum()


def generate(
    n_items: int = 120,
    n_learners: int = 800,
    seq_len: int = 40,
    text_dim: int = 64,
    K: int = 3,
    seed: int = 0,
) -> SyntheticContentData:
    """Generate a content-determines-difficulty GPCM dataset.

    Each learner answers ``seq_len`` items sampled with replacement from the
    full bank, so every item is observed many times (clean calibration).
    """
    rng = np.random.default_rng(seed)

    # Content vectors -- the "text" of each item.
    content = rng.standard_normal((n_items, text_dim)).astype(np.float32)

    # Difficulty and discrimination are DETERMINISTIC functions of content.
    w_b = rng.standard_normal(text_dim)
    w_b /= np.linalg.norm(w_b)
    b_raw = content @ w_b
    b_true = (b_raw - b_raw.mean()) / (b_raw.std() + 1e-8)   # unit-scale difficulty

    w_a = rng.standard_normal(text_dim)
    w_a /= np.linalg.norm(w_a)
    a_true = _softplus(0.7 * (content @ w_a) + 0.5) + 0.3    # positive, mild spread

    theta_true = rng.standard_normal(n_learners)

    # K-1 equally spaced step thresholds centred on each item's b_true so the
    # mean step threshold equals b_true (matches the recovery's mean-b scalar).
    if K > 2:
        offsets = np.linspace(-0.8, 0.8, K - 1)
    else:
        offsets = np.array([0.0])

    item_ids = rng.integers(0, n_items, size=(n_learners, seq_len))
    responses = np.empty((n_learners, seq_len), dtype=np.int64)
    for n in range(n_learners):
        th = theta_true[n]
        for t in range(seq_len):
            i = item_ids[n, t]
            steps = b_true[i] + offsets
            p = _gpcm_probs(th, a_true[i], steps)
            responses[n, t] = rng.choice(K, p=p)

    return SyntheticContentData(
        content=content,
        b_true=b_true,
        a_true=a_true,
        item_ids=item_ids.astype(np.int64),
        responses=responses,
        theta_true=theta_true,
        n_items=n_items,
        text_dim=text_dim,
        K=K,
    )


def shuffle_content(data: SyntheticContentData, seed: int = 1) -> np.ndarray:
    """Return content rows permuted across items (negative control).

    Breaks the content<->difficulty link while preserving the marginal content
    distribution and leaving responses untouched.  A model given these features
    has no usable content signal for cold-start.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.n_items)
    return data.content[perm].copy()
