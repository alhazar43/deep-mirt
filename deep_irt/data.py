"""
data.py  --  synthetic GPCM data generation for the anchoring-drift v0 experiment.

Ground truth:
  theta_learner ~ N(0, 1)
  a_j           ~ LogNormal(0, 0.3)      [discrimination, >0]
  b_jk          ~ sorted K-1 draws from N(0, 1)   [step thresholds, ascending]

Every learner answers all B base items.
A fixed random half of learners (the "anchor set") also answer all E extension items.
Responses are drawn from the true GPCM and presented in random per-learner order.

Dynamic theta (theta_dynamics="random_walk"):
  theta_t = theta_0 + cumsum(N(0, drift_sigma) steps)
  The TRUE theta at each interaction is recorded in base_seqs[i]["theta_true"] (length B).
  For ext_seqs the theta is fixed at the learner's end-of-B theta (see anchoring note).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DataConfig:
    n_learners: int = 2000
    n_base: int = 100       # B
    n_ext: int = 30         # E
    n_cats: int = 4         # K  (categories 0 .. K-1)
    seed: int = 0
    ext_shift: float = 0.0  # additive shift applied to E step-threshold means
                             # (0.0 = same-distribution as B; >0 = harder items)
    # --- dynamic theta ---
    theta_dynamics: str = "static"   # "static" | "random_walk"
    drift_sigma: float = 0.12        # per-step std for random_walk
                                     # over ~100 steps -> total drift ~1.2 SD in expectation


@dataclass
class GroundTruth:
    theta: np.ndarray       # (N,) initial (static) or mean ability
    a_base: np.ndarray      # (B,)
    b_base: np.ndarray      # (B, K-1)
    a_ext: np.ndarray       # (E,)
    b_ext: np.ndarray       # (E, K-1)
    # dynamic-only: true theta at each interaction, aligned with base_seqs item order
    theta_traj: Optional[np.ndarray] = None   # (N, B) or None for static


@dataclass
class Dataset:
    # sequences for the B-only phase
    # For static:  {"items": [...], "responses": [...]}
    # For dynamic: {"items": [...], "responses": [...], "theta_true": [...]}
    #              theta_true[t] = the learner's true theta WHEN they answered item[t]
    base_seqs: list
    # sequences for the anchor learners over E items
    ext_seqs: list           # list of |anchor_set| dicts
    anchor_mask: np.ndarray  # boolean (N,) -- True for the half who answer E items
    gt: GroundTruth
    cfg: DataConfig


def _gpcm_probs(theta: float, a: float, b: np.ndarray) -> np.ndarray:
    """
    Compute GPCM category probabilities for a single (learner, item) pair.
    K = len(b) + 1 categories.

    psi_0 = 0
    psi_k = sum_{c=1..k}  a * (theta - b[c-1])   for k=1..K-1
    P(r=k) = exp(psi_k) / sum_m exp(psi_m)
    """
    K = len(b) + 1
    psi = np.zeros(K)
    for k in range(1, K):
        psi[k] = np.sum(a * (theta - b[:k]))
    # numerically stable softmax
    psi -= psi.max()
    exp_psi = np.exp(psi)
    return exp_psi / exp_psi.sum()


def _sample_response(theta: float, a: float, b: np.ndarray, rng: np.random.Generator) -> int:
    probs = _gpcm_probs(theta, a, b)
    return int(rng.choice(len(probs), p=probs))


def generate(cfg: DataConfig) -> Dataset:
    if cfg.theta_dynamics == "random_walk":
        return _generate_random_walk(cfg)
    return _generate_static(cfg)


def _generate_static(cfg: DataConfig) -> Dataset:
    """Original static-theta generation -- unchanged from v0/v0.1."""
    rng = np.random.default_rng(cfg.seed)
    N, B, E, K = cfg.n_learners, cfg.n_base, cfg.n_ext, cfg.n_cats

    # --- ground truth ---
    theta = rng.standard_normal(N)

    def sample_item_params(n_items):
        a = rng.lognormal(mean=0.0, sigma=0.3, size=n_items)
        b_raw = rng.standard_normal((n_items, K - 1))
        b = np.sort(b_raw, axis=1)
        return a, b

    a_base, b_base = sample_item_params(B)
    a_ext, b_ext_unshifted = sample_item_params(E)
    b_ext = b_ext_unshifted + cfg.ext_shift

    gt = GroundTruth(theta=theta, a_base=a_base, b_base=b_base,
                     a_ext=a_ext, b_ext=b_ext, theta_traj=None)

    anchor_mask = np.zeros(N, dtype=bool)
    anchor_idx = rng.choice(N, size=N // 2, replace=False)
    anchor_mask[anchor_idx] = True

    base_seqs = []
    for i in range(N):
        item_order = rng.permutation(B).tolist()
        responses = [_sample_response(theta[i], a_base[j], b_base[j], rng)
                     for j in item_order]
        base_seqs.append({"items": item_order, "responses": responses})

    ext_seqs = []
    for i in np.where(anchor_mask)[0]:
        item_order = rng.permutation(E).tolist()
        responses = [_sample_response(theta[i], a_ext[j], b_ext[j], rng)
                     for j in item_order]
        ext_seqs.append({"items": item_order, "responses": responses,
                         "learner_idx": int(i)})

    return Dataset(base_seqs=base_seqs, ext_seqs=ext_seqs,
                   anchor_mask=anchor_mask, gt=gt, cfg=cfg)


def _generate_random_walk(cfg: DataConfig) -> Dataset:
    """
    Dynamic-theta generation: random walk ability.

    Each learner i has:
      theta_0[i] ~ N(0, 1)   (initial ability)
      theta_t[i] = theta_0[i] + sum_{s=1..t} eps_s,  eps_s ~ N(0, drift_sigma)

    Learner answers B items in a random order.  At timestep t the response is
    sampled from GPCM(theta_t[i], item[t]).  The TRUE theta at each step is
    stored in base_seqs[i]["theta_true"] (length B, aligned with item_order).

    For extension items (anchor learners), we fix each learner's ability at
    their END-OF-B theta (theta_{B-1}[i]).  This is an approximation noted in
    the runner; the focus is trajectory tracking over B, not perfect E calibration.

    theta_traj shape: (N, B)  -- theta_traj[i, t] = true theta of learner i at step t.
    """
    rng = np.random.default_rng(cfg.seed)
    N, B, E, K = cfg.n_learners, cfg.n_base, cfg.n_ext, cfg.n_cats

    # --- item parameters (same as static) ---
    def sample_item_params(n_items):
        a = rng.lognormal(mean=0.0, sigma=0.3, size=n_items)
        b_raw = rng.standard_normal((n_items, K - 1))
        b = np.sort(b_raw, axis=1)
        return a, b

    a_base, b_base = sample_item_params(B)
    a_ext, b_ext_unshifted = sample_item_params(E)
    b_ext = b_ext_unshifted + cfg.ext_shift

    # --- initial abilities ---
    theta_0 = rng.standard_normal(N)

    # --- random walk trajectories over B steps ---
    # steps shape: (N, B) -- increments at each of the B interactions
    steps = rng.normal(0.0, cfg.drift_sigma, size=(N, B))
    # theta_traj[i, t] = theta_0[i] + sum(steps[i, 0:t])  (t=0 uses no steps yet)
    # i.e. ability BEFORE answering item t (the ability the learner brings to item t)
    cumsteps = np.cumsum(steps, axis=1)               # (N, B)  cumsteps[:,0] = steps[:,0]
    # shift right: theta at step 0 = theta_0 (no drift yet), theta at step t = theta_0 + cumsum[:t]
    theta_traj = np.zeros((N, B))
    theta_traj[:, 0] = theta_0
    theta_traj[:, 1:] = theta_0[:, None] + cumsteps[:, :-1]   # (N, B-1)

    # --- anchor mask ---
    anchor_mask = np.zeros(N, dtype=bool)
    anchor_idx = rng.choice(N, size=N // 2, replace=False)
    anchor_mask[anchor_idx] = True

    # --- base sequences: item order shuffled, response drawn from per-step theta ---
    base_seqs = []
    for i in range(N):
        item_order = rng.permutation(B).tolist()
        responses = []
        theta_true = []
        for t, j in enumerate(item_order):
            th = theta_traj[i, t]
            resp = _sample_response(th, a_base[j], b_base[j], rng)
            responses.append(resp)
            theta_true.append(float(th))
        base_seqs.append({
            "items": item_order,
            "responses": responses,
            "theta_true": theta_true,  # per-interaction true theta (aligned with item_order)
        })

    # --- extension sequences: anchor learners only, fixed at end-of-B theta ---
    ext_seqs = []
    for i in np.where(anchor_mask)[0]:
        theta_end = float(theta_traj[i, -1])   # end-of-B ability
        item_order = rng.permutation(E).tolist()
        responses = [_sample_response(theta_end, a_ext[j], b_ext[j], rng)
                     for j in item_order]
        ext_seqs.append({
            "items": item_order,
            "responses": responses,
            "learner_idx": int(i),
            "theta_anchor": theta_end,   # the fixed theta used for E responses
        })

    gt = GroundTruth(
        theta=theta_0,
        a_base=a_base, b_base=b_base,
        a_ext=a_ext, b_ext=b_ext,
        theta_traj=theta_traj,
    )

    return Dataset(base_seqs=base_seqs, ext_seqs=ext_seqs,
                   anchor_mask=anchor_mask, gt=gt, cfg=cfg)
