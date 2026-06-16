"""
data_gen.py -- Synthetic data generation for the dynamic-encoder joint experiment.

Mirrors jointfmt/data_gen.py but adds DYNAMIC theta: each learner's ability
follows a random walk over their answer sequence (drift_sigma ~ 0.2).

Design
------
N items, each with a true latent difficulty s_i ~ N(0,1).
Partitioned identically to jointfmt: DIRECT-ONLY / PAIRWISE-ONLY / BOTH.

Persons:
  - Each has a TRUE TRAJECTORY theta_true[person, t] (random walk).
  - GPCM responses are generated under the MOVING theta_t at step t.
  - BT pairwise comparisons are person-free: P(i beats j) = sigmoid(s_i - s_j).

Dynamic ability model:
  theta[p, 0] ~ N(0, 1)
  theta[p, t] = theta[p, t-1] + eps,  eps ~ N(0, drift_sigma^2)

This makes the true signal HARDER to recover than the static case in jointfmt
(the encoder must track a moving target, not a fixed scalar).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ground truth (identical to jointfmt)
# ---------------------------------------------------------------------------

def generate_ground_truth(
    n_items: int = 300,
    n_direct_only: int = 100,
    n_pairwise_only: int = 100,
    K: int = 4,
    seed: int = 0,
) -> dict:
    """
    Sample ground-truth item parameters and assign partition groups.
    Identical to jointfmt/data_gen.generate_ground_truth().
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_items = int(n_items)
    n_both = n_items - n_direct_only - n_pairwise_only
    assert n_both > 0, "Need at least some items in BOTH partition"

    s = torch.randn(n_items, generator=gen)
    log_a = 0.3 * torch.randn(n_items, generator=gen)
    a = torch.exp(log_a)
    offsets = torch.tensor([-1.0, 0.0, 1.0])
    betas = s.unsqueeze(1) + offsets.unsqueeze(0)  # (n_items, K-1)

    perm = torch.randperm(n_items, generator=gen)
    direct_only_idx = perm[:n_direct_only]
    pairwise_only_idx = perm[n_direct_only : n_direct_only + n_pairwise_only]
    both_idx = perm[n_direct_only + n_pairwise_only:]

    direct_mask = torch.zeros(n_items, dtype=torch.bool)
    direct_mask[direct_only_idx] = True
    direct_mask[both_idx] = True

    pairwise_mask = torch.zeros(n_items, dtype=torch.bool)
    pairwise_mask[pairwise_only_idx] = True
    pairwise_mask[both_idx] = True

    return {
        "s": s,
        "a": a,
        "betas": betas,
        "K": K,
        "n_items": n_items,
        "direct_mask": direct_mask,
        "pairwise_mask": pairwise_mask,
        "direct_only_idx": direct_only_idx.sort().values,
        "pairwise_only_idx": pairwise_only_idx.sort().values,
        "both_idx": both_idx.sort().values,
        "n_direct_only": n_direct_only,
        "n_pairwise_only": n_pairwise_only,
        "n_both": n_both,
    }


# ---------------------------------------------------------------------------
# GPCM probability helper
# ---------------------------------------------------------------------------

def gpcm_probs(
    theta: torch.Tensor,
    a: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """GPCM category probabilities. Identical to jointfmt."""
    step_logits = a.unsqueeze(-1) * (theta.unsqueeze(-1) - betas)
    cumsum = torch.cumsum(step_logits, dim=-1)
    log_num = torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum], dim=-1)
    return F.softmax(log_num, dim=-1)


# ---------------------------------------------------------------------------
# Mode A: dynamic GPCM sequences
# ---------------------------------------------------------------------------

def generate_mode_a_dynamic(
    ground_truth: dict,
    n_respondents: int = 500,
    seq_len: int = 40,
    drift_sigma: float = 0.2,
    seed: int = 0,
) -> dict:
    """
    Generate GPCM ordinal responses for items in direct_mask (DIRECT-ONLY + BOTH)
    under a MOVING theta trajectory.

    Each learner answers exactly seq_len items drawn uniformly from the eligible
    pool.  Their ability follows a random walk.

    Returns
    -------
    dict with:
        seq_item_ids   : (n_respondents, seq_len)  LongTensor -- ordered sequences
        seq_responses  : (n_respondents, seq_len)  LongTensor -- ordered responses
        theta_true_traj: (n_respondents, seq_len)  -- true theta at each step
        theta_true_init: (n_respondents,)          -- initial ability theta[p, 0]
        theta_true_final:(n_respondents,)          -- final ability theta[p, -1]
        net_drift      : (n_respondents,)          -- theta_final - theta_init
        n_respondents  : int
        eligible_items : LongTensor
    """
    gen = torch.Generator()
    gen.manual_seed(seed + 100)

    s = ground_truth["s"]
    a = ground_truth["a"]
    betas = ground_truth["betas"]
    K = ground_truth["K"]

    eligible = torch.where(ground_truth["direct_mask"])[0]
    n_eligible = len(eligible)
    per_resp = min(seq_len, n_eligible)

    # Initial ability ~ N(0,1)
    theta_init = torch.randn(n_respondents, generator=gen)

    seq_item_ids_list = []
    seq_responses_list = []
    theta_traj_list = []

    for p in range(n_respondents):
        # Item order: random permutation of eligible items
        perm = torch.randperm(n_eligible, generator=gen)[:per_resp]
        item_global = eligible[perm]  # (per_resp,)

        # Build theta trajectory: random walk
        theta_p = torch.zeros(per_resp)
        theta_p[0] = theta_init[p]
        for t in range(1, per_resp):
            eps = drift_sigma * torch.randn(1, generator=gen).item()
            theta_p[t] = theta_p[t - 1] + eps

        # Generate GPCM responses under the moving theta_t
        a_p = a[item_global]       # (per_resp,)
        betas_p = betas[item_global]  # (per_resp, K-1)
        probs = gpcm_probs(theta_p, a_p, betas_p)  # (per_resp, K)
        resp = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

        seq_item_ids_list.append(item_global.unsqueeze(0))
        seq_responses_list.append(resp.unsqueeze(0))
        theta_traj_list.append(theta_p.unsqueeze(0))

    seq_item_ids = torch.cat(seq_item_ids_list, dim=0)    # (N, T)
    seq_responses = torch.cat(seq_responses_list, dim=0)  # (N, T)
    theta_traj = torch.cat(theta_traj_list, dim=0)        # (N, T)

    return {
        "seq_item_ids": seq_item_ids,
        "seq_responses": seq_responses,
        "theta_true_traj": theta_traj,
        "theta_true_init": theta_traj[:, 0],
        "theta_true_final": theta_traj[:, -1],
        "net_drift": theta_traj[:, -1] - theta_traj[:, 0],
        "n_respondents": n_respondents,
        "eligible_items": eligible,
        "seq_len": per_resp,
    }


# ---------------------------------------------------------------------------
# Mode B: pairwise comparisons (person-free, identical to jointfmt)
# ---------------------------------------------------------------------------

def generate_mode_b(
    ground_truth: dict,
    comparisons_per_item: int = 60,
    seed: int = 0,
) -> dict:
    """
    Generate BT pairwise comparisons for items in pairwise_mask.
    Identical to jointfmt/data_gen.generate_mode_b().
    """
    gen = torch.Generator()
    gen.manual_seed(seed + 200)

    s = ground_truth["s"]
    eligible = torch.where(ground_truth["pairwise_mask"])[0]
    n_eligible = len(eligible)
    n_pairs = int(n_eligible * comparisons_per_item / 2)

    item_i_list = []
    item_j_list = []
    outcome_list = []
    generated = 0

    while generated < n_pairs:
        batch = min(2000, n_pairs - generated)
        li = torch.randint(0, n_eligible, (batch,), generator=gen)
        lj = torch.randint(0, n_eligible, (batch,), generator=gen)
        valid = li != lj
        li, lj = li[valid], lj[valid]
        if len(li) == 0:
            continue

        gi = eligible[li]
        gj = eligible[lj]
        delta = s[gi] - s[gj]
        prob_i = torch.sigmoid(delta)
        outcome = torch.bernoulli(prob_i, generator=gen).long()

        item_i_list.append(gi)
        item_j_list.append(gj)
        outcome_list.append(outcome)
        generated += len(gi)

    return {
        "item_i": torch.cat(item_i_list)[:n_pairs],
        "item_j": torch.cat(item_j_list)[:n_pairs],
        "outcome": torch.cat(outcome_list)[:n_pairs],
        "n_pairs": n_pairs,
        "eligible_items": eligible,
    }
