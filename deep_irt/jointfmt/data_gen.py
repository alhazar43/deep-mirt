"""
data_gen.py -- Synthetic data generation for the joint-format transfer experiment.

Design
------
N items, each with a true latent difficulty s_i ~ N(0,1).

Items are partitioned into three groups:
  - DIRECT-ONLY  : observed only via GPCM ordinal responses
  - PAIRWISE-ONLY: observed only via BT pairwise comparisons
  - BOTH         : observed via both formats

Data generated:
  Mode A (GPCM): respondent population answers DIRECT-ONLY and BOTH items.
  Mode B (BT)  : pairwise comparisons over PAIRWISE-ONLY and BOTH items.

Sign convention (same as bt_demo):
  s_i is item difficulty/strength. Higher s_i -> harder item.
  In GPCM: higher s_i -> higher step thresholds -> lower expected response.
  In BT  : item i beats j with prob sigmoid(s_i - s_j).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ground truth
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

    Returns
    -------
    dict with:
        s               : (n_items,)       true latent difficulty
        a               : (n_items,)       GPCM discrimination
        betas           : (n_items, K-1)   GPCM step thresholds
        K               : int
        n_items         : int
        direct_mask     : (n_items,) bool  -- items in DIRECT-ONLY or BOTH
        pairwise_mask   : (n_items,) bool  -- items in PAIRWISE-ONLY or BOTH
        direct_only_idx : LongTensor       -- indices of DIRECT-ONLY items
        pairwise_only_idx: LongTensor      -- indices of PAIRWISE-ONLY items
        both_idx        : LongTensor       -- indices of BOTH items
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_items = int(n_items)
    n_both = n_items - n_direct_only - n_pairwise_only
    assert n_both > 0, "Need at least some items in BOTH partition"

    # True latent difficulty
    s = torch.randn(n_items, generator=gen)

    # Discrimination: lognormal(0, 0.3)
    log_a = 0.3 * torch.randn(n_items, generator=gen)
    a = torch.exp(log_a)

    # Step thresholds: centered on s_i with spacing [-1, 0, +1]
    offsets = torch.tensor([-1.0, 0.0, 1.0])           # (K-1,)
    betas = s.unsqueeze(1) + offsets.unsqueeze(0)       # (n_items, K-1)

    # Assign item indices to groups (first shuffle, then split)
    perm = torch.randperm(n_items, generator=gen)
    direct_only_idx = perm[:n_direct_only]
    pairwise_only_idx = perm[n_direct_only : n_direct_only + n_pairwise_only]
    both_idx = perm[n_direct_only + n_pairwise_only:]

    # Masks
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
    """
    Compute GPCM category probabilities.

    Parameters
    ----------
    theta : (B,)
    a     : (B,)
    betas : (B, K-1)

    Returns
    -------
    probs : (B, K)
    """
    step_logits = a.unsqueeze(-1) * (theta.unsqueeze(-1) - betas)  # (B, K-1)
    cumsum = torch.cumsum(step_logits, dim=-1)                       # (B, K-1)
    log_num = torch.cat(
        [torch.zeros_like(cumsum[..., :1]), cumsum], dim=-1
    )  # (B, K)
    return F.softmax(log_num, dim=-1)


# ---------------------------------------------------------------------------
# Mode A: direct ordinal responses (GPCM)
# ---------------------------------------------------------------------------

def generate_mode_a(
    ground_truth: dict,
    n_respondents: int = 2000,
    items_per_respondent: int = 40,
    seed: int = 0,
) -> dict:
    """
    Generate GPCM ordinal responses for items in direct_mask
    (DIRECT-ONLY + BOTH).

    Returns
    -------
    dict with:
        respondent_ids : (N_obs,) int
        item_ids       : (N_obs,) int   -- global item indices
        responses      : (N_obs,) int in {0, ..., K-1}
        theta_true     : (n_respondents,)
        n_respondents  : int
        eligible_items : LongTensor    -- which global item indices are in the pool
    """
    gen = torch.Generator()
    gen.manual_seed(seed + 100)

    s = ground_truth["s"]
    a = ground_truth["a"]
    betas = ground_truth["betas"]
    K = ground_truth["K"]

    # Pool: only items in direct_mask
    eligible = torch.where(ground_truth["direct_mask"])[0]  # global indices
    n_eligible = len(eligible)

    theta_true = torch.randn(n_respondents, generator=gen)

    respondent_ids_list = []
    item_ids_list = []
    responses_list = []

    per_resp = min(items_per_respondent, n_eligible)

    for p in range(n_respondents):
        perm = torch.randperm(n_eligible, generator=gen)[:per_resp]
        item_subset_local = perm                     # indices into eligible
        item_subset_global = eligible[item_subset_local]  # global item ids

        theta_p = theta_true[p].expand(per_resp)
        a_p = a[item_subset_global]
        betas_p = betas[item_subset_global]

        probs = gpcm_probs(theta_p, a_p, betas_p)
        resp = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

        respondent_ids_list.append(torch.full((per_resp,), p, dtype=torch.long))
        item_ids_list.append(item_subset_global)
        responses_list.append(resp)

    return {
        "respondent_ids": torch.cat(respondent_ids_list),
        "item_ids": torch.cat(item_ids_list),
        "responses": torch.cat(responses_list),
        "theta_true": theta_true,
        "n_respondents": n_respondents,
        "eligible_items": eligible,
    }


# ---------------------------------------------------------------------------
# Mode B: pairwise comparisons (BT)
# ---------------------------------------------------------------------------

def generate_mode_b(
    ground_truth: dict,
    comparisons_per_item: int = 60,
    seed: int = 0,
) -> dict:
    """
    Generate BT pairwise comparisons for items in pairwise_mask
    (PAIRWISE-ONLY + BOTH).

    Returns
    -------
    dict with:
        item_i    : (N_pairs,) int  -- global item indices
        item_j    : (N_pairs,) int
        outcome   : (N_pairs,) int  {0,1}
        n_pairs   : int
        eligible_items : LongTensor
    """
    gen = torch.Generator()
    gen.manual_seed(seed + 200)

    s = ground_truth["s"]

    eligible = torch.where(ground_truth["pairwise_mask"])[0]  # global indices
    n_eligible = len(eligible)

    n_pairs = int(n_eligible * comparisons_per_item / 2)

    item_i_list = []
    item_j_list = []
    outcome_list = []

    generated = 0
    while generated < n_pairs:
        batch = min(2000, n_pairs - generated)
        # Sample local indices, then map to global
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
