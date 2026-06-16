"""
Ground-truth data generation for the Bradley-Terry / GPCM bridge demo.

Sign convention (fixed throughout):
    s_i is item "strength" on a single latent scale.
    Higher s_i means the item is harder / stronger:
        - In pairwise (BT): item i beats item j with prob sigmoid(s_i - s_j).
        - In GPCM: higher s_i acts as higher difficulty (more negative
          effective difficulty), so respondents with lower theta answer
          in lower categories more often.

Both recovery methods should return values that are monotonically
increasing with s_i, so Spearman / Pearson between them should be high.
"""

import torch
import torch.nn.functional as F


def generate_ground_truth(
    n_items: int,
    seed: int = 0,
) -> dict:
    """
    Sample ground-truth item parameters.

    Returns
    -------
    dict with keys:
        s       : (n_items,)  latent strength (N(0,1))
        a       : (n_items,)  discrimination  (lognormal(0, 0.3))
        betas   : (n_items, K-1)  GPCM step thresholds
                  centered on the item's difficulty = -s_i so that
                  higher s -> more negative intercepts -> harder item.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_items = int(n_items)
    K = 4  # ordinal categories 0..3, so K-1=3 thresholds

    # Latent strength: s_i ~ N(0,1).  Higher = harder/stronger.
    s = torch.randn(n_items, generator=gen)

    # Discrimination: a_i ~ lognormal(0, 0.3)
    log_a = 0.3 * torch.randn(n_items, generator=gen)
    a = torch.exp(log_a)

    # Step thresholds for GPCM.
    # Difficulty is tied to s:  d_i = s_i  (so higher s -> harder).
    # Three thresholds spaced evenly around d_i:
    #   beta_{i,k} = d_i + offset_k,  offsets = [-1, 0, +1]
    # This ensures the item difficulty centroid equals s_i.
    offsets = torch.tensor([-1.0, 0.0, 1.0])          # (K-1,)
    betas = s.unsqueeze(1) + offsets.unsqueeze(0)      # (n_items, K-1)

    return {"s": s, "a": a, "betas": betas, "K": K, "n_items": n_items}


def gpcm_probs(theta: torch.Tensor, a: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """
    Compute GPCM category probabilities.

    Parameters
    ----------
    theta : (B,)       respondent ability
    a     : (B,)       item discrimination
    betas : (B, K-1)   item step thresholds

    All tensors are already aligned (same batch B = respondents x items).

    Returns
    -------
    probs : (B, K)  category probabilities
    """
    K = betas.shape[-1] + 1  # number of categories

    # Cumulative logit for category k:  z_k = k*a*(theta - beta_k)
    # Standard GPCM: unnormalized score for category k =
    #   sum_{m=1}^{k} a*(theta - beta_m), with sum_{m=1}^{0} = 0.
    # numerator_k = exp( sum_{m=1}^{k} a*(theta - beta_m) )

    # Shape: (B, K-1)
    step_logits = a.unsqueeze(-1) * (theta.unsqueeze(-1) - betas)  # (B, K-1)

    # Cumulative sum over thresholds to get log-numerator for k=1..K-1
    cumsum = torch.cumsum(step_logits, dim=-1)  # (B, K-1)

    # log-numerator for all K categories: cat([0, cumsum], dim=-1)
    log_num = torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum], dim=-1)  # (B, K)

    # log-sum-exp normalisation
    probs = F.softmax(log_num, dim=-1)  # (B, K)
    return probs


def generate_mode_a(
    ground_truth: dict,
    n_respondents: int = 2000,
    items_per_respondent: int = 40,
    seed: int = 0,
) -> dict:
    """
    Mode A: direct ordinal responses (human-style GPCM).

    Each respondent answers a random subset of items.

    Returns
    -------
    dict with keys:
        respondent_ids : (N_obs,) int
        item_ids       : (N_obs,) int
        responses      : (N_obs,) int in {0, ..., K-1}
        theta_true     : (n_respondents,) true ability
    """
    gen = torch.Generator()
    gen.manual_seed(seed + 100)

    s = ground_truth["s"]
    a = ground_truth["a"]
    betas = ground_truth["betas"]
    K = ground_truth["K"]
    n_items = ground_truth["n_items"]

    # Sample respondent abilities
    theta_true = torch.randn(n_respondents, generator=gen)

    respondent_ids_list = []
    item_ids_list = []
    responses_list = []

    for p in range(n_respondents):
        # Random subset of items (without replacement)
        perm = torch.randperm(n_items, generator=gen)[:items_per_respondent]
        item_subset = perm  # (items_per_respondent,)

        theta_p = theta_true[p].expand(items_per_respondent)     # (m,)
        a_p = a[item_subset]                                       # (m,)
        betas_p = betas[item_subset]                               # (m, K-1)

        probs = gpcm_probs(theta_p, a_p, betas_p)                 # (m, K)

        # Sample responses
        resp = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)  # (m,)

        respondent_ids_list.append(torch.full((items_per_respondent,), p, dtype=torch.long))
        item_ids_list.append(item_subset)
        responses_list.append(resp)

    return {
        "respondent_ids": torch.cat(respondent_ids_list),
        "item_ids": torch.cat(item_ids_list),
        "responses": torch.cat(responses_list),
        "theta_true": theta_true,
        "n_respondents": n_respondents,
    }


def generate_mode_b(
    ground_truth: dict,
    comparisons_per_item: int = 50,
    seed: int = 0,
) -> dict:
    """
    Mode B: pairwise comparisons (judge-style Bradley-Terry).

    Generates enough random pairs so each item appears in ~comparisons_per_item
    comparisons. Outcome: item i beats item j with prob sigmoid(s_i - s_j).

    Returns
    -------
    dict with keys:
        item_i   : (N_pairs,) int
        item_j   : (N_pairs,) int
        outcome  : (N_pairs,) int  -- 1 if i beats j, 0 otherwise
    """
    gen = torch.Generator()
    gen.manual_seed(seed + 200)

    s = ground_truth["s"]
    n_items = ground_truth["n_items"]

    # Total pairs needed so each item appears ~comparisons_per_item times.
    # Each pair involves 2 items, so total_pairs ~ n_items * comparisons_per_item / 2.
    n_pairs = int(n_items * comparisons_per_item / 2)

    item_i_list = []
    item_j_list = []
    outcome_list = []

    pairs_generated = 0
    while pairs_generated < n_pairs:
        # Sample a batch of random pairs (i != j)
        batch = min(1000, n_pairs - pairs_generated)
        i_idx = torch.randint(0, n_items, (batch,), generator=gen)
        j_idx = torch.randint(0, n_items, (batch,), generator=gen)

        # Filter out self-comparisons
        valid = i_idx != j_idx
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]

        if len(i_idx) == 0:
            continue

        # BT probability: sigmoid(s_i - s_j)
        delta = s[i_idx] - s[j_idx]
        prob_i_wins = torch.sigmoid(delta)
        outcome = torch.bernoulli(prob_i_wins, generator=gen).long()

        item_i_list.append(i_idx)
        item_j_list.append(j_idx)
        outcome_list.append(outcome)
        pairs_generated += len(i_idx)

    return {
        "item_i": torch.cat(item_i_list)[:n_pairs],
        "item_j": torch.cat(item_j_list)[:n_pairs],
        "outcome": torch.cat(outcome_list)[:n_pairs],
        "n_pairs": n_pairs,
    }
