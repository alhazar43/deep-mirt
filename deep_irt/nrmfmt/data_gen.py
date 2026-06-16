"""
data_gen.py -- Synthetic data for the binary-vs-NRM cross-format experiment.

Design (mirrors deep_irt/jointfmt/data_gen.py)
-----------------------------------------------
N items, each with a true latent difficulty s_i ~ N(0,1).  Every item is a
K-option multiple-choice item with one designated-correct option.  The
correct-option NRM location is set to s_i, so s_i is SIMULTANEOUSLY the 2PL
difficulty and the NRM correct-option location -- one shared scale.

Items are partitioned into three groups:
  - BINARY-ONLY : observed only as 2PL right/wrong responses
  - NRM-ONLY    : observed only as NRM chosen-option responses
  - BOTH        : observed via both formats (the bridge)

Mode A (binary 2PL): respondents answer BINARY-ONLY + BOTH items, observed as
    correct/incorrect.  P(correct) = sigmoid(a_i * (theta - s_i)).
Mode B (NRM): respondents answer NRM-ONLY + BOTH items, observed as the chosen
    option 0..K-1 from softmax_k(alpha_k * theta + gamma_k), where the correct
    option's location equals s_i.

Sign convention: higher s_i -> harder item (lower P(correct), correct option
needs higher theta to overtake).  Consistent across both heads.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def generate_ground_truth(
    n_items: int = 300,
    n_binary_only: int = 100,
    n_nrm_only: int = 100,
    K: int = 4,
    correct_option: int = 0,
    seed: int = 0,
) -> dict:
    """
    Sample ground-truth item params + partition groups.

    Returns
    -------
    dict with:
        s              : (n_items,)       true latent difficulty (shared scale)
        a2pl           : (n_items,)       2PL discrimination
        nrm_a          : (n_items, K)     NRM per-option slopes (centered)
        nrm_c          : (n_items, K)     NRM per-option intercepts (centered)
        K, correct_option, n_items
        binary_mask    : (n_items,) bool  BINARY-ONLY or BOTH
        nrm_mask       : (n_items,) bool  NRM-ONLY or BOTH
        binary_only_idx, nrm_only_idx, both_idx : LongTensors (sorted)
        n_binary_only, n_nrm_only, n_both
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_items = int(n_items)
    n_both = n_items - n_binary_only - n_nrm_only
    assert n_both > 0, "Need at least some items in BOTH partition"

    # Shared latent difficulty
    s = torch.randn(n_items, generator=gen)

    # 2PL discrimination: lognormal(0, 0.3), positive
    a2pl = torch.exp(0.3 * torch.randn(n_items, generator=gen))

    # NRM per-option params.  Correct option: slope alpha_corr > 0, intercept
    # gamma_corr = -alpha_corr * s_i  so its location -gamma/alpha = s_i.
    alpha_corr = torch.exp(0.3 * torch.randn(n_items, generator=gen))   # >0
    gamma_corr = -alpha_corr * s

    nrm_a = 0.6 * torch.randn(n_items, K, generator=gen)
    nrm_c = 0.6 * torch.randn(n_items, K, generator=gen)
    nrm_a[:, correct_option] = alpha_corr
    nrm_c[:, correct_option] = gamma_corr

    # Center across options (Bock sum-to-zero); recompute s from centered
    # correct params so the planted location survives centering exactly.
    nrm_a = nrm_a - nrm_a.mean(dim=1, keepdim=True)
    nrm_c = nrm_c - nrm_c.mean(dim=1, keepdim=True)
    s_centered = -nrm_c[:, correct_option] / nrm_a[:, correct_option]
    s = s_centered  # shared scale == centered NRM correct-option location

    # Partition
    perm = torch.randperm(n_items, generator=gen)
    binary_only_idx = perm[:n_binary_only]
    nrm_only_idx = perm[n_binary_only : n_binary_only + n_nrm_only]
    both_idx = perm[n_binary_only + n_nrm_only:]

    binary_mask = torch.zeros(n_items, dtype=torch.bool)
    binary_mask[binary_only_idx] = True
    binary_mask[both_idx] = True

    nrm_mask = torch.zeros(n_items, dtype=torch.bool)
    nrm_mask[nrm_only_idx] = True
    nrm_mask[both_idx] = True

    return {
        "s": s,
        "a2pl": a2pl,
        "nrm_a": nrm_a,
        "nrm_c": nrm_c,
        "K": K,
        "correct_option": correct_option,
        "n_items": n_items,
        "binary_mask": binary_mask,
        "nrm_mask": nrm_mask,
        "binary_only_idx": binary_only_idx.sort().values,
        "nrm_only_idx": nrm_only_idx.sort().values,
        "both_idx": both_idx.sort().values,
        "n_binary_only": n_binary_only,
        "n_nrm_only": n_nrm_only,
        "n_both": n_both,
    }


# ---------------------------------------------------------------------------
# Mode A: binary 2PL responses
# ---------------------------------------------------------------------------

def generate_mode_a(
    ground_truth: dict,
    n_respondents: int = 2000,
    items_per_respondent: int = 40,
    seed: int = 0,
) -> dict:
    """Binary correct/incorrect for items in binary_mask (BINARY-ONLY + BOTH)."""
    gen = torch.Generator()
    gen.manual_seed(seed + 100)

    s = ground_truth["s"]
    a = ground_truth["a2pl"]

    eligible = torch.where(ground_truth["binary_mask"])[0]
    n_eligible = len(eligible)
    theta_true = torch.randn(n_respondents, generator=gen)
    per_resp = min(items_per_respondent, n_eligible)

    resp_ids, item_ids, responses = [], [], []
    for p in range(n_respondents):
        sub = torch.randperm(n_eligible, generator=gen)[:per_resp]
        g = eligible[sub]
        logit = a[g] * (theta_true[p] - s[g])
        correct = torch.bernoulli(torch.sigmoid(logit), generator=gen).long()
        resp_ids.append(torch.full((per_resp,), p, dtype=torch.long))
        item_ids.append(g)
        responses.append(correct)

    return {
        "respondent_ids": torch.cat(resp_ids),
        "item_ids": torch.cat(item_ids),
        "responses": torch.cat(responses),
        "theta_true": theta_true,
        "n_respondents": n_respondents,
        "eligible_items": eligible,
    }


# ---------------------------------------------------------------------------
# Mode B: NRM chosen-option responses
# ---------------------------------------------------------------------------

def generate_mode_b(
    ground_truth: dict,
    n_respondents: int = 2000,
    items_per_respondent: int = 40,
    seed: int = 0,
) -> dict:
    """Chosen option 0..K-1 for items in nrm_mask (NRM-ONLY + BOTH)."""
    gen = torch.Generator()
    gen.manual_seed(seed + 200)

    nrm_a = ground_truth["nrm_a"]
    nrm_c = ground_truth["nrm_c"]

    eligible = torch.where(ground_truth["nrm_mask"])[0]
    n_eligible = len(eligible)
    theta_true = torch.randn(n_respondents, generator=gen)
    per_resp = min(items_per_respondent, n_eligible)

    resp_ids, item_ids, responses = [], [], []
    for p in range(n_respondents):
        sub = torch.randperm(n_eligible, generator=gen)[:per_resp]
        g = eligible[sub]
        logits = nrm_a[g] * theta_true[p] + nrm_c[g]      # (per_resp, K)
        probs = F.softmax(logits, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
        resp_ids.append(torch.full((per_resp,), p, dtype=torch.long))
        item_ids.append(g)
        responses.append(choice)

    return {
        "respondent_ids": torch.cat(resp_ids),
        "item_ids": torch.cat(item_ids),
        "responses": torch.cat(responses),
        "theta_true": theta_true,
        "n_respondents": n_respondents,
        "eligible_items": eligible,
    }
