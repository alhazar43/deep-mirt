"""data_gen.py -- parametric learning-curve synthetic data for E0.

Each respondent has a true ability trajectory

    theta(t) = theta_inf - (theta_inf - theta_0) exp(-r t),   t = 0..T-1,

with per-respondent initial ability theta_0, asymptote theta_inf, and rate
r. The rate is the estimand. Items carry fixed IRT parameters; responses
at each step are sampled from the GPCM (binary is K=2) under the moving
theta_t. This mirrors dynjoint/data_gen.py but swaps the random walk for a
parametric curve with a known rate, and reuses its gpcm_probs helper.
"""

import math

import torch

from deep_irt.dynjoint.data_gen import gpcm_probs


def sample_item_params(n_items: int, K: int, seed: int = 0) -> dict:
    """Sample fixed item parameters.

    s ~ N(0,1) sets difficulty; a = exp(0.3 z) is the discrimination; the
    K-1 step thresholds are s plus evenly spaced offsets centered at 0
    (a single 0 offset at K=2, recovering the 2PL).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    s = torch.randn(n_items, generator=gen)
    a = torch.exp(0.3 * torch.randn(n_items, generator=gen))
    if K == 2:
        offsets = torch.tensor([0.0])
    else:
        offsets = torch.linspace(-1.0, 1.0, K - 1)
    betas = s.unsqueeze(1) + offsets.unsqueeze(0)          # (n_items, K-1)
    return {"s": s, "a": a, "betas": betas, "K": K, "n_items": n_items}


def generate_learning_curves(
    n_items: int = 120,
    n_respondents: int = 300,
    seq_len: int = 40,
    K: int = 4,
    rate_lo: float = 0.04,
    rate_hi: float = 0.6,
    theta0_mean: float = -0.8,
    theta0_sd: float = 0.6,
    gain_lo: float = 0.8,
    gain_hi: float = 2.5,
    seed: int = 0,
) -> dict:
    """Generate responses under per-respondent parametric learning curves.

    Per respondent: theta_0 ~ N(theta0_mean, theta0_sd), a positive gain
    ~ U(gain_lo, gain_hi) so theta_inf = theta_0 + gain (genuine learning),
    and a log-uniform rate r ~ logU(rate_lo, rate_hi) so the curvature
    varies across respondents (the spread the recovery is scored on). Each
    respondent answers seq_len items drawn uniformly without replacement.

    Returns a dict with seq_item_ids (N,T), seq_responses (N,T),
    theta_true_traj (N,T), rate_true (N,), theta0_true (N,),
    thetainf_true (N,), and the item params s/a/betas.
    """
    ip = sample_item_params(n_items, K, seed)
    s, a, betas = ip["s"], ip["a"], ip["betas"]

    gen = torch.Generator()
    gen.manual_seed(seed + 100)
    per_resp = min(seq_len, n_items)

    theta0 = theta0_mean + theta0_sd * torch.randn(n_respondents, generator=gen)
    gain = gain_lo + (gain_hi - gain_lo) * torch.rand(n_respondents, generator=gen)
    thetainf = theta0 + gain
    log_lo, log_hi = math.log(rate_lo), math.log(rate_hi)
    rate = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(n_respondents, generator=gen))

    t = torch.arange(per_resp, dtype=torch.float32)                 # (T,)
    decay = torch.exp(-rate.unsqueeze(1) * t.unsqueeze(0))          # (N, T)
    theta_traj = thetainf.unsqueeze(1) - (thetainf - theta0).unsqueeze(1) * decay

    seq_items = torch.zeros(n_respondents, per_resp, dtype=torch.long)
    seq_resp = torch.zeros(n_respondents, per_resp, dtype=torch.long)
    for p in range(n_respondents):
        perm = torch.randperm(n_items, generator=gen)[:per_resp]
        seq_items[p] = perm
        probs = gpcm_probs(theta_traj[p], a[perm], betas[perm])     # (T, K)
        seq_resp[p] = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

    return {
        "seq_item_ids": seq_items,
        "seq_responses": seq_resp,
        "theta_true_traj": theta_traj,
        "rate_true": rate,
        "theta0_true": theta0,
        "thetainf_true": thetainf,
        "s": s,
        "a": a,
        "betas": betas,
        "K": K,
        "n_items": n_items,
        "seq_len": per_resp,
        "n_respondents": n_respondents,
    }
