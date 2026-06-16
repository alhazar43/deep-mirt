"""Tests for the three extended response-format decoders (decoders_ext.py).

CPU-only.  Each decoder gets:
  - structural unit tests (shapes, valid density/pmf, gradient flow,
    identifiability anchor where applicable), and
  - a small synthetic TRUE-MODEL recovery check: generate from the true model
    with known item params + person latent, fit through the decoder's own
    item_params / log_* code path, and measure Spearman recovery of the item
    parameter and the person latent.

GATE (per decoder): PASS if item-parameter recovery Spearman > 0.7 AND
person-latent recovery > 0.5.  The synthetic configs are deliberately small and
fast (N ~ 600 persons, ~60 items, a few hundred epochs, device='cpu').

Run:  python -m pytest deep_irt/tests/test_decoders_ext.py -q
Or:   python deep_irt/tests/test_decoders_ext.py   (prints a recovery table)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sp_stats

from deep_irt.core.decoders_ext import (
    LogNormalRTDecoder,
    BetaResponseDecoder,
    PoissonCountDecoder,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(x, y) -> float:
    r, _ = sp_stats.spearmanr(np.ravel(x), np.ravel(y))
    return float(r)


def _align(v, ref):
    """Flip sign of v if it is negatively correlated with the reference."""
    r, _ = sp_stats.pearsonr(np.ravel(v), np.ravel(ref))
    return -np.asarray(v) if r < 0 else np.asarray(v)


def _center(v):
    v = np.asarray(v, dtype=float)
    return v - v.mean()


class _ItemEmb(nn.Module):
    """One free embedding row per item -> a decoder maps it to item params.

    This routes the fit through the decoder's real item_params / log_* code so
    the recovery check exercises exactly the code that ships in decoders_ext.py.
    """

    def __init__(self, n_items: int, emb_dim: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Parameter(0.1 * torch.randn(n_items, emb_dim, generator=g))

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[item_ids]


def _flatten(theta_person, item_param_true, n_persons, n_items):
    """Build a fully-crossed (person x item) flat index set."""
    pid = torch.arange(n_persons).repeat_interleave(n_items)
    iid = torch.arange(n_items).repeat(n_persons)
    return pid, iid


# ---------------------------------------------------------------------------
# 1. Log-normal response-time decoder
# ---------------------------------------------------------------------------

def _gen_rt(n_persons, n_items, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    tau = torch.randn(n_persons, generator=g)                 # person speed
    beta = torch.randn(n_items, generator=g)                  # item time-intensity
    pid, iid = _flatten(tau, beta, n_persons, n_items)
    mean = tau[pid] - beta[iid]
    log_rt = mean + sigma * torch.randn(mean.shape, generator=g)
    return tau, beta, pid, iid, log_rt


def fit_rt(n_persons=600, n_items=60, sigma=0.4, emb_dim=4,
           n_epochs=400, lr=0.05, seed=0, verbose=False):
    torch.manual_seed(seed)
    tau, beta, pid, iid, log_rt = _gen_rt(n_persons, n_items, sigma, seed)

    dec = LogNormalRTDecoder(emb_dim=emb_dim, per_item_sigma=False).to(DEVICE)
    item_emb = _ItemEmb(n_items, emb_dim, seed=seed).to(DEVICE)
    tau_hat = nn.Parameter(torch.zeros(n_persons))
    opt = torch.optim.Adam(
        list(dec.parameters()) + list(item_emb.parameters()) + [tau_hat], lr=lr
    )

    for ep in range(n_epochs):
        opt.zero_grad()
        emb = item_emb(iid)
        loss = dec.nll(tau_hat[pid], emb, log_rt)
        # Anchor person speed to mean 0 (van der Linden location constraint).
        loss = loss + 1e-3 * tau_hat.mean().pow(2)
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            print(f"  [rt] ep {ep} nll {loss.item():.4f}")

    with torch.no_grad():
        beta_hat = dec.item_params(item_emb.emb)["beta"].squeeze(-1).cpu().numpy()
        tau_rec = tau_hat.detach().cpu().numpy()
    return {
        "beta_true": beta.numpy(), "beta_hat": beta_hat,
        "tau_true": tau.numpy(), "tau_hat": tau_rec,
    }


def recover_rt(seed=0):
    r = fit_rt(seed=seed)
    # beta enters as (tau - beta): recovered beta is anti-correlated with the
    # network's raw sign, _align handles the arbitrary sign of the linear map.
    beta_hat = _align(_center(r["beta_hat"]), _center(r["beta_true"]))
    tau_hat = _align(_center(r["tau_hat"]), _center(r["tau_true"]))
    return {
        "item_sp": _spearman(beta_hat, _center(r["beta_true"])),
        "person_sp": _spearman(tau_hat, _center(r["tau_true"])),
    }


def test_rt_log_density_shapes():
    dec = LogNormalRTDecoder(emb_dim=6)
    emb = torch.randn(5, 6)
    tau = torch.randn(5)
    log_rt = torch.randn(5)
    p = dec.item_params(emb)
    assert p["beta"].shape == (5, 1)
    ld = dec.log_density(tau, p["beta"], p["log_sigma"], log_rt)
    assert ld.shape == (5,)
    assert torch.isfinite(ld).all()


def test_rt_density_matches_normal():
    """Decoder log-density equals a hand-computed Normal log-density."""
    dec = LogNormalRTDecoder(emb_dim=4)
    emb = torch.randn(3, 4)
    tau = torch.randn(3)
    log_rt = torch.randn(3)
    p = dec.item_params(emb)
    ld = dec.log_density(tau, p["beta"], p["log_sigma"], log_rt)
    mean = tau - p["beta"].squeeze(-1)
    sigma = p["log_sigma"].squeeze(-1).exp()
    ref = torch.distributions.Normal(mean, sigma).log_prob(log_rt)
    assert torch.allclose(ld, ref, atol=1e-5)


def test_rt_gradient_flow():
    dec = LogNormalRTDecoder(emb_dim=4)
    emb = torch.randn(5, 4, requires_grad=True)
    tau = torch.randn(5)
    log_rt = torch.randn(5)
    loss = dec.nll(tau, emb, log_rt)
    loss.backward()
    assert torch.isfinite(emb.grad).all()
    assert dec.log_sigma.grad is not None


def test_rt_recovery():
    rec = recover_rt(seed=0)
    assert rec["item_sp"] > 0.7, f"RT time-intensity recovery weak: {rec}"
    assert rec["person_sp"] > 0.5, f"RT person-speed recovery weak: {rec}"


# ---------------------------------------------------------------------------
# 2. Beta continuous-response decoder
# ---------------------------------------------------------------------------

def _gen_beta(n_persons, n_items, phi, seed):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n_persons, generator=g)
    a = torch.exp(0.3 * torch.randn(n_items, generator=g))    # >0
    b = torch.randn(n_items, generator=g)
    pid, iid = _flatten(theta, b, n_persons, n_items)
    mu = torch.sigmoid(a[iid] * (theta[pid] - b[iid])).clamp(1e-4, 1 - 1e-4)
    alpha = mu * phi
    beta_p = (1.0 - mu) * phi
    y = torch.distributions.Beta(alpha, beta_p).sample()
    y = y.clamp(1e-5, 1 - 1e-5)
    return theta, a, b, pid, iid, y


def fit_beta(n_persons=600, n_items=60, phi=12.0, emb_dim=4,
             n_epochs=500, lr=0.05, seed=0, verbose=False):
    torch.manual_seed(seed)
    theta, a, b, pid, iid, y = _gen_beta(n_persons, n_items, phi, seed)

    dec = BetaResponseDecoder(emb_dim=emb_dim, per_item_phi=False).to(DEVICE)
    item_emb = _ItemEmb(n_items, emb_dim, seed=seed).to(DEVICE)
    theta_hat = nn.Parameter(torch.zeros(n_persons))
    opt = torch.optim.Adam(
        list(dec.parameters()) + list(item_emb.parameters()) + [theta_hat], lr=lr
    )

    for ep in range(n_epochs):
        opt.zero_grad()
        emb = item_emb(iid)
        loss = dec.nll(theta_hat[pid], emb, y)
        # Soft N(0,1) anchor on theta (fixes 2PL location/scale).
        loss = loss + 1e-2 * (theta_hat.mean().pow(2)
                              + (theta_hat.std() - 1.0).pow(2))
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            print(f"  [beta] ep {ep} nll {loss.item():.4f}")

    with torch.no_grad():
        p = dec.item_params(item_emb.emb)
        b_hat = p["b"].squeeze(-1).cpu().numpy()
        theta_rec = theta_hat.detach().cpu().numpy()
    return {
        "b_true": b.numpy(), "b_hat": b_hat,
        "theta_true": theta.numpy(), "theta_hat": theta_rec,
    }


def recover_beta(seed=0):
    r = fit_beta(seed=seed)
    b_hat = _align(r["b_hat"], r["b_true"])
    theta_hat = _align(r["theta_hat"], r["theta_true"])
    return {
        "item_sp": _spearman(b_hat, r["b_true"]),
        "person_sp": _spearman(theta_hat, r["theta_true"]),
    }


def test_beta_density_valid():
    dec = BetaResponseDecoder(emb_dim=6)
    emb = torch.randn(5, 6)
    theta = torch.randn(5)
    y = torch.rand(5).clamp(1e-3, 1 - 1e-3)
    p = dec.item_params(emb)
    ld = dec.log_density(theta, p["a"], p["b"], p["phi"], y)
    assert ld.shape == (5,)
    assert torch.isfinite(ld).all()


def test_beta_density_matches_torch_beta():
    """Decoder log-density equals torch.distributions.Beta log_prob."""
    dec = BetaResponseDecoder(emb_dim=4)
    emb = torch.randn(4, 4)
    theta = torch.randn(4)
    y = torch.rand(4).clamp(1e-3, 1 - 1e-3)
    p = dec.item_params(emb)
    ld = dec.log_density(theta, p["a"], p["b"], p["phi"], y)
    mu = torch.sigmoid(p["a"] * (theta.unsqueeze(1) - p["b"])).squeeze(1)
    mu = mu.clamp(1e-6, 1 - 1e-6)
    phi = p["phi"].squeeze(1)
    ref = torch.distributions.Beta(mu * phi, (1 - mu) * phi).log_prob(y)
    assert torch.allclose(ld, ref, atol=1e-4)


def test_beta_positive_discrimination():
    dec = BetaResponseDecoder(emb_dim=5)
    emb = torch.randn(10, 5)
    a = dec.item_params(emb)["a"]
    assert (a > 0).all()


def test_beta_gradient_flow():
    dec = BetaResponseDecoder(emb_dim=4)
    emb = torch.randn(6, 4, requires_grad=True)
    theta = torch.randn(6)
    y = torch.rand(6).clamp(1e-3, 1 - 1e-3)
    loss = dec.nll(theta, emb, y)
    loss.backward()
    assert torch.isfinite(emb.grad).all()


def test_beta_recovery():
    rec = recover_beta(seed=0)
    assert rec["item_sp"] > 0.7, f"Beta difficulty recovery weak: {rec}"
    assert rec["person_sp"] > 0.5, f"Beta ability recovery weak: {rec}"


# ---------------------------------------------------------------------------
# 3. Poisson count decoder
# ---------------------------------------------------------------------------

def _gen_poisson(n_persons, n_items, seed):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n_persons, generator=g)
    a = torch.exp(0.2 * torch.randn(n_items, generator=g))    # >0, mild spread
    b = torch.randn(n_items, generator=g)
    pid, iid = _flatten(theta, b, n_persons, n_items)
    log_lambda = a[iid] * (theta[pid] - b[iid])
    # Clamp the log-rate so counts stay in a sane range for a fast CPU fit.
    log_lambda = log_lambda.clamp(-4.0, 3.0)
    lam = log_lambda.exp()
    y = torch.poisson(lam, generator=g)
    return theta, a, b, pid, iid, y


def fit_poisson(n_persons=600, n_items=60, emb_dim=4,
                n_epochs=500, lr=0.05, seed=0, verbose=False):
    torch.manual_seed(seed)
    theta, a, b, pid, iid, y = _gen_poisson(n_persons, n_items, seed)

    dec = PoissonCountDecoder(emb_dim=emb_dim, discrimination=True).to(DEVICE)
    item_emb = _ItemEmb(n_items, emb_dim, seed=seed).to(DEVICE)
    theta_hat = nn.Parameter(torch.zeros(n_persons))
    opt = torch.optim.Adam(
        list(dec.parameters()) + list(item_emb.parameters()) + [theta_hat], lr=lr
    )

    for ep in range(n_epochs):
        opt.zero_grad()
        emb = item_emb(iid)
        loss = dec.nll(theta_hat[pid], emb, y)
        loss = loss + 1e-2 * (theta_hat.mean().pow(2)
                              + (theta_hat.std() - 1.0).pow(2))
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            print(f"  [pois] ep {ep} nll {loss.item():.4f}")

    with torch.no_grad():
        b_hat = dec.item_params(item_emb.emb)["b"].squeeze(-1).cpu().numpy()
        theta_rec = theta_hat.detach().cpu().numpy()
    return {
        "b_true": b.numpy(), "b_hat": b_hat,
        "theta_true": theta.numpy(), "theta_hat": theta_rec,
    }


def recover_poisson(seed=0):
    r = fit_poisson(seed=seed)
    b_hat = _align(r["b_hat"], r["b_true"])
    theta_hat = _align(r["theta_hat"], r["theta_true"])
    return {
        "item_sp": _spearman(b_hat, r["b_true"]),
        "person_sp": _spearman(theta_hat, r["theta_true"]),
    }


def test_poisson_logpmf_valid():
    dec = PoissonCountDecoder(emb_dim=6, discrimination=True)
    emb = torch.randn(5, 6)
    theta = torch.randn(5)
    y = torch.randint(0, 5, (5,)).float()
    p = dec.item_params(emb)
    lp = dec.log_probs(theta, p["b"], y, a=p["a"])
    assert lp.shape == (5,)
    assert torch.isfinite(lp).all()


def test_poisson_logpmf_matches_torch():
    """Decoder log-pmf equals torch.distributions.Poisson log_prob."""
    dec = PoissonCountDecoder(emb_dim=4, discrimination=True)
    emb = torch.randn(4, 4)
    theta = torch.randn(4)
    y = torch.randint(0, 6, (4,)).float()
    p = dec.item_params(emb)
    lp = dec.log_probs(theta, p["b"], y, a=p["a"])
    log_lambda = (p["a"] * (theta.unsqueeze(1) - p["b"])).squeeze(1)
    ref = torch.distributions.Poisson(log_lambda.exp()).log_prob(y)
    assert torch.allclose(lp, ref, atol=1e-4)


def test_poisson_no_discrimination_params():
    dec = PoissonCountDecoder(emb_dim=4, discrimination=False)
    emb = torch.randn(3, 4)
    p = dec.item_params(emb)
    assert "a" not in p and "b" in p


def test_poisson_gradient_flow():
    dec = PoissonCountDecoder(emb_dim=4, discrimination=True)
    emb = torch.randn(6, 4, requires_grad=True)
    theta = torch.randn(6)
    y = torch.randint(0, 5, (6,)).float()
    loss = dec.nll(theta, emb, y)
    loss.backward()
    assert torch.isfinite(emb.grad).all()


def test_poisson_recovery():
    rec = recover_poisson(seed=0)
    assert rec["item_sp"] > 0.7, f"Poisson difficulty recovery weak: {rec}"
    assert rec["person_sp"] > 0.5, f"Poisson ability recovery weak: {rec}"


# ---------------------------------------------------------------------------
# __main__: print a recovery table with explicit PASS/FAIL gate verdicts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print("Synthetic recovery (CPU-only, true-model gen -> fit -> Spearman)\n")
    rows = [
        ("LogNormalRT (priority)", recover_rt(seed=0), "time-intensity beta", "speed tau"),
        ("BetaResponse",          recover_beta(seed=0), "difficulty b",        "ability theta"),
        ("PoissonCount",          recover_poisson(seed=0), "difficulty b",     "ability theta"),
    ]
    print(f"{'decoder':<24}{'item':>8}{'person':>9}   gate (item>0.7 & person>0.5)")
    print("-" * 70)
    all_pass = True
    for name, rec, ip_name, pp_name in rows:
        item_sp, person_sp = rec["item_sp"], rec["person_sp"]
        ok = (item_sp > 0.7) and (person_sp > 0.5)
        all_pass = all_pass and ok
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<24}{item_sp:>8.3f}{person_sp:>9.3f}   {verdict}")
    print("-" * 70)
    print(f"item param = first label, person latent = second.")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAIL'}")
