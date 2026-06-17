"""Spine analysis: the GPCM stiffness ratio E[I(theta)]/E[I(alpha)] vs K, overlaid
on the empirical dynamic-alpha benefit delta_alpha(K) from RQ1.

Fisher info of a categorical with logits psi_k(phi):
    I(phi) = Var_{k~P}( d psi_k / d phi )
For the GPCM, psi_k = a * sum_{c=1..k}(theta - b_{c-1}), psi_0 = 0, so
    d psi_k / d theta = a * k
    d psi_k / d a     = sum_{c=1..k}(theta - b_{c-1}) = psi_k / a
Averaged over theta ~ N(0,1) and over items drawn from the bench priors
(a ~ lognormal(0,0.3), b ~ sorted N(0,1)^{K-1}).

Sanity (printed): at K=2 this must reduce to the closed-form 2PL,
I(theta)=a^2 P(1-P), I(a)=(theta-b)^2 P(1-P).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"

KS = [2, 4, 6, 8, 11]
Q_ITEMS = 4000            # items per K for a stable expectation over the priors
SEED = 12345


def _fisher_at(a: float, b: np.ndarray, theta: float):
    """Return (I_theta, I_alpha) for one item at one ability."""
    K = len(b) + 1
    psi = np.zeros(K)
    for k in range(1, K):
        psi[k] = a * np.sum(theta - b[:k])
    psi -= psi.max()
    P = np.exp(psi)
    P /= P.sum()
    kk = np.arange(K, dtype=float)
    d_theta = a * kk                                   # d psi_k / d theta
    d_a = np.array([np.sum(theta - b[:k]) for k in range(K)])  # d psi_k / d a
    I_theta = float((P * d_theta**2).sum() - (P * d_theta).sum()**2)
    I_alpha = float((P * d_a**2).sum() - (P * d_a).sum()**2)
    return I_theta, I_alpha


def _expected_fisher(K: int, rng, theta_grid, theta_w):
    a = rng.lognormal(mean=0.0, sigma=0.3, size=Q_ITEMS)
    b = np.sort(rng.standard_normal((Q_ITEMS, K - 1)), axis=1)
    It = np.zeros(Q_ITEMS)
    Ia = np.zeros(Q_ITEMS)
    for j in range(Q_ITEMS):
        it = ia = 0.0
        for th, w in zip(theta_grid, theta_w):
            t, al = _fisher_at(a[j], b[j], th)
            it += w * t
            ia += w * al
        It[j] = it
        Ia[j] = ia
    return It.mean(), Ia.mean()


def _sanity_k2():
    """Closed-form 2PL cross-check at a few (a,b,theta)."""
    print("K=2 sanity (numeric vs closed form a^2 P(1-P), (theta-b)^2 P(1-P)):")
    for a, b0, th in [(1.0, 0.0, 0.5), (1.4, -0.3, 1.0), (0.7, 0.8, -0.4)]:
        It, Ia = _fisher_at(a, np.array([b0]), th)
        P = 1.0 / (1.0 + np.exp(-a * (th - b0)))
        It_cf = a**2 * P * (1 - P)
        Ia_cf = (th - b0)**2 * P * (1 - P)
        print(f"  a={a} b={b0} th={th}:  I_theta {It:.5f} vs {It_cf:.5f} | "
              f"I_alpha {Ia:.5f} vs {Ia_cf:.5f}")


def main():
    _sanity_k2()
    # Gauss-Hermite over theta ~ N(0,1): nodes x, weights w; integral of f(theta)
    # phi(theta) dtheta = sum w_i/sqrt(pi) f(sqrt(2) x_i).
    x, w = np.polynomial.hermite_e.hermegauss(40)     # probabilists' Hermite, N(0,1)
    theta_grid = x
    theta_w = w / w.sum()
    rng = np.random.default_rng(SEED)

    rows = []
    for K in KS:
        It, Ia = _expected_fisher(K, rng, theta_grid, theta_w)
        rows.append({"K": K, "I_theta": It, "I_alpha": Ia, "ratio": It / Ia})
        print(f"K={K:2d}  E[I_theta]={It:.4f}  E[I_alpha]={Ia:.4f}  "
              f"stiffness I_theta/I_alpha={It/Ia:.4f}")

    # empirical delta_alpha(K) from RQ1
    ab = json.loads((OUT / "alpha_beta_asymmetry.json").read_text())
    d_by_k = {s["K"]: s["delta_alpha"] for s in ab["summary"]}
    ratio = np.array([r["ratio"] for r in rows])
    dalpha = np.array([d_by_k[r["K"]] for r in rows])
    from scipy.stats import spearmanr, pearsonr
    rho_k = spearmanr([r["K"] for r in rows], dalpha).correlation
    rho_ratio = spearmanr(ratio, dalpha).correlation
    print(f"\nSpearman(delta_alpha, K)      = {rho_k:+.3f}")
    print(f"Spearman(delta_alpha, ratio)  = {rho_ratio:+.3f}")
    print(f"Pearson(delta_alpha, log ratio) = "
          f"{pearsonr(np.log(ratio), dalpha)[0]:+.3f}")

    (OUT / "fisher_ratio.json").write_text(json.dumps(
        {"rows": rows, "delta_alpha": d_by_k,
         "spearman_dalpha_K": rho_k, "spearman_dalpha_ratio": rho_ratio},
        indent=2))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ks = np.array([r["K"] for r in rows])
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    c1, c2 = "#1f77b4", "#d62728"
    ax1.plot(Ks, ratio, "o-", color=c1, lw=2, label="stiffness I(theta)/I(alpha)")
    ax1.set_xlabel("number of ordinal categories K")
    ax1.set_ylabel("GPCM stiffness  I(theta)/I(alpha)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2 = ax1.twinx()
    ax2.plot(Ks, dalpha, "s--", color=c2, lw=2,
             label="empirical benefit delta_alpha")
    ax2.axhline(0, color="0.7", lw=0.8, ls=":")
    ax2.set_ylabel("dynamic-alpha benefit  delta_alpha (Pearson r)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title(f"Discrimination benefit tracks Fisher stiffness\n"
                  f"Spearman(delta_alpha, stiffness) = {rho_ratio:+.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "fisher_ratio.png", dpi=150)
    print(f"wrote {OUT/'fisher_ratio.png'} and fisher_ratio.json")


if __name__ == "__main__":
    main()
