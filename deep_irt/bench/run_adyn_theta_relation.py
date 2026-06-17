"""Study the a_dynamic <-> theta relation on the NULL (static-alpha data).

a_dynamic is item-demeaned per-occurrence alpha, so its variation is WITHIN item,
across the different-ability learners who saw that item. Linear corr was ~0, but
alpha's Fisher term is (theta-beta)^2, so probe nonlinear and gap-driven shape and
ask how much of a_dynamic a flexible function of theta can actually explain. If the
null relation is genuine noise (R^2 ~ 0, flat bins) the detector is clean; if the
architecture imposes spurious theta-structure even here, that is a bias to flag.
Temp probe; delete after.
"""
from __future__ import annotations

import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import pearsonr

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine

K = 4
EPOCHS = 150
N = 2000
SEED = 0


@torch.no_grad()
def extract(model, items, resp, device):
    enc, dec = model.encoder, model.decoder
    it = torch.as_tensor(items, dtype=torch.long, device=device)
    rp = torch.as_tensor(resp, dtype=torch.long, device=device)
    theta, state = enc.aligned_theta_and_state(it, rp)
    Nn, T = it.shape
    H = state.shape[-1]
    val = enc.item_val_emb(it)
    key = enc.item_key_emb(it)
    p = dec.item_params(val.reshape(Nn * T, -1), state=state.reshape(Nn * T, H),
                        item_key=key.reshape(Nn * T, -1))
    a = p["alpha"].reshape(Nn, T).cpu().numpy()
    b = p["beta"].reshape(Nn, T, -1).cpu().numpy()
    th = theta.reshape(Nn, T).cpu().numpy()
    return a, b, th


def r2(y, X):
    Xc = np.hstack([np.ones((len(y), 1)), X])
    coef, *_ = la.lstsq(Xc, y, rcond=None)
    pred = Xc @ coef
    return 1.0 - np.var(y - pred) / np.var(y)


def main():
    ds = generate(BenchDataConfig(name="s", kind="static", n_cats=K,
                                  n_learners=N, n_items=60, seq_len=60, seed=SEED))
    eng = DeepIRTEngine(ds, decoder="gpcm", emb_dim=8, hidden_dim=32,
                        state_alpha=True, item_key_dim=64, alpha_log_scale=1.0,
                        encoder="lstm", device="cuda", seed=SEED)
    eng.fit(n_epochs=EPOCHS, batch_size=256)

    items = ds.items0
    a, b, th_m = extract(eng.model, items, ds.responses, "cuda")
    th_t = ds.theta_at_step.ravel().astype(np.float64)

    Q = ds.cfg.n_items
    fi = items.ravel()
    fa = a.ravel().astype(np.float64)
    cnt = np.zeros(Q); asum = np.zeros(Q)
    np.add.at(asum, fi, fa); np.add.at(cnt, fi, 1.0)
    a_static = asum / np.maximum(cnt, 1.0)
    a_dyn = fa - a_static[fi]

    beta_item = np.sort(b.reshape(-1, K - 1), axis=1).mean(axis=1)  # item difficulty
    th_mf = th_m.ravel().astype(np.float64)
    gap_t = th_t - beta_item        # ability - difficulty (true theta)
    gap_m = th_mf - beta_item

    def c(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return float(pearsonr(x[m], y[m])[0])

    print(f"\n=== a_dynamic <-> theta relation, static-alpha K={K}, N={N} ===")
    print(f"a_static recovery (vs true alpha): {c(a_static, ds.gt.a):.3f}")
    print(f"a_dynamic std (CV):                {np.std(a_dyn)/np.mean(np.abs(a_static)):.3f}")
    print("\nlinear:")
    print(f"  corr(a_dyn, theta_true)   = {c(a_dyn, th_t):+.3f}")
    print(f"  corr(a_dyn, theta_model)  = {c(a_dyn, th_mf):+.3f}")
    print("gap and nonlinear (true theta):")
    print(f"  corr(a_dyn, gap=th-beta)  = {c(a_dyn, gap_t):+.3f}")
    print(f"  corr(a_dyn, |gap|)        = {c(a_dyn, np.abs(gap_t)):+.3f}")
    print(f"  corr(a_dyn, gap^2)        = {c(a_dyn, gap_t**2):+.3f}")
    print(f"  corr(a_dyn, beta_item)    = {c(a_dyn, beta_item):+.3f}")
    print("variance of a_dyn explained (R^2):")
    print(f"  cubic(theta_true)         = {r2(a_dyn, np.vstack([th_t, th_t**2, th_t**3]).T):.4f}")
    print(f"  cubic(gap_true)           = {r2(a_dyn, np.vstack([gap_t, gap_t**2, gap_t**3]).T):.4f}")
    print(f"  cubic(theta_model)        = {r2(a_dyn, np.vstack([th_mf, th_mf**2, th_mf**3]).T):.4f}")
    print(f"  cubic(gap_model)          = {r2(a_dyn, np.vstack([gap_m, gap_m**2, gap_m**3]).T):.4f}")

    bins = np.quantile(th_mf, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(th_mf, bins[1:-1]), 0, 9)
    print("\na_dyn conditional mean by theta_model decile:")
    for d in range(10):
        m = idx == d
        print(f"  theta~{np.mean(th_mf[m]):+.2f}  a_dyn_mean={np.mean(a_dyn[m]):+.4f}  "
              f"a_dyn_std={np.std(a_dyn[m]):.3f}  n={m.sum()}")


if __name__ == "__main__":
    main()
