"""Phase 1 of the a_static / a_dynamic study.

On STATIC-alpha synthetic data the true discrimination is occurrence-invariant,
so the true a_dynamic is exactly zero.  The state-conditioned alpha head still
produces a per-occurrence alpha; decompose it as
    a_static(j) = mean_t alpha_jt   (the would-be instrument property)
    a_dynamic(j,t) = alpha_jt - a_static(j)
and sweep N.  If a_dynamic is finite-data SCAFFOLD it shrinks toward zero as N
grows; if the model imposes spurious dynamic structure it persists (a bias).
beta is item-key-only (no state) so it has no dynamic part by construction; theta
is purely dynamic.  Temp probe; delete after.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine

K = 4
EPOCHS = 150
SEEDS = (0, 1, 2)
NS = (200, 800, 3200)


@torch.no_grad()
def alpha_per_pos(model, items, resp, device):
    enc, dec = model.encoder, model.decoder
    it = torch.as_tensor(items, dtype=torch.long, device=device)
    rp = torch.as_tensor(resp, dtype=torch.long, device=device)
    theta, state = enc.aligned_theta_and_state(it, rp)        # (N,T), (N,T,H)
    N, T = it.shape
    H = state.shape[-1]
    val = enc.item_val_emb(it)
    key = enc.item_key_emb(it)
    p = dec.item_params(val.reshape(N * T, -1),
                        state=state.reshape(N * T, H),
                        item_key=key.reshape(N * T, -1))
    a = p["alpha"].reshape(N, T).cpu().numpy()
    th = theta.reshape(N, T).cpu().numpy()
    return a, th


def run(N, seed):
    ds = generate(BenchDataConfig(name="s", kind="static", n_cats=K,
                                  n_learners=N, n_items=60, seq_len=60, seed=seed))
    eng = DeepIRTEngine(ds, decoder="gpcm", emb_dim=8, hidden_dim=32,
                        state_alpha=True, item_key_dim=64, alpha_log_scale=1.0,
                        encoder="lstm", device="cuda", seed=seed)
    eng.fit(n_epochs=EPOCHS, batch_size=256)

    items = ds.items0
    a, th_model = alpha_per_pos(eng.model, items, ds.responses, "cuda")
    th_true = ds.theta_at_step                                # (N,T)
    Q = ds.cfg.n_items

    fi = items.ravel()
    fa = a.ravel()
    cnt = np.zeros(Q)
    asum = np.zeros(Q)
    np.add.at(asum, fi, fa)
    np.add.at(cnt, fi, 1.0)
    a_static = asum / np.maximum(cnt, 1.0)
    a_dyn = fa - a_static[fi]

    # within-item std of alpha (the size of the dynamic part)
    sq = np.zeros(Q)
    np.add.at(sq, fi, a_dyn ** 2)
    within_std = np.sqrt(sq / np.maximum(cnt, 1.0))
    cv = float(np.mean(within_std / np.maximum(np.abs(a_static), 1e-6)))

    rec = float(pearsonr(a_static, ds.gt.a)[0])               # a_static recovery
    ct = float(pearsonr(a_dyn, th_true.ravel())[0])           # spurious vs TRUE theta
    cm = float(pearsonr(a_dyn, th_model.ravel())[0])          # mechanical vs model theta
    return rec, cv, ct, cm


def main():
    print(f"\n=== a_dynamic null probe, static-alpha K={K}, {len(SEEDS)} seeds, "
          f"{EPOCHS} epochs ===")
    print(f"{'N':>5s} {'a_static_rec':>12s} {'dyn_CV':>7s} "
          f"{'corr(adyn,theta_true)':>22s} {'corr(adyn,theta_model)':>22s}")
    for N in NS:
        rs, cvs, cts, cms = [], [], [], []
        for seed in SEEDS:
            rec, cv, ct, cm = run(N, seed)
            rs.append(rec); cvs.append(cv); cts.append(ct); cms.append(cm)
        print(f"{N:5d} {np.mean(rs):12.3f} {np.mean(cvs):7.3f} "
              f"{np.mean(cts):+22.3f} {np.mean(cms):+22.3f}")


if __name__ == "__main__":
    main()
