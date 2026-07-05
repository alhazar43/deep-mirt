"""_qm3_mml.py -- GATE C: C-dimensional marginal calibration of the
multidimensional GPCM bank, and the cross-loading recovery gate.

Marginal ML over theta ~ MVN(0, R(r)) via product Gauss-Hermite quadrature
(probabilists' nodes, Cholesky-correlated). Loadings constrained positive
on the Q-support (sign identification); prior unit variances carry the
scale identification; thresholds free (centered in reporting).

Pre-registration: docs/qmirt_experiment_results.md, Session 1, GATE C.
Run from repo root: python deep_irt/bench/_qm3_mml.py [--quick]
Outputs: outputs/qm2/gatec/cell_*.json + gatec_summary.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qm3_datagen import corr_matrix, generate_static_cohort   # noqa: E402
from _qm2_metrics import seed_agg, spearman                     # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs", "qm2", "gatec")


def gh_nodes(C, n, R, device):
    """Product probabilists' Gauss-Hermite grid correlated by chol(R).
    Returns z_nodes (n^C, C), logw (n^C,)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / np.sqrt(2 * np.pi)
    grids = list(itertools.product(range(n), repeat=C))
    idx = np.array(grids)                                   # (M, C)
    nodes = x[idx]                                          # (M, C)
    logw = np.log(w)[idx].sum(axis=1)
    L = np.linalg.cholesky(R)
    z = nodes @ L.T
    return (torch.as_tensor(z, dtype=torch.float, device=device),
            torch.as_tensor(logw, dtype=torch.float, device=device))


def _logp_table(A, b, z_nodes):
    """LP (M, J, K): log P(category | item, node)."""
    eta = z_nodes @ A.T                                     # (M, J)
    steps = eta.unsqueeze(-1) - b.unsqueeze(0)              # (M, J, K-1)
    csum = torch.cumsum(steps, dim=-1)
    logits = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
    return F.log_softmax(logits, dim=-1)


def fit_mml(ds, n_nodes=7, epochs=200, lr=2e-2, device="cpu",
            verbose=False, prior_R=None):
    """prior_R: if given, the quadrature prior uses THIS correlation matrix
    instead of the generating one (misspecification/estimated-R arms; the
    2026-07-05 math review's demand -- the default known-R fit is an
    oracle-prior advantage and must be reported as such)."""
    bank = ds["bank"]
    J, C, K = bank["J"], bank["C"], bank["K"]
    R = prior_R if prior_R is not None \
        else corr_matrix(C, ds["config"]["r"])
    z_nodes, logw = gh_nodes(C, n_nodes, R, device)
    Qm = torch.as_tensor(bank["Q"], dtype=torch.float, device=device)
    seq = torch.as_tensor(ds["seq"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    N, T = seq.shape
    flat = (seq * K + resp).reshape(-1)                     # (N*T,)

    A_raw = torch.zeros(J, C, device=device, requires_grad=True)
    b0 = np.tile(np.linspace(-0.8, 0.8, K - 1), (J, 1))
    b = torch.tensor(b0, dtype=torch.float, device=device,
                     requires_grad=True)
    opt = torch.optim.Adam([A_raw, b], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        A = Qm * torch.exp(A_raw)
        LP = _logp_table(A, b, z_nodes)                     # (M, J, K)
        ll = LP.reshape(LP.shape[0], -1)[:, flat]           # (M, N*T)
        ll = ll.reshape(-1, N, T).sum(dim=2)                # (M, N)
        loss = -torch.logsumexp(logw.unsqueeze(1) + ll, dim=0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A_raw, b], 5.0)
        opt.step()
        if verbose and ep % 50 == 0:
            print(f"    ep{ep} nll={loss.item():.4f}")
    A = (Qm * torch.exp(A_raw)).detach().cpu().numpy()
    return A, b.detach().cpu().numpy(), float(loss.item())


def eap_corr(ds, A_hat, b_hat, n_nodes=7, device="cpu"):
    """Plug-in ability-correlation estimate: EAP scores per learner under
    an independence prior and the fitted bank, then their correlation
    (attenuated; one-iteration plug-in, reported as such)."""
    bank = ds["bank"]
    C, K = bank["C"], bank["K"]
    R0 = corr_matrix(C, 0.0)
    z_nodes, logw = gh_nodes(C, n_nodes, R0, device)
    A = torch.as_tensor(A_hat, dtype=torch.float, device=device)
    b = torch.as_tensor(b_hat, dtype=torch.float, device=device)
    seq = torch.as_tensor(ds["seq"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    N, T = seq.shape
    LP = _logp_table(A, b, z_nodes)
    flat = (seq * K + resp).reshape(-1)
    ll = LP.reshape(LP.shape[0], -1)[:, flat].reshape(-1, N, T).sum(2)
    post = torch.softmax(logw.unsqueeze(1) + ll, dim=0)      # (M, N)
    eap = post.T @ z_nodes                                    # (N, C)
    Rh = np.corrcoef(eap.cpu().numpy().T)
    Rh = np.clip(Rh, -0.95, 0.95)
    np.fill_diagonal(Rh, 1.0)
    return Rh


def fit_oracle(ds, epochs=200, lr=2e-2, device="cpu"):
    bank = ds["bank"]
    J, C, K = bank["J"], bank["C"], bank["K"]
    Qm = torch.as_tensor(bank["Q"], dtype=torch.float, device=device)
    theta = torch.as_tensor(ds["theta"], dtype=torch.float, device=device)
    seq = torch.as_tensor(ds["seq"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    A_raw = torch.zeros(J, C, device=device, requires_grad=True)
    b0 = np.tile(np.linspace(-0.8, 0.8, K - 1), (J, 1))
    b = torch.tensor(b0, dtype=torch.float, device=device,
                     requires_grad=True)
    opt = torch.optim.Adam([A_raw, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        A = Qm * torch.exp(A_raw)
        eta = (theta.unsqueeze(1) * A[seq]).sum(-1)          # (N, T)
        steps = eta.unsqueeze(-1) - b[seq]
        csum = torch.cumsum(steps, dim=-1)
        lg = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
        ll = F.log_softmax(lg, dim=-1)
        loss = -torch.gather(ll, 2, resp.unsqueeze(-1)).squeeze(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A_raw, b], 5.0)
        opt.step()
    A = (Qm * torch.exp(A_raw)).detach().cpu().numpy()
    return A, b.detach().cpu().numpy(), float(loss.item())


# ------------------------------------------------------------- metrics
def recovery(A_hat, b_hat, bank):
    A_true, Q = bank["A"], bank["Q"] > 0
    la_h, la_t, cols = [], [], []
    for c in range(bank["C"]):
        on = Q[:, c]
        h = np.log(np.maximum(A_hat[on, c], 1e-9))
        t = np.log(A_true[on, c])
        la_h.append(h - h.mean())
        la_t.append(t - t.mean())
        cols.append(float(np.polyfit(t - t.mean(), h - h.mean(), 1)[0]))
    alpha_rho = spearman(np.concatenate(la_h), np.concatenate(la_t))
    cross = ~bank["pure"]
    rh, rt = [], []
    for j in np.where(cross)[0]:
        c_on = np.where(Q[j])[0]
        rh.append(np.log(max(A_hat[j, c_on[0]], 1e-9))
                  - np.log(max(A_hat[j, c_on[1]], 1e-9)))
        rt.append(np.log(bank["A"][j, c_on[0]])
                  - np.log(bank["A"][j, c_on[1]]))
    ratio_rho = spearman(rh, rt)
    ratio_rmse = float(np.sqrt(np.mean((np.array(rh) - np.array(rt)) ** 2)))
    loc_h = b_hat.mean(axis=1)
    loc_t = bank["b"].mean(axis=1)
    prim = bank["primary"]
    for c in range(bank["C"]):
        m = prim == c
        loc_h[m] -= loc_h[m].mean()
        loc_t[m] -= loc_t[m].mean()
    return {"alpha_rho": alpha_rho, "alpha_slope": float(np.mean(cols)),
            "ratio_rho": ratio_rho, "ratio_rmse": ratio_rmse,
            "d_rho": spearman(loc_h, loc_t)}


def run_misspec(args, device):
    out = OUT + "_misspec"
    os.makedirs(out, exist_ok=True)
    C = 3
    rows = []
    for r in (0.3, 0.6):
        for seed in args.data_seeds:
            ds = generate_static_cohort(seed, n_pure=3, r=r, N=args.n,
                                        T=args.t)
            bank = ds["bank"]
            row = {"r": r, "seed": seed}
            arms = {"true": None, "zero": corr_matrix(C, 0.0)}
            fitted = {}
            for arm, R in arms.items():
                A_h, b_h, _ = fit_mml(ds, n_nodes=args.nodes,
                                      epochs=args.epochs, device=device,
                                      prior_R=R)
                fitted[arm] = (A_h, b_h)
                row[arm] = recovery(A_h, b_h, bank)
            Rh = eap_corr(ds, *fitted["zero"], device=device)
            row["Rhat_offdiag"] = float(Rh[np.triu_indices(C, 1)].mean())
            A_h, b_h, _ = fit_mml(ds, n_nodes=args.nodes,
                                  epochs=args.epochs, device=device,
                                  prior_R=Rh)
            row["est"] = recovery(A_h, b_h, bank)
            rows.append(row)
            print(f"[misspec r={r} s={seed}] " + " ".join(
                f"{a}: a={row[a]['alpha_rho']:.3f} "
                f"ratio={row[a]['ratio_rho']:.3f}"
                for a in ("true", "zero", "est"))
                + f" Rhat={row['Rhat_offdiag']:.2f}", flush=True)
    summary = {}
    for r in (0.3, 0.6):
        ks = [x for x in rows if x["r"] == r]
        summary[f"r{r}"] = {
            arm: {m: seed_agg([x[arm][m] for x in ks])
                  for m in ("alpha_rho", "ratio_rho")}
            for arm in ("true", "zero", "est")}
        summary[f"r{r}"]["Rhat"] = seed_agg([x["Rhat_offdiag"] for x in ks])
    with open(os.path.join(out, "misspec_summary.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=1)
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--t", type=int, default=60)
    p.add_argument("--nodes", type=int, default=7)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--rs", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    p.add_argument("--purities", type=int, nargs="+", default=[1, 3, 6])
    p.add_argument("--data-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--misspec", action="store_true",
                   help="known-R vs zero-R vs plug-in-estimated-R arms at "
                        "the gate purity (review demand)")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.misspec:
        run_misspec(args, device)
        return
    N = 150 if args.quick else args.n
    epochs = 60 if args.quick else args.epochs
    rs = args.rs[1:2] if args.quick else args.rs      # smoke the gate cell
    purities = args.purities[1:2] if args.quick else args.purities
    seeds = args.data_seeds[:1] if args.quick else args.data_seeds

    out_dir = OUT + ("_quick" if args.quick else "")  # never clobber
    globals()["OUT"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    cells = []
    for r in rs:
        for n_pure in purities:
            for seed in seeds:
                ds = generate_static_cohort(seed, n_pure=n_pure, r=r,
                                            N=N, T=args.t)
                row = {"r": r, "n_pure": n_pure, "seed": seed}
                t0 = time.time()
                A_hat, b_hat, nll = fit_mml(ds, n_nodes=args.nodes,
                                            epochs=epochs, device=device)
                row["mml"] = recovery(A_hat, b_hat, ds["bank"])
                row["mml"]["nll"] = nll
                A_o, b_o, _ = fit_oracle(ds, epochs=epochs, device=device)
                row["oracle"] = recovery(A_o, b_o, ds["bank"])
                row["secs"] = round(time.time() - t0, 1)
                cells.append(row)
                m, o = row["mml"], row["oracle"]
                print(f"[r={r} pure={n_pure} s={seed}] "
                      f"mml a={m['alpha_rho']:.3f} ratio={m['ratio_rho']:.3f}"
                      f" d={m['d_rho']:.3f} | oracle a={o['alpha_rho']:.3f} "
                      f"ratio={o['ratio_rho']:.3f} ({row['secs']}s)",
                      flush=True)
                with open(os.path.join(
                        OUT, f"cell_{r}_{n_pure}_{seed}.json"), "w") as f:
                    json.dump(row, f, indent=1)

    # gate evaluation per pre-registration
    def cellpass(row):
        m, o = row["mml"], row["oracle"]
        return (m["alpha_rho"] >= 0.85 and m["ratio_rho"] >= 0.80
                and m["alpha_rho"] >= o["alpha_rho"] - 0.10
                and m["ratio_rho"] >= o["ratio_rho"] - 0.10)

    summary = {}
    for r in rs:
        for n_pure in purities:
            ks = [c for c in cells
                  if c["r"] == r and c["n_pure"] == n_pure]
            summary[f"r{r}_p{n_pure}"] = {
                "pass_count": sum(cellpass(c) for c in ks),
                "n": len(ks),
                "alpha_rho": seed_agg([c["mml"]["alpha_rho"] for c in ks]),
                "ratio_rho": seed_agg([c["mml"]["ratio_rho"] for c in ks]),
                "oracle_alpha": seed_agg([c["oracle"]["alpha_rho"]
                                          for c in ks]),
                "oracle_ratio": seed_agg([c["oracle"]["ratio_rho"]
                                          for c in ks]),
            }
    gate_cell = summary.get("r0.3_p3")
    summary["GATE_C"] = ("PASS" if gate_cell and gate_cell["pass_count"] >= 2
                         else "FAIL" if gate_cell else "N/A(quick)")
    with open(os.path.join(OUT, "gatec_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
