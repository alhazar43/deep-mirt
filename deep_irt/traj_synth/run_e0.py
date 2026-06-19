"""run_e0.py -- E0 sweep: synthetic ability-trajectory rate recovery.

Trains the encoder-decoder under prediction loss on respondents with known
learning-curve rates, recovers the per-step ability with model.track(),
fits the curve, and scores the recovered rate against the truth. Sweeps
response format (binary, ordinal) and sequence density (seq_len), over
several seeds. Reports the encoder's rate recovery, an oracle ceiling
(rate MLE with known item params and curve family), and a fitter sanity
check (curve fit to the noiseless true trajectory).

Run from the repo root with the project env:
    python -m deep_irt.traj_synth.run_e0            # full sweep
    python -m deep_irt.traj_synth.run_e0 --quick    # fast smoke
"""

import argparse
import json
import os

import numpy as np
import torch

from deep_irt.core.model import DeepIRTModel
from deep_irt.traj_synth.data_gen import generate_learning_curves
from deep_irt.traj_synth import metrics as M


def _train_model(data, decoder, n_cats, n_epochs, lr, device, seed):
    item_ids = data["seq_item_ids"]
    responses = data["seq_responses"]
    model = DeepIRTModel(
        num_items=data["n_items"], emb_dim=8, hidden_dim=32, n_cats=n_cats,
        decoder=decoder, encoder="lstm", decouple=True, device=device, seed=seed,
    )
    model.fit(item_ids, responses, n_epochs=n_epochs, lr=lr, verbose=False)
    return model


def _responsive_theta(model, item_ids, responses):
    """theta_t that has seen the response at t (the dynamic track readout)."""
    return model.track(item_ids, responses).cpu().numpy()


def _aligned_theta(model, item_ids, responses):
    """Prediction-aligned theta_t (history strictly before t), less jumpy."""
    enc = model.encoder
    enc.eval()
    ii = item_ids.to(model.device)
    rr = responses.to(model.device)
    with torch.no_grad():
        try:
            th, _ = enc.aligned_theta_and_state(ii, rr)
        except Exception:
            th = enc.theta_for_prediction(ii, rr)
    return th.cpu().numpy()


def run_condition(fmt, seq_len, seeds, n_items, n_resp, n_epochs, lr, device,
                  oracle_max=150):
    """One (format, seq_len) cell, averaged over seeds. Oracle on seed 0."""
    K = 2 if fmt == "binary" else 4
    decoder = "binary" if fmt == "binary" else "gpcm"

    per_seed = []
    per_seed_aligned = []
    headline = None
    oracle_summary = None
    sanity_summary = None
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")

    for si, seed in enumerate(seeds):
        data = generate_learning_curves(
            n_items=n_items, n_respondents=n_resp, seq_len=seq_len, K=K, seed=seed,
        )
        r_true = data["rate_true"].numpy()
        theta_true = data["theta_true_traj"].numpy()
        item_ids, responses = data["seq_item_ids"], data["seq_responses"]

        model = _train_model(data, decoder, K, n_epochs, lr, device, seed)
        theta_resp = _responsive_theta(model, item_ids, responses)       # (N, T)
        theta_align = _aligned_theta(model, item_ids, responses)         # (N, T)

        # Hardened readout (light smoothing + robust loss) on each theta stream.
        r_enc = M.recover_rates(theta_resp, robust=True, smooth=1)
        r_aln = M.recover_rates(theta_align, robust=True, smooth=1)
        enc_sum = M.rate_recovery_summary(r_enc, r_true)
        aln_sum = M.rate_recovery_summary(r_aln, r_true)
        perstep_mean, _ = M.perstep_ability_corr(theta_resp, theta_true)
        enc_sum["perstep_corr"] = perstep_mean
        per_seed.append(enc_sum)
        per_seed_aligned.append(aln_sum)

        if si == 0:
            r_truecurve = M.recover_rates(theta_true, robust=False, smooth=0)
            sanity_summary = M.rate_recovery_summary(r_truecurve, r_true)
            r_oracle = M.oracle_rates(
                data["seq_responses"], data["seq_item_ids"],
                data["a"], data["betas"], max_resp=oracle_max,
            )
            oracle_summary = M.rate_recovery_summary(r_oracle, r_true[: len(r_oracle)])
            headline = {
                "r_true": r_true.tolist(),
                "r_enc": [None if not np.isfinite(x) else float(x) for x in r_enc],
                "r_oracle": [None if not np.isfinite(x) else float(x)
                             for x in r_oracle],
            }
            # Save trajectories so readout choices can be re-explored offline.
            np.savez_compressed(
                os.path.join(out_dir, f"traj_{fmt}_T{seq_len}_seed{seed}.npz"),
                theta_resp=theta_resp, theta_align=theta_align,
                theta_true=theta_true, r_true=r_true,
            )
        print(f"  [{fmt:6s} T={seq_len:3d} seed={seed}] "
              f"enc rho={enc_sum['spearman']:.3f} aligned rho={aln_sum['spearman']:.3f} "
              f"mae={enc_sum['median_abs_err']:.3f} perstep={perstep_mean:.3f}")

    def agg(rows, key):
        vals = np.array([d[key] for d in rows], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    p_m, p_s = agg(per_seed, "pearson")
    sp_m, sp_s = agg(per_seed, "spearman")
    mae_m, mae_s = agg(per_seed, "median_abs_err")
    ps_m, ps_s = agg(per_seed, "perstep_corr")
    ff_m, _ = agg(per_seed, "frac_fit")
    aln_m, aln_s = agg(per_seed_aligned, "spearman")

    return {
        "format": fmt, "seq_len": seq_len, "K": K,
        "enc": {
            "pearson_mean": p_m, "pearson_std": p_s,
            "spearman_mean": sp_m, "spearman_std": sp_s,
            "median_abs_err_mean": mae_m, "median_abs_err_std": mae_s,
            "perstep_corr_mean": ps_m, "perstep_corr_std": ps_s,
            "frac_fit_mean": ff_m, "per_seed": per_seed,
            "aligned_spearman_mean": aln_m, "aligned_spearman_std": aln_s,
            "per_seed_aligned": per_seed_aligned,
        },
        "oracle_seed0": oracle_summary,
        "sanity_truecurve_seed0": sanity_summary,
        "headline_arrays_seed0": headline,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke, one cell")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n_resp", type=int, default=300)
    ap.add_argument("--n_items", type=int, default=120)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = os.path.dirname(__file__)
    os.makedirs(os.path.join(here, "outputs"), exist_ok=True)

    if args.quick:
        formats = ["gpcm"]
        seq_lens = [30]
        seeds = [0]
        n_epochs, n_resp, n_items = 120, 64, 60
        out = args.out or os.path.join(here, "outputs", "results_e0_quick.json")
    else:
        formats = ["binary", "gpcm"]
        seq_lens = [20, 40, 80]
        seeds = [0, 1, 2]
        n_epochs, n_resp, n_items = args.epochs, args.n_resp, args.n_items
        out = args.out or os.path.join(here, "outputs", "results_e0.json")

    print(f"E0 trajectory-rate recovery | device={device} | epochs={n_epochs} "
          f"| N={n_resp} | items={n_items} | seeds={seeds}")

    conditions = []
    for fmt in formats:
        for T in seq_lens:
            conditions.append(run_condition(
                fmt, T, seeds, n_items, n_resp, n_epochs, 1e-2, device,
            ))

    result = {
        "config": {
            "formats": formats, "seq_lens": seq_lens, "seeds": seeds,
            "n_epochs": n_epochs, "n_resp": n_resp, "n_items": n_items,
            "device": str(device), "lr": 1e-2,
            "curve": "theta(t)=theta_inf-(theta_inf-theta_0)exp(-r t)",
        },
        "conditions": conditions,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== E0 summary (rate recovery) ===")
    print(f"{'fmt':6s} {'T':>4s} {'enc_rho':>8s} {'aln_rho':>8s} {'enc_mae':>8s} "
          f"{'oracle_rho':>11s} {'sanity_rho':>11s} {'perstep':>8s}")
    for c in conditions:
        e = c["enc"]
        orc = c["oracle_seed0"]["spearman"] if c["oracle_seed0"] else float("nan")
        san = (c["sanity_truecurve_seed0"]["spearman"]
               if c["sanity_truecurve_seed0"] else float("nan"))
        print(f"{c['format']:6s} {c['seq_len']:4d} {e['spearman_mean']:8.3f} "
              f"{e['aligned_spearman_mean']:8.3f} {e['median_abs_err_mean']:8.3f} "
              f"{orc:11.3f} {san:11.3f} {e['perstep_corr_mean']:8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
