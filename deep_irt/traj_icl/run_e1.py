"""run_e1.py -- in-context adaptation curves on a shared IRT scale.

Reads the per-model response files, fits a joint 2PL to place every
(model, shot count, condition) examinee on one shared ARC item scale,
reads off theta(k) as the in-context adaptation curve, fits the adaptation
rate, and compares the true-label curve against the shuffled-label control
to separate genuine in-context learning from format priming. Writes
results and plots to outputs/.

Run from the repo root with the project env, after generate.py has written
the response files:
    python -m deep_irt.traj_icl.run_e1
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_irt.traj_icl.irt_fit import load_responses, fit_2pl
from deep_irt.traj_synth.metrics import fit_rate

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT, help="dir holding responses_*.json")
    ap.add_argument("--tag", default="e1", help="output filename tag (e1, e1b)")
    args = ap.parse_args()
    out_dir, tag = args.dir, args.tag

    R, index, item_ids, acc = load_responses(out_dir)
    fit = fit_2pl(R, n_epochs=3000, lr=0.05)
    theta = fit["theta"]

    models = sorted({r["model"] for r in index})
    conds = sorted({r["condition"] for r in index})
    ks = sorted({r["k"] for r in index})

    def series(model, cond, vals):
        out = {}
        for i, r in enumerate(index):
            if r["model"] == model and r["condition"] == cond:
                out[r["k"]] = vals[i]
        return np.array([out[k] for k in ks])

    results = {
        "models": models, "conditions": conds, "k_values": ks,
        "n_items": int(R.shape[1]), "n_examinees": int(R.shape[0]),
        "final_bce": fit["final_bce"],
        "item_difficulty": {"mean": float(np.mean(fit["b"])),
                            "sd": float(np.std(fit["b"]))},
        "per_model": {},
    }

    print(f"E1 in-context adaptation | {len(models)} models | {R.shape[1]} items "
          f"| k={ks} | conds={conds} | 2PL BCE={fit['final_bce']:.4f}")
    print(f"\n{'model':28s} {'cond':9s} {'theta(k) ->':>40s}   rate")
    for m in models:
        rec = {"theta_true": None, "theta_shuffled": None, "acc_true": None,
               "acc_shuffled": None, "rate_true": None, "rate_shuffled": None,
               "priming_corrected_gain": None}
        for cond in conds:
            th = series(m, cond, theta)
            ac = series(m, cond, acc)
            r_hat, _, _ = fit_rate(th, robust=False, smooth=0, r_max=2.0, t=ks)
            rec[f"theta_{cond}"] = th.tolist()
            rec[f"acc_{cond}"] = ac.tolist()
            rec[f"rate_{cond}"] = float(r_hat)
            curve = " ".join(f"{v:+.2f}" for v in th)
            print(f"{m:28s} {cond:9s} {curve:>40s}   r={r_hat:.3f}")
        if rec["theta_true"] is not None and rec["theta_shuffled"] is not None:
            tt = np.array(rec["theta_true"])
            ts = np.array(rec["theta_shuffled"])
            # genuine in-context gain above the priming baseline, at max k
            rec["priming_corrected_gain"] = float((tt[-1] - tt[0]) - (ts[-1] - ts[0]))
        results["per_model"][m] = rec

    with open(os.path.join(out_dir, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: theta(k) per model, true solid vs shuffled dashed
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap("viridis")
    for mi, m in enumerate(models):
        c = cmap(mi / max(1, len(models) - 1))
        rec = results["per_model"][m]
        mtag = m.split("/")[-1]
        if rec["theta_true"]:
            ax.plot(ks, rec["theta_true"], "-o", color=c, label=f"{mtag} true")
        if rec["theta_shuffled"]:
            ax.plot(ks, rec["theta_shuffled"], "--x", color=c, alpha=0.7,
                    label=f"{mtag} shuffled")
    ax.set_xlabel("in-context shot count k")
    ax.set_ylabel("IRT ability theta (shared scale)")
    ax.set_title("E1: in-context adaptation curves")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_adaptation_curves.png"), dpi=130)
    plt.close(fig)

    print("\n--- priming-corrected gain (true minus shuffled, theta at max k) ---")
    for m in models:
        g = results["per_model"][m]["priming_corrected_gain"]
        print(f"{m:28s} {g:+.3f}" if g is not None else f"{m:28s} n/a")
    print(f"\nwrote results_{tag}.json, {tag}_adaptation_curves.png in {out_dir}")


if __name__ == "__main__":
    main()
