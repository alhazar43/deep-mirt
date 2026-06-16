"""
theta_read_probe.py -- decisive probe: is ma-irt-lstm_gpcm's theta gap a
REPORTING artifact (clean read closes it) or a TRAINING-time smear (it persists)?

This EXTENDS the engine head-to-head bench. It does not edit ma-irt (frozen,
run-only) and does not touch run_bench.py's outputs. It reuses datagen.generate
(same seeds, same ground truth) and the existing engines, trains both engines
on the SAME synthetic datasets, then extracts ma-irt's per-step theta SEVERAL
ways and rescores recovery with the SAME metrics_bench functions the headline
bench uses.

What couples theta to the discrimination heads in ma-irt
--------------------------------------------------------
The GPCM step value (models/components/irt.py::GPCMLogits) is, per step,

    phi_k = sum_{h<=k} ( <alpha, theta>  -  ||alpha|| * beta_h )

For D=1 this is alpha*(theta - beta_h). The likelihood only ever sees the
PRODUCT alpha*theta (the interaction) and the term ||alpha||*beta. theta is
NEVER seen alone. So the absolute scale/orientation of the reported theta is
whatever the optimiser settled into, coupled per-step to the learned alpha
through the orbit (theta, alpha, beta) <-> (theta/c, c*alpha, beta/c).

ma-irt reports theta = ability_network(student_summary) * ability_scale
(decoders/gpcm.py L104). It is the encoder's DIRECT per-step ability output;
there is NO explicit ||alpha|| rescaling at report time. The "rescaling" lives
inside the likelihood (the ||alpha||*beta term), not in the theta readout.

Theta-read variants scored here (all per-step (N,T), then static final-step /
dynamic net-drift, exactly as metrics_bench does):

  (a) reported            : out["theta"]  -- reproduces the headline number.
  (b) raw_ability         : identical to (a) for this model (no report-time
                            rescale exists); kept as an explicit control so the
                            "raw theta" hypothesis is on the table and shown to
                            equal (a).
  (c) interaction         : <alpha, theta> = the effective ability the
                            likelihood actually identifies (alpha*theta for D=1).
                            This is the quantity the data constrains; reading it
                            removes nothing the model used, it reads the model's
                            OWN effective-ability signal directly.
  (d) interaction_over_an : <alpha, theta> / ||alpha||  = ability put back on
                            the beta (difficulty) scale. For D=1 equals theta
                            again analytically, but with per-step alpha it is a
                            distinct numerical read; this is the read most
                            consistent with how deep_irt reads a bare scalar
                            ability on the difficulty scale.

The deep_irt-LSTM scalar theta (its encoder's per-step ability, read with no
discrimination coupling at all) is the clean-ability reference target.

Run
---
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="deep_irt;rl/src;ma-irt" \
      python deep_irt/bench/theta_read_probe.py [--device cuda] [--epochs N]
      [--seeds 0 1 2]

Prints a small table (theta-read variant -> static / dynamic recovery) for
ma-irt-lstm_gpcm next to the deep_irt-LSTM baseline, aggregated over seeds.
Writes nothing unless --save is passed (then a sibling json/md, not the
headline files).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench import metrics_bench as M
from deep_irt.bench.engines import DeepIRTEngine, MaIrtEngine


# ---------------------------------------------------------------------------
# theta-read extraction for the ma-irt lstm_gpcm engine
# ---------------------------------------------------------------------------

@torch.no_grad()
def ma_irt_theta_reads(engine: MaIrtEngine) -> dict:
    """Return a dict {variant_name: theta_traj (N,T)} for the ma-irt engine.

    One forward pass over ALL learners (full sequences), then the per-step
    (theta, alpha, beta) tensors are combined four ways. alpha/theta carry a
    trailing trait dim D (=1 here); we keep the D-aware ops so this stays
    correct if a D>1 run is ever probed.
    """
    q_all = torch.tensor(engine.ds.items0 + 1, dtype=torch.long, device=engine.device)
    r_all = torch.tensor(engine.ds.responses, dtype=torch.long, device=engine.device)
    engine.model.eval()
    out = engine.model(q_all, r_all)

    theta = out["theta"]          # (N, T, D)
    alpha = out["alpha"]          # (N, T, D)

    # (a)/(b) reported / raw ability: the bare ability_network output.
    reported = theta.squeeze(-1)                         # (N, T)

    # (c) interaction <alpha, theta>: the effective ability the GPCM uses.
    interaction = (alpha * theta).sum(dim=-1)            # (N, T)

    # (d) interaction / ||alpha||: ability on the difficulty scale.
    alpha_norm = alpha.norm(dim=-1).clamp_min(1e-8)      # (N, T)
    interaction_over_an = interaction / alpha_norm       # (N, T)

    return {
        "(a) reported  out[theta]":      reported.cpu().numpy(),
        "(b) raw_ability (==a)":         reported.cpu().numpy(),
        "(c) interaction <a,th>":        interaction.cpu().numpy(),
        "(d) <a,th>/||a||":              interaction_over_an.cpu().numpy(),
    }


def score_theta(theta_traj: np.ndarray, ds) -> tuple[float, float]:
    """Return (static_spearman, dynamic_netdrift_spearman) for a (N,T) read.

    Static uses the final-step level vs the constant true theta0; dynamic uses
    within-learner net drift vs the true trajectory. Exactly metrics_bench.
    """
    if ds.cfg.kind == "dynamic":
        d = M.theta_recovery_dynamic(theta_traj, ds.gt.theta_traj)
        return float("nan"), d["theta_netdrift_spearman"]
    else:
        s = M.theta_recovery_static(theta_traj[:, -1], ds.gt.theta0)
        return s["theta_spearman"], float("nan")


# ---------------------------------------------------------------------------
# deep_irt clean-ability reference
# ---------------------------------------------------------------------------

@torch.no_grad()
def deep_irt_theta_traj(engine: DeepIRTEngine) -> np.ndarray:
    it = torch.tensor(engine.ds.items0, dtype=torch.long, device=engine.device)
    rp = torch.tensor(engine.ds.responses, dtype=torch.long, device=engine.device)
    return engine.model.track(it, rp).cpu().numpy()      # (N, T)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--Q", type=int, default=60)
    ap.add_argument("--T", type=int, default=60)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"=== theta-read probe: device={device} epochs={args.epochs} "
          f"seeds={args.seeds} N={args.N} Q={args.Q} T={args.T} ===")

    variant_names = [
        "(a) reported  out[theta]",
        "(b) raw_ability (==a)",
        "(c) interaction <a,th>",
        "(d) <a,th>/||a||",
    ]

    # accumulate per (dataset, variant) across seeds
    acc = {ds: {v: [] for v in variant_names} for ds in ("static_k4", "dynamic_k4")}
    sub_acc = {"static_k4": [], "dynamic_k4": []}

    for seed in args.seeds:
        data_cfgs = {
            "static_k4": BenchDataConfig(name="static_k4", kind="static", n_cats=4,
                                         n_learners=args.N, n_items=args.Q,
                                         seq_len=args.T, seed=seed),
            "dynamic_k4": BenchDataConfig(name="dynamic_k4", kind="dynamic", n_cats=4,
                                          n_learners=args.N, n_items=args.Q,
                                          seq_len=args.T, drift_sigma=0.15, seed=seed),
        }
        for dsname, cfg in data_cfgs.items():
            ds = generate(cfg)
            kind = ds.cfg.kind

            # ma-irt lstm_gpcm (the engine under test)
            mai = MaIrtEngine(ds, encoder="lstm_gpcm", device=device, seed=seed)
            mai.fit(n_epochs=args.epochs)
            reads = ma_irt_theta_reads(mai)
            for v in variant_names:
                st, dy = score_theta(reads[v], ds)
                acc[dsname][v].append(dy if kind == "dynamic" else st)

            # deep_irt clean-ability reference
            sub = DeepIRTEngine(ds, decoder="gpcm", device=device, seed=seed)
            sub.fit(n_epochs=args.epochs)
            st, dy = score_theta(deep_irt_theta_traj(sub), ds)
            sub_acc[dsname].append(dy if kind == "dynamic" else st)

            print(f"  seed={seed} {dsname}: "
                  f"ma-irt reads="
                  + ", ".join(f"{v.split()[0]}={acc[dsname][v][-1]:.3f}" for v in variant_names)
                  + f" | deep_irt={sub_acc[dsname][-1]:.3f}")

    # ---- render aggregate table ----
    def agg(xs):
        xs = [x for x in xs if x == x]
        if not xs:
            return float("nan"), float("nan")
        return float(np.mean(xs)), float(np.std(xs))

    print("\n" + "=" * 78)
    print("THETA-READ COMPARISON  (ma-irt-lstm_gpcm), mean +- std over "
          f"seeds {args.seeds}")
    print("=" * 78)
    header = f"{'theta read':30s} | {'static (sp)':>16s} | {'dynamic net-drift':>18s}"
    print(header)
    print("-" * len(header))
    for v in variant_names:
        sm, ss = agg(acc["static_k4"][v])
        dm, dsd = agg(acc["dynamic_k4"][v])
        print(f"{v:30s} | {sm:6.3f} +- {ss:5.3f}  | {dm:6.3f} +- {dsd:5.3f}")
    print("-" * len(header))
    sm, ss = agg(sub_acc["static_k4"])
    dm, dsd = agg(sub_acc["dynamic_k4"])
    print(f"{'deep_irt-LSTM (clean ref)':30s} | {sm:6.3f} +- {ss:5.3f}  | "
          f"{dm:6.3f} +- {dsd:5.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
