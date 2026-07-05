"""_qm3_gates.py -- GATE A (encoder-off regression) and GATE B (encoder
honesty), per the Session-1 pre-registrations in
docs/qmirt_experiment_results.md.

GATE A: the qm3 model, encoder OFF, one-hot converted qm2 bank FROZEN at
truth, on the qm2 forecast beds (pos/null/neg x data seeds x model seeds,
converged budgets). Must land in the pre-registered bands (pos score gap
in [0.02, 0.06], 9/9 positive; null |gap| <= 0.005, |G_BA| <= 0.01; neg
sign 9/9). Kill thresholds for the qm3 class are then re-registered from
these cells.

GATE B: encoder ON (LSTM recognition network proposing z0, lambda, gamma;
geometric-mean-1 constraints), gamma free, two inference windows (early =
first 20 steps; full conditioning window). Beds: null and pos twins.
FABRICATION metrics: population score gap on B; G_hat_BA; and the
PER-LEARNER effect e_i = mean over forecast B steps of (predicted score
with-G minus predicted score no-G) -- under null, 95th percentile of
|e_i| must be <= 0.01 (pre-registered), else the encoder channel
fabricates transfer for a tail of learners that population means hide.

Run from repo root:
    python deep_irt/bench/_qm3_gates.py --gate A [--quick]
    python deep_irt/bench/_qm3_gates.py --gate B [--quick]
Outputs: outputs/qm2/gate_a/ and outputs/qm2/gate_b/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qm2_datagen import A as CA, B as CB, U as CU              # noqa: E402
from _qm2_datagen import generate, sparse_g                      # noqa: E402
from _qm2_metrics import seed_agg                                # noqa: E402
from _qm3_datagen import mgpcm_probs                             # noqa: E402
from _qm3_model import QM3Model, bank_from_qm2, fit_two_stage    # noqa: E402

OUTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "qm2")
KINDS = {"pos": +0.025, "null": 0.0, "neg": -0.025}


def run_cell(kind, g_val, ds_seed, m_seed, N, e1, e2, device,
             use_encoder=False, enc_window=None, enc_gamma=True,
             freeze_traits=False):
    g = None if g_val == 0.0 else sparse_g(val=g_val)
    ds = generate(ds_seed, kind="matched", N=N, g_true=g,
                  schedule="forecast",
                  theta0_mu_B=(0.5 if g_val < 0 else None))
    items, marks = ds["items"], ds["marks"]
    bank = bank_from_qm2(items)
    T_cond = marks["T_cond"]
    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    cond_mask = torch.zeros_like(resp, dtype=torch.float)
    cond_mask[:, :T_cond] = 1.0

    torch.manual_seed(m_seed)
    # LEAK GUARD (caught 2026-07-05 pre-run): the encoder may read AT MOST
    # the conditioning window; "cond" resolves to T_cond here. A raw None
    # would read the full sequence including masked forecast responses and
    # void Gate B.
    ew = T_cond if enc_window in (None, "cond") else int(enc_window)
    model = QM3Model(bank, N, inert_items=ds["inert_items"],
                     use_encoder=use_encoder,
                     enc_window=(ew if use_encoder else None),
                     enc_gamma=enc_gamma).to(device)
    if freeze_traits:            # Gate A matched control: lambda=gamma=1
        with torch.no_grad():
            model.ul_free.zero_()
            model.ug_free.zero_()
        model.ul_free.requires_grad_(False)
        model.ug_free.requires_grad_(False)
    fit_two_stage(model, seq_eff, resp, cond_mask, e1=e1, e2=e2)

    with torch.no_grad():
        z_w = model.unroll(seq_eff, resp).cpu().numpy()
        z_n = model.unroll(seq_eff, resp, g_off_from=T_cond).cpu().numpy()
        G_hat = model.G_signed.cpu().numpy()

    K = items["K"]

    def pred_scores(z_arm, steps):
        out = np.zeros((len(steps), N))
        for si, t in enumerate(steps):
            for i in range(N):
                j = int(ds["seq_eff"][i, t])
                p = mgpcm_probs(bank["A"][j], bank["b"][j], z_arm[i, t])
                out[si, i] = (p * np.arange(K)).sum() / (K - 1)
        return out

    obs = {c: ds["resp"][:, marks[f"fc_{n}"]].T / (K - 1)
           for n, c in (("B", CB), ("U", CU))}
    res = {"kind": kind, "data_seed": ds_seed, "model_seed": m_seed,
           "G_hat_BA": float(G_hat[CB, CA]),
           "G_hat_UA": float(G_hat[CU, CA]),
           "enc": bool(use_encoder), "enc_window": enc_window}
    for name, c in (("B", CB), ("U", CU)):
        steps = marks[f"fc_{name}"]
        pw, pn = pred_scores(z_w, steps), pred_scores(z_n, steps)
        ob = ds["resp"][:, steps].astype(float).T / (K - 1)
        err_w = float(np.abs(pw - ob).mean())
        err_n = float(np.abs(pn - ob).mean())
        res[f"score_gap_{name}"] = err_n - err_w
        if name == "B":
            e_i = (pw - pn).mean(axis=0)          # per-learner effect
            res["perlearner_p95_absB"] = float(
                np.percentile(np.abs(e_i), 95))
            res["perlearner_mean_B"] = float(e_i.mean())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate", choices=["A", "B", "B2", "ACTRL"],
                   required=True)
    p.add_argument("--b-kinds", nargs="+", default=["null", "pos"],
                   help="chunking filter for encoder gates")
    p.add_argument("--b-windows", nargs="+", default=["20", "cond"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--e1", type=int, default=500)
    p.add_argument("--e2", type=int, default=300)
    p.add_argument("--data-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 100 if args.quick else args.n
    e1 = 40 if args.quick else args.e1
    e2 = 25 if args.quick else args.e2
    dss = args.data_seeds[:1] if args.quick else args.data_seeds
    mss = args.model_seeds[:1] if args.quick else args.model_seeds

    out = os.path.join(OUTROOT, "gate_" + args.gate.lower())
    if args.quick:
        out += "_quick"                      # never clobber official cells
    os.makedirs(out, exist_ok=True)
    cells = []
    if args.gate in ("A", "ACTRL"):
        combos = [(k, None, False) for k in ("pos", "null", "neg")]
    else:
        wins = [20 if w == "20" else "cond" for w in args.b_windows]
        combos = [(k, w, True) for k in args.b_kinds
                  for w in wins]                 # early vs full CONDITIONING
                                                 # window (never the forecast)
    for kind, win, enc in combos:
        for ds_seed in dss:
            for m_seed in mss:
                t0 = time.time()
                r = run_cell(kind, KINDS[kind], ds_seed, m_seed, N, e1, e2,
                             device, use_encoder=enc, enc_window=win,
                             enc_gamma=(args.gate != "B2"),
                             freeze_traits=(args.gate == "ACTRL"))
                r["secs"] = round(time.time() - t0, 1)
                cells.append(r)
                tag = f"w{win}" if enc else "free"
                print(f"[{kind}/{tag} ds={ds_seed} ms={m_seed}] "
                      f"gapB={r['score_gap_B']:+.4f} "
                      f"gapU={r['score_gap_U']:+.4f} "
                      f"G_BA={r['G_hat_BA']:+.4f} "
                      f"p95={r.get('perlearner_p95_absB', float('nan')):.4f}"
                      f" ({r['secs']}s)", flush=True)
                with open(os.path.join(
                        out, f"cell_{kind}_{tag}_{ds_seed}_{m_seed}.json"),
                        "w") as f:
                    json.dump(r, f, indent=1)

    summary = {}
    keyfn = (lambda c: c["kind"]) if args.gate in ("A", "ACTRL") else \
        (lambda c: f"{c['kind']}_w{c['enc_window']}")
    for key in sorted({keyfn(c) for c in cells}):
        ks = [c for c in cells if keyfn(c) == key]
        summary[key] = {
            "score_gap_B": seed_agg([c["score_gap_B"] for c in ks]),
            "score_gap_U": seed_agg([c["score_gap_U"] for c in ks]),
            "G_hat_BA": seed_agg([c["G_hat_BA"] for c in ks]),
            "perlearner_p95": seed_agg(
                [c["perlearner_p95_absB"] for c in ks]),
        }
    if args.gate in ("A", "ACTRL"):
        pos, nul = summary.get("pos"), summary.get("null")
        ok = (pos and nul
              and 0.02 <= pos["score_gap_B"]["mean"] <= 0.06
              and pos["score_gap_B"]["n_pos"] == pos["score_gap_B"]["n"]
              and abs(nul["score_gap_B"]["mean"]) <= 0.005
              and abs(nul["G_hat_BA"]["mean"]) <= 0.01)
        summary["GATE_A"] = "PASS" if ok else "FAIL"
    else:
        verdicts = {}
        for w in ("w20", "wcond"):
            nul = summary.get(f"null_{w}")
            if nul:
                verdicts[w] = ("PASS" if
                               abs(nul["score_gap_B"]["mean"]) <= 0.005
                               and nul["perlearner_p95"]["mean"] <= 0.01
                               else "FAIL")
        summary["GATE_B"] = verdicts
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
