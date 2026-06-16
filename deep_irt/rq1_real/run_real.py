"""run_real.py -- RQ1 on REAL human dual-format data (HelpSteer2).

Format-agnostic shared quality scale, NO model judge. Reuses the validated
essay pipeline (ContentExtractor + sign-consistent GradedHead + BTHead +
JointEssayModel + IndepGraded/IndepBT; held-out-format transfer + trap detector).
Only the data layer changes: REAL human helpfulness ratings (graded) + REAL
signed human preferences (pairwise).

Synchronous, torch-only. No Ollama.

Usage (from repo root):
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="rl/src;ma-irt" \
    python deep_irt/rq1_real/run_real.py --n-pairs 4000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ESSAY = os.path.join(os.path.dirname(HERE), "rq1_essay")
sys.path.insert(0, HERE)
sys.path.insert(0, ESSAY)  # reuse the validated model/train/metrics modules

from helpsteer2_data import build_split  # noqa: E402
from essay_train import (  # noqa: E402
    train_joint, train_indep_graded, train_indep_bt,
)
from essay_metrics import compute_metrics  # noqa: E402

OUTPUT_DIR = Path(HERE) / "outputs"


def format_report(split, m, params) -> str:
    n = split.n
    L = [
        "=" * 72,
        "RQ1 REAL -- Format-agnostic shared QUALITY scale on REAL human data",
        "Venue: HelpSteer2 (human helpfulness ratings + REAL human preferences)",
        "PATH: pairwise comparisons are HUMAN annotations (NO model judge).",
        "=" * 72,
        "",
        "DESIGN",
        f"  Latent:               response helpfulness quality",
        f"  Graded format:        human helpfulness rating 0-4 (K={split.K})",
        f"  Pairwise format:      REAL signed human preference",
        f"  Total responses:      {n}",
        f"  GRADED-ONLY:          {len(split.graded_only_idx)}  "
        f"(rating only; preferences held out)",
        f"  PAIRWISE-ONLY:        {len(split.pairwise_only_idx)}  "
        f"(preferences only; rating held out)",
        f"  BOTH (bridge):        {len(split.both_idx)}",
        f"  Content features:     {split.feat_kind} (FROZEN, dim={split.features.shape[1]})",
        f"  Comparisons (used):   {len(split.comparisons)}",
        f"  Epochs:               {params.get('n_epochs')}",
        "",
        "OVERALL RECOVERY vs HUMAN HELPFULNESS  (Spearman, all responses)",
        f"  Joint (shared extractor):        {m['joint_overall']:+.4f}",
        f"  Indep graded (content-only):     {m['indep_graded_overall']:+.4f}",
        f"  Indep BT     (content-only):     {m['indep_bt_overall']:+.4f}",
        "",
        "HELD-OUT-FORMAT TRANSFER  (the RQ1 test)",
        "",
        "  [A] PAIRWISE-ONLY responses (helpfulness HELD OUT):",
        f"    Joint d_i vs human rating:             {m['joint_on_pairwise_only']:+.4f}  <- transfer",
        f"    Indep-BT readout (PI baseline):        {m['indep_bt_on_pairwise_only']:+.4f}",
        f"    Indep-GRADED content-only (trap chk):  {m['indep_graded_on_pairwise_only']:+.4f}",
        f"      gap vs indep-BT baseline:            {m['pairwise_only_gap_vs_indepBT']:+.4f}",
        f"      gap vs content-only (genuine xfer):  {m['pairwise_only_gap_vs_content']:+.4f}",
        "",
        "  [B] GRADED-ONLY responses (preferences HELD OUT):",
        f"    Joint d_i vs human rating:             {m['joint_on_graded_only']:+.4f}  <- transfer",
        f"    Indep-GRADED readout (PI baseline):    {m['indep_graded_on_graded_only']:+.4f}",
        f"    Indep-BT content-only (trap chk):      {m['indep_bt_on_graded_only']:+.4f}",
        f"      gap vs indep-graded baseline:        {m['graded_only_gap_vs_indepGraded']:+.4f}",
        f"      gap vs content-only (genuine xfer):  {m['graded_only_gap_vs_content']:+.4f}",
        "",
        "BRIDGE / DIAGNOSTICS",
        f"  Joint on BOTH responses vs human:        {m['joint_on_both']:+.4f}",
        f"  pref-vs-human-rating pair agreement:     {m['llm_human_pair_agreement_both_po']:.4f}",
        "",
    ]

    po_xfer = m["pairwise_only_gap_vs_content"]
    po_joint = m["joint_on_pairwise_only"]
    go_xfer = m["graded_only_gap_vs_content"]
    go_joint = m["joint_on_graded_only"]
    L.append("VERDICT (genuine content-mediated transfer = joint beats content-only)")
    for tag, joint, xfer in [("PAIRWISE-ONLY", po_joint, po_xfer),
                             ("GRADED-ONLY", go_joint, go_xfer)]:
        if np.isnan(joint):
            L.append(f"  [{tag}] INDETERMINATE (degenerate variance / too few).")
        elif joint >= 0.30 and xfer >= 0.05:
            L.append(f"  [{tag}] GENUINE TRANSFER: lands on held-out scale "
                     f"(rho={joint:+.3f}) ABOVE content-only (gap={xfer:+.3f}).")
        elif joint >= 0.30 and xfer < 0.05:
            L.append(f"  [{tag}] TRAP/INCONCLUSIVE: lands (rho={joint:+.3f}) but "
                     f"content-only matches it (gap={xfer:+.3f}). Content does the "
                     f"work, not cross-format unification.")
        else:
            L.append(f"  [{tag}] WEAK/NULL: rho={joint:+.3f}.")
    L.append("")
    L.append("DOMAIN CAVEAT: general-domain LLM-response helpfulness (HelpSteer2),")
    L.append("not educational item difficulty. Both formats are HUMAN judgments.")
    L.append("=" * 72)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=800)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--bt-weight", type=float, default=1.0)
    ap.add_argument("--no-st", action="store_true")
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--tag", default="hs2")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 72)
    print("RQ1 REAL (HelpSteer2)")
    print("=" * 72)

    print(f"\n[1/4] Building HelpSteer2 split (n_pairs={args.n_pairs}) ...", flush=True)
    split = build_split(
        n_pairs=args.n_pairs, seed=args.seed,
        prefer_st=not args.no_st, max_chars=args.max_chars,
    )
    print(f"  responses={split.n}  K={split.K}  features={split.feat_kind} "
          f"(dim={split.features.shape[1]})", flush=True)
    print(f"  graded-only={len(split.graded_only_idx)} "
          f"pairwise-only={len(split.pairwise_only_idx)} "
          f"both={len(split.both_idx)}", flush=True)
    print(f"  comparisons used={len(split.comparisons)}", flush=True)
    print(f"  helpfulness range observed: "
          f"{split.raw_scores.min():.2f}-{split.raw_scores.max():.2f}", flush=True)

    comparisons = split.comparisons
    if len(comparisons) < 50:
        print("BLOCKER: too few comparisons after partition. Increase --n-pairs.")
        sys.exit(1)

    # Pref-vs-rating agreement on the used comparisons (sanity: real-data signal)
    dec = [(i, j, o) for (i, j, o) in comparisons
           if split.raw_scores[i] != split.raw_scores[j]]
    agree = sum(1 for i, j, o in dec
                if (split.raw_scores[i] > split.raw_scores[j]) == (o == 1))
    print(f"  pref-vs-rating agreement on non-tied pairs: "
          f"{agree}/{len(dec)} = {agree/max(len(dec),1):.3f}", flush=True)

    print(f"\n[2/4] Training joint shared-extractor model ...", flush=True)
    joint = train_joint(split, comparisons, hidden_dim=args.hidden_dim,
                        n_epochs=args.n_epochs, bt_weight=args.bt_weight,
                        seed=args.seed, verbose=True)
    print(f"\n[3/4] Training independent baselines ...", flush=True)
    indep_graded = train_indep_graded(split, hidden_dim=args.hidden_dim,
                                      n_epochs=args.n_epochs, seed=args.seed,
                                      verbose=True)
    indep_bt = train_indep_bt(split, comparisons, hidden_dim=args.hidden_dim,
                             n_epochs=args.n_epochs, seed=args.seed, verbose=True)

    joint_q = joint.get_quality().cpu().numpy()
    indep_graded_q = indep_graded.get_quality().cpu().numpy()
    indep_bt_q = indep_bt.get_quality().cpu().numpy()

    print(f"\n[4/4] Computing metrics ...", flush=True)
    m = compute_metrics(split, comparisons, joint_q, indep_graded_q, indep_bt_q)
    params = {
        "venue": "helpsteer2", "n_responses": split.n,
        "n_comparisons": len(comparisons), "n_pairs_requested": args.n_pairs,
        "n_epochs": args.n_epochs, "seed": args.seed,
        "feat_kind": split.feat_kind, "bt_weight": args.bt_weight,
        "pref_rating_agreement": agree / max(len(dec), 1),
    }
    report = format_report(split, m, params)
    print("\n" + report, flush=True)

    suffix = f"_{args.tag}_seed{args.seed}"
    (OUTPUT_DIR / f"report{suffix}.txt").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / f"results{suffix}.json").write_text(
        json.dumps({k: (round(float(v), 6) if v == v else None)
                    for k, v in m.items()}, indent=2), encoding="utf-8")
    (OUTPUT_DIR / f"params{suffix}.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8")
    print(f"\nOutputs -> {OUTPUT_DIR}/  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
