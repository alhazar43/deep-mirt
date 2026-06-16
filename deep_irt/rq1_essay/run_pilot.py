"""run_pilot.py -- RQ1 essay pilot orchestrator (bounded, synchronous).

Pipeline:
  1. Build split: sample essays from one ASAP set, frozen content features,
     partition graded-only / pairwise-only / both.
  2. Sample pairwise comparison pairs over (pairwise-only + both) essays.
  3. Elicit LLM judgments via Ollama (ONE judge model, bounded count).
  4. Train joint shared-extractor model + indep-graded + indep-BT baselines.
  5. Held-out-format transfer metrics + the both-recover-truth trap detector.

Usage (from repo root):
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="deep_irt;rl/src;ma-irt" \
    python deep_irt/rq1_essay/run_pilot.py \
      --essay-set 1 --n-essays 80 --n-comparisons 400 \
      --judge-model llama3.2:3b --seed 0
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
sys.path.insert(0, HERE)

from essay_data import build_split, load_asap_set  # noqa: E402
from elicit import elicit_comparisons, list_models  # noqa: E402
from essay_train import train_joint, train_indep_graded, train_indep_bt  # noqa: E402
from essay_metrics import compute_metrics, format_report  # noqa: E402

OUTPUT_DIR = Path(HERE) / "outputs"

# Per-set prompt topic (short, fed to the judge for context).
SET_PROMPT_TOPIC = {
    1: "Effects of computers on people and society (persuasive letter).",
    5: "Describe the mood created by the author in the memoir excerpt.",
}


def sample_pairs(pairwise_idx, n_comparisons, seed):
    """Sample unique unordered pairs uniformly from pairwise-eligible essays.

    Uniform sampling does NOT use human scores -> no leakage into pair choice.
    """
    rng = np.random.default_rng(seed + 7)
    idx = np.asarray(pairwise_idx)
    seen = set()
    pairs = []
    max_pairs = len(idx) * (len(idx) - 1) // 2
    n_target = min(n_comparisons, max_pairs)
    tries = 0
    while len(pairs) < n_target and tries < n_target * 50:
        i, j = rng.choice(idx, size=2, replace=False)
        i, j = int(i), int(j)
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append((i, j))
        tries += 1
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--essay-set", type=int, default=1)
    ap.add_argument("--n-essays", type=int, default=80)
    ap.add_argument("--n-comparisons", type=int, default=400)
    ap.add_argument("--judge-model", default="llama3.2:3b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=800)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--no-st", action="store_true",
                    help="force handcrafted features (skip sentence-transformers)")
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="truncate stored essay text (judge prompt caps separately)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("RQ1 ESSAY PILOT")
    print("=" * 70)

    # 1. Build split ------------------------------------------------------
    print(f"\n[1/5] Building split (set {args.essay_set}, n={args.n_essays}) ...",
          flush=True)
    split = build_split(
        essay_set=args.essay_set, n_essays=args.n_essays, seed=args.seed,
        prefer_st=not args.no_st, max_chars=args.max_chars,
    )
    print(f"  essays={split.n}  K={split.K}  features={split.feat_kind} "
          f"(dim={split.features.shape[1]})", flush=True)
    print(f"  graded-only={len(split.graded_only_idx)} "
          f"pairwise-only={len(split.pairwise_only_idx)} "
          f"both={len(split.both_idx)}", flush=True)
    print(f"  human score range observed: "
          f"{int(split.raw_scores.min())}-{int(split.raw_scores.max())}", flush=True)

    # 2. Sample pairs -----------------------------------------------------
    pairs = sample_pairs(split.pairwise_idx, args.n_comparisons, args.seed)
    print(f"\n[2/5] Sampled {len(pairs)} unique comparison pairs over "
          f"{len(split.pairwise_idx)} pairwise-eligible essays.", flush=True)

    # 3. Elicit -----------------------------------------------------------
    print(f"\n[3/5] Eliciting LLM judgments (judge={args.judge_model}) ...",
          flush=True)
    try:
        models = [m["name"] for m in list_models()]
    except Exception as e:  # noqa: BLE001
        print(f"BLOCKER: Ollama unreachable: {e}")
        sys.exit(1)
    if args.judge_model not in models:
        print(f"BLOCKER: judge model {args.judge_model} not in {models}")
        sys.exit(1)

    cache_path = OUTPUT_DIR / (
        f"elicit_set{args.essay_set}_n{args.n_essays}_"
        f"{args.judge_model.replace(':','_').replace('/','_')}_seed{args.seed}.json"
    )
    rng = np.random.default_rng(args.seed + 99)
    raw_results = elicit_comparisons(
        pairs=pairs, texts=split.texts,
        prompt_text=SET_PROMPT_TOPIC.get(args.essay_set, "Essay prompt."),
        model=args.judge_model, cache_path=cache_path, rng=rng, verbose=True,
    )
    comparisons = [(r["i"], r["j"], r["outcome"]) for r in raw_results
                   if r["outcome"] is not None]
    n_unparsed = len(raw_results) - len(comparisons)
    print(f"  parsed comparisons: {len(comparisons)}  (unparsed/error: {n_unparsed})",
          flush=True)
    # Judge-vs-human agreement on decided pairs (sanity on judge quality)
    dec = [(i, j, o) for (i, j, o) in comparisons
           if split.raw_scores[i] != split.raw_scores[j]]
    agree = sum(1 for i, j, o in dec
                if (split.raw_scores[i] > split.raw_scores[j]) == (o == 1))
    print(f"  judge-vs-human agreement on non-tied pairs: "
          f"{agree}/{len(dec)} = {agree/max(len(dec),1):.3f}", flush=True)

    if len(comparisons) < 20:
        print("BLOCKER: too few parsed comparisons to fit BT. Aborting.")
        sys.exit(1)

    # 4. Train ------------------------------------------------------------
    print(f"\n[4/5] Training joint + baselines ...", flush=True)
    joint = train_joint(split, comparisons, hidden_dim=args.hidden_dim,
                        n_epochs=args.n_epochs, seed=args.seed, verbose=True)
    indep_graded = train_indep_graded(split, hidden_dim=args.hidden_dim,
                                      n_epochs=args.n_epochs, seed=args.seed,
                                      verbose=True)
    indep_bt = train_indep_bt(split, comparisons, hidden_dim=args.hidden_dim,
                             n_epochs=args.n_epochs, seed=args.seed, verbose=True)

    joint_q = joint.get_quality().cpu().numpy()
    indep_graded_q = indep_graded.get_quality().cpu().numpy()
    indep_bt_q = indep_bt.get_quality().cpu().numpy()

    # 5. Metrics ----------------------------------------------------------
    print(f"\n[5/5] Computing metrics ...", flush=True)
    m = compute_metrics(split, comparisons, joint_q, indep_graded_q, indep_bt_q)
    params = {
        "essay_set": args.essay_set, "n_essays": split.n,
        "n_comparisons": len(comparisons), "judge_model": args.judge_model,
        "n_epochs": args.n_epochs, "seed": args.seed,
        "feat_kind": split.feat_kind, "judge_human_agreement":
            agree / max(len(dec), 1),
    }
    report = format_report(split, comparisons, m, params)
    print("\n" + report, flush=True)

    # Save
    (OUTPUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps({k: (round(float(v), 6) if v == v else None)
                    for k, v in m.items()}, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "params.json").write_text(json.dumps(params, indent=2),
                                            encoding="utf-8")
    np.savez(
        OUTPUT_DIR / "essay_arrays.npz",
        raw_scores=split.raw_scores, joint_q=joint_q,
        indep_graded_q=indep_graded_q, indep_bt_q=indep_bt_q,
        graded_only_idx=split.graded_only_idx,
        pairwise_only_idx=split.pairwise_only_idx, both_idx=split.both_idx,
    )
    print(f"\nOutputs -> {OUTPUT_DIR}/  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
