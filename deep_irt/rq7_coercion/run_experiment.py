"""run_experiment.py -- RQ7 COERCION-ROBUSTNESS, with built-in validation.

Question (real_data_agenda.md:79): does the learned ability/item scale survive
HOW raw behaviour is coerced into a response format?  I.e. is the scale RANK-
invariant to the arbitrary binning that turns raw behaviour (correctness +
response time) into a K-category format, or is "format-agnosticism" trivial
because the formats are just relabelings of the same data?

Defensible claim (NOT "coercion is free").  The collapsing-categories IRT
literature says coarsening LOSES scale information and precision -- coercion is
lossy.  So we test the weaker, defensible claim: the anchored learned scale stays
RANK-invariant across coercion schemes even as absolute precision drops on the
coarser bins.

Design.
 * ONE raw EdNet behaviour (correctness + RT) on a SHARED item/learner set.
 * Coerce it K=2 / K=3 / K=4 (data.py).  Bin cuts fit on TRAIN learners only.
 * Independently fit one ma-irt LSTMGPCM (separate_theta=True) per coercion, a
   DISTINCT seed each so no shared-init leak.
 * Read frozen per-item difficulty (item-only GPCM location) and per-learner
   ability (final-step item-blind theta) on each coercion's own scale.
 * PRIMARY METRIC: pairwise Spearman of item-difficulty rankings (K2-K3, K2-K4,
   K3-K4) and of ability rankings.  High => the scale survives coercion.

Validation (built in, so the real number is interpretable).
 * POSITIVE CONTROL (synthetic): one TRUE 1D IRT latent drives both correctness
   and RT; coerce the same way and confirm HIGH cross-coercion Spearman -> the
   achievable CEILING (proves the metric is not trivially low).
 * NEGATIVE CONTROL: a coercion that re-bins on a latent-IRRELEVANT random signal;
   confirm LOW cross-coercion Spearman -> the FLOOR (proves the metric can DETECT
   non-invariance, is not trivially high).
 * Interpret the REAL EdNet numbers BETWEEN these bounds, with the precision side
   (seed-to-seed reliability per coercion) reported to show the expected info-loss
   on coarser bins even when rank is preserved.

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="rl/src;ma-irt"
    python deep_irt/rq7_coercion/run_experiment.py

Env knobs (bounded runs): RQ7_N_LEARNERS, RQ7_EPOCHS, RQ7_MIN_ITEM_RESP,
RQ7_MAX_SEQ, RQ7_LR, RQ7_BATCH, RQ7_DMODEL, RQ7_SEED, RQ7_SYN_LEARNERS,
RQ7_SYN_ITEMS.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from deep_irt.rq7_coercion.data import (
    load_raw_ednet, coerce, scramble_coercion, make_synthetic_positive,
    RawBehaviour, CoercedData, _fit_cuts,
)
from deep_irt.rq7_coercion.model import (
    fit_coercion, item_difficulty, learner_ability, item_seen_counts,
)

OUTPUT_DIR = Path(os.environ.get(
    "RQ7_OUTPUT_DIR", str(Path(__file__).resolve().parent / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_LEARNERS = int(os.environ.get("RQ7_N_LEARNERS", "3000"))
EPOCHS = int(os.environ.get("RQ7_EPOCHS", "80"))
MIN_ITEM_RESP = int(os.environ.get("RQ7_MIN_ITEM_RESP", "50"))
MAX_SEQ = int(os.environ.get("RQ7_MAX_SEQ", "200"))
LR = float(os.environ.get("RQ7_LR", "0.005"))
BATCH = int(os.environ.get("RQ7_BATCH", "256"))
DMODEL = int(os.environ.get("RQ7_DMODEL", "64"))
SEED = int(os.environ.get("RQ7_SEED", "0"))
SYN_LEARNERS = int(os.environ.get("RQ7_SYN_LEARNERS", "2500"))
SYN_ITEMS = int(os.environ.get("RQ7_SYN_ITEMS", "700"))
MIN_ITEM_SCORE = int(os.environ.get("RQ7_MIN_ITEM_SCORE", "30"))  # min resp to score an item
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Distinct seed per coercion so the cross-coercion Spearman cannot be a
# shared-init artifact.  (label -> seed offset)
COERCION_SEED = {"K2": 0, "K3": 11, "K4": 22}
RELIABILITY_SEED_OFFSET = 500  # second seed for the seed-to-seed reliability fit


def tprint(*a, **k):
    print(*a, **k, flush=True)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(spearmanr(x, y).correlation)


# ---------------------------------------------------------------------------
# Fit one coercion and read its frozen item + learner scale.
# ---------------------------------------------------------------------------

def fit_and_read(
    coerced: CoercedData, n_items: int, seed: int, label: str,
) -> Dict:
    model, info = fit_coercion(
        coerced, n_items, d_model=DMODEL, key_dim=DMODEL, value_dim=DMODEL,
        n_epochs=EPOCHS, lr=LR, batch_size=BATCH, device=DEVICE, seed=seed,
        verbose=True,
    )
    b_loc, a_disc = item_difficulty(model, n_items, coerced=coerced,
                                    batch_size=BATCH, device=DEVICE)
    sids, theta = learner_ability(model, coerced, batch_size=BATCH, device=DEVICE)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "b_loc": b_loc, "a_disc": a_disc,
            "sids": sids, "theta": theta, "info": info}


def pairwise_item_spearman(
    reads: Dict[str, Dict], seen_mask: np.ndarray, labels: List[str],
) -> Dict[str, float]:
    """Pairwise Spearman of item-difficulty rankings over scored items."""
    out = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            out[f"{a}-{b}"] = _spearman(reads[a]["b_loc"][seen_mask],
                                        reads[b]["b_loc"][seen_mask])
    return out


def pairwise_ability_spearman(
    reads: Dict[str, Dict], labels: List[str],
) -> Dict[str, float]:
    """Pairwise Spearman of per-learner ability rankings.

    Aligns learners by student_id (the same learner set is fit under every
    coercion, but learner ORDER can differ if a coercion drops a learner; align
    on the intersection by id).
    """
    out = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            ra, rb = reads[a], reads[b]
            map_a = {int(s): t for s, t in zip(ra["sids"], ra["theta"])}
            map_b = {int(s): t for s, t in zip(rb["sids"], rb["theta"])}
            common = sorted(set(map_a) & set(map_b))
            xa = np.array([map_a[s] for s in common])
            xb = np.array([map_b[s] for s in common])
            out[f"{a}-{b}"] = _spearman(xa, xb)
    return out


# ---------------------------------------------------------------------------
# One full block (real / synthetic): fit K2/K3/K4, score invariance + reliability.
# ---------------------------------------------------------------------------

def run_block(
    raw: RawBehaviour, block_label: str, score_items_min: int,
    do_reliability: bool = True,
) -> Dict:
    n_items = raw.n_items
    cuts = _fit_cuts(raw)
    tprint(f"\n{'='*68}\nBLOCK [{block_label}]  n_items={n_items}  "
           f"n_learners={len(raw.records)}")
    tprint(f"  bin cuts (TRAIN only): median_correct={cuts['median_correct']:.1f} "
           f"median_wrong={cuts['median_wrong']:.1f}")

    coercions = {k: coerce(raw, K, cuts=cuts) for k, K in
                 (("K2", 2), ("K3", 3), ("K4", 4))}

    # Response-level distribution per coercion (audit).
    level_dist = {}
    for lab, cd in coercions.items():
        allr = np.concatenate([r["responses"] for r in cd.records])
        level_dist[lab] = {int(v): int((allr == v).sum()) for v in range(cd.K)}

    # Score on items with enough responses under EVERY coercion (same set across K).
    seen_counts = item_seen_counts(coercions["K4"], n_items)  # identical across K
    seen_mask = seen_counts >= score_items_min
    tprint(f"  scored items (>= {score_items_min} resp): {int(seen_mask.sum())}/{n_items}")

    labels = ["K2", "K3", "K4"]
    reads = {lab: fit_and_read(coercions[lab], n_items,
                               seed=SEED + COERCION_SEED[lab], label=lab)
             for lab in labels}

    item_inv = pairwise_item_spearman(reads, seen_mask, labels)
    abil_inv = pairwise_ability_spearman(reads, labels)

    # --- precision side: seed-to-seed reliability per coercion ---
    # Refit each coercion with a SECOND distinct seed; correlate item locations.
    # This is the within-coercion ceiling AND the precision/stability measure --
    # the coarser the bins, the lower the reliability the literature predicts.
    reliability = {}
    if do_reliability:
        for lab in labels:
            seed2 = SEED + COERCION_SEED[lab] + RELIABILITY_SEED_OFFSET
            r2 = fit_and_read(coercions[lab], n_items, seed=seed2, label=lab + "_s2")
            rel_item = _spearman(reads[lab]["b_loc"][seen_mask],
                                 r2["b_loc"][seen_mask])
            # ability reliability: align ids
            map1 = {int(s): t for s, t in zip(reads[lab]["sids"], reads[lab]["theta"])}
            map2 = {int(s): t for s, t in zip(r2["sids"], r2["theta"])}
            common = sorted(set(map1) & set(map2))
            rel_abil = _spearman(np.array([map1[s] for s in common]),
                                 np.array([map2[s] for s in common]))
            reliability[lab] = {"item": rel_item, "ability": rel_abil}
            tprint(f"  [{lab}] seed-to-seed reliability: item={rel_item:.3f} "
                   f"ability={rel_abil:.3f}")

    nlls = {lab: reads[lab]["info"]["final_nll"] for lab in labels}
    return {
        "block": block_label,
        "n_items": int(n_items),
        "n_learners": int(len(raw.records)),
        "n_scored_items": int(seen_mask.sum()),
        "cuts": cuts,
        "level_dist": level_dist,
        "item_invariance": item_inv,
        "ability_invariance": abil_inv,
        "reliability": reliability,
        "final_nll": nlls,
    }


def run_negative_block(raw: RawBehaviour, block_label: str,
                       score_items_min: int) -> Dict:
    """NEGATIVE control: bin on a latent-irrelevant random signal.

    Cross-coercion Spearman among scrambled coercions (and a real K2 vs a
    scrambled K3/K4) should collapse to ~0 -> the FLOOR.
    """
    n_items = raw.n_items
    cuts = _fit_cuts(raw)
    tprint(f"\n{'='*68}\nNEGATIVE CONTROL [{block_label}]  (random-signal bins)")

    real_k2 = coerce(raw, 2, cuts=cuts)
    scram = {f"K{K}s": scramble_coercion(raw, K, seed=K) for K in (2, 3, 4)}

    seen_counts = item_seen_counts(real_k2, n_items)
    seen_mask = seen_counts >= score_items_min

    # Fit real K2 (genuine scale) + the three scrambles.
    reads = {"K2": fit_and_read(real_k2, n_items, seed=SEED + COERCION_SEED["K2"],
                                label="K2")}
    for lab, cd in scram.items():
        reads[lab] = fit_and_read(cd, n_items, seed=SEED + hash(lab) % 97 + 300,
                                  label=lab)

    labels = ["K2", "K2s", "K3s", "K4s"]
    item_inv = pairwise_item_spearman(reads, seen_mask, labels)
    abil_inv = pairwise_ability_spearman(reads, labels)
    return {
        "block": block_label,
        "n_scored_items": int(seen_mask.sum()),
        "item_invariance": item_inv,
        "ability_invariance": abil_inv,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt_pairwise(d: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items())


def mean_finite(d: Dict[str, float]) -> float:
    vals = [v for v in d.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def write_report(results: Dict):
    lines: List[str] = []
    A = lines.append
    A("=" * 70)
    A("RQ7  COERCION-ROBUSTNESS  --  does the learned scale survive coercion?")
    A("=" * 70)
    A(f"Engine: ma-irt LSTMGPCM (separate_theta=True), item-only difficulty +")
    A(f"final-step ability readouts.  Distinct seed per coercion (no init leak).")
    A(f"Config: N_LEARNERS={N_LEARNERS} EPOCHS={EPOCHS} MIN_ITEM_RESP={MIN_ITEM_RESP}")
    A(f"        MAX_SEQ={MAX_SEQ} LR={LR} D_MODEL={DMODEL} DEVICE={DEVICE}")
    A("")

    pos = results.get("positive_control", {})
    neg = results.get("negative_control", {})
    real = results.get("real", {})

    A("-" * 70)
    A("PRIMARY: pairwise cross-coercion Spearman (ITEM DIFFICULTY ranking)")
    A("-" * 70)
    A(f"{'pair':<10}{'REAL EdNet':>14}{'POS ceiling':>14}{'NEG floor':>14}")
    real_it = real.get("item_invariance", {})
    pos_it = pos.get("item_invariance", {})
    neg_it = neg.get("item_invariance", {})
    for pair in ["K2-K3", "K2-K4", "K3-K4"]:
        rv = real_it.get(pair, float("nan"))
        pv = pos_it.get(pair, float("nan"))
        A(f"{pair:<10}{rv:>14.3f}{pv:>14.3f}{'--':>14}")
    A(f"{'mean':<10}{mean_finite(real_it):>14.3f}{mean_finite(pos_it):>14.3f}"
      f"{mean_finite(neg_it):>14.3f}")
    A(f"(neg-floor pairs: {fmt_pairwise(neg_it)})")
    A("")

    A("-" * 70)
    A("PRIMARY: pairwise cross-coercion Spearman (LEARNER ABILITY ranking)")
    A("-" * 70)
    A(f"{'pair':<10}{'REAL EdNet':>14}{'POS ceiling':>14}")
    real_ab = real.get("ability_invariance", {})
    pos_ab = pos.get("ability_invariance", {})
    neg_ab = neg.get("ability_invariance", {})
    for pair in ["K2-K3", "K2-K4", "K3-K4"]:
        A(f"{pair:<10}{real_ab.get(pair, float('nan')):>14.3f}"
          f"{pos_ab.get(pair, float('nan')):>14.3f}")
    A(f"{'mean':<10}{mean_finite(real_ab):>14.3f}{mean_finite(pos_ab):>14.3f}")
    A(f"(neg-floor ability pairs: {fmt_pairwise(neg_ab)})")
    A("")

    A("-" * 70)
    A("PRECISION SIDE: seed-to-seed reliability per coercion (within-K ceiling)")
    A("  (the lossy-coarsening check: coarser bins should be LESS reliable)")
    A("-" * 70)
    rel = real.get("reliability", {})
    A(f"{'coercion':<10}{'item rel':>12}{'ability rel':>14}{'NLL':>10}")
    nll = real.get("final_nll", {})
    for lab in ["K2", "K3", "K4"]:
        r = rel.get(lab, {})
        A(f"{lab:<10}{r.get('item', float('nan')):>12.3f}"
          f"{r.get('ability', float('nan')):>14.3f}{nll.get(lab, float('nan')):>10.3f}")
    A("")

    A("-" * 70)
    A("RESPONSE-LEVEL DISTRIBUTION (real EdNet, audit)")
    A("-" * 70)
    for lab, dist in real.get("level_dist", {}).items():
        A(f"  {lab}: {dist}")
    A(f"  bin cuts (ms, TRAIN only): {real.get('cuts', {})}")
    A(f"  scored items: {real.get('n_scored_items')}  learners: {real.get('n_learners')}")
    A("")

    # Verdict.
    A("=" * 70)
    A("VERDICT")
    A("=" * 70)
    rmean = mean_finite(real_it)
    pmean = mean_finite(pos_it)
    nmean = mean_finite(neg_it)
    if np.isfinite(rmean) and np.isfinite(pmean) and np.isfinite(nmean):
        denom = pmean - nmean
        placement = (rmean - nmean) / denom if abs(denom) > 1e-6 else float("nan")
        A(f"Item-difficulty invariance: REAL mean {rmean:.3f}, between NEG floor "
          f"{nmean:.3f} and POS ceiling {pmean:.3f}.")
        A(f"Placement in [floor, ceiling]: {placement:.2f} "
          f"(1.0 = at ceiling, 0.0 = at floor).")
    A("See report body for the candid interpretation.")

    report = "\n".join(lines)
    (OUTPUT_DIR / "report.txt").write_text(report)
    tprint("\n" + report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = OUTPUT_DIR / "run.log"
    log_fh = open(log_path, "w", buffering=1)
    orig_stdout = sys.stdout
    sys.stdout = log_fh
    try:
        t0 = time.time()
        tprint("=" * 68)
        tprint("RQ7 COERCION-ROBUSTNESS")
        tprint("=" * 68)
        tprint(f"Config: N_LEARNERS={N_LEARNERS} EPOCHS={EPOCHS} "
               f"MIN_ITEM_RESP={MIN_ITEM_RESP} MAX_SEQ={MAX_SEQ} LR={LR} "
               f"D_MODEL={DMODEL} SEED={SEED} DEVICE={DEVICE}")

        results: Dict = {}

        # ---- POSITIVE CONTROL (synthetic, one true latent) ----
        tprint("\n##### POSITIVE CONTROL (synthetic, one true 1D IRT latent) #####")
        syn_raw, true_theta, true_b = make_synthetic_positive(
            n_learners=SYN_LEARNERS, n_items=SYN_ITEMS, seq_len=120,
            train_frac=0.5, seed=SEED,
        )
        results["positive_control"] = run_block(
            syn_raw, "POSITIVE(synthetic)", score_items_min=MIN_ITEM_SCORE,
            do_reliability=False,
        )
        # Sanity: recovered K2 difficulty vs TRUE b (should be strong).
        # (Re-fit not needed; recovered numbers are in the block's invariance.)

        # ---- REAL EdNet ----
        tprint("\n##### REAL EdNet #####")
        raw = load_raw_ednet(
            n_learners=N_LEARNERS, min_item_responses=MIN_ITEM_RESP,
            min_seq_len=10, max_seq_len=MAX_SEQ, train_frac=0.5, seed=SEED,
        )
        results["real"] = run_block(
            raw, "REAL(EdNet)", score_items_min=MIN_ITEM_SCORE,
            do_reliability=True,
        )

        # ---- NEGATIVE CONTROL (random-signal bins on the SAME raw) ----
        tprint("\n##### NEGATIVE CONTROL (random-signal bins) #####")
        results["negative_control"] = run_negative_block(
            raw, "NEGATIVE(random-bins)", score_items_min=MIN_ITEM_SCORE,
        )

        results["wall_clock_total"] = time.time() - t0

        # Persist JSON (floats only).
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(_clean(results), f, indent=2)

        write_report(results)
        tprint(f"\nTotal wall clock: {results['wall_clock_total']:.1f}s")
    finally:
        sys.stdout = orig_stdout
        log_fh.close()
    # Echo the report to the real stdout for the caller.
    print((OUTPUT_DIR / "report.txt").read_text())


if __name__ == "__main__":
    main()
