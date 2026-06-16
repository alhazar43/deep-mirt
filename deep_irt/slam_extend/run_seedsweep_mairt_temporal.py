"""
run_seedsweep_mairt_temporal.py -- RQ2 reliability-ceiling precision cleanup.

Why
---
run_experiment_mairt_temporal.py reports anchored new-item recovery as a
FRACTION of a full-recalibration seed-to-seed reliability CEILING, but it
estimates that ceiling from only TWO full-recal seeds (RECAL_SEEDS=(0,1)).
A 2-seed ceiling has wide sampling error, so the headline fractions
(difficulty 0.910x, discrimination 0.843x) are normalised by a noisy
denominator. This script re-estimates the ceiling from N=12-20 full-recal
seeds, propagates the uncertainty into the fraction, and reports whether the
headline holds.

No new method. SAME setup as the convergence run:
  - full learners (MAIRT_SUBSAMPLE=0), RANDOM split (ITEM_SPLIT_SEED=42),
  - gold-standard temporal item-only readout (recover_full over real
    interleaved base+new sequences),
  - base 80 / ext 300 / recal 80 epochs.

What changes vs run_experiment_mairt_temporal.py
------------------------------------------------
1. BASE FIT once (seed 0). ANCHORED EXTENSION once -- it is recal-seed
   INDEPENDENT (assemble_anchored_model uses seed=0 and the fit RNG is a
   fixed manual_seed(123)). So the anchored params are computed a single time.
2. FULL RECAL run N=MAIRT_RECAL_NSEEDS times (seeds 0..N-1). Each recal's
   recovered new-item (a, b_mean, seen) is cached to a .npz so a kill resumes.
3. Statistics:
   (a) CEILING distribution: across all C(N,2) distinct seed PAIRS, the
       seed-to-seed Spearman of item difficulty and of discrimination ->
       mean, sd, and a bootstrap 95% CI over the N recal fits.
   (b) ANCHORED RECOVERY: anchored vs each recal, Spearman, mean +/- sd over N.
   (c) FRACTION = anchored / ceiling, difficulty and discrimination, with a
       bootstrap 95% CI propagated from the N-recal seed distribution.

The 2-seed path is preserved verbatim in run_experiment_mairt_temporal.py;
this script is fully additive (ma-irt frozen). Runs INLINE / FOREGROUND / GPU.

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="rl/src;ma-irt"
    MAIRT_SUBSAMPLE=0 MAIRT_BASE_EPOCHS=80 MAIRT_EXT_EPOCHS=300 \
      MAIRT_RECAL_EPOCHS=80 MAIRT_RECAL_NSEEDS=16 \
      python deep_irt/slam_extend/run_seedsweep_mairt_temporal.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: repo root, rl/src, ma-irt importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Force the seed-sweep output dir BEFORE importing the parent (its OUTPUT_DIR is
# read at import time). Do not clobber outputs_mairt_temporal*/.
DEFAULT_OUT = str(Path(__file__).resolve().parent / "outputs_mairt_seedsweep")
os.environ.setdefault("MAIRT_OUTPUT_DIR", DEFAULT_OUT)

# Reuse the gold-standard temporal anchored fit + the parent data/model layer.
from deep_irt.slam_extend.run_experiment_mairt_temporal import (  # noqa: E402
    run_anchored_extension_temporal,
)
from deep_irt.slam_extend.run_experiment_mairt import (  # noqa: E402
    tprint,
    load_slam,
    get_catchall_ids,
    get_n_named,
    filter_catchall,
    make_item_split,
    subsequence,
    remap_ids_to_local,
    spearman,
    mean_b,
    ITEM_SPLIT_SEED,
    BASE_EPOCHS,
    EXT_EPOCHS,
    RECAL_EPOCHS,
    KEY_DIM,
    DEVICE,
    train_full,
    recover_full,
)

import torch  # noqa: E402

OUTPUT_DIR = Path(os.environ["MAIRT_OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Number of full-recal seeds for the tighter ceiling (additive knob; the
# original 2-seed path lives in run_experiment_mairt_temporal.py).
N_SEEDS = int(os.environ.get("MAIRT_RECAL_NSEEDS", "16"))
N_BOOT = int(os.environ.get("MAIRT_NBOOT", "10000"))
BOOT_SEED = int(os.environ.get("MAIRT_BOOT_SEED", "12345"))

# Old 2-seed headline (from outputs_mairt_temporal_conv/results.json, RANDOM).
OLD_2SEED = {
    "ceiling_difficulty": 0.7863167772729661,
    "ceiling_discrimination": 0.6978554390065117,
    "anchored_difficulty": 0.7154016233341142,
    "anchored_discrimination": 0.5880340770126234,
    "frac_difficulty": 0.9098135052074132,
    "frac_discrimination": 0.8426302127124877,
}


# ---------------------------------------------------------------------------
# Per-seed full-recal: train + recover new-item (a, b_mean, seen). Cached.
# ---------------------------------------------------------------------------

def recal_for_seed(seed: int, all_local: List[Dict], n_named: int,
                   new_local_in_all: np.ndarray) -> dict:
    """Train a full-recal model at `seed`, recover new-item params, cache, and
    return {"a": (n_new,), "b_mean": (n_new,), "seen": (n_new,) bool}."""
    cache = CACHE_DIR / f"recal_seed{seed}.npz"
    if cache.exists():
        d = np.load(cache)
        tprint(f"  [Recal s{seed}] loaded cache {cache.name}")
        return {"a": d["a"], "b_mean": d["b_mean"], "seen": d["seen"]}

    m, info = train_full(all_local, n_named, seed=seed,
                         n_epochs=RECAL_EPOCHS, label=f"FullRecal_s{seed}")
    a_all, b_all, seen_all = recover_full(m, all_local, n_named)
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = {
        "a": a_all[new_local_in_all].astype(np.float64),
        "b_mean": mean_b(b_all[new_local_in_all]).astype(np.float64),
        "seen": seen_all[new_local_in_all].astype(bool),
    }
    np.savez(cache, a=out["a"], b_mean=out["b_mean"], seen=out["seen"],
             wall_clock=float(info["wall_clock"]))
    return out


# ---------------------------------------------------------------------------
# Statistics: ceiling distribution, anchored recovery, fraction + bootstrap CI.
# ---------------------------------------------------------------------------

def _pct_ci(x: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> Tuple[float, float]:
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def compute_stats(recals: List[dict], anc: dict, seen_common: np.ndarray) -> dict:
    """recals: list of {"a","b_mean","seen"} (new-item, in new_local_in_all order).
    anc: {"a","b_mean","seen"}. seen_common: bool mask of items kept in all fits.

    Returns the full stats block (ceiling/anchored/fraction with CIs)."""
    N = len(recals)
    s = seen_common

    # Stack recal params over seeds on the common item set: (N, M).
    A = np.stack([r["a"][s] for r in recals], axis=0)       # discrimination
    B = np.stack([r["b_mean"][s] for r in recals], axis=0)  # difficulty
    a_anc = anc["a"][s]
    b_anc = anc["b_mean"][s]

    # (a) CEILING -- seed-to-seed Spearman over all C(N,2) distinct pairs.
    pair_diff: List[float] = []
    pair_disc: List[float] = []
    for i, j in combinations(range(N), 2):
        pair_diff.append(spearman(B[i], B[j]))
        pair_disc.append(spearman(A[i], A[j]))
    pair_diff = np.asarray(pair_diff)
    pair_disc = np.asarray(pair_disc)

    # (b) ANCHORED RECOVERY -- anchored vs each recal, Spearman, over N.
    anc_diff = np.asarray([spearman(b_anc, B[i]) for i in range(N)])
    anc_disc = np.asarray([spearman(a_anc, A[i]) for i in range(N)])

    # (c) BOOTSTRAP over the N recal fits (the independent unit of variation).
    #     Resample seeds WITH replacement, and on each resample recompute BOTH
    #     the mean pairwise ceiling and the mean anchored-vs-recal, then the
    #     fraction. This propagates seed-sampling error into the fraction CI.
    rng = np.random.default_rng(BOOT_SEED)
    boot_ceil_diff = np.empty(N_BOOT)
    boot_ceil_disc = np.empty(N_BOOT)
    boot_anc_diff = np.empty(N_BOOT)
    boot_anc_disc = np.empty(N_BOOT)
    boot_frac_diff = np.empty(N_BOOT)
    boot_frac_disc = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, N, size=N)
        # Ceiling on the resample: mean Spearman over all distinct ordered-unique
        # pairs of the (possibly repeated) drawn seeds. Skip i==j (self-pair has
        # Spearman 1.0 and would inflate the ceiling); require >=2 distinct.
        uniq = np.unique(idx)
        if uniq.size >= 2:
            cd = [spearman(B[i], B[j]) for i, j in combinations(uniq, 2)]
            cc = [spearman(A[i], A[j]) for i, j in combinations(uniq, 2)]
            ceil_d = float(np.mean(cd)); ceil_c = float(np.mean(cc))
        else:
            ceil_d = np.nan; ceil_c = np.nan
        anc_d = float(np.mean([spearman(b_anc, B[i]) for i in idx]))
        anc_c = float(np.mean([spearman(a_anc, A[i]) for i in idx]))
        boot_ceil_diff[b] = ceil_d
        boot_ceil_disc[b] = ceil_c
        boot_anc_diff[b] = anc_d
        boot_anc_disc[b] = anc_c
        boot_frac_diff[b] = anc_d / ceil_d if abs(ceil_d) > 1e-9 else np.nan
        boot_frac_disc[b] = anc_c / ceil_c if abs(ceil_c) > 1e-9 else np.nan

    # Point estimates: full-sample means (not bootstrap means).
    ceil_diff_mean = float(pair_diff.mean())
    ceil_disc_mean = float(pair_disc.mean())
    anc_diff_mean = float(anc_diff.mean())
    anc_disc_mean = float(anc_disc.mean())
    frac_diff = anc_diff_mean / ceil_diff_mean
    frac_disc = anc_disc_mean / ceil_disc_mean

    def block(point, samp_sd, boot):
        boot = boot[~np.isnan(boot)]
        lo, hi = _pct_ci(boot)
        return {"mean": float(point), "sd": float(samp_sd),
                "ci95_lo": lo, "ci95_hi": hi, "boot_sd": float(boot.std(ddof=1))}

    return {
        "n_seeds": N,
        "n_pairs": int(len(pair_diff)),
        "n_items_common": int(s.sum()),
        "ceiling": {
            "difficulty": {**block(ceil_diff_mean, pair_diff.std(ddof=1), boot_ceil_diff),
                           "pair_min": float(pair_diff.min()),
                           "pair_max": float(pair_diff.max())},
            "discrimination": {**block(ceil_disc_mean, pair_disc.std(ddof=1), boot_ceil_disc),
                               "pair_min": float(pair_disc.min()),
                               "pair_max": float(pair_disc.max())},
        },
        "anchored": {
            "difficulty": block(anc_diff_mean, anc_diff.std(ddof=1), boot_anc_diff),
            "discrimination": block(anc_disc_mean, anc_disc.std(ddof=1), boot_anc_disc),
        },
        "fraction": {
            "difficulty": {"mean": frac_diff,
                           **{k: v for k, v in
                              zip(("ci95_lo", "ci95_hi"), _pct_ci(boot_frac_diff[~np.isnan(boot_frac_diff)]))},
                           "boot_sd": float(np.nanstd(boot_frac_diff, ddof=1))},
            "discrimination": {"mean": frac_disc,
                               **{k: v for k, v in
                                  zip(("ci95_lo", "ci95_hi"), _pct_ci(boot_frac_disc[~np.isnan(boot_frac_disc)]))},
                               "boot_sd": float(np.nanstd(boot_frac_disc, ddof=1))},
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = OUTPUT_DIR / "run.log"
    log_fh = open(log_path, "w", buffering=1)
    sys.stdout = log_fh
    tprint("=" * 70)
    tprint("RQ2 [ma-irt TEMPORAL]: TIGHTER RELIABILITY-CEILING (N-seed sweep)")
    tprint("=" * 70)
    tprint(f"Config: KEY_DIM={KEY_DIM} BASE_EP={BASE_EPOCHS} EXT_EP={EXT_EPOCHS} "
           f"RECAL_EP={RECAL_EPOCHS} N_SEEDS={N_SEEDS} DEVICE={DEVICE}")
    tprint(f"Bootstrap: N_BOOT={N_BOOT} BOOT_SEED={BOOT_SEED}")

    # --- Data layer: identical to the convergence run (full learners, random). ---
    adapter = load_slam()
    catchall_ids = get_catchall_ids(adapter)
    n_named = get_n_named(adapter)
    full_records = filter_catchall(adapter, catchall_ids, split="train")
    tprint(f"[Data] n_named={n_named}, learners(train,>=5)={len(full_records)}")

    SUBSAMPLE = int(os.environ.get("MAIRT_SUBSAMPLE", "1000"))
    if SUBSAMPLE and SUBSAMPLE < len(full_records):
        rng = np.random.default_rng(7)
        keep = rng.permutation(len(full_records))[:SUBSAMPLE]
        full_records = [full_records[i] for i in sorted(keep.tolist())]
        tprint(f"[Data] SUBSAMPLED to {len(full_records)} learners (seed=7)")
    else:
        tprint(f"[Data] FULL learners (no subsample).")

    base_ids, new_ids = make_item_split(n_named, seed=ITEM_SPLIT_SEED)
    n_base = len(base_ids); n_new = len(new_ids)
    tprint(f"[Split/random] BASE={n_base}, NEW={n_new}")

    base_set = set(base_ids.tolist()); new_set = set(new_ids.tolist())
    base_seqs = subsequence(full_records, base_set, min_events=2)
    new_seqs = subsequence(full_records, new_set, min_events=2)
    shared_sid = {r["student_id"] for r in base_seqs} & {r["student_id"] for r in new_seqs}
    tprint(f"  Shared learners (base & new): {len(shared_sid)}")

    base_seqs_shared = sorted([r for r in base_seqs if r["student_id"] in shared_sid],
                              key=lambda r: r["student_id"])
    base_local = remap_ids_to_local(base_seqs_shared, base_ids)

    all_named_ids = np.sort(np.concatenate([base_ids, new_ids]))
    all_seqs = subsequence(full_records, set(all_named_ids.tolist()), min_events=2)
    all_local = remap_ids_to_local(all_seqs, all_named_ids)
    all_g2l = {int(g): i for i, g in enumerate(all_named_ids)}
    new_local_in_all = np.array([all_g2l[int(g)] for g in new_ids], dtype=np.int64)
    new_set_local0 = set(int(x) for x in new_local_in_all.tolist())

    # --- ANCHORED EXTENSION (once -- recal-seed independent). The BASE FIT is
    #     only needed to produce it, so a resume with anchored.npz present skips
    #     the base fit entirely (saves ~50s). ---
    anc_cache = CACHE_DIR / "anchored.npz"
    if anc_cache.exists():
        d = np.load(anc_cache)
        anc_new = {"a": d["a"], "b_mean": d["b_mean"], "seen": d["seen"]}
        tprint(f"[Anchored] loaded cache {anc_cache.name} (base fit skipped)")
    else:
        # BASE FIT (once, seed 0) -- feeds the anchored extension only.
        base_model, _ = train_full(base_local, n_base, seed=0,
                                   n_epochs=BASE_EPOCHS, label="BaseFit")
        anc = run_anchored_extension_temporal(
            base_model, n_named, base_ids, new_ids, all_named_ids,
            all_local, new_set_local0)
        a_anc_all = anc["a_all"]; b_anc_all = anc["b_all"]
        anc_new = {
            "a": a_anc_all[new_local_in_all].astype(np.float64),
            "b_mean": mean_b(b_anc_all[new_local_in_all]).astype(np.float64),
            "seen": anc["seen_all"][new_local_in_all].astype(bool),
        }
        np.savez(anc_cache, a=anc_new["a"], b_mean=anc_new["b_mean"],
                 seen=anc_new["seen"])
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- FULL RECAL x N SEEDS (cached per seed; resumable) ---
    tprint(f"\n{'='*70}\nFULL RECAL SWEEP: {N_SEEDS} seeds (0..{N_SEEDS-1})\n{'='*70}")
    recals: List[dict] = []
    t_sweep = time.time()
    for seed in range(N_SEEDS):
        recals.append(recal_for_seed(seed, all_local, n_named, new_local_in_all))
    tprint(f"[Sweep] {N_SEEDS} recals done in {time.time()-t_sweep:.1f}s "
           f"(incl. cache hits).")

    # --- Common item set: seen in anchored AND every recal ---
    seen_common = anc_new["seen"].copy()
    for r in recals:
        seen_common &= r["seen"]
    tprint(f"[Coverage] new items seen in anchored + all {N_SEEDS} recals: "
           f"{int(seen_common.sum())}/{n_new}")

    # --- Statistics ---
    stats = compute_stats(recals, anc_new, seen_common)

    # --- Persist ---
    payload = {
        "split_label": "random",
        "n_named": int(n_named), "n_base": int(n_base), "n_new": int(n_new),
        "n_shared_learners": int(len(shared_sid)),
        "config": {"base_epochs": BASE_EPOCHS, "ext_epochs": EXT_EPOCHS,
                   "recal_epochs": RECAL_EPOCHS, "n_seeds": N_SEEDS,
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED},
        "old_2seed": OLD_2SEED,
        "stats": stats,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    # --- Report ---
    cd = stats["ceiling"]["difficulty"]; cc = stats["ceiling"]["discrimination"]
    ad = stats["anchored"]["difficulty"]; ac = stats["anchored"]["discrimination"]
    fd = stats["fraction"]["difficulty"]; fc = stats["fraction"]["discrimination"]
    tprint("\n" + "=" * 70)
    tprint(f"TIGHTER CEILING (N={N_SEEDS} seeds, {stats['n_pairs']} pairs, "
           f"{stats['n_items_common']} items)")
    tprint("=" * 70)
    tprint(f"CEILING difficulty:      {cd['mean']:.4f}  "
           f"sd(pairs)={cd['sd']:.4f}  95%CI[{cd['ci95_lo']:.4f},{cd['ci95_hi']:.4f}]  "
           f"pair-range[{cd['pair_min']:.4f},{cd['pair_max']:.4f}]")
    tprint(f"  old 2-seed ceiling:    {OLD_2SEED['ceiling_difficulty']:.4f}")
    tprint(f"CEILING discrimination:  {cc['mean']:.4f}  "
           f"sd(pairs)={cc['sd']:.4f}  95%CI[{cc['ci95_lo']:.4f},{cc['ci95_hi']:.4f}]  "
           f"pair-range[{cc['pair_min']:.4f},{cc['pair_max']:.4f}]")
    tprint(f"  old 2-seed ceiling:    {OLD_2SEED['ceiling_discrimination']:.4f}")
    tprint("")
    tprint(f"ANCHORED difficulty:     {ad['mean']:.4f}  sd={ad['sd']:.4f}  "
           f"95%CI[{ad['ci95_lo']:.4f},{ad['ci95_hi']:.4f}]")
    tprint(f"  old 2-seed anchored:   {OLD_2SEED['anchored_difficulty']:.4f}")
    tprint(f"ANCHORED discrimination: {ac['mean']:.4f}  sd={ac['sd']:.4f}  "
           f"95%CI[{ac['ci95_lo']:.4f},{ac['ci95_hi']:.4f}]")
    tprint(f"  old 2-seed anchored:   {OLD_2SEED['anchored_discrimination']:.4f}")
    tprint("")
    tprint(f"FRACTION difficulty:     {fd['mean']:.4f}  "
           f"95%CI[{fd['ci95_lo']:.4f},{fd['ci95_hi']:.4f}]   "
           f"(old 2-seed {OLD_2SEED['frac_difficulty']:.4f})")
    tprint(f"FRACTION discrimination: {fc['mean']:.4f}  "
           f"95%CI[{fc['ci95_lo']:.4f},{fc['ci95_hi']:.4f}]   "
           f"(old 2-seed {OLD_2SEED['frac_discrimination']:.4f})")
    tprint(f"\nOutputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
