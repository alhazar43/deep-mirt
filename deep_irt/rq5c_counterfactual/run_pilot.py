"""run_pilot.py -- RQ5c COUNTERFACTUAL-RESPONDENT bounded pilot.

The most original design in the battery. Human data cannot produce a respondent
whose TRUE ability is held constant across two instruments, but a FIXED LLM can.
So any gap between a fixed LLM's ability estimated on bank A vs bank B is PURE
instrument effect. This tests whether the learned/anchored scale SEPARATES
ability from instrument.

Pipeline
--------
1. Build two instrument banks (two text-amenable SLAM formats):
       Bank A = reverse_translate, Bank B = reverse_tap.
   Item difficulties come from the SAME human calibration source
   (human_difficulty.csv, population token mistake rate).
2. A pool of fixed LLM respondents. Each answers EVERY item of BOTH banks, so
   its TRUE theta is constant by construction. (Optionally widen the pool with
   cheap temperature variants of a base model.)
3. Grade by token overlap; reduce to binary correctness (full overlap) for the
   anchored Rasch scoring.
4. Score each respondent's ability per bank by ANCHORED fixed-parameter Rasch
   estimation: item difficulties FIXED at logit(human mistake rate), estimate
   the scalar theta (MAP). Raw proportion-correct reported as a backup.
5. PRIMARY METRIC: Spearman(theta_A, theta_B) across the pool. HIGH => the scale
   is instrument-SEPARABLE. Also report the paired LEVEL shift (mean theta_A vs
   theta_B): order-separability and level-separability are distinct claims.
6. VALIDATION CONTROL: within-bank random-half split-half reliability per bank.
   Cross-bank consistency is read AGAINST this floor.

Safety: all Ollama calls go through elicit.call_ollama (HTTP API, num_ctx=2048,
45s hard per-call timeout, incremental disk persistence + run.log). Bounded:
small model pool x ~100 items x 2 banks.

Usage (from repo root, env activated)::

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="rl/src;ma-irt" \\
        python deep_irt/rq5c_counterfactual/run_pilot.py \\
        --n-items 100 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anchored_irt import (
    estimate_theta_rasch,
    mistake_rate_to_difficulty,
    proportion_correct,
    spearman,
    pearson,
    split_half_reliability,
)
from banks import build_bank
from elicit import elicit_model_on_bank, list_models

# Bank A / Bank B = two text-amenable SLAM formats (the two instruments).
BANK_A_FORMAT = "reverse_translate"
BANK_B_FORMAT = "reverse_tap"

# Proven pool that runs reliably on this box, spanning ability (see slam_pilot
# outputs_diverse: mean error 0.13-0.32). Smallest/weakest first for spread.
DEFAULT_POOL = [
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen3.5:2B",
    "phi4-mini:3.8b",
    "gemma4:e2b",
]

# Optional cheap temperature-variant respondents to widen N (base model, temp,
# seed). Same model at higher temperature is a different (noisier) respondent.
TEMP_VARIANTS: List[Tuple[str, float, int]] = [
    ("llama3.2:3b", 0.7, 11),
    ("qwen3.5:2B", 0.7, 11),
]

# Binary-correct threshold for the Rasch scoring. 1.0 = reproduce ALL reference
# tokens (matches SLAM all-correct). A softer value is a sensitivity knob.
CORRECT_THRESHOLD = 1.0


def _binarize(overlap: float, threshold: float = CORRECT_THRESHOLD) -> int:
    return 1 if overlap >= threshold - 1e-9 else 0


def score_bank(
    responses: Dict[str, Dict[str, Any]],
    items: List[Dict[str, Any]],
    item_hashes: List[str],
    prior_sd: float,
    threshold: float = CORRECT_THRESHOLD,
) -> Tuple[float, float, List[int], List[float]]:
    """Anchored theta + proportion-correct for one respondent on one bank.

    Returns (theta, prop_correct, correct_vector, difficulty_vector) restricted
    to ``item_hashes`` (an ordered subset, e.g. a random half).
    """
    diff_by_hash = {it["item_hash"]: it["human_difficulty"] for it in items}
    correct: List[int] = []
    diffs: List[float] = []
    for h in item_hashes:
        if h not in responses:
            continue
        correct.append(_binarize(responses[h]["overlap"], threshold))
        diffs.append(mistake_rate_to_difficulty(diff_by_hash[h]))
    theta = estimate_theta_rasch(correct, diffs, prior_sd=prior_sd)
    pc = proportion_correct(correct)
    return theta, pc, correct, diffs


def run(
    human_csv: Path,
    raw_dir: Path,
    out_dir: Path,
    n_items: int,
    seed: int,
    prior_sd: float,
    pool: List[str],
    use_temp_variants: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / f"cache_seed{seed}_n{n_items}"
    log_path = out_dir / "run.log"
    t_run = time.time()

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== RUN {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"n_items={n_items} seed={seed} prior_sd={prior_sd} ===\n")

    # --- Ollama reachability + pool validation -----------------------------
    print("\n=== Checking Ollama + model pool ===", flush=True)
    try:
        avail = {m["name"] for m in list_models()}
    except Exception as e:  # noqa: BLE001
        print(f"BLOCKER: cannot reach Ollama: {e}", flush=True)
        return
    pool = [m for m in pool if m in avail]
    if not pool:
        print("BLOCKER: none of the requested pool models are available.", flush=True)
        return
    print(f"Pool ({len(pool)}): {pool}", flush=True)

    respondents: List[Tuple[str, Optional[float], int, str]] = [
        (m, None, 0, m) for m in pool
    ]
    if use_temp_variants:
        for base, temp, vseed in TEMP_VARIANTS:
            if base in avail:
                respondents.append((base, temp, vseed, f"{base}@t{temp:g}"))
    print(f"Respondents ({len(respondents)}): "
          f"{[r[3] for r in respondents]}", flush=True)

    # --- Build the two banks ----------------------------------------------
    print(f"\n=== Building Bank A ({BANK_A_FORMAT}) ===", flush=True)
    bank_a = build_bank(human_csv, raw_dir, BANK_A_FORMAT, n_items, seed)
    print(f"\n=== Building Bank B ({BANK_B_FORMAT}) ===", flush=True)
    bank_b = build_bank(human_csv, raw_dir, BANK_B_FORMAT, n_items, seed)
    if not bank_a or not bank_b:
        print("BLOCKER: a bank came back empty.", flush=True)
        return

    hashes_a = [it["item_hash"] for it in bank_a]
    hashes_b = [it["item_hash"] for it in bank_b]
    total_calls = len(respondents) * (len(bank_a) + len(bank_b))
    print(f"\nBank A: {len(bank_a)} items | Bank B: {len(bank_b)} items | "
          f"{len(respondents)} respondents | <= {total_calls} calls", flush=True)

    # --- Elicit every respondent on both banks ----------------------------
    print("\n=== Elicitation ===", flush=True)
    # responses[tag][bank] = {item_hash: {response, overlap, error}}
    responses: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model_name, temp, vseed, tag in respondents:
        responses[tag] = {}
        for bank_tag, items in (("A", bank_a), ("B", bank_b)):
            print(f"  {tag} on Bank {bank_tag}...", flush=True)
            responses[tag][bank_tag] = elicit_model_on_bank(
                model_name=model_name,
                bank_tag=bank_tag,
                items=items,
                cache_dir=cache_dir,
                log_path=log_path,
                temperature=temp,
                seed=vseed,
            )

    # --- Anchored scoring: full-bank theta per respondent -----------------
    print("\n=== Anchored Rasch scoring (full banks) ===", flush=True)
    rows: List[Dict[str, Any]] = []
    theta_a_list: List[float] = []
    theta_b_list: List[float] = []
    for model_name, temp, vseed, tag in respondents:
        ta, pca, _, _ = score_bank(responses[tag]["A"], bank_a, hashes_a, prior_sd)
        tb, pcb, _, _ = score_bank(responses[tag]["B"], bank_b, hashes_b, prior_sd)
        theta_a_list.append(ta)
        theta_b_list.append(tb)
        rows.append({
            "respondent": tag, "model": model_name,
            "temp": "" if temp is None else temp,
            "theta_A": ta, "theta_B": tb,
            "acc_A": pca, "acc_B": pcb,
            "theta_shift": ta - tb,
        })
        print(f"  {tag:22s}  thetaA={ta:+.3f} (acc {pca:.3f})  "
              f"thetaB={tb:+.3f} (acc {pcb:.3f})  shift={ta-tb:+.3f}", flush=True)

    # --- Within-bank split-half reliability floor -------------------------
    print("\n=== Within-bank split-half reliability (the floor) ===", flush=True)
    rng = random.Random(seed + 100)

    def halves(hs: List[str]) -> Tuple[List[str], List[str]]:
        idx = list(range(len(hs)))
        rng.shuffle(idx)
        mid = len(idx) // 2
        h1 = [hs[i] for i in sorted(idx[:mid])]
        h2 = [hs[i] for i in sorted(idx[mid:])]
        return h1, h2

    a1, a2 = halves(hashes_a)
    b1, b2 = halves(hashes_b)

    def half_thetas(items, resp_key, hs) -> List[float]:
        out = []
        for _, _, _, tag in respondents:
            t, _, _, _ = score_bank(responses[tag][resp_key], items, hs, prior_sd)
            out.append(t)
        return out

    a_h1 = half_thetas(bank_a, "A", a1)
    a_h2 = half_thetas(bank_a, "A", a2)
    b_h1 = half_thetas(bank_b, "B", b1)
    b_h2 = half_thetas(bank_b, "B", b2)

    rel_a = split_half_reliability(a_h1, a_h2)
    rel_b = split_half_reliability(b_h1, b_h2)
    rel_floor = (
        (rel_a + rel_b) / 2.0
        if math.isfinite(rel_a) and math.isfinite(rel_b)
        else float("nan")
    )
    print(f"  Bank A split-half (Spearman-Brown): {rel_a:+.3f}", flush=True)
    print(f"  Bank B split-half (Spearman-Brown): {rel_b:+.3f}", flush=True)
    print(f"  Reliability floor (mean of banks):  {rel_floor:+.3f}", flush=True)

    # --- Primary cross-bank consistency -----------------------------------
    print("\n=== Cross-bank ability consistency (PRIMARY) ===", flush=True)
    sp_cross = spearman(theta_a_list, theta_b_list)
    pe_cross = pearson(theta_a_list, theta_b_list)
    # Raw proportion-correct cross-bank (sanity backup, no model assumption).
    acc_a = [r["acc_A"] for r in rows]
    acc_b = [r["acc_B"] for r in rows]
    sp_cross_acc = spearman(acc_a, acc_b)

    n_resp = len(respondents)
    mean_ta = sum(theta_a_list) / n_resp
    mean_tb = sum(theta_b_list) / n_resp
    mean_shift = mean_ta - mean_tb
    shifts = [a - b for a, b in zip(theta_a_list, theta_b_list)]
    sd_shift = (
        math.sqrt(sum((s - mean_shift) ** 2 for s in shifts) / (n_resp - 1))
        if n_resp > 1 else float("nan")
    )

    print(f"  N respondents               = {n_resp}", flush=True)
    print(f"  Spearman(thetaA, thetaB)    = {sp_cross:+.3f}  (PRIMARY)", flush=True)
    print(f"  Pearson (thetaA, thetaB)    = {pe_cross:+.3f}", flush=True)
    print(f"  Spearman(accA, accB) backup = {sp_cross_acc:+.3f}", flush=True)
    print(f"  Mean theta_A = {mean_ta:+.3f} | Mean theta_B = {mean_tb:+.3f}", flush=True)
    print(f"  Mean level shift A-B = {mean_shift:+.3f}  (sd {sd_shift:.3f})", flush=True)

    # --- Verdict -----------------------------------------------------------
    verdict, label = _verdict(sp_cross, rel_floor, n_resp, mean_shift, sd_shift)
    print(f"\n=== VERDICT ===\n  {label}\n  {verdict}", flush=True)

    elapsed = time.time() - t_run
    _write_outputs(
        out_dir, rows, bank_a, bank_b, respondents, prior_sd, seed, n_items,
        rel_a, rel_b, rel_floor, sp_cross, pe_cross, sp_cross_acc,
        mean_ta, mean_tb, mean_shift, sd_shift, verdict, label, elapsed,
    )
    print(f"\n  Total run time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)


def _verdict(
    sp_cross: float, rel_floor: float, n_resp: int,
    mean_shift: float, sd_shift: float,
) -> Tuple[str, str]:
    """Interpret cross-bank consistency against the within-bank floor."""
    caveat = (
        f" CAVEATS: pilot N={n_resp} respondents -> the Spearman is noisy; this "
        "is a proof-of-concept of the counterfactual design, not a high-N result. "
        "Weak local models can cluster near floor/chance on hard items, "
        "compressing the ability range (see per-model per-bank accuracy)."
    )
    if not math.isfinite(sp_cross) or not math.isfinite(rel_floor):
        return ("INDETERMINATE: a correlation could not be computed (degenerate "
                "theta variance, likely range restriction)." + caveat, "INDETERMINATE")

    # Order-separability: cross-bank rank consistency relative to the floor.
    ratio = sp_cross / rel_floor if abs(rel_floor) > 1e-6 else float("nan")
    level_note = (
        f" A systematic LEVEL shift of {mean_shift:+.3f} logits (sd {sd_shift:.3f}) "
        "remains between instruments: even where rank order is preserved, the "
        "absolute ability level is offset by the instrument's calibration. "
        "Order-separability and level-separability are distinct claims."
    )
    if sp_cross >= 0.9 * rel_floor and rel_floor > 0:
        label = "INSTRUMENT-SEPARABLE (order)"
        body = (
            f"Cross-bank Spearman {sp_cross:+.3f} ~ within-bank reliability floor "
            f"{rel_floor:+.3f} (ratio {ratio:.2f}). The fixed respondent's ABILITY "
            "RANK is essentially as consistent across the two instruments as it is "
            "within one instrument: the anchored scale separates ability ordering "
            "from instrument."
        )
    elif sp_cross >= 0.5 * rel_floor and rel_floor > 0:
        label = "PARTIALLY SEPARABLE"
        body = (
            f"Cross-bank Spearman {sp_cross:+.3f} is below but not far from the "
            f"reliability floor {rel_floor:+.3f} (ratio {ratio:.2f}). Some "
            "instrument effect on ability rank is present beyond measurement noise."
        )
    else:
        label = "INSTRUMENT-CONTAMINATED"
        body = (
            f"Cross-bank Spearman {sp_cross:+.3f} << reliability floor "
            f"{rel_floor:+.3f} (ratio {ratio:.2f}). The respondent's ability rank "
            "is far less consistent across instruments than within one: the scale "
            "conflates ability with instrument."
        )
    return body + level_note + caveat, label


def _write_outputs(
    out_dir, rows, bank_a, bank_b, respondents, prior_sd, seed, n_items,
    rel_a, rel_b, rel_floor, sp_cross, pe_cross, sp_cross_acc,
    mean_ta, mean_tb, mean_shift, sd_shift, verdict, label, elapsed,
) -> None:
    # Per-respondent CSV.
    csv_path = out_dir / "rq5c_respondents.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "respondent", "model", "temp",
            "theta_A", "theta_B", "acc_A", "acc_B", "theta_shift"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    report = out_dir / "rq5c_report.txt"
    lines = [
        "RQ5c COUNTERFACTUAL-RESPONDENT pilot",
        "=" * 60,
        "",
        f"Date:           {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed:           {seed}   n_items/bank (target): {n_items}",
        f"Bank A:         {BANK_A_FORMAT}  ({len(bank_a)} items)",
        f"Bank B:         {BANK_B_FORMAT}  ({len(bank_b)} items)",
        f"Respondents:    {len(respondents)}  -> {[r[3] for r in respondents]}",
        f"Rasch prior_sd: {prior_sd}",
        f"Correct rule:   token_overlap >= {CORRECT_THRESHOLD} (all reference tokens)",
        f"Run time:       {elapsed:.0f}s ({elapsed/60:.1f} min)",
        "",
        "Design: two text-amenable SLAM formats are two instruments. A FIXED LLM",
        "answers both banks, so its TRUE ability is constant by construction. Any",
        "gap between its anchored ability on bank A vs B is pure instrument effect.",
        "Item difficulties fixed at logit(human population mistake rate); only the",
        "respondent's scalar Rasch theta is estimated per bank (MAP).",
        "",
        "--- Per-respondent ability (anchored Rasch theta + raw accuracy) ---",
        f"  {'respondent':22s}  thetaA   accA    thetaB   accB    shift",
    ]
    for r in rows:
        lines.append(
            f"  {r['respondent']:22s}  {r['theta_A']:+.3f}  {r['acc_A']:.3f}  "
            f"{r['theta_B']:+.3f}  {r['acc_B']:.3f}  {r['theta_shift']:+.3f}"
        )
    lines += [
        "",
        "--- Within-bank split-half reliability (the floor) ---",
        f"  Bank A (Spearman-Brown):  {rel_a:+.3f}",
        f"  Bank B (Spearman-Brown):  {rel_b:+.3f}",
        f"  Reliability floor (mean): {rel_floor:+.3f}",
        "",
        "--- Cross-bank ability consistency (PRIMARY) ---",
        f"  N respondents               = {len(respondents)}",
        f"  Spearman(thetaA, thetaB)    = {sp_cross:+.3f}   <-- PRIMARY",
        f"  Pearson (thetaA, thetaB)    = {pe_cross:+.3f}",
        f"  Spearman(accA, accB) backup = {sp_cross_acc:+.3f}",
        f"  Mean theta_A = {mean_ta:+.3f}  Mean theta_B = {mean_tb:+.3f}",
        f"  Mean level shift A-B = {mean_shift:+.3f}  (sd {sd_shift:.3f})",
        "",
        "--- Verdict ---",
        f"  {label}",
        f"  {verdict}",
        "",
        "--- Output files ---",
        f"  {csv_path}",
        f"  {report}",
    ]
    text = "\n".join(lines) + "\n"
    with report.open("w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text, flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="RQ5c counterfactual-respondent pilot.")
    p.add_argument("--human-csv", default="deep_irt/slam_pilot/outputs/human_difficulty.csv")
    p.add_argument("--raw-dir", default="rl/data/slam_raw")
    p.add_argument("--out-dir", default="deep_irt/rq5c_counterfactual/outputs")
    p.add_argument("--n-items", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prior-sd", type=float, default=1.0)
    p.add_argument("--no-temp-variants", action="store_true",
                   help="Disable the temperature-variant pseudo-respondents.")
    p.add_argument("--pool", default="",
                   help="Comma-separated model pool override.")
    args = p.parse_args(argv)

    pool = ([m.strip() for m in args.pool.split(",") if m.strip()]
            if args.pool else list(DEFAULT_POOL))

    run(
        human_csv=Path(args.human_csv),
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        n_items=args.n_items,
        seed=args.seed,
        prior_sd=args.prior_sd,
        pool=pool,
        use_temp_variants=not args.no_temp_variants,
    )


if __name__ == "__main__":
    main()
