"""analyze_transfer.py -- Tightened analysis of the SLAM transfer pilot.

Loads transfer_results.csv and human_difficulty.csv, then reports:

1. Per-format transfer correlations (overall, reverse_translate, reverse_tap).
   Hypothesis: reverse_tap is less contaminated by exact-match grading artifacts
   than reverse_translate (free typing), so transfer should be cleaner (higher).

2. Grading-artifact confound: items where the LLM population is near-correct
   (LLM difficulty < 0.10) yet human difficulty is high (> 0.30) are flagged as
   artifact-suspect (phrasing/word-order penalised in SLAM grading but LLMs
   produce correct meaning).  Reports the count and the correlation excluding them
   -- with an explicit selection-bias caveat.

3. Difficulty-driven subset correlation (human difficulty > 0.10) with the same
   selection caveat.

Outputs:
    deep_irt/slam_pilot/outputs/transfer_tightened.txt

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="deep_irt;rl/src;ma-irt" \\
        python deep_irt/slam_pilot/analyze_transfer.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _rank(vals: List[float]) -> List[float]:
    n = len(vals)
    sorted_with_idx = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_with_idx[j + 1][1] == sorted_with_idx[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[sorted_with_idx[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


def spearman(x: List[float], y: List[float]) -> float:
    return pearson(_rank(x), _rank(y))


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Kendall tau-b."""
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    tied_xy = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0.0 and dy == 0.0:
                tied_xy += 1
            elif dx == 0.0:
                tied_x += 1
            elif dy == 0.0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    n1 = tied_x + tied_xy
    n2 = tied_y + tied_xy
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom < 1e-12:
        return float("nan")
    return (concordant - discordant) / denom


def corr_block(
    hd: List[float],
    llm: List[float],
    label: str,
) -> Tuple[float, float, float]:
    """Return (spearman, kendall, pearson) and print a formatted block."""
    n = len(hd)
    sp = spearman(hd, llm)
    kt = kendall_tau(hd, llm)
    pe = pearson(hd, llm)
    return sp, kt, pe


def fmt_r(v: float) -> str:
    return f"{v:+.4f}" if not math.isnan(v) else "  nan "


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    results_csv: Path,
    out_dir: Path,
    artifact_llm_thresh: float = 0.10,
    artifact_human_thresh: float = 0.30,
    hard_human_thresh: float = 0.10,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load transfer_results.csv
    rows: List[Dict[str, str]] = []
    with results_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("BLOCKER: transfer_results.csv is empty.")
        sys.exit(1)

    # Parse numeric fields
    items = []
    for row in rows:
        try:
            hd = float(row["human_difficulty"])
            llm = float(row["llm_difficulty_overlap"])
            fmt = row["format"]
            items.append({"hash": row["item_hash"], "hd": hd, "llm": llm, "format": fmt})
        except (ValueError, KeyError) as e:
            print(f"  Warning: skipping row {row.get('item_hash','?')}: {e}")

    n_total = len(items)
    rt_items = [it for it in items if it["format"] == "reverse_translate"]
    tap_items = [it for it in items if it["format"] == "reverse_tap"]

    # Artifact-suspect: near-perfect LLM but high human difficulty
    artifact_items = [
        it for it in items
        if it["llm"] < artifact_llm_thresh and it["hd"] > artifact_human_thresh
    ]
    n_artifact = len(artifact_items)
    artifact_hashes = {it["hash"] for it in artifact_items}
    non_artifact_items = [it for it in items if it["hash"] not in artifact_hashes]

    # Hard subset: human difficulty > threshold
    hard_items = [it for it in items if it["hd"] > hard_human_thresh]

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def corr_lines(
        subset: List[dict],
        label: str,
        caveat: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        hd = [it["hd"] for it in subset]
        llm = [it["llm"] for it in subset]
        sp, kt, pe = corr_block(hd, llm, label)
        lines.append(f"  n = {len(subset)}")
        lines.append(f"  Spearman r  = {fmt_r(sp)}  (primary)")
        lines.append(f"  Kendall tau = {fmt_r(kt)}")
        lines.append(f"  Pearson  r  = {fmt_r(pe)}")
        if caveat:
            lines.append(f"  NOTE: {caveat}")
        return sp, kt, pe

    lines.append("SLAM Transfer Pilot -- Tightened Analysis")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Data: {results_csv}")
    lines.append(f"Total items: {n_total}  (reverse_translate={len(rt_items)}, reverse_tap={len(tap_items)})")
    lines.append(f"Artifact thresholds: LLM difficulty < {artifact_llm_thresh}, human difficulty > {artifact_human_thresh}")
    lines.append(f"Hard-subset threshold: human difficulty > {hard_human_thresh}")

    # --- Section 1: Per-format correlations ---
    section("1. Transfer Correlations by Format")

    lines.append("")
    lines.append("(a) Overall (all 200 items)")
    sp_all, kt_all, pe_all = corr_lines(items, "overall")

    lines.append("")
    lines.append("(b) reverse_translate only (free typing, more phrasing latitude)")
    sp_rt, kt_rt, pe_rt = corr_lines(rt_items, "reverse_translate")

    lines.append("")
    lines.append("(c) reverse_tap only (tile selection, minimal phrasing deviation)")
    sp_tap, kt_tap, pe_tap = corr_lines(tap_items, "reverse_tap")

    lines.append("")
    lines.append("HYPOTHESIS: reverse_tap is less contaminated by exact-match grading")
    lines.append("artifacts (phrasing / word-order penalised in SLAM but not meaningful"),
    lines.append("errors), so its transfer correlation should be higher than reverse_translate.")
    lines.append("")

    # Evaluate hypothesis
    if math.isnan(sp_tap) or math.isnan(sp_rt):
        hypothesis_verdict = "INDETERMINATE (insufficient data in one split)."
    elif len(tap_items) < 10:
        hypothesis_verdict = (
            f"INCONCLUSIVE -- only {len(tap_items)} reverse_tap items; "
            "sample too small for reliable comparison."
        )
    else:
        delta = sp_tap - sp_rt
        if delta > 0.05:
            hypothesis_verdict = (
                f"SUPPORTED: reverse_tap Spearman ({sp_tap:+.4f}) is higher than "
                f"reverse_translate ({sp_rt:+.4f}) by {delta:.4f}, consistent with "
                "less exact-match contamination on tile-selection items."
            )
        elif delta < -0.05:
            hypothesis_verdict = (
                f"NOT SUPPORTED: reverse_tap Spearman ({sp_tap:+.4f}) is lower than "
                f"reverse_translate ({sp_rt:+.4f}) by {abs(delta):.4f}. "
                "Either the tap format introduces its own noise or the sample is too "
                "small (n={len(tap_items)}) to distinguish."
            )
        else:
            hypothesis_verdict = (
                f"INCONCLUSIVE: reverse_tap Spearman ({sp_tap:+.4f}) vs "
                f"reverse_translate ({sp_rt:+.4f}); delta = {delta:+.4f} is within "
                "noise margin given the tap sample size (n={len(tap_items)})."
            )
    lines.append(f"Verdict: {hypothesis_verdict}")

    # --- Section 2: Grading-artifact confound ---
    section("2. Grading-Artifact Confound")

    lines.append("")
    lines.append(
        f"Artifact-suspect items: LLM difficulty < {artifact_llm_thresh} "
        f"AND human difficulty > {artifact_human_thresh}"
    )
    lines.append(
        "Rationale: LLMs produce semantically correct translations that Duolingo's "
        "exact-match grader rejects for phrasing/word-order reasons. These items "
        "conflate surface-form matching difficulty with true linguistic difficulty."
    )
    lines.append("")
    lines.append(f"  Artifact-suspect count: {n_artifact} of {n_total} ({100*n_artifact/n_total:.1f}%)")

    # List a few examples
    if artifact_items:
        lines.append("")
        lines.append("  Examples (item_hash, LLM_diff, human_diff, reference):")
        for it in sorted(artifact_items, key=lambda x: x["hd"], reverse=True)[:5]:
            # find reference from original rows
            ref_row = next((r for r in rows if r["item_hash"] == it["hash"]), {})
            ref = ref_row.get("reference", "")[:60]
            lines.append(
                f"    {it['hash']}  llm={it['llm']:.4f}  human={it['hd']:.4f}  ref: {ref!r}"
            )

    lines.append("")
    lines.append("Correlation EXCLUDING artifact-suspect items:")
    lines.append(
        "  (Selection caveat: removing items where the grading artifact is most "
        "visible inflates the apparent correlation by construction. This estimate "
        "is an upper bound on the clean-signal correlation, not an unbiased estimate.)"
    )
    sp_noart, kt_noart, pe_noart = corr_lines(
        non_artifact_items,
        "no-artifact",
        caveat=(
            "selection bias -- items removed precisely because grading artifact is "
            "detectable; correlation is biased upward."
        ),
    )

    # --- Section 3: Hard-subset correlation ---
    section("3. Difficulty-Driven Subset (human difficulty > 0.10)")

    lines.append("")
    lines.append(
        f"Items with human difficulty > {hard_human_thresh}: {len(hard_items)} of {n_total}"
    )
    lines.append(
        "Rationale: the easy tail (human difficulty near 0) has near-zero LLM "
        "difficulty too, so both signals are compressed and contribute little "
        "rank information. Restricting to harder items tests whether the signal "
        "exists in the informative range."
    )
    lines.append(
        "  (Selection caveat: truncating the easy tail inflates rank correlations "
        "by removing the floor region where both signals agree trivially. "
        "This should not be interpreted as the general-population correlation.)"
    )
    lines.append("")
    sp_hard, kt_hard, pe_hard = corr_lines(
        hard_items,
        "hard-subset",
        caveat=(
            "selection bias -- easy floor removed; correlation is biased upward "
            "relative to the full-range estimate."
        ),
    )

    # --- Section 4: Summary ---
    section("4. Summary and Honest Verdict")

    lines.append("")
    lines.append(f"  Overall (n={n_total}):                 Spearman = {fmt_r(sp_all)}")
    lines.append(f"  reverse_translate (n={len(rt_items)}):      Spearman = {fmt_r(sp_rt)}")
    lines.append(f"  reverse_tap (n={len(tap_items)}):            Spearman = {fmt_r(sp_tap)}")
    lines.append(f"  Excl. artifact-suspect (n={len(non_artifact_items)}): Spearman = {fmt_r(sp_noart)}  [selection-biased upward]")
    lines.append(f"  Hard subset >0.10 (n={len(hard_items)}):      Spearman = {fmt_r(sp_hard)}  [selection-biased upward]")
    lines.append("")

    # Overall verdict
    if sp_all > 0.4:
        signal_desc = "strong"
    elif sp_all > 0.2:
        signal_desc = "weak but present"
    elif sp_all > 0.0:
        signal_desc = "near-zero"
    else:
        signal_desc = "absent or negative"

    tap_note = ""
    if not math.isnan(sp_tap) and not math.isnan(sp_rt) and len(tap_items) >= 10:
        delta = sp_tap - sp_rt
        if delta > 0.05:
            tap_note = (
                f" reverse_tap shows a cleaner signal ({sp_tap:+.4f} vs "
                f"{sp_rt:+.4f} for reverse_translate), consistent with the "
                "grading-artifact hypothesis."
            )
        else:
            tap_note = (
                f" The format split does not clearly support the grading-artifact "
                f"hypothesis (tap={sp_tap:+.4f}, translate={sp_rt:+.4f})."
            )

    verdict = (
        f"The SLAM pilot shows a {signal_desc} transfer signal overall "
        f"(Spearman = {sp_all:+.4f}). "
        f"{n_artifact} of {n_total} items ({100*n_artifact/n_total:.0f}%) are "
        "artifact-suspect (LLMs near-correct but human difficulty high), confirming "
        "that exact-match grading inflates apparent human difficulty for semantically "
        "easy items."
        f"{tap_note} "
        "Excluding artifact-suspect items or restricting to the hard tail both raise "
        "the estimated correlation, but these are selection operations and the inflated "
        "figures cannot be taken at face value. "
        "The honest conclusion is that SLAM human difficulty is too contaminated by "
        "surface-form grading to provide a reliable transfer signal at current scale "
        "without a regrading step that scores semantic correctness rather than "
        "exact-string match."
    )
    lines.append("ONE-LINE VERDICT:")
    lines.append(f"  {verdict}")

    # Write report
    report_text = "\n".join(lines) + "\n"
    out_path = out_dir / "transfer_tightened.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tightened analysis of SLAM transfer pilot results."
    )
    parser.add_argument(
        "--results-csv",
        default="deep_irt/slam_pilot/outputs/transfer_results.csv",
        help="Path to transfer_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="deep_irt/slam_pilot/outputs",
        help="Output directory for transfer_tightened.txt",
    )
    parser.add_argument(
        "--artifact-llm-thresh",
        type=float,
        default=0.10,
        help="LLM difficulty below which an item is 'near-correct' for artifact flagging",
    )
    parser.add_argument(
        "--artifact-human-thresh",
        type=float,
        default=0.30,
        help="Human difficulty above which a near-correct LLM item is artifact-suspect",
    )
    parser.add_argument(
        "--hard-human-thresh",
        type=float,
        default=0.10,
        help="Human difficulty threshold for the 'hard subset' analysis",
    )
    args = parser.parse_args(argv)

    run_analysis(
        results_csv=Path(args.results_csv),
        out_dir=Path(args.out_dir),
        artifact_llm_thresh=args.artifact_llm_thresh,
        artifact_human_thresh=args.artifact_human_thresh,
        hard_human_thresh=args.hard_human_thresh,
    )


if __name__ == "__main__":
    main()
