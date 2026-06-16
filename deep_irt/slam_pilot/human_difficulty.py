"""human_difficulty.py -- Compute per-item human difficulty from SLAM 2018 en_es.

An "item" is a unique exercise type identified by a 16-hex MD5 hash of
(format, lowercased word sequence), exactly as in SlamAdapter (rl/src/ordrec/data/slam.py).

Human difficulty for an item is defined as the proportion of INCORRECT token
responses pooled across all learners and all splits (train + dev + test).
Specifically: total_mistakes / total_tokens across every attempt of that item.

This is a token-level mistake rate, consistent with the SLAM 2018 task definition,
but aggregated to exercise level. An item's difficulty ranges from 0.0 (always
all-correct) to 1.0 (always all-wrong).

Only items that appear in at least MIN_RESPONSES total token observations are
retained in the final table (default threshold: 20 tokens).

Usage:
    python deep_irt/slam_pilot/human_difficulty.py \\
        --raw-dir rl/data/slam_raw \\
        --out-dir deep_irt/slam_pilot/outputs \\
        [--min-responses 20]

Outputs:
    outputs/human_difficulty.csv   -- item_hash, format, words_sample,
                                      n_exercises, n_tokens, n_mistakes,
                                      human_difficulty
    outputs/human_difficulty_report.txt -- text summary of the distribution
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SLAM raw-file parsing (self-contained, no rl/ imports needed)
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^# user:(?P<user>\S+)\s+.*?format:(?P<format>\w+)")


def _exercise_hash(format_: str, words: List[str]) -> str:
    """16-hex MD5 of (format, lowercased word sequence) -- matches SlamAdapter."""
    canonical = format_ + "|" + " ".join(w.lower() for w in words)
    return hashlib.md5(canonical.encode()).hexdigest()[:16]


def _load_key_file(path: Path) -> Dict[str, int]:
    """Load a .key file into {token_id: label}."""
    lookup: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    lookup[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    return lookup


def _parse_slam_file(
    path: Path,
    has_labels: bool,
    label_lookup: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Parse one SLAM file into a list of exercise dicts.

    Returns list of dicts with keys:
        user (str), format (str), days (float),
        words (list[str], lowercased), labels (list[int], 0=correct 1=mistake)
    """
    if label_lookup is None:
        label_lookup = {}

    exercises: List[Dict[str, Any]] = []
    cur_user: Optional[str] = None
    cur_fmt: Optional[str] = None
    cur_days: float = 0.0
    cur_words: List[str] = []
    cur_labels: List[int] = []

    def flush() -> None:
        if cur_words and cur_user is not None:
            exercises.append(
                {
                    "user": cur_user,
                    "format": cur_fmt,
                    "days": cur_days,
                    "words": list(cur_words),
                    "labels": list(cur_labels),
                }
            )
        cur_words.clear()
        cur_labels.clear()

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("# user:"):
                flush()
                m = _HEADER_RE.match(line)
                if m:
                    cur_user = m.group("user")
                    cur_fmt = m.group("format")
                    dm = re.search(r"days:([0-9.]+)", line)
                    cur_days = float(dm.group(1)) if dm else 0.0
            elif line.startswith("#") or line.strip() == "":
                flush()
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                token_id = parts[0]
                word = parts[1]
                if has_labels:
                    try:
                        label = int(parts[-1])
                    except (ValueError, IndexError):
                        label = 0
                else:
                    label = label_lookup.get(token_id, 0)
                cur_words.append(word.lower())
                cur_labels.append(label)

    flush()
    return exercises


def _locate_slam_files(raw_dir: Path) -> Tuple[Path, Path, Path, Path, Path]:
    """Return (train, dev, dev_key, test, test_key) paths."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")
    train_files = [
        p for p in raw_dir.glob("*.slam.*.train") if not p.name.startswith(".")
    ]
    if not train_files:
        raise FileNotFoundError(f"No *.slam.*.train file found under {raw_dir}")
    if len(train_files) > 1:
        raise ValueError(f"Multiple train files under {raw_dir}; expected exactly one.")
    train = train_files[0]
    base = train.name.replace(".train", "")
    dev = raw_dir / f"{base}.dev"
    dev_key = raw_dir / f"{base}.dev.key"
    test = raw_dir / f"{base}.test"
    test_key = raw_dir / f"{base}.test.key"
    for p, name in [(dev, "dev"), (dev_key, "dev.key"), (test, "test"), (test_key, "test.key")]:
        if not p.exists():
            raise FileNotFoundError(f"Expected {name} file at {p}")
    return train, dev, dev_key, test, test_key


# ---------------------------------------------------------------------------
# Difficulty computation
# ---------------------------------------------------------------------------

def compute_human_difficulty(
    raw_dir: Path,
    min_responses: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse all SLAM splits and compute per-item token-level mistake rate.

    Args:
        raw_dir: Path to the SLAM raw data directory.
        min_responses: Minimum total token observations for an item to be
            included in the output table. Items below this threshold are
            excluded (long-tail suppression).

    Returns:
        (rows, stats) where rows is a list of dicts (one per surviving item)
        and stats is a summary dict.
    """
    train_path, dev_path, dev_key_path, test_path, test_key_path = _locate_slam_files(raw_dir)

    print(f"Parsing train: {train_path.name}")
    train_exs = _parse_slam_file(train_path, has_labels=True)

    print(f"Parsing dev (+ key): {dev_path.name}")
    dev_key = _load_key_file(dev_key_path)
    dev_exs = _parse_slam_file(dev_path, has_labels=False, label_lookup=dev_key)

    print(f"Parsing test (+ key): {test_path.name}")
    test_key = _load_key_file(test_key_path)
    test_exs = _parse_slam_file(test_path, has_labels=False, label_lookup=test_key)

    all_exs = train_exs + dev_exs + test_exs
    n_total_exercises = len(all_exs)
    n_total_tokens = sum(len(e["labels"]) for e in all_exs)
    n_learners = len(set(e["user"] for e in all_exs))

    print(
        f"Loaded {n_total_exercises:,} exercises, "
        f"{n_total_tokens:,} tokens, "
        f"{n_learners:,} learners."
    )

    # Accumulate per-item: n_exercises, n_tokens, n_mistakes, a sample of words + format
    item_n_ex: Dict[str, int] = defaultdict(int)
    item_n_tok: Dict[str, int] = defaultdict(int)
    item_n_mis: Dict[str, int] = defaultdict(int)
    item_format: Dict[str, str] = {}
    item_words: Dict[str, str] = {}  # store first-seen word list as sample text

    for ex in all_exs:
        sig = _exercise_hash(ex["format"], ex["words"])
        item_n_ex[sig] += 1
        item_n_tok[sig] += len(ex["labels"])
        item_n_mis[sig] += sum(ex["labels"])
        if sig not in item_format:
            item_format[sig] = ex["format"]
            item_words[sig] = " ".join(ex["words"])

    n_unique_all = len(item_n_ex)

    # Apply minimum-response threshold (token count, not exercise count)
    surviving = {
        sig for sig, n_tok in item_n_tok.items() if n_tok >= min_responses
    }
    n_surviving = len(surviving)

    print(f"Unique item signatures (all splits): {n_unique_all:,}")
    print(
        f"Items with >= {min_responses} token observations: "
        f"{n_surviving:,} ({100 * n_surviving / n_unique_all:.1f}%)"
    )

    rows = []
    for sig in sorted(surviving):
        n_tok = item_n_tok[sig]
        n_mis = item_n_mis[sig]
        rows.append(
            {
                "item_hash": sig,
                "format": item_format[sig],
                "words_sample": item_words[sig],
                "n_exercises": item_n_ex[sig],
                "n_tokens": n_tok,
                "n_mistakes": n_mis,
                "human_difficulty": n_mis / n_tok,
            }
        )

    # Sort by difficulty descending (hardest first)
    rows.sort(key=lambda r: r["human_difficulty"], reverse=True)

    # Compute distribution stats
    difficulties = [r["human_difficulty"] for r in rows]
    difficulties_sorted = sorted(difficulties)
    n = len(difficulties_sorted)
    median = difficulties_sorted[n // 2] if n > 0 else float("nan")

    # Histogram buckets (10 equal-width bins from 0 to 1)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist = [0] * (len(bins) - 1)
    for d in difficulties:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                hist[i] += 1
                break
        else:
            # d == 1.0 goes into the last bucket
            hist[-1] += 1

    stats = {
        "n_total_exercises": n_total_exercises,
        "n_total_tokens": n_total_tokens,
        "n_learners": n_learners,
        "n_unique_sigs_all": n_unique_all,
        "min_responses_threshold": min_responses,
        "n_items_surviving": n_surviving,
        "difficulty_min": difficulties_sorted[0] if n else float("nan"),
        "difficulty_median": median,
        "difficulty_max": difficulties_sorted[-1] if n else float("nan"),
        "difficulty_mean": sum(difficulties) / n if n else float("nan"),
        "histogram_bins": bins,
        "histogram_counts": hist,
    }

    return rows, stats


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fields = ["item_hash", "format", "words_sample", "n_exercises", "n_tokens",
              "n_mistakes", "human_difficulty"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows):,} rows -> {out_path}")


def _write_report(stats: Dict[str, Any], out_path: Path) -> None:
    bins = stats["histogram_bins"]
    hist = stats["histogram_counts"]
    n_items = stats["n_items_surviving"]

    lines = [
        "SLAM 2018 en_es -- Human Item Difficulty Report",
        "=" * 52,
        "",
        f"Total exercises (train+dev+test): {stats['n_total_exercises']:>10,}",
        f"Total tokens:                     {stats['n_total_tokens']:>10,}",
        f"Unique learners:                  {stats['n_learners']:>10,}",
        f"Unique item signatures (all):     {stats['n_unique_sigs_all']:>10,}",
        "",
        f"Min-response threshold:           {stats['min_responses_threshold']:>10} tokens",
        f"Surviving items:                  {n_items:>10,}",
        "",
        "Difficulty distribution (proportion incorrect, token-level):",
        f"  Min:    {stats['difficulty_min']:.4f}",
        f"  Median: {stats['difficulty_median']:.4f}",
        f"  Mean:   {stats['difficulty_mean']:.4f}",
        f"  Max:    {stats['difficulty_max']:.4f}",
        "",
        "Histogram (proportion of surviving items per bin):",
        "  [low  -- high)  count    bar",
    ]

    max_count = max(hist) if hist else 1
    bar_width = 40
    for i, count in enumerate(hist):
        lo, hi = bins[i], bins[i + 1]
        bar_len = int(round(bar_width * count / max_count))
        bar = "#" * bar_len
        pct = 100.0 * count / n_items if n_items else 0.0
        lines.append(f"  [{lo:.1f} -- {hi:.1f})   {count:6d}  {pct:5.1f}%  {bar}")

    report_text = "\n".join(lines) + "\n"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved report -> {out_path}")
    print()
    print(report_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-item human difficulty from SLAM 2018 en_es raw data."
    )
    parser.add_argument(
        "--raw-dir",
        default="rl/data/slam_raw",
        help="Path to the SLAM raw data directory (default: rl/data/slam_raw)",
    )
    parser.add_argument(
        "--out-dir",
        default="deep_irt/slam_pilot/outputs",
        help="Directory for output files (default: deep_irt/slam_pilot/outputs)",
    )
    parser.add_argument(
        "--min-responses",
        type=int,
        default=20,
        help="Minimum total token observations for an item to be retained (default: 20)",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, stats = compute_human_difficulty(raw_dir, min_responses=args.min_responses)

    _write_csv(rows, out_dir / "human_difficulty.csv")
    _write_report(stats, out_dir / "human_difficulty_report.txt")


if __name__ == "__main__":
    main()
