#!/usr/bin/env python3
"""Prepare ASSISTments data for proxy-ordinality experiment (RQ5).

Converts ASSISTments 2009-2010 skill-builder dataset into the kt-gpcm
sequence format with K=3 ordinal categories derived from attempt counts:
    1 attempt  → category 0  (mastery)
    2-3 attempts → category 1  (partial)
    4+ attempts  → category 2  (struggle)

Expected input: the ASSISTments 2009-2010 skill-builder CSV file.
Download from: https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data
Typical filename: skill_builder_data_corrected.csv

Usage::

    PYTHONPATH=src python scripts/prepare_assistments.py \\
        --input path/to/skill_builder_data_corrected.csv \\
        --output_dir data \\
        --name assistments_k3 \\
        --min_seq 10 \\
        --max_seq 500

Output (inside <output_dir>/<name>/):
    sequences.json          — list of {questions, responses} dicts
    metadata.json           — dataset parameters
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bin_attempts(n_attempts: int) -> int:
    """Map attempt count to ordinal category 0/1/2."""
    if n_attempts <= 1:
        return 0
    elif n_attempts <= 3:
        return 1
    else:
        return 2


def load_assistments(csv_path: Path) -> list[dict]:
    """Parse ASSISTments CSV and return per-student sequences.

    Handles both comma and tab delimiters. Looks for columns:
        user_id, problem_id (or skill_id), attempt_count (or attempts)
    Falls back to order_id for sequencing if timestamp is absent.
    """
    import csv

    with csv_path.open(encoding="utf-8-sig") as f:
        sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t")

    rows = []
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, dialect=dialect)
        headers = [h.strip().lower() for h in reader.fieldnames]
        reader.fieldnames = headers
        for row in reader:
            rows.append({k.strip().lower(): v.strip() for k, v in row.items()})

    print(f"Loaded {len(rows)} rows. Columns: {list(rows[0].keys())[:10]}")

    # Identify relevant columns
    def find_col(candidates):
        for c in candidates:
            if c in rows[0]:
                return c
        return None

    uid_col      = find_col(["user_id", "student_id", "anon_student_id"])
    prob_col     = find_col(["problem_id", "skill_id", "skill_name", "problem_name"])
    attempt_col  = find_col(["attempt_count", "attempts", "attempt_number"])
    order_col    = find_col(["order_id", "timestamp", "start_time"])

    if uid_col is None or prob_col is None or attempt_col is None:
        raise ValueError(
            f"Could not find required columns. Found: {list(rows[0].keys())}\n"
            f"Need: user_id, problem_id/skill_id, attempt_count/attempts"
        )

    print(f"Using columns: uid={uid_col}, prob={prob_col}, attempt={attempt_col}, order={order_col}")

    # Group by student, sort by order
    student_rows = defaultdict(list)
    for row in rows:
        try:
            uid = row[uid_col]
            prob = row[prob_col]
            attempts = int(float(row[attempt_col]))
            order = int(float(row[order_col])) if order_col else 0
            student_rows[uid].append((order, prob, attempts))
        except (ValueError, KeyError):
            continue

    # Build problem → integer ID mapping (1-based)
    all_probs = sorted({r[1] for rows_s in student_rows.values() for r in rows_s})
    prob_to_id = {p: i + 1 for i, p in enumerate(all_probs)}
    n_questions = len(prob_to_id)
    print(f"Unique problems/skills: {n_questions}")
    print(f"Unique students: {len(student_rows)}")

    return student_rows, prob_to_id, n_questions


def build_sequences(
    student_rows: dict,
    prob_to_id: dict,
    min_seq: int,
    max_seq: int,
) -> list[dict]:
    sequences = []
    for uid, rows in student_rows.items():
        rows_sorted = sorted(rows, key=lambda x: x[0])
        q_seq = [prob_to_id[r[1]] for r in rows_sorted]
        r_seq = [bin_attempts(r[2]) for r in rows_sorted]

        # Truncate to max_seq
        if len(q_seq) > max_seq:
            q_seq = q_seq[:max_seq]
            r_seq = r_seq[:max_seq]

        if len(q_seq) >= min_seq:
            sequences.append({"questions": q_seq, "responses": r_seq})

    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ASSISTments data for proxy-ordinality experiment."
    )
    parser.add_argument("--input", required=True,
                        help="Path to ASSISTments CSV file.")
    parser.add_argument("--output_dir", default="data",
                        help="Root data directory.")
    parser.add_argument("--name", default="assistments_k3",
                        help="Dataset name (subdirectory under output_dir).")
    parser.add_argument("--min_seq", type=int, default=10,
                        help="Minimum sequence length to include.")
    parser.add_argument("--max_seq", type=int, default=500,
                        help="Maximum sequence length (truncate longer).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    student_rows, prob_to_id, n_questions = load_assistments(csv_path)
    sequences = build_sequences(student_rows, prob_to_id, args.min_seq, args.max_seq)

    seq_lens = [len(s["questions"]) for s in sequences]
    print(f"\nSequences kept: {len(sequences)}")
    print(f"Seq length: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={np.mean(seq_lens):.1f}, median={np.median(seq_lens):.1f}")

    # Category distribution
    all_responses = [r for s in sequences for r in s["responses"]]
    for k in range(3):
        count = all_responses.count(k)
        print(f"  Category {k}: {count} ({100*count/len(all_responses):.1f}%)")

    out = Path(args.output_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    with (out / "sequences.json").open("w") as f:
        json.dump(sequences, f)
    print(f"\nWrote sequences -> {out / 'sequences.json'}")

    meta = {
        "dataset_name": args.name,
        "n_students": len(sequences),
        "n_questions": n_questions,
        "n_categories": 3,
        "seq_len_range": [args.min_seq, args.max_seq],
        "model_type": "proxy_ordinal",
        "source": "ASSISTments 2009-2010 skill-builder",
        "binning": {"0": "1 attempt", "1": "2-3 attempts", "2": "4+ attempts"},
    }
    with (out / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata  -> {out / 'metadata.json'}")
    print("\nNo true_irt_parameters.json written (real data — no ground truth).")
    print("Done.")


if __name__ == "__main__":
    main()
