"""banks.py -- Build the two SLAM instrument banks for RQ5c.

Two banks = two TEXT-amenable SLAM en_es exercise formats, the two instruments:

    Bank A: reverse_translate  (free-typed translation)
    Bank B: reverse_tap        (tile-selection translation)

For a free-generation LLM both are the SAME underlying task ("translate this
Spanish to English"); the instrument difference (typing vs picking word tiles)
exists only in the human UI. That is exactly the counterfactual property we
want: the LLM's TRUE ability on the construct is identical across banks, so any
gap in its anchored ability estimate is pure instrument effect.

Each bank carries, per item:
    item_hash, format, prompt (Spanish), reference (English token list),
    human_difficulty (population mistake rate, the anchoring source).

Items are sampled STRATIFIED on human difficulty so the bank spans the range
rather than collapsing onto the easy floor (reverse_tap is heavily left-skewed:
mean human difficulty ~0.06).
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Formats that an LLM can answer by free text generation. ``listen`` is audio,
# excluded by design.
TEXT_FORMATS = ("reverse_translate", "reverse_tap")

_HEADER_RE = re.compile(r"^# user:(?P<user>\S+)\s+.*?format:(?P<format>\w+)")


def exercise_hash(format_: str, words: List[str]) -> str:
    """16-hex MD5 of (format, lowercased words) -- matches SlamAdapter."""
    canonical = format_ + "|" + " ".join(w.lower() for w in words)
    return hashlib.md5(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Human-difficulty candidates per format
# ---------------------------------------------------------------------------

def load_candidates(human_csv: Path, fmt: str) -> List[Dict[str, Any]]:
    """Load all human-calibrated items of one format from human_difficulty.csv."""
    out: List[Dict[str, Any]] = []
    with human_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["format"] == fmt:
                out.append({
                    "item_hash": row["item_hash"],
                    "format": fmt,
                    "human_difficulty": float(row["human_difficulty"]),
                })
    return out


def stratified_sample(
    candidates: List[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample ``n`` items stratified on human difficulty to span the range.

    Five strata on human_difficulty with hand-tuned target proportions that
    oversample the hard tail. Robust to formats that have few hard items
    (caps each stratum at availability and reallocates the slack).
    """
    rng = random.Random(seed)
    bounds = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]
    target_fracs = [0.20, 0.15, 0.20, 0.25, 0.20]

    groups: Dict[int, List[Dict]] = defaultdict(list)
    for item in candidates:
        d = item["human_difficulty"]
        s = 4
        for i, (lo, hi) in enumerate(bounds):
            if lo <= d < hi:
                s = i
                break
        groups[s].append(item)

    # Hardest stratum: take all (there are few).
    target = [0] * 5
    target[4] = min(len(groups[4]), n)
    remaining_n = max(0, n - target[4])
    fr = target_fracs[:4]
    fs = sum(fr)
    for i in range(4):
        t = int(round(remaining_n * fr[i] / fs)) if fs > 0 else 0
        target[i] = min(t, len(groups[i]))

    # True-up to exactly n where availability allows.
    allocated = sum(target)
    diff = n - allocated
    if diff != 0:
        order = sorted(range(5), key=lambda s: len(groups[s]) - target[s], reverse=True)
        idx = 0
        guard = 0
        while diff != 0 and guard < 10000:
            s = order[idx % 5]
            if diff > 0 and target[s] < len(groups[s]):
                target[s] += 1
                diff -= 1
            elif diff < 0 and target[s] > 0:
                target[s] -= 1
                diff += 1
            idx += 1
            guard += 1

    sampled: List[Dict[str, Any]] = []
    for s in range(5):
        g = groups[s]
        t = max(0, min(target[s], len(g)))
        if t > 0:
            sampled.extend(rng.sample(g, t))
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Targeted SLAM prompt loading (only the sampled hashes)
# ---------------------------------------------------------------------------

def _parse_targeted(path: Path, target_hashes: set) -> Dict[str, Dict[str, Any]]:
    """Parse one SLAM file, return {hash: {format, prompt, reference}} for hits."""
    found: Dict[str, Dict[str, Any]] = {}
    cur_prompt: Optional[str] = None
    cur_fmt: Optional[str] = None
    cur_words: List[str] = []

    def flush() -> None:
        if cur_words and cur_fmt in TEXT_FORMATS:
            h = exercise_hash(cur_fmt, cur_words)
            if h in target_hashes and h not in found:
                found[h] = {
                    "format": cur_fmt,
                    "prompt": cur_prompt,
                    "reference": list(cur_words),
                }
        cur_words.clear()

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("# prompt:"):
                flush()
                cur_prompt = line[len("# prompt:"):].strip()
                cur_fmt = None
            elif line.startswith("# user:"):
                m = _HEADER_RE.match(line)
                if m:
                    cur_fmt = m.group("format")
            elif line.startswith("#") or line.strip() == "":
                if not line.startswith("# prompt:") and not line.startswith("# user:"):
                    flush()
                    cur_prompt = None
                    cur_fmt = None
            else:
                parts = line.split()
                if len(parts) >= 2:
                    cur_words.append(parts[1].lower())
    flush()
    return found


def attach_prompts(raw_dir: Path, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach Spanish prompt + English reference to sampled items; drop misses."""
    target = {it["item_hash"] for it in items}
    files = [
        raw_dir / "en_es.slam.20190204.train",
        raw_dir / "en_es.slam.20190204.dev",
        raw_dir / "en_es.slam.20190204.test",
    ]
    found: Dict[str, Dict[str, Any]] = {}
    remaining = set(target)
    for p in files:
        if not p.exists():
            print(f"  Warning: {p} missing, skipping.", flush=True)
            continue
        if not remaining:
            break
        print(f"  Parsing {p.name} (seeking {len(remaining)})...", flush=True)
        t0 = time.time()
        batch = _parse_targeted(p, remaining)
        found.update(batch)
        remaining -= set(batch.keys())
        print(f"    +{len(batch)} hashes in {time.time()-t0:.1f}s, {len(remaining)} left.", flush=True)

    out: List[Dict[str, Any]] = []
    for it in items:
        h = it["item_hash"]
        if h in found and found[h]["prompt"] is not None:
            it = dict(it)
            it["prompt"] = found[h]["prompt"]
            it["reference"] = found[h]["reference"]
            out.append(it)
    return out


def build_bank(
    human_csv: Path,
    raw_dir: Path,
    fmt: str,
    n_items: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Build one instrument bank: sample + attach prompts. Returns valid items."""
    cands = load_candidates(human_csv, fmt)
    print(f"  [{fmt}] {len(cands)} human-calibrated candidates.", flush=True)
    sampled = stratified_sample(cands, min(n_items, len(cands)), seed)
    print(f"  [{fmt}] sampled {len(sampled)} items, attaching prompts...", flush=True)
    valid = attach_prompts(raw_dir, sampled)
    print(f"  [{fmt}] {len(valid)} items with valid Spanish prompt.", flush=True)
    return valid


__all__ = [
    "TEXT_FORMATS",
    "exercise_hash",
    "load_candidates",
    "stratified_sample",
    "attach_prompts",
    "build_bank",
]
