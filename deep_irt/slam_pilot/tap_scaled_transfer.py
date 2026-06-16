"""tap_scaled_transfer.py -- Scaled respondent-agnostic transfer test on reverse_tap items.

Scale-up of the SLAM pilot, restricted to reverse_tap format only (the CLEAN
format identified in the tightened analysis).  reverse_tap items have minimal
phrasing variation because learners select from given English word tiles, so the
Duolingo exact-match grading artifact is structurally absent.

Design:
  - Sample ~500 reverse_tap items STRATIFIED by human difficulty.
    Oversamples the hard tail to span the difficulty range rather than collapsing
    onto the easy mass (>80% of tap items have human_difficulty < 0.10).
  - For each item x each Ollama model: ask the model for the English translation
    (genuine attempt, one-shot, no chain-of-thought, think=False).
  - Grade: token-level overlap against the SLAM reference.
  - LLM item difficulty = mean (1 - token_overlap) across models.
  - Report Spearman (primary), Kendall tau, Pearson vs human difficulty.
  - Gate: Spearman >= ~0.35 => respondent-agnostic premise SUPPORTED on clean data.
  - Artifact-suspect rate: items where LLM difficulty < 0.10 AND human difficulty
    > 0.30.  Should be low for reverse_tap by design.

Outputs (all prefixed tap_scaled_ to avoid clobbering pilot files):
  tap_scaled_results.csv    -- per-item results
  tap_scaled_report.txt     -- full correlations + gate verdict

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="deep_irt;rl/src;ma-irt" \\
        python deep_irt/slam_pilot/tap_scaled_transfer.py \\
        --raw-dir rl/data/slam_raw \\
        --human-csv deep_irt/slam_pilot/outputs/human_difficulty.csv \\
        --out-dir deep_irt/slam_pilot/outputs \\
        --n-items 500 \\
        --seed 7
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import string
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OLLAMA_BASE = "http://localhost:11434"


# ---------------------------------------------------------------------------
# SLAM raw-file parsing
# ---------------------------------------------------------------------------

def _exercise_hash(format_: str, words: List[str]) -> str:
    canonical = format_ + "|" + " ".join(w.lower() for w in words)
    return hashlib.md5(canonical.encode()).hexdigest()[:16]


def _parse_slam_file_targeted(
    path: Path,
    target_hashes: set,
) -> Dict[str, Dict[str, Any]]:
    """Parse one SLAM file, return only items whose hash is in target_hashes."""
    found: Dict[str, Dict[str, Any]] = {}

    cur_prompt: Optional[str] = None
    cur_fmt: Optional[str] = None
    cur_words: List[str] = []

    HEADER_RE = re.compile(r"^# user:(?P<user>\S+)\s+.*?format:(?P<format>\w+)")

    def flush() -> None:
        if cur_words and cur_fmt in ("reverse_translate", "reverse_tap"):
            h = _exercise_hash(cur_fmt, cur_words)
            if h in target_hashes and h not in found:
                found[h] = {
                    "format": cur_fmt,
                    "prompt": cur_prompt,
                    "reference": list(cur_words),
                }
        cur_words.clear()

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("# prompt:"):
                flush()
                cur_prompt = line[len("# prompt:"):].strip()
                cur_fmt = None
            elif line.startswith("# user:"):
                m = HEADER_RE.match(line)
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


def load_targeted_items(
    raw_dir: Path,
    target_hashes: set,
) -> Dict[str, Dict[str, Any]]:
    """Load Spanish prompts and English references for the target item hashes."""
    files = [
        raw_dir / "en_es.slam.20190204.train",
        raw_dir / "en_es.slam.20190204.dev",
        raw_dir / "en_es.slam.20190204.test",
    ]
    found: Dict[str, Dict[str, Any]] = {}
    remaining = set(target_hashes)

    for p in files:
        if not p.exists():
            print(f"  Warning: {p} not found, skipping.")
            continue
        if not remaining:
            break
        print(f"  Parsing {p.name} ({p.stat().st_size/1e6:.0f}MB, seeking {len(remaining)} hashes)...", flush=True)
        t0 = time.time()
        batch = _parse_slam_file_targeted(p, remaining)
        found.update(batch)
        remaining -= set(batch.keys())
        print(f"    Found {len(batch)} new hashes in {time.time()-t0:.1f}s, {len(remaining)} remaining.", flush=True)

    return found


# ---------------------------------------------------------------------------
# Stratified sampling -- oversamples the hard tail
# ---------------------------------------------------------------------------

def stratified_sample_tap(
    candidates: List[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample n reverse_tap items with heavy oversampling of the hard tail.

    reverse_tap difficulty distribution is extremely left-skewed: ~80% of items
    have human_difficulty < 0.10.  A uniform sample would be dominated by the
    easy floor.  This function uses 5 strata with hand-tuned weights to ensure
    the full difficulty range is represented.

    Stratum boundaries and target proportions:
      [0.00, 0.05): ~50% of items -- allocate 20% of sample
      [0.05, 0.10): ~15% of items -- allocate 15% of sample
      [0.10, 0.20): ~12% of items -- allocate 20% of sample
      [0.20, 0.40): ~5%  of items -- allocate 25% of sample
      [0.40, 1.00]: ~1%  of items -- include ALL (or up to 20% of sample)
    """
    import random
    rng = random.Random(seed)

    # Define strata and target proportions (will be rescaled to available items)
    strata_bounds = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]
    target_fracs = [0.20, 0.15, 0.20, 0.25, 0.20]  # desired proportions (sum ~1.0)

    # Assign strata
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for item in candidates:
        d = item["human_difficulty"]
        s = 4
        for i, (lo, hi) in enumerate(strata_bounds):
            if lo <= d < hi:
                s = i
                break
        groups[s].append(item)

    print("  Stratum sizes (available items):")
    for i, (lo, hi) in enumerate(strata_bounds):
        print(f"    [{lo:.2f}-{hi:.2f}): {len(groups[i])}", flush=True)

    # For hardest stratum, include ALL items (there are so few)
    target_per_stratum = [0] * 5
    hard_stratum_items = groups[4]
    target_per_stratum[4] = len(hard_stratum_items)  # include all
    remaining_n = n - target_per_stratum[4]
    remaining_fracs = target_fracs[:4]
    frac_sum = sum(remaining_fracs)

    for i in range(4):
        t = int(round(remaining_n * remaining_fracs[i] / frac_sum))
        # Cap at available items
        t = min(t, len(groups[i]))
        target_per_stratum[i] = t

    # Adjust total to exactly n
    allocated = sum(target_per_stratum)
    diff = n - allocated
    if diff != 0:
        # Adjust from largest available strata first
        order = sorted(range(5), key=lambda s: len(groups[s]) - target_per_stratum[s], reverse=True)
        for i in range(abs(diff)):
            s = order[i % 5]
            if diff > 0 and target_per_stratum[s] < len(groups[s]):
                target_per_stratum[s] += 1
            elif diff < 0 and target_per_stratum[s] > 0:
                target_per_stratum[s] -= 1

    sampled = []
    for s in range(5):
        group = groups[s]
        t = max(0, min(target_per_stratum[s], len(group)))
        if t > 0 and group:
            chosen = rng.sample(group, t)
            sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def list_models() -> List[Dict[str, Any]]:
    url = f"{OLLAMA_BASE}/api/tags"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    return data.get("models", [])


def call_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    """Call Ollama generate API (non-streaming) and return response text."""
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # disable thinking mode for Qwen3.5 models
        "keep_alive": "30m",  # keep model resident to avoid reload stalls between calls
        "options": {
            "temperature": 0.0,
            "num_predict": 80,
            "top_p": 1.0,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[ERROR: {e}]"
    except Exception as e:
        return f"[ERROR: {e}]"


def build_prompt(spanish: str) -> str:
    """Build the one-shot translation prompt."""
    return (
        "Translate this Spanish text to English. "
        "Reply with only the English translation, nothing else.\n\n"
        f"Spanish: {spanish}\n"
        "English:"
    )


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


def grade_token_overlap(response: str, reference: List[str]) -> float:
    """Fraction of reference tokens present in the response."""
    if not reference:
        return 0.0
    ref_tokens = [normalize(w) for w in reference if normalize(w)]
    if not ref_tokens:
        return 0.0
    resp_tokens = set(normalize(response).split())
    matched = sum(1 for t in ref_tokens if t in resp_tokens)
    return matched / len(ref_tokens)


# ---------------------------------------------------------------------------
# Model selection: widest ability spread, include smallest/weakest
# ---------------------------------------------------------------------------

def select_models(all_models: List[Dict[str, Any]], n_target: int = 5) -> List[str]:
    """Select n_target models spanning sizes; always include the smallest/weakest."""
    sorted_models = sorted(all_models, key=lambda m: m.get("size", 0))
    names = [m["name"] for m in sorted_models]

    if len(names) <= n_target:
        return names

    selected = {names[0]}  # always include smallest (weakest, most discriminating errors)
    step = (len(names) - 1) / (n_target - 1)
    for i in range(1, n_target):
        idx = min(int(round(i * step)), len(names) - 1)
        selected.add(names[idx])

    return [n for n in names if n in selected]


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
            dx_ = x[i] - x[j]
            dy_ = y[i] - y[j]
            if dx_ == 0.0 and dy_ == 0.0:
                tied_xy += 1
            elif dx_ == 0.0:
                tied_x += 1
            elif dy_ == 0.0:
                tied_y += 1
            elif (dx_ > 0) == (dy_ > 0):
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


def std(vals: List[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Main scaled test
# ---------------------------------------------------------------------------

def run_scaled_test(
    raw_dir: Path,
    human_csv: Path,
    out_dir: Path,
    n_items: int = 500,
    seed: int = 7,
    n_models_target: int = 5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_run_start = time.time()

    # Step 1: Models
    print("\n=== Step 1: Checking Ollama models ===", flush=True)
    try:
        all_models = list_models()
    except Exception as e:
        print(f"BLOCKER: Cannot reach Ollama at {OLLAMA_BASE}: {e}")
        sys.exit(1)

    if not all_models:
        print("BLOCKER: No models found at Ollama.")
        sys.exit(1)

    print(f"Available models ({len(all_models)}) sorted by size:")
    for m in sorted(all_models, key=lambda x: x.get("size", 0)):
        print(f"  {m['name']:55s}  {m.get('size',0)/1e9:.2f} GB", flush=True)

    _env_models = os.environ.get("RQ3_MODELS", "").strip()
    if _env_models:
        selected_models = [m.strip() for m in _env_models.split(",") if m.strip()]
    else:
        selected_models = select_models(all_models, n_target=n_models_target)
    print(f"\nSelected {len(selected_models)} models (widest spread, smallest=weakest included):")
    for name in selected_models:
        sz = next((m.get("size", 0) for m in all_models if m["name"] == name), 0)
        print(f"  {name:55s}  {sz/1e9:.2f} GB", flush=True)

    # Step 2: Human difficulty -- reverse_tap only
    print("\n=== Step 2: Loading human difficulty CSV (reverse_tap only) ===", flush=True)
    tap_candidates: List[Dict[str, Any]] = []
    with human_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["format"] == "reverse_tap":
                tap_candidates.append({
                    "item_hash": row["item_hash"],
                    "human_difficulty": float(row["human_difficulty"]),
                    "format": "reverse_tap",
                })
    print(f"  reverse_tap items in human_difficulty.csv: {len(tap_candidates):,}", flush=True)

    if len(tap_candidates) < n_items:
        print(f"  Warning: only {len(tap_candidates)} candidates; reducing n_items to {len(tap_candidates)}.")
        n_items = len(tap_candidates)

    # Step 3: Stratified sample (heavy oversampling of hard tail)
    print(f"\n=== Step 3: Stratified sample ({n_items} items, seed={seed}) ===", flush=True)
    sampled = stratified_sample_tap(tap_candidates, n_items, seed=seed)
    print(f"  Sampled {len(sampled)} items.", flush=True)

    diffs = [item["human_difficulty"] for item in sampled]
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    print("  Human difficulty distribution of sample:")
    for i in range(len(bins) - 1):
        cnt = sum(1 for d in diffs if bins[i] <= d < bins[i + 1])
        bar = "#" * (cnt * 30 // max(1, len(sampled) // 10))
        print(f"    [{bins[i]:.2f}-{min(bins[i+1],1.0):.2f}): {cnt:4d}  {bar}", flush=True)

    # Step 4: Load SLAM prompts for sampled items
    print("\n=== Step 4: Loading SLAM prompts for sampled items ===", flush=True)
    target_hashes = {item["item_hash"] for item in sampled}
    slam_items = load_targeted_items(raw_dir, target_hashes)

    valid_sampled = []
    n_no_prompt = 0
    for item in sampled:
        h = item["item_hash"]
        if h in slam_items and slam_items[h]["prompt"] is not None:
            item["prompt"] = slam_items[h]["prompt"]
            item["reference"] = slam_items[h]["reference"]
            valid_sampled.append(item)
        else:
            n_no_prompt += 1

    print(f"  Items with valid Spanish prompt: {len(valid_sampled)}", flush=True)
    print(f"  Items excluded (no prompt found): {n_no_prompt}", flush=True)

    if not valid_sampled:
        print("BLOCKER: No items with valid prompts found.")
        sys.exit(1)

    # Step 5: LLM attempts (with per-model checkpointing for resumability)
    n_models = len(selected_models)
    n_valid = len(valid_sampled)
    total_calls = n_valid * n_models
    print(
        f"\n=== Step 5: LLM attempts ({n_valid} items x {n_models} models = {total_calls} calls) ===",
        flush=True,
    )

    # Per-model checkpoint cache. Keyed on (seed, n_items) so resume uses the same
    # sample. Each completed model's responses are written as JSON, so a restart
    # skips already-finished models rather than re-running 1000s of calls.
    cache_dir = out_dir / f"tap_scaled_cache_seed{seed}_n{n_items}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(name: str) -> str:
        return name.replace("/", "_").replace(":", "_")

    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        item["item_hash"]: {} for item in valid_sampled
    }

    call_count = 0
    t_start = time.time()

    def _write_cache(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
        """Atomic JSON write so a kill mid-flush cannot corrupt the cache."""
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as cf:
            json.dump(data, cf)
        tmp.replace(path)

    for model_name in selected_models:
        cache_path = cache_dir / f"{_safe_name(model_name)}.json"

        # Resume: load whatever items are already cached for this model.
        # Incremental caching (every 50 items) means a partial cache is common
        # after a kill; we keep finished items and only run the remainder.
        model_cache: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as cf:
                    model_cache = json.load(cf)
            except (json.JSONDecodeError, ValueError):
                print(f"\n  Model: {model_name}  [corrupt cache, restarting model]", flush=True)
                model_cache = {}

        done_hashes = set(model_cache.keys())
        # Seed in-memory results from cache
        for h, r in model_cache.items():
            if h in results:
                results[h][model_name] = r

        if len(done_hashes) >= n_valid:
            call_count += n_valid
            mean_ov_c = sum(r["overlap"] for r in model_cache.values()) / len(model_cache)
            print(
                f"\n  Model: {model_name}  [CACHED, skipping] "
                f"mean_error={1.0 - mean_ov_c:.4f}  n={len(model_cache)}",
                flush=True,
            )
            continue

        if done_hashes:
            print(f"\n  Model: {model_name}  [resuming from {len(done_hashes)}/{n_valid} cached]", flush=True)
            call_count += len(done_hashes)
        else:
            print(f"\n  Model: {model_name}", flush=True)

        model_overlaps = [r["overlap"] for r in model_cache.values()]
        since_flush = 0

        for item in valid_sampled:
            h = item["item_hash"]
            if h in done_hashes:
                continue
            spanish = item["prompt"]
            reference = item["reference"]

            prompt = build_prompt(spanish)
            response = call_ollama(model_name, prompt, timeout=90)

            # Strip echoed prefix if model repeats "English:" etc.
            resp_clean = response
            for prefix in ["English:", "Translation:", "Answer:", "english:", "translation:"]:
                if resp_clean.lower().startswith(prefix.lower()):
                    resp_clean = resp_clean[len(prefix):].strip()
                    break

            overlap = grade_token_overlap(resp_clean, reference)
            error = 1.0 - overlap

            entry = {
                "response": resp_clean,
                "overlap": overlap,
                "error": error,
            }
            results[h][model_name] = entry
            model_cache[h] = entry
            model_overlaps.append(overlap)

            call_count += 1
            since_flush += 1
            if since_flush >= 50:
                _write_cache(cache_path, model_cache)
                since_flush = 0
                elapsed = time.time() - t_start
                rate = call_count / elapsed if elapsed > 0 else 0
                remaining_calls = (total_calls - call_count) / rate if rate > 0 else 0
                print(
                    f"    [{call_count}/{total_calls}] elapsed={elapsed:.0f}s "
                    f"est_remaining={remaining_calls:.0f}s  (checkpoint {len(model_cache)}/{n_valid})",
                    flush=True,
                )

        # Final flush for this model
        _write_cache(cache_path, model_cache)

        mean_ov = sum(model_overlaps) / len(model_overlaps) if model_overlaps else float("nan")
        mean_err = 1.0 - mean_ov
        print(
            f"    DONE: mean_error={mean_err:.4f}  mean_overlap={mean_ov:.4f}  "
            f"n={len(model_overlaps)}  [cached -> {cache_path.name}]",
            flush=True,
        )

    elapsed_total = time.time() - t_start
    print(f"\n  All attempts done in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min).", flush=True)

    # Step 6: Per-item LLM difficulty
    print("\n=== Step 6: Per-item LLM difficulty ===", flush=True)
    item_llm_difficulty: Dict[str, float] = {}

    for item in valid_sampled:
        h = item["item_hash"]
        model_results = results[h]
        if not model_results:
            continue
        errors = [r["error"] for r in model_results.values()]
        item_llm_difficulty[h] = sum(errors) / len(errors)

    llm_diffs = list(item_llm_difficulty.values())
    llm_mean = sum(llm_diffs) / len(llm_diffs) if llm_diffs else float("nan")
    llm_min = min(llm_diffs) if llm_diffs else float("nan")
    llm_max = max(llm_diffs) if llm_diffs else float("nan")
    llm_std = std(llm_diffs) if llm_diffs else float("nan")

    print(f"  LLM difficulty: mean={llm_mean:.4f}  std={llm_std:.4f}  "
          f"min={llm_min:.4f}  max={llm_max:.4f}", flush=True)

    # LLM difficulty histogram
    llm_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    llm_bin_counts = [0] * (len(llm_bins) - 1)
    for d in llm_diffs:
        for i in range(len(llm_bins) - 1):
            if llm_bins[i] <= d < llm_bins[i + 1]:
                llm_bin_counts[i] += 1
                break
    print("  LLM difficulty histogram:")
    max_cnt = max(llm_bin_counts, default=1)
    for i, cnt in enumerate(llm_bin_counts):
        lo, hi = llm_bins[i], llm_bins[i + 1]
        bar = "#" * (cnt * 30 // max(max_cnt, 1))
        print(f"    [{lo:.1f}-{min(hi,1.0):.1f}): {cnt:4d}  {bar}", flush=True)

    # Check variance
    n_zero = sum(1 for d in llm_diffs if d < 1e-6)
    n_one = sum(1 for d in llm_diffs if d > 1 - 1e-6)
    pct_zero = 100 * n_zero / len(llm_diffs) if llm_diffs else 0
    pct_one = 100 * n_one / len(llm_diffs) if llm_diffs else 0
    print(f"  Degenerate items: {n_zero} zero-error ({pct_zero:.1f}%), {n_one} max-error ({pct_one:.1f}%)", flush=True)

    # Step 7: Per-model stats (ceiling check)
    print("\n=== Step 7: Per-model error rates (ceiling check) ===", flush=True)
    per_model_stats: Dict[str, Dict[str, float]] = {}
    for model_name in selected_models:
        errs, ovlps = [], []
        for h, mr in results.items():
            if model_name in mr:
                errs.append(mr[model_name]["error"])
                ovlps.append(mr[model_name]["overlap"])
        me = sum(errs) / len(errs) if errs else float("nan")
        mo = sum(ovlps) / len(ovlps) if ovlps else float("nan")
        sd = std(errs) if errs else float("nan")
        per_model_stats[model_name] = {
            "mean_error": me, "mean_overlap": mo, "std_error": sd, "n": len(errs)
        }
        print(f"  {model_name:55s}  err={me:.4f}  sd={sd:.4f}  overlap={mo:.4f}  n={len(errs)}", flush=True)

    all_mean_errors = [s["mean_error"] for s in per_model_stats.values() if not math.isnan(s["mean_error"])]
    ceiling_flag = all(e < 0.10 for e in all_mean_errors) if all_mean_errors else False
    floor_flag = all(e > 0.90 for e in all_mean_errors) if all_mean_errors else False

    error_spread = max(all_mean_errors) - min(all_mean_errors) if len(all_mean_errors) >= 2 else 0.0
    print(f"  Model error spread (max - min): {error_spread:.4f}", flush=True)
    print(f"  Ceiling (all models < 10% error): {'YES' if ceiling_flag else 'NO'}", flush=True)
    print(f"  Floor   (all models > 90% error): {'YES' if floor_flag else 'NO'}", flush=True)

    # Step 8: Artifact rate (should be low for reverse_tap)
    print("\n=== Step 8: Artifact-suspect rate ===", flush=True)
    paired = [item for item in valid_sampled if item["item_hash"] in item_llm_difficulty]
    artifact_items = [
        item for item in paired
        if item_llm_difficulty[item["item_hash"]] < 0.10 and item["human_difficulty"] > 0.30
    ]
    n_artifact = len(artifact_items)
    artifact_rate = n_artifact / len(paired) if paired else 0.0
    print(f"  Artifact-suspect (LLM<0.10 AND human>0.30): {n_artifact} / {len(paired)} ({100*artifact_rate:.1f}%)", flush=True)
    print("  (For reverse_tap, this should be low by design -- minimal phrasing variation)", flush=True)
    if artifact_items:
        print("  Examples:")
        for it in sorted(artifact_items, key=lambda x: x["human_difficulty"], reverse=True)[:5]:
            ref = " ".join(it.get("reference", []))[:60]
            llm_d = item_llm_difficulty[it["item_hash"]]
            print(f"    {it['item_hash']}  llm={llm_d:.4f}  human={it['human_difficulty']:.4f}  ref: {ref!r}", flush=True)

    # Step 9: Correlations
    print("\n=== Step 9: Correlations ===", flush=True)
    hd_vals = [item["human_difficulty"] for item in paired]
    llm_vals = [item_llm_difficulty[item["item_hash"]] for item in paired]
    n_paired = len(hd_vals)

    sp_r = spearman(hd_vals, llm_vals)
    kt_r = kendall_tau(hd_vals, llm_vals)
    pe_r = pearson(hd_vals, llm_vals)

    print(f"  n_paired  = {n_paired}")
    print(f"  Spearman r = {sp_r:+.4f}  (primary)")
    print(f"  Kendall tau= {kt_r:+.4f}")
    print(f"  Pearson  r = {pe_r:+.4f}", flush=True)

    # Also report for hard subset (human_difficulty > 0.10)
    hard_paired = [it for it in paired if it["human_difficulty"] > 0.10]
    if len(hard_paired) >= 10:
        hd_hard = [it["human_difficulty"] for it in hard_paired]
        llm_hard = [item_llm_difficulty[it["item_hash"]] for it in hard_paired]
        sp_hard = spearman(hd_hard, llm_hard)
        kt_hard = kendall_tau(hd_hard, llm_hard)
        pe_hard = pearson(hd_hard, llm_hard)
        print(f"\n  Hard subset (human_difficulty > 0.10, n={len(hard_paired)}):")
        print(f"    Spearman r = {sp_hard:+.4f}")
        print(f"    Kendall tau= {kt_hard:+.4f}")
        print(f"    Pearson  r = {pe_hard:+.4f}", flush=True)
    else:
        sp_hard = kt_hard = pe_hard = float("nan")
        print("  Hard subset: too few items.", flush=True)

    # Step 10: Gate verdict
    print("\n=== Step 10: Gate verdict ===", flush=True)

    if ceiling_flag:
        verdict = (
            f"ELICITATION CEILING: all models have mean error < 10%. "
            f"Spearman = {sp_r:.3f}. LLM difficulty variance is insufficient. "
            "Try harder prompts or perplexity-based scoring."
        )
        gate_passed = False
    elif llm_std < 0.05:
        verdict = (
            f"DEGENERATE VARIANCE: LLM difficulty std = {llm_std:.4f} < 0.05. "
            f"Spearman = {sp_r:.3f}. The LLM difficulty signal is too compressed to rank items. "
            f"Model homogeneity or ceiling/floor collapse is likely the limiting factor. "
            f"Error spread across models: {error_spread:.4f}."
        )
        gate_passed = False
    elif math.isnan(sp_r):
        verdict = "INDETERMINATE: Spearman could not be computed."
        gate_passed = False
    elif sp_r >= 0.35:
        verdict = (
            f"GATE PASSED: Spearman r = {sp_r:.3f} >= 0.35 threshold. "
            f"LLM item difficulty predicts human item difficulty on {n_paired} "
            f"reverse_tap items with non-degenerate variance (std={llm_std:.4f}). "
            f"Artifact-suspect rate = {100*artifact_rate:.1f}% (expected low for tap). "
            "The respondent-agnostic premise is SUPPORTED on clean real data. "
            "This is the keystone result."
        )
        gate_passed = True
    elif sp_r >= 0.20:
        verdict = (
            f"WEAK SIGNAL: Spearman r = {sp_r:.3f} (0.20-0.35 range). "
            f"n={n_paired}, LLM difficulty std={llm_std:.4f}. "
            f"Some transfer signal present but below the robust gate threshold. "
            f"Artifact-suspect rate = {100*artifact_rate:.1f}%. "
            f"Possible limiting factors: model homogeneity (error spread={error_spread:.4f}), "
            "residual grading noise, or insufficient hard items."
        )
        gate_passed = False
    else:
        verdict = (
            f"NULL: Spearman r = {sp_r:.3f} < 0.20. "
            f"n={n_paired}, LLM difficulty std={llm_std:.4f}, "
            f"error spread={error_spread:.4f}. "
            "LLM difficulty does not predict human difficulty at this threshold. "
            f"Artifact-suspect rate = {100*artifact_rate:.1f}%."
        )
        gate_passed = False

    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  GATE: {'PASSED' if gate_passed else 'NOT PASSED'}", flush=True)

    total_elapsed = time.time() - t_run_start
    print(f"\n  Total run time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)

    # Step 11: Write outputs
    print("\n=== Step 11: Writing outputs ===", flush=True)

    # tap_scaled_results.csv
    results_path = out_dir / "tap_scaled_results.csv"
    base_fields = ["item_hash", "format", "prompt", "reference",
                   "human_difficulty", "llm_difficulty"]
    model_fields = []
    for m in selected_models:
        safe = m.replace("/", "_").replace(":", "_")
        model_fields += [f"{safe}_response", f"{safe}_overlap", f"{safe}_error"]

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + model_fields, extrasaction="ignore")
        writer.writeheader()
        for item in paired:
            h = item["item_hash"]
            row: Dict[str, Any] = {
                "item_hash": h,
                "format": item["format"],
                "prompt": item.get("prompt", ""),
                "reference": " ".join(item.get("reference", [])),
                "human_difficulty": item["human_difficulty"],
                "llm_difficulty": item_llm_difficulty.get(h, ""),
            }
            for m in selected_models:
                safe = m.replace("/", "_").replace(":", "_")
                mr = results[h].get(m, {})
                row[f"{safe}_response"] = mr.get("response", "")
                row[f"{safe}_overlap"] = mr.get("overlap", "")
                row[f"{safe}_error"] = mr.get("error", "")
            writer.writerow(row)
    print(f"  Saved {len(paired)} rows -> {results_path}", flush=True)

    # tap_scaled_report.txt
    report_path = out_dir / "tap_scaled_report.txt"

    hd_hist = [0] * (len(bins) - 1)
    for d in hd_vals:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                hd_hist[i] += 1
                break

    lines = [
        "SLAM Scaled Transfer Test -- reverse_tap CLEAN FORMAT",
        "=" * 60,
        "",
        f"Date:          {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed:          {seed}",
        f"Format:        reverse_tap ONLY (clean: tile selection, minimal phrasing variation)",
        f"Items sampled: {n_items}  -> valid (Spanish prompt found): {len(valid_sampled)}",
        f"Items paired:  {n_paired}  (excluded: {n_no_prompt} no prompt found)",
        f"Models:        {len(selected_models)}",
        f"Total calls:   {total_calls}",
        f"Run time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)",
        "",
        "--- Models used (size order, widest spread) ---",
    ]
    for name in selected_models:
        sz = next((m.get("size", 0) for m in all_models if m["name"] == name), 0)
        lines.append(f"  {name:55s}  {sz/1e9:.2f} GB")

    lines += [
        "",
        "--- Per-model error rates (ceiling/floor check) ---",
        f"  {'Model':55s}  mean_error  std_error  mean_overlap  n",
    ]
    for name in selected_models:
        s = per_model_stats[name]
        lines.append(
            f"  {name:55s}  {s['mean_error']:.4f}      {s['std_error']:.4f}     "
            f"{s['mean_overlap']:.4f}        {s['n']}"
        )
    lines.append(f"  Model error spread (max - min): {error_spread:.4f}")
    lines.append(f"  Ceiling (all models < 10% error): {'YES' if ceiling_flag else 'NO'}")
    lines.append(f"  Floor   (all models > 90% error): {'YES' if floor_flag else 'NO'}")

    lines += [
        "",
        "--- LLM difficulty distribution (1 - token_overlap) ---",
        f"  mean={llm_mean:.4f}  std={llm_std:.4f}  min={llm_min:.4f}  max={llm_max:.4f}",
        f"  Zero-error items: {n_zero} ({pct_zero:.1f}%)  Max-error items: {n_one} ({pct_one:.1f}%)",
        "  Histogram:",
    ]
    for i, cnt in enumerate(llm_bin_counts):
        lo, hi = llm_bins[i], llm_bins[i + 1]
        bar = "#" * (cnt * 30 // max(max_cnt, 1))
        lines.append(f"    [{lo:.1f}-{min(hi,1.0):.1f}): {cnt:4d}  {bar}")

    lines += [
        "",
        "--- Human difficulty distribution of sampled items ---",
        "  Histogram:",
    ]
    for i, cnt in enumerate(hd_hist):
        lo, hi = bins[i], bins[i + 1]
        lines.append(f"    [{lo:.2f}-{min(hi,1.0):.2f}): {cnt:4d}")

    lines += [
        "",
        "--- Artifact-suspect rate (LLM<0.10 AND human>0.30) ---",
        f"  {n_artifact} / {n_paired} ({100*artifact_rate:.1f}%)",
        "  (For reverse_tap, expected low: tile selection eliminates most phrasing artifacts)",
    ]

    lines += [
        "",
        "--- Correlations (LLM difficulty vs human difficulty) ---",
        f"  n = {n_paired}",
        f"  Spearman r  = {sp_r:+.4f}  (primary)",
        f"  Kendall tau = {kt_r:+.4f}",
        f"  Pearson  r  = {pe_r:+.4f}",
    ]
    if not math.isnan(sp_hard):
        lines += [
            f"",
            f"  Hard subset (human_difficulty > 0.10, n={len(hard_paired)}):",
            f"    Spearman r  = {sp_hard:+.4f}",
            f"    Kendall tau = {kt_hard:+.4f}",
            f"    Pearson  r  = {pe_hard:+.4f}",
            f"  NOTE: hard-subset correlation is selection-biased upward (easy floor removed).",
        ]

    lines += [
        "",
        "--- Gate Verdict ---",
        f"  Gate threshold: Spearman >= 0.35 with non-degenerate variance",
        f"  VERDICT: {verdict}",
        f"  GATE: {'PASSED' if gate_passed else 'NOT PASSED'}",
        "",
        "--- Output files ---",
        f"  {results_path}",
        f"  {report_path}",
    ]

    report_text = "\n".join(lines) + "\n"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Saved report -> {report_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(report_text, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scaled LLM transfer test on reverse_tap SLAM items."
    )
    parser.add_argument("--raw-dir", default="rl/data/slam_raw")
    parser.add_argument("--human-csv", default="deep_irt/slam_pilot/outputs/human_difficulty.csv")
    parser.add_argument("--out-dir", default="deep_irt/slam_pilot/outputs")
    parser.add_argument("--n-items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-models", type=int, default=5)
    args = parser.parse_args(argv)

    run_scaled_test(
        raw_dir=Path(args.raw_dir),
        human_csv=Path(args.human_csv),
        out_dir=Path(args.out_dir),
        n_items=args.n_items,
        seed=args.seed,
        n_models_target=args.n_models,
    )


if __name__ == "__main__":
    main()
