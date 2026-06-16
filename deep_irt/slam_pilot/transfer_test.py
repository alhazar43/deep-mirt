"""transfer_test.py -- Respondent-agnostic LLM transfer test for SLAM items.

Design:
  - Parse SLAM raw files to recover (Spanish prompt, English reference) pairs
    for reverse_translate and reverse_tap items only.
  - Stratified sample ~200 items by human difficulty to span the full range.
  - For each item x each Ollama model: ask the model to actually attempt the
    translation (one-shot, no chain-of-thought).
  - Grade: (a) normalized exact match, (b) token-level overlap.
  - LLM item difficulty = mean per-attempt error (1 - token overlap) across
    all model attempts on that item.
  - Correlate LLM difficulty with human difficulty (Spearman + Pearson).
  - Report ceiling check and gate verdict.

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="deep_irt;rl/src;ma-irt" \\
        python deep_irt/slam_pilot/transfer_test.py \\
        --raw-dir rl/data/slam_raw \\
        --human-csv deep_irt/slam_pilot/outputs/human_difficulty.csv \\
        --out-dir deep_irt/slam_pilot/outputs \\
        --n-items 200 \\
        --seed 42

Outputs (all under --out-dir):
    transfer_results.csv   -- per-item results
    transfer_report.txt    -- full report with correlations and verdict
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
# SLAM raw-file parsing -- FAST: stops once target hashes found
# ---------------------------------------------------------------------------

def _exercise_hash(format_: str, words: List[str]) -> str:
    """16-hex MD5 of (format, lowercased word sequence)."""
    canonical = format_ + "|" + " ".join(w.lower() for w in words)
    return hashlib.md5(canonical.encode()).hexdigest()[:16]


def _parse_slam_file_targeted(
    path: Path,
    target_hashes: set,
) -> Dict[str, Dict[str, Any]]:
    """Parse one SLAM file, returning only items whose hash is in target_hashes.

    Returns dict: hash -> {format, prompt, reference}
    """
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
    """Load Spanish prompts and English references for the target item hashes.

    Reads SLAM files in order (train, dev, test) and stops early for each
    hash once found. Returns dict: hash -> {format, prompt, reference}.
    """
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
        print(f"  Parsing {p.name} ({p.stat().st_size/1e6:.0f}MB, seeking {len(remaining)} hashes)...")
        t0 = time.time()
        batch = _parse_slam_file_targeted(p, remaining)
        found.update(batch)
        remaining -= set(batch.keys())
        print(f"    Found {len(batch)} new hashes in {time.time()-t0:.1f}s, {len(remaining)} remaining.")

    return found


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    candidates: List[Dict[str, Any]],
    n: int,
    seed: int,
    n_strata: int = 10,
) -> List[Dict[str, Any]]:
    """Sample n items stratified by human_difficulty, oversampling the hard tail."""
    import random
    rng = random.Random(seed)

    cands = sorted(candidates, key=lambda x: x["human_difficulty"])
    total = len(cands)

    for i, item in enumerate(cands):
        item["_stratum"] = min(int(n_strata * i / total), n_strata - 1)

    strata_groups: Dict[int, List[Dict]] = defaultdict(list)
    for item in cands:
        strata_groups[item["_stratum"]].append(item)

    # Weight harder strata (>= 4, i.e. difficulty >= 0.4) by 2x to oversample
    stratum_weights = []
    for s in range(n_strata):
        count = len(strata_groups.get(s, []))
        weight = 2.0 if s >= 4 else 1.0
        stratum_weights.append(weight * count)

    total_weight = sum(stratum_weights)
    target_per_stratum = []
    for w in stratum_weights:
        t = int(round(n * w / total_weight)) if total_weight > 0 else 0
        target_per_stratum.append(t)

    # Adjust to hit exactly n
    allocated = sum(target_per_stratum)
    diff = n - allocated
    if diff != 0:
        sorted_strata = sorted(range(n_strata), key=lambda s: target_per_stratum[s], reverse=True)
        for i in range(abs(diff)):
            s = sorted_strata[i % n_strata]
            target_per_stratum[s] += 1 if diff > 0 else -1

    sampled = []
    for s in range(n_strata):
        group = strata_groups.get(s, [])
        t = max(0, min(target_per_stratum[s], len(group)))
        if t > 0 and group:
            sampled.extend(rng.sample(group, t))

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


def call_ollama(model: str, prompt: str, timeout: int = 45) -> str:
    """Call Ollama generate API (non-streaming) and return response text.

    Uses think=False to disable chain-of-thought for Qwen3.5 thinking models,
    so the response field contains the actual answer rather than being empty.
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # disable thinking mode for Qwen3.5 models
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


def build_prompt(spanish: str, fmt: str) -> str:
    """Build the one-shot translation prompt."""
    return (
        f"Translate this Spanish text to English. "
        f"Reply with only the English translation, nothing else.\n\n"
        f"Spanish: {spanish}\n"
        f"English:"
    )


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


def grade_exact(response: str, reference: List[str]) -> int:
    ref_str = normalize(" ".join(reference))
    resp_str = normalize(response)
    return int(resp_str == ref_str)


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
# Model selection: span sizes, always include smallest
# ---------------------------------------------------------------------------

def select_models(all_models: List[Dict[str, Any]], n_target: int = 5) -> List[str]:
    """Select n_target models spanning sizes, always including smallest."""
    sorted_models = sorted(all_models, key=lambda m: m.get("size", 0))
    names = [m["name"] for m in sorted_models]

    if len(names) <= n_target:
        return names

    selected = {names[0]}  # always include smallest
    step = (len(names) - 1) / (n_target - 1)
    for i in range(1, n_target):
        idx = min(int(round(i * step)), len(names) - 1)
        selected.add(names[idx])

    # Return in size order
    return [n for n in names if n in selected]


# ---------------------------------------------------------------------------
# Statistics helpers
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


# ---------------------------------------------------------------------------
# Main pilot
# ---------------------------------------------------------------------------

def run_pilot(
    raw_dir: Path,
    human_csv: Path,
    out_dir: Path,
    n_items: int = 200,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Models
    print("\n=== Step 1: Checking Ollama models ===", flush=True)
    try:
        all_models = list_models()
    except Exception as e:
        print(f"BLOCKER: Cannot reach Ollama at {OLLAMA_BASE}: {e}")
        sys.exit(1)

    if not all_models:
        print("BLOCKER: No models found.")
        sys.exit(1)

    print(f"Available models ({len(all_models)}):")
    for m in sorted(all_models, key=lambda x: x.get("size", 0)):
        print(f"  {m['name']:50s}  {m.get('size',0)/1e9:.2f} GB", flush=True)

    selected_models = select_models(all_models, n_target=4)
    print(f"\nSelected {len(selected_models)} models:")
    for name in selected_models:
        sz = next((m.get("size", 0) for m in all_models if m["name"] == name), 0)
        print(f"  {name:50s}  {sz/1e9:.2f} GB", flush=True)

    # Step 2: Human difficulty
    print("\n=== Step 2: Loading human difficulty CSV ===", flush=True)
    human_items: Dict[str, float] = {}
    human_formats: Dict[str, str] = {}
    with human_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["format"] in ("reverse_translate", "reverse_tap"):
                h = row["item_hash"]
                human_items[h] = float(row["human_difficulty"])
                human_formats[h] = row["format"]

    print(f"  Text-format items: {len(human_items):,}", flush=True)

    # Step 3: Stratified sample FIRST, then only parse what we need
    print(f"\n=== Step 3: Stratified sample ({n_items} items, seed={seed}) ===", flush=True)
    candidates = [
        {"item_hash": h, "human_difficulty": d, "format": human_formats[h]}
        for h, d in human_items.items()
    ]
    if len(candidates) < n_items:
        n_items = len(candidates)
        print(f"  Warning: only {len(candidates)} candidates; reducing n_items to {n_items}.")

    sampled = stratified_sample(candidates, n_items, seed=seed)
    print(f"  Sampled {len(sampled)} items.")

    diffs = [item["human_difficulty"] for item in sampled]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_counts = [0] * (len(bins) - 1)
    for d in diffs:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                bin_counts[i] += 1
                break
    print("  Human difficulty distribution of sample:")
    for i, cnt in enumerate(bin_counts):
        print(f"    [{bins[i]:.1f}-{min(bins[i+1], 1.0):.1f}): {cnt:3d}", flush=True)

    # Step 4: Load SLAM prompts only for sampled items
    print("\n=== Step 4: Loading SLAM prompts for sampled items ===", flush=True)
    target_hashes = {item["item_hash"] for item in sampled}
    slam_items = load_targeted_items(raw_dir, target_hashes)

    # Filter to items with a valid prompt
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

    # Step 5: LLM attempts
    n_models = len(selected_models)
    n_valid = len(valid_sampled)
    total_calls = n_valid * n_models
    print(f"\n=== Step 5: LLM attempts ({n_valid} items x {n_models} models = {total_calls} calls) ===", flush=True)

    # results[item_hash][model_name] = {response, exact, overlap, error}
    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        item["item_hash"]: {} for item in valid_sampled
    }

    call_count = 0
    t_start = time.time()

    for model_name in selected_models:
        print(f"\n  Model: {model_name}", flush=True)
        model_errors = []
        model_overlaps = []
        model_exacts = []

        for item in valid_sampled:
            h = item["item_hash"]
            spanish = item["prompt"]
            reference = item["reference"]
            fmt = item["format"]

            prompt = build_prompt(spanish, fmt)
            response = call_ollama(model_name, prompt, timeout=60)

            # Strip echoed prefix if model repeats "English:" etc.
            resp_clean = response
            for prefix in ["English:", "Translation:", "Answer:", "english:"]:
                if resp_clean.lower().startswith(prefix.lower()):
                    resp_clean = resp_clean[len(prefix):].strip()
                    break

            exact = grade_exact(resp_clean, reference)
            overlap = grade_token_overlap(resp_clean, reference)
            error = 1.0 - overlap

            results[h][model_name] = {
                "response": resp_clean,
                "exact": exact,
                "overlap": overlap,
                "error": error,
            }
            model_errors.append(error)
            model_overlaps.append(overlap)
            model_exacts.append(exact)

            call_count += 1
            if call_count % 25 == 0:
                elapsed = time.time() - t_start
                rate = call_count / elapsed
                remaining = (total_calls - call_count) / rate if rate > 0 else 0
                print(
                    f"    [{call_count}/{total_calls}] elapsed={elapsed:.0f}s "
                    f"est_remaining={remaining:.0f}s",
                    flush=True,
                )

        mean_err = sum(model_errors) / len(model_errors) if model_errors else float("nan")
        mean_overlap = sum(model_overlaps) / len(model_overlaps) if model_overlaps else float("nan")
        exact_rate = sum(model_exacts) / len(model_exacts) if model_exacts else float("nan")
        print(
            f"    DONE: mean_error={mean_err:.4f}  mean_overlap={mean_overlap:.4f}  "
            f"exact_match_rate={exact_rate:.4f}  n={len(model_errors)}",
            flush=True,
        )

    elapsed_total = time.time() - t_start
    print(f"\n  All attempts done in {elapsed_total:.1f}s.", flush=True)

    # Step 6: Per-item LLM difficulty
    print("\n=== Step 6: Per-item LLM difficulty ===", flush=True)
    item_llm_difficulty: Dict[str, float] = {}
    item_llm_exact_err: Dict[str, float] = {}

    for item in valid_sampled:
        h = item["item_hash"]
        model_results = results[h]
        if not model_results:
            continue
        errors = [r["error"] for r in model_results.values()]
        exact_errs = [1 - r["exact"] for r in model_results.values()]
        item_llm_difficulty[h] = sum(errors) / len(errors)
        item_llm_exact_err[h] = sum(exact_errs) / len(exact_errs)

    llm_diffs = list(item_llm_difficulty.values())
    llm_mean = sum(llm_diffs) / len(llm_diffs) if llm_diffs else float("nan")
    llm_min = min(llm_diffs) if llm_diffs else float("nan")
    llm_max = max(llm_diffs) if llm_diffs else float("nan")
    llm_std = math.sqrt(
        sum((x - llm_mean) ** 2 for x in llm_diffs) / len(llm_diffs)
    ) if llm_diffs else float("nan")

    print(f"  LLM difficulty: mean={llm_mean:.4f}  std={llm_std:.4f}  "
          f"min={llm_min:.4f}  max={llm_max:.4f}", flush=True)

    # LLM difficulty histogram
    llm_bin_counts = [0] * 10
    for d in llm_diffs:
        b = min(int(d * 10), 9)
        llm_bin_counts[b] += 1
    print("  LLM difficulty histogram:")
    for i, cnt in enumerate(llm_bin_counts):
        lo, hi = i * 0.1, (i + 1) * 0.1
        bar = "#" * (cnt * 30 // max(llm_bin_counts, default=1))
        print(f"    [{lo:.1f}-{hi:.1f}): {cnt:3d}  {bar}", flush=True)

    # Step 7: Per-model stats
    print("\n=== Step 7: Per-model error rates (ceiling check) ===", flush=True)
    per_model_stats: Dict[str, Dict[str, float]] = {}
    for model_name in selected_models:
        errs, ovlps, exacts = [], [], []
        for h, mr in results.items():
            if model_name in mr:
                errs.append(mr[model_name]["error"])
                ovlps.append(mr[model_name]["overlap"])
                exacts.append(mr[model_name]["exact"])
        me = sum(errs) / len(errs) if errs else float("nan")
        mo = sum(ovlps) / len(ovlps) if ovlps else float("nan")
        ex = sum(exacts) / len(exacts) if exacts else float("nan")
        per_model_stats[model_name] = {"mean_error": me, "mean_overlap": mo, "exact_rate": ex, "n": len(errs)}
        print(f"  {model_name:50s}  err={me:.4f}  overlap={mo:.4f}  exact={ex:.4f}  n={len(errs)}", flush=True)

    all_mean_errors = [s["mean_error"] for s in per_model_stats.values() if not math.isnan(s["mean_error"])]
    ceiling_flag = all(e < 0.10 for e in all_mean_errors) if all_mean_errors else False

    # Step 8: Correlations
    print("\n=== Step 8: Correlations ===", flush=True)
    paired = [item for item in valid_sampled if item["item_hash"] in item_llm_difficulty]
    hd_vals = [item["human_difficulty"] for item in paired]
    llm_vals = [item_llm_difficulty[item["item_hash"]] for item in paired]
    n_paired = len(hd_vals)

    sp_r = spearman(hd_vals, llm_vals)
    pe_r = pearson(hd_vals, llm_vals)

    print(f"  n_paired = {n_paired}")
    print(f"  Spearman r = {sp_r:.4f}")
    print(f"  Pearson  r = {pe_r:.4f}", flush=True)

    # Step 9: Gate verdict
    print("\n=== Step 9: Gate verdict ===", flush=True)
    if ceiling_flag:
        verdict = (
            "ELICITATION CEILING: all models have mean error < 10%. "
            "LLM difficulty variance is insufficient to predict human difficulty. "
            "This is fixable (use harder prompts, masked-token elicitation, or "
            "perplexity-based scoring). NOT evidence that transfer fails."
        )
    elif math.isnan(sp_r):
        verdict = "INDETERMINATE: Spearman could not be computed (insufficient variance in LLM difficulty)."
    elif sp_r > 0.4:
        verdict = (
            f"TRANSFER SIGNAL PRESENT: Spearman r = {sp_r:.3f} > 0.4 threshold. "
            "LLM item difficulty predicts human item difficulty. "
            "Recommend scaling to the full item set."
        )
    elif sp_r > 0.2:
        verdict = (
            f"WEAK SIGNAL: Spearman r = {sp_r:.3f} (0.2-0.4 range). "
            "Some transfer signal but below the strong gate threshold. "
            "Ceiling on easy items may be suppressing the correlation."
        )
    else:
        verdict = (
            f"NULL: Spearman r = {sp_r:.3f} < 0.2. "
            "LLM difficulty does not predict human difficulty at this threshold. "
            "Check for ceiling on easy items, distribution mismatch, or genuine null."
        )

    print(f"  VERDICT: {verdict}", flush=True)

    # Step 10: Write outputs
    print("\n=== Step 10: Writing outputs ===", flush=True)

    # transfer_results.csv
    results_path = out_dir / "transfer_results.csv"
    base_fields = ["item_hash", "format", "prompt", "reference",
                   "human_difficulty", "llm_difficulty_overlap", "llm_difficulty_exact"]
    model_fields = []
    for m in selected_models:
        safe = m.replace("/", "_").replace(":", "_")
        model_fields += [f"{safe}_response", f"{safe}_exact", f"{safe}_overlap", f"{safe}_error"]

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
                "llm_difficulty_overlap": item_llm_difficulty.get(h, ""),
                "llm_difficulty_exact": item_llm_exact_err.get(h, ""),
            }
            for m in selected_models:
                safe = m.replace("/", "_").replace(":", "_")
                mr = results[h].get(m, {})
                row[f"{safe}_response"] = mr.get("response", "")
                row[f"{safe}_exact"] = mr.get("exact", "")
                row[f"{safe}_overlap"] = mr.get("overlap", "")
                row[f"{safe}_error"] = mr.get("error", "")
            writer.writerow(row)
    print(f"  Saved {len(paired)} rows -> {results_path}", flush=True)

    # transfer_report.txt
    report_path = out_dir / "transfer_report.txt"

    # Build human difficulty histogram for report
    hd_hist = [0] * 10
    for d in hd_vals:
        b = min(int(d * 10), 9)
        hd_hist[b] += 1

    lines = [
        "SLAM Transfer Test Report",
        "=" * 60,
        "",
        f"Date:        {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed:        {seed}",
        f"Items:       {n_paired}  (of {n_items} sampled; {n_no_prompt} excluded: no Spanish prompt)",
        f"Models:      {len(selected_models)}",
        "",
        "--- Models used (size order) ---",
    ]
    for name in selected_models:
        sz = next((m.get("size", 0) for m in all_models if m["name"] == name), 0)
        lines.append(f"  {name:50s}  {sz/1e9:.2f} GB")

    lines += [
        "",
        "--- Per-model error rates (ceiling check) ---",
        f"  {'Model':50s}  mean_error  mean_overlap  exact_rate  n",
    ]
    for name in selected_models:
        s = per_model_stats[name]
        lines.append(
            f"  {name:50s}  {s['mean_error']:.4f}      {s['mean_overlap']:.4f}        "
            f"{s['exact_rate']:.4f}    {s['n']}"
        )
    lines.append(f"\n  Ceiling (all models < 10% error): {'YES -- ELICITATION CEILING' if ceiling_flag else 'NO'}")

    lines += [
        "",
        "--- LLM difficulty distribution (1 - token_overlap) ---",
        f"  mean={llm_mean:.4f}  std={llm_std:.4f}  min={llm_min:.4f}  max={llm_max:.4f}",
        "  Histogram:",
    ]
    for i, cnt in enumerate(llm_bin_counts):
        lo, hi = i * 0.1, (i + 1) * 0.1
        bar = "#" * (cnt * 30 // max(llm_bin_counts, default=1))
        lines.append(f"    [{lo:.1f}-{hi:.1f}): {cnt:3d}  {bar}")

    lines += [
        "",
        "--- Human difficulty distribution of sampled items ---",
        "  Histogram:",
    ]
    for i, cnt in enumerate(hd_hist):
        lo, hi = i * 0.1, (i + 1) * 0.1
        lines.append(f"    [{lo:.1f}-{hi:.1f}): {cnt:3d}")

    lines += [
        "",
        "--- Correlations (LLM difficulty vs human difficulty) ---",
        f"  n = {n_paired}",
        f"  Spearman r = {sp_r:.4f}",
        f"  Pearson  r = {pe_r:.4f}",
        "",
        "--- Gate Verdict ---",
        f"  {verdict}",
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
        description="LLM respondent-agnostic transfer test for SLAM items."
    )
    parser.add_argument("--raw-dir", default="rl/data/slam_raw")
    parser.add_argument("--human-csv", default="deep_irt/slam_pilot/outputs/human_difficulty.csv")
    parser.add_argument("--out-dir", default="deep_irt/slam_pilot/outputs")
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_pilot(
        raw_dir=Path(args.raw_dir),
        human_csv=Path(args.human_csv),
        out_dir=Path(args.out_dir),
        n_items=args.n_items,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
