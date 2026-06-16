"""run_pregate.py -- Judge pre-gate screening for the RQ1 essay experiment.

A prior RQ1 pilot found the local judge llama3.2:3b near-chance at comparing
essay quality (agreement with human scores = 0.535). Before spending a
multi-hour full elicitation, we cheaply SCREEN candidate judges: on pairs whose
human answer is KNOWN (scores differ by a clear margin), how often does each
judge agree with the human ordering. Keep judges clearing ~0.65.

Two protocols per judge
-----------------------
  ab    : show both essays (capped), randomize A/B to control position bias,
          ask "which is better, A or B", parse, map back. 1 call per pair.
  score : score each UNIQUE essay once on 0-10, derive each pair's winner as
          the higher score (skip ties). Cheaper (unique-essay calls, not 2x
          pairs) and usually a better discriminator than head-to-head.

Reuse
-----
essay_data.load_asap_set (ASAP latin-1 loading) for the essays, and
elicit.call_ollama / build_pair_prompt / parse_choice for the Ollama call
pattern, the bounded prompt, and the robust A/B parse. num_ctx is capped at
3072 and temperature pinned to 0 on every call (the 128K default spills to CPU
and is slow).

Usage (from repo root)
----------------------
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="rl/src;ma-irt" \
    python deep_irt/rq1_essay/run_pregate.py \
      --judges qwen2.5-coder:7b,gemma4:e2b,phi4-mini:3.8b,llama3.2:3b \
      --protocols ab,score --n-pairs 100 --seed 0 --gap-min 2

Smoke (<=5 real calls, confirms the pipeline only):
  ... python deep_irt/rq1_essay/run_pregate.py \
      --judges llama3.2:3b --protocols ab --n-pairs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from essay_data import load_asap_set  # noqa: E402
from elicit import (  # noqa: E402
    build_pair_prompt,
    call_ollama,
    list_models,
    parse_choice,
)

OUTPUT_DIR = Path(HERE) / "outputs_pregate"
ESSAY_SET = 1                # set 1: persuasive letter, human score 2-12
PASS_THRESHOLD = 0.65        # agreement at/above this flags PASS
NUM_CTX = 3072               # cap context (128K default spills to CPU, slow)

DEFAULT_JUDGES = [
    "qwen2.5-coder:7b",
    "gemma4:e2b",
    "phi4-mini:3.8b",
    "llama3.2:3b",            # CALIBRATION: expect ~0.53 (pilot) to confirm
                             # this harness measures the same thing as the pilot
]
DEFAULT_PROTOCOLS = ["ab", "score"]

SET_PROMPT_TOPIC = "Effects of computers on people and society (persuasive letter)."


# ---------------------------------------------------------------------------
# Labeled pair sampling (network-free, deterministic, unit-tested)
# ---------------------------------------------------------------------------

def sample_labeled_pairs(rows, n_pairs, seed, gap_min=2):
    """Sample non-tied essay pairs whose human scores differ by >= gap_min.

    The human winner is unambiguous: the higher-scored essay. Pairs are
    stratified across the score range by bucketing on the LOWER score of the
    pair, so the screen spans weak-vs-strong and mid-vs-high contrasts rather
    than collapsing onto one band. Deterministic given (rows order, seed).

    Returns list of dicts: {a, b, score_a, score_b, winner} where winner is the
    essay_id of the higher-scored essay (a and b are essay_ids, a<b by id).
    """
    rng = np.random.default_rng(seed)
    ids = np.array([r["essay_id"] for r in rows])
    scores = np.array([r["score"] for r in rows], dtype=float)
    order = np.argsort(ids)              # stable canonical order by essay_id
    ids, scores = ids[order], scores[order]
    n = len(ids)
    if n < 2:
        return []

    lo, hi = int(scores.min()), int(scores.max())
    # Buckets keyed by the lower score of a valid pair: lo .. hi-gap_min.
    bucket_keys = list(range(lo, hi - gap_min + 1))
    if not bucket_keys:
        return []
    target_per = max(1, n_pairs // len(bucket_keys))

    seen = set()
    by_bucket = {k: [] for k in bucket_keys}
    # Cap total attempts so a degenerate score distribution can't spin forever.
    max_tries = max(n_pairs * 200, 20000)
    tries = 0
    while sum(len(v) for v in by_bucket.values()) < n_pairs and tries < max_tries:
        tries += 1
        a_pos, b_pos = rng.integers(0, n, size=2)
        if a_pos == b_pos:
            continue
        sa, sb = scores[a_pos], scores[b_pos]
        if abs(sa - sb) < gap_min:
            continue
        ia, ib = int(ids[a_pos]), int(ids[b_pos])
        lo_id, hi_id = (ia, ib) if ia < ib else (ib, ia)
        if (lo_id, hi_id) in seen:
            continue
        bkey = int(min(sa, sb))
        if bkey not in by_bucket:
            continue
        # Let a hot bucket overflow modestly but not unboundedly, unless we
        # still need pairs overall and no other bucket is filling.
        if len(by_bucket[bkey]) >= target_per * 3 and \
           sum(len(v) for v in by_bucket.values()) >= n_pairs:
            continue
        seen.add((lo_id, hi_id))
        # winner = essay_id of the higher-scored essay (gap >= gap_min, no tie)
        winner = ia if sa > sb else ib
        score_lo = float(sa if ia < ib else sb)
        score_hi = float(sb if ia < ib else sa)
        by_bucket[bkey].append({
            "a": lo_id, "b": hi_id,
            "score_a": score_lo, "score_b": score_hi,
            "winner": int(winner),
        })

    pairs = []
    for k in bucket_keys:
        pairs.extend(by_bucket[k])
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def compute_agreement(decisions):
    """Agreement = fraction of decided pairs where judge winner == human winner.

    decisions: list of (human_winner_id, judge_winner_id_or_None). Pairs the
    judge could not decide (None) count toward `unparsed`, not the denominator.
    Returns (agreement, n_decided, n_unparsed).
    """
    decided = [(h, j) for (h, j) in decisions if j is not None]
    unparsed = len(decisions) - len(decided)
    if not decided:
        return float("nan"), 0, unparsed
    agree = sum(1 for h, j in decided if h == j)
    return agree / len(decided), len(decided), unparsed


# ---------------------------------------------------------------------------
# Resumable per-call cache
# ---------------------------------------------------------------------------

def _cache_path(judge):
    safe = judge.replace(":", "_").replace("/", "_")
    return OUTPUT_DIR / f"cache_{safe}.json"


def _load_cache(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_cache(path, cache):
    path.write_text(json.dumps(cache), encoding="utf-8")


def _ckey(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Protocol AB: head-to-head, randomized A/B, parse, map back
# ---------------------------------------------------------------------------

def run_ab(judge, pairs, text_by_id, cache, cache_path, seed):
    """Return list of (human_winner, judge_winner_or_None), one per pair."""
    rng = np.random.default_rng(seed + 11)
    decisions = []
    total = len(pairs)
    since_flush = 0
    for k, p in enumerate(pairs):
        a_id, b_id = p["a"], p["b"]
        key = _ckey(judge, "ab", a_id, b_id)
        if key in cache:
            decisions.append((p["winner"], cache[key]["judge_winner"]))
            continue
        # Randomize which essay is shown as A vs B; remember the mapping.
        flip = bool(rng.integers(0, 2))
        if flip:
            shown_A, shown_B = b_id, a_id   # A<-b, B<-a
        else:
            shown_A, shown_B = a_id, b_id   # A<-a, B<-b
        prompt = build_pair_prompt(
            SET_PROMPT_TOPIC, text_by_id[shown_A], text_by_id[shown_B]
        )
        resp = call_ollama(judge, prompt, num_ctx=NUM_CTX, num_predict=8)
        choice = parse_choice(resp)
        if choice is None:
            judge_winner = None
        else:
            judge_winner = shown_A if choice == "A" else shown_B
        cache[key] = {"judge_winner": judge_winner, "raw": resp[:40], "flip": flip}
        decisions.append((p["winner"], judge_winner))
        since_flush += 1
        if since_flush >= 10:
            _save_cache(cache_path, cache)
            since_flush = 0
        print(f"    [ab] {judge} {k + 1}/{total}", flush=True)
    _save_cache(cache_path, cache)
    return decisions


# ---------------------------------------------------------------------------
# Protocol SCORE: score each unique essay once, winner = higher score
# ---------------------------------------------------------------------------

def build_score_prompt(prompt_text, essay):
    from elicit import CAP_CHARS
    return (
        "You are an expert essay grader. Rate the QUALITY of the student essay "
        "below on a 0 to 10 scale (0 = very poor, 10 = excellent), judging "
        "development, organization, language, and support. Reply with ONLY a "
        "single number from 0 to 10. No explanation.\n\n"
        f"PROMPT TOPIC: {prompt_text}\n\n"
        f"=== ESSAY ===\n{essay[:CAP_CHARS]}\n\n"
        "Quality score (0-10), just the number:"
    )


_NUM_PREFIXES = ("score", "rating", "answer", ":", "-", "=", " ")


def parse_score(response):
    """Extract a 0-10 number from the response, else None."""
    import re
    r = response.strip()
    if not r or r.startswith("[ERROR"):
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", r)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    if val < 0 or val > 10:
        return None
    return val


def run_score(judge, pairs, text_by_id, cache, cache_path):
    """Score each unique essay once, then derive each pair's winner.

    Returns list of (human_winner, judge_winner_or_None), one per pair (ties on
    the judge's score yield judge_winner=None -> counted as unparsed/undecided).
    """
    unique_ids = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    total = len(unique_ids)
    since_flush = 0
    for k, eid in enumerate(unique_ids):
        key = _ckey(judge, "score", eid)
        if key in cache:
            continue
        prompt = build_score_prompt(SET_PROMPT_TOPIC, text_by_id[eid])
        resp = call_ollama(judge, prompt, num_ctx=NUM_CTX, num_predict=8)
        val = parse_score(resp)
        cache[key] = {"score": val, "raw": resp[:40]}
        since_flush += 1
        if since_flush >= 10:
            _save_cache(cache_path, cache)
            since_flush = 0
        print(f"    [score] {judge} essay {k + 1}/{total}", flush=True)
    _save_cache(cache_path, cache)

    decisions = []
    for p in pairs:
        sa = cache[_ckey(judge, "score", p["a"])]["score"]
        sb = cache[_ckey(judge, "score", p["b"])]["score"]
        if sa is None or sb is None or sa == sb:
            judge_winner = None   # unscored or tie -> undecided
        else:
            judge_winner = p["a"] if sa > sb else p["b"]
        decisions.append((p["winner"], judge_winner))
    return decisions


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES),
                    help="comma-separated judge model names")
    ap.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS),
                    help="comma-separated: ab,score")
    ap.add_argument("--n-pairs", type=int, default=100,
                    help="number of labeled non-tied pairs to screen on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-min", type=int, default=2,
                    help="min human-score gap for a pair to be labeled")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    for p in protocols:
        if p not in ("ab", "score"):
            print(f"BLOCKER: unknown protocol '{p}' (expected ab|score)")
            sys.exit(2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 70, flush=True)
    print("RQ1 JUDGE PRE-GATE SCREENING", flush=True)
    print("=" * 70, flush=True)

    # Load essays + sample labeled pairs (network-free) -------------------
    rows = load_asap_set(ESSAY_SET)
    if not rows:
        print(f"BLOCKER: no essays for essay_set={ESSAY_SET}")
        sys.exit(1)
    text_by_id = {r["essay_id"]: r["essay"] for r in rows}
    pairs = sample_labeled_pairs(rows, args.n_pairs, args.seed, args.gap_min)
    if not pairs:
        print("BLOCKER: could not sample any labeled pairs "
              f"(gap_min={args.gap_min} may be too large)")
        sys.exit(1)
    gaps = [abs(p["score_a"] - p["score_b"]) for p in pairs]
    n_unique = len({p["a"] for p in pairs} | {p["b"] for p in pairs})
    print(f"\n[setup] set {ESSAY_SET}: {len(rows)} essays, "
          f"sampled {len(pairs)} labeled pairs "
          f"(gap_min={args.gap_min}, mean_gap={np.mean(gaps):.1f}, "
          f"{n_unique} unique essays)", flush=True)

    # Verify Ollama + judges reachable ------------------------------------
    try:
        installed = {m["name"] for m in list_models()}
    except Exception as e:  # noqa: BLE001
        print(f"BLOCKER: Ollama unreachable: {e}")
        sys.exit(1)
    missing = [j for j in judges if j not in installed]
    if missing:
        print(f"BLOCKER: judges not installed in Ollama: {missing}\n"
              f"  installed: {sorted(installed)}")
        sys.exit(1)

    # Screen each judge x protocol ----------------------------------------
    results = {}   # judge -> protocol -> {agreement, n, unparsed, pass}
    for judge in judges:
        cache_path = _cache_path(judge)
        cache = _load_cache(cache_path)
        results[judge] = {}
        for proto in protocols:
            print(f"\n[run] judge={judge} protocol={proto}", flush=True)
            if proto == "ab":
                decisions = run_ab(judge, pairs, text_by_id, cache,
                                   cache_path, args.seed)
            else:
                decisions = run_score(judge, pairs, text_by_id, cache,
                                      cache_path)
            agreement, n, unparsed = compute_agreement(decisions)
            passed = (agreement == agreement) and agreement >= PASS_THRESHOLD
            results[judge][proto] = {
                "agreement": agreement, "n": n,
                "unparsed": unparsed, "pass": bool(passed),
            }
            agr_str = "nan" if agreement != agreement else f"{agreement:.3f}"
            print(f"    -> agreement={agr_str} n={n} unparsed={unparsed} "
                  f"{'PASS' if passed else 'FAIL'}", flush=True)

    # Report --------------------------------------------------------------
    report = format_report(results, judges, protocols, pairs, args)
    print("\n" + report, flush=True)
    (out_dir / "pregate_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "pregate_report.json").write_text(
        json.dumps({
            "params": {
                "judges": judges, "protocols": protocols,
                "n_pairs_requested": args.n_pairs, "n_pairs_sampled": len(pairs),
                "seed": args.seed, "gap_min": args.gap_min,
                "essay_set": ESSAY_SET, "pass_threshold": PASS_THRESHOLD,
                "num_ctx": NUM_CTX,
            },
            "results": results,
        }, indent=2, default=lambda x: None if x != x else x),
        encoding="utf-8")
    print(f"\nOutputs -> {out_dir}/  (total {time.time() - t0:.0f}s)", flush=True)


def format_report(results, judges, protocols, pairs, args):
    lines = []
    lines.append("=" * 70)
    lines.append("RQ1 JUDGE PRE-GATE REPORT")
    lines.append("=" * 70)
    lines.append(f"essay_set={ESSAY_SET}  n_pairs={len(pairs)}  "
                 f"gap_min={args.gap_min}  seed={args.seed}  "
                 f"pass_threshold={PASS_THRESHOLD}")
    lines.append("")
    header = f"{'judge':<22} {'protocol':<8} {'agree':>7} {'n':>5} {'unpars':>7} {'verdict':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for judge in judges:
        for proto in protocols:
            r = results[judge][proto]
            agr = "nan" if r["agreement"] != r["agreement"] else f"{r['agreement']:.3f}"
            verdict = "PASS" if r["pass"] else "FAIL"
            lines.append(f"{judge:<22} {proto:<8} {agr:>7} {r['n']:>5} "
                         f"{r['unparsed']:>7} {verdict:>8}")
    lines.append("-" * len(header))
    passing = sorted({j for j in judges
                      for p in protocols if results[j][p]["pass"]})
    lines.append(f"PASS (>= {PASS_THRESHOLD}): "
                 f"{', '.join(passing) if passing else '(none)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
