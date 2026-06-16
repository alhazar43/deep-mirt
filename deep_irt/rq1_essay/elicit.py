"""elicit.py -- Local-LLM pairwise essay judging via Ollama.

Reuses the slam_pilot call_ollama pattern: POST to {OLLAMA_BASE}/api/generate,
non-streaming, temperature 0. Here the task is: given two essays, pick the
better one. Path A: comparisons are MODEL-generated (state plainly).

Design choices to keep it bounded and unbiased:
  - Cap essay text fed to the judge (essays are long; CAP_CHARS keeps latency
    and context bounded).
  - RANDOMIZE A/B order per comparison to control position bias; record the
    mapping so the parsed winner maps back to the true essay.
  - Robust parse: look for a leading 'A'/'B' token; fall back to scanning.
  - Per-comparison caching so a kill mid-run can resume.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"
CAP_CHARS = 1500   # truncate each essay in the prompt (bounded context)


def list_models() -> list:
    with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as resp:
        return json.loads(resp.read()).get("models", [])


def call_ollama(
    model: str,
    prompt: str,
    timeout: int = 90,
    num_ctx: int = 3072,
    num_predict: int = 8,
) -> str:
    """Non-streaming generate; returns response text or '[ERROR: ...]'.

    num_ctx is capped (default 3072) because the Ollama default 128K context
    spills to CPU and is very slow; temperature is pinned to 0 for determinism.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
            "top_p": 1.0,
            "num_ctx": num_ctx,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except (urllib.error.URLError, Exception) as e:  # noqa: BLE001
        return f"[ERROR: {e}]"


def build_pair_prompt(prompt_text: str, essay_a: str, essay_b: str) -> str:
    """Ask the judge to pick the better-quality essay. Demand a single letter."""
    a = essay_a[:CAP_CHARS]
    b = essay_b[:CAP_CHARS]
    return (
        "You are an expert essay grader. Two student essays answer the same "
        "prompt. Decide which essay is HIGHER QUALITY overall (development, "
        "organization, language, support). Reply with ONLY a single letter: "
        "A or B. No explanation.\n\n"
        f"PROMPT TOPIC: {prompt_text}\n\n"
        f"=== ESSAY A ===\n{a}\n\n"
        f"=== ESSAY B ===\n{b}\n\n"
        "Which essay is higher quality? Answer with only A or B:"
    )


_LETTER_RE = re.compile(r"\b([AB])\b", re.IGNORECASE)


def parse_choice(response: str) -> Optional[str]:
    """Return 'A', 'B', or None. Prefers a leading letter, else first match."""
    r = response.strip()
    if not r or r.startswith("[ERROR"):
        return None
    head = r[:3].upper()
    if head.startswith("A"):
        return "A"
    if head.startswith("B"):
        return "B"
    m = _LETTER_RE.search(r)
    if m:
        return m.group(1).upper()
    return None


def _pair_key(model: str, i: int, j: int) -> str:
    return hashlib.md5(f"{model}|{i}|{j}".encode()).hexdigest()[:16]


def elicit_comparisons(
    pairs: list,           # list of (i, j) essay indices (global within split)
    texts: list,           # essay texts indexed by global essay index
    prompt_text: str,
    model: str,
    cache_path: Path,
    rng,                   # numpy Generator for order randomization
    verbose: bool = True,
) -> list:
    """Run the judge over pairs. Returns list of dicts with parsed outcome.

    outcome == 1 means essay i judged better than essay j (true-index space,
    AFTER undoing the randomized A/B presentation order).
    """
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            cache = {}

    results = []
    t0 = time.time()
    n_err = 0
    n_unparsed = 0
    since_flush = 0

    for n, (i, j) in enumerate(pairs):
        key = _pair_key(model, i, j)
        if key in cache:
            results.append(cache[key])
            continue

        # Randomize which essay is shown as A vs B
        flip = bool(rng.integers(0, 2))
        if flip:
            ea, eb, a_is = texts[j], texts[i], j   # A=j, B=i
        else:
            ea, eb, a_is = texts[i], texts[j], i   # A=i, B=j

        prompt = build_pair_prompt(prompt_text, ea, eb)
        resp = call_ollama(model, prompt)
        choice = parse_choice(resp)

        if choice is None:
            n_unparsed += 1
            if resp.startswith("[ERROR"):
                n_err += 1
            entry = {"i": int(i), "j": int(j), "outcome": None,
                     "raw": resp[:40], "flip": flip}
        else:
            winner_idx = (j if (choice == "A") else i) if flip else \
                         (i if (choice == "A") else j)
            outcome = 1 if winner_idx == i else 0  # 1: i beats j
            entry = {"i": int(i), "j": int(j), "outcome": int(outcome),
                     "raw": resp[:40], "flip": flip}

        cache[key] = entry
        results.append(entry)
        since_flush += 1
        if since_flush >= 25:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            since_flush = 0
            if verbose:
                el = time.time() - t0
                rate = (n + 1) / el if el > 0 else 0
                rem = (len(pairs) - n - 1) / rate if rate > 0 else 0
                print(f"    [{n+1}/{len(pairs)}] elapsed={el:.0f}s "
                      f"est_remaining={rem:.0f}s unparsed={n_unparsed} err={n_err}",
                      flush=True)

    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    if verbose:
        el = time.time() - t0
        print(f"    DONE {len(pairs)} comparisons in {el:.0f}s "
              f"({el/60:.1f} min); unparsed={n_unparsed} err={n_err}", flush=True)
    return results
