"""elicit.py -- Safe, bounded LLM elicitation for RQ5c.

Wraps the proven Ollama HTTP-API call from deep_irt/slam_pilot with the
RQ5c safety caps (NON-NEGOTIABLE -- local Ollama has thrown runaways):

    - HTTP API only (urllib POST), never interactive ``ollama run``.
    - options.num_ctx = 2048 on every call.
    - Hard per-call timeout = 45s; a hung call aborts and is recorded as an
      error, never blocks.
    - Per (model, bank) responses persisted to disk incrementally, so a restart
      resumes instead of re-eliciting.

Grading reuses token-overlap from the slam_pilot pilot. Each item is also
reduced to a BINARY correctness for the Rasch anchoring: correct iff the model
reproduces ALL reference tokens (token_overlap == 1.0), matching SLAM's
"all-correct" category. A softer threshold is available for sensitivity checks.
"""

from __future__ import annotations

import json
import string
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

OLLAMA_BASE = "http://localhost:11434"

# Safety caps (do not relax).
NUM_CTX = 2048
PER_CALL_TIMEOUT_S = 45

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def list_models() -> List[Dict[str, Any]]:
    url = f"{OLLAMA_BASE}/api/tags"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read()).get("models", [])


def call_ollama(model: str, prompt: str, timeout: int = PER_CALL_TIMEOUT_S) -> str:
    """Call Ollama generate API (non-streaming), capped context, hard timeout.

    Returns the response text, or ``"[ERROR: ...]"`` on any failure or timeout
    (the timeout is enforced by urlopen; a hung call raises and is caught here,
    so it can never block the run).
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "num_predict": 80,
            "num_ctx": NUM_CTX,
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
            return json.loads(resp.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[ERROR: {e}]"
    except Exception as e:  # noqa: BLE001 -- any failure must not block the run
        return f"[ERROR: {e}]"


def call_ollama_temp(model: str, prompt: str, temperature: float,
                     seed: int, timeout: int = PER_CALL_TIMEOUT_S) -> str:
    """call_ollama variant with an explicit sampling temperature + seed.

    Used to add cheap temperature-variant pseudo-respondents to the pool to
    widen N. A fixed seed keeps each variant reproducible.
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_predict": 80,
            "num_ctx": NUM_CTX,
            "top_p": 0.95,
            "seed": seed,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[ERROR: {e}]"
    except Exception as e:  # noqa: BLE001
        return f"[ERROR: {e}]"


def build_prompt(spanish: str) -> str:
    return (
        "Translate this Spanish text to English. "
        "Reply with only the English translation, nothing else.\n\n"
        f"Spanish: {spanish}\n"
        "English:"
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


def grade_token_overlap(response: str, reference: List[str]) -> float:
    """Fraction of reference tokens present in the response (0..1)."""
    if not reference:
        return 0.0
    ref = [_normalize(w) for w in reference if _normalize(w)]
    if not ref:
        return 0.0
    resp = set(_normalize(response).split())
    return sum(1 for t in ref if t in resp) / len(ref)


def strip_echo(response: str) -> str:
    out = response
    for pre in ["English:", "Translation:", "Answer:", "english:", "translation:"]:
        if out.lower().startswith(pre.lower()):
            return out[len(pre):].strip()
    return out


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def elicit_model_on_bank(
    model_name: str,
    bank_tag: str,
    items: List[Dict[str, Any]],
    cache_dir: Path,
    log_path: Path,
    temperature: Optional[float] = None,
    seed: int = 0,
    flush_every: int = 25,
) -> Dict[str, Dict[str, Any]]:
    """Elicit ``model_name`` on every item of one bank, with resume + logging.

    Returns ``{item_hash: {response, overlap, error}}``. Persists incrementally
    to ``cache_dir/<model>__<bank>.json`` and appends a progress line to
    ``log_path`` when the (model, bank) pair finishes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{_safe(model_name)}__{bank_tag}"
    if temperature is not None:
        tag = f"{tag}__t{temperature:g}_s{seed}"
    cache_path = cache_dir / f"{tag}.json"

    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, ValueError):
            cache = {}

    done = set(cache.keys())
    n = len(items)
    if len(done) >= n and n > 0:
        return cache

    t0 = time.time()
    since = 0
    n_err = 0
    for it in items:
        h = it["item_hash"]
        if h in done:
            continue
        prompt = build_prompt(it["prompt"])
        if temperature is None:
            raw = call_ollama(model_name, prompt)
        else:
            raw = call_ollama_temp(model_name, prompt, temperature, seed)
        resp = strip_echo(raw)
        if raw.startswith("[ERROR"):
            n_err += 1
        overlap = grade_token_overlap(resp, it["reference"])
        cache[h] = {"response": resp, "overlap": overlap, "error": 1.0 - overlap}
        since += 1
        if since >= flush_every:
            _atomic_write(cache_path, cache)
            since = 0
    _atomic_write(cache_path, cache)

    elapsed = time.time() - t0
    overlaps = [r["overlap"] for r in cache.values()]
    mean_ov = sum(overlaps) / len(overlaps) if overlaps else float("nan")
    line = (
        f"{time.strftime('%H:%M:%S')}  {tag:55s}  n={len(cache)}/{n}  "
        f"mean_overlap={mean_ov:.4f}  errors={n_err}  {elapsed:.0f}s\n"
    )
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(line)
    print("  " + line.rstrip(), flush=True)
    return cache


__all__ = [
    "OLLAMA_BASE",
    "NUM_CTX",
    "PER_CALL_TIMEOUT_S",
    "list_models",
    "call_ollama",
    "call_ollama_temp",
    "build_prompt",
    "grade_token_overlap",
    "strip_echo",
    "elicit_model_on_bank",
]
