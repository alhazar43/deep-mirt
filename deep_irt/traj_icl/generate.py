"""generate.py -- Score ARC bank items with a Qwen2.5-Instruct model ladder.

For each model, each shot count k in {0,1,2,4,8,16}, and each condition in
{"true","shuffled"}, every bank item is scored by taking the model's
next-token log-probabilities at the answer position over the four option-letter
tokens and calling argmax the predicted letter. Accuracy is the fraction of
bank items where the predicted letter matches gold_letter.

Conditions.
    "true"      The k demo items use their actual gold labels.
    "shuffled"  The k demo items use a fixed random permutation of the gold
                labels (seeded), breaking the input-label mapping -- the Min
                et al. 2022 priming control. At k=0 there are no demos, so
                true and shuffled are identical.

Output contract (one file per model, loader-compatible with irt_fit.load_responses):

    outputs/responses_<safetag>.json
    {
      "model": "<full HF id>",
      "item_ids": [400 id strings, same order for every model],
      "k_values": [0,1,2,4,8,16],
      "conditions": ["true","shuffled"],
      "correct": {
          "true":     {"0": [400 ints 0/1], "1": [...], ...},
          "shuffled": {same keys}
      },
      "accuracy": {
          "true":     {"0": float, ...},
          "shuffled": {...}
      }
    }

Run from the repo root:
    python -m deep_irt.traj_icl.generate [--smoke] [--items N]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from typing import Any, Dict, List

import torch

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")

K_VALUES = [0, 1, 2, 4, 8, 16]
CONDITIONS = ["true", "shuffled"]
SHUFFLE_SEED = 7

# Option-letter tokens as the model tokenizer sees them.
OPTION_LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Shuffled-label generation
# ---------------------------------------------------------------------------

def _make_shuffled_labels(
    demos: List[Dict[str, Any]],
    seed: int = SHUFFLE_SEED,
) -> List[str]:
    """Return a fixed permutation of the demo gold labels.

    The permutation is seeded so every call with the same seed returns the
    same mapping. This ensures the shuffled condition is reproducible.
    """
    rng = random.Random(seed)
    labels = [d["gold_letter"] for d in demos]
    shuffled = list(labels)
    while shuffled == labels and len(labels) > 1:
        rng.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, device: str = "cuda"):
    """Load a model and tokenizer in fp16 on device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def free_model(model) -> None:
    """Release a model from GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Scoring a single item
# ---------------------------------------------------------------------------

def _get_option_token_ids(tokenizer) -> List[int]:
    """Return the single-token ids for 'A','B','C','D'.

    We probe several encodings because different tokenizers assign different
    ids: bare letter, space-prefixed letter, and newline-prefixed letter. We
    prefer the bare encoding if it is a single token; otherwise fall back.
    """
    ids = []
    for letter in OPTION_LETTERS:
        candidates = [letter, f" {letter}", f"\n{letter}"]
        chosen = None
        for cand in candidates:
            toks = tokenizer.encode(cand, add_special_tokens=False)
            if len(toks) == 1:
                chosen = toks[0]
                break
        if chosen is None:
            # Last resort: take the first token of the bare letter encoding.
            chosen = tokenizer.encode(letter, add_special_tokens=False)[0]
        ids.append(chosen)
    return ids


@torch.inference_mode()
def score_item(
    model,
    tokenizer,
    prompt_text: str,
    option_token_ids: List[int],
    device: str = "cuda",
) -> str:
    """Return the predicted option letter for one item.

    The prompt_text already ends with "Answer:" (or the model's equivalent
    from apply_chat_template with add_generation_prompt=True). We forward
    the prompt and read logprobs at the last token position over the four
    option-letter tokens; argmax is the predicted letter.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # Last token logits: shape (vocab_size,)
    logits = outputs.logits[0, -1, :]
    option_logits = torch.tensor(
        [logits[tid].item() for tid in option_token_ids],
        dtype=torch.float32,
    )
    pred_idx = int(option_logits.argmax().item())
    return OPTION_LETTERS[pred_idx]


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(
    model_id: str,
    bank: List[Dict[str, Any]],
    demos: List[Dict[str, Any]],
    out_dir: str = OUT,
    device: str = "cuda",
    max_items: int | None = None,
) -> Dict[str, Any]:
    """Score all bank items (or the first max_items) for one model.

    Returns the response dict and writes it to disk.
    """
    from deep_irt.traj_icl.items import build_prompt

    os.makedirs(out_dir, exist_ok=True)

    # Truncate for smoke test.
    if max_items is not None:
        bank = bank[:max_items]

    item_ids = [it["id"] for it in bank]
    shuffled_labels_full = _make_shuffled_labels(demos, seed=SHUFFLE_SEED)

    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    option_token_ids = _get_option_token_ids(tokenizer)
    print(f"  Option token ids: {dict(zip(OPTION_LETTERS, option_token_ids))}", flush=True)

    correct: Dict[str, Dict[str, List[int]]] = {
        "true": {str(k): [] for k in K_VALUES},
        "shuffled": {str(k): [] for k in K_VALUES},
    }
    accuracy: Dict[str, Dict[str, float]] = {
        "true": {},
        "shuffled": {},
    }

    t0 = time.time()
    n = len(bank)

    for k in K_VALUES:
        for cond in CONDITIONS:
            # At k=0 both conditions are the same (no demos to shuffle).
            demo_labels = None
            if cond == "shuffled" and k > 0:
                demo_labels = shuffled_labels_full[:k]
            elif cond == "true" and k > 0:
                demo_labels = None  # use gold from demos

            preds = []
            for j, item in enumerate(bank):
                prompt_text = build_prompt(
                    target_item=item,
                    k=k,
                    demos=demos,
                    demo_labels=demo_labels,
                    tokenizer=tokenizer,
                )
                pred = score_item(model, tokenizer, prompt_text,
                                  option_token_ids, device=device)
                preds.append(1 if pred == item["gold_letter"] else 0)
                if (j + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    acc_so_far = sum(preds) / len(preds)
                    print(
                        f"  k={k} {cond:9s} {j+1:4d}/{n} "
                        f"acc={acc_so_far:.3f} elapsed={elapsed:.0f}s",
                        flush=True,
                    )

            correct[cond][str(k)] = preds
            accuracy[cond][str(k)] = float(sum(preds) / len(preds)) if preds else 0.0
            print(
                f"  k={k} {cond:9s} done | acc={accuracy[cond][str(k)]:.3f}",
                flush=True,
            )

    # VRAM peak.
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak_gb:.2f} GB", flush=True)
    else:
        peak_gb = 0.0

    free_model(model)

    result = {
        "model": model_id,
        "item_ids": item_ids,
        "k_values": K_VALUES,
        "conditions": CONDITIONS,
        "correct": correct,
        "accuracy": accuracy,
        "meta": {
            "n_items": len(bank),
            "peak_vram_gb": peak_gb,
            "wall_clock_s": time.time() - t0,
        },
    }

    safetag = model_id.split("/")[-1]
    out_path = os.path.join(out_dir, f"responses_{safetag}.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_path)
    print(f"  Wrote {out_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ARC responses for ICL experiment")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run only the first 40 bank items on the 0.5B model")
    parser.add_argument("--items", type=int, default=None,
                        help="Limit to this many bank items (overrides --smoke)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override the model ladder")
    args = parser.parse_args()

    from deep_irt.traj_icl.items import build_bank_and_demos, load_bank_and_demos

    bank_path = os.path.join(OUT, "item_bank.json")
    if os.path.exists(bank_path):
        print("Loading existing bank and demos from disk.", flush=True)
        bank, demos = load_bank_and_demos(OUT)
    else:
        print("Building bank and demos...", flush=True)
        bank, demos = build_bank_and_demos(OUT)

    print(f"Bank: {len(bank)} items | Demos: {len(demos)}", flush=True)

    if args.smoke:
        max_items = args.items if args.items is not None else 40
        model_ladder = ["Qwen/Qwen2.5-0.5B-Instruct"]
        print(f"SMOKE TEST: {max_items} items, model {model_ladder[0]}", flush=True)
    else:
        max_items = args.items
        model_ladder = args.models or [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
        ]

    t_total = time.time()
    for model_id in model_ladder:
        print(f"\n{'='*60}\n{model_id}\n{'='*60}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            result = run_model(
                model_id=model_id,
                bank=bank,
                demos=demos,
                out_dir=OUT,
                device="cuda",
                max_items=max_items,
            )
            print(f"\nDone {model_id} | wall={result['meta']['wall_clock_s']:.0f}s "
                  f"| peak VRAM={result['meta']['peak_vram_gb']:.2f}GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM: {model_id} at fp16, skipping.", flush=True)
            free_model(None)

    print(f"\nTotal wall clock: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
