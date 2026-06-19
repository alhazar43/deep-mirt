"""items.py -- Build the shared ARC item bank and demonstration pool for E1.

Two artifacts are written to deep_irt/traj_icl/outputs/:

    item_bank.json  400 items (200 ARC-Easy test + 200 ARC-Challenge test),
                    sampled with a fixed seed, only items with exactly 4
                    choices, re-indexed so options are always A/B/C/D by
                    position and gold_letter is the letter at the matching
                    position.

    demos.json      16 demonstration items drawn from the ARC train splits,
                    disjoint from the bank, with their gold_letter for the
                    few-shot prompts.

The module also exports ``build_prompt`` which assembles a chat prompt for a
given target item at shot count k. The system turn instructs the model to
answer with a single option letter. Each demo turn shows the question, lettered
options, and "Answer: <letter>". The target turn ends without an answer so the
model fills it in.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

from datasets import load_dataset

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")

SEED = 42
BANK_SIZE_EASY = 200
BANK_SIZE_CHALLENGE = 200
DEMO_POOL_SIZE = 16

LETTERS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "Answer the following multiple-choice question by responding with the single "
    "letter of the correct option (A, B, C, or D). "
    "Do not explain your reasoning. Output only the letter."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_item(raw: Dict[str, Any], source: str) -> Dict[str, Any] | None:
    """Convert a raw ARC record to a normalized bank item.

    Returns None if the item does not have exactly 4 choices.
    """
    choices = raw["choices"]
    texts = choices["text"]
    labels = choices["label"]
    if len(texts) != 4:
        return None

    # Re-present as A/B/C/D by position.
    options = list(texts)

    # Map the gold answerKey to the positional letter.
    gold_raw = raw["answerKey"]
    # answerKey may be a letter (A/B/C/D) or a digit (1/2/3/4).
    if gold_raw in LETTERS:
        # Find the original position of that label in the label list.
        try:
            pos = labels.index(gold_raw)
        except ValueError:
            # Fallback: treat as 0-indexed letter.
            pos = LETTERS.index(gold_raw)
    elif gold_raw in ("1", "2", "3", "4"):
        pos = int(gold_raw) - 1
    else:
        return None

    if pos >= len(options):
        return None

    gold_letter = LETTERS[pos]
    return {
        "id": raw["id"],
        "question": raw["question"],
        "options": options,
        "gold_letter": gold_letter,
        "source": source,
    }


def _load_split(subset: str, split: str) -> List[Dict[str, Any]]:
    """Load and normalize one ARC split, keeping only 4-choice items."""
    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    out = []
    for raw in ds:
        item = _normalize_item(raw, source=f"{subset}/{split}")
        if item is not None:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Bank and demo construction
# ---------------------------------------------------------------------------

def build_bank_and_demos(
    out_dir: str = OUT,
    seed: int = SEED,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build the item bank and demo pool, write JSON, and return both.

    The bank is 400 items (200 Easy-test + 200 Challenge-test) sampled with
    a fixed seed. The demo pool is 16 items from the train splits, disjoint
    from the bank by item id.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    # Load test splits for the bank.
    easy_test = _load_split("ARC-Easy", "test")
    challenge_test = _load_split("ARC-Challenge", "test")

    # Sample with fixed seed.
    rng.shuffle(easy_test)
    rng.shuffle(challenge_test)
    bank_easy = easy_test[: min(BANK_SIZE_EASY, len(easy_test))]
    bank_challenge = challenge_test[: min(BANK_SIZE_CHALLENGE, len(challenge_test))]
    bank = bank_easy + bank_challenge

    bank_ids = {it["id"] for it in bank}

    # Load train splits for demos, disjoint from the bank.
    easy_train = _load_split("ARC-Easy", "train")
    challenge_train = _load_split("ARC-Challenge", "train")
    demo_pool_all = [it for it in easy_train + challenge_train
                     if it["id"] not in bank_ids]
    rng.shuffle(demo_pool_all)
    demos = demo_pool_all[:DEMO_POOL_SIZE]

    # Write files.
    bank_path = os.path.join(out_dir, "item_bank.json")
    demos_path = os.path.join(out_dir, "demos.json")

    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    with open(demos_path, "w", encoding="utf-8") as f:
        json.dump(demos, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(bank)} bank items to {bank_path}", flush=True)
    print(f"Wrote {len(demos)} demo items to {demos_path}", flush=True)
    return bank, demos


def load_bank_and_demos(
    out_dir: str = OUT,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load pre-built bank and demos from disk."""
    bank_path = os.path.join(out_dir, "item_bank.json")
    demos_path = os.path.join(out_dir, "demos.json")
    with open(bank_path, encoding="utf-8") as f:
        bank = json.load(f)
    with open(demos_path, encoding="utf-8") as f:
        demos = json.load(f)
    return bank, demos


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    target_item: Dict[str, Any],
    k: int,
    demos: List[Dict[str, Any]],
    demo_labels: List[str] | None = None,
    tokenizer: Any = None,
) -> str | List[Dict[str, str]]:
    """Build a chat prompt for one target item at shot count k.

    Args:
        target_item: a bank item dict.
        k: number of few-shot demonstrations to prepend (0 for zero-shot).
        demos: the full demo pool; the first k items are used.
        demo_labels: if provided, override the gold_letter for each demo
            (used for the shuffled-label condition). Must have len >= k.
        tokenizer: a HuggingFace tokenizer with apply_chat_template. If
            provided the function returns the rendered string; otherwise it
            returns the list of message dicts.

    Returns:
        The rendered prompt string (if tokenizer given) or the message list.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in range(k):
        demo = demos[i]
        label = demo_labels[i] if demo_labels is not None else demo["gold_letter"]
        user_text = demo["question"]
        for letter, text in zip(LETTERS, demo["options"]):
            user_text += f"\n{letter}. {text}"
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": f"Answer: {label}"})

    target_text = target_item["question"]
    for letter, text in zip(LETTERS, target_item["options"]):
        target_text += f"\n{letter}. {text}"
    target_text += "\nAnswer:"
    messages.append({"role": "user", "content": target_text})

    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return messages


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bank, demos = build_bank_and_demos()
    print(f"Bank size: {len(bank)}, Demo pool size: {len(demos)}")
    print("Sample bank item:", json.dumps(bank[0], indent=2))
    print("Sample demo item:", json.dumps(demos[0], indent=2))
