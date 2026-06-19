"""synth_remap.py -- E1b: synthetic arbitrary-mapping in-context label experiment.

WHY: ARC is a poor ICL probe because the model already knows the task -- demos
only prime. This experiment uses an ARBITRARY bijection between 4 semantic
categories (ANIMAL/FRUIT/VEHICLE/TOOL) and abstract single-letter labels
(A/B/C/D) that is NEVER stated, only implied by (word, label) demo pairs.  At
k=0 the model has no signal and must be near chance (0.25). Under the "true"
condition accuracy rises as the model infers the mapping from k demos. Under
"shuffled" demos there is no consistent mapping, so accuracy stays near chance
-- this is the unlearnable control.

Design.
    - 4 categories, ~50 words each, split into demo pool and query pool.
    - Seed-0 bijection: ANIMAL->C, FRUIT->A, VEHICLE->D, TOOL->B (arbitrary).
    - 200 query items (50 per category), fixed order, saved to outputs_e1b/item_bank.json.
    - k in {0,1,2,4,8,16,32}; for each k, a fixed balanced demo set is sampled.
    - true  condition: demos use the consistent mapping.
    - shuffled condition: each demo gets a uniformly random letter (seeded), breaking
      the mapping. Unlearnable control.
    - At k=0 both conditions are identical (no demos).

Output contract: outputs_e1b/responses_<safetag>.json, identical schema to E1
so irt_fit.load_responses works unchanged.

Run from the repo root:
    python -m deep_irt.traj_icl.synth_remap [--smoke] [--items N]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch

HERE = os.path.dirname(__file__)
OUT_E1B = os.path.join(HERE, "outputs_e1b")

K_VALUES = [0, 1, 2, 4, 8, 16, 32]
CONDITIONS = ["true", "shuffled"]
SEED = 42
SHUFFLE_SEED = 7
OPTION_LETTERS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# Mapping seeds: three additional arbitrary bijections for robustness study.
# Seed 0 is the original (ANIMAL->C, FRUIT->A, VEHICLE->D, TOOL->B).
# Seeds 1-3 are generated deterministically from a permutation RNG.
# ---------------------------------------------------------------------------

def _make_bijection(seed: int) -> Dict[str, str]:
    """Return an arbitrary but reproducible category->letter bijection for seed."""
    if seed == 0:
        # Original bijection from E1b; keep exactly as is for backward compat.
        return {"ANIMAL": "C", "FRUIT": "A", "VEHICLE": "D", "TOOL": "B"}
    rng = random.Random(seed * 1000 + 17)
    letters = list(OPTION_LETTERS)
    cats = list(["ANIMAL", "FRUIT", "VEHICLE", "TOOL"])
    rng.shuffle(letters)
    return dict(zip(cats, letters))

# ---------------------------------------------------------------------------
# Word pools (demo + query, ~50 per category, disjoint)
# ---------------------------------------------------------------------------

# Each category has 100 words: first 50 are the DEMO pool, last 50 are the
# QUERY pool (held-out). Kept here so the harness is fully self-contained.

_ANIMALS_ALL = [
    # demo pool (indices 0-49)
    "cat", "dog", "horse", "elephant", "tiger", "lion", "bear", "wolf",
    "fox", "deer", "rabbit", "squirrel", "raccoon", "otter", "beaver",
    "zebra", "giraffe", "hippo", "rhino", "gorilla", "chimpanzee", "baboon",
    "parrot", "eagle", "owl", "penguin", "flamingo", "pelican", "sparrow",
    "crow", "salmon", "tuna", "shark", "whale", "dolphin", "seal", "walrus",
    "crab", "lobster", "octopus", "frog", "turtle", "lizard", "gecko",
    "crocodile", "cobra", "python", "moth", "beetle", "ant",
    # query pool (indices 50-99)
    "butterfly", "dragonfly", "grasshopper", "mosquito", "ladybug",
    "firefly", "wasp", "bumblebee", "caterpillar", "scorpion",
    "tarantula", "snail", "earthworm", "jellyfish", "starfish",
    "clam", "oyster", "mussel", "shrimp", "squid",
    "moose", "bison", "elk", "caribou", "coyote",
    "bobcat", "lynx", "jaguar", "cheetah", "leopard",
    "meerkat", "mongoose", "armadillo", "platypus", "kangaroo",
    "koala", "wombat", "cassowary", "kiwi bird", "albatross",
    "heron", "whooping crane", "stork", "peacock", "toucan",
    "macaw", "cockatoo", "finch", "hummingbird", "woodpecker",
]

_FRUITS_ALL = [
    # demo pool
    "apple", "banana", "orange", "grape", "strawberry", "watermelon",
    "pineapple", "mango", "peach", "plum", "cherry", "pear", "apricot",
    "blueberry", "raspberry", "blackberry", "lemon", "lime", "grapefruit",
    "kiwi", "papaya", "guava", "lychee", "pomegranate", "fig",
    "date", "coconut", "avocado", "melon", "cantaloupe", "honeydew",
    "tangerine", "clementine", "mandarin", "nectarine", "sapote", "persimmon",
    "dragonfruit", "starfruit", "jackfruit", "breadfruit", "durian",
    "rambutan", "mangosteen", "passion fruit", "tamarind", "quince",
    "mulberry", "gooseberry", "cranberry",
    # query pool
    "boysenberry", "elderberry", "currant", "loganberry", "marionberry",
    "serviceberry", "cloudberry", "bilberry", "lingonberry", "bearberry",
    "salmonberry", "thimbleberry", "dewberry", "wineberry", "buffaloberry",
    "feijoa", "cherimoya", "soursop", "atemoya", "sapodilla",
    "cupuacu", "jabuticaba", "pitaya", "longan", "caimito",
    "santol", "marang", "langsat", "pulasan", "salak",
    "carambola", "bael", "jujube", "kumquat", "uglifruit",
    "yuzu", "bergamot", "citron", "pomelo", "blood orange",
    "cara cara", "satsuma", "etrog", "finger lime", "Buddha's hand",
    "muscadine", "scuppernong", "concord grape", "champagne grape", "sultana",
]

_VEHICLES_ALL = [
    # demo pool
    "car", "truck", "bus", "bicycle", "motorcycle", "train", "airplane",
    "helicopter", "boat", "ship", "submarine", "tractor", "bulldozer",
    "crane", "forklift", "scooter", "skateboard", "kayak", "canoe",
    "sailboat", "ferry", "tanker", "cargo ship", "hot air balloon",
    "glider", "jet", "spaceship", "rocket", "shuttle", "blimp",
    "dirigible", "hang glider", "paraglider", "ultralight", "drone",
    "hovercraft", "snowmobile", "ATV", "golf cart", "rickshaw",
    "chariot", "stagecoach", "gondola", "punt", "rowboat",
    "speedboat", "motorboat", "catamaran", "trimaran", "yacht",
    # query pool
    "dinghy", "skiff", "barge", "tugboat", "icebreaker",
    "aircraft carrier", "destroyer", "frigate", "cruiser", "battleship",
    "ambulance", "fire truck", "police car", "taxi", "limousine",
    "minivan", "pickup truck", "semi truck", "dump truck", "cement mixer",
    "garbage truck", "delivery van", "mail truck", "armored car", "tank",
    "jeep", "SUV", "convertible", "coupe", "sedan",
    "hatchback", "station wagon", "minibus", "trolleybus", "tram",
    "monorail", "cable car", "funicular", "maglev", "hyperloop",
    "biplane", "seaplane", "turboprop", "jumbo jet", "supersonic jet",
    "fighter jet", "bomber", "stealth bomber", "VTOL", "tiltrotor",
]

_TOOLS_ALL = [
    # demo pool
    "hammer", "screwdriver", "wrench", "pliers", "saw", "drill",
    "chisel", "crowbar", "level", "tape measure", "ruler", "compass",
    "protractor", "calculator", "stapler", "hole punch", "scissors",
    "knife", "spatula", "tongs", "ladle", "whisk", "peeler",
    "grater", "colander", "mortar", "pestle", "rolling pin", "brush",
    "paintbrush", "roller", "shovel", "rake", "hoe", "pitchfork",
    "axe", "hatchet", "machete", "pickaxe", "sledgehammer", "mallet",
    "file", "rasp", "plane", "router", "jigsaw", "circular saw",
    "handsaw", "coping saw",
    # query pool
    "hacksaw", "band saw", "table saw", "miter saw", "reciprocating saw",
    "oscillating tool", "angle grinder", "bench grinder", "belt sander",
    "orbital sander", "palm sander", "detail sander", "heat gun",
    "soldering iron", "wire stripper", "crimper", "multimeter", "voltage tester",
    "clamp", "vise", "C-clamp", "bar clamp", "pipe clamp",
    "torque wrench", "socket wrench", "Allen wrench", "box wrench", "open-end wrench",
    "needle-nose pliers", "slip-joint pliers", "locking pliers", "channel-lock pliers", "wire cutters",
    "awl", "bradawl", "center punch", "nail set", "scribe",
    "marking gauge", "combination square", "speed square", "T-bevel", "dividers",
    "caulking gun", "grease gun", "rivet gun", "staple gun", "glue gun",
    "spokeshave", "drawknife", "burnisher",
]

CATEGORIES = ["ANIMAL", "FRUIT", "VEHICLE", "TOOL"]
_ALL_WORDS: Dict[str, List[str]] = {
    "ANIMAL": _ANIMALS_ALL,
    "FRUIT": _FRUITS_ALL,
    "VEHICLE": _VEHICLES_ALL,
    "TOOL": _TOOLS_ALL,
}

# Seed-0 arbitrary bijection (category -> letter). Arbitrary but fixed.
# ANIMAL->C, FRUIT->A, VEHICLE->D, TOOL->B.
BIJECTION: Dict[str, str] = {
    "ANIMAL": "C",
    "FRUIT": "A",
    "VEHICLE": "D",
    "TOOL": "B",
}

DEMO_POOL_SIZE = 50  # first 50 words per category
QUERY_POOL_SIZE = 50  # last 50 words per category


# ---------------------------------------------------------------------------
# Bank and demo construction
# ---------------------------------------------------------------------------

def build_bank_and_demos(
    out_dir: str = OUT_E1B,
    seed: int = SEED,
    bijection: Optional[Dict[str, str]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build query bank (200 items) and demo pool (200 items), write JSON, return both.

    Query items are drawn from the QUERY pool (indices 50-99 per category).
    Demo items are drawn from the DEMO pool (indices 0-49 per category).
    The two pools are disjoint by construction.

    Args:
        bijection: category->letter mapping. Defaults to BIJECTION (seed-0).
    """
    if bijection is None:
        bijection = BIJECTION
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    bank: List[Dict[str, Any]] = []
    demos: List[Dict[str, Any]] = []

    for cat in CATEGORIES:
        all_words = _ALL_WORDS[cat]
        demo_words = all_words[:DEMO_POOL_SIZE]
        query_words = all_words[DEMO_POOL_SIZE: DEMO_POOL_SIZE + QUERY_POOL_SIZE]

        for i, w in enumerate(query_words):
            bank.append({
                "id": f"{cat}_{i:03d}",
                "word": w,
                "category": cat,
                "gold_letter": bijection[cat],
            })

        for i, w in enumerate(demo_words):
            demos.append({
                "id": f"demo_{cat}_{i:03d}",
                "word": w,
                "category": cat,
                "gold_letter": bijection[cat],
            })

    # Shuffle query bank order for variety; fix seed so every run is the same.
    rng.shuffle(bank)

    bank_path = os.path.join(out_dir, "item_bank.json")
    demos_path = os.path.join(out_dir, "demos.json")
    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    with open(demos_path, "w", encoding="utf-8") as f:
        json.dump(demos, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(bank)} query items to {bank_path}", flush=True)
    print(f"Wrote {len(demos)} demo items to {demos_path}", flush=True)
    print(f"Bijection: {bijection}", flush=True)
    return bank, demos


def load_bank_and_demos(
    out_dir: str = OUT_E1B,
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
# Demo selection: balanced across categories, fixed seed per k
# ---------------------------------------------------------------------------

def _select_demos_for_k(
    demos: List[Dict[str, Any]],
    k: int,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Return k demos balanced across categories, fixed for a given k.

    With k < 4 categories, some categories may have 0 demos. Categories
    are filled in round-robin order to keep balance.
    """
    if k == 0:
        return []
    rng = random.Random(seed + k)
    by_cat: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}
    for d in demos:
        by_cat[d["category"]].append(d)
    # Shuffle each pool independently.
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])

    selected: List[Dict[str, Any]] = []
    per_cat, extra = divmod(k, len(CATEGORIES))
    # Fill base quota per category.
    for c in CATEGORIES:
        selected.extend(by_cat[c][:per_cat])
    # Add one extra from the first `extra` categories.
    for i, c in enumerate(CATEGORIES[:extra]):
        idx = per_cat  # next item after the base quota
        if idx < len(by_cat[c]):
            selected.append(by_cat[c][idx])
    rng.shuffle(selected)
    return selected[:k]


# ---------------------------------------------------------------------------
# Shuffled-label generation (control condition)
# ---------------------------------------------------------------------------

def _make_shuffled_labels(
    demos: List[Dict[str, Any]],
    seed: int = SHUFFLE_SEED,
) -> List[str]:
    """Assign each demo a random letter, seeded, no consistent mapping."""
    rng = random.Random(seed)
    return [rng.choice(OPTION_LETTERS) for _ in demos]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Each word belongs to one of four categories. "
    "The categories are labeled A, B, C, or D in an arbitrary order "
    "that you must infer from the examples. "
    "Given the examples, output the label for the final word as a single letter."
)


def build_prompt(
    target_item: Dict[str, Any],
    k: int,
    demos: List[Dict[str, Any]],
    demo_labels: Optional[List[str]] = None,
    tokenizer: Any = None,
) -> str:
    """Build a chat prompt for one target item at shot count k.

    Args:
        target_item: a bank item dict with 'word' and 'gold_letter'.
        k: number of few-shot demo pairs to include.
        demos: pre-selected demo items (len >= k).
        demo_labels: if provided, override gold_letter for each demo
            (shuffled condition). Must have len >= k.
        tokenizer: HuggingFace tokenizer with apply_chat_template.

    Returns:
        Rendered prompt string.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Each demo is a user/assistant exchange.
    for i in range(min(k, len(demos))):
        demo = demos[i]
        label = demo_labels[i] if demo_labels is not None else demo["gold_letter"]
        messages.append({"role": "user", "content": f"Word: {demo['word']}"})
        messages.append({"role": "assistant", "content": label})

    # Target: append and leave for model to complete.
    messages.append({"role": "user", "content": f"Word: {target_item['word']}"})

    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback: raw text (not used in practice).
    lines = []
    for m in messages:
        lines.append(f"[{m['role'].upper()}] {m['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reuse generate.py helpers for model loading and scoring
# ---------------------------------------------------------------------------

from deep_irt.traj_icl.generate import (
    load_model_and_tokenizer,
    free_model,
    _get_option_token_ids,
    score_item,
)


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(
    model_id: str,
    bank: List[Dict[str, Any]],
    demos: List[Dict[str, Any]],
    out_dir: str = OUT_E1B,
    device: str = "cuda",
    max_items: Optional[int] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Score all bank items (or the first max_items) for one model.

    Args:
        k_values: shot-count grid to evaluate. Defaults to K_VALUES.
    """
    if k_values is None:
        k_values = K_VALUES
    os.makedirs(out_dir, exist_ok=True)

    if max_items is not None:
        bank = bank[:max_items]

    item_ids = [it["id"] for it in bank]
    n = len(bank)

    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    option_token_ids = _get_option_token_ids(tokenizer)
    print(f"  Option token ids: {dict(zip(OPTION_LETTERS, option_token_ids))}", flush=True)

    correct: Dict[str, Dict[str, List[int]]] = {
        "true": {str(k): [] for k in k_values},
        "shuffled": {str(k): [] for k in k_values},
    }
    accuracy: Dict[str, Dict[str, float]] = {"true": {}, "shuffled": {}}

    t0 = time.time()

    for k in k_values:
        # Pre-select a fixed demo set for this k.
        demo_set = _select_demos_for_k(demos, k, seed=SEED)

        # Shuffled labels for this demo set.
        shuffled_labels = _make_shuffled_labels(demo_set, seed=SHUFFLE_SEED) if k > 0 else []

        for cond in CONDITIONS:
            if cond == "shuffled" and k > 0:
                dlabels = shuffled_labels
            else:
                dlabels = None  # use gold_letter from demo items

            preds = []
            for j, item in enumerate(bank):
                prompt_text = build_prompt(
                    target_item=item,
                    k=k,
                    demos=demo_set,
                    demo_labels=dlabels,
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

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak_gb:.2f} GB", flush=True)
    else:
        peak_gb = 0.0

    free_model(model)

    result = {
        "model": model_id,
        "item_ids": item_ids,
        "k_values": k_values,
        "conditions": CONDITIONS,
        "correct": correct,
        "accuracy": accuracy,
        "meta": {
            "n_items": n,
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
    parser = argparse.ArgumentParser(
        description="E1b: synthetic label-remapping ICL experiment"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: first 40 items, 0.5B only")
    parser.add_argument("--items", type=int, default=None,
                        help="Override item count (overrides --smoke)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model ladder")
    parser.add_argument(
        "--k-grid", nargs="+", type=int, default=None,
        help="Shot-count grid (space-separated ints). Default: 0 1 2 4 8 16 32",
    )
    parser.add_argument(
        "--mapping-seed", type=int, default=0,
        help="Bijection seed (0=original E1b mapping, 1-3=robustness seeds). "
             "Controls which category->letter assignment is used.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: outputs_e1b for seed 0, "
             "outputs_e1b_robust/seed<S> for seed>0).",
    )
    args = parser.parse_args()

    mapping_seed = args.mapping_seed
    bijection = _make_bijection(mapping_seed)

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif mapping_seed == 0:
        out_dir = OUT_E1B
    else:
        out_dir = os.path.join(HERE, "outputs_e1b_robust", f"seed{mapping_seed}")

    k_grid = args.k_grid if args.k_grid is not None else K_VALUES

    print(f"Mapping seed: {mapping_seed} | Bijection: {bijection}", flush=True)
    print(f"k-grid: {k_grid}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)

    bank_path = os.path.join(out_dir, "item_bank.json")
    if os.path.exists(bank_path):
        print("Loading existing bank and demos from disk.", flush=True)
        bank, demos = load_bank_and_demos(out_dir)
    else:
        print("Building bank and demos...", flush=True)
        bank, demos = build_bank_and_demos(out_dir, bijection=bijection)

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
                out_dir=out_dir,
                device="cuda",
                max_items=max_items,
                k_values=k_grid,
            )
            print(
                f"\nDone {model_id} | wall={result['meta']['wall_clock_s']:.0f}s "
                f"| peak VRAM={result['meta']['peak_vram_gb']:.2f}GB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"OOM: {model_id} at fp16, skipping.", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nTotal wall clock: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
