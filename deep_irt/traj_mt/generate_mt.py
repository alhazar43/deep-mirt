"""generate_mt.py -- E4: in-context adaptation on real low-resource MT (FLORES-200).

Language pair: English -> Dinka (dik_Latn), from hlillemark/flores200_8_val_test.
Dinka is a Nilo-Saharan language of South Sudan, extremely low-resource and
essentially absent from Qwen2.5 training data. At k=0 the model is expected to
produce near-zero chrF; genuine ICL gain requires learning the target-language
surface form from the demonstrations themselves.

Data source: hlillemark/flores200_8_val_test (not gated, parquet-based, genuine
FLORES-200 sentences extracted from Wikipedia).

- Item bank: all en->dik_Latn samples from the val split (~93 items).
- Demo pool: first 32 en->dik_Latn samples from the test split (disjoint).

Output contract matches deep_irt.traj_icl.irt_fit.load_responses exactly.

Self-gate (1.5B, 30 items): gate passes if EITHER the absolute k=16 gain over
k=0 >= 2 chrF, OR the true-vs-shuffled gap at k=16 >= 2 chrF, AND the k=0
baseline >= 3 chrF (model can form words). This dual criterion handles both the
case where the model shows learning AND the case where shuffled demos actively
hurt (which is also a signal that content matters).

Threshold for binarization: median chrF from the 1.5B true k=16 full run
(93 items). Applied universally. Floor at 3.0.

Wall-time design: 1.5B runs the full ladder in one pass (gate uses a subset,
threshold uses the k=16 slice from the full run, not a separate pass). Gate and
full run together = one model load. Threshold is extracted from saved chrf_raw.
"""

import gc
import json
import os
import random
import statistics
import sys
import time

try:
    from sacrebleu.metrics import CHRF
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "sacrebleu"], check=True)
    from sacrebleu.metrics import CHRF

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

LOG_PATH = os.path.join(OUT, "run_log.txt")
_log_fh = open(LOG_PATH, "w", buffering=1, encoding="utf-8")


def log(msg: str = ""):
    """Write to both stdout and the buffered log file."""
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
K_VALUES = [0, 1, 2, 4, 8, 16]
CONDITIONS = ["true", "shuffled"]

SRC_LANG = "eng_Latn"
TGT_LANG = "dik_Latn"
LANG_NAME = "Dinka"

N_DEMOS = 32
MAX_NEW_TOKENS = 80

GATE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GATE_N_ITEMS = 30
GATE_K = 16
GATE_MIN_GAIN = 2.0    # true_k16 - true_k0 chrF
GATE_MIN_BASE = 3.0    # k=0 chrF floor
GATE_MIN_GAP = 2.0     # alternate: true_k16 - shuffled_k16 chrF

CHRF_METRIC = CHRF()


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def load_flores_data():
    """Load item bank (val) and demo pool (test) from hlillemark/flores200_8_val_test."""
    ds = load_dataset("hlillemark/flores200_8_val_test")

    val_items = sorted(
        [x for x in ds["val"] if x["source_lang"] == SRC_LANG and x["target_lang"] == TGT_LANG],
        key=lambda x: x["id"],
    )
    test_items = sorted(
        [x for x in ds["test"] if x["source_lang"] == SRC_LANG and x["target_lang"] == TGT_LANG],
        key=lambda x: x["id"],
    )
    demo_pool = test_items[:N_DEMOS]
    log(f"Item bank (val):  {len(val_items)} sentences")
    log(f"Demo pool (test): {len(demo_pool)} sentences (first {N_DEMOS} of {len(test_items)})")
    return val_items, demo_pool


# -------------------------------------------------------------------------
# Prompt construction
# -------------------------------------------------------------------------

def build_prompt(query_src: str, demos: list, shuffled: bool, rng: random.Random) -> str:
    """Build a few-shot MT prompt.

    At k=0 true and shuffled are identical (no demos).
    Shuffled: demo sources paired with mismatched targets from the same demo set.
    """
    header = f"Translate from English to {LANG_NAME}.\n\n"
    if not demos:
        return header + f"English: {query_src}\n{LANG_NAME}:"

    if shuffled:
        targets = [d["target"] for d in demos]
        shuffled_t = targets[:]
        while shuffled_t == targets and len(targets) > 1:
            rng.shuffle(shuffled_t)
        demo_list = [{"source": d["source"], "target": t} for d, t in zip(demos, shuffled_t)]
    else:
        demo_list = demos

    lines = []
    for i, d in enumerate(demo_list, 1):
        lines += [f"[Demo {i}]", f"English: {d['source']}", f"{LANG_NAME}: {d['target']}", ""]
    lines += [f"English: {query_src}", f"{LANG_NAME}:"]
    return header + "\n".join(lines)


# -------------------------------------------------------------------------
# Model management
# -------------------------------------------------------------------------

def load_model(model_name: str):
    log(f"\n--- Loading {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def free_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_translation(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.split("\n")[0].strip()


def score_chrf(hypothesis: str, reference: str) -> float:
    return CHRF_METRIC.sentence_score(hypothesis, [reference]).score


# -------------------------------------------------------------------------
# Self-gate (uses a subset of the 1.5B full run, no separate pass)
# -------------------------------------------------------------------------

def check_gate(chrf_raw_15b: dict) -> tuple:
    """Check gate from already-computed 1.5B chrf_raw.

    chrf_raw_15b[cond][k_str] = list of floats (all 93 items).
    Uses first GATE_N_ITEMS items and k=0, GATE_K.
    """
    n = GATE_N_ITEMS
    k0_true = chrf_raw_15b["true"]["0"][:n]
    kmax_true = chrf_raw_15b["true"][str(GATE_K)][:n]
    kmax_shuf = chrf_raw_15b["shuffled"][str(GATE_K)][:n]

    base = sum(k0_true) / len(k0_true)
    k_true = sum(kmax_true) / len(kmax_true)
    k_shuf = sum(kmax_shuf) / len(kmax_shuf)

    gain = k_true - base
    gap = k_true - k_shuf

    passes = (base >= GATE_MIN_BASE) and ((gain >= GATE_MIN_GAIN) or (gap >= GATE_MIN_GAP))

    log("")
    log("=" * 60)
    log(f"SELF-GATE (1.5B, {n} items)")
    log("=" * 60)
    log(f"k=0   true:   {base:5.1f}  shuffled: {base:5.1f}")
    log(f"k={GATE_K:2d}  true:   {k_true:5.1f}  shuffled: {k_shuf:5.1f}")
    log(f"Gain k=0->k={GATE_K}: {gain:+.1f} chrF  (need >= {GATE_MIN_GAIN})")
    log(f"True-shuffled gap at k={GATE_K}: {gap:+.1f} chrF  (alternate: need >= {GATE_MIN_GAP})")
    log(f"Base chrF at k=0: {base:.1f}  (need >= {GATE_MIN_BASE})")
    log(f"SELF-GATE: {'PASS' if passes else 'FAIL'}")

    return passes, {"k0_true": base, "k16_true": k_true, "k16_shuffled": k_shuf,
                    "gain": gain, "gap": gap}


# -------------------------------------------------------------------------
# Full run for one model
# -------------------------------------------------------------------------

def run_model(model_name: str, val_items: list, demo_pool: list,
              ref_threshold: float) -> dict:
    """Run the full k-ladder for one model. Returns (response_dict, chrf_raw_dict)."""
    rng = random.Random(42)
    model, tokenizer = load_model(model_name)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    n_items = len(val_items)
    item_ids = [f"flores_val_{i:03d}" for i in range(n_items)]
    chrf_raw = {cond: {str(k): [] for k in K_VALUES} for cond in CONDITIONS}

    for k in K_VALUES:
        for cond in CONDITIONS:
            scores = []
            for item in val_items:
                demos = rng.sample(demo_pool, k) if k > 0 else []
                prompt = build_prompt(
                    item["source"], demos,
                    shuffled=(cond == "shuffled" and k > 0),
                    rng=rng,
                )
                hyp = generate_translation(model, tokenizer, prompt)
                scores.append(score_chrf(hyp, item["target"]))
            chrf_raw[cond][str(k)] = scores
            mean_chrf = sum(scores) / len(scores)
            log(f"  [{model_name.split('/')[-1]}] k={k:2d} {cond:9s}: "
                f"mean chrF={mean_chrf:.2f}")

    wall_sec = time.time() - t0
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    free_model(model, tokenizer)

    # Binarize
    correct = {cond: {} for cond in CONDITIONS}
    accuracy = {cond: {} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for k in K_VALUES:
            sc = chrf_raw[cond][str(k)]
            binary = [1 if s >= ref_threshold else 0 for s in sc]
            correct[cond][str(k)] = binary
            accuracy[cond][str(k)] = float(sum(binary) / len(binary))

    safe_tag = model_name.replace("/", "__")
    response = {
        "model": model_name,
        "item_ids": item_ids,
        "k_values": K_VALUES,
        "conditions": CONDITIONS,
        "correct": correct,
        "accuracy": accuracy,
        "meta": {
            "lang_pair": f"{SRC_LANG} -> {TGT_LANG}",
            "lang_name": LANG_NAME,
            "n_items": n_items,
            "threshold": ref_threshold,
            "wall_sec": round(wall_sec, 1),
            "peak_vram_gb": round(peak_vram_gb, 2),
        },
    }
    chrf_out = {
        "model": model_name,
        "item_ids": item_ids,
        "k_values": K_VALUES,
        "conditions": CONDITIONS,
        "chrf": chrf_raw,
    }

    with open(os.path.join(OUT, f"responses_{safe_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)
    with open(os.path.join(OUT, f"chrf_raw_{safe_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(chrf_out, f, indent=2)

    log(f"  {model_name.split('/')[-1]}: wall={wall_sec:.0f}s  peak_vram={peak_vram_gb:.2f}GB  "
        f"-> responses written")
    return response, chrf_raw


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main():
    log(f"E4: English -> {LANG_NAME} ({TGT_LANG}) on FLORES-200")
    log(f"Models: {[m.split('/')[-1] for m in MODELS]}")
    log(f"k values: {K_VALUES}")
    log(f"Conditions: {CONDITIONS}")

    val_items, demo_pool = load_flores_data()

    # Run 1.5B first (gate + threshold embedded; no separate pass needed).
    # The gate uses the first GATE_N_ITEMS items from the 1.5B k=0 and k=GATE_K slices.
    # The threshold is the median chrF at (1.5B, k=16, true) over all 93 items.
    # A placeholder threshold of 5.0 is used for the 1.5B run itself (self-referential
    # binarization would require a second pass; the threshold is set from raw scores).
    log("\nStep 1: Run 1.5B full ladder (gate + threshold source)")
    _, chrf_15b = run_model(GATE_MODEL, val_items, demo_pool, ref_threshold=5.0)

    # Check gate from saved chrf_raw
    gate_passed, gate_info = check_gate(chrf_15b)

    if not gate_passed:
        log("\nSELF-GATE FAILED -- writing null report and stopping.")
        null = {
            "gate_passed": False,
            "gate_info": gate_info,
            "lang": TGT_LANG,
            "note": "Gain and gap both below threshold at k=16 on 1.5B, 30 items.",
        }
        with open(os.path.join(OUT, "gate_failed.json"), "w") as f:
            json.dump(null, f, indent=2)
        _log_fh.close()
        return

    # Compute threshold from 1.5B k=16 true scores
    k16_true_scores = chrf_15b["true"]["16"]
    threshold = max(statistics.median(k16_true_scores), 3.0)
    log(f"\nThreshold: median chrF at (1.5B, k=16, true) = {threshold:.2f}")
    with open(os.path.join(OUT, "threshold.json"), "w") as f:
        json.dump({"threshold": threshold, "method": "median_1.5B_k16_true"}, f, indent=2)

    # Re-binarize the 1.5B response with the correct threshold
    log("Re-binarizing 1.5B responses with final threshold...")
    safe_15b = GATE_MODEL.replace("/", "__")
    resp_path = os.path.join(OUT, f"responses_{safe_15b}.json")
    with open(resp_path) as f:
        resp_15b = json.load(f)
    for cond in CONDITIONS:
        for k in K_VALUES:
            sc = chrf_15b[cond][str(k)]
            binary = [1 if s >= threshold else 0 for s in sc]
            resp_15b["correct"][cond][str(k)] = binary
            resp_15b["accuracy"][cond][str(k)] = float(sum(binary) / len(binary))
    resp_15b["meta"]["threshold"] = threshold
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(resp_15b, f, indent=2)

    # Run 0.5B and 3B
    other_models = [m for m in MODELS if m != GATE_MODEL]
    for model_name in other_models:
        log(f"\nStep: Run {model_name.split('/')[-1]} full ladder")
        run_model(model_name, val_items, demo_pool, ref_threshold=threshold)

    log("\nAll models done.")
    _log_fh.close()


if __name__ == "__main__":
    main()
