"""
run_experiment_mairt.py -- ma-irt engine replication of the RQ4/RQ5 EdNet
separability study (the engine-swap of ``run_experiment.py``).

WHY this file exists.  The prior separability passes (cross-part consistency,
separability index, person x part interaction, per-part reliability, the
construct-distance gradient) were established on the SUBSTRATE LSTM encoder.
The PI asked for a robustness/replication check: do those anchored-readout
passes HOLD or IMPROVE on the better ma-irt engine (LSTMGPCM, separate_theta
=True, the audited-correct ability pathway)?  We re-run the SAME protocol on
the SAME EdNet parts-1-7 data with the ma-irt joint fit swapped in, keeping the
anchored fixed-parameter readout and the analysis UNCHANGED.

WHAT changes vs run_experiment.py.  Only the engine-specific pieces:
  * joint fit: ``model.train_joint`` (DeepIRTModel) -> ``model_mairt.
    train_joint_mairt`` (LSTMGPCM).
  * item-param recovery for the anchored readout: ``anchored_readout.
    recover_fixed_item_params`` (deep_irt) -> ``model_mairt.
    recover_fixed_item_params_mairt`` (ma-irt; alpha averaged over occurrences,
    item-only unsorted beta).
  * encoder readout baseline: ma-irt final-step theta over part-restricted
    subsequences (local helpers below), the analog of ``model.part_ability`` /
    ``model.part_representation``.

EVERYTHING downstream is reused VERBATIM from the deep_irt study:
  * the anchored MAP per-part theta estimator (``anchored_readout`` functions),
  * the variance decomposition, cross-part alignment, leakage probe
    (``analysis`` functions),
  * the data layer (``data.load_records`` etc.) on the identical learner subset.

This is legitimate because the anchored readout + analysis are ENGINE-AGNOSTIC
given a fixed per-item (a, b) table, and ma-irt's D=1 GPCM likelihood is
identical to the readout's (alpha*(theta-beta) == a*(theta-b)); see model_mairt.

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="rl/src;ma-irt"
    python deep_irt/ednet_sep/run_experiment_mairt.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Path setup: deep_irt (parent of core/), rl/src, and ma-irt importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from deep_irt.core.realdata import collate_adapter_items

from deep_irt.ednet_sep.data import (
    load_records, part_subsequences, split_half, LearnerRecord,
)
from deep_irt.ednet_sep.model_mairt import (
    train_joint_mairt, recover_fixed_item_params_mairt, _q1_batch_from_records,
)
from deep_irt.ednet_sep.analysis import (
    variance_decomposition, leakage_probe, cross_part_alignment, spearman_brown,
)
from deep_irt.ednet_sep.anchored_readout import (
    build_anchored_ability_table, anchored_split_half_reliability,
)
from deep_irt.ednet_sep.partmap import PARTS, part_section


# ---------------------------------------------------------------------------
# Constants -- matched to the deep_irt run (run_experiment.py) for an
# apples-to-apples comparison.  ma-irt widths use key/d/value=64 to mirror the
# deep_irt emb_dim=16 + hidden_dim=64 capacity envelope (the deep_irt splits
# its 16-d item embedding and 64-d hidden; ma-irt's q_embed/d_model are the
# closest analogs and 64 is the ma-irt LSTM default).
# ---------------------------------------------------------------------------
RAW_CSV = REPO_ROOT / "rl" / "data" / "ednet_raw" / "ednet.csv"
ARTEFACT_ROOT = REPO_ROOT / "rl" / "data" / "ednet_artefacts"
OUTPUT_DIR = Path(os.environ.get(
    "EDNET_SEP_MAIRT_OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "outputs_mairt")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_LEARNERS = int(os.environ.get("EDNET_SEP_N_LEARNERS", "4000"))
MIN_PART_RESP = int(os.environ.get("EDNET_SEP_MIN_PART_RESP", "10"))
MIN_PARTS = 2
MAX_SEQ_LEN = int(os.environ.get("EDNET_SEP_MAX_SEQ_LEN", "400"))
KEY_DIM = int(os.environ.get("EDNET_SEP_MAIRT_KEY_DIM", "64"))
D_MODEL = int(os.environ.get("EDNET_SEP_MAIRT_D_MODEL", "64"))
VALUE_DIM = int(os.environ.get("EDNET_SEP_MAIRT_VALUE_DIM", "64"))
N_EPOCHS = int(os.environ.get("EDNET_SEP_EPOCHS", "60"))
LR = float(os.environ.get("EDNET_SEP_LR", "0.01"))
BATCH_SIZE = int(os.environ.get("EDNET_SEP_BATCH", "128"))
K = 4
SEED = 0

# Fixed-parameter alpha-readout mode for the anchored study.  Default reproduces
# the original ma-irt run (occurrence-averaged alpha).  "canonical_item_only"
# reads a deterministic item-only alpha at the population-mean joint state.
ALPHA_MODE = os.environ.get("EDNET_SEP_MAIRT_ALPHA_MODE", "occurrence_avg")

ANCHOR_PRIOR_SD = float(os.environ.get("EDNET_SEP_ANCHOR_PRIOR_SD", "1.0"))
ANCHOR_STEPS = int(os.environ.get("EDNET_SEP_ANCHOR_STEPS", "200"))
ANCHOR_LR = 0.2


def tprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# ma-irt encoder readout (final-step theta over a part-restricted subsequence)
# the analog of model.part_ability / model.part_representation.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mairt_part_theta(
    model,
    subseq_items: List[Dict],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Final-step ma-irt theta for a list of part-restricted subsequences.

    Runs the frozen ma-irt encoder+decoder over each learner's part-only history
    and reads theta (the item-blind ability) at the final valid step.  This is
    the OOD-confounded encoder readout the anchored readout is meant to correct;
    we reproduce it on ma-irt so the encoder-vs-anchored delta is comparable.
    """
    if len(subseq_items) == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    b = collate_adapter_items(subseq_items)
    q = torch.where(b.mask, b.item_ids + 1, torch.zeros_like(b.item_ids)).to(device)
    r = b.responses.to(device)
    seq_lens = b.seq_lens
    model.eval()
    out = model(q, r)
    theta = out["theta"].squeeze(-1)                  # (M, S)  D=1
    idx = (seq_lens - 1).to(device)
    rows = torch.arange(theta.size(0), device=device)
    theta_final = theta[rows, idx]                    # (M,)
    return (
        theta_final.detach().cpu().numpy().astype(np.float64),
        b.student_ids.cpu().numpy().astype(np.int64),
    )


def build_encoder_ability_table_mairt(
    model, records: List[LearnerRecord], device: torch.device,
) -> Dict[int, Dict[int, float]]:
    """ma-irt encoder-readout per-part ability table (the confounded baseline)."""
    ability: Dict[int, Dict[int, float]] = {}
    for part in PARTS:
        subseqs = part_subsequences(records, part, min_events=MIN_PART_RESP)
        if not subseqs:
            continue
        theta, sids = _mairt_part_theta(model, subseqs, device=device)
        for j, sid in enumerate(sids):
            ability.setdefault(int(sid), {})[part] = float(theta[j])
    return ability


def build_encoder_reliability_mairt(
    model, records: List[LearnerRecord], device: torch.device,
) -> Dict[int, dict]:
    """Per-part split-half reliability under the ma-irt ENCODER readout."""
    out: Dict[int, dict] = {}
    for part in PARTS:
        halves_a: List[dict] = []
        halves_b: List[dict] = []
        for r in records:
            split = split_half(r, part, seed=SEED)
            if split is None:
                continue
            ha, hb = split
            if len(ha["questions"]) == 0 or len(hb["questions"]) == 0:
                continue
            halves_a.append(ha)
            halves_b.append(hb)
        if len(halves_a) < 30:
            out[part] = {"n": len(halves_a), "split_half_r": float("nan"),
                         "reliability": float("nan")}
            continue
        theta_a, sid_a = _mairt_part_theta(model, halves_a, device=device)
        theta_b, sid_b = _mairt_part_theta(model, halves_b, device=device)
        map_b = {int(s): theta_b[i] for i, s in enumerate(sid_b)}
        xs, ys = [], []
        for i, s in enumerate(sid_a):
            if int(s) in map_b:
                xs.append(theta_a[i]); ys.append(map_b[int(s)])
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 30 or np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            r_half = float("nan")
        else:
            r_half = float(pearsonr(xs, ys)[0])
        out[part] = {"n": len(xs), "split_half_r": r_half,
                     "reliability": float(spearman_brown(r_half))}
    return out


def _mean_corrected(align: dict) -> float:
    vals = [p["corrected"] for p in align["pairs"] if np.isfinite(p["corrected"])]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_raw(align: dict) -> float:
    vals = [p["pearson"] for p in align["pairs"]]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Prior deep_irt anchored numbers (from deep_irt/ednet_sep/outputs) for the
# side-by-side table.  Loaded from the deep_irt results.json if present, else
# the documented constants.
# ---------------------------------------------------------------------------

def load_prior_deep_irt() -> dict:
    path = REPO_ROOT / "deep_irt" / "ednet_sep" / "outputs" / "results.json"
    fallback = {
        "anchored_consistency": 0.7223,
        "anchored_sep_index": 0.5831,
        "anchored_interaction_share": 0.4104,
        "anchored_person_share": 0.5831,
        "anchored_part_share": 0.0065,
        "within_corrected": 0.7897,
        "cross_corrected": 0.6718,
        "reliability": {1: 0.797, 2: 0.820, 3: 0.875, 4: 0.796,
                        5: 0.855, 6: 0.801, 7: 0.798},
        "balanced": "373 persons x 5 parts [1, 2, 3, 5, 6]",
    }
    if not path.exists():
        return fallback
    try:
        r = json.load(open(path))
        v = r["verification_anchored_readout"]
        avd = v["anchored_rq5b_variance_decomposition"]
        ag = v["anchored_rq4_cross_part_alignment_own_ceilings"]["gradient"]
        arel = v.get("anchored_reliability", {})
        return {
            "anchored_consistency": v.get("anchored_disattenuated_consistency_own_ceilings"),
            "anchored_sep_index": v.get("anchored_separability_index"),
            "anchored_interaction_share": avd.get("interaction_var_share"),
            "anchored_person_share": avd.get("person_var_share"),
            "anchored_part_share": avd.get("part_var_share"),
            "within_corrected": ag.get("within_section_mean_corrected"),
            "cross_corrected": ag.get("cross_section_mean_corrected"),
            "reliability": {int(p): d.get("reliability") for p, d in arel.items()},
            "balanced": f"{avd.get('balanced_n_persons')} persons x "
                        f"{avd.get('balanced_n_parts')} parts {avd.get('balanced_parts')}",
        }
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(
    meta: dict, prior: dict,
    enc_vd: dict, enc_align: dict, enc_rel: dict,
    anc_vd: dict, anc_align: dict, anc_rel: dict,
) -> str:
    enc_mc = _mean_corrected(enc_align)
    anc_mc = _mean_corrected(anc_align)
    enc_raw = _mean_raw(enc_align)
    anc_raw = _mean_raw(anc_align)
    anc_g = anc_align["gradient"]
    enc_sep = enc_vd.get("separability_index", float("nan"))
    anc_sep = anc_vd.get("separability_index", float("nan"))

    L = ["=" * 78,
         "RQ4/RQ5 EdNet SEPARABILITY -- ma-irt ENGINE REPLICATION",
         "(engine = LSTMGPCM, separate_theta=True; anchored readout + analysis UNCHANGED)",
         "=" * 78, "",
         "DESIGN",
         f"  Engine        : ma-irt LSTMGPCM (key={KEY_DIM}, d_model={D_MODEL}, "
         f"value={VALUE_DIM}, separate_theta=True)",
         f"  Decoder       : K={K} GPCM (alpha state-conditioned, beta item-only)",
         f"  Alpha readout : {meta.get('alpha_mode', 'occurrence_avg')} "
         f"(frozen per-item discrimination)",
         f"  Joint fit     : {meta['n_learners']} learners, {meta['n_questions']} items, "
         f"{N_EPOCHS} epochs, batch {BATCH_SIZE}, lr {LR}",
         f"  Final NLL     : {meta['final_nll']:.4f}   (train {meta['wall_clock']:.1f}s)",
         f"  Anchored read : fix ma-irt per-item (a=mean alpha over occurrences, "
         f"b=item-only threshold); per-part theta by MAP (N(0,{ANCHOR_PRIOR_SD:.1f}^2))",
         f"  Sections      : LISTENING = parts 1-4,  READING = parts 5-7",
         "",
         "-" * 78,
         "ANCHORED PER-PART SPLIT-HALF RELIABILITY (Spearman-Brown)",
         "-" * 78]
    for p in PARTS:
        rr = anc_rel.get(p, {})
        prior_rel = prior["reliability"].get(p, float("nan"))
        L.append(f"  part {p} ({part_section(p):9s}): reliability={rr.get('reliability', float('nan')):.3f}  "
                 f"(prior deep_irt {prior_rel:.3f})  r_half={rr.get('split_half_r', float('nan')):.3f}  n={rr.get('n', 0)}")

    L += ["",
          "=" * 78,
          "HEADLINE COMPARISON -- ma-irt (this run) vs prior SUBSTRATE",
          "(both anchored fixed-parameter readout; own-ceiling disattenuation)",
          "=" * 78,
          "",
          "  metric                                   deep_irt    ma-irt     delta",
          "  " + "-" * 70,
          f"  RQ4 cross-part disattenuated consistency   {prior['anchored_consistency']:6.3f}     {anc_mc:6.3f}    {anc_mc - prior['anchored_consistency']:+.3f}",
          f"  RQ5b separability index (person share)     {prior['anchored_sep_index']:6.3f}     {anc_sep:6.3f}    {anc_sep - prior['anchored_sep_index']:+.3f}",
          f"  RQ5b person x part interaction share       {prior['anchored_interaction_share']:6.3f}     {anc_vd.get('interaction_var_share', float('nan')):6.3f}    {anc_vd.get('interaction_var_share', float('nan')) - prior['anchored_interaction_share']:+.3f}",
          f"  RQ5b part main-effect share                {prior['anchored_part_share']:6.3f}     {anc_vd.get('part_var_share', float('nan')):6.3f}    {anc_vd.get('part_var_share', float('nan')) - prior['anchored_part_share']:+.3f}",
          f"  RQ4 gradient within-section (corrected)    {prior['within_corrected']:6.3f}     {anc_g['within_section_mean_corrected']:6.3f}    {anc_g['within_section_mean_corrected'] - prior['within_corrected']:+.3f}",
          f"  RQ4 gradient cross-section  (corrected)    {prior['cross_corrected']:6.3f}     {anc_g['cross_section_mean_corrected']:6.3f}    {anc_g['cross_section_mean_corrected'] - prior['cross_corrected']:+.3f}",
          "  " + "-" * 70,
          f"  prior deep_irt balanced grid : {prior['balanced']}",
          f"  ma-irt   balanced grid : {anc_vd.get('balanced_n_persons')} persons x "
          f"{anc_vd.get('balanced_n_parts')} parts {anc_vd.get('balanced_parts')}",
          ""]

    L += ["-" * 78,
          "ma-irt ENCODER READOUT vs ANCHORED READOUT (does anchoring still help?)",
          "-" * 78,
          "  metric                                   encoder    anchored    delta",
          "  " + "-" * 70,
          f"  RQ4 disattenuated consistency (own ceil)   {enc_mc:6.3f}     {anc_mc:6.3f}    {anc_mc - enc_mc:+.3f}",
          f"  RQ4 raw mean cross-part pearson            {enc_raw:6.3f}     {anc_raw:6.3f}    {anc_raw - enc_raw:+.3f}",
          f"  RQ5b separability index                    {enc_sep:6.3f}     {anc_sep:6.3f}    {anc_sep - enc_sep:+.3f}",
          ""]

    L += ["-" * 78,
          "RQ4 ANCHORED CROSS-PART ALIGNMENT (per pair, ma-irt)",
          "-" * 78,
          "  pair        n     pearson  ceiling  corrected   section"]
    for pr in anc_align["pairs"]:
        tag = "within" if pr["same_section"] else "CROSS"
        L.append(f"  P{pr['part_a']}-P{pr['part_b']}   {pr['n_shared']:5d}   "
                 f"{pr['pearson']:+.3f}   {pr['attenuation_ceiling']:.3f}    "
                 f"{pr['corrected']:+.3f}    {tag}")
    L += ["",
          f"  within-section corrected mean = {anc_g['within_section_mean_corrected']:.3f}  "
          f"(n={anc_g['n_within_pairs']})",
          f"  cross-section  corrected mean = {anc_g['cross_section_mean_corrected']:.3f}  "
          f"(n={anc_g['n_cross_pairs']})",
          f"  gradient gap (within-cross, raw) = {anc_g['gradient_gap_raw']:+.3f}  "
          f"holds = {anc_g['gradient_holds_raw']}",
          f"  DISATTENUATED person-consistency (mean corrected over pairs) = {anc_mc:.3f}"]

    # ---- Verdict ----
    L += ["", "=" * 78, "VERDICT (PI 'investigate then continue' rule)", "=" * 78]

    def _delta_tag(name, ma, sub, higher_better=True):
        d = ma - sub
        # tolerance for "AGREE": within +/-0.05 of the deep_irt number
        if abs(d) <= 0.05:
            return f"  {name}: AGREE  (ma-irt {ma:.3f} vs deep_irt {sub:.3f}, delta {d:+.3f})"
        improve = (d > 0) == higher_better
        word = "IMPROVE" if improve else "DEGRADE"
        return f"  {name}: {word}  (ma-irt {ma:.3f} vs deep_irt {sub:.3f}, delta {d:+.3f})"

    L.append(_delta_tag("RQ4 cross-part consistency", anc_mc, prior["anchored_consistency"], True))
    L.append(_delta_tag("RQ5b separability index", anc_sep, prior["anchored_sep_index"], True))
    L.append(_delta_tag("RQ5b interaction share", anc_vd.get("interaction_var_share", float("nan")),
                        prior["anchored_interaction_share"], False))
    grad_ma = anc_g["within_section_mean_corrected"] - anc_g["cross_section_mean_corrected"]
    grad_sub = prior["within_corrected"] - prior["cross_corrected"]
    gtag = "HOLDS" if grad_ma > 0 else "INVERTED"
    L.append(f"  RQ4 construct-distance gradient: within {anc_g['within_section_mean_corrected']:.3f} "
             f"{'>' if grad_ma > 0 else '<='} cross {anc_g['cross_section_mean_corrected']:.3f} "
             f"(gap {grad_ma:+.3f}; deep_irt gap {grad_sub:+.3f}) -> {gtag}")

    # Anchoring still helps on ma-irt?
    anc_delta = anc_mc - enc_mc
    if anc_delta > 0.03:
        L.append(f"  ANCHORING TIE-IN: on ma-irt the anchored readout still RAISES cross-part "
                 f"consistency over the encoder ({anc_mc:.3f} vs {enc_mc:.3f}, +{anc_delta:.3f}) -- "
                 f"the identifiability mechanism does double duty on the better engine too.")
    elif anc_delta < -0.03:
        L.append(f"  ANCHORING TIE-IN: on ma-irt the anchored readout is LOWER than the encoder "
                 f"({anc_mc:.3f} vs {enc_mc:.3f}, {anc_delta:+.3f}) -- investigate.")
    else:
        L.append(f"  ANCHORING TIE-IN: anchored ~ encoder on ma-irt ({anc_mc:.3f} vs {enc_mc:.3f}, "
                 f"{anc_delta:+.3f}).")

    L += ["", "=" * 78]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = OUTPUT_DIR / "run.log"
    log_fh = open(log_path, "w", buffering=1)
    real_stdout = sys.stdout
    sys.stdout = log_fh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tprint("=" * 78)
    tprint("EDNET SEPARABILITY -- ma-irt ENGINE REPLICATION")
    tprint(f"device={device}  seed={SEED}")
    tprint("=" * 78)

    # 1. Data -- IDENTICAL subset to the deep_irt run.
    tprint("\n[1/6] Loading + subsampling EdNet learners (same subset as deep_irt)...")
    records, n_questions, _part_of_local = load_records(
        repo_root=REPO_ROOT, artefact_root=ARTEFACT_ROOT, raw_csv=RAW_CSV,
        n_learners=N_LEARNERS, min_part_resp=MIN_PART_RESP, min_parts=MIN_PARTS,
        max_seq_len=MAX_SEQ_LEN, seed=SEED, split="train",
    )
    tprint(f"  learners={len(records)}  items={n_questions}")

    # 2. Joint ma-irt fit.
    tprint("\n[2/6] Fitting joint ma-irt LSTMGPCM (separate_theta=True, K=4)...")
    model, info = train_joint_mairt(
        records, n_questions, n_cats=K, key_dim=KEY_DIM, d_model=D_MODEL,
        value_dim=VALUE_DIM, n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        device=device, seed=SEED, verbose=True,
    )
    tprint(f"  final NLL={info['final_nll']:.4f}  params={info['n_params']:,}  "
           f"t={info['wall_clock']:.1f}s")

    # 3. Encoder readout baseline (ma-irt final-step theta over part subseqs).
    tprint("\n[3/6] ma-irt ENCODER readout (per-part theta over part subsequences)...")
    enc_ability = build_encoder_ability_table_mairt(model, records, device)
    enc_rel = build_encoder_reliability_mairt(model, records, device)
    enc_rel_simple = {p: enc_rel[p]["reliability"] for p in enc_rel}
    enc_vd = variance_decomposition(enc_ability, parts=PARTS)
    enc_align = cross_part_alignment(enc_ability, enc_rel_simple, parts=PARTS, min_pairs=30)
    tprint(f"  encoder sep_index={enc_vd.get('separability_index', float('nan')):.3f}  "
           f"disatten consistency={_mean_corrected(enc_align):.3f}")

    # 4. Anchored readout (fix ma-irt item params, per-part MAP theta).
    tprint(f"\n[4/6] Recovering fixed ma-irt item params (alpha_mode={ALPHA_MODE}, "
           f"b=item-only)...")
    a_table, b_table = recover_fixed_item_params_mairt(
        model, records, n_questions, batch_size=BATCH_SIZE, device=device,
        alpha_mode=ALPHA_MODE)
    tprint(f"  a {a_table.shape}  b {b_table.shape}  "
           f"a[min/med/max]={a_table.min():.3f}/{np.median(a_table):.3f}/{a_table.max():.3f}")

    tprint("\n[5/6] Anchored fixed-parameter per-part MAP theta + reliability...")
    anc_ability = build_anchored_ability_table(
        records, a_table, b_table, min_events=MIN_PART_RESP,
        prior_sd=ANCHOR_PRIOR_SD, n_steps=ANCHOR_STEPS, lr=ANCHOR_LR, device=device)
    anc_rel = anchored_split_half_reliability(
        records, a_table, b_table, prior_sd=ANCHOR_PRIOR_SD,
        n_steps=ANCHOR_STEPS, lr=ANCHOR_LR, seed=SEED, device=device)
    anc_rel_simple = {p: anc_rel[p]["reliability"] for p in anc_rel}
    anc_vd = variance_decomposition(anc_ability, parts=PARTS)
    anc_align = cross_part_alignment(anc_ability, anc_rel_simple, parts=PARTS, min_pairs=30)
    tprint(f"  anchored sep_index={anc_vd.get('separability_index', float('nan')):.3f}  "
           f"disatten consistency={_mean_corrected(anc_align):.3f}")

    # 6. Report + JSON.
    tprint("\n[6/6] Writing report + results.json...")
    prior = load_prior_deep_irt()
    meta = {
        "engine": "ma-irt LSTMGPCM separate_theta=True",
        "n_learners": len(records), "n_questions": n_questions,
        "final_nll": info["final_nll"], "wall_clock": info["wall_clock"],
        "n_params": info["n_params"], "key_dim": KEY_DIM, "d_model": D_MODEL,
        "value_dim": VALUE_DIM, "K": K, "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
        "max_seq_len": MAX_SEQ_LEN, "min_part_resp": MIN_PART_RESP,
        "device": str(device), "alpha_mode": ALPHA_MODE,
    }
    report = format_report(meta, prior, enc_vd, enc_align, enc_rel,
                           anc_vd, anc_align, anc_rel)
    tprint("\n" + report)
    (OUTPUT_DIR / "report.txt").write_text(report)

    results = {
        "meta": meta,
        "prior_deep_irt_anchored": prior,
        "encoder_readout": {
            "reliability": enc_rel,
            "rq5b_variance_decomposition": enc_vd,
            "rq4_cross_part_alignment": enc_align,
            "disattenuated_consistency_own_ceilings": _mean_corrected(enc_align),
            "separability_index": enc_vd.get("separability_index", float("nan")),
        },
        "anchored_readout": {
            "prior_sd": ANCHOR_PRIOR_SD, "anchor_steps": ANCHOR_STEPS,
            "anchor_lr": ANCHOR_LR,
            "reliability": anc_rel,
            "rq5b_variance_decomposition": anc_vd,
            "rq4_cross_part_alignment": anc_align,
            "disattenuated_consistency_own_ceilings": _mean_corrected(anc_align),
            "separability_index": anc_vd.get("separability_index", float("nan")),
        },
    }

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (OUTPUT_DIR / "results.json").write_text(json.dumps(_san(results), indent=2))
    tprint(f"\nOutputs written to {OUTPUT_DIR}/")

    # Restore stdout and close the log so the interpreter shuts down cleanly
    # (otherwise the redirected stream teardown reports a non-zero exit).
    sys.stdout = real_stdout
    log_fh.close()


if __name__ == "__main__":
    main()
