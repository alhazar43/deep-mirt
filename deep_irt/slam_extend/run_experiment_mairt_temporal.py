"""
run_experiment_mairt_temporal.py -- RQ2 anchored extension, ma-irt engine,
APPLES-TO-APPLES discrimination measurement (the aggregation fix).

Motivation (the confound)
-------------------------
run_experiment_mairt.py concluded new-item DISCRIMINATION recovers at only
~0.44x its seed-reliability ceiling (a provisional "structural cap"), while
DIFFICULTY passes at ~0.80x. That comparison is NOT apples-to-apples for
discrimination:

  ma-irt's alpha is STATE-CONDITIONED PER STEP:
      alpha_t = exp(log_scale * disc_net([joint_summary_t, item_emb]))
  refined over the running encoder state joint_summary_t.

  - Reference, recover_full(): runs the FULL model over each learner's REAL
    interleaved sequence; per-item alpha = mean over all (learner, step)
    occurrences at the REAL encounter state. Temporal + student-averaged.
    CORRECT.
  - Anchored extraction in the prior script: conditions alpha on a SINGLE
    pinned final-base joint_summary per learner, .expand()ed constant across
    all steps. The per-step refinement is DISCARDED before the average. The
    two sides therefore average alpha over DIFFERENT state distributions; the
    0.44x null is partly an aggregation artifact, not necessarily a cap.

Beta (= threshold(item_embed), item-only) is NOT affected and already passes;
its logic and conclusion are reused verbatim and only confirmed not to regress.

This script (GOLD-STANDARD faithful variant)
--------------------------------------------
Variant ANCHORED: anchor BOTH the new-item KEY embedding (q_embed, feeds the
decoder + the per-step current-question signal) AND the new-item VALUE
embedding (LearnedEmbedding.item_embed, feeds the past-interaction value
stream so the FROZEN encoder can process new items IN-SEQUENCE). This is a
legitimate new-item parameter under Kolen-Brennan fixed-parameter anchoring.

  1. BASE FIT on base items (reused unchanged from run_experiment_mairt).
  2. Assemble an all-named LSTMGPCM: copy ALL base params + base embedding
     rows (key & value) FROZEN; the new-item key & value rows are fresh and
     the ONLY trainable parameters.
  3. Fit those new rows IN-SEQUENCE over the REAL interleaved (base+new)
     sequences, cross-entropy at NEW-item positions only. The encoder runs
     the real running history, so joint_summary at each new-item step is the
     REAL encounter state.
  4. Measure anchored item alpha with the SAME recover_full machinery over
     the REAL interleaved sequences -- per-step alpha at real encounter
     states, averaged over (learner, step). IDENTICAL aggregation to the
     reference. apples-to-apples.

Everything else (data layer, splits, base fit, full-recal, ceiling, difficulty
recovery, recover_full) is imported unchanged from run_experiment_mairt.py.
ma-irt is FROZEN: this script is additive only, it edits nothing under ma-irt/.

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="<repo-root>;rl/src;ma-irt"
    python deep_irt/slam_extend/run_experiment_mairt_temporal.py
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
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: repo root, rl/src, ma-irt importable (same as the parent script).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Force a NEW output dir BEFORE importing the parent (its OUTPUT_DIR is read at
# import time). We do not want to clobber outputs_mairt/.
DEFAULT_OUT = str(Path(__file__).resolve().parent / "outputs_mairt_temporal")
os.environ.setdefault("MAIRT_OUTPUT_DIR", DEFAULT_OUT)

# Reuse the parent ma-irt script's data layer, model build, base fit,
# full-recal, ceiling, difficulty recovery, and recover_full -- UNCHANGED.
from deep_irt.slam_extend.run_experiment_mairt import (
    tprint,
    load_slam,
    get_catchall_ids,
    get_n_named,
    filter_catchall,
    make_item_split,
    make_difficulty_split,
    subsequence,
    remap_ids_to_local,
    spearman,
    mean_b,
    ITEM_SPLIT_SEED,
    K,
    KEY_DIM,
    BATCH,
    BASE_EPOCHS,
    EXT_EPOCHS,
    RECAL_EPOCHS,
    EXT_LR,
    RECAL_SEEDS,
    DEVICE,
    build_model,
    train_full,
    recover_full,
    _q1_batch,
)
from models.lstm_gpcm import LSTMGPCM

OUTPUT_DIR = Path(os.environ["MAIRT_OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Gold-standard anchored fit: anchor new-item KEY + VALUE rows in-sequence.
# ---------------------------------------------------------------------------

# Embedding tables are 1-based indexed: row 0 is padding_idx, item local id g
# lives at row g+1. base_g2l/all_g2l map global 1-based id -> 0-based local id.

_KEY_TABLE = "encoder.q_embed.weight"
_VAL_TABLE = "encoder.embedding.item_embed.weight"


def assemble_anchored_model(base_model: LSTMGPCM,
                            n_named: int,
                            base_ids: np.ndarray,
                            new_ids: np.ndarray,
                            all_named_ids: np.ndarray,
                            seed: int) -> Tuple[LSTMGPCM, torch.Tensor, torch.Tensor]:
    """Build an all-named LSTMGPCM with base params + base embedding rows frozen
    and FRESH new-item KEY & VALUE rows.

    Returns (model, key_new_mask, val_new_mask) where the masks are
    (n_named+1,) bool tensors, True exactly on the new-item rows of each table
    (the only rows allowed to receive gradient).
    """
    model = build_model(n_named, seed=seed)  # fresh all-named LSTMGPCM on DEVICE
    base_sd = base_model.state_dict()
    new_sd = model.state_dict()

    # 1) Copy every shape-invariant (non-item-table) parameter from base.
    for k, v in base_sd.items():
        if k in (_KEY_TABLE, _VAL_TABLE):
            continue
        if k not in new_sd:
            raise KeyError(f"base key {k} absent from all-named model")
        if new_sd[k].shape != v.shape:
            raise ValueError(f"shape mismatch on {k}: {tuple(new_sd[k].shape)} "
                             f"vs base {tuple(v.shape)}")
        new_sd[k].copy_(v)

    # 2) Row-map the two item tables: base rows -> their all-named slot, frozen.
    #    Row index = local id + 1 (row 0 = padding_idx, left at fresh init = 0).
    base_g2l = {int(g): i for i, g in enumerate(base_ids)}
    all_g2l = {int(g): i for i, g in enumerate(all_named_ids)}
    new_set = set(int(g) for g in new_ids.tolist())

    key_new_mask = torch.zeros(n_named + 1, dtype=torch.bool, device=DEVICE)
    val_new_mask = torch.zeros(n_named + 1, dtype=torch.bool, device=DEVICE)

    base_key = base_sd[_KEY_TABLE]
    base_val = base_sd[_VAL_TABLE]
    new_key = new_sd[_KEY_TABLE]
    new_val = new_sd[_VAL_TABLE]
    # zero the pad row explicitly (padding_idx=0 -> kept at zero).
    new_key[0].zero_(); new_val[0].zero_()
    for g in all_named_ids.tolist():
        g = int(g)
        dst = all_g2l[g] + 1
        if g in new_set:
            key_new_mask[dst] = True
            val_new_mask[dst] = True
            # leave fresh init in new_key[dst], new_val[dst]
        else:
            src = base_g2l[g] + 1
            new_key[dst].copy_(base_key[src])
            new_val[dst].copy_(base_val[src])

    model.load_state_dict(new_sd)
    return model, key_new_mask, val_new_mask


def run_anchored_extension_temporal(base_model: LSTMGPCM,
                                    n_named: int,
                                    base_ids: np.ndarray,
                                    new_ids: np.ndarray,
                                    all_named_ids: np.ndarray,
                                    all_local: List[Dict],
                                    new_set_local0: set) -> dict:
    """Fit ONLY the new-item key & value rows over the REAL interleaved
    (base+new) sequences, cross-entropy at new-item positions only. Then
    recover anchored params with recover_full (temporal + student-averaged).

    new_set_local0: set of 0-based local ids (in the all_named_ids space) that
                    are NEW items -- used to mask the CE loss to new positions.
    """
    model, key_new_mask, val_new_mask = assemble_anchored_model(
        base_model, n_named, base_ids, new_ids, all_named_ids, seed=0)

    # Freeze everything, then re-enable grad ONLY on the two item tables.
    for p in model.parameters():
        p.requires_grad_(False)
    key_w = model.encoder.q_embed.weight
    val_w = model.encoder.embedding.item_embed.weight
    key_w.requires_grad_(True)
    val_w.requires_grad_(True)

    # Membership tensor for the new-item positions (0-based local id -> bool).
    is_new_row = torch.zeros(n_named + 1, dtype=torch.bool, device=DEVICE)
    for lid in new_set_local0:
        is_new_row[int(lid)] = True  # lid is 0-based local; q-1 used below

    q, r, mask, seq_lens, sids = _q1_batch(all_local)
    N = q.size(0)
    opt = torch.optim.Adam([key_w, val_w], lr=EXT_LR)
    gen = torch.Generator().manual_seed(123)
    n_params = int(key_new_mask.sum().item() * KEY_DIM
                   + val_new_mask.sum().item() * val_w.size(1))

    tprint(f"\n[Anchored Ext / TEMPORAL gold] {N} learners, "
           f"{int(key_new_mask.sum())} new items, {EXT_EPOCHS} epochs, "
           f"batch={BATCH}, trainable new-rows params={n_params:,} "
           f"(key+value)...")
    t0 = time.time()
    final = float("nan")
    model.train()  # base modules are frozen; train() only affects dropout (=0).
    for ep in range(1, EXT_EPOCHS + 1):
        perm = torch.randperm(N, generator=gen)
        tot_loss = 0.0; tot_tok = 0
        for s in range(0, N, BATCH):
            bi = perm[s:s + BATCH]
            qb = q[bi].to(DEVICE); rb = r[bi].to(DEVICE); mb = mask[bi].to(DEVICE)
            # New-item position mask: valid AND the item id is a new item.
            # ids0 is 0-based local (pad positions clamp to 0, then masked by mb).
            ids0 = (qb - 1).clamp(min=0)
            new_pos = mb & is_new_row.index_select(
                0, ids0.reshape(-1)).reshape(qb.shape)
            opt.zero_grad()
            logits = model(qb, rb)["logits"]          # (B,S,K)
            B, S, Kk = logits.shape
            ce = F.cross_entropy(logits.reshape(B * S, Kk),
                                 rb.reshape(B * S), reduction="none")
            mflat = new_pos.reshape(B * S).float()
            denom = mflat.sum().clamp(min=1)
            loss = (ce * mflat).sum() / denom
            loss.backward()
            # Keep base rows fixed: zero their grad on both tables.
            with torch.no_grad():
                key_w.grad[~key_new_mask] = 0.0
                val_w.grad[~val_new_mask] = 0.0
                key_w.grad[0] = 0.0; val_w.grad[0] = 0.0  # pad row
            opt.step()
            tot_loss += float(loss.item()) * int(mflat.sum().item())
            tot_tok += int(mflat.sum().item())
        final = tot_loss / max(tot_tok, 1)
        if ep % 50 == 0 or ep == 1:
            tprint(f"  [AnchorExt-T] epoch {ep:4d}  NLL(new-pos)={final:.4f}")
    wall = time.time() - t0

    # Recover anchored new-item params with the SAME recover_full machinery
    # over the REAL interleaved sequences -- identical aggregation to reference.
    a_all, b_all, seen_all = recover_full(model, all_local, n_named)
    tprint(f"  [AnchorExt-T] done. NLL={final:.4f}, t={wall:.1f}s")
    del model
    return {"a_all": a_all, "b_all": b_all, "seen_all": seen_all,
            "n_params": n_params, "train_time": wall, "final_nll": final}


# ---------------------------------------------------------------------------
# Core protocol -- mirrors run_experiment_mairt.run_protocol; only the anchored
# discrimination measurement changes (now temporal + student-averaged).
# ---------------------------------------------------------------------------

def run_protocol(full_records: List[Dict], base_ids: np.ndarray,
                 new_ids: np.ndarray, n_named: int, split_label: str) -> dict:
    base_set = set(base_ids.tolist())
    new_set = set(new_ids.tolist())
    n_base = len(base_ids); n_new = len(new_ids)

    tprint(f"\n{'='*65}\nPROTOCOL [ma-irt TEMPORAL]: {split_label.upper()} SPLIT")
    tprint(f"  Named items: {n_named}  BASE: {n_base}  NEW: {n_new}\n{'='*65}")

    base_seqs = subsequence(full_records, base_set, min_events=2)
    new_seqs = subsequence(full_records, new_set, min_events=2)
    base_sid = {r["student_id"] for r in base_seqs}
    new_sid = {r["student_id"] for r in new_seqs}
    shared_sid = base_sid & new_sid
    tprint(f"  Shared learners (base & new): {len(shared_sid)}")

    # BASE FIT uses BASE-only subsequences of the SHARED learners -- identical
    # to the parent script (so the base model is the same engine on the same
    # base history).
    base_seqs_shared = sorted([r for r in base_seqs if r["student_id"] in shared_sid],
                              key=lambda r: r["student_id"])
    base_local = remap_ids_to_local(base_seqs_shared, base_ids)

    # Real interleaved (base+new) sequences in the all-named local space.
    all_named_ids = np.sort(np.concatenate([base_ids, new_ids]))
    all_seqs = subsequence(full_records, set(all_named_ids.tolist()), min_events=2)
    all_local = remap_ids_to_local(all_seqs, all_named_ids)
    all_g2l = {int(g): i for i, g in enumerate(all_named_ids)}
    new_local_in_all = np.array([all_g2l[int(g)] for g in new_ids], dtype=np.int64)
    new_set_local0 = set(int(x) for x in new_local_in_all.tolist())

    # --- BASE FIT (unchanged engine) ---
    base_model, base_info = train_full(base_local, n_base, seed=0,
                                       n_epochs=BASE_EPOCHS, label="BaseFit")

    # --- ANCHORED EXTENSION (TEMPORAL gold) over REAL interleaved sequences ---
    anc = run_anchored_extension_temporal(
        base_model, n_named, base_ids, new_ids, all_named_ids,
        all_local, new_set_local0)
    a_anc_all = anc["a_all"]; b_anc_all = anc["b_all"]
    a_anc = a_anc_all[new_local_in_all]
    b_anc_mean = mean_b(b_anc_all[new_local_in_all])
    seen_anc = anc["seen_all"][new_local_in_all]

    # --- FULL RECAL x2 SEEDS (unchanged) ---
    recal_new = []
    recal_infos = []
    for seed in RECAL_SEEDS:
        m, info = train_full(all_local, n_named, seed=seed,
                             n_epochs=RECAL_EPOCHS, label=f"FullRecal_s{seed}")
        a_all, b_all, seen_all = recover_full(m, all_local, n_named)
        recal_new.append({"a": a_all[new_local_in_all],
                          "b_mean": mean_b(b_all[new_local_in_all]),
                          "seen": seen_all[new_local_in_all]})
        recal_infos.append(info)
        del m

    # --- CEILING (seed-to-seed) over items seen in both recal fits + anchored ---
    seen = seen_anc & recal_new[0]["seen"] & recal_new[1]["seen"]
    tprint(f"\n[Coverage] new items scored (seen in all fits): {int(seen.sum())}/{n_new}")
    ceil_diff = spearman(recal_new[0]["b_mean"][seen], recal_new[1]["b_mean"][seen])
    ceil_disc = spearman(recal_new[0]["a"][seen], recal_new[1]["a"][seen])
    tprint(f"[Ceiling] difficulty={ceil_diff:.4f}, discrimination={ceil_disc:.4f}")

    # --- PRIMARY: anchored vs full-recal(seed0) ---
    prim_diff = spearman(b_anc_mean[seen], recal_new[0]["b_mean"][seen])
    prim_disc = spearman(a_anc[seen], recal_new[0]["a"][seen])
    f_diff = prim_diff / ceil_diff if abs(ceil_diff) > 1e-6 else float("nan")
    f_disc = prim_disc / ceil_disc if abs(ceil_disc) > 1e-6 else float("nan")
    tprint(f"\n[PRIMARY] Anchored(TEMPORAL) vs FullRecal(seed=0)")
    tprint(f"  Difficulty Spearman:     {prim_diff:.4f}  (ceiling {ceil_diff:.4f}, frac {f_diff:.3f})")
    tprint(f"  Discrimination Spearman: {prim_disc:.4f}  (ceiling {ceil_disc:.4f}, frac {f_disc:.3f})")

    return {
        "split_label": split_label, "n_named": int(n_named),
        "n_base": int(n_base), "n_new": int(n_new),
        "n_shared_learners": int(len(shared_sid)),
        "n_new_scored": int(seen.sum()),
        "variant": "anchored_temporal_key_and_value",
        "ceiling_difficulty": ceil_diff, "ceiling_discrimination": ceil_disc,
        "primary_difficulty_spearman": prim_diff,
        "primary_discrimination_spearman": prim_disc,
        "primary_difficulty_frac": f_diff,
        "primary_discrimination_frac": f_disc,
        "ext_only_params": anc["n_params"], "ext_only_time": anc["train_time"],
        "ext_final_nll": anc["final_nll"],
        "base_time": base_info["wall_clock"], "recal_time": recal_infos[0]["wall_clock"],
        "recal_params": recal_infos[0]["n_params"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = OUTPUT_DIR / "run.log"
    log_fh = open(log_path, "w", buffering=1)
    sys.stdout = log_fh
    tprint("=" * 65)
    tprint("RQ2 [ma-irt TEMPORAL]: APPLES-TO-APPLES DISCRIMINATION RE-MEASURE")
    tprint("=" * 65)
    tprint(f"Variant: anchored KEY+VALUE rows fit in-sequence over REAL "
           f"interleaved (base+new); alpha recovered by recover_full "
           f"(temporal + student-averaged), identical aggregation to reference.")
    tprint(f"Config: KEY_DIM={KEY_DIM} BASE_EP={BASE_EPOCHS} EXT_EP={EXT_EPOCHS} "
           f"RECAL_EP={RECAL_EPOCHS} EXT_LR={EXT_LR} DEVICE={DEVICE}")

    adapter = load_slam()
    catchall_ids = get_catchall_ids(adapter)
    n_named = get_n_named(adapter)
    full_records = filter_catchall(adapter, catchall_ids, split="train")
    tprint(f"[Data] n_named={n_named}, learners(train,>=5)={len(full_records)}")

    SUBSAMPLE = int(os.environ.get("MAIRT_SUBSAMPLE", "1000"))
    if SUBSAMPLE and SUBSAMPLE < len(full_records):
        rng = np.random.default_rng(7)
        keep = rng.permutation(len(full_records))[:SUBSAMPLE]
        full_records = [full_records[i] for i in sorted(keep.tolist())]
        tprint(f"[Data] SUBSAMPLED to {len(full_records)} learners (seed=7)")

    base_ids, new_ids = make_item_split(n_named, seed=ITEM_SPLIT_SEED)
    tprint(f"[Split/random] BASE={len(base_ids)}, NEW={len(new_ids)}")
    res_random = run_protocol(full_records, base_ids, new_ids, n_named, "random")

    res_diff = None
    if os.environ.get("MAIRT_SKIP_STRESS", "0") != "1":
        try:
            bd, nd = make_difficulty_split(n_named)
            tprint(f"\n[Split/difficulty] BASE={len(bd)}, NEW={len(nd)}")
            res_diff = run_protocol(full_records, bd, nd, n_named, "difficulty")
        except Exception as e:
            tprint(f"[Stress] FAILED: {e}")

    combined = {"random": res_random}
    if res_diff is not None:
        combined["difficulty"] = res_diff
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({s: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                       for k, v in r.items()} for s, r in combined.items()},
                  f, indent=2)

    # Old numbers (from outputs_mairt/results.json) for side-by-side context.
    OLD = {
        "random": {"disc": 0.3067961676275506, "disc_ceil": 0.6978554390065117,
                   "disc_frac": 0.4396271068178747, "diff_frac": 0.8128336187016737},
        "difficulty": {"disc": 0.2955677545457256, "disc_ceil": 0.6498645792454649,
                       "disc_frac": 0.45481437823384524, "diff_frac": 0.790022677756395},
    }

    tprint("\n" + "=" * 65)
    tprint("SUMMARY (ma-irt TEMPORAL anchored extension, new-item recovery)")
    tprint("=" * 65)
    for s, r in combined.items():
        o = OLD.get(s, {})
        tprint(f"\n[{s.upper()}] scored {r['n_new_scored']}/{r['n_new']} new items, "
               f"{r['n_shared_learners']} shared learners")
        tprint(f"  ceiling   diff={r['ceiling_difficulty']:.3f}  disc={r['ceiling_discrimination']:.3f}")
        tprint(f"  anchored  diff={r['primary_difficulty_spearman']:.3f}  disc={r['primary_discrimination_spearman']:.3f}")
        tprint(f"  fraction  diff={r['primary_difficulty_frac']:.3f}  disc={r['primary_discrimination_frac']:.3f}")
        if o:
            tprint(f"  -- OLD (pinned-state aggregation): disc={o['disc']:.3f} "
                   f"(ceil {o['disc_ceil']:.3f}, frac {o['disc_frac']:.3f}); "
                   f"diff_frac={o['diff_frac']:.3f}")
            tprint(f"  -- DELTA disc fraction: {r['primary_discrimination_frac'] - o['disc_frac']:+.3f}")
    tprint(f"\nOutputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
