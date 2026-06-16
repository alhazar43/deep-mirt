"""
run_experiment.py -- RQ2: Item-agnosticism on real SLAM data.

Tests whether ANCHORED EXTENSION (freeze encoder + base item embeddings,
fit only new-item embeddings with theta pinned to base) reproduces
FULL RECALIBRATION for new items, at a fraction of the cost, without
disturbing the existing scale.

Protocol
--------
1. Load SLAM en_es via SlamAdapter (min_count=10). Drop catch-all items.
2. Split named items 70/30 BASE/NEW (seeded, deterministic).
3. BASE FIT: train DeepIRTModel on each learner's BASE-item subsequence.
4. ANCHORED EXTENSION: freeze encoder+base embeddings; fit NEW-item
   embeddings only, theta pinned to theta_base (final step per learner).
5. FULL RECALIBRATION x2 seeds: fresh model on ALL named items jointly.
6. RELIABILITY CEILING: Spearman between the two full-recal seeds.
7. PRIMARY: Spearman(anchored, full_recal_seed0) vs ceiling.
8. SECONDARY: base param/theta stability (should be ~exact after freeze).
9. TERTIARY: trainable params + wall-clock ratio.
10. STRESS: repeat with DIFFICULTY split (hardest 30% named items as NEW).

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="rl/src;ma-irt"
    python deep_irt/slam_extend/run_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr


def tprint(*args, **kwargs):
    """Print with immediate flush so output appears in redirected files."""
    print(*args, **kwargs)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Path setup: ensure deep_irt/ (parent of core/) and rl/src are importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rl" / "src"))

from ordrec.data.slam import SlamAdapter
from ordrec.data.base import AdapterConfig

from deep_irt.core.model import DeepIRTModel
from deep_irt.core.realdata import collate_adapter_items, CollatedBatch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLAM_RAW_DIR = REPO_ROOT / "rl" / "data" / "slam_raw"
ARTEFACT_DIR = REPO_ROOT / "rl" / "data" / "slam_artefact_mc10"
OUTPUT_DIR = Path(os.environ.get(
    "SLAM_OUTPUT_DIR", str(Path(__file__).resolve().parent / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 3            # ordinal categories
EMB_DIM = 16
HIDDEN_DIM = 64
BASE_EPOCHS = int(os.environ.get("SLAM_BASE_EPOCHS", "100"))
EXT_EPOCHS = int(os.environ.get("SLAM_EXT_EPOCHS", "100"))
RECAL_EPOCHS = int(os.environ.get("SLAM_RECAL_EPOCHS", "100"))
LR = 0.02
BATCH_SIZE = 256
ITEM_SPLIT_SEED = 42
RECAL_SEEDS = (0, 1)


# ---------------------------------------------------------------------------
# Step 1: Load SLAM adapter
# ---------------------------------------------------------------------------

def load_slam() -> SlamAdapter:
    cfg = AdapterConfig(
        name="slam_en_es_mc10",
        raw_dir=SLAM_RAW_DIR,
        out_dir=ARTEFACT_DIR,
        min_seq_len=5,
        max_seq_len=0,          # no cap
        chunk_long_sequences=False,
    )
    adapter = SlamAdapter(cfg, min_count=10)
    artefact_path = ARTEFACT_DIR / "slam_en_es_mc10" / "sequences.json"
    if not artefact_path.exists():
        tprint("[Data] Materialising SLAM artefact (first run)...")
        adapter.materialise()
    tprint("[Data] Loading SLAM artefact...")
    adapter.load()
    return adapter


def get_catchall_ids(adapter: SlamAdapter) -> set:
    """Return the set of 1-based catch-all item ids from coercion artefacts."""
    coercion = adapter._coercion
    catchall_map = coercion["item_selection"]["catchall_item_ids"]
    return set(catchall_map.values())  # 1-based


def get_n_named(adapter: SlamAdapter) -> int:
    return int(adapter._coercion["item_selection"]["n_named_items"])


# ---------------------------------------------------------------------------
# Step 2: Build per-learner filtered sequences (drop catch-all events)
# ---------------------------------------------------------------------------

def filter_catchall(
    adapter: SlamAdapter,
    catchall_ids: set,
    split: str = "train",
) -> List[Dict]:
    """
    Return list of dicts {student_id, questions (1-based), responses}
    for the requested split, with catch-all events removed.
    Only learners with >=5 remaining events are kept.
    """
    split_idx = adapter.get_split(split)
    records = []
    for i in split_idx:
        item = adapter[i]
        qs = item["questions"].tolist()
        rs = item["responses"].tolist()
        pairs = [(q, r) for q, r in zip(qs, rs) if q not in catchall_ids]
        if len(pairs) < 5:
            continue
        qs_f, rs_f = zip(*pairs)
        records.append({
            "student_id": int(item["student_id"]),
            "questions": torch.tensor(list(qs_f), dtype=torch.long),
            "responses": torch.tensor(list(rs_f), dtype=torch.long),
        })
    return records


# ---------------------------------------------------------------------------
# Step 3: Item split (BASE 70% / NEW 30%), deterministic
# ---------------------------------------------------------------------------

def make_item_split(
    n_named: int,
    seed: int = ITEM_SPLIT_SEED,
    base_frac: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (base_ids_1based, new_ids_1based) -- sorted numpy arrays.
    Named item IDs run from 1 to n_named inclusive.
    """
    rng = np.random.default_rng(seed)
    all_ids = np.arange(1, n_named + 1)
    perm = rng.permutation(all_ids)
    cut = int(len(perm) * base_frac)
    base_ids = np.sort(perm[:cut])
    new_ids = np.sort(perm[cut:])
    return base_ids, new_ids


def make_difficulty_split(
    n_named: int,
    new_frac: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NEW = hardest named items by human mistake-rate.
    Reads human_difficulty.csv; maps item_hash -> 1-based id via adapter.
    Returns (base_ids_1based, new_ids_1based).
    """
    csv_path = REPO_ROOT / "deep_irt" / "slam_pilot" / "outputs" / "human_difficulty.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"human_difficulty.csv not found at {csv_path}")

    import csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["item_hash"], float(row["human_difficulty"])))

    # Sort descending by difficulty (hardest first)
    rows.sort(key=lambda x: x[1], reverse=True)

    # We only have n_named items with IDs; the csv may have fewer or more.
    # We'll use up to n_named named items.
    n_new = int(n_named * new_frac)
    # We don't have a direct hash->id map here; load adapter coercion instead.
    # We'll load the slam artefact coercion from the adapter.
    coercion_path = ARTEFACT_DIR / "slam_en_es_mc10" / "coercion_artefacts.json"
    with open(coercion_path) as f:
        coercion = json.load(f)
    question_id_map = {}  # hash -> 1-based id
    # The question_id_map is in metadata, not coercion. Load metadata.
    meta_path = ARTEFACT_DIR / "slam_en_es_mc10" / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    question_id_map = {h: int(v) for h, v in meta["question_id_map"].items()}

    new_ids_list = []
    for item_hash, _ in rows:
        if item_hash in question_id_map:
            new_ids_list.append(question_id_map[item_hash])
        if len(new_ids_list) >= n_new:
            break

    new_ids = np.sort(np.array(new_ids_list, dtype=np.int64))
    base_ids = np.sort(np.setdiff1d(np.arange(1, n_named + 1), new_ids))
    return base_ids, new_ids


# ---------------------------------------------------------------------------
# Step 4: Sub-sequence extraction (only BASE or only NEW items per learner)
# ---------------------------------------------------------------------------

def subsequence(
    records: List[Dict],
    keep_ids_1based: set,
    min_events: int = 2,
) -> List[Dict]:
    """Filter each learner's sequence to only events from keep_ids_1based."""
    out = []
    for rec in records:
        qs = rec["questions"].tolist()
        rs = rec["responses"].tolist()
        pairs = [(q, r) for q, r in zip(qs, rs) if q in keep_ids_1based]
        if len(pairs) < min_events:
            continue
        qs_f, rs_f = zip(*pairs)
        out.append({
            "student_id": rec["student_id"],
            "questions": torch.tensor(list(qs_f), dtype=torch.long),
            "responses": torch.tensor(list(rs_f), dtype=torch.long),
        })
    return out


def remap_ids_to_local(records: List[Dict], ids_1based: np.ndarray) -> List[Dict]:
    """
    Remap 1-based global item ids to 0-based local ids in [0, len(ids)-1].
    ids_1based must be sorted.
    """
    global_to_local = {int(g): i for i, g in enumerate(ids_1based)}
    out = []
    for rec in records:
        local_qs = [global_to_local[int(q)] for q in rec["questions"].tolist()]
        out.append({
            "student_id": rec["student_id"],
            "questions": torch.tensor(local_qs, dtype=torch.long),
            "responses": rec["responses"].clone(),
        })
    return out


# ---------------------------------------------------------------------------
# Step 5: Model training helpers
# ---------------------------------------------------------------------------

def train_base_model(
    base_records_local: List[Dict],
    n_base: int,
    seed: int = 0,
    verbose: bool = True,
    label: str = "BASE",
) -> Tuple[DeepIRTModel, dict]:
    tprint(f"\n[{label}] Training on {len(base_records_local)} learners, "
          f"{n_base} items, {BASE_EPOCHS} epochs...")
    batch = collate_adapter_items(base_records_local)
    # item_ids are already 0-based from remap; collate_adapter_items subtracts 1
    # from 1-based ids -- but our records already have 0-based ids stored in
    # the questions field.  We need to pass them as 1-based to collate so it
    # converts them to 0-based.  Add 1 here.
    # Actually, collate_adapter_items subtracts 1 from questions to make 0-based.
    # Our local records have 0-based ids, so we need to pass them as (local+1).
    # Re-do: pass 1-based local (local+1).
    records_1based = []
    for rec in base_records_local:
        records_1based.append({
            "student_id": rec["student_id"],
            "questions": rec["questions"] + 1,  # 0-based -> 1-based for collate
            "responses": rec["responses"],
        })
    batch = collate_adapter_items(records_1based)

    model = DeepIRTModel(
        num_items=n_base,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        n_cats=K,
        decoder="gpcm",
        device=torch.device("cpu"),
        seed=seed,
    )
    t0 = time.time()
    info = model.fit(
        item_ids=batch.item_ids,
        responses=batch.responses,
        n_epochs=BASE_EPOCHS,
        lr=LR,
        verbose=verbose,
        mask=batch.mask,
        batch_size=BATCH_SIZE,
    )
    info["wall_clock"] = time.time() - t0
    n_params = sum(p.numel() for p in model.encoder.parameters()) + \
               sum(p.numel() for p in model.decoder.parameters())
    info["n_params"] = n_params
    tprint(f"  [{label}] Done. NLL={info['final_nll']:.4f}, "
          f"params={n_params}, t={info['wall_clock']:.1f}s")
    return model, info


def extract_theta_base(
    model: DeepIRTModel,
    base_records_local: List[Dict],
) -> Tuple[torch.Tensor, List[int]]:
    """
    Return final per-learner theta from base model and student_ids.
    Only learners present in base_records_local are included.
    """
    records_1based = [
        {
            "student_id": rec["student_id"],
            "questions": rec["questions"] + 1,
            "responses": rec["responses"],
        }
        for rec in base_records_local
    ]
    batch = collate_adapter_items(records_1based)
    theta_traj = model.track(batch.item_ids, batch.responses)  # (N, T_max)
    # Pick the theta at the last valid position for each learner.
    seq_lens = batch.seq_lens  # (N,)
    theta_final = theta_traj[torch.arange(len(seq_lens)), seq_lens - 1]  # (N,)
    student_ids = batch.student_ids.tolist()
    return theta_final.detach(), student_ids


def run_anchored_extension(
    base_model: DeepIRTModel,
    theta_base: torch.Tensor,
    new_records_local: List[Dict],
    n_new: int,
    verbose: bool = True,
) -> dict:
    """
    Run anchored extension.  theta_base is (N_base_learners,).
    new_records_local contains 0-based local ids for NEW items only,
    for learners that appear in the new split.

    The anchored_extend API expects:
        anchor_theta : (n_anchor,)        -- scalar per learner
        ext_item_ids : (n_anchor, E)      -- local ids {0..E-1}
        ext_responses: (n_anchor, E)      -- responses

    For variable-length sequences we pad.  Since anchor_theta is a scalar
    per learner (the final base theta), we expand it across all NEW-item
    positions.  The decoder NLL averages over all (learner, new-item) pairs.
    """
    # Build padded tensors for NEW items.
    records_1based = [
        {
            "student_id": rec["student_id"],
            "questions": rec["questions"] + 1,
            "responses": rec["responses"],
        }
        for rec in new_records_local
    ]
    batch = collate_adapter_items(records_1based)
    # batch.item_ids: (N, T_max) 0-based new-item local ids

    n_anchor = batch.item_ids.size(0)
    T_max = batch.item_ids.size(1)

    # anchor_theta must be (n_anchor,) -- match order of new_records_local.
    # new_records_local may have a different learner set than base_records.
    # theta_base was computed from base_records in order; we need to align.
    # We pass the full theta_base as-is: if lengths differ we use what we have.
    # Caller must ensure alignment. See run_protocol().

    # Masked extension: we feed the decoder only valid (non-padded) positions.
    # anchored_extend doesn't accept a mask natively; we adapt by passing
    # valid positions only via a compact representation.
    # Strategy: each learner has a different number of NEW events. We flatten
    # valid positions only: build (n_valid,) theta and item-emb tensors.
    # However anchored_extend needs (n_anchor, E) shape. Use max-padding with
    # a mask; we'll adapt anchor.py minimally or work around it here.

    # Cleanest approach: call anchored_extend with the padded batch but
    # use a custom wrapper that masks the NLL, mirroring DeepIRTModel._compute_nll.
    # We inline the anchored extension logic here to support masking.

    from deep_irt.core.anchor import _ExtEmbeddings
    import torch.optim as optim

    encoder = base_model.encoder
    decoder = base_model.decoder
    device = base_model.device

    # Freeze base model
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    decoder.eval()

    ext_mod = _ExtEmbeddings(n_ext=n_new, emb_dim=EMB_DIM).to(device)
    optimizer = optim.Adam(ext_mod.parameters(), lr=LR)
    n_params = sum(p.numel() for p in ext_mod.parameters())

    item_ids_dev = batch.item_ids.to(device)   # (N, T_max) 0-based local new ids
    responses_dev = batch.responses.to(device) # (N, T_max)
    mask_dev = batch.mask.to(device)           # (N, T_max) bool
    theta_dev = theta_base.to(device)          # (N,)

    tprint(f"\n[Anchored Ext] {n_anchor} learners, {n_new} new items, "
          f"{EXT_EPOCHS} epochs, trainable params={n_params}...")

    t0 = time.time()
    final_nll = float("nan")

    for epoch in range(1, EXT_EPOCHS + 1):
        ext_mod.train()
        optimizer.zero_grad()

        # Get embeddings for all new-item positions
        embs = ext_mod(item_ids_dev)               # (N, T_max, emb_dim)
        N, T, D = embs.shape

        # Expand theta across time: (N, T_max)
        theta_expanded = theta_dev.unsqueeze(1).expand(N, T)

        # Flatten
        embs_flat = embs.reshape(N * T, D)
        theta_flat = theta_expanded.reshape(N * T)
        resp_flat = responses_dev.reshape(N * T)
        mask_flat = mask_dev.reshape(N * T)

        # Select valid positions only
        embs_valid = embs_flat[mask_flat]
        theta_valid = theta_flat[mask_flat]
        resp_valid = resp_flat[mask_flat]

        loss = decoder.nll(theta_valid, embs_valid, resp_valid)
        loss.backward()
        optimizer.step()

        final_nll = loss.item()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [AnchorExt] epoch {epoch:4d}  NLL={final_nll:.4f}")

    train_time = time.time() - t0

    ext_mod.eval()
    with torch.no_grad():
        ext_emb_weight = ext_mod.ext_emb.weight.cpu().clone()  # (n_new, emb_dim)
        params_out = decoder.item_params_sorted(ext_emb_weight.to(device))
        a_ext = params_out["a"].squeeze(-1).cpu().numpy()
        b_ext = params_out["b"].cpu().numpy()

    tprint(f"  [AnchorExt] Done. NLL={final_nll:.4f}, t={train_time:.1f}s")

    return {
        "a_ext": a_ext,
        "b_ext": b_ext,
        "n_params": n_params,
        "train_time": train_time,
        "final_nll": final_nll,
    }


def train_full_recal(
    all_records_local: List[Dict],
    n_all: int,
    seed: int = 0,
    verbose: bool = True,
    label: str = "FullRecal",
) -> Tuple[DeepIRTModel, dict]:
    """Train a fresh full model on ALL named items (base+new) jointly."""
    tprint(f"\n[{label} seed={seed}] Training on {len(all_records_local)} learners, "
          f"{n_all} items, {RECAL_EPOCHS} epochs...")
    records_1based = [
        {
            "student_id": rec["student_id"],
            "questions": rec["questions"] + 1,
            "responses": rec["responses"],
        }
        for rec in all_records_local
    ]
    batch = collate_adapter_items(records_1based)
    model = DeepIRTModel(
        num_items=n_all,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        n_cats=K,
        decoder="gpcm",
        device=torch.device("cpu"),
        seed=seed,
    )
    t0 = time.time()
    info = model.fit(
        item_ids=batch.item_ids,
        responses=batch.responses,
        n_epochs=RECAL_EPOCHS,
        lr=LR,
        verbose=verbose,
        mask=batch.mask,
        batch_size=BATCH_SIZE,
    )
    info["wall_clock"] = time.time() - t0
    n_params = sum(p.numel() for p in model.encoder.parameters()) + \
               sum(p.numel() for p in model.decoder.parameters())
    info["n_params"] = n_params
    tprint(f"  [{label}] Done. NLL={info['final_nll']:.4f}, "
          f"params={n_params}, t={info['wall_clock']:.1f}s")
    return model, info


# ---------------------------------------------------------------------------
# Step 6: Metric helpers
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def mean_b(b: np.ndarray) -> np.ndarray:
    """Collapse (M, K-1) step thresholds to a scalar difficulty per item."""
    return b.mean(axis=1)


def recover_params_for_ids(
    model: DeepIRTModel,
    global_ids_1based: np.ndarray,
    local_id_map: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover (a, b) for a set of global 1-based item ids using their local
    indices in model's embedding table.
    Returns a (M,), b_mean (M,).
    """
    local_ids = np.array([local_id_map[g] for g in global_ids_1based], dtype=np.int64)
    params = model.recover_item_params(torch.tensor(local_ids, dtype=torch.long))
    a = params["a"]           # (M,)
    b_mean = mean_b(params["b"])  # (M,)
    return a, b_mean


# ---------------------------------------------------------------------------
# Core experiment protocol
# ---------------------------------------------------------------------------

def run_protocol(
    full_records: List[Dict],
    base_ids: np.ndarray,
    new_ids: np.ndarray,
    n_named: int,
    split_label: str = "random",
) -> dict:
    """
    Run the full RQ2 protocol for a given item split.

    full_records: all named-item-only learner sequences (0-based ids BUT
                  NOTE: they are stored as original 1-based from filter_catchall,
                  so we need to be careful here).

    Actually full_records has 1-based questions. We need to:
    - subsequence() to filter to base_ids or new_ids (1-based comparison)
    - remap_ids_to_local() to get 0-based local indices
    """

    base_set = set(base_ids.tolist())
    new_set = set(new_ids.tolist())
    n_base = len(base_ids)
    n_new = len(new_ids)

    tprint(f"\n{'='*65}")
    tprint(f"PROTOCOL: {split_label.upper()} SPLIT")
    tprint(f"  Named items: {n_named}  BASE: {n_base}  NEW: {n_new}")
    tprint(f"{'='*65}")

    # Sub-sequences
    base_seqs = subsequence(full_records, base_set, min_events=2)
    new_seqs = subsequence(full_records, new_set, min_events=2)

    # Align learners: keep only those that have BOTH base and new events.
    base_sid = {r["student_id"] for r in base_seqs}
    new_sid = {r["student_id"] for r in new_seqs}
    shared_sid = base_sid & new_sid
    tprint(f"  Learners with base events: {len(base_sid)}")
    tprint(f"  Learners with new events:  {len(new_sid)}")
    tprint(f"  Shared (used for anchor):  {len(shared_sid)}")

    base_seqs_shared = [r for r in base_seqs if r["student_id"] in shared_sid]
    new_seqs_shared  = [r for r in new_seqs  if r["student_id"] in shared_sid]
    # Sort both by student_id to ensure consistent ordering
    base_seqs_shared.sort(key=lambda r: r["student_id"])
    new_seqs_shared.sort(key=lambda r: r["student_id"])

    # Remap to local 0-based ids
    base_local = remap_ids_to_local(base_seqs_shared, base_ids)  # local 0-based
    new_local  = remap_ids_to_local(new_seqs_shared,  new_ids)

    # All-named sequences for full recal (local ids remapped to 0..n_named-1)
    all_named_ids = np.sort(np.concatenate([base_ids, new_ids]))
    all_seqs = subsequence(full_records, set(all_named_ids.tolist()), min_events=2)
    all_local = remap_ids_to_local(all_seqs, all_named_ids)

    # Global->local maps for parameter recovery
    base_g2l = {int(g): i for i, g in enumerate(base_ids)}
    new_g2l  = {int(g): i for i, g in enumerate(new_ids)}
    all_g2l  = {int(g): i for i, g in enumerate(all_named_ids)}

    # -----------------------------------------------------------------------
    # Step 3: BASE FIT
    # -----------------------------------------------------------------------
    base_model, base_info = train_base_model(
        base_local, n_base, seed=0, verbose=True, label="BaseFit"
    )

    # Extract per-learner theta_base (final step)
    theta_base, base_sids = extract_theta_base(base_model, base_local)
    # theta_base is aligned with base_seqs_shared order (same sort order)

    # -----------------------------------------------------------------------
    # Step 4: ANCHORED EXTENSION
    # -----------------------------------------------------------------------
    t_anchor_start = time.time()
    anchor_result = run_anchored_extension(
        base_model, theta_base, new_local, n_new, verbose=True
    )
    anchor_total_time = time.time() - t_anchor_start

    # Anchored params for new items (local index order matches new_ids sort order)
    a_anc = anchor_result["a_ext"]     # (n_new,) -- sorted by local id
    b_anc = anchor_result["b_ext"]     # (n_new, K-1)
    b_anc_mean = mean_b(b_anc)

    # -----------------------------------------------------------------------
    # Step 5: FULL RECALIBRATION x2 seeds
    # -----------------------------------------------------------------------
    recal_models = []
    recal_infos = []
    for seed in RECAL_SEEDS:
        m, info = train_full_recal(
            all_local, n_named, seed=seed, verbose=True,
            label=f"FullRecal_seed{seed}"
        )
        recal_models.append(m)
        recal_infos.append(info)

    # Recover NEW-item params from full-recal models
    # In all_local, new items have global ids in new_ids; their local index
    # in the all-named embedding is all_g2l[new_id].
    new_local_in_all = np.array([all_g2l[int(g)] for g in new_ids], dtype=np.int64)

    recal_params = []
    for m in recal_models:
        params = m.recover_item_params(torch.tensor(new_local_in_all, dtype=torch.long))
        recal_params.append({
            "a": params["a"],
            "b_mean": mean_b(params["b"]),
        })

    # -----------------------------------------------------------------------
    # Step 6: RELIABILITY CEILING
    # -----------------------------------------------------------------------
    ceil_diff = spearman(recal_params[0]["b_mean"], recal_params[1]["b_mean"])
    ceil_disc = spearman(recal_params[0]["a"],       recal_params[1]["a"])
    tprint(f"\n[Ceiling] Seed reliability -- difficulty: {ceil_diff:.4f}, "
          f"discrimination: {ceil_disc:.4f}")

    # -----------------------------------------------------------------------
    # Step 7: PRIMARY -- anchored vs full-recal
    # -----------------------------------------------------------------------
    # Use seed-0 full-recal as reference
    prim_diff = spearman(b_anc_mean, recal_params[0]["b_mean"])
    prim_disc = spearman(a_anc, recal_params[0]["a"])
    prim_diff_frac = prim_diff / ceil_diff if abs(ceil_diff) > 1e-6 else float("nan")
    prim_disc_frac = prim_disc / ceil_disc if abs(ceil_disc) > 1e-6 else float("nan")

    tprint(f"\n[PRIMARY] Anchored vs FullRecal(seed=0)")
    tprint(f"  Difficulty Spearman:      {prim_diff:.4f}  "
          f"(ceiling {ceil_diff:.4f}, fraction {prim_diff_frac:.3f})")
    tprint(f"  Discrimination Spearman:  {prim_disc:.4f}  "
          f"(ceiling {ceil_disc:.4f}, fraction {prim_disc_frac:.3f})")
    pass_fail = "PASS" if prim_diff_frac >= 0.8 else "PARTIAL" if prim_diff_frac >= 0.5 else "FAIL"
    tprint(f"  PRIMARY VERDICT: {pass_fail} (threshold 0.8 x ceiling)")

    # -----------------------------------------------------------------------
    # Step 8: SECONDARY -- base-scale stability
    # -----------------------------------------------------------------------
    # Base params from base_model vs from full-recal seed-0
    # In full-recal, base items have global ids base_ids; local in all_named = all_g2l.
    base_local_in_all = np.array([all_g2l[int(g)] for g in base_ids], dtype=np.int64)
    base_params_from_base = base_model.recover_item_params(
        torch.arange(n_base, dtype=torch.long)
    )
    base_params_from_recal = recal_models[0].recover_item_params(
        torch.tensor(base_local_in_all, dtype=torch.long)
    )

    # Sort both by same base_ids order for comparison
    sec_diff_base_spear = spearman(
        mean_b(base_params_from_base["b"]),
        mean_b(base_params_from_recal["b"])
    )
    sec_disc_base_spear = spearman(
        base_params_from_base["a"],
        base_params_from_recal["a"]
    )
    # Check theta stability: theta_base vs theta from full-recal
    # Recompute theta from full-recal on same learners
    # Use base_seqs_shared (same learners, same base events)
    all_seqs_shared = [r for r in all_seqs if r["student_id"] in shared_sid]
    all_seqs_shared.sort(key=lambda r: r["student_id"])
    all_local_shared = remap_ids_to_local(all_seqs_shared, all_named_ids)
    records_1based_all_sh = [
        {"student_id": r["student_id"],
         "questions": r["questions"] + 1,
         "responses": r["responses"]}
        for r in all_local_shared
    ]
    batch_all_sh = collate_adapter_items(records_1based_all_sh)
    theta_recal_traj = recal_models[0].track(batch_all_sh.item_ids, batch_all_sh.responses)
    seq_lens_sh = batch_all_sh.seq_lens
    theta_recal_final = theta_recal_traj[torch.arange(len(seq_lens_sh)), seq_lens_sh - 1]
    theta_base_np = theta_base.numpy()
    theta_recal_np = theta_recal_final.detach().numpy()

    sec_theta_spear = spearman(theta_base_np, theta_recal_np)
    tprint(f"\n[SECONDARY] Base-scale stability")
    tprint(f"  Base item difficulty Spearman (base vs recal):       {sec_diff_base_spear:.4f}")
    tprint(f"  Base item discrimination Spearman (base vs recal):   {sec_disc_base_spear:.4f}")
    tprint(f"  Per-learner theta Spearman (base vs recal):          {sec_theta_spear:.4f}")
    tprint(f"  (theta_base is frozen by construction; Spearman ~1 expected)")

    # -----------------------------------------------------------------------
    # Step 9: TERTIARY -- cost
    # -----------------------------------------------------------------------
    anchor_params = base_info["n_params"] + anchor_result["n_params"]
    recal_params_count = recal_infos[0]["n_params"]
    recal_time = recal_infos[0]["wall_clock"]
    # Anchored total time = base fit + extension
    anchor_total_time_full = base_info["wall_clock"] + anchor_result["train_time"]
    param_ratio = anchor_params / recal_params_count
    time_ratio = anchor_total_time_full / recal_time

    tprint(f"\n[TERTIARY] Cost comparison (base fit + anchor ext vs full recal)")
    tprint(f"  Anchored total params:    {anchor_params:,}  "
          f"(base={base_info['n_params']:,} + ext={anchor_result['n_params']:,})")
    tprint(f"  Full recal params:        {recal_params_count:,}")
    tprint(f"  Param ratio:              {param_ratio:.3f}x")
    tprint(f"  Anchored total time:      {anchor_total_time_full:.1f}s "
          f"(base={base_info['wall_clock']:.1f}s + ext={anchor_result['train_time']:.1f}s)")
    tprint(f"  Full recal time:          {recal_time:.1f}s")
    tprint(f"  Time ratio:               {time_ratio:.2f}x")
    tprint(f"  Extension-only params:    {anchor_result['n_params']:,}  "
          f"(ratio to recal: {anchor_result['n_params']/recal_params_count:.3f}x)")
    tprint(f"  Extension-only time:      {anchor_result['train_time']:.1f}s  "
          f"(ratio to recal: {anchor_result['train_time']/recal_time:.2f}x)")

    return {
        "split_label": split_label,
        "n_named": n_named,
        "n_base": n_base,
        "n_new": n_new,
        "n_shared_learners": len(shared_sid),
        # PRIMARY
        "ceiling_difficulty": ceil_diff,
        "ceiling_discrimination": ceil_disc,
        "primary_difficulty_spearman": prim_diff,
        "primary_discrimination_spearman": prim_disc,
        "primary_difficulty_frac": prim_diff_frac,
        "primary_discrimination_frac": prim_disc_frac,
        "primary_verdict": pass_fail,
        # SECONDARY
        "secondary_base_difficulty_spearman": sec_diff_base_spear,
        "secondary_base_discrimination_spearman": sec_disc_base_spear,
        "secondary_theta_spearman": sec_theta_spear,
        # TERTIARY
        "anchor_total_params": anchor_params,
        "recal_params": recal_params_count,
        "param_ratio": param_ratio,
        "anchor_total_time": anchor_total_time_full,
        "recal_time": recal_time,
        "time_ratio": time_ratio,
        "ext_only_params": anchor_result["n_params"],
        "ext_only_time": anchor_result["train_time"],
        "ext_only_param_ratio": anchor_result["n_params"] / recal_params_count,
        "ext_only_time_ratio": anchor_result["train_time"] / recal_time,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(results_random: dict, results_difficulty: Optional[dict]) -> str:
    lines = [
        "=" * 65,
        "RQ2: ITEM-AGNOSTICISM -- ANCHORED EXTENSION ON REAL SLAM DATA",
        "=" * 65,
        "",
        "DESIGN",
        "  Dataset:     SLAM 2018 en_es, SlamAdapter min_count=10",
        f"  Named items: {results_random['n_named']}",
        f"  Catch-all buckets excluded from all sequences.",
        f"  BASE items:  {results_random['n_base']} (70%, random split seed={ITEM_SPLIT_SEED})",
        f"  NEW items:   {results_random['n_new']} (30%)",
        f"  K = {K}  |  emb_dim = {EMB_DIM}  |  hidden_dim = {HIDDEN_DIM}",
        f"  Base epochs: {BASE_EPOCHS}  |  Ext epochs: {EXT_EPOCHS}  "
        f"|  Recal epochs: {RECAL_EPOCHS}  |  batch_size: {BATCH_SIZE}",
        "",
    ]

    def add_section(r: dict, label: str):
        lines.append(f"{'='*65}")
        lines.append(f"RESULTS: {label.upper()} SPLIT")
        lines.append(f"  Shared learners (have both base and new events): {r['n_shared_learners']}")
        lines.append("")
        lines.append("RELIABILITY CEILING (seed-to-seed, full-recal x2 seeds)")
        lines.append(f"  Difficulty ceiling:         {r['ceiling_difficulty']:.4f}")
        lines.append(f"  Discrimination ceiling:     {r['ceiling_discrimination']:.4f}")
        lines.append("")
        lines.append("PRIMARY: Anchored vs Full-Recal (seed=0)")
        lines.append(f"  Difficulty Spearman:        {r['primary_difficulty_spearman']:.4f}  "
                     f"[fraction of ceiling: {r['primary_difficulty_frac']:.3f}]")
        lines.append(f"  Discrimination Spearman:    {r['primary_discrimination_spearman']:.4f}  "
                     f"[fraction of ceiling: {r['primary_discrimination_frac']:.3f}]")
        lines.append(f"  VERDICT: {r['primary_verdict']}  (pass >= 0.80 x ceiling)")
        lines.append("")
        lines.append("SECONDARY: Base-scale stability (frozen by construction)")
        lines.append(f"  Base item difficulty Spearman (base vs recal):       {r['secondary_base_difficulty_spearman']:.4f}")
        lines.append(f"  Base item discrimination Spearman (base vs recal):   {r['secondary_base_discrimination_spearman']:.4f}")
        lines.append(f"  Per-learner theta Spearman (base vs recal):          {r['secondary_theta_spearman']:.4f}")
        lines.append("")
        lines.append("TERTIARY: Cost (extension-only params/time vs full recal)")
        lines.append(f"  Ext-only trainable params:  {r['ext_only_params']:,}  "
                     f"(ratio: {r['ext_only_param_ratio']:.3f}x full recal)")
        lines.append(f"  Ext-only wall-clock:        {r['ext_only_time']:.1f}s  "
                     f"(ratio: {r['ext_only_time_ratio']:.2f}x full recal)")
        lines.append(f"  Full recal params:          {r['recal_params']:,}  "
                     f"wall-clock: {r['recal_time']:.1f}s")
        lines.append("")

    add_section(results_random, "random")

    if results_difficulty is not None:
        add_section(results_difficulty, "difficulty (stress)")
        # Degradation comparison
        deg_diff = results_random["primary_difficulty_frac"] - results_difficulty["primary_difficulty_frac"]
        deg_disc = results_random["primary_discrimination_frac"] - results_difficulty["primary_discrimination_frac"]
        lines.append("DEGRADATION (random vs difficulty split)")
        lines.append(f"  Difficulty fraction:  {results_random['primary_difficulty_frac']:.3f}  ->  "
                     f"{results_difficulty['primary_difficulty_frac']:.3f}  "
                     f"(delta = {deg_diff:+.3f})")
        lines.append(f"  Discrimination frac:  {results_random['primary_discrimination_frac']:.3f}  ->  "
                     f"{results_difficulty['primary_discrimination_frac']:.3f}  "
                     f"(delta = {deg_disc:+.3f})")
        lines.append("")

    # Final verdict
    lines.append("=" * 65)
    r = results_random
    if r["primary_verdict"] == "PASS":
        verdict = (
            f"PASS. Anchored extension achieves {r['primary_difficulty_frac']:.2f}x the "
            f"seed-reliability ceiling on new-item difficulty (Spearman={r['primary_difficulty_spearman']:.3f} "
            f"vs ceiling={r['ceiling_difficulty']:.3f}), meeting the 0.80x threshold. "
            f"Discrimination recovery is {r['primary_discrimination_spearman']:.3f} "
            f"({r['primary_discrimination_frac']:.2f}x ceiling), consistent with known noisier "
            f"discrimination estimation. Extension-only cost: {r['ext_only_param_ratio']:.2f}x params "
            f"and {r['ext_only_time_ratio']:.2f}x wall-clock vs full recal. "
            f"Base-scale theta stability (Spearman vs recal): {r['secondary_theta_spearman']:.3f}."
        )
    elif r["primary_verdict"] == "PARTIAL":
        verdict = (
            f"PARTIAL. Anchored extension achieves {r['primary_difficulty_frac']:.2f}x the "
            f"seed-reliability ceiling on new-item difficulty (Spearman={r['primary_difficulty_spearman']:.3f} "
            f"vs ceiling={r['ceiling_difficulty']:.3f}), below the 0.80x pass threshold but above 0.50x. "
            f"This is a meaningful but incomplete recovery. "
            f"Discrimination: {r['primary_discrimination_spearman']:.3f} ({r['primary_discrimination_frac']:.2f}x ceiling). "
            f"Extension cost: {r['ext_only_param_ratio']:.2f}x params, {r['ext_only_time_ratio']:.2f}x time vs full recal."
        )
    else:
        verdict = (
            f"FAIL. Anchored extension achieves only {r['primary_difficulty_frac']:.2f}x the "
            f"seed-reliability ceiling on new-item difficulty (Spearman={r['primary_difficulty_spearman']:.3f} "
            f"vs ceiling={r['ceiling_difficulty']:.3f}). This falls below both the 0.80x pass "
            f"and 0.50x partial thresholds. The anchored-extension claim does not hold on real SLAM "
            f"data under these conditions. "
            f"Discrimination: {r['primary_discrimination_spearman']:.3f} ({r['primary_discrimination_frac']:.2f}x ceiling)."
        )

    lines.append("OVERALL VERDICT")
    lines.append(f"  {verdict}")
    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Redirect stdout to log file with explicit flush on every write.
    # We do NOT tee to the original stdout (task-runner pipe) because the
    # pipe becomes broken when the bash wrapper exits, causing BrokenPipeError
    # that silently stops all further output.  The log file is the only output.
    log_path = OUTPUT_DIR / "run.log"
    log_fh = open(log_path, "w", buffering=1)  # line-buffered text mode
    sys.stdout = log_fh
    tprint("\n" + "=" * 65)
    tprint("RQ2: ITEM-AGNOSTICISM ON REAL SLAM DATA")
    tprint("=" * 65)

    # Step 1: Load data
    adapter = load_slam()
    catchall_ids = get_catchall_ids(adapter)
    n_named = get_n_named(adapter)
    tprint(f"\n[Data] n_named={n_named}, catch-all ids={sorted(catchall_ids)}")
    tprint(f"[Data] Total adapter sequences: {len(adapter)}")

    # Filter to train split, drop catch-all
    full_records = filter_catchall(adapter, catchall_ids, split="train")
    tprint(f"[Data] After catch-all filter (train, >=5 events): {len(full_records)} learners")

    # Step 2: Item splits
    base_ids_rand, new_ids_rand = make_item_split(n_named, seed=ITEM_SPLIT_SEED)
    tprint(f"\n[Split/random] BASE={len(base_ids_rand)}, NEW={len(new_ids_rand)}")

    # Run PRIMARY + CEILING (random split)
    results_random = run_protocol(
        full_records, base_ids_rand, new_ids_rand, n_named,
        split_label="random"
    )

    # STRESS: difficulty split (skippable via SLAM_SKIP_STRESS=1 for fast probes)
    results_difficulty = None
    if os.environ.get("SLAM_SKIP_STRESS", "0") != "1":
        tprint("\n[Stress] Building difficulty split...")
        try:
            base_ids_diff, new_ids_diff = make_difficulty_split(n_named)
            tprint(f"[Split/difficulty] BASE={len(base_ids_diff)}, NEW={len(new_ids_diff)}")
            results_difficulty = run_protocol(
                full_records, base_ids_diff, new_ids_diff, n_named,
                split_label="difficulty"
            )
        except Exception as e:
            tprint(f"[Stress] FAILED: {e}")
            results_difficulty = None

    # Build report
    report = format_report(results_random, results_difficulty)
    tprint("\n" + report)

    # Save outputs
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    combined = {"random": results_random}
    if results_difficulty is not None:
        combined["difficulty"] = results_difficulty
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {split: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in res.items()}
             for split, res in combined.items()},
            f, indent=2,
        )
    tprint(f"\nOutputs written to {OUTPUT_DIR}/")
    tprint(f"  {report_path}")
    tprint(f"  {results_path}")


if __name__ == "__main__":
    main()
