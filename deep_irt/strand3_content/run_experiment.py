"""run_experiment.py -- Strand 3: the content channel and its cold-start payoff.

Two tests, one verdict.

TEST 1 -- SYNTHETIC MECHANISM PROOF (positive + negative control)
    Manufacture a world where content DETERMINES difficulty.  Train the content
    channel on a BASE item set, hold out items, predict their difficulty from
    CONTENT ALONE (cold-start), correlate with truth.
      POSITIVE: real content -> recovers well.
      NEGATIVE: shuffled content (content uncorrelated with difficulty) ->
                cold-start collapses to ~0.
    The gap (positive minus negative) is the proof that the channel places items
    via content, not leakage.

TEST 2 -- REAL SLAM (en_es, in hand)
    Embed each SLAM item's text.  Content cold-start on held-out items: predict
    difficulty from text alone, correlate with the item's OBSERVED human
    difficulty.  Compare against:
      (a) ID-only on the same held-out items -- CANNOT cold-start (its held-out
          rows are untrained); its correlation is the no-content floor.
      (b) WARM anchored placement -- new items placed using their RESPONSES
          (RQ2-style), the upper reference of what response data recovers.
    Honest expectation: partial (SLAM difficulty is construct-capped by
    exact-match grading).  ABOVE the negative-control / ID-only floor means the
    lever works.

VERDICT: does content alone place items materially above the negative-control
floor (synthetic clean, SLAM capped)?  How far toward the warm placement?

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export PYTHONPATH=".;rl/src;ma-irt"
    export KMP_DUPLICATE_LIB_OK=TRUE
    python -m deep_irt.strand3_content.run_experiment
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
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rl" / "src"))
sys.path.insert(0, str(REPO_ROOT / "ma-irt"))

from deep_irt.core.model import DeepIRTModel
from deep_irt.core.realdata import collate_adapter_items

from deep_irt.strand3_content.content_model import ContentModel
from deep_irt.strand3_content import synthetic as syn
from deep_irt.strand3_content.slam_content import build_bank
from deep_irt.strand3_content.slam_sequences import (
    load_adapter,
    named_global_ids,
    sequences_over_items,
    to_collate_records,
)

SLAM_RAW_DIR = REPO_ROOT / "rl" / "data" / "slam_raw"
ARTEFACT_DIR = REPO_ROOT / "rl" / "data" / "slam_artefact_mc10"
OUTPUT_DIR = Path(os.environ.get(
    "STRAND3_OUTPUT_DIR", str(Path(__file__).resolve().parent / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if (os.environ.get("STRAND3_CPU", "0") != "1"
               and torch.cuda.is_available())
    else "cpu"
)


def tprint(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


# ===========================================================================
# TEST 1 -- synthetic mechanism proof
# ===========================================================================

def _train_content_on_base(
    content_feats: np.ndarray,
    base_item_ids: np.ndarray,
    item_seq: np.ndarray,
    resp_seq: np.ndarray,
    K: int,
    emb_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    mode: str = "content",
) -> Tuple[ContentModel, np.ndarray]:
    """Train a content model on BASE-item events only.

    The content table holds ALL items' features (so held-out items can be
    cold-started later) but training events touch only base items, and the
    content table is remapped so base items occupy a contiguous 0..n_base-1
    block trained against, with held-out features carried for cold-start.
    Returned local->global map lets us index back.
    """
    # Restrict the content table + sequences to base items for training, but we
    # keep the FULL features for cold-start in the caller.  Here base items get
    # local ids 0..n_base-1.
    base_set = set(int(i) for i in base_item_ids)
    global_to_local = {int(g): j for j, g in enumerate(sorted(base_set))}
    base_global_sorted = np.array(sorted(base_set), dtype=np.int64)
    base_feats = content_feats[base_global_sorted]

    # Filter sequences to base events, remap to local.
    records = []
    for n in range(item_seq.shape[0]):
        pairs = [
            (global_to_local[int(q)], int(r))
            for q, r in zip(item_seq[n], resp_seq[n])
            if int(q) in base_set
        ]
        if len(pairs) < 2:
            continue
        lq, lresp = zip(*pairs)
        records.append({
            "student_id": n + 1,
            "questions": torch.tensor(list(lq), dtype=torch.long) + 1,  # for collate
            "responses": torch.tensor(list(lresp), dtype=torch.long),
        })
    batch = collate_adapter_items(records)

    model = ContentModel(
        text_features=torch.from_numpy(base_feats),
        emb_dim=emb_dim, hidden_dim=hidden_dim, n_cats=K,
        mode=mode, state_alpha=True, device=DEVICE, seed=seed,
    )
    model.fit(
        batch.item_ids, batch.responses, n_epochs=epochs, lr=lr,
        verbose=False, mask=batch.mask, batch_size=256,
    )
    return model, base_global_sorted


def run_synthetic(seed: int = 0) -> dict:
    tprint("\n" + "=" * 70)
    tprint("TEST 1 -- SYNTHETIC MECHANISM PROOF (content determines difficulty)")
    tprint(f"device={DEVICE}")
    tprint("=" * 70)

    data = syn.generate(
        n_items=120, n_learners=800, seq_len=40, text_dim=64, K=3, seed=seed
    )
    tprint(f"  items={data.n_items}  learners=800  K={data.K}  text_dim={data.text_dim}")

    # 70/30 base/held-out item split.
    rng = np.random.default_rng(seed + 100)
    perm = rng.permutation(data.n_items)
    cut = int(0.70 * data.n_items)
    base_ids = np.sort(perm[:cut])
    held_ids = np.sort(perm[cut:])
    tprint(f"  base items={len(base_ids)}  held-out items={len(held_ids)}")

    out: Dict[str, dict] = {}

    for control, feats in [
        ("positive", data.content),
        ("negative", syn.shuffle_content(data, seed=seed + 1)),
    ]:
        tprint(f"  [{control}] training content model on base items...")
        t0 = time.time()
        model, _ = _train_content_on_base(
            content_feats=feats,
            base_item_ids=base_ids,
            item_seq=data.item_ids,
            resp_seq=data.responses,
            K=data.K, emb_dim=16, hidden_dim=64,
            epochs=150, lr=0.02, seed=seed, mode="content",
        )
        tprint(f"    trained in {time.time() - t0:.1f}s")
        # Cold-start held-out difficulty from text alone.
        held_feats = torch.from_numpy(feats[held_ids])
        cold = model.cold_start_difficulty(held_feats)        # (n_held,)
        b_true_held = data.b_true[held_ids]

        sp = spearman(cold, b_true_held)
        pe = pearson(cold, b_true_held)
        # Also the in-bank (base) recovery as a sanity ceiling.
        base_cold = model.cold_start_difficulty(
            torch.from_numpy(feats[base_ids])
        )
        base_sp = spearman(base_cold, data.b_true[base_ids])

        out[control] = {
            "coldstart_spearman_held": sp,
            "coldstart_pearson_held": pe,
            "base_spearman": base_sp,
            "n_held": int(len(held_ids)),
        }
        tprint(f"  [{control:8s}] cold-start held-out Spearman={sp:+.3f}  "
               f"Pearson={pe:+.3f}  (base recov Spearman={base_sp:+.3f})")

    gap = out["positive"]["coldstart_spearman_held"] - out["negative"]["coldstart_spearman_held"]
    out["gap_pos_minus_neg"] = gap
    tprint(f"  GAP (positive - negative) = {gap:+.3f}")
    return out


# ===========================================================================
# TEST 2 -- real SLAM content cold-start
# ===========================================================================

def _build_id_only_records(records_local, n_items):
    """collate records for an ID-only / content model sharing the same local
    id space (records already 0-based local)."""
    return collate_adapter_items(to_collate_records(records_local))


def run_slam(seed: int = 0, prefer_st: bool = True) -> dict:
    tprint("\n" + "=" * 70)
    tprint("TEST 2 -- REAL SLAM en_es content cold-start")
    tprint("=" * 70)

    adapter = load_adapter(SLAM_RAW_DIR, ARTEFACT_DIR)
    named = named_global_ids(adapter)
    tprint(f"  named items (mc10): {len(named)}")

    # Content bank for all named items that have text + observed difficulty.
    bank = build_bank(REPO_ROOT, ARTEFACT_DIR, keep_global_ids=named,
                      prefer_st=prefer_st)
    tprint(f"  items with text + difficulty: {len(bank.global_ids)}  "
           f"features={bank.feat_kind}  dim={bank.text_features.shape[1]}")

    # Local id space = bank rows.  Load sequences over exactly these items.
    global_to_local = {int(g): int(l) for g, l in zip(bank.global_ids, bank.local_ids)}
    records_local = sequences_over_items(
        adapter, bank.global_ids, global_to_local, min_events=3
    )
    tprint(f"  learners (>=3 named events): {len(records_local)}")

    M = len(bank.global_ids)
    K = 3
    EMB_DIM, HIDDEN_DIM = 16, 64
    EPOCHS, LR = 100, 0.02

    # 70/30 base/held-out item split (random, seeded).
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(M)
    cut = int(0.70 * M)
    base_local = np.sort(perm[:cut])
    held_local = np.sort(perm[cut:])
    base_set = set(int(i) for i in base_local)
    held_set = set(int(i) for i in held_local)
    tprint(f"  base items={len(base_local)}  held-out items={len(held_local)}")

    obs_held = bank.obs_difficulty[held_local]
    obs_base = bank.obs_difficulty[base_local]

    # --- Split each learner's sequence into base-only and held-only events ---
    def restrict(records, keep_set, remap):
        out = []
        for rec in records:
            qs = rec["questions"].tolist()
            rs = rec["responses"].tolist()
            pairs = [(remap[q], r) for q, r in zip(qs, rs) if q in keep_set]
            if len(pairs) < 2:
                continue
            lq, lr = zip(*pairs)
            out.append({
                "student_id": rec["student_id"],
                "questions": torch.tensor(list(lq), dtype=torch.long),
                "responses": torch.tensor(list(lr), dtype=torch.long),
            })
        return out

    base_remap = {int(g): j for j, g in enumerate(base_local)}
    held_remap = {int(g): j for j, g in enumerate(held_local)}
    base_records = restrict(records_local, base_set, base_remap)
    held_records = restrict(records_local, held_set, held_remap)
    tprint(f"  base-event learners={len(base_records)}  "
           f"held-event learners={len(held_records)}")

    base_feats = bank.text_features[base_local]
    held_feats = bank.text_features[held_local]

    results: Dict[str, object] = {
        "n_items": M, "n_base": int(len(base_local)),
        "n_held": int(len(held_local)), "feat_kind": bank.feat_kind,
    }

    # ----- (1) CONTENT channel: train on base, cold-start held-out -----
    tprint("\n  [content] training on base items, cold-starting held-out...")
    cm = ContentModel(
        text_features=torch.from_numpy(base_feats),
        emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, n_cats=K,
        mode="content", state_alpha=True, device=DEVICE, seed=seed,
    )
    cb = collate_adapter_items(to_collate_records(base_records))
    cm.fit(cb.item_ids, cb.responses, n_epochs=EPOCHS, lr=LR,
           verbose=False, mask=cb.mask, batch_size=256)
    cold = cm.cold_start_difficulty(torch.from_numpy(held_feats))
    content_sp = spearman(cold, obs_held)
    content_pe = pearson(cold, obs_held)
    # in-bank base recovery for reference
    base_recov = cm.recover_difficulty()
    content_base_sp = spearman(base_recov, obs_base)
    results["content_coldstart_spearman"] = content_sp
    results["content_coldstart_pearson"] = content_pe
    results["content_base_spearman"] = content_base_sp
    tprint(f"    cold-start Spearman={content_sp:+.3f}  Pearson={content_pe:+.3f}  "
           f"(in-bank base Spearman={content_base_sp:+.3f})")

    # ----- (2) ID-only channel: CANNOT cold-start (held rows untrained) -----
    tprint("\n  [id-only] training on base items, 'cold-start' held-out "
           "(untrained rows -> floor)...")
    # Build an ID-only model whose item table covers ALL M items but train only
    # base events; held-out rows stay at init -> their difficulty is noise.
    full_records = restrict(
        records_local, base_set, base_remap
    )  # base events, base-local ids
    idm = DeepIRTModel(
        num_items=M, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, n_cats=K,
        decoder="gpcm", state_alpha=True, device=DEVICE, seed=seed,
    )
    # Train on base events but using FULL-M item ids so held rows exist.
    # Remap base events to their ORIGINAL local ids in [0, M).
    base_full = []
    for rec in records_local:
        qs = rec["questions"].tolist()
        rs = rec["responses"].tolist()
        pairs = [(q, r) for q, r in zip(qs, rs) if q in base_set]
        if len(pairs) < 2:
            continue
        lq, lr = zip(*pairs)
        base_full.append({
            "student_id": rec["student_id"],
            "questions": torch.tensor(list(lq), dtype=torch.long) + 1,
            "responses": torch.tensor(list(lr), dtype=torch.long),
        })
    ib = collate_adapter_items(base_full)
    idm.fit(ib.item_ids, ib.responses, n_epochs=EPOCHS, lr=LR,
            verbose=False, mask=ib.mask, batch_size=256)
    # "Cold-start" the held-out items = read their UNTRAINED rows.
    with torch.no_grad():
        held_ids_t = torch.tensor(held_local, dtype=torch.long, device=DEVICE)
        held_embs = idm.encoder.item_emb(held_ids_t)
        b_held = idm.decoder.item_params_sorted(held_embs)["b"]
        id_cold = b_held.mean(dim=-1).cpu().numpy()
    id_sp = spearman(id_cold, obs_held)
    results["idonly_coldstart_spearman"] = id_sp
    tprint(f"    ID-only held-out Spearman={id_sp:+.3f}  (no-content floor)")

    # ----- (3) WARM anchored placement (RQ2-style): uses RESPONSES -----
    tprint("\n  [warm-anchored] placing held-out items from their RESPONSES "
           "(upper reference)...")
    try:
        warm = _warm_anchored_difficulty(
            base_records=base_records, held_records=held_records,
            n_base=len(base_local), n_held=len(held_local),
            K=K, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
            epochs=EPOCHS, lr=LR, seed=seed, device=DEVICE,
        )
    except torch.cuda.OutOfMemoryError:
        tprint("    CUDA OOM on warm baseline; retrying on CPU...")
        torch.cuda.empty_cache()
        warm = _warm_anchored_difficulty(
            base_records=base_records, held_records=held_records,
            n_base=len(base_local), n_held=len(held_local),
            K=K, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
            epochs=EPOCHS, lr=LR, seed=seed, device=torch.device("cpu"),
        )
    warm_sp = spearman(warm, obs_held)
    results["warm_anchored_spearman"] = warm_sp
    tprint(f"    warm-anchored held-out Spearman={warm_sp:+.3f}")

    # ----- fractions -----
    if not np.isnan(warm_sp) and abs(warm_sp) > 1e-6:
        frac = content_sp / warm_sp
        results["content_over_warm_fraction"] = float(frac)
        tprint(f"\n  content cold-start / warm placement = {frac:+.2f}x")
    results["content_minus_idonly"] = content_sp - id_sp
    tprint(f"  content - ID-only floor = {content_sp - id_sp:+.3f}")
    return results


def _warm_anchored_difficulty(
    base_records, held_records, n_base, n_held, K, emb_dim, hidden_dim,
    epochs, lr, seed, device,
) -> np.ndarray:
    """RQ2-style anchored placement of held-out items from responses.

    Train a base ID model on base events; pin each learner's final theta from
    base; freeze encoder+decoder; fit free ID rows for the held-out items
    against the held-out responses; read mean-b difficulty.  This is the
    response-based upper reference (mirrors slam_extend).
    """
    import torch.optim as optim
    from deep_irt.core.anchor import _ExtEmbeddings
    base_model = DeepIRTModel(
        num_items=n_base, emb_dim=emb_dim, hidden_dim=hidden_dim, n_cats=K,
        decoder="gpcm", state_alpha=True, device=device, seed=seed,
    )
    cb = collate_adapter_items(to_collate_records(base_records))
    base_model.fit(cb.item_ids, cb.responses, n_epochs=epochs, lr=lr,
                   verbose=False, mask=cb.mask, batch_size=256)

    # Per-learner final theta from base (direct stream in dual mode).  Batched
    # over learners so the LSTM forward fits in GPU memory (full-batch track
    # over ~7700 learners x long T_max overflows an 8 GB card).
    theta_final_chunks = []
    bs = 256
    N_all = cb.item_ids.size(0)
    for start in range(0, N_all, bs):
        ii = cb.item_ids[start:start + bs]
        rr = cb.responses[start:start + bs]
        sl = cb.seq_lens[start:start + bs]
        tt = base_model.track(ii, rr)                          # (b, T)
        theta_final_chunks.append(
            tt[torch.arange(len(sl)), sl - 1].detach().cpu()
        )
    theta_final = torch.cat(theta_final_chunks)
    sid_to_theta = {int(s): float(t) for s, t in zip(cb.student_ids.tolist(),
                                                     theta_final.tolist())}

    # Align held-out learners to a pinned theta (those present in base).
    hb = collate_adapter_items(to_collate_records(held_records))
    thetas = []
    keep_rows = []
    for r, sid in enumerate(hb.student_ids.tolist()):
        if int(sid) in sid_to_theta:
            thetas.append(sid_to_theta[int(sid)])
            keep_rows.append(r)
    if not keep_rows:
        return np.full(n_held, np.nan)
    keep_rows = torch.tensor(keep_rows, dtype=torch.long)
    theta_dev = torch.tensor(thetas, dtype=torch.float32).to(device)
    item_ids = hb.item_ids[keep_rows].to(device)
    responses = hb.responses[keep_rows].to(device)
    mask = hb.mask[keep_rows].to(device)

    encoder, decoder = base_model.encoder, base_model.decoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    decoder.eval()

    ext = _ExtEmbeddings(n_ext=n_held, emb_dim=emb_dim).to(device)
    opt = optim.Adam(ext.parameters(), lr=lr)
    N_held_learn = item_ids.size(0)
    bs = 512
    for _ in range(epochs):
        ext.train()
        # Mini-batch over learners: only the trainable ext embeddings are in the
        # graph (encoder/decoder frozen), so each chunk's gradient accumulates
        # the same loss the full batch would, at a fraction of the memory.
        opt.zero_grad()
        for start in range(0, N_held_learn, bs):
            ii = item_ids[start:start + bs]
            rr = responses[start:start + bs]
            mm = mask[start:start + bs]
            embs = ext(ii)                            # (b, T, emb)
            b_, T, D = embs.shape
            th = theta_dev[start:start + bs].unsqueeze(1).expand(b_, T)
            ef = embs.reshape(b_ * T, D)
            tf = th.reshape(b_ * T)
            rf = rr.reshape(b_ * T)
            mf = mm.reshape(b_ * T)
            # Warm placement uses the static item-only alpha head (state=None),
            # exactly the anchored-extension recipe in slam_extend.
            loss = decoder.nll(tf[mf], ef[mf], rf[mf])
            loss.backward()
        opt.step()

    ext.eval()
    with torch.no_grad():
        b = decoder.item_params_sorted(ext.ext_emb.weight)["b"]
        return b.mean(dim=-1).cpu().numpy()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)

    syn_res = run_synthetic(seed=0)
    slam_res = run_slam(seed=0, prefer_st=True)

    # ----- verdict -----
    tprint("\n" + "=" * 70)
    tprint("VERDICT -- content channel as the foundation-model lever")
    tprint("=" * 70)
    pos = syn_res["positive"]["coldstart_spearman_held"]
    neg = syn_res["negative"]["coldstart_spearman_held"]
    tprint(f"  SYNTHETIC  positive cold-start = {pos:+.3f}  "
           f"negative = {neg:+.3f}  gap = {syn_res['gap_pos_minus_neg']:+.3f}")
    c_sp = slam_res["content_coldstart_spearman"]
    i_sp = slam_res["idonly_coldstart_spearman"]
    w_sp = slam_res["warm_anchored_spearman"]
    tprint(f"  REAL SLAM  content cold-start = {c_sp:+.3f}  "
           f"ID-only floor = {i_sp:+.3f}  warm-anchored = {w_sp:+.3f}")
    if "content_over_warm_fraction" in slam_res:
        tprint(f"             content / warm = "
               f"{slam_res['content_over_warm_fraction']:+.2f}x")

    syn_pass = syn_res["gap_pos_minus_neg"] > 0.3 and pos > 0.5
    slam_pass = (c_sp - i_sp) > 0.05 and c_sp > 0.05
    tprint(f"\n  synthetic mechanism proof: {'PASS' if syn_pass else 'FAIL'}")
    tprint(f"  real SLAM above floor:     {'PASS' if slam_pass else 'FAIL'}")
    go = syn_pass and slam_pass
    tprint(f"\n  GO/NO-GO: {'GO' if go else 'NO-GO'} for the content channel "
           f"as the foundation-model lever.")

    payload = {
        "synthetic": syn_res,
        "slam": slam_res,
        "verdict": {
            "synthetic_pass": bool(syn_pass),
            "slam_pass": bool(slam_pass),
            "go": bool(go),
        },
        "wall_clock_s": time.time() - t0,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    tprint(f"\n  results -> {OUTPUT_DIR / 'results.json'}  "
           f"(t={payload['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
