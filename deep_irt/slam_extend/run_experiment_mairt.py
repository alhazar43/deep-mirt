"""
run_experiment_mairt.py -- RQ2 anchored extension on real SLAM, ma-irt engine.

Same protocol and DATA LAYER as run_experiment.py (the deep_irt version),
but the engine is ma-irt's LSTM + GPCM decoder with separate_theta=True (the
audited-correct ability pathway). The motivating question:

  The deep_irt RQ2 found new-item DIFFICULTY transfers under anchored
  extension (~0.76x ceiling, PARTIAL) but DISCRIMINATION does NOT (~0.46x).
  A corrected benchmark traced the discrimination gap to the DECODER: ma-irt's
  GPCM decoder recovers alpha at ~0.93 Spearman vs the deep_irt's ~0.71.
  So: does running the SAME anchored protocol on ma-irt lift the new-item
  discrimination fraction toward a PASS?

ma-irt anchoring analog (decoder-direct, the structural twin of the deep_irt)
------------------------------------------------------------------------------
The deep_irt anchors by pinning per-learner theta (a scalar from the base-only
fit) and fitting only the new-item embedding through the FROZEN decoder, whose
alpha/beta are pure functions of that embedding. ma-irt's decoder differs: beta
is item-only (threshold(item_embed)) but alpha is STATE-CONDITIONED,
alpha = exp(disc_net([joint_summary, item_embed])). So a new item's alpha needs
a per-learner summary, not just the new embedding.

We therefore pin, per learner, the BASE-ONLY final-step summaries:
  theta_L  = ability_network(student_summary_final)     # item-blind ability
  s_L      = joint_summary_final                         # state for alpha
Then we fit only a fresh new-item KEY embedding table (the q_embed analog),
feeding the FROZEN decoder sub-networks:
  beta  = threshold(new_emb)
  alpha = exp(disc_net([s_L, new_emb]))
  theta = theta_L
GPCM logits -> cross-entropy at new-item positions only -> grad flows ONLY into
the new-item embedding. Everything that maps embedding->params (the encoder, the
IRTParameterExtractor, the base embeddings) is frozen. This is the faithful
analog of deep_irt anchoring and never re-runs the full interleaved sequence
(so the value-side table cannot leak into theta).

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="<repo-root>;rl/src;ma-irt"
    python deep_irt/slam_extend/run_experiment_mairt.py
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup: deep_irt (parent of core/), rl/src, and ma-irt importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Reuse the deep_irt RQ2 data layer verbatim.
from deep_irt.slam_extend.run_experiment import (
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
)
from deep_irt.core.realdata import collate_adapter_items

from models.lstm_gpcm import LSTMGPCM
from models.components.irt import alpha_from_raw

# ---------------------------------------------------------------------------
# Constants (bounded run knobs via env)
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get(
    "MAIRT_OUTPUT_DIR", str(Path(__file__).resolve().parent / "outputs_mairt")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 3
KEY_DIM = int(os.environ.get("MAIRT_KEY_DIM", "32"))     # q_embed / item-key width
D_MODEL = int(os.environ.get("MAIRT_D_MODEL", "32"))     # LSTM hidden
VALUE_DIM = int(os.environ.get("MAIRT_VALUE_DIM", "32"))
BASE_EPOCHS = int(os.environ.get("MAIRT_BASE_EPOCHS", "60"))
EXT_EPOCHS = int(os.environ.get("MAIRT_EXT_EPOCHS", "200"))
RECAL_EPOCHS = int(os.environ.get("MAIRT_RECAL_EPOCHS", "60"))
LR = float(os.environ.get("MAIRT_LR", "0.01"))
EXT_LR = float(os.environ.get("MAIRT_EXT_LR", "0.02"))
SUBSAMPLE = int(os.environ.get("MAIRT_SUBSAMPLE", "1000"))  # 0 = all learners
RECAL_SEEDS = (0, 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model build + train helpers
# ---------------------------------------------------------------------------

def build_model(n_questions: int, seed: int) -> LSTMGPCM:
    torch.manual_seed(seed)
    m = LSTMGPCM(
        n_questions=n_questions,
        n_categories=K,
        d_model=D_MODEL,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        dropout_rate=0.0,
        separate_theta=True,
    )
    return m.to(DEVICE)


BATCH = int(os.environ.get("MAIRT_BATCH", "256"))


def _q1_batch(records_local: List[Dict]):
    """Collate local-0based records into 1-indexed CPU tensors.

    Returns (q (N,S) 1-based pad=0, r (N,S), mask (N,S), seq_lens (N,), sids).
    Tensors stay on CPU; minibatches are moved to DEVICE by the callers.
    """
    b = collate_adapter_items([
        {"student_id": r["student_id"],
         "questions": r["questions"] + 1,  # 0-based local -> 1-based for collate
         "responses": r["responses"]}
        for r in records_local
    ])
    # collate subtracts 1 -> back to 0-based; ma-irt wants 1-based -> add 1,
    # padded positions become id 0 (the padding_idx).
    q = torch.where(b.mask, b.item_ids + 1, torch.zeros_like(b.item_ids))
    return q, b.responses, b.mask, b.seq_lens, b.student_ids.tolist()


def train_full(records_local: List[Dict], n_questions: int, seed: int,
               n_epochs: int, label: str) -> Tuple[LSTMGPCM, dict]:
    """Train ma-irt to minimise masked cross-entropy over all valid positions.

    Minibatched (BATCH learners/step) so memory stays bounded on the 8 GB GPU.
    """
    q, r, mask, seq_lens, sids = _q1_batch(records_local)
    N = q.size(0)
    model = build_model(n_questions, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 1)
    tprint(f"\n[{label}] {N} learners, {n_questions} items, {n_epochs} epochs, "
           f"batch={BATCH}, device={DEVICE}...")
    t0 = time.time()
    model.train()
    final = float("nan")
    for ep in range(1, n_epochs + 1):
        perm = torch.randperm(N, generator=gen)
        tot_loss = 0.0; tot_tok = 0
        for s in range(0, N, BATCH):
            bi = perm[s:s + BATCH]
            qb = q[bi].to(DEVICE); rb = r[bi].to(DEVICE); mb = mask[bi].to(DEVICE)
            opt.zero_grad()
            logits = model(qb, rb)["logits"]            # (B,S,K)
            B, S, Kk = logits.shape
            ce = F.cross_entropy(logits.reshape(B * S, Kk),
                                 rb.reshape(B * S), reduction="none")
            mflat = mb.reshape(B * S).float()
            loss = (ce * mflat).sum() / mflat.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.item()) * int(mflat.sum().item())
            tot_tok += int(mflat.sum().item())
        final = tot_loss / max(tot_tok, 1)
        if ep % 20 == 0 or ep == 1:
            tprint(f"  [{label}] epoch {ep:4d}  NLL={final:.4f}")
    wall = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    tprint(f"  [{label}] done. NLL={final:.4f}, params={n_params:,}, t={wall:.1f}s")
    return model, {"final_nll": final, "wall_clock": wall, "n_params": n_params,
                   "student_ids": sids}


# ---------------------------------------------------------------------------
# Pin per-learner base-only summaries (theta + alpha-state)
# ---------------------------------------------------------------------------

@torch.no_grad()
def pin_learner_states(base_model: LSTMGPCM,
                       base_records_local: List[Dict]
                       ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Return (theta_pinned (N,D), joint_pinned (N,d_model), student_ids).

    theta_pinned   = ability_network(student_summary at final valid step).
    joint_pinned   = joint_summary at final valid step (alpha conditioning state).
    Both from BASE-ONLY history -- the structural twin of deep_irt's pinned theta.
    """
    base_model.eval()
    q, r, mask, seq_lens, sids = _q1_batch(base_records_local)
    N = q.size(0)
    irt = base_model.decoder.irt
    student_finals = []; joint_finals = []
    for s in range(0, N, BATCH):
        sl = slice(s, s + BATCH)
        qb = q[sl].to(DEVICE); rb = r[sl].to(DEVICE)
        enc = base_model.encoder(qb, rb)
        idx = (seq_lens[sl].to(DEVICE) - 1).clamp(min=0)
        rows = torch.arange(qb.size(0), device=DEVICE)
        student_finals.append(enc.student_summary[rows, idx].cpu())
        joint_finals.append(enc.joint_summary[rows, idx].cpu())
    student_final = torch.cat(student_finals, dim=0).to(DEVICE)  # (N,d_model)
    joint_final = torch.cat(joint_finals, dim=0).to(DEVICE)      # (N,d_model)
    theta_pinned = irt.ability_network(student_final) * irt.ability_scale  # (N,D)
    return theta_pinned.detach(), joint_final.detach(), sids


# ---------------------------------------------------------------------------
# Anchored extension: fit ONLY new-item KEY embeddings through frozen decoder
# ---------------------------------------------------------------------------

def run_anchored_extension(base_model: LSTMGPCM,
                           theta_pinned: torch.Tensor,
                           joint_pinned: torch.Tensor,
                           new_records_local: List[Dict],
                           n_new: int) -> dict:
    """Fit a fresh (n_new, key_dim) embedding table against new-item responses,
    with the decoder frozen and per-learner state pinned. Grad flows ONLY into
    the new embedding rows.
    """
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    irt = base_model.decoder.irt
    gpcm_logits = base_model.decoder.gpcm_logits
    log_scale = base_model.decoder.alpha_log_scale

    # Collate new-item positions (0-based local ids -> keep as local index).
    b = collate_adapter_items([
        {"student_id": r["student_id"],
         "questions": r["questions"] + 1,
         "responses": r["responses"]}
        for r in new_records_local
    ])
    new_local_ids = b.item_ids               # (N,S) CPU, 0-based local new-item ids
    resp = b.responses                       # (N,S) CPU
    mask = b.mask                            # (N,S) CPU
    N, S = new_local_ids.shape
    D = theta_pinned.size(-1)

    theta_pinned = theta_pinned.cpu()        # (N,D)
    joint_pinned = joint_pinned.cpu()        # (N,d_model)

    # Fresh trainable new-item key embeddings (q_embed analog).
    ext_emb = nn.Embedding(n_new, KEY_DIM).to(DEVICE)
    nn.init.kaiming_normal_(ext_emb.weight)
    opt = torch.optim.Adam(ext_emb.parameters(), lr=EXT_LR)
    n_params = sum(p.numel() for p in ext_emb.parameters())
    gen = torch.Generator().manual_seed(123)

    tprint(f"\n[Anchored Ext] {N} learners, {n_new} new items, {EXT_EPOCHS} epochs, "
           f"batch={BATCH}, trainable params={n_params:,}...")
    t0 = time.time()
    final = float("nan")
    for ep in range(1, EXT_EPOCHS + 1):
        ext_emb.train()
        perm = torch.randperm(N, generator=gen)
        tot_loss = 0.0; tot_tok = 0
        for s in range(0, N, BATCH):
            bi = perm[s:s + BATCH]
            ids_b = new_local_ids[bi].to(DEVICE)            # (b,S)
            resp_b = resp[bi].to(DEVICE)
            mask_b = mask[bi].to(DEVICE)
            theta_b = theta_pinned[bi].to(DEVICE).unsqueeze(1).expand(-1, S, D)
            joint_b = joint_pinned[bi].to(DEVICE).unsqueeze(1).expand(-1, S, joint_pinned.size(-1))
            opt.zero_grad()
            emb = ext_emb(ids_b)                            # (b,S,key_dim)
            beta = irt.threshold(emb)                       # (b,S,K-1)
            raw_alpha = irt.discrimination_network(
                torch.cat([joint_b, emb], dim=-1))          # (b,S,D)
            alpha = alpha_from_raw(raw_alpha, log_scale)    # (b,S,D)
            logits = gpcm_logits(theta_b, alpha, beta)      # (b,S,K)
            B, T, Kk = logits.shape
            ce = F.cross_entropy(logits.reshape(B * T, Kk),
                                 resp_b.reshape(B * T), reduction="none")
            mflat = mask_b.reshape(B * T).float()
            loss = (ce * mflat).sum() / mflat.sum().clamp(min=1)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * int(mflat.sum().item())
            tot_tok += int(mflat.sum().item())
        final = tot_loss / max(tot_tok, 1)
        if ep % 50 == 0 or ep == 1:
            tprint(f"  [AnchorExt] epoch {ep:4d}  NLL={final:.4f}")
    wall = time.time() - t0

    # Recover anchored new-item params.
    # beta: item-only -> one value per new item from its embedding.
    # alpha: state-conditioned -> average over learners who saw the item, using
    #        that learner's pinned joint state (mirrors "average over occurrences").
    ext_emb.eval()
    a_sum = np.zeros(n_new); a_cnt = np.zeros(n_new)
    with torch.no_grad():
        all_emb = ext_emb.weight                            # (n_new, key_dim)
        beta_item = irt.threshold(all_emb).cpu().numpy()    # (n_new, K-1)
        for s in range(0, N, BATCH):
            sl = slice(s, s + BATCH)
            ids_b = new_local_ids[sl].to(DEVICE)
            mask_b = mask[sl].to(DEVICE)
            joint_b = joint_pinned[sl].to(DEVICE).unsqueeze(1).expand(-1, S, joint_pinned.size(-1))
            emb = ext_emb(ids_b)
            raw_a = irt.discrimination_network(torch.cat([joint_b, emb], dim=-1))
            a_seq = alpha_from_raw(raw_a, log_scale).squeeze(-1)  # (b,S)
            a_np = a_seq.cpu().numpy(); m_np = mask_b.cpu().numpy()
            ids_np = ids_b.cpu().numpy()
            np.add.at(a_sum, ids_np[m_np], a_np[m_np])
            np.add.at(a_cnt, ids_np[m_np], 1.0)
        alpha_item = a_sum / np.maximum(a_cnt, 1.0)         # (n_new,)

    tprint(f"  [AnchorExt] done. NLL={final:.4f}, t={wall:.1f}s, "
           f"items covered={int((a_cnt > 0).sum())}/{n_new}")
    return {"a": alpha_item, "b": beta_item, "seen": a_cnt > 0,
            "n_params": n_params, "train_time": wall, "final_nll": final}


# ---------------------------------------------------------------------------
# Full-recal recovery (bench convention: average alpha over occurrences)
# ---------------------------------------------------------------------------

@torch.no_grad()
def recover_full(model: LSTMGPCM, records_local: List[Dict],
                 n_questions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full model over all sequences; per-item alpha = mean over all
    (learner, step) occurrences, beta = mean over occurrences. Returns
    (a (Q,), b (Q,K-1), seen (Q,)). Item ids here are 0-based local."""
    model.eval()
    q, r, mask, seq_lens, sids = _q1_batch(records_local)
    N = q.size(0)
    a_sum = np.zeros(n_questions); a_cnt = np.zeros(n_questions)
    b_sum = np.zeros((n_questions, K - 1)); b_cnt = np.zeros(n_questions)
    for s in range(0, N, BATCH):
        sl = slice(s, s + BATCH)
        qb = q[sl].to(DEVICE); rb = r[sl].to(DEVICE); mb = mask[sl]
        out = model(qb, rb)
        alpha = out["alpha"].squeeze(-1).cpu().numpy()      # (b,S)
        beta = out["beta"].cpu().numpy()                    # (b,S,K-1)
        ids0 = (qb.cpu().numpy() - 1)                       # (b,S) 0-based; pad=-1
        m_np = mb.numpy()
        fi = ids0[m_np]; fa = alpha[m_np]; fb = beta[m_np]
        np.add.at(a_sum, fi, fa)
        np.add.at(a_cnt, fi, 1.0)
        np.add.at(b_sum, fi, fb)
        np.add.at(b_cnt, fi, 1.0)
    a = a_sum / np.maximum(a_cnt, 1.0)
    b = b_sum / np.maximum(b_cnt[:, None], 1.0)
    return a, b, (a_cnt > 0)


# ---------------------------------------------------------------------------
# Core protocol (mirrors run_experiment.run_protocol, ma-irt engine)
# ---------------------------------------------------------------------------

def run_protocol(full_records: List[Dict], base_ids: np.ndarray,
                 new_ids: np.ndarray, n_named: int, split_label: str) -> dict:
    base_set = set(base_ids.tolist())
    new_set = set(new_ids.tolist())
    n_base = len(base_ids); n_new = len(new_ids)

    tprint(f"\n{'='*65}\nPROTOCOL [ma-irt]: {split_label.upper()} SPLIT")
    tprint(f"  Named items: {n_named}  BASE: {n_base}  NEW: {n_new}\n{'='*65}")

    base_seqs = subsequence(full_records, base_set, min_events=2)
    new_seqs = subsequence(full_records, new_set, min_events=2)
    base_sid = {r["student_id"] for r in base_seqs}
    new_sid = {r["student_id"] for r in new_seqs}
    shared_sid = base_sid & new_sid
    tprint(f"  Shared learners (base & new): {len(shared_sid)}")

    base_seqs_shared = sorted([r for r in base_seqs if r["student_id"] in shared_sid],
                              key=lambda r: r["student_id"])
    new_seqs_shared = sorted([r for r in new_seqs if r["student_id"] in shared_sid],
                             key=lambda r: r["student_id"])

    base_local = remap_ids_to_local(base_seqs_shared, base_ids)
    new_local = remap_ids_to_local(new_seqs_shared, new_ids)

    all_named_ids = np.sort(np.concatenate([base_ids, new_ids]))
    all_seqs = subsequence(full_records, set(all_named_ids.tolist()), min_events=2)
    all_local = remap_ids_to_local(all_seqs, all_named_ids)
    all_g2l = {int(g): i for i, g in enumerate(all_named_ids)}
    new_local_in_all = np.array([all_g2l[int(g)] for g in new_ids], dtype=np.int64)

    # --- BASE FIT ---
    base_model, base_info = train_full(base_local, n_base, seed=0,
                                       n_epochs=BASE_EPOCHS, label="BaseFit")
    theta_pinned, joint_pinned, _ = pin_learner_states(base_model, base_local)

    # --- ANCHORED EXTENSION ---
    anc = run_anchored_extension(base_model, theta_pinned, joint_pinned,
                                 new_local, n_new)
    a_anc = anc["a"]; b_anc_mean = mean_b(anc["b"])

    # --- FULL RECAL x2 SEEDS ---
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
    seen = anc["seen"] & recal_new[0]["seen"] & recal_new[1]["seen"]
    tprint(f"\n[Coverage] new items scored (seen in all fits): {int(seen.sum())}/{n_new}")
    ceil_diff = spearman(recal_new[0]["b_mean"][seen], recal_new[1]["b_mean"][seen])
    ceil_disc = spearman(recal_new[0]["a"][seen], recal_new[1]["a"][seen])
    tprint(f"[Ceiling] difficulty={ceil_diff:.4f}, discrimination={ceil_disc:.4f}")

    # --- PRIMARY: anchored vs full-recal(seed0) ---
    prim_diff = spearman(b_anc_mean[seen], recal_new[0]["b_mean"][seen])
    prim_disc = spearman(a_anc[seen], recal_new[0]["a"][seen])
    f_diff = prim_diff / ceil_diff if abs(ceil_diff) > 1e-6 else float("nan")
    f_disc = prim_disc / ceil_disc if abs(ceil_disc) > 1e-6 else float("nan")
    tprint(f"\n[PRIMARY] Anchored vs FullRecal(seed=0)")
    tprint(f"  Difficulty Spearman:     {prim_diff:.4f}  (ceiling {ceil_diff:.4f}, frac {f_diff:.3f})")
    tprint(f"  Discrimination Spearman: {prim_disc:.4f}  (ceiling {ceil_disc:.4f}, frac {f_disc:.3f})")

    return {
        "split_label": split_label, "n_named": int(n_named),
        "n_base": int(n_base), "n_new": int(n_new),
        "n_shared_learners": int(len(shared_sid)),
        "n_new_scored": int(seen.sum()),
        "ceiling_difficulty": ceil_diff, "ceiling_discrimination": ceil_disc,
        "primary_difficulty_spearman": prim_diff,
        "primary_discrimination_spearman": prim_disc,
        "primary_difficulty_frac": f_diff,
        "primary_discrimination_frac": f_disc,
        "ext_only_params": anc["n_params"], "ext_only_time": anc["train_time"],
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
    tprint("RQ2 [ma-irt engine]: ITEM-AGNOSTICISM ON REAL SLAM DATA")
    tprint("=" * 65)
    tprint(f"Config: KEY_DIM={KEY_DIM} D_MODEL={D_MODEL} VALUE_DIM={VALUE_DIM} "
           f"BASE_EP={BASE_EPOCHS} EXT_EP={EXT_EPOCHS} RECAL_EP={RECAL_EPOCHS} "
           f"LR={LR} EXT_LR={EXT_LR} SUBSAMPLE={SUBSAMPLE} DEVICE={DEVICE}")

    adapter = load_slam()
    catchall_ids = get_catchall_ids(adapter)
    n_named = get_n_named(adapter)
    full_records = filter_catchall(adapter, catchall_ids, split="train")
    tprint(f"[Data] n_named={n_named}, learners(train,>=5)={len(full_records)}")

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

    # Console-style report.
    tprint("\n" + "=" * 65)
    tprint("SUMMARY (ma-irt anchored extension, new-item recovery)")
    tprint("=" * 65)
    for s, r in combined.items():
        tprint(f"\n[{s.upper()}] scored {r['n_new_scored']}/{r['n_new']} new items, "
               f"{r['n_shared_learners']} shared learners")
        tprint(f"  ceiling   diff={r['ceiling_difficulty']:.3f}  disc={r['ceiling_discrimination']:.3f}")
        tprint(f"  anchored  diff={r['primary_difficulty_spearman']:.3f}  disc={r['primary_discrimination_spearman']:.3f}")
        tprint(f"  fraction  diff={r['primary_difficulty_frac']:.3f}  disc={r['primary_discrimination_frac']:.3f}")
    tprint(f"\nOutputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
