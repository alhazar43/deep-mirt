"""run_experiment.py -- REAL-DATA RQ1: does ONE shared ma-irt item scale unify
BINARY (2PL) and NOMINAL (NRM) responses on EdNet, via held-out-format transfer?

The encoder / sequential, real-data analogue of deep_irt/nrmfmt (which proved
the binary-vs-NRM pair SYNTHETICALLY with STATIC per-respondent theta).  Here
theta is per-step, produced by the ma-irt LSTM encoder (separate_theta=True),
trained on interleaved real EdNet sequences.

There is NO synthetic ground truth.  The reference is a FULL-INFORMATION fit
(a JointModel where every item is BOTH, so its d_i uses both views) -- the best
location estimate the data allows.  Cross-format transfer:

    NRM-ONLY items (NRM head shaped their d_i):  read d_i, compare to full-info.
    BINARY-ONLY items (2PL head shaped their d_i): read d_i, compare to full-info.

Independent baselines (separate encoder + embedding per format) place held-out-
format items at NOISE -- the noise floor.  Transfer materially above the floor =
the shared scale genuinely unifies the two formats.

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="rl/src;ma-irt"
    python deep_irt/rq1_ednet_nrm/run_experiment.py
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
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from deep_irt.rq1_ednet_nrm.data import load_ednet, EdNetData
from deep_irt.rq1_ednet_nrm.model import (
    JointModel, IndepBinary, IndepNRM,
)

OUTPUT_DIR = Path(os.environ.get("RQ1_OUTPUT_DIR", str(HERE / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- bounded run knobs (env-overridable) ----
N_LEARNERS = int(os.environ.get("RQ1_N_LEARNERS", "3000"))
EPOCHS = int(os.environ.get("RQ1_EPOCHS", "120"))
LR = float(os.environ.get("RQ1_LR", "0.005"))
BATCH = int(os.environ.get("RQ1_BATCH", "256"))
D_MODEL = int(os.environ.get("RQ1_D_MODEL", "64"))
KEY_DIM = int(os.environ.get("RQ1_KEY_DIM", "64"))
VALUE_DIM = int(os.environ.get("RQ1_VALUE_DIM", "64"))
SEED = int(os.environ.get("RQ1_SEED", "0"))
MAX_SEQ = int(os.environ.get("RQ1_MAX_SEQ", "200"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 4

_log_fh = None


def tprint(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


# ---------------------------------------------------------------------------
# Collate: list of records -> padded tensors with both views + both masks
# ---------------------------------------------------------------------------

def collate(records: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad to a rectangular batch.

    Returns CPU tensors:
        q        (N,T) long, 1-based item ids, 0 = pad
        nominal  (N,T) long, 0..K-1, pad 0
        binary   (N,T) long, 0/1, pad 0
        bin_mask (N,T) bool, binary-visible AND non-pad
        nrm_mask (N,T) bool, nrm-visible AND non-pad
    """
    N = len(records)
    T = max(len(r["questions"]) for r in records)
    q = torch.zeros(N, T, dtype=torch.long)
    nominal = torch.zeros(N, T, dtype=torch.long)
    binary = torch.zeros(N, T, dtype=torch.long)
    bin_mask = torch.zeros(N, T, dtype=torch.bool)
    nrm_mask = torch.zeros(N, T, dtype=torch.bool)
    for i, r in enumerate(records):
        L = len(r["questions"])
        q[i, :L] = torch.from_numpy(r["questions"])
        nominal[i, :L] = torch.from_numpy(r["nominal"])
        binary[i, :L] = torch.from_numpy(r["binary"])
        bin_mask[i, :L] = torch.from_numpy(r["bin_vis"])
        nrm_mask[i, :L] = torch.from_numpy(r["nrm_vis"])
    return {"q": q, "nominal": nominal, "binary": binary,
            "bin_mask": bin_mask, "nrm_mask": nrm_mask}


def _full_info_masks(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Override the format masks so EVERY non-pad position is visible to BOTH
    heads (the full-information reference).  Padding stays masked."""
    nonpad = batch["q"] > 0
    out = dict(batch)
    out["bin_mask"] = nonpad.clone()
    out["nrm_mask"] = nonpad.clone()
    return out


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _iter_minibatches(N: int, gen: torch.Generator):
    perm = torch.randperm(N, generator=gen)
    for s in range(0, N, BATCH):
        yield perm[s:s + BATCH]


def train_joint(batch, n_items, label, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    model = JointModel(n_items, K=K, d_model=D_MODEL, key_dim=KEY_DIM,
                       value_dim=VALUE_DIM, correct_option=0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 1)
    N = batch["q"].size(0)
    tprint(f"\n[{label}] {N} learners, {n_items} items, {epochs} ep, "
           f"batch={BATCH}, device={DEVICE}")
    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        tot_b = tot_n = 0.0
        nb = nn_ = 0
        for bi in _iter_minibatches(N, gen):
            qb = batch["q"][bi].to(DEVICE)
            vr = batch["nominal"][bi].to(DEVICE)     # encoder value stream
            binb = batch["binary"][bi].to(DEVICE)
            nomb = batch["nominal"][bi].to(DEVICE)
            bmb = batch["bin_mask"][bi].to(DEVICE)
            nmb = batch["nrm_mask"][bi].to(DEVICE)
            opt.zero_grad()
            bin_nll, nrm_nll = model.losses(qb, vr, binb, nomb, bmb, nmb)
            loss = bin_nll + nrm_nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_b += float(bin_nll.item()); tot_n += float(nrm_nll.item())
            nb += 1; nn_ += 1
        if ep % 20 == 0 or ep == 1:
            tprint(f"  [{label}] ep {ep:4d}  bin_nll={tot_b/max(nb,1):.4f}  "
                   f"nrm_nll={tot_n/max(nn_,1):.4f}")
    tprint(f"  [{label}] done. t={time.time()-t0:.1f}s")
    return model


def train_indep_binary(batch, n_items, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed + 10)
    model = IndepBinary(n_items, K=K, d_model=D_MODEL, key_dim=KEY_DIM,
                        value_dim=VALUE_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 11)
    N = batch["q"].size(0)
    tprint(f"\n[IndepBinary] {N} learners, {epochs} ep")
    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0; nb = 0
        for bi in _iter_minibatches(N, gen):
            qb = batch["q"][bi].to(DEVICE)
            vr = batch["nominal"][bi].to(DEVICE)
            binb = batch["binary"][bi].to(DEVICE)
            bmb = batch["bin_mask"][bi].to(DEVICE)
            opt.zero_grad()
            loss = model.loss(qb, vr, binb, bmb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        if ep % 40 == 0 or ep == 1:
            tprint(f"  [IndepBinary] ep {ep:4d}  nll={tot/max(nb,1):.4f}")
    tprint(f"  [IndepBinary] done. t={time.time()-t0:.1f}s")
    return model


def train_indep_nrm(batch, n_items, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed + 20)
    model = IndepNRM(n_items, K=K, d_model=D_MODEL, key_dim=KEY_DIM,
                     value_dim=VALUE_DIM, correct_option=0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 21)
    N = batch["q"].size(0)
    tprint(f"\n[IndepNRM] {N} learners, {epochs} ep")
    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0; nb = 0
        for bi in _iter_minibatches(N, gen):
            qb = batch["q"][bi].to(DEVICE)
            vr = batch["nominal"][bi].to(DEVICE)
            nomb = batch["nominal"][bi].to(DEVICE)
            nmb = batch["nrm_mask"][bi].to(DEVICE)
            opt.zero_grad()
            loss = model.loss(qb, vr, nomb, nmb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        if ep % 40 == 0 or ep == 1:
            tprint(f"  [IndepNRM] ep {ep:4d}  nll={tot/max(nb,1):.4f}")
    tprint(f"  [IndepNRM] done. t={time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sp(x, y):
    r, _ = spearmanr(x, y)
    return float(r)


def _align(rec, ref):
    """Flip sign of rec to match ref (Spearman sign-free, but report signed)."""
    r, _ = spearmanr(rec, ref)
    return -rec if r < 0 else rec


def compute_metrics(d_joint, d_full, d_ib, d_inn, data: EdNetData) -> Dict:
    bo = data.binary_only_idx
    no = data.nrm_only_idx
    both = data.both_idx

    jd = _align(d_joint, d_full)
    ibd = _align(d_ib, d_full)
    ind = _align(d_inn, d_full)

    # Overall recovery vs the full-info reference.
    joint_overall = _sp(jd, d_full)

    # NRM-only: shared d_i (shaped by NRM head) read on the full-info scale.
    # transfer = NRM -> 2PL direction (the softer side, real data widens it).
    joint_nrm_only = _sp(jd[no], d_full[no])
    indep_bin_nrm_only = _sp(ibd[no], d_full[no])      # 2PL never saw them -> noise
    nrm_only_gap = joint_nrm_only - indep_bin_nrm_only

    # Binary-only: shared d_i (shaped by 2PL head) read as the NRM location.
    # transfer = 2PL -> NRM direction (binary->NRM, expected stronger).
    joint_bin_only = _sp(jd[bo], d_full[bo])
    indep_nrm_bin_only = _sp(ind[bo], d_full[bo])      # NRM never saw them -> noise
    bin_only_gap = joint_bin_only - indep_nrm_bin_only

    # Sanity: each indep model on its OWN items (should be solid).
    indep_bin_on_bin = _sp(ibd[bo], d_full[bo])
    indep_nrm_on_nrm = _sp(ind[no], d_full[no])

    joint_both = _sp(jd[both], d_full[both])

    return {
        "joint_overall_vs_fullinfo": joint_overall,
        "joint_on_nrm_only": joint_nrm_only,
        "indep_binary_nrm_only_noise": indep_bin_nrm_only,
        "nrm_only_transfer_gap": nrm_only_gap,
        "joint_on_binary_only": joint_bin_only,
        "indep_nrm_binary_only_noise": indep_nrm_bin_only,
        "binary_only_transfer_gap": bin_only_gap,
        "indep_binary_on_binary_only": indep_bin_on_bin,
        "indep_nrm_on_nrm_only": indep_nrm_on_nrm,
        "joint_on_both_bridge": joint_both,
        "n_binary_only": int(len(bo)),
        "n_nrm_only": int(len(no)),
        "n_both": int(len(both)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(data: EdNetData, n_learners: int, m: Dict) -> str:
    a = m["joint_on_nrm_only"]; b = m["joint_on_binary_only"]
    ga = m["nrm_only_transfer_gap"]; gb = m["binary_only_transfer_gap"]
    L = [
        "=" * 70,
        "REAL-DATA RQ1: BINARY-vs-NRM CROSS-FORMAT TRANSFER (ma-irt encoder)",
        "One shared item scale read by a 2PL head and an item-only NRM head?",
        "=" * 70, "",
        "DESIGN (real EdNet, interleaved per-learner sequences)",
        f"  Items (4-option, kept):  {data.n_items}",
        f"  BINARY-ONLY items:       {m['n_binary_only']}  (2PL view only)",
        f"  NRM-ONLY items:          {m['n_nrm_only']}  (chosen-option view only)",
        f"  BOTH (bridge) items:     {m['n_both']}  (seen by both heads)",
        f"  Learners:                {n_learners}",
        f"  Options K:               {K}   correct option index: 0 (remapped)", "",
        "REFERENCE = FULL-INFORMATION fit (every item BOTH; no synthetic truth)", "",
        "OVERALL",
        f"  Joint d_i vs full-info (all items):   {m['joint_overall_vs_fullinfo']:.4f}", "",
        "CROSS-FORMAT TRANSFER  (held-out-format placement)", "",
        "  [A] NRM-ONLY items  (NRM head shaped d_i; read on the 2PL/full scale)",
        f"      = NRM -> 2PL direction (softer side expected)",
        f"      Joint:                {a:.4f}   <-- cross-format",
        f"      Indep 2PL (noise):    {m['indep_binary_nrm_only_noise']:.4f}   (never saw them)",
        f"      Transfer gap:         {ga:+.4f}", "",
        "  [B] BINARY-ONLY items  (2PL head shaped d_i; read as NRM location)",
        f"      = binary -> NRM direction (stronger side expected)",
        f"      Joint:                {b:.4f}   <-- cross-format",
        f"      Indep NRM (noise):    {m['indep_nrm_binary_only_noise']:.4f}   (never saw them)",
        f"      Transfer gap:         {gb:+.4f}", "",
        "SANITY (each indep model on its OWN items vs full-info)",
        f"  Indep 2PL on BINARY-ONLY:  {m['indep_binary_on_binary_only']:.4f}",
        f"  Indep NRM on NRM-ONLY:     {m['indep_nrm_on_nrm_only']:.4f}",
        f"  Joint on BOTH bridge:      {m['joint_on_both_bridge']:.4f}", "",
        "VERDICT",
    ]
    if a >= 0.50 and b >= 0.50 and ga >= 0.25 and gb >= 0.25:
        v = (f"PASS. The shared ma-irt scale unifies binary and nominal on REAL "
             f"data. NRM-only items land on the 2PL scale (rho={a:.3f} vs noise "
             f"{m['indep_binary_nrm_only_noise']:.3f}), binary-only items land on "
             f"the NRM location scale (rho={b:.3f} vs noise "
             f"{m['indep_nrm_binary_only_noise']:.3f}). Both materially above the "
             f"independent-baseline noise floor.")
    elif (a >= 0.30 or b >= 0.30) and (ga >= 0.15 or gb >= 0.15):
        v = (f"PARTIAL. Measurable but soft / asymmetric transfer "
             f"(NRM->2PL {a:.3f}, binary->NRM {b:.3f}). Above noise but below a "
             f"clean unification claim.")
    else:
        v = (f"WEAK/FAIL. Transfer not materially above the noise floor "
             f"(NRM->2PL {a:.3f}, binary->NRM {b:.3f}).")
    L += [f"  {v}", "", "=" * 70]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _log_fh
    _log_fh = open(OUTPUT_DIR / "run.log", "w", buffering=1)
    t_start = time.time()
    tprint("=" * 70)
    tprint("REAL-DATA RQ1 binary-vs-NRM format-agnosticism (ma-irt encoder)")
    tprint("=" * 70)
    tprint(f"Config: N_LEARNERS={N_LEARNERS} EPOCHS={EPOCHS} LR={LR} BATCH={BATCH} "
           f"D_MODEL={D_MODEL} KEY={KEY_DIM} VALUE={VALUE_DIM} MAX_SEQ={MAX_SEQ} "
           f"SEED={SEED} DEVICE={DEVICE}")

    tprint("\n[1/6] Loading EdNet (bounded subset)...")
    data = load_ednet(n_learners=N_LEARNERS, max_seq_len=MAX_SEQ, seed=SEED)
    n_obs = sum(len(r["questions"]) for r in data.records)
    tprint(f"  learners={len(data.records)} items={data.n_items} obs={n_obs}")
    tprint(f"  partition: binary-only={len(data.binary_only_idx)} "
           f"nrm-only={len(data.nrm_only_idx)} both={len(data.both_idx)}")

    batch = collate(data.records)
    full_batch = _full_info_masks(batch)

    tprint("\n[2/6] Training JOINT model (shared encoder, 2PL + NRM heads)...")
    joint = train_joint(batch, data.n_items, "Joint")
    d_joint = joint.item_difficulty().cpu().numpy()

    tprint("\n[3/6] Training FULL-INFORMATION reference (every item BOTH)...")
    full = train_joint(full_batch, data.n_items, "FullInfo")
    d_full = full.item_difficulty().cpu().numpy()

    tprint("\n[4/6] Training INDEP binary baseline (separate encoder)...")
    ib = train_indep_binary(batch, data.n_items)
    d_ib = ib.item_difficulty().cpu().numpy()

    tprint("\n[5/6] Training INDEP NRM baseline (separate encoder)...")
    inn = train_indep_nrm(batch, data.n_items)
    d_inn = inn.item_difficulty().cpu().numpy()

    tprint("\n[6/6] Computing cross-format transfer metrics...")
    m = compute_metrics(d_joint, d_full, d_ib, d_inn, data)

    report = format_report(data, len(data.records), m)
    tprint("\n" + report)
    tprint(f"\nTotal wall: {time.time()-t_start:.1f}s")

    with open(OUTPUT_DIR / "report.txt", "w") as f:
        f.write(report + "\n")
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({k: (round(float(v), 6) if isinstance(v, (int, float, np.floating))
                       else v) for k, v in m.items()}, f, indent=2)
    group = data.group.copy()
    np.savez(OUTPUT_DIR / "item_arrays.npz",
             group=group, d_joint=d_joint, d_full=d_full,
             d_indep_binary=d_ib, d_indep_nrm=d_inn)
    tprint(f"Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
