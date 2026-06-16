"""run_strictblind.py -- STRICT-FORMAT-BLIND rerun of the real-data RQ1
binary-vs-NRM cross-format transfer (deep_irt/rq1_ednet_nrm).

Identical to run_experiment.py in EVERY respect (same learner/item subset, same
held-out-format partition, same full-information reference, same independent
single-format noise floor, same epochs/optimizer/seed) EXCEPT one thing: the
encoder VALUE/interaction stream is made STRICTLY FORMAT-BLIND.

Original run: the encoder value stream fed the 4-way NOMINAL option at every
position, including BINARY-ONLY items -- so a binary-only item's chosen option
entered the encoder's theta/history pathway (the honest caveat).

This run: at BINARY-ONLY positions the encoder value stream carries ONLY the
binary label (correct vs incorrect); the chosen distractor cannot enter
theta/history.  NRM-only and bridge positions are unchanged.  See
data_strictblind.py for the exact mapping.

The model, heads, shared d_i, targets, masks, and metric are all unchanged --
the JOINT / FullInfo / Indep models are imported from model.py verbatim.  Only
the value tensor fed to the encoder differs.  Writes to outputs_strictblind/
(the original outputs/ is NOT touched).

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="rl/src;ma-irt"
    python deep_irt/rq1_ednet_nrm/run_strictblind.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from deep_irt.rq1_ednet_nrm.data import EdNetData
from deep_irt.rq1_ednet_nrm.data_strictblind import load_ednet_strictblind
from deep_irt.rq1_ednet_nrm.model import JointModel, IndepBinary, IndepNRM
# Metric + report are format-agnostic -- reuse verbatim so numbers line up.
from deep_irt.rq1_ednet_nrm.run_experiment import compute_metrics, format_report

OUTPUT_DIR = Path(os.environ.get("RQ1_OUTPUT_DIR", str(HERE / "outputs_strictblind")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- bounded run knobs (env-overridable) -- SAME DEFAULTS as run_experiment ----
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
# Collate: adds the strict-blind ``value`` stream alongside the original tensors
# ---------------------------------------------------------------------------

def collate(records: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad to a rectangular batch.

    Identical to run_experiment.collate plus one extra tensor:
        value    (N,T) long, strict-blind encoder value stream (see
                 data_strictblind): nominal everywhere EXCEPT binary-only
                 positions, which carry only the binary label.
    """
    N = len(records)
    T = max(len(r["questions"]) for r in records)
    q = torch.zeros(N, T, dtype=torch.long)
    nominal = torch.zeros(N, T, dtype=torch.long)
    value = torch.zeros(N, T, dtype=torch.long)
    binary = torch.zeros(N, T, dtype=torch.long)
    bin_mask = torch.zeros(N, T, dtype=torch.bool)
    nrm_mask = torch.zeros(N, T, dtype=torch.bool)
    for i, r in enumerate(records):
        L = len(r["questions"])
        q[i, :L] = torch.from_numpy(r["questions"])
        nominal[i, :L] = torch.from_numpy(r["nominal"])
        value[i, :L] = torch.from_numpy(r["value"])
        binary[i, :L] = torch.from_numpy(r["binary"])
        bin_mask[i, :L] = torch.from_numpy(r["bin_vis"])
        nrm_mask[i, :L] = torch.from_numpy(r["nrm_vis"])
    return {"q": q, "nominal": nominal, "value": value, "binary": binary,
            "bin_mask": bin_mask, "nrm_mask": nrm_mask}


def _full_info_masks(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Full-information reference: every non-pad position visible to BOTH heads.

    IMPORTANT -- the full-information reference must also be format-blind-CONSISTENT:
    when every item is BOTH, no position is binary-only, so the encoder value
    stream should be the true NOMINAL everywhere (the richest signal the
    full-info fit is allowed).  We therefore set value <- nominal here so the
    reference d_i is the best location the data allows, exactly as in the
    original run.  Padding stays masked.
    """
    nonpad = batch["q"] > 0
    out = dict(batch)
    out["bin_mask"] = nonpad.clone()
    out["nrm_mask"] = nonpad.clone()
    out["value"] = batch["nominal"].clone()   # full-info: all positions BOTH
    return out


# ---------------------------------------------------------------------------
# Training loops -- identical to run_experiment EXCEPT vr = batch["value"]
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
            vr = batch["value"][bi].to(DEVICE)       # STRICT-BLIND value stream
            binb = batch["binary"][bi].to(DEVICE)
            nomb = batch["nominal"][bi].to(DEVICE)   # NRM target = true nominal
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
            vr = batch["value"][bi].to(DEVICE)       # STRICT-BLIND value stream
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
            vr = batch["value"][bi].to(DEVICE)       # STRICT-BLIND value stream
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
# Main
# ---------------------------------------------------------------------------

def main():
    global _log_fh
    _log_fh = open(OUTPUT_DIR / "run.log", "w", buffering=1)
    t_start = time.time()
    tprint("=" * 70)
    tprint("STRICT-FORMAT-BLIND RQ1 binary-vs-NRM (ma-irt encoder)")
    tprint("Encoder value stream is format-blind: binary-only positions carry")
    tprint("ONLY the binary label (no distractor identity into theta/history).")
    tprint("=" * 70)
    tprint(f"Config: N_LEARNERS={N_LEARNERS} EPOCHS={EPOCHS} LR={LR} BATCH={BATCH} "
           f"D_MODEL={D_MODEL} KEY={KEY_DIM} VALUE={VALUE_DIM} MAX_SEQ={MAX_SEQ} "
           f"SEED={SEED} DEVICE={DEVICE}")

    tprint("\n[1/6] Loading EdNet (bounded subset, strict-blind value stream)...")
    data = load_ednet_strictblind(n_learners=N_LEARNERS, max_seq_len=MAX_SEQ, seed=SEED)
    n_obs = sum(len(r["questions"]) for r in data.records)
    tprint(f"  learners={len(data.records)} items={data.n_items} obs={n_obs}")
    tprint(f"  partition: binary-only={len(data.binary_only_idx)} "
           f"nrm-only={len(data.nrm_only_idx)} both={len(data.both_idx)}")

    # Sanity: confirm binary-only positions in the value stream are 2-level only.
    bo_vals = set()
    for r in data.records:
        g = data.group[r["questions"] - 1]
        bo = (g == 1)  # G_BINARY_ONLY
        bo_vals.update(np.unique(r["value"][bo]).tolist())
    tprint(f"  strict-blind check: binary-only value tokens = {sorted(bo_vals)} "
           f"(expect subset of {{0, {K-1}}})")

    batch = collate(data.records)
    full_batch = _full_info_masks(batch)

    tprint("\n[2/6] Training JOINT model (strict-blind encoder, 2PL + NRM heads)...")
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
    # Re-title the report so it is unambiguous which run it is.
    report = report.replace(
        "REAL-DATA RQ1: BINARY-vs-NRM CROSS-FORMAT TRANSFER (ma-irt encoder)",
        "STRICT-FORMAT-BLIND RQ1: BINARY-vs-NRM TRANSFER (ma-irt encoder)")
    tprint("\n" + report)
    tprint(f"\nTotal wall: {time.time()-t_start:.1f}s")

    with open(OUTPUT_DIR / "report.txt", "w") as f:
        f.write(report + "\n")
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({k: (round(float(v), 6) if isinstance(v, (int, float, np.floating))
                       else v) for k, v in m.items()}, f, indent=2)
    np.savez(OUTPUT_DIR / "item_arrays.npz",
             group=data.group.copy(), d_joint=d_joint, d_full=d_full,
             d_indep_binary=d_ib, d_indep_nrm=d_inn)
    tprint(f"Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
