"""
run_pipeline.py -- End-to-end demonstration of the dynamic assessment deep_irt.

This script exercises the full pipeline in-process on CPU with synthetic data:

  1. DYNAMIC TRAJECTORY RECOVERY
     Generate random-walk ability data + base item bank.
     Fit DeepIRTModel (LSTM encoder + GPCM decoder).
     Report within-learner net-drift correlation (Spearman) and pooled RMSE.

  2. ANCHORED EXTENSION + TRACKING SURVIVAL
     Add E new items; anchor-calibrate against half the learners.
     Report new-item parameter recovery (Spearman discrimination, mean-threshold
     RMSE).  Then run B+E interleaved sequences through the frozen extended
     encoder and report tracking survival (Spearman of final theta B+E vs B-only).

  3. DECODER VERSATILITY
     a) Binary decoder: binarise responses and re-fit.  Check that the recovered
        item difficulty ranking agrees with the GPCM ranking (Spearman).
     b) Bradley-Terry decoder: generate pairwise comparison data tied to the
        GPCM ground-truth item strengths; fit BT; check cross-format agreement
        (Spearman BT-strength vs GPCM-difficulty, per bt_demo logic).

Usage
-----
  cd deep-mirt
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt python deep_irt/core/run_pipeline.py

Output saved to deep_irt/core/outputs/pipeline_report.txt
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

# Ensure deep_irt package is importable.
# __file__ is deep_irt/core/run_pipeline.py; we need the repo root (two levels up)
# so that "import deep_irt.core" resolves correctly.
_HERE = os.path.dirname(os.path.abspath(__file__))      # .../deep_irt/core/
_SUBSTRATE_DIR = os.path.dirname(_HERE)                  # .../deep_irt/
_REPO_ROOT = os.path.dirname(_SUBSTRATE_DIR)             # .../deep-mirt/
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deep_irt.core import DeepIRTModel, BradleyTerryDecoder

SEED = 0
DEVICE = torch.device("cpu")
OUTPUT_DIR = os.path.join(_HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# Utilities
# ===========================================================================

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    r, _ = sp_stats.spearmanr(x.ravel(), y.ravel())
    return float(r)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    r, _ = sp_stats.pearsonr(x.ravel(), y.ravel())
    return float(r)


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x.ravel() - y.ravel()) ** 2)))


def align_sign(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip sign of v if negatively correlated with ref."""
    r = pearson(v, ref)
    return -v if r < 0 else v


def section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}"


# ===========================================================================
# Data generation (self-contained; mirrors deep_irt/data.py logic)
# ===========================================================================

def _gpcm_probs_np(theta: float, a: float, b: np.ndarray) -> np.ndarray:
    K = len(b) + 1
    psi = np.zeros(K)
    for k in range(1, K):
        psi[k] = np.sum(a * (theta - b[:k]))
    psi -= psi.max()
    e = np.exp(psi)
    return e / e.sum()


def _sample_resp_np(theta, a, b, rng):
    probs = _gpcm_probs_np(theta, a, b)
    return int(rng.choice(len(probs), p=probs))


def generate_dynamic_data(
    n_learners: int = 800,
    n_base: int = 60,
    n_ext: int = 20,
    n_cats: int = 4,
    drift_sigma: float = 0.12,
    seed: int = 0,
) -> dict:
    """
    Synthetic random-walk ability data with base and extension items.

    Returns a dict of numpy arrays and python lists ready for tensor conversion.
    """
    rng = np.random.default_rng(seed)
    N, B, E, K = n_learners, n_base, n_ext, n_cats

    # Item parameters
    def sample_items(n):
        a = rng.lognormal(0.0, 0.3, size=n)
        b = np.sort(rng.standard_normal((n, K - 1)), axis=1)
        return a, b

    a_base, b_base = sample_items(B)
    a_ext, b_ext = sample_items(E)

    # Ability trajectories: random walk over B steps
    theta_0 = rng.standard_normal(N)
    steps = rng.normal(0.0, drift_sigma, size=(N, B))
    theta_traj = np.zeros((N, B))
    theta_traj[:, 0] = theta_0
    theta_traj[:, 1:] = theta_0[:, None] + np.cumsum(steps, axis=1)[:, :-1]

    # Anchor mask: random half
    anchor_mask = np.zeros(N, dtype=bool)
    anchor_idx = rng.choice(N, size=N // 2, replace=False)
    anchor_mask[anchor_idx] = True

    # Base sequences
    base_seqs = []
    for i in range(N):
        item_order = rng.permutation(B).tolist()
        responses, theta_true = [], []
        for t, j in enumerate(item_order):
            th = theta_traj[i, t]
            r = _sample_resp_np(th, a_base[j], b_base[j], rng)
            responses.append(r)
            theta_true.append(float(th))
        base_seqs.append({"items": item_order, "responses": responses,
                          "theta_true": theta_true})

    # Extension sequences (anchor learners only; fixed at end-of-B theta)
    ext_seqs = []
    for i in np.where(anchor_mask)[0]:
        theta_end = float(theta_traj[i, -1])
        item_order = rng.permutation(E).tolist()
        responses = [_sample_resp_np(theta_end, a_ext[j], b_ext[j], rng)
                     for j in item_order]
        ext_seqs.append({"items": item_order, "responses": responses,
                         "learner_idx": int(i), "theta_anchor": theta_end})

    return {
        "n_learners": N, "n_base": B, "n_ext": E, "n_cats": K,
        "a_base": a_base, "b_base": b_base,
        "a_ext": a_ext, "b_ext": b_ext,
        "theta_0": theta_0, "theta_traj": theta_traj,
        "anchor_mask": anchor_mask,
        "base_seqs": base_seqs,
        "ext_seqs": ext_seqs,
    }


def seqs_to_tensors(seqs, device=DEVICE):
    items = torch.tensor([s["items"] for s in seqs], dtype=torch.long, device=device)
    resps = torch.tensor([s["responses"] for s in seqs], dtype=torch.long, device=device)
    return items, resps


# ===========================================================================
# Stage 1: Dynamic trajectory recovery
# ===========================================================================

def stage1_dynamic(data: dict, n_epochs: int = 300) -> dict:
    print(section("STAGE 1: Dynamic Trajectory Recovery (GPCM encoder)"))

    N, B, K = data["n_learners"], data["n_base"], data["n_cats"]
    theta_traj = data["theta_traj"]  # (N, B) ground truth

    items_t, resps_t = seqs_to_tensors(data["base_seqs"])

    model = DeepIRTModel(
        num_items=B, emb_dim=8, hidden_dim=32, n_cats=K,
        decoder="gpcm", device=DEVICE, seed=SEED,
    )
    t0 = time.time()
    model.fit(items_t, resps_t, n_epochs=n_epochs, verbose=True)
    fit_time = time.time() - t0

    # Per-step theta trajectory: (N, B)
    theta_rec = model.track(items_t, resps_t).cpu().numpy()

    # Pooled Pearson across all (N, B) pairs
    pooled_r = pearson(theta_rec.ravel(), theta_traj.ravel())
    pooled_rmse = rmse(theta_rec.ravel(), theta_traj.ravel())

    # Per-learner net drift: (theta_last - theta_first)
    true_drift = theta_traj[:, -1] - theta_traj[:, 0]   # (N,)
    rec_drift = theta_rec[:, -1] - theta_rec[:, 0]        # (N,)
    nd_spearman = spearman(rec_drift, true_drift)
    nd_pearson = pearson(rec_drift, true_drift)
    nd_rmse = rmse(rec_drift, true_drift)

    print(f"\n  Fit time          : {fit_time:.1f}s")
    print(f"  Pooled Pearson r  : {pooled_r:.4f}  (all learner x timestep)")
    print(f"  Pooled RMSE       : {pooled_rmse:.4f}")
    print(f"  Net-drift Spearman: {nd_spearman:.4f}  (per-learner)")
    print(f"  Net-drift Pearson : {nd_pearson:.4f}")
    print(f"  Net-drift RMSE    : {nd_rmse:.4f}")

    return {
        "model": model,
        "theta_rec": theta_rec,
        "fit_time": fit_time,
        "pooled_pearson": pooled_r,
        "pooled_rmse": pooled_rmse,
        "nd_spearman": nd_spearman,
        "nd_pearson": nd_pearson,
        "nd_rmse": nd_rmse,
    }


# ===========================================================================
# Stage 2: Anchored extension + tracking survival
# ===========================================================================

def stage2_extension(data: dict, stage1: dict, n_epochs: int = 300) -> dict:
    print(section("STAGE 2: Anchored Extension + Tracking Survival"))

    model: DeepIRTModel = stage1["model"]
    B, E, K = data["n_base"], data["n_ext"], data["n_cats"]
    anchor_mask = data["anchor_mask"]
    anchor_indices = np.where(anchor_mask)[0]
    n_anchor = len(anchor_indices)

    # Pin theta for anchor learners from the base encoder
    base_items_t, base_resps_t = seqs_to_tensors(data["base_seqs"])
    model.encoder.eval()
    with torch.no_grad():
        theta_all = model.encoder.get_final_theta(
            base_items_t.to(DEVICE), base_resps_t.to(DEVICE)
        ).cpu()
    anchor_theta = theta_all[anchor_indices]  # (n_anchor,)

    # Build extension tensors (local item ids 0..E-1)
    ext_item_ids = torch.tensor(
        [s["items"] for s in data["ext_seqs"]], dtype=torch.long
    )  # (n_anchor, E)
    ext_resps = torch.tensor(
        [s["responses"] for s in data["ext_seqs"]], dtype=torch.long
    )  # (n_anchor, E)

    t0 = time.time()
    ext_result = model.extend(
        n_ext=E,
        anchor_theta=anchor_theta,
        ext_item_ids=ext_item_ids,
        ext_responses=ext_resps,
        n_epochs=n_epochs,
        verbose=True,
    )
    ext_time = time.time() - t0

    a_ext_rec = ext_result["a_ext"]            # (E,)
    b_ext_rec = ext_result["b_ext"]            # (E, K-1)
    a_ext_true = data["a_ext"]                 # (E,)
    b_ext_true = data["b_ext"]                 # (E, K-1)

    a_spearman = spearman(a_ext_rec, a_ext_true)
    b_rmse = rmse(b_ext_rec.ravel(), b_ext_true.ravel())
    b_pearson = pearson(b_ext_rec.ravel(), b_ext_true.ravel())

    print(f"\n  Extension fit time          : {ext_time:.1f}s")
    print(f"  Trainable params            : {ext_result['n_params']}")
    print(f"  Discrimination Spearman (a) : {a_spearman:.4f}")
    print(f"  Step-threshold Pearson (b)  : {b_pearson:.4f}")
    print(f"  Step-threshold RMSE (b)     : {b_rmse:.4f}")

    # --- Tracking survival: interleaved B+E sequences ---
    rng_shuffle = np.random.default_rng(SEED + 999)
    seq_len = B + E

    items_np = np.zeros((n_anchor, seq_len), dtype=np.int64)
    resps_np = np.zeros((n_anchor, seq_len), dtype=np.int64)

    ext_seq_map = {s["learner_idx"]: s for s in data["ext_seqs"]}

    for row, learner_i in enumerate(anchor_indices):
        base_s = data["base_seqs"][learner_i]
        ext_s = ext_seq_map[int(learner_i)]
        b_items = np.array(base_s["items"], dtype=np.int64)
        b_resps = np.array(base_s["responses"], dtype=np.int64)
        e_items = np.array(ext_s["items"], dtype=np.int64) + B  # shift to global ids
        e_resps = np.array(ext_s["responses"], dtype=np.int64)

        all_items = np.concatenate([b_items, e_items])
        all_resps = np.concatenate([b_resps, e_resps])
        perm = rng_shuffle.permutation(seq_len)
        items_np[row] = all_items[perm]
        resps_np[row] = all_resps[perm]

    items_be_t = torch.tensor(items_np, device=DEVICE)
    resps_be_t = torch.tensor(resps_np, device=DEVICE)

    # Final theta from extended model on B+E sequences
    theta_be = model.track(items_be_t, resps_be_t, use_extended=True).cpu().numpy()
    theta_be_final = theta_be[:, -1]                    # (n_anchor,)

    # Reference: B-only final theta from base model
    theta_b_final = anchor_theta.numpy()                # (n_anchor,)

    track_spearman = spearman(theta_be_final, theta_b_final)
    track_pearson = pearson(theta_be_final, theta_b_final)
    track_rmse = rmse(theta_be_final, theta_b_final)

    # Also check against ground-truth (end-of-trajectory theta)
    theta_gt_end = data["theta_traj"][anchor_indices, -1]
    be_vs_truth_sp = spearman(theta_be_final, theta_gt_end)

    print(f"\n  Tracking survival (B+E final theta vs B-only final theta):")
    print(f"    Spearman : {track_spearman:.4f}")
    print(f"    Pearson  : {track_pearson:.4f}")
    print(f"    RMSE     : {track_rmse:.4f}")
    print(f"  B+E final theta vs ground-truth end-theta (Spearman): {be_vs_truth_sp:.4f}")

    return {
        "ext_result": ext_result,
        "a_spearman": a_spearman,
        "b_pearson": b_pearson,
        "b_rmse": b_rmse,
        "track_spearman": track_spearman,
        "track_pearson": track_pearson,
        "track_rmse": track_rmse,
        "be_vs_truth_sp": be_vs_truth_sp,
        "ext_time": ext_time,
    }


# ===========================================================================
# Stage 3a: Binary decoder (GPCM K=2 agreement)
# ===========================================================================

def stage3a_binary(data: dict, stage1: dict, n_epochs: int = 300) -> dict:
    print(section("STAGE 3a: Binary Decoder (K=2 reduction)"))

    B, K = data["n_base"], data["n_cats"]

    # Binarise responses: 0 if below K//2, 1 otherwise
    binary_seqs = []
    for s in data["base_seqs"]:
        bin_resp = [1 if r >= K // 2 else 0 for r in s["responses"]]
        binary_seqs.append({"items": s["items"], "responses": bin_resp})

    items_t, resps_t = seqs_to_tensors(binary_seqs)

    model_bin = DeepIRTModel(
        num_items=B, emb_dim=8, hidden_dim=32, n_cats=2,
        decoder="binary", device=DEVICE, seed=SEED,
    )
    t0 = time.time()
    model_bin.fit(items_t, resps_t, n_epochs=n_epochs, verbose=True)
    fit_time = time.time() - t0

    # Recover item difficulty from binary model
    params_bin = model_bin.recover_item_params()
    # beta has shape (B, 1) for K=2; squeeze to (B,)
    d_bin = params_bin["beta"].squeeze(-1)                # (B,) "difficulty"

    # Recover item difficulty from GPCM model (stage 1): mean of thresholds
    gpcm_model: DeepIRTModel = stage1["model"]
    params_gpcm = gpcm_model.recover_item_params()
    d_gpcm = params_gpcm["beta"].mean(axis=1)            # (B,) mean of K-1 thresholds

    # Align signs before comparing
    d_bin_aligned = align_sign(d_bin, d_gpcm)

    bin_vs_gpcm_sp = spearman(d_bin_aligned, d_gpcm)
    bin_vs_gpcm_pe = pearson(d_bin_aligned, d_gpcm)

    print(f"\n  Fit time                         : {fit_time:.1f}s")
    print(f"  Binary difficulty vs GPCM difficulty:")
    print(f"    Spearman : {bin_vs_gpcm_sp:.4f}")
    print(f"    Pearson  : {bin_vs_gpcm_pe:.4f}")

    return {
        "model_bin": model_bin,
        "d_bin": d_bin,
        "d_gpcm": d_gpcm,
        "bin_vs_gpcm_spearman": bin_vs_gpcm_sp,
        "bin_vs_gpcm_pearson": bin_vs_gpcm_pe,
        "fit_time": fit_time,
    }


# ===========================================================================
# Stage 3b: Bradley-Terry cross-format agreement
# ===========================================================================

def _generate_pairwise(a_true: np.ndarray, b_true: np.ndarray,
                        s_true: np.ndarray,
                        comparisons_per_item: int = 50,
                        seed: int = 0):
    """
    Generate pairwise comparison data whose ground truth is s_true.
    P(i beats j) = sigmoid(s_i - s_j).
    Also returns mode-A (ordinal GPCM) observations for the same items.
    """
    rng_t = torch.Generator()
    rng_t.manual_seed(seed + 200)

    n_items = len(s_true)
    s_t = torch.tensor(s_true, dtype=torch.float32)
    n_pairs = int(n_items * comparisons_per_item / 2)

    item_i_list, item_j_list, outcome_list = [], [], []
    generated = 0
    while generated < n_pairs:
        batch = min(1000, n_pairs - generated)
        ii = torch.randint(0, n_items, (batch,), generator=rng_t)
        jj = torch.randint(0, n_items, (batch,), generator=rng_t)
        valid = ii != jj
        ii, jj = ii[valid], jj[valid]
        if len(ii) == 0:
            continue
        delta = s_t[ii] - s_t[jj]
        outcome = torch.bernoulli(torch.sigmoid(delta), generator=rng_t).long()
        item_i_list.append(ii)
        item_j_list.append(jj)
        outcome_list.append(outcome)
        generated += len(ii)

    item_i = torch.cat(item_i_list)[:n_pairs]
    item_j = torch.cat(item_j_list)[:n_pairs]
    outcome = torch.cat(outcome_list)[:n_pairs]
    return {"item_i": item_i, "item_j": item_j, "outcome": outcome, "n_pairs": n_pairs}


def stage3b_bt(data: dict, stage1: dict, n_epochs: int = 500) -> dict:
    print(section("STAGE 3b: Bradley-Terry Cross-Format Agreement"))

    B = data["n_base"]
    a_true = data["a_base"]                  # (B,)
    b_true = data["b_base"]                  # (B, K-1)
    # Ground-truth item strength: use mean step threshold as difficulty proxy
    s_true = b_true.mean(axis=1)             # (B,)

    print(f"\n  Generating {B}-item pairwise comparison data...")
    pairs = _generate_pairwise(a_true, b_true, s_true, comparisons_per_item=50, seed=SEED)
    print(f"  Pairs generated: {pairs['n_pairs']}")

    # Build a BT DeepIRTModel sharing the same emb_dim
    model_bt = DeepIRTModel(
        num_items=B, emb_dim=8, hidden_dim=32, n_cats=4,
        decoder="bt", device=DEVICE, seed=SEED,
    )

    # Use the *frozen* base encoder's item embeddings as fixed item representations.
    # The BT decoder fc_s maps these embeddings to scalar strengths.
    # We freeze the encoder embeddings and only train the fc_s layer.
    gpcm_model: DeepIRTModel = stage1["model"]
    gpcm_enc = gpcm_model.encoder

    # Copy base encoder embedding weights into BT model encoder (for consistent scale)
    with torch.no_grad():
        model_bt.encoder.item_val_emb.weight.copy_(gpcm_enc.item_val_emb.weight)
    # Freeze the BT encoder's item embeddings (only train fc_s in BT decoder)
    for p in model_bt.encoder.parameters():
        p.requires_grad_(False)

    # Precompute item embeddings for all pairs (using frozen encoder)
    model_bt.encoder.eval()
    with torch.no_grad():
        all_embs = model_bt.encoder.item_val_emb.weight    # (B, emb_dim)
        emb_i = all_embs[pairs["item_i"].to(DEVICE)]   # (N_pairs, emb_dim)
        emb_j = all_embs[pairs["item_j"].to(DEVICE)]   # (N_pairs, emb_dim)

    t0 = time.time()
    model_bt.fit_pairs(
        item_emb_i=emb_i,
        item_emb_j=emb_j,
        outcome=pairs["outcome"].to(DEVICE),
        n_epochs=n_epochs,
        verbose=True,
    )
    fit_time = time.time() - t0

    # Recover BT strengths for all items
    bt_params = model_bt.recover_item_params()
    s_bt = bt_params["strength"]                  # (B,)
    s_bt = s_bt - s_bt.mean()                     # mean-center

    # GPCM difficulty from stage1 (mean threshold)
    gpcm_params = gpcm_model.recover_item_params()
    d_gpcm = gpcm_params["beta"].mean(axis=1)     # (B,)

    # Align signs before cross-format comparison
    s_true_np = s_true
    s_bt_aligned = align_sign(s_bt, s_true_np)
    d_gpcm_aligned = align_sign(d_gpcm, s_true_np)

    # vs ground truth
    bt_vs_truth_sp = spearman(s_bt_aligned, s_true_np)
    gpcm_vs_truth_sp = spearman(d_gpcm_aligned, s_true_np)

    # Cross-format: BT strength vs GPCM difficulty
    bt_vs_gpcm_sp = spearman(s_bt_aligned, d_gpcm_aligned)
    bt_vs_gpcm_pe = pearson(s_bt_aligned, d_gpcm_aligned)

    print(f"\n  Fit time                             : {fit_time:.1f}s")
    print(f"  BT strength vs ground truth (Spearman)  : {bt_vs_truth_sp:.4f}")
    print(f"  GPCM difficulty vs ground truth (Spearman): {gpcm_vs_truth_sp:.4f}")
    print(f"\n  Cross-format agreement (BT strength vs GPCM difficulty):")
    print(f"    Spearman : {bt_vs_gpcm_sp:.4f}")
    print(f"    Pearson  : {bt_vs_gpcm_pe:.4f}")

    return {
        "model_bt": model_bt,
        "s_bt": s_bt_aligned,
        "d_gpcm": d_gpcm_aligned,
        "bt_vs_truth_spearman": bt_vs_truth_sp,
        "gpcm_vs_truth_spearman": gpcm_vs_truth_sp,
        "bt_vs_gpcm_spearman": bt_vs_gpcm_sp,
        "bt_vs_gpcm_pearson": bt_vs_gpcm_pe,
        "fit_time": fit_time,
    }


# ===========================================================================
# Consolidated report
# ===========================================================================

def build_report(data: dict, s1: dict, s2: dict, s3a: dict, s3b: dict) -> str:
    lines = [
        "=" * 70,
        "SUBSTRATE PIPELINE -- CONSOLIDATED METRICS REPORT",
        "=" * 70,
        "",
        "SETUP",
        f"  Learners (N)          : {data['n_learners']}",
        f"  Base items (B)        : {data['n_base']}",
        f"  Extension items (E)   : {data['n_ext']}",
        f"  Categories (K)        : {data['n_cats']}",
        f"  Encoder               : LSTM (emb_dim=8, hidden_dim=32)",
        f"  Seed                  : {SEED}",
        "",
        "-" * 70,
        "STAGE 1: Dynamic Trajectory Recovery (GPCM)",
        "-" * 70,
        f"  Pooled Pearson r (all learner x timestep) : {s1['pooled_pearson']:.4f}",
        f"  Pooled RMSE                               : {s1['pooled_rmse']:.4f}",
        f"  Net-drift Spearman (per-learner)          : {s1['nd_spearman']:.4f}",
        f"  Net-drift Pearson                         : {s1['nd_pearson']:.4f}",
        f"  Net-drift RMSE                            : {s1['nd_rmse']:.4f}",
        f"  Fit time                                  : {s1['fit_time']:.1f}s",
        "",
        "-" * 70,
        "STAGE 2: Anchored Extension + Tracking Survival",
        "-" * 70,
        f"  Trainable params (Phase 2)                : {s2['ext_result']['n_params']}",
        f"  New-item discrimination Spearman (a)      : {s2['a_spearman']:.4f}",
        f"  New-item step-threshold Pearson (b)       : {s2['b_pearson']:.4f}",
        f"  New-item step-threshold RMSE (b)          : {s2['b_rmse']:.4f}",
        f"  Tracking survival Spearman (B+E vs B-only): {s2['track_spearman']:.4f}",
        f"  Tracking survival Pearson                 : {s2['track_pearson']:.4f}",
        f"  Tracking survival RMSE                    : {s2['track_rmse']:.4f}",
        f"  B+E final theta vs truth Spearman         : {s2['be_vs_truth_sp']:.4f}",
        f"  Extension fit time                        : {s2['ext_time']:.1f}s",
        "",
        "-" * 70,
        "STAGE 3a: Binary Decoder (K=2 agreement with GPCM)",
        "-" * 70,
        f"  Binary difficulty vs GPCM difficulty:",
        f"    Spearman : {s3a['bin_vs_gpcm_spearman']:.4f}",
        f"    Pearson  : {s3a['bin_vs_gpcm_pearson']:.4f}",
        f"  Fit time : {s3a['fit_time']:.1f}s",
        "",
        "-" * 70,
        "STAGE 3b: Bradley-Terry Cross-Format Agreement",
        "-" * 70,
        f"  BT strength vs ground truth (Spearman)    : {s3b['bt_vs_truth_spearman']:.4f}",
        f"  GPCM difficulty vs ground truth (Spearman): {s3b['gpcm_vs_truth_spearman']:.4f}",
        f"  Cross-format BT vs GPCM (Spearman)        : {s3b['bt_vs_gpcm_spearman']:.4f}",
        f"  Cross-format BT vs GPCM (Pearson)         : {s3b['bt_vs_gpcm_pearson']:.4f}",
        f"  Fit time : {s3b['fit_time']:.1f}s",
        "",
        "=" * 70,
        "VERDICT",
        "=" * 70,
    ]

    checks = []

    # Trajectory recovery: net-drift Spearman > 0.40 considered meaningful
    nd_sp = s1["nd_spearman"]
    if nd_sp >= 0.60:
        checks.append(f"  PASS  Trajectory recovery: net-drift Spearman={nd_sp:.3f} (>= 0.60)")
    elif nd_sp >= 0.40:
        checks.append(f"  OK    Trajectory recovery: net-drift Spearman={nd_sp:.3f} (>= 0.40)")
    else:
        checks.append(f"  WARN  Trajectory recovery: net-drift Spearman={nd_sp:.3f} (< 0.40)")

    # Extension recovery: discrimination Spearman > 0.40
    a_sp = s2["a_spearman"]
    if a_sp >= 0.60:
        checks.append(f"  PASS  Extension item recovery: a Spearman={a_sp:.3f} (>= 0.60)")
    elif a_sp >= 0.40:
        checks.append(f"  OK    Extension item recovery: a Spearman={a_sp:.3f} (>= 0.40)")
    else:
        checks.append(f"  WARN  Extension item recovery: a Spearman={a_sp:.3f} (< 0.40)")

    # Tracking survival: > 0.80 tight agreement expected (frozen encoder, same scale)
    tr_sp = s2["track_spearman"]
    if tr_sp >= 0.85:
        checks.append(f"  PASS  Tracking survival: Spearman={tr_sp:.3f} (>= 0.85)")
    elif tr_sp >= 0.70:
        checks.append(f"  OK    Tracking survival: Spearman={tr_sp:.3f} (>= 0.70)")
    else:
        checks.append(f"  WARN  Tracking survival: Spearman={tr_sp:.3f} (< 0.70)")

    # Cross-format: BT vs GPCM Spearman > 0.75
    xf_sp = s3b["bt_vs_gpcm_spearman"]
    if xf_sp >= 0.85:
        checks.append(f"  PASS  Cross-format agreement: Spearman={xf_sp:.3f} (>= 0.85)")
    elif xf_sp >= 0.70:
        checks.append(f"  OK    Cross-format agreement: Spearman={xf_sp:.3f} (>= 0.70)")
    else:
        checks.append(f"  WARN  Cross-format agreement: Spearman={xf_sp:.3f} (< 0.70)")

    lines.extend(checks)
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 70)
    print("SUBSTRATE CORE -- End-to-End Pipeline")
    print("CPU only | synthetic data | seed=0")
    print("=" * 70)

    t_total = time.time()

    # ---- Generate data ----
    print("\nGenerating synthetic random-walk data...")
    data = generate_dynamic_data(
        n_learners=800, n_base=60, n_ext=20, n_cats=4,
        drift_sigma=0.12, seed=SEED,
    )
    print(f"  {data['n_learners']} learners  |  "
          f"{data['n_base']} base items  |  {data['n_ext']} ext items  |  "
          f"K={data['n_cats']}")
    print(f"  Anchor learners: {data['anchor_mask'].sum()}")

    # ---- Stage 1 ----
    s1 = stage1_dynamic(data, n_epochs=300)

    # ---- Stage 2 ----
    s2 = stage2_extension(data, s1, n_epochs=300)

    # ---- Stage 3a ----
    s3a = stage3a_binary(data, s1, n_epochs=300)

    # ---- Stage 3b ----
    s3b = stage3b_bt(data, s1, n_epochs=500)

    # ---- Report ----
    report = build_report(data, s1, s2, s3a, s3b)

    print("\n\n" + report)

    out_path = os.path.join(OUTPUT_DIR, "pipeline_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")
    print(f"\nTotal wall-clock time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
