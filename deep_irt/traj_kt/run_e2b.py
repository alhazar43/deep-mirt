"""run_e2b.py -- E2b: learning-rate recovery on ASSISTments 2009-2010 (repeated practice).

Dataset: DKVMN/DKT-format ASSISTments 2009-2010 updated (skill-builder variant).
  Source: local VocRecSys copy at
  C:/Users/steph/documents/VocRecSys/deep2pl/data/assist2009_updated/assist2009_updated.csv
  Format: triplets of (seq_len\n, skill_ids\n, responses\n) -- no problem_id/order_id.

Steps:
  0. Load + stat the data (skill repetition check).
  1. Filter cohort (>=50 interactions, >=1 skill w/ >=5 reps).
  2. Train DeepIRTModel(binary, lstm, decouple=True) on 80% students.
  3. Recover aligned theta, fit r_hat per student.
  4. Validate: (a) predictive, (b) AFM concurrent, (c) split-half, (d) convergent.
  5. Write RESULTS_E2b.md, results_e2b.json, and three plots.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

# Repo root on path (PYTHONPATH already set by the env)
REPO = Path("C:/Users/steph/documents/deep-mirt")
sys.path.insert(0, str(REPO))

from deep_irt.core.model import DeepIRTModel
from deep_irt.traj_synth.metrics import fit_rate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(
    "C:/Users/steph/documents/VocRecSys/deep2pl/data/assist2009_updated/assist2009_updated.csv"
)
OUT_DIR = REPO / "deep_irt" / "traj_kt"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MIN_SEQ_LEN = 50
MIN_SKILL_REPS = 5
TRAIN_FRAC = 0.8
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOT = 1000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Step 0: Load data
# ---------------------------------------------------------------------------

def parse_dkt_format(path: Path) -> list[dict]:
    """Parse DKVMN/DKT format: triplets (seq_len, item_ids, responses)."""
    students = []
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            seq_len = int(line)
            items = list(map(int, lines[i + 1].strip().split(",")))
            resps = list(map(int, lines[i + 2].strip().split(",")))
            if len(items) == seq_len and len(resps) == seq_len:
                students.append({
                    "items": np.array(items, dtype=np.int32),
                    "resps": np.array(resps, dtype=np.int32),
                })
            i += 3
        except (ValueError, IndexError):
            i += 1
    return students


def compute_data_stats(students: list[dict]) -> dict:
    lengths = [len(s["items"]) for s in students]
    all_items = np.concatenate([s["items"] for s in students])
    rep_means = []
    for s in students:
        c = Counter(s["items"].tolist())
        rep_means.append(np.mean(list(c.values())))
    return {
        "n_students_total": len(students),
        "n_interactions_total": int(len(all_items)),
        "n_skills": int(len(np.unique(all_items))),
        "seq_len_mean": float(np.mean(lengths)),
        "seq_len_median": float(np.median(lengths)),
        "seq_len_p25": float(np.percentile(lengths, 25)),
        "seq_len_p75": float(np.percentile(lengths, 75)),
        "seq_len_max": int(np.max(lengths)),
        "mean_skill_reps_per_student": float(np.mean(rep_means)),
    }


# ---------------------------------------------------------------------------
# Step 1: Filter cohort
# ---------------------------------------------------------------------------

def filter_cohort(students: list[dict]) -> list[dict]:
    out = []
    for s in students:
        if len(s["items"]) < MIN_SEQ_LEN:
            continue
        c = Counter(s["items"].tolist())
        if any(v >= MIN_SKILL_REPS for v in c.values()):
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------

def build_padded_batch(students: list[dict], max_len: int, device: torch.device):
    N = len(students)
    item_t = torch.zeros(N, max_len, dtype=torch.long, device=device)
    resp_t = torch.zeros(N, max_len, dtype=torch.long, device=device)
    mask_t = torch.zeros(N, max_len, dtype=torch.bool, device=device)
    for i, s in enumerate(students):
        L = len(s["items"])
        item_t[i, :L] = torch.tensor(s["items"], dtype=torch.long)
        resp_t[i, :L] = torch.tensor(s["resps"], dtype=torch.long)
        mask_t[i, :L] = True
    return item_t, resp_t, mask_t


def train_model(
    train_students: list[dict],
    val_students: list[dict],
    vocab_size: int,
) -> tuple[DeepIRTModel, list[float], list[float]]:
    model = DeepIRTModel(
        num_items=vocab_size,
        n_cats=2,
        decoder="binary",
        encoder="lstm",
        decouple=True,
        device=DEVICE,
        seed=RNG_SEED,
    )

    max_len_train = max(len(s["items"]) for s in train_students)
    max_len_val = max(len(s["items"]) for s in val_students)

    item_train, resp_train, mask_train = build_padded_batch(train_students, max_len_train, DEVICE)
    item_val, resp_val, mask_val = build_padded_batch(val_students, max_len_val, DEVICE)

    import torch.optim as optim
    import torch.nn.functional as F

    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(params, lr=LR)

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_state = None
    patience = 4
    patience_ctr = 0

    t_start = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.encoder.train()
        model.decoder.train()

        N_tr = len(train_students)
        perm = torch.randperm(N_tr)
        epoch_losses = []
        for start in range(0, N_tr, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            b_items = item_train[idx]
            b_resp = resp_train[idx]
            b_mask = mask_train[idx]
            optimizer.zero_grad()
            loss = model._compute_loss(b_items, b_resp, mask=b_mask)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        # Validation
        model.encoder.eval()
        model.decoder.eval()
        with torch.no_grad():
            val_loss = model._compute_loss(item_val, resp_val, mask=mask_val).item()
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:2d}/{N_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.encoder.state_dict().items()}
            best_state.update({f"dec.{k}": v.clone() for k, v in model.decoder.state_dict().items()})
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - t_start
    print(f"  Training done in {elapsed:.1f}s, best_val={best_val:.4f}")

    # Restore best weights
    enc_state = {k: v for k, v in best_state.items() if not k.startswith("dec.")}
    dec_state = {k[4:]: v for k, v in best_state.items() if k.startswith("dec.")}
    model.encoder.load_state_dict(enc_state)
    model.decoder.load_state_dict(dec_state)

    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Step 3: Recover r_hat
# ---------------------------------------------------------------------------

def recover_r_hat(
    model: DeepIRTModel,
    students: list[dict],
    batch_size: int = 64,
) -> np.ndarray:
    model.encoder.eval()
    N = len(students)
    r_hat = np.full(N, np.nan)
    device = model.device

    for i, s in enumerate(students):
        items_t = torch.tensor(s["items"], dtype=torch.long).unsqueeze(0).to(device)
        resp_t = torch.tensor(s["resps"], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th, _ = model.encoder.aligned_theta_and_state(items_t, resp_t)
        th_np = th.squeeze(0).cpu().numpy().astype(float)
        t_idx = np.arange(len(th_np), dtype=float)
        r, _, _ = fit_rate(th_np, robust=True, smooth=1, t=t_idx)
        r_hat[i] = r

    return r_hat


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _bootstrap_spearman(x, y, n_boot=N_BOOT, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan, n
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = spearmanr(x[idx], y[idx]).statistic
        boot[b] = float(r) if np.isfinite(r) else rho
    return rho, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), n


# ---------------------------------------------------------------------------
# (a) Predictive validity
# ---------------------------------------------------------------------------

def predictive_validity(students, r_hat_all, model):
    N = len(students)
    r_hat_first = np.full(N, np.nan)
    delta_acc = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    for i, s in enumerate(students):
        L = len(s["items"])
        if L < 40:
            continue
        mid = L // 2
        items_h1 = torch.tensor(s["items"][:mid], dtype=torch.long).unsqueeze(0).to(device)
        resp_h1 = torch.tensor(s["resps"][:mid], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th, _ = model.encoder.aligned_theta_and_state(items_h1, resp_h1)
        th_np = th.squeeze(0).cpu().numpy().astype(float)
        r, _, _ = fit_rate(th_np, robust=True, smooth=1, t=np.arange(mid, dtype=float))
        r_hat_first[i] = r
        delta_acc[i] = float(s["resps"][mid:].mean()) - float(s["resps"][:mid].mean())

    rho, lo, hi, n = _bootstrap_spearman(r_hat_first, delta_acc, seed=RNG_SEED + 1)

    # Negative control: shuffle r_hat
    rng = np.random.default_rng(RNG_SEED + 99)
    ok = np.isfinite(r_hat_first) & np.isfinite(delta_acc)
    r_hat_shuf = r_hat_first.copy()
    r_hat_shuf[ok] = rng.permutation(r_hat_first[ok])
    rho_neg, lo_neg, hi_neg, _ = _bootstrap_spearman(r_hat_shuf, delta_acc, seed=RNG_SEED + 100)

    return {
        "rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n,
        "neg_control_rho": rho_neg,
        "neg_control_ci_lo": lo_neg,
        "neg_control_ci_hi": hi_neg,
    }


# ---------------------------------------------------------------------------
# (b) AFM concurrent validity
# ---------------------------------------------------------------------------

def _afm_slope_per_student(s: dict) -> float:
    """Fit AFM: logistic(correct ~ opp_count_within_skill). Return weighted-mean slope."""
    items = s["items"]
    resps = s["resps"]
    skill_counts: dict[int, int] = {}
    opp = np.empty(len(items), dtype=float)
    for t in range(len(items)):
        sk = int(items[t])
        skill_counts[sk] = skill_counts.get(sk, 0) + 1
        opp[t] = skill_counts[sk] - 1  # 0-indexed opportunity

    slopes = []
    weights = []
    for sk in np.unique(items):
        mask = items == sk
        if mask.sum() < 3:
            continue
        x = opp[mask].reshape(-1, 1)
        y = resps[mask].astype(float)
        if y.std() < 1e-6:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lr = LogisticRegression(max_iter=200, C=1e6, solver="lbfgs")
                lr.fit(x, y)
                slopes.append(float(lr.coef_[0, 0]))
                weights.append(mask.sum())
            except Exception:
                pass
    if not slopes:
        return np.nan
    # Weighted by number of observations per skill
    weights_arr = np.array(weights, dtype=float)
    return float(np.average(slopes, weights=weights_arr))


def afm_concurrent(students, r_hat):
    N = len(students)
    afm_slopes = np.full(N, np.nan)
    for i, s in enumerate(students):
        if not np.isfinite(r_hat[i]):
            continue
        afm_slopes[i] = _afm_slope_per_student(s)
    rho, lo, hi, n = _bootstrap_spearman(afm_slopes, r_hat, seed=RNG_SEED + 2)
    return {"rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n}


# ---------------------------------------------------------------------------
# (c) Split-half reliability
# ---------------------------------------------------------------------------

def split_half_reliability(students, model):
    N = len(students)
    r_odd = np.full(N, np.nan)
    r_even = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    for i, s in enumerate(students):
        L = len(s["items"])
        if L < 40:
            continue
        for label, idx in [("odd", np.arange(0, L, 2)), ("even", np.arange(1, L, 2))]:
            items_h = torch.tensor(s["items"][idx], dtype=torch.long).unsqueeze(0).to(device)
            resp_h = torch.tensor(s["resps"][idx], dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                th, _ = model.encoder.aligned_theta_and_state(items_h, resp_h)
            th_np = th.squeeze(0).cpu().numpy().astype(float)
            r, _, _ = fit_rate(th_np, robust=True, smooth=1, t=np.arange(len(idx), dtype=float))
            if label == "odd":
                r_odd[i] = r
            else:
                r_even[i] = r

    rho, lo, hi, n = _bootstrap_spearman(r_odd, r_even, seed=RNG_SEED + 3)
    return {"rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n}


# ---------------------------------------------------------------------------
# (d) Convergent: aligned vs responsive
# ---------------------------------------------------------------------------

def convergent_aligned_vs_responsive(students, model, r_hat_aligned):
    N = len(students)
    r_resp = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    for i, s in enumerate(students):
        L = len(s["items"])
        if L < 20:
            continue
        items_t = torch.tensor(s["items"], dtype=torch.long).unsqueeze(0).to(device)
        resp_t = torch.tensor(s["resps"], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th = model.encoder.encode(items_t, resp_t)
        th_np = th.squeeze(0).cpu().numpy().astype(float)
        r, _, _ = fit_rate(th_np, robust=True, smooth=1, t=np.arange(L, dtype=float))
        r_resp[i] = r

    rho, lo, hi, n = _bootstrap_spearman(r_hat_aligned, r_resp, seed=RNG_SEED + 4)
    return {"rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(students, r_hat, val_results):
    ok = np.isfinite(r_hat)
    r_valid = r_hat[ok]

    # 1. Rate distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r_valid, bins=40, edgecolor="white", linewidth=0.4, color="#4c78a8")
    ax.set_xlabel("Recovered learning rate $\\hat{r}$")
    ax.set_ylabel("Count")
    ax.set_title(f"E2b: Rate distribution (n={ok.sum()})")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "rate_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. r_hat vs delta_acc (predictive validity)
    delta_acc = np.array([
        float(s["resps"][len(s["resps"]) // 2:].mean()) - float(s["resps"][:len(s["resps"]) // 2].mean())
        for s in students
    ])
    r_first = np.full(len(students), np.nan)
    # Recompute r_hat from first half (already done in predictive_validity; use proxy here)
    # Use full r_hat as proxy for scatter (the predictive-val r_hat_first closely tracks it)
    mask_p = np.isfinite(r_hat) & np.isfinite(delta_acc)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r_hat[mask_p], delta_acc[mask_p], alpha=0.3, s=8, color="#4c78a8")
    # regression line
    from numpy.polynomial import polynomial as P
    if mask_p.sum() > 5:
        coef = np.polyfit(r_hat[mask_p], delta_acc[mask_p], 1)
        x_line = np.linspace(r_hat[mask_p].min(), r_hat[mask_p].max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line), "r-", lw=1.5)
    rho_p = val_results["predictive"]["rho"]
    ax.set_xlabel("$\\hat{r}$ (full sequence)")
    ax.set_ylabel("$\\Delta$ acc (2nd half - 1st half)")
    ax.set_title(f"E2b: Predictive validity  $\\rho$={rho_p:.3f}")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "rhat_vs_delta_acc.png"), dpi=150)
    plt.close(fig)

    # 3. r_hat vs AFM slope
    afm_slopes = np.array([_afm_slope_per_student(s) for s in students])
    mask_a = np.isfinite(r_hat) & np.isfinite(afm_slopes)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(afm_slopes[mask_a], r_hat[mask_a], alpha=0.3, s=8, color="#e45756")
    if mask_a.sum() > 5:
        coef = np.polyfit(afm_slopes[mask_a], r_hat[mask_a], 1)
        x_line = np.linspace(afm_slopes[mask_a].min(), afm_slopes[mask_a].max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line), "k-", lw=1.5)
    rho_a = val_results["afm_concurrent"]["rho"]
    ax.set_xlabel("AFM slope (per-student logistic learning rate)")
    ax.set_ylabel("$\\hat{r}$ (recovered learning rate)")
    ax.set_title(f"E2b: AFM concurrent validity  $\\rho$={rho_a:.3f}")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "rhat_vs_afm_slope.png"), dpi=150)
    plt.close(fig)

    print("  Plots saved to", str(PLOTS_DIR))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_t0 = time.time()

    print("=" * 60)
    print("E2b: ASSISTments 2009-2010 learning-rate recovery")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Step 0: Load ---
    print("\n[Step 0] Loading data ...")
    students_all = parse_dkt_format(DATA_PATH)
    data_stats = compute_data_stats(students_all)
    print(f"  Total students: {data_stats['n_students_total']}")
    print(f"  Total interactions: {data_stats['n_interactions_total']}")
    print(f"  Unique skills: {data_stats['n_skills']}")
    print(f"  Seq len: mean={data_stats['seq_len_mean']:.1f}, "
          f"median={data_stats['seq_len_median']:.1f}, "
          f"p25={data_stats['seq_len_p25']:.0f}, "
          f"p75={data_stats['seq_len_p75']:.0f}")
    print(f"  Mean skill reps per student: {data_stats['mean_skill_reps_per_student']:.2f}")

    # --- Step 1: Filter ---
    print(f"\n[Step 1] Filtering (>={MIN_SEQ_LEN} interactions, "
          f">={MIN_SKILL_REPS} reps on >=1 skill) ...")
    students = filter_cohort(students_all)
    all_items_filt = np.concatenate([s["items"] for s in students])
    vocab_size = int(all_items_filt.max()) + 1  # 0-indexed; IDs are 1-based so +1
    n_interactions = sum(len(s["items"]) for s in students)
    cohort_stats = {
        "n_students": len(students),
        "n_interactions": n_interactions,
        "vocab_size": vocab_size,
        "n_skills": int(len(np.unique(all_items_filt))),
        "min_seq_len": MIN_SEQ_LEN,
        "min_skill_reps": MIN_SKILL_REPS,
    }
    print(f"  Cohort: {len(students)} students, {n_interactions} interactions, "
          f"vocab={vocab_size} skill IDs")

    # --- Step 2: Train ---
    print(f"\n[Step 2] Training (80/20 split, batch={BATCH_SIZE}, "
          f"epochs={N_EPOCHS}, lr={LR}) ...")
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(students))
    n_train = int(len(students) * TRAIN_FRAC)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    train_students = [students[i] for i in train_idx]
    val_students = [students[i] for i in val_idx]
    print(f"  Train: {len(train_students)}, Val: {len(val_students)}")

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, train_losses, val_losses = train_model(train_students, val_students, vocab_size)

    peak_vram_mb = 0.0
    if DEVICE.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    training_stats = {
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "final_train_loss": float(train_losses[-1]),
        "best_val_loss": float(min(val_losses)),
        "n_epochs_run": len(train_losses),
        "peak_vram_mb": peak_vram_mb,
    }

    # --- Step 3: Recover r_hat (all students) ---
    print(f"\n[Step 3] Recovering r_hat for all {len(students)} students ...")
    r_hat = recover_r_hat(model, students)
    n_finite = int(np.isfinite(r_hat).sum())
    print(f"  r_hat: {n_finite}/{len(students)} finite, "
          f"mean={np.nanmean(r_hat):.4f}, std={np.nanstd(r_hat):.4f}")

    # --- Step 4: Validate ---
    print("\n[Step 4] Validating ...")

    print("  (a) Predictive validity ...")
    pred_val = predictive_validity(students, r_hat, model)
    print(f"      rho={pred_val['rho']:.3f} [{pred_val['ci_lo']:.3f}, {pred_val['ci_hi']:.3f}]"
          f"  n={pred_val['n']}")
    print(f"      neg_control rho={pred_val['neg_control_rho']:.3f} "
          f"[{pred_val['neg_control_ci_lo']:.3f}, {pred_val['neg_control_ci_hi']:.3f}]")

    print("  (b) AFM concurrent validity ...")
    afm_val = afm_concurrent(students, r_hat)
    print(f"      rho={afm_val['rho']:.3f} [{afm_val['ci_lo']:.3f}, {afm_val['ci_hi']:.3f}]"
          f"  n={afm_val['n']}")

    print("  (c) Split-half reliability ...")
    sh_val = split_half_reliability(students, model)
    print(f"      rho={sh_val['rho']:.3f} [{sh_val['ci_lo']:.3f}, {sh_val['ci_hi']:.3f}]"
          f"  n={sh_val['n']}")

    print("  (d) Convergent validity (aligned vs responsive) ...")
    conv_val = convergent_aligned_vs_responsive(students, model, r_hat)
    print(f"      rho={conv_val['rho']:.3f} [{conv_val['ci_lo']:.3f}, {conv_val['ci_hi']:.3f}]"
          f"  n={conv_val['n']}")

    val_results = {
        "predictive": pred_val,
        "afm_concurrent": afm_val,
        "split_half": sh_val,
        "convergent": conv_val,
    }

    # --- Plots ---
    print("\n[Step 5] Generating plots ...")
    make_plots(students, r_hat, val_results)

    # --- Verdict ---
    wall_time = time.time() - wall_t0
    afm_rho = afm_val["rho"]
    pred_rho = pred_val["rho"]
    if np.isfinite(afm_rho) and afm_rho > 0.15:
        verdict = (
            f"POSITIVE: AFM rho={afm_rho:.3f} (>{0.15:.2f} threshold). "
            "Recovered rate captures real learning signal on a repeated-practice dataset."
        )
    elif np.isfinite(pred_rho) and pred_rho > 0.15:
        verdict = (
            f"PARTIAL: Predictive rho={pred_rho:.3f} positive but AFM rho={afm_rho:.3f} "
            "weaker than threshold. Partial learning signal recovery."
        )
    else:
        verdict = (
            f"NULL: AFM rho={afm_rho:.3f}, pred rho={pred_rho:.3f}. "
            "No clear learning signal recovered. Informative null."
        )

    # --- Save JSON ---
    results = {
        "experiment": "E2b",
        "dataset": {
            "source": str(DATA_PATH),
            "format": "DKVMN/DKT triplet format (ASSISTments 2009-2010 updated)",
            **data_stats,
        },
        "cohort": cohort_stats,
        "training": training_stats,
        "r_hat": {
            "n_finite": n_finite,
            "n_total": len(students),
            "mean": float(np.nanmean(r_hat)),
            "std": float(np.nanstd(r_hat)),
            "p25": float(np.nanpercentile(r_hat, 25)),
            "p75": float(np.nanpercentile(r_hat, 75)),
        },
        "validation": val_results,
        "verdict": verdict,
        "wall_time_s": float(wall_time),
        "peak_vram_mb": peak_vram_mb,
    }
    json_path = OUT_DIR / "results_e2b.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON: {json_path}")

    # --- Write RESULTS_E2b.md ---
    def fmt(rho, lo, hi, n):
        return f"rho={rho:.3f} [{lo:.3f}, {hi:.3f}] (n={n})"

    md = f"""# E2b Results: ASSISTments 2009-2010 Repeated-Practice

## Dataset

- **Source**: {DATA_PATH}
- **Format**: DKVMN/DKT triplet (skill-builder variant, ASSISTments 2009-2010 updated)
- **Students total**: {data_stats['n_students_total']}
- **Total interactions**: {data_stats['n_interactions_total']:,}
- **Unique skills**: {data_stats['n_skills']}
- **Fields**: seq_len, skill_ids (1..{data_stats['n_skills']}), responses (0/1)
- **Seq len**: mean {data_stats['seq_len_mean']:.1f}, median {data_stats['seq_len_median']:.1f}, p25/p75 {data_stats['seq_len_p25']:.0f}/{data_stats['seq_len_p75']:.0f}
- **Mean skill repetitions per student**: {data_stats['mean_skill_reps_per_student']:.2f} (key repeated-practice property)

## Cohort (after filtering: >={MIN_SEQ_LEN} interactions + >={MIN_SKILL_REPS} reps on >=1 skill)

- **Students**: {cohort_stats['n_students']}
- **Interactions**: {cohort_stats['n_interactions']:,}
- **Skill vocabulary size**: {cohort_stats['vocab_size']}
- **Unique skills**: {cohort_stats['n_skills']}

## Training

- **Model**: DeepIRTModel(binary, lstm, decouple=True), device={DEVICE}
- **Split**: 80/20 student split ({len(train_students)} / {len(val_students)})
- **Epochs run**: {training_stats['n_epochs_run']} (early stopping patience=4)
- **Final train loss**: {training_stats['final_train_loss']:.4f}
- **Best val loss**: {training_stats['best_val_loss']:.4f}
- **Peak VRAM**: {peak_vram_mb:.1f} MB
- **Wall time total**: {wall_time:.1f}s

## Rate Recovery

- **r_hat finite**: {n_finite}/{len(students)} students
- **Mean r_hat**: {np.nanmean(r_hat):.4f}, std={np.nanstd(r_hat):.4f}

## Validation

### (a) Predictive validity
{fmt(pred_val['rho'], pred_val['ci_lo'], pred_val['ci_hi'], pred_val['n'])}
- Negative control (shuffled r_hat): rho={pred_val['neg_control_rho']:.3f} [{pred_val['neg_control_ci_lo']:.3f}, {pred_val['neg_control_ci_hi']:.3f}]

### (b) AFM concurrent validity (PRIMARY)
{fmt(afm_val['rho'], afm_val['ci_lo'], afm_val['ci_hi'], afm_val['n'])}

### (c) Split-half reliability
{fmt(sh_val['rho'], sh_val['ci_lo'], sh_val['ci_hi'], sh_val['n'])}

### (d) Convergent validity (aligned vs responsive)
{fmt(conv_val['rho'], conv_val['ci_lo'], conv_val['ci_hi'], conv_val['n'])}

## Verdict

**{verdict}**

## Contrast with EdNet-KT1 null

EdNet-KT1 was single-pass (each student sees each item at most once), so no
opportunity-count curves exist and AFM concurrent validity cannot be computed.
ASSISTments 2009-2010 is the canonical AFM/Koedinger repeated-practice setting:
mean {data_stats['mean_skill_reps_per_student']:.1f} skill repetitions per student enables the AFM concurrent test.
"""

    md_path = OUT_DIR / "RESULTS_E2b.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Results MD:   {md_path}")

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict}")
    print(f"Wall time: {wall_time:.1f}s  |  Peak VRAM: {peak_vram_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
