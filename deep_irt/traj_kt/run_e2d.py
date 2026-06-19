"""run_e2d.py -- E2d: GRADED-response variant of E2c on KDD Cup 2010 (Algebra 2008-2009).

WHY E2d: E2c used the binary Correct First Attempt response.  The per-student
learning rate was unreliable (split-half rho=0.17) because the binary signal is
near-saturated (~80% correct = little dynamic range), so the non-circular AFM
concurrent test was uninterpretable.  E0 showed ordinal beats binary for rate
recovery.  KDD's per-step columns Incorrects and Hints give a richer signal we
grade into K=4 ordinal proficiency, restoring dynamic range.

Hypothesis: graded response -> higher rate reliability -> interpretable
(ideally positive) non-circular AFM concurrent.

CHANGES from E2c:
  1. Read extra columns Incorrects and Hints in the KDD parser.
  2. Graded K=4 response from errors = Incorrects + Hints:
       3 if errors==0, 2 if errors==1, 1 if errors in {2,3}, 0 if errors>=4.
  3. decoder='gpcm', n_cats=4. Recover gpcm item params (a, K-1=3 betas).
  4. Existence gate uses GPCM NLL (not binary NLL).
  5. Oracle magnitude via oracle_rate_mle on recovered GPCM params.
  6. Non-circular AFM concurrent: AFM slope on binary-correct ~ opportunity
     within KC, correlated with graded oracle r_hat.
  7. Split-half reliability of r_oracle.
  8. Aligned-vs-responsive convergent.
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
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression

REPO = Path("C:/Users/steph/documents/deep-mirt")
sys.path.insert(0, str(REPO))

from deep_irt.core.model import DeepIRTModel
from deep_irt.traj_synth.metrics import fit_rate, oracle_rate_mle

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR  = REPO / "data" / "kdd"
KDD_FILE  = DATA_DIR / "algebra_2008_2009_train.txt"
OUT_DIR   = REPO / "deep_irt" / "traj_kt"

# Cohort (same as E2c)
MIN_SEQ_LEN    = 200
TARGET_N_HI    = 5000
MAX_ITEM_VOCAB = 50_000

# Training (same as E2c)
MAX_SEQ_LEN    = 500
BATCH_SIZE     = 64
N_EPOCHS       = 30
LR             = 1e-3
RNG_SEED       = 42
N_BOOT         = 1000

# Existence gate
TAIL_FRAC      = 0.20

# Oracle magnitude
MAX_ORACLE     = 5000

# Graded response: K=4 (categories 0..3)
N_CATS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Step 1: Parse + cohort (adapted: adds Incorrects, Hints, graded response)
# ---------------------------------------------------------------------------

def parse_kdd_graded(path: Path) -> tuple[list[dict], dict, dict]:
    """Parse KDD Cup 2010 with Incorrects and Hints for graded response.

    K=4 graded response from errors = Incorrects + Hints:
      3 if errors==0, 2 if errors==1, 1 if errors in {2,3}, 0 if errors>=4.

    Returns (students, item_vocab, kc_vocab).
    Each student dict has:
      item_ids  (np.int32) -- 0-based step-level item index
      responses (np.int32) -- 0..3 graded proficiency
      binary    (np.int32) -- 0/1 Correct First Attempt (for AFM)
      kc_ids    (np.int32) -- first KC index per step
      kc_strs   (np.object_)
      has_kc    (np.bool_)
      opp       (np.int32)
    """
    import pandas as pd

    print(f"[parse] Reading {path} ...")
    t0 = time.time()

    USE_COLS = [
        "Row",
        "Anon Student Id",
        "Problem Name",
        "Step Name",
        "Correct First Attempt",
        "Incorrects",
        "Hints",
        "KC(KTracedSkills)",
        "Opportunity(KTracedSkills)",
    ]

    chunks = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=USE_COLS,
        dtype={
            "Row": "int32",
            "Anon Student Id": str,
            "Problem Name": str,
            "Step Name": str,
            "Correct First Attempt": "float32",
            "Incorrects": "float32",
            "Hints": "float32",
            "KC(KTracedSkills)": str,
            "Opportunity(KTracedSkills)": str,
        },
        low_memory=False,
        chunksize=500_000,
    ):
        # Drop rows where Correct First Attempt is null (those have no outcome)
        chunk = chunk.dropna(subset=["Correct First Attempt"])
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"[parse] Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    df = df.sort_values("Row").reset_index(drop=True)

    # Step-level item key
    df["_step_key"] = df["Problem Name"].str.strip() + "|" + df["Step Name"].str.strip()

    # Binary response (for AFM concurrent test)
    df["_binary"] = df["Correct First Attempt"].astype(np.int32)

    # Graded K=4 response
    df["Incorrects"] = df["Incorrects"].fillna(0.0)
    df["Hints"]      = df["Hints"].fillna(0.0)
    df["_errors"] = (df["Incorrects"] + df["Hints"]).astype(np.int32)

    def grade(e):
        if e == 0:
            return 3
        elif e == 1:
            return 2
        elif e <= 3:
            return 1
        else:
            return 0

    df["_resp"] = df["_errors"].map(grade).astype(np.int32)

    # First KC
    df["_kc_first"] = df["KC(KTracedSkills)"].str.split("~~").str[0].str.strip()
    df["_has_kc"] = df["KC(KTracedSkills)"].notna()

    df["_opp_first"] = (
        df["Opportunity(KTracedSkills)"]
        .fillna("0")
        .str.split("~~")
        .str[0]
        .str.strip()
        .replace("", "0")
    )
    df["_opp_first"] = pd.to_numeric(df["_opp_first"], errors="coerce").fillna(0).astype(np.int32)

    # Item vocab
    step_counts = df["_step_key"].value_counts()
    step_keys = df["_step_key"].unique().tolist()
    print(f"[parse] Raw step vocab size: {len(step_keys):,}")

    if len(step_keys) > MAX_ITEM_VOCAB:
        vocab = {s: i for i, s in enumerate(step_counts.index[:MAX_ITEM_VOCAB - 1].tolist())}
        oov_idx = MAX_ITEM_VOCAB - 1
        df["_item_idx"] = df["_step_key"].map(lambda s: vocab.get(s, oov_idx)).astype(np.int32)
        print(f"[parse] Capped vocab to {MAX_ITEM_VOCAB} (OOV bucket = {oov_idx})")
    else:
        vocab = {s: i for i, s in enumerate(step_counts.index.tolist())}
        df["_item_idx"] = df["_step_key"].map(vocab).astype(np.int32)
        print(f"[parse] Full vocab: {len(vocab):,} steps")

    # KC vocab
    kc_keys = [k for k in df["_kc_first"].unique().tolist() if isinstance(k, str)]
    kc_vocab: dict[str, int] = {k: i for i, k in enumerate(kc_keys)}
    df["_kc_idx"] = df["_kc_first"].map(kc_vocab).fillna(-1).astype(np.int32)

    print(f"[parse] Unique KCs: {len(kc_vocab):,}")

    # Report K=4 histogram
    hist = df["_resp"].value_counts().sort_index()
    total = len(df)
    print(f"[parse] Graded K=4 response histogram:")
    for cat in range(4):
        cnt = int(hist.get(cat, 0))
        print(f"  category {cat}: {cnt:,}  ({100*cnt/total:.1f}%)")

    # Group by student
    print("[parse] Grouping by student ...")
    grouped = df.groupby("Anon Student Id", sort=False)
    students: list[dict] = []
    for sid, grp in grouped:
        items   = grp["_item_idx"].to_numpy(dtype=np.int32)
        resps   = grp["_resp"].to_numpy(dtype=np.int32)
        binary  = grp["_binary"].to_numpy(dtype=np.int32)
        kc_strs = grp["_kc_first"].to_numpy(dtype=object)
        has_kc  = grp["_has_kc"].to_numpy(dtype=bool)
        kc_ids  = grp["_kc_idx"].to_numpy(dtype=np.int32)
        opp_raw = grp["_opp_first"].to_numpy(dtype=np.int32)
        students.append({
            "sid":       str(sid),
            "item_ids":  items,
            "responses": resps,    # graded 0..3
            "binary":    binary,   # 0/1 for AFM
            "kc_ids":    kc_ids,
            "kc_strs":   kc_strs,
            "has_kc":    has_kc,
            "opp":       opp_raw,
        })

    print(f"[parse] Total students: {len(students):,}")

    # Global histogram over all student sequences
    all_resp = np.concatenate([s["responses"] for s in students])
    hist_global = Counter(all_resp.tolist())
    print("[parse] Global graded histogram (student sequences):")
    for cat in range(4):
        cnt = hist_global.get(cat, 0)
        print(f"  category {cat}: {cnt:,}  ({100*cnt/len(all_resp):.1f}%)")

    del df
    return students, vocab, kc_vocab


def filter_cohort(students_all: list[dict], min_seq: int = MIN_SEQ_LEN) -> list[dict]:
    return [s for s in students_all if len(s["item_ids"]) >= min_seq]


def compute_mean_steps_per_kc(students: list[dict]) -> float:
    totals = []
    for s in students:
        kc_ctr: Counter = Counter(s["kc_ids"].tolist())
        totals.extend(kc_ctr.values())
    return float(np.mean(totals)) if totals else 0.0


# ---------------------------------------------------------------------------
# Step 2: Train (GPCM, n_cats=4)
# ---------------------------------------------------------------------------

def pad_sequences(
    seqs: list[np.ndarray],
    max_len: int,
    pad_val: int = 0,
    dtype=np.int32,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(seqs)
    data = np.full((N, max_len), pad_val, dtype=dtype)
    mask = np.zeros((N, max_len), dtype=bool)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        data[i, :L] = s[:L]
        mask[i, :L] = True
    return data, mask


def train_model(
    train_students: list[dict],
    vocab_size: int,
) -> tuple[DeepIRTModel, dict]:
    items_list = [s["item_ids"][:MAX_SEQ_LEN] for s in train_students]
    resp_list  = [s["responses"][:MAX_SEQ_LEN] for s in train_students]

    T_max = MAX_SEQ_LEN
    items_np, mask_np = pad_sequences(items_list, T_max, pad_val=0)
    resp_np, _        = pad_sequences(resp_list,  T_max, pad_val=0)

    items_t = torch.tensor(items_np, dtype=torch.long)
    resp_t  = torch.tensor(resp_np,  dtype=torch.long)
    mask_t  = torch.tensor(mask_np,  dtype=torch.bool)

    model = DeepIRTModel(
        num_items=vocab_size,
        n_cats=N_CATS,          # 4 graded categories
        decoder="gpcm",
        encoder="lstm",
        decouple=True,
        device=DEVICE,
        seed=RNG_SEED,
    )

    import torch.optim as optim

    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(params, lr=LR)

    train_losses: list[float] = []
    t0 = time.time()

    N_tr = len(train_students)
    rng_pt = torch.Generator(device="cpu")
    rng_pt.manual_seed(RNG_SEED + 1)

    for epoch in range(1, N_EPOCHS + 1):
        model.encoder.train()
        model.decoder.train()

        perm = torch.randperm(N_tr, generator=rng_pt)
        epoch_losses = []
        for start in range(0, N_tr, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            b_items = items_t[idx].to(DEVICE)
            b_resp  = resp_t[idx].to(DEVICE)
            b_mask  = mask_t[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = model._compute_loss(b_items, b_resp, mask=b_mask)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        epoch_loss = float(np.mean(epoch_losses))
        train_losses.append(epoch_loss)
        print(f"  [train] epoch {epoch:2d}/{N_EPOCHS}  loss={epoch_loss:.4f}")

    wall = time.time() - t0
    peak_vram_mb = 0.0
    if DEVICE.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6

    print(f"  [train] done in {wall:.1f}s  peak_vram={peak_vram_mb:.0f} MB")

    model.encoder.eval()
    print("\n[train] Example aligned-theta curves (first 3 train students):")
    for i in range(min(3, len(train_students))):
        s = train_students[i]
        L = min(len(s["item_ids"]), MAX_SEQ_LEN)
        ids_t  = torch.tensor(s["item_ids"][:L], dtype=torch.long).unsqueeze(0).to(DEVICE)
        rsp_t  = torch.tensor(s["responses"][:L], dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            th, _ = model.encoder.aligned_theta_and_state(ids_t, rsp_t)
        th_np = th.squeeze(0).cpu().numpy()
        steps = list(range(0, L, max(1, L // 10)))
        vals  = [f"{th_np[j]:.3f}" for j in steps]
        print(f"  student {i}: L={L}  theta@[{','.join(str(j) for j in steps)}] = [{','.join(vals)}]")

    return model, {
        "train_losses": [float(x) for x in train_losses],
        "final_train_loss": float(train_losses[-1]),
        "wall_time_s": float(wall),
        "peak_vram_mb": float(peak_vram_mb),
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "vocab_size": vocab_size,
        "n_cats": N_CATS,
        "decoder": "gpcm",
    }


# ---------------------------------------------------------------------------
# Step 3a: Existence gate (GPCM NLL)
# ---------------------------------------------------------------------------

def gpcm_nll(theta: float, resps: np.ndarray, a: np.ndarray, betas: np.ndarray) -> float:
    """NLL of GPCM for a constant theta scalar.

    resps: (T,) int in 0..K-1
    a: (T,)
    betas: (T, K-1)
    """
    T = len(resps)
    theta_vec = np.full(T, theta)
    step = a[:, None] * (theta_vec[:, None] - betas)      # (T, K-1)
    cum  = np.cumsum(step, axis=1)
    lognum = np.concatenate([np.zeros((T, 1)), cum], axis=1)  # (T, K)
    lognum = lognum - lognum.max(axis=1, keepdims=True)
    logZ   = np.log(np.exp(lognum).sum(axis=1))
    ll = lognum[np.arange(T), resps] - logZ
    return float(-ll.sum())


def mle_theta_gpcm(resps: np.ndarray, a: np.ndarray, betas: np.ndarray) -> float:
    result = minimize_scalar(
        lambda t: gpcm_nll(t, resps, a, betas),
        bounds=(-6.0, 6.0),
        method="bounded",
    )
    return float(result.x)


def existence_gate(
    all_students: list[dict],
    model: DeepIRTModel,
    item_params: dict,
    tail_frac: float = TAIL_FRAC,
) -> tuple[dict, np.ndarray]:
    """Held-out existence gate using GPCM NLL.

    delta_NLL[i] = NLL_static[i] - NLL_dynamic[i]  (positive = dynamic wins).
    """
    a_hat = item_params["alpha"].astype(np.float64)       # (Q,)
    betas_hat = item_params["beta"].astype(np.float64)    # (Q, K-1)
    if betas_hat.ndim == 1:
        betas_hat = betas_hat[:, None]

    N = len(all_students)
    delta_nll   = np.full(N, np.nan)
    nll_static  = np.full(N, np.nan)
    nll_dynamic = np.full(N, np.nan)

    model.encoder.eval()
    device = model.device

    print(f"[existence] Computing per-student held-out GPCM NLL (N={N}) ...")
    for i, s in enumerate(all_students):
        items = s["item_ids"]
        resps = s["responses"]   # graded 0..3
        L = min(len(items), MAX_SEQ_LEN)
        if L < 40:
            continue

        T_fit  = max(int(L * (1 - tail_frac)), 20)
        T_tail = L - T_fit
        if T_tail < 5:
            continue

        fit_items  = items[:T_fit].astype(np.int64)
        fit_resps  = resps[:T_fit].astype(np.int32)
        tail_items = items[T_fit:L].astype(np.int64)
        tail_resps = resps[T_fit:L].astype(np.int32)

        a_fit   = a_hat[fit_items]
        b_fit   = betas_hat[fit_items]      # (T_fit, K-1)
        a_tail  = a_hat[tail_items]
        b_tail  = betas_hat[tail_items]     # (T_tail, K-1)

        # Static: MLE constant theta on full fit window
        theta_static = mle_theta_gpcm(fit_resps, a_fit, b_fit)
        nll_s = gpcm_nll(theta_static, tail_resps, a_tail, b_tail) / T_tail
        nll_static[i] = nll_s

        # Dynamic: aligned theta at last fit step
        ids_t = torch.tensor(fit_items, dtype=torch.long).unsqueeze(0).to(device)
        rsp_t = torch.tensor(fit_resps.astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th, _ = model.encoder.aligned_theta_and_state(ids_t, rsp_t)
        theta_dyn = float(th.squeeze(0)[-1].cpu().numpy())
        nll_d = gpcm_nll(theta_dyn, tail_resps, a_tail, b_tail) / T_tail
        nll_dynamic[i] = nll_d

        delta_nll[i] = nll_s - nll_d

        if i % 500 == 0:
            print(f"  ... {i}/{N}")

    ok = np.isfinite(delta_nll)
    d_ok = delta_nll[ok]

    mean_d   = float(np.mean(d_ok))
    frac_pos = float((d_ok > 0).mean())

    rng = np.random.default_rng(RNG_SEED + 10)
    boot_means = np.array([
        np.mean(rng.choice(d_ok, len(d_ok), replace=True))
        for _ in range(N_BOOT)
    ])
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    try:
        stat_w, p_w = wilcoxon(d_ok, alternative="greater")
    except Exception:
        stat_w, p_w = float("nan"), float("nan")

    result = {
        "n_students": int(ok.sum()),
        "mean_delta_nll": mean_d,
        "mean_delta_nll_ci_lo": ci_lo,
        "mean_delta_nll_ci_hi": ci_hi,
        "frac_pos": frac_pos,
        "wilcoxon_stat": float(stat_w),
        "wilcoxon_p": float(p_w),
        "tail_frac": tail_frac,
        "comparator": "full_window_static_theta_gpcm",
    }
    return result, delta_nll


# ---------------------------------------------------------------------------
# Step 3b: Oracle magnitude (GPCM)
# ---------------------------------------------------------------------------

def compute_oracle_rates(
    all_students: list[dict],
    item_params: dict,
    max_students: int = MAX_ORACLE,
) -> np.ndarray:
    """Fit the full learning curve theta_0, theta_inf, r per student under GPCM."""
    a_hat     = item_params["alpha"].astype(np.float64)   # (Q,)
    betas_hat = item_params["beta"].astype(np.float64)    # (Q, K-1)
    if betas_hat.ndim == 1:
        betas_hat = betas_hat[:, None]

    N = min(len(all_students), max_students)
    r_oracle = np.full(N, np.nan)

    print(f"[oracle] Fitting per-student oracle_rate_mle GPCM (N={N}) ...")
    for i in range(N):
        s = all_students[i]
        items = s["item_ids"]
        resps = s["responses"]   # graded 0..3
        L = min(len(items), MAX_SEQ_LEN)
        if L < 20:
            continue

        a_i = a_hat[items[:L].astype(np.int64)]              # (L,)
        b_i = betas_hat[items[:L].astype(np.int64), :]       # (L, K-1)
        resp_i = resps[:L].astype(np.int64)

        try:
            r_oracle[i] = oracle_rate_mle(resp_i, a_i, b_i)
        except Exception:
            pass

        if i % 500 == 0:
            print(f"  ... {i}/{N}")

    ok = np.isfinite(r_oracle)
    print(f"[oracle] r_oracle: {ok.sum()}/{N} finite, "
          f"mean={np.nanmean(r_oracle):.4f}, "
          f"median={np.nanmedian(r_oracle):.4f}, "
          f"p90={np.nanpercentile(r_oracle, 90):.4f}")
    return r_oracle


# ---------------------------------------------------------------------------
# Step 3c: Non-circular AFM concurrent
# ---------------------------------------------------------------------------
# NOTE: AFM slope is computed on BINARY correct (Correct First Attempt), not
# on the graded response, because AFM is a classic binary-correct model.
# The oracle r_hat is from the GPCM graded response.
# This is non-circular: encoder item key = Problem|Step, AFM KC = KTracedSkills.

def afm_slope_per_student(s: dict) -> float:
    """Logistic slope binary_correct ~ opportunity_count_within_KC."""
    L = min(len(s["item_ids"]), MAX_SEQ_LEN)
    has_kc = s["has_kc"][:L]
    kc_ids = s["kc_ids"][:L]
    resps  = s["binary"][:L].astype(float)   # binary correct

    if not has_kc.any():
        return np.nan

    kc_ids_kc = kc_ids[has_kc]
    resps_kc  = resps[has_kc]

    opp_kc = np.empty(len(kc_ids_kc), dtype=float)
    kc_ctr: dict[int, int] = {}
    for t, kc in enumerate(kc_ids_kc.tolist()):
        kc_ctr[kc] = kc_ctr.get(kc, 0)
        opp_kc[t] = kc_ctr[kc]
        kc_ctr[kc] += 1

    slopes = []
    weights = []
    for kc in np.unique(kc_ids_kc):
        mask = kc_ids_kc == kc
        if mask.sum() < 3:
            continue
        x_kc = opp_kc[mask].reshape(-1, 1)
        y_kc = resps_kc[mask]
        if y_kc.std() < 1e-6:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lr = LogisticRegression(max_iter=300, C=1e6, solver="lbfgs")
                lr.fit(x_kc, y_kc)
                slopes.append(float(lr.coef_[0, 0]))
                weights.append(int(mask.sum()))
            except Exception:
                pass
    if not slopes:
        return np.nan
    w_arr = np.array(weights, dtype=float)
    return float(np.average(slopes, weights=w_arr))


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(ok.sum())
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


def afm_concurrent(
    all_students: list[dict],
    r_oracle: np.ndarray,
) -> tuple[dict, np.ndarray]:
    N = min(len(all_students), len(r_oracle))
    afm_slopes = np.full(N, np.nan)
    print(f"[afm] Computing AFM slopes on binary-correct (N={N}) ...")
    for i in range(N):
        if not np.isfinite(r_oracle[i]):
            continue
        afm_slopes[i] = afm_slope_per_student(all_students[i])
        if i % 500 == 0:
            print(f"  ... {i}/{N}")

    rho, lo, hi, n = _bootstrap_spearman(afm_slopes, r_oracle, seed=RNG_SEED + 20)
    n_finite_afm = int(np.isfinite(afm_slopes).sum())
    print(f"[afm] AFM slopes finite: {n_finite_afm}/{N}")
    print(f"[afm] Spearman(AFM_slope, r_oracle) = {rho:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
    return {
        "rho": rho, "ci_lo": lo, "ci_hi": hi,
        "n": n, "n_afm_finite": n_finite_afm,
    }, afm_slopes


# ---------------------------------------------------------------------------
# Step 3d: Secondary checks
# ---------------------------------------------------------------------------

def split_half_reliability(
    all_students: list[dict],
    item_params: dict,
    max_students: int = MAX_ORACLE,
) -> dict:
    """Split-half reliability of r_oracle (odd vs even steps, GPCM)."""
    a_hat     = item_params["alpha"].astype(np.float64)
    betas_hat = item_params["beta"].astype(np.float64)
    if betas_hat.ndim == 1:
        betas_hat = betas_hat[:, None]

    N = min(len(all_students), max_students)
    r_odd  = np.full(N, np.nan)
    r_even = np.full(N, np.nan)

    print(f"[split-half] Computing odd/even oracle rates (N={N}) ...")
    for i in range(N):
        s = all_students[i]
        items = s["item_ids"]
        resps = s["responses"]   # graded
        L = min(len(items), MAX_SEQ_LEN)
        if L < 40:
            continue

        for label, idx in [("odd", np.arange(0, L, 2)), ("even", np.arange(1, L, 2))]:
            a_i = a_hat[items[idx].astype(np.int64)]
            b_i = betas_hat[items[idx].astype(np.int64), :]
            resp_i = resps[idx].astype(np.int64)
            if len(resp_i) < 10:
                continue
            try:
                r = oracle_rate_mle(resp_i, a_i, b_i)
            except Exception:
                r = np.nan
            if label == "odd":
                r_odd[i] = r
            else:
                r_even[i] = r

    rho, lo, hi, n = _bootstrap_spearman(r_odd, r_even, seed=RNG_SEED + 30)
    print(f"[split-half] Spearman(r_odd, r_even) = {rho:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
    return {"rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n}


def compute_aligned_rates(
    all_students: list[dict],
    model: DeepIRTModel,
    max_students: int = MAX_ORACLE,
) -> np.ndarray:
    N = min(len(all_students), max_students)
    r_hat = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    for i in range(N):
        s = all_students[i]
        items = s["item_ids"]
        resps = s["responses"]
        L = min(len(items), MAX_SEQ_LEN)
        if L < 20:
            continue
        ids_t = torch.tensor(items[:L].astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        rsp_t = torch.tensor(resps[:L].astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th, _ = model.encoder.aligned_theta_and_state(ids_t, rsp_t)
        th_np = th.squeeze(0).cpu().numpy().astype(float)
        r, _, _ = fit_rate(th_np, robust=True, smooth=1, t=np.arange(L, dtype=float))
        r_hat[i] = r

    return r_hat


def convergent_aligned_vs_responsive(
    all_students: list[dict],
    model: DeepIRTModel,
    r_hat_aligned: np.ndarray,
    max_students: int = MAX_ORACLE,
) -> dict:
    N = min(len(all_students), max_students, len(r_hat_aligned))
    r_resp = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    print(f"[convergent] Computing responsive-theta rates (N={N}) ...")
    for i in range(N):
        s = all_students[i]
        items = s["item_ids"]
        resps = s["responses"]
        L = min(len(items), MAX_SEQ_LEN)
        if L < 20:
            continue
        ids_t = torch.tensor(items[:L].astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        rsp_t = torch.tensor(resps[:L].astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th = model.encoder.encode(ids_t, rsp_t)
        th_np = th.squeeze(0).cpu().numpy().astype(float)
        r, _, _ = fit_rate(th_np, robust=True, smooth=1, t=np.arange(L, dtype=float))
        r_resp[i] = r

    rho, lo, hi, n = _bootstrap_spearman(r_hat_aligned, r_resp[:N], seed=RNG_SEED + 40)
    print(f"[convergent] Spearman(r_aligned, r_responsive) = {rho:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
    return {"rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n}


# ---------------------------------------------------------------------------
# Model-free diagnostic
# ---------------------------------------------------------------------------

def within_student_accuracy_gain(all_students: list[dict]) -> dict:
    """Model-free check on GRADED response: does mean category rise over the sequence?

    We compute first-quartile vs last-quartile mean graded response (0..3).
    Also computes binary correct rate for context.
    """
    deltas_graded = []
    deltas_binary = []
    first_graded  = []
    last_graded   = []
    binary_rates  = []
    for s in all_students:
        L = min(len(s["responses"]), MAX_SEQ_LEN)
        rg = s["responses"][:L].astype(float)
        rb = s["binary"][:L].astype(float)
        q = L // 4
        if q < 5:
            continue
        fg = float(rg[:q].mean())
        lg = float(rg[-q:].mean())
        first_graded.append(fg)
        last_graded.append(lg)
        deltas_graded.append(lg - fg)
        deltas_binary.append(float(rb[-q:].mean()) - float(rb[:q].mean()))
        binary_rates.append(float(rb.mean()))

    deltas_graded = np.array(deltas_graded)
    deltas_binary = np.array(deltas_binary)

    rng = np.random.default_rng(RNG_SEED + 50)
    boot = np.array([
        np.mean(rng.choice(deltas_graded, len(deltas_graded), replace=True))
        for _ in range(N_BOOT)
    ])
    return {
        "n": int(len(deltas_graded)),
        "first_quartile_graded": float(np.mean(first_graded)),
        "last_quartile_graded": float(np.mean(last_graded)),
        "mean_gain_graded": float(deltas_graded.mean()),
        "mean_gain_graded_ci_lo": float(np.percentile(boot, 2.5)),
        "mean_gain_graded_ci_hi": float(np.percentile(boot, 97.5)),
        "frac_improving_graded": float((deltas_graded > 0).mean()),
        "mean_gain_binary": float(deltas_binary.mean()),
        "overall_binary_rate": float(np.mean(binary_rates)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    all_students: list[dict],
    r_oracle: np.ndarray,
    delta_nll: np.ndarray,
    afm_result: dict,
    afm_slopes: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    plots_dir = out_dir / "plots_e2d"
    plots_dir.mkdir(exist_ok=True)
    paths = []

    # 1. delta_NLL distribution
    ok = np.isfinite(delta_nll)
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = delta_nll[ok]
    bins = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 50)
    ax.hist(np.clip(vals, bins[0], bins[-1]), bins=bins, color="#4c78a8", edgecolor="white", lw=0.4)
    ax.axvline(0, color="red", linestyle="--", lw=1.2)
    ax.set_xlabel("delta_NLL GPCM (static - dynamic)  [nats/step]")
    ax.set_ylabel("Count")
    mean_d = float(np.mean(vals))
    frac_p = float((vals > 0).mean())
    ax.set_title(f"E2d Existence gate  mean={mean_d:.4f}  frac>0={frac_p:.3f}  n={ok.sum()}")
    fig.tight_layout()
    p = plots_dir / "existence_gate.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. Oracle rate distribution
    rok = np.isfinite(r_oracle)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r_oracle[rok], bins=40, color="#e45756", edgecolor="white", lw=0.4)
    ax.set_xlabel("Oracle learning rate $\\hat{r}$ (GPCM)")
    ax.set_ylabel("Count")
    ax.set_title(f"E2d Oracle rate distribution  n={rok.sum()}")
    fig.tight_layout()
    p = plots_dir / "oracle_rate_dist.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3. AFM slope vs oracle rate
    N = min(len(afm_slopes), len(r_oracle))
    m_ok = np.isfinite(afm_slopes[:N]) & np.isfinite(r_oracle[:N])
    if m_ok.sum() > 5:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(afm_slopes[:N][m_ok], r_oracle[:N][m_ok], alpha=0.3, s=8, color="#54a24b")
        coef = np.polyfit(afm_slopes[:N][m_ok], r_oracle[:N][m_ok], 1)
        x_line = np.linspace(afm_slopes[:N][m_ok].min(), afm_slopes[:N][m_ok].max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line), "k-", lw=1.5)
        rho_a = afm_result["rho"]
        ax.set_xlabel("AFM slope (binary correct ~ opp, per KC)")
        ax.set_ylabel("Oracle $\\hat{r}$ GPCM")
        ax.set_title(f"E2d Non-circular AFM concurrent  $\\rho$={rho_a:.3f}")
        fig.tight_layout()
        p = plots_dir / "afm_concurrent.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        paths.append(p)

    # 4. Graded response histogram
    all_resp = np.concatenate([s["responses"][:min(len(s["responses"]), MAX_SEQ_LEN)]
                               for s in all_students])
    hist = Counter(all_resp.tolist())
    fig, ax = plt.subplots(figsize=(5, 4))
    cats = [0, 1, 2, 3]
    counts = [hist.get(c, 0) for c in cats]
    ax.bar(cats, counts, color=["#e45756", "#f58518", "#54a24b", "#4c78a8"])
    ax.set_xlabel("Graded response (0=struggled, 3=mastered first try)")
    ax.set_ylabel("Count")
    total = sum(counts)
    ax.set_title(f"E2d Graded K=4 histogram  total={total:,}")
    for c, cnt in zip(cats, counts):
        ax.text(c, cnt * 1.01, f"{100*cnt/total:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    p = plots_dir / "graded_histogram.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_t0 = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    print("=" * 65)
    print("E2d: KDD Cup 2010 -- GRADED response (K=4) variant of E2c")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    # --- Step 1: Parse ---
    print("\n[Step 1] Parsing KDD Cup 2010 (graded response) ...")
    students_all, item_vocab, kc_vocab = parse_kdd_graded(KDD_FILE)
    vocab_size = max(len(item_vocab), MAX_ITEM_VOCAB)

    print(f"\n  RAW COHORT:")
    print(f"  Students total         : {len(students_all):,}")
    seq_lens_all = [len(s["item_ids"]) for s in students_all]
    print(f"  Seq len median / p90   : {float(np.median(seq_lens_all)):.0f} / "
          f"{float(np.percentile(seq_lens_all, 90)):.0f}")
    print(f"  Item vocab size        : {vocab_size:,}")
    print(f"  Distinct KCs           : {len(kc_vocab):,}")

    students = filter_cohort(students_all, MIN_SEQ_LEN)
    del students_all
    mean_steps_kc = compute_mean_steps_per_kc(students)
    lengths = [len(s["item_ids"]) for s in students]
    print(f"\n  FILTERED COHORT (>= {MIN_SEQ_LEN} steps):")
    print(f"  Students               : {len(students):,}")
    print(f"  Seq len median / p90   : {float(np.median(lengths)):.0f} / {float(np.percentile(lengths,90)):.0f}")
    print(f"  Mean steps/KC/student  : {mean_steps_kc:.2f}")

    if len(students) < 200:
        print("ERROR: fewer than 200 dense students; aborting.")
        return
    print(f"\n  [SELF-GATE PASS] {len(students)} students >= {MIN_SEQ_LEN} steps.")

    if len(students) > TARGET_N_HI:
        rng_np = np.random.default_rng(RNG_SEED)
        idx = rng_np.choice(len(students), TARGET_N_HI, replace=False)
        students = [students[i] for i in idx]
        print(f"  Subsampled to {len(students)} students (TARGET_N_HI={TARGET_N_HI})")

    # --- Step 2: Train ---
    print(f"\n[Step 2] Training GPCM (K=4) on 80/20 student split ...")
    rng_np = np.random.default_rng(RNG_SEED)
    perm = rng_np.permutation(len(students))
    n_train = int(len(students) * 0.8)
    train_students = [students[i] for i in perm[:n_train]]
    val_students   = [students[i] for i in perm[n_train:]]
    print(f"  Train: {len(train_students)}  Val: {len(val_students)}")

    model, train_stats = train_model(train_students, vocab_size)

    losses = train_stats["train_losses"]
    if not np.isfinite(losses[-1]):
        print("ERROR: training loss is NaN; aborting.")
        return
    print(f"\n  [SELF-GATE PASS] Training complete. "
          f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # --- Recover item parameters ---
    print("\n[Step 2b] Recovering GPCM item parameters ...")
    items_list = [s["item_ids"][:MAX_SEQ_LEN] for s in train_students]
    resp_list  = [s["responses"][:MAX_SEQ_LEN] for s in train_students]

    items_np, mask_np = pad_sequences(items_list, MAX_SEQ_LEN, pad_val=0)
    resp_np, _        = pad_sequences(resp_list,  MAX_SEQ_LEN, pad_val=0)
    items_t = torch.tensor(items_np, dtype=torch.long).to(DEVICE)
    resp_t  = torch.tensor(resp_np,  dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        item_params = model.recover_item_params(items_t, resp_t)

    print(f"  alpha shape: {item_params['alpha'].shape}  "
          f"min={item_params['alpha'].min():.3f}  max={item_params['alpha'].max():.3f}  "
          f"mean={item_params['alpha'].mean():.3f}")
    print(f"  beta shape:  {item_params['beta'].shape}  "
          f"min={item_params['beta'].min():.3f}  max={item_params['beta'].max():.3f}  "
          f"mean={item_params['beta'].mean():.3f}")

    # --- Step 3-pre: Model-free trend ---
    print(f"\n[Step 3-pre] Model-free within-student graded response gain ...")
    gain_result = within_student_accuracy_gain(students)
    print(f"  First-quartile graded  : {gain_result['first_quartile_graded']:.3f}")
    print(f"  Last-quartile graded   : {gain_result['last_quartile_graded']:.3f}")
    print(f"  Mean gain graded       : {gain_result['mean_gain_graded']:+.4f}  "
          f"[{gain_result['mean_gain_graded_ci_lo']:.4f}, {gain_result['mean_gain_graded_ci_hi']:.4f}]")
    print(f"  Frac improving (graded): {gain_result['frac_improving_graded']:.3f}")
    print(f"  Mean gain binary       : {gain_result['mean_gain_binary']:+.4f}")
    print(f"  Overall binary rate    : {gain_result['overall_binary_rate']:.3f}")

    # --- Step 3a: Existence gate ---
    print(f"\n[Step 3a] Existence gate GPCM (tail_frac={TAIL_FRAC}) ...")
    exist_result, delta_nll = existence_gate(students, model, item_params)
    print(f"\n  EXISTENCE GATE:")
    print(f"  N students             : {exist_result['n_students']}")
    print(f"  mean delta_NLL         : {exist_result['mean_delta_nll']:.4f}  "
          f"[{exist_result['mean_delta_nll_ci_lo']:.4f}, {exist_result['mean_delta_nll_ci_hi']:.4f}]")
    print(f"  frac delta_NLL > 0     : {exist_result['frac_pos']:.3f}")
    print(f"  Wilcoxon p (>0)        : {exist_result['wilcoxon_p']:.3e}")

    ci_clears_zero = exist_result["mean_delta_nll_ci_lo"] > 0
    sig  = exist_result["wilcoxon_p"] < 0.05
    pos  = exist_result["mean_delta_nll"] > 0
    if pos and sig and ci_clears_zero:
        exist_verdict = ("PASS: trajectory model beats static at held-out GPCM prediction "
                         "(Wilcoxon p < 0.05 AND mean delta_NLL CI clears zero).")
    elif pos and sig:
        exist_verdict = ("WEAK PASS: dynamic > static (Wilcoxon p < 0.05) but effect at "
                         "the measurement floor (mean delta_NLL CI crosses zero).")
    elif pos:
        exist_verdict = "MARGINAL: positive mean delta_NLL but not significant (p >= 0.05)."
    else:
        exist_verdict = "FAIL: static model not beaten by trajectory model."
    print(f"  Existence verdict      : {exist_verdict}")

    # --- Step 3b: Oracle magnitude ---
    print(f"\n[Step 3b] Oracle magnitude GPCM (N up to {MAX_ORACLE}) ...")
    r_oracle = compute_oracle_rates(students, item_params, max_students=MAX_ORACLE)

    # --- Step 3c: Non-circular AFM concurrent ---
    print(f"\n[Step 3c] Non-circular AFM concurrent ...")
    print("  (encoder item key = Problem|Step; AFM KC = KTracedSkills -- non-circular)")
    print("  (AFM slope on BINARY correct; oracle r_hat from GPCM graded response)")
    afm_result, afm_slopes = afm_concurrent(students, r_oracle)
    print(f"  Spearman(AFM_slope, r_oracle) = {afm_result['rho']:.3f} "
          f"[{afm_result['ci_lo']:.3f}, {afm_result['ci_hi']:.3f}]  n={afm_result['n']}")

    # --- Step 3d: Secondary checks ---
    print(f"\n[Step 3d] Secondary checks ...")
    print("  (a) Split-half reliability of r_oracle ...")
    sh_result = split_half_reliability(students, item_params, max_students=MAX_ORACLE)

    print("  (b) Aligned vs responsive convergent ...")
    r_hat_aligned = compute_aligned_rates(students, model, max_students=MAX_ORACLE)
    conv_result = convergent_aligned_vs_responsive(
        students, model, r_hat_aligned, max_students=MAX_ORACLE
    )

    # --- Plots ---
    print("\n[Step 4] Generating plots ...")
    plot_paths = make_plots(students, r_oracle, delta_nll, afm_result, afm_slopes, OUT_DIR)
    for p in plot_paths:
        print(f"  Saved: {p}")

    # --- Verdict ---
    wall_time = time.time() - wall_t0
    peak_vram_mb = 0.0
    if DEVICE.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6

    trend_exists  = gain_result["mean_gain_graded_ci_lo"] > 0
    exist_strong  = pos and sig and ci_clears_zero
    exist_weak    = pos and sig and not exist_strong
    afm_pass      = np.isfinite(afm_result["rho"]) and afm_result["rho"] > 0.10
    rate_reliable = np.isfinite(sh_result["rho"]) and sh_result["rho"] > 0.40

    # E2c baseline for comparison table
    e2c_sh_rho    = 0.17
    e2c_afm_rho   = -0.005
    e2c_delta_nll = 0.0008

    if exist_strong and afm_pass and rate_reliable:
        final_verdict = (
            "STRONG POSITIVE: existence gate PASS (significant and material) + "
            "non-circular AFM concurrent positive + reliable r_oracle. "
            f"delta_NLL mean={exist_result['mean_delta_nll']:.4f} "
            f"[{exist_result['mean_delta_nll_ci_lo']:.4f}, {exist_result['mean_delta_nll_ci_hi']:.4f}], "
            f"AFM Spearman={afm_result['rho']:.3f} [{afm_result['ci_lo']:.3f}, {afm_result['ci_hi']:.3f}], "
            f"split-half reliability={sh_result['rho']:.2f}. "
            "Graded scoring rescued the human-front rate: higher dynamic range delivered "
            "reliable rate estimation and an interpretable non-circular AFM concurrent."
        )
    elif (exist_strong or exist_weak) and afm_pass and rate_reliable:
        final_verdict = (
            f"QUALIFIED POSITIVE: graded scoring lifted rate reliability "
            f"(split-half rho={sh_result['rho']:.2f} vs E2c {e2c_sh_rho:.2f}) and produced a "
            f"positive non-circular AFM concurrent (rho={afm_result['rho']:.3f} vs E2c {e2c_afm_rho:.3f}). "
            f"Existence gate: {exist_verdict}. "
            "Graded scoring rescued the human-front magnitude signal."
        )
    elif rate_reliable and not afm_pass:
        final_verdict = (
            f"MIXED: rate is now reliable (split-half rho={sh_result['rho']:.2f} vs E2c {e2c_sh_rho:.2f}) "
            f"but non-circular AFM concurrent remains null (rho={afm_result['rho']:.3f}). "
            f"Graded scoring improved rate reliability, but the AFM concurrent still does not confirm "
            f"the non-circular magnitude signal. Existence gate: {exist_verdict}."
        )
    elif not rate_reliable:
        reliab_vs_e2c = ("improved vs" if sh_result["rho"] > e2c_sh_rho else "similar to or worse than")
        final_verdict = (
            f"NULL on MAGNITUDE: rate reliability {reliab_vs_e2c} E2c "
            f"(split-half rho={sh_result['rho']:.2f} vs E2c {e2c_sh_rho:.2f}). "
            f"Non-circular AFM concurrent: rho={afm_result['rho']:.3f} "
            f"[{afm_result['ci_lo']:.3f}, {afm_result['ci_hi']:.3f}] vs E2c {e2c_afm_rho:.3f}. "
            f"Existence gate: {exist_verdict}. "
            "Graded scoring did NOT rescue the human rate: even with K=4 dynamic range, "
            "the per-student oracle rate is not sufficiently self-consistent to yield an "
            "interpretable AFM concurrent. The limiting factor may be insufficient sequence "
            "length or structural misfit of the exponential learning curve to KDD practice data."
        )
    else:
        final_verdict = (
            f"INCONCLUSIVE: split-half rho={sh_result['rho']:.2f}, "
            f"AFM rho={afm_result['rho']:.3f}, existence verdict: {exist_verdict}."
        )

    print(f"\n{'=' * 65}")
    print(f"FINAL VERDICT: {final_verdict}")
    print(f"Wall time: {wall_time:.1f}s  |  Peak VRAM: {peak_vram_mb:.0f} MB")
    print("=" * 65)

    # E2c vs E2d comparison table
    print(f"\n{'=' * 65}")
    print("E2c vs E2d COMPARISON TABLE")
    print(f"{'=' * 65}")
    print(f"{'Metric':<35} {'E2c (binary)':>15} {'E2d (graded K=4)':>18}")
    print(f"{'-'*35} {'-'*15} {'-'*18}")
    print(f"{'Response type':<35} {'binary (0/1)':>15} {'graded (0..3)':>18}")
    print(f"{'Split-half reliability (r_oracle)':<35} {e2c_sh_rho:>15.3f} {sh_result['rho'] if np.isfinite(sh_result['rho']) else float('nan'):>18.3f}")
    print(f"{'Non-circular AFM concurrent rho':<35} {e2c_afm_rho:>15.3f} {afm_result['rho'] if np.isfinite(afm_result['rho']) else float('nan'):>18.3f}")
    print(f"{'Existence gate mean delta_NLL':<35} {e2c_delta_nll:>15.4f} {exist_result['mean_delta_nll']:>18.4f}")
    print(f"{'Existence gate Wilcoxon p':<35} {'<0.05':>15} {exist_result['wilcoxon_p']:>18.3e}")
    print("=" * 65)

    # --- Save JSON ---
    results = {
        "experiment": "E2d",
        "dataset": "KDD Cup 2010, algebra_2008_2009_train",
        "response_type": f"graded K={N_CATS} (errors = Incorrects+Hints)",
        "grading_rule": "3 if errors==0, 2 if errors==1, 1 if errors in {2,3}, 0 if errors>=4",
        "cohort": {
            "n_students_filtered": len(students),
            "min_seq_len": MIN_SEQ_LEN,
            "seq_len_median": float(np.median(lengths)),
            "seq_len_p90": float(np.percentile(lengths, 90)),
            "seq_len_max": int(max(lengths)),
            "item_vocab_size": vocab_size,
            "n_kcs": len(kc_vocab),
            "mean_steps_per_kc_per_student": float(mean_steps_kc),
        },
        "training": train_stats,
        "validation": {
            "within_student_graded_gain": gain_result,
            "existence_gate": exist_result,
            "existence_verdict": exist_verdict,
            "oracle_magnitude": {
                "n_finite": int(np.isfinite(r_oracle).sum()),
                "mean": float(np.nanmean(r_oracle)),
                "median": float(np.nanmedian(r_oracle)),
                "p90": float(np.nanpercentile(r_oracle, 90)),
            },
            "afm_concurrent_noncircular": afm_result,
            "split_half_reliability": sh_result,
            "convergent_aligned_vs_responsive": conv_result,
        },
        "comparison_e2c": {
            "e2c_sh_rho": e2c_sh_rho,
            "e2c_afm_rho": e2c_afm_rho,
            "e2c_delta_nll": e2c_delta_nll,
            "e2d_sh_rho": float(sh_result["rho"]) if np.isfinite(sh_result["rho"]) else None,
            "e2d_afm_rho": float(afm_result["rho"]) if np.isfinite(afm_result["rho"]) else None,
            "e2d_delta_nll": float(exist_result["mean_delta_nll"]),
        },
        "verdict": final_verdict,
        "wall_time_s": float(wall_time),
        "peak_vram_mb": float(peak_vram_mb),
    }
    json_path = OUT_DIR / "results_e2d.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON: {json_path}")

    _write_results_md(results, exist_result, exist_verdict, afm_result, sh_result,
                      conv_result, gain_result, r_oracle, delta_nll,
                      e2c_sh_rho, e2c_afm_rho, e2c_delta_nll, final_verdict)
    print(f"Results MD:  {OUT_DIR / 'RESULTS_E2d.md'}")


def _write_results_md(
    results, exist_result, exist_verdict, afm_result, sh_result, conv_result,
    gain_result, r_oracle, delta_nll,
    e2c_sh_rho, e2c_afm_rho, e2c_delta_nll, final_verdict,
):
    wall_time = results["wall_time_s"]
    peak_vram = results["peak_vram_mb"]
    n_students = results["cohort"]["n_students_filtered"]
    train_stats = results["training"]
    oracle_stats = results["validation"]["oracle_magnitude"]

    ok_d = np.isfinite(delta_nll)
    d_ok = delta_nll[ok_d]

    def fmt(rho, lo, hi, n):
        return f"rho={rho:.3f} [{lo:.3f}, {hi:.3f}] (n={n})"

    sh_rho = sh_result["rho"]
    afm_rho = afm_result["rho"]
    delta_nll_mean = exist_result["mean_delta_nll"]

    graded_gain = gain_result["mean_gain_graded"]
    graded_gain_lo = gain_result["mean_gain_graded_ci_lo"]
    graded_gain_hi = gain_result["mean_gain_graded_ci_hi"]

    md = f"""# E2d Results: KDD Cup 2010 -- Graded Response (K=4) Variant of E2c

## Motivation

E2c used the binary Correct First Attempt response (E2c split-half rho=0.17,
AFM rho=-0.005). The near-saturated binary signal (~{gain_result['overall_binary_rate']:.0%} correct)
left insufficient dynamic range for reliable per-student rate estimation.
E0 showed ordinal beats binary for rate recovery. E2d grades each step into
K=4 ordinal proficiency from the Incorrects and Hints columns:

  errors = Incorrects + Hints
  category 3: errors==0 (mastered first try)
  category 2: errors==1 (one slip)
  category 1: errors in {{2, 3}} (struggled)
  category 0: errors>=4 (severe difficulty)

Hypothesis: graded response -> more dynamic range -> higher rate reliability
-> interpretable (ideally positive) non-circular AFM concurrent.

## Dataset and Cohort

- Source: algebra_2008_2009_train.txt
- Item key: Problem Name + '|' + Step Name (step-level, non-circular with KC).
- KC label: KC(KTracedSkills) (for AFM only).

| Stat | Value |
|---|---|
| Students (>= {MIN_SEQ_LEN} steps) | {n_students:,} |
| Seq len median / p90 / max | {results['cohort']['seq_len_median']:.0f} / {results['cohort']['seq_len_p90']:.0f} / {results['cohort']['seq_len_max']:,} |
| Item vocab size (step-level) | {results['cohort']['item_vocab_size']:,} |
| Distinct KCs | {results['cohort']['n_kcs']:,} |
| Mean steps per KC per student | {results['cohort']['mean_steps_per_kc_per_student']:.2f} |

## Training

- Model: DeepIRTModel(n_cats={N_CATS}, decoder='gpcm', encoder='lstm', decouple=True)
- Seq cap: {MAX_SEQ_LEN} steps, batch={BATCH_SIZE}, epochs={train_stats['n_epochs']}, lr={LR}
- 80/20 student split
- Final train loss: {train_stats['final_train_loss']:.4f}
- Wall time (total): {wall_time:.1f}s
- Peak VRAM: {peak_vram:.0f} MB

## Validation

### (a0) Model-free trend (graded response)

| Metric | Value |
|---|---|
| First-quartile mean graded | {gain_result['first_quartile_graded']:.3f} |
| Last-quartile mean graded | {gain_result['last_quartile_graded']:.3f} |
| Mean gain graded (last - first) | {graded_gain:+.4f} [{graded_gain_lo:.4f}, {graded_gain_hi:.4f}] |
| Fraction improving (graded) | {gain_result['frac_improving_graded']:.3f} |
| Mean gain binary (for context) | {gain_result['mean_gain_binary']:+.4f} |
| Overall binary correct rate | {gain_result['overall_binary_rate']:.3f} |

### (a) Existence Gate (GPCM NLL)

Holds out last {int(TAIL_FRAC*100)}% of each student. Dynamic predictor:
aligned theta at last fit step. Static null: MLE constant theta (GPCM).
delta_NLL = NLL_static - NLL_dynamic (positive = dynamic wins).

| Metric | Value |
|---|---|
| N students | {exist_result['n_students']} |
| mean delta_NLL | {delta_nll_mean:.4f} [{exist_result['mean_delta_nll_ci_lo']:.4f}, {exist_result['mean_delta_nll_ci_hi']:.4f}] |
| frac delta_NLL > 0 | {exist_result['frac_pos']:.3f} |
| Wilcoxon p (one-sided, > 0) | {exist_result['wilcoxon_p']:.3e} |

**Existence verdict**: {exist_verdict}

### (b) Oracle Magnitude (GPCM)

Per-student oracle learning rate via oracle_rate_mle (GPCM a, K-1=3 betas).

| Metric | Value |
|---|---|
| N finite | {oracle_stats['n_finite']} |
| mean r_oracle | {oracle_stats['mean']:.4f} |
| median r_oracle | {oracle_stats['median']:.4f} |
| p90 r_oracle | {oracle_stats['p90']:.4f} |

### (c) Non-Circular AFM Concurrent

AFM slope on BINARY correct ~ opportunity within KC (logistic per-KC, weighted).
Oracle r_hat from GPCM graded response.
NON-CIRCULAR: encoder item key = Problem|Step; AFM KC = KTracedSkills.

{fmt(afm_result['rho'], afm_result['ci_lo'], afm_result['ci_hi'], afm_result['n'])}

### (d) Split-Half Reliability of r_oracle

{fmt(sh_result['rho'], sh_result['ci_lo'], sh_result['ci_hi'], sh_result['n'])}

### (e) Convergent: Aligned vs Responsive Theta

{fmt(conv_result['rho'], conv_result['ci_lo'], conv_result['ci_hi'], conv_result['n'])}

## E2c vs E2d Comparison

| Metric | E2c (binary) | E2d (graded K=4) |
|---|---|---|
| Response type | binary (0/1) | graded (0..3) |
| Split-half reliability (r_oracle) | {e2c_sh_rho:.3f} | {sh_rho:.3f} |
| Non-circular AFM concurrent rho | {e2c_afm_rho:.3f} | {afm_rho:.3f} |
| Existence gate mean delta_NLL | {e2c_delta_nll:.4f} | {delta_nll_mean:.4f} |
| Existence gate Wilcoxon p | <0.05 | {exist_result['wilcoxon_p']:.2e} |

## Verdict

**{final_verdict}**

Wall time: {wall_time:.1f}s  |  Peak VRAM: {peak_vram:.0f} MB
"""
    md_path = OUT_DIR / "RESULTS_E2d.md"
    with open(md_path, "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
