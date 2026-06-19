"""run_poscontrol.py -- Positive control for the E2/E2b predictive-validity test.

On real data (EdNet, ASSISTments) the predictive-validity test
  Spearman(r_hat_firsthalf, delta_acc = late_acc - early_acc)
came back null to negative (-0.04, -0.08).  Before concluding the method
fails on humans we must check whether the metric itself is valid when a
real learning rate is present.

Suspected confound: a fast learner saturates in the first half of the
sequence, so their delta_acc (late minus early) is small or even
negative, while a slow learner keeps improving and shows large delta_acc.
This makes rate NEGATIVELY correlated with delta_acc by construction.

This script verifies the confound on synthetic data where ground-truth
rates are known:

    A. corr(r_hat, r_true)               -- recovery works?  ~0.4 expected
    B. corr(r_hat_firsthalf, delta_acc)  -- the E2/E2b metric as applied
    C. corr(r_true, delta_acc)           -- does TRUE rate predict late gain?
    D. tercile diagnostic                -- saturation by rate group
    E. alternative criteria              -- corr(r_hat, r_true) for
                                            identifiable learners, and
                                            corr(r_hat, total_acc_gain)

All Spearman correlations with bootstrap 95% CI (1000 resamples).
Outputs: RESULTS_poscontrol.md, poscontrol_results.json, and two PNG
plots (delta_acc vs r_true; r_hat vs r_true).

Run from repo root:
    python -m deep_irt.traj_synth.run_poscontrol
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from deep_irt.core.model import DeepIRTModel
from deep_irt.traj_synth.data_gen import generate_learning_curves
from deep_irt.traj_synth.metrics import fit_rate


# ---------------------------------------------------------------------------
# Bootstrap Spearman CI
# ---------------------------------------------------------------------------

def spearman_ci(x, y, n_boot=1000, alpha=0.05, seed=0):
    """Spearman r with bootstrap 95% CI.  Returns (rho, lo, hi, n_valid)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) < 1e-9 or np.std(yi) < 1e-9:
            continue
        boot.append(float(spearmanr(xi, yi)[0]))
    if len(boot) < 10:
        return rho, float("nan"), float("nan"), n
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return rho, lo, hi, n


def fmt_corr(rho, lo, hi, n):
    if not np.isfinite(rho):
        return "nan"
    return f"{rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (n={n})"


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_model(data, device, n_epochs=300, seed=0):
    item_ids = data["seq_item_ids"].to(device)
    responses = data["seq_responses"].to(device)
    model = DeepIRTModel(
        num_items=data["n_items"],
        emb_dim=8,
        hidden_dim=32,
        n_cats=data["K"],
        decoder="gpcm",
        encoder="lstm",
        decouple=True,
        device=device,
        seed=seed,
    )
    print("  Training DeepIRTModel (300 epochs) ...")
    model.fit(item_ids, responses, n_epochs=n_epochs, lr=1e-2, verbose=True)
    return model


# ---------------------------------------------------------------------------
# Theta extraction (prediction-aligned, the E2 recipe)
# ---------------------------------------------------------------------------

def get_aligned_theta(model, item_ids, responses):
    """Return prediction-aligned theta (N, T) on cpu."""
    enc = model.encoder
    enc.eval()
    ii = item_ids.to(model.device)
    rr = responses.to(model.device)
    with torch.no_grad():
        th, _ = enc.aligned_theta_and_state(ii, rr)
    return th.cpu().numpy()


# ---------------------------------------------------------------------------
# Fit rates from a theta array (N, T) -- thin wrapper
# ---------------------------------------------------------------------------

def recover_rates_array(theta_arr):
    """Per-respondent fit_rate(robust=True, smooth=1), returns (N,) array."""
    T = theta_arr.shape[1]
    t = np.arange(T, dtype=float)
    out = np.array(
        [fit_rate(theta_arr[i], robust=True, smooth=1, t=t)[0]
         for i in range(theta_arr.shape[0])],
        dtype=float,
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Positive control | device={device}")

    # ---- 1. Generate synthetic data ----------------------------------------
    print("\n[1] Generating synthetic learning curves ...")
    data = generate_learning_curves(
        n_items=120, n_respondents=600, seq_len=80, K=4, seed=0
    )
    r_true = data["rate_true"].numpy()          # (600,)
    theta_true = data["theta_true_traj"].numpy()  # (600, 80)
    item_ids = data["seq_item_ids"]             # (600, 80)
    responses = data["seq_responses"]           # (600, 80)
    N, T = item_ids.shape
    mid = T // 2  # 40

    print(f"  N={N}, T={T}, K={data['K']}, mid={mid}")
    print(f"  r_true range: [{r_true.min():.3f}, {r_true.max():.3f}], "
          f"median={np.median(r_true):.3f}")

    # ---- 2. Train and recover full-sequence theta --------------------------
    print("\n[2] Training model and recovering theta ...")
    model = train_model(data, device, n_epochs=300, seed=0)
    theta_hat = get_aligned_theta(model, item_ids, responses)  # (N, T)

    # Full-sequence r_hat
    r_hat = recover_rates_array(theta_hat)   # (N,)

    # First-half r_hat (first mid steps only)
    r_hat_first = recover_rates_array(theta_hat[:, :mid])  # (N,)

    print(f"  r_hat finite: {np.isfinite(r_hat).sum()}/{N}")
    print(f"  r_hat_first finite: {np.isfinite(r_hat_first).sum()}/{N}")

    # ---- 3. Compute delta_acc and total_acc_gain ---------------------------
    resp_np = responses.numpy()  # (N, T), 0-indexed categories
    # Normalise responses to [0,1] (max category = K-1)
    K = data["K"]
    resp01 = resp_np / (K - 1)

    acc_first = resp01[:, :mid].mean(axis=1)   # (N,) mean 1st half
    acc_second = resp01[:, mid:].mean(axis=1)  # (N,) mean 2nd half
    delta_acc = acc_second - acc_first          # late minus early (E2/E2b criterion)
    total_gain = resp01[:, -10:].mean(axis=1) - resp01[:, :10].mean(axis=1)

    print(f"  delta_acc: mean={delta_acc.mean():.3f}, std={delta_acc.std():.3f}")
    print(f"  total_gain: mean={total_gain.mean():.3f}, std={total_gain.std():.3f}")

    # ---- 4. Compute all correlations with bootstrap CIs --------------------
    print("\n[3/4] Computing correlations ...")

    # A. Recovery: corr(r_hat, r_true)
    A_rho, A_lo, A_hi, A_n = spearman_ci(r_hat, r_true)
    print(f"  A. corr(r_hat, r_true):              {fmt_corr(A_rho, A_lo, A_hi, A_n)}")

    # B. E2/E2b metric: corr(r_hat_first, delta_acc)
    B_rho, B_lo, B_hi, B_n = spearman_ci(r_hat_first, delta_acc)
    print(f"  B. corr(r_hat_first, delta_acc):     {fmt_corr(B_rho, B_lo, B_hi, B_n)}")

    # B_perm. Permuted-order control (shuffle r_hat_first)
    rng_perm = np.random.default_rng(42)
    r_perm = r_hat_first[rng_perm.permutation(N)]
    Bp_rho, Bp_lo, Bp_hi, Bp_n = spearman_ci(r_perm, delta_acc, seed=42)
    print(f"  B_perm (shuffled r_hat):             {fmt_corr(Bp_rho, Bp_lo, Bp_hi, Bp_n)}")

    # C. Metric validity: corr(r_true, delta_acc)
    C_rho, C_lo, C_hi, C_n = spearman_ci(r_true, delta_acc)
    print(f"  C. corr(r_true, delta_acc):          {fmt_corr(C_rho, C_lo, C_hi, C_n)}")

    # D. Tercile diagnostic
    tercile_edges = np.percentile(r_true, [0, 33.3, 66.7, 100])
    tercile_labels = ["slow (low r)", "mid", "fast (high r)"]
    tercile_stats = []
    print("\n[D] Tercile diagnostic (by r_true):")
    print(f"  {'group':18s} {'n':>5s} {'r_true mean':>12s} {'delta_acc mean':>15s} {'total_gain mean':>16s}")
    for ti in range(3):
        lo_edge = tercile_edges[ti]
        hi_edge = tercile_edges[ti + 1]
        mask = (r_true >= lo_edge) & (r_true <= hi_edge)
        g_r = r_true[mask].mean()
        g_da = delta_acc[mask].mean()
        g_tg = total_gain[mask].mean()
        g_n = int(mask.sum())
        print(f"  {tercile_labels[ti]:18s} {g_n:5d} {g_r:12.3f} {g_da:15.4f} {g_tg:16.4f}")
        tercile_stats.append({
            "group": tercile_labels[ti],
            "n": g_n,
            "r_true_mean": float(g_r),
            "delta_acc_mean": float(g_da),
            "total_gain_mean": float(g_tg),
        })

    # E. Alternative criteria
    # E1: corr(r_hat, r_true) restricted to identifiable regime (r_true <= median)
    low_mid_mask = r_true <= np.median(r_true)
    E1_rho, E1_lo, E1_hi, E1_n = spearman_ci(
        r_hat[low_mid_mask], r_true[low_mid_mask]
    )
    print(f"\n  E1. corr(r_hat, r_true) [low/mid rate]: {fmt_corr(E1_rho, E1_lo, E1_hi, E1_n)}")

    # E2: corr(r_hat, total_acc_gain)
    E2_rho, E2_lo, E2_hi, E2_n = spearman_ci(r_hat, total_gain)
    print(f"  E2. corr(r_hat, total_acc_gain):        {fmt_corr(E2_rho, E2_lo, E2_hi, E2_n)}")

    # E3: corr(r_true, total_acc_gain)  -- sanity that total gain is ground-truth-sensitive
    E3_rho, E3_lo, E3_hi, E3_n = spearman_ci(r_true, total_gain)
    print(f"  E3. corr(r_true, total_acc_gain):       {fmt_corr(E3_rho, E3_lo, E3_hi, E3_n)}")

    # ---- 5. Verdict --------------------------------------------------------
    recovery_works = np.isfinite(A_rho) and A_rho > 0.2
    metric_fires = np.isfinite(B_rho) and B_rho > 0.15
    metric_valid = np.isfinite(C_rho) and C_rho > 0.15

    if recovery_works and not metric_fires and not metric_valid:
        verdict = (
            "METRIC ARTIFACT. Recovery works (A positive) but the predictive-"
            "validity metric does not fire (B null/negative) because the true "
            "rate does not predict late-half gain (C null/negative). The "
            "saturation confound is confirmed: fast learners plateau in the "
            "first half and show little late gain, inverting the expected "
            "correlation. The null/negative predictive validity in E2/E2b "
            "is a metric artifact, NOT evidence that recovery fails."
        )
    elif recovery_works and metric_fires and metric_valid:
        verdict = (
            "ALL POSITIVE. Recovery works (A), the predictive-validity metric "
            "fires (B), and the true rate predicts late gain (C). The E2/E2b "
            "null must reflect a genuine failure on real data, not a metric "
            "artifact."
        )
    elif not recovery_works:
        verdict = (
            "RECOVERY FAILURE. Even on synthetic data recovery is weak (A "
            "negative), indicating a model or training problem that must be "
            "fixed before interpreting human-data results."
        )
    elif recovery_works and metric_fires and not metric_valid:
        verdict = (
            "AMBIGUOUS. Recovery works (A) and B fires, but C is near zero, "
            "meaning r_hat predicts delta_acc even though r_true does not. "
            "This points to r_hat capturing something other than true rate."
        )
    else:
        verdict = (
            f"MIXED. A={A_rho:.3f}, B={B_rho:.3f}, C={C_rho:.3f}. Inspect the "
            "tercile diagnostic and full numbers for interpretation."
        )

    print(f"\n[VERDICT] {verdict}")

    # ---- 6. Save JSON results ----------------------------------------------
    results = {
        "config": {
            "n_items": int(data["n_items"]),
            "n_respondents": N,
            "seq_len": T,
            "K": int(K),
            "mid": mid,
            "n_epochs": 300,
            "seed": 0,
        },
        "correlations": {
            "A_recovery": {
                "description": "corr(r_hat, r_true) -- full sequence",
                "rho": A_rho, "ci_lo": A_lo, "ci_hi": A_hi, "n": A_n,
            },
            "B_predictive_validity": {
                "description": "corr(r_hat_first, delta_acc) -- E2/E2b metric",
                "rho": B_rho, "ci_lo": B_lo, "ci_hi": B_hi, "n": B_n,
            },
            "B_perm_control": {
                "description": "corr(shuffled_r_hat_first, delta_acc)",
                "rho": Bp_rho, "ci_lo": Bp_lo, "ci_hi": Bp_hi, "n": Bp_n,
            },
            "C_metric_validity": {
                "description": "corr(r_true, delta_acc) -- does TRUE rate predict late gain?",
                "rho": C_rho, "ci_lo": C_lo, "ci_hi": C_hi, "n": C_n,
            },
            "E1_recovery_lowmid": {
                "description": "corr(r_hat, r_true) restricted to low/mid rate learners",
                "rho": E1_rho, "ci_lo": E1_lo, "ci_hi": E1_hi, "n": E1_n,
            },
            "E2_rhat_total_gain": {
                "description": "corr(r_hat, total_acc_gain) -- alternative criterion",
                "rho": E2_rho, "ci_lo": E2_lo, "ci_hi": E2_hi, "n": E2_n,
            },
            "E3_rtrue_total_gain": {
                "description": "corr(r_true, total_acc_gain) -- ground-truth sanity",
                "rho": E3_rho, "ci_lo": E3_lo, "ci_hi": E3_hi, "n": E3_n,
            },
        },
        "tercile_diagnostic": tercile_stats,
        "verdict": verdict,
    }

    json_path = os.path.join(out_dir, "poscontrol_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Wrote {json_path}")

    # ---- 7. Plots ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Plot 1: delta_acc vs r_true (the confound diagnostic)
    ax = axes[0]
    scatter = ax.scatter(r_true, delta_acc, c=r_true, cmap="viridis",
                         s=14, alpha=0.55, linewidths=0)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("r_true (ground-truth learning rate)", fontsize=11)
    ax.set_ylabel("delta_acc (late half - early half)", fontsize=11)
    ax.set_title(
        f"Saturation confound: r_true vs delta_acc\n"
        f"Spearman rho = {C_rho:+.3f}  [{C_lo:+.3f}, {C_hi:+.3f}]",
        fontsize=10,
    )
    plt.colorbar(scatter, ax=ax, label="r_true")

    # Plot 2: r_hat vs r_true (recovery quality)
    ax = axes[1]
    ok = np.isfinite(r_hat) & np.isfinite(r_true)
    ax.scatter(r_true[ok], r_hat[ok], s=14, alpha=0.45, color="steelblue",
               linewidths=0)
    ax.set_xlabel("r_true (ground-truth learning rate)", fontsize=11)
    ax.set_ylabel("r_hat (recovered rate, full sequence)", fontsize=11)
    ax.set_title(
        f"Recovery quality: r_hat vs r_true\n"
        f"Spearman rho = {A_rho:+.3f}  [{A_lo:+.3f}, {A_hi:+.3f}]",
        fontsize=10,
    )

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "poscontrol_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Wrote {plot_path}")

    # ---- 8. Write RESULTS_poscontrol.md ------------------------------------
    _write_md(here, results, A_rho, A_lo, A_hi, A_n,
              B_rho, B_lo, B_hi, B_n,
              Bp_rho, Bp_lo, Bp_hi,
              C_rho, C_lo, C_hi, C_n,
              E1_rho, E1_lo, E1_hi, E1_n,
              E2_rho, E2_lo, E2_hi, E2_n,
              E3_rho, E3_lo, E3_hi, E3_n,
              tercile_stats, verdict, mid, T, N)

    print("\nDone.")


def _write_md(
    here,
    results,
    A_rho, A_lo, A_hi, A_n,
    B_rho, B_lo, B_hi, B_n,
    Bp_rho, Bp_lo, Bp_hi,
    C_rho, C_lo, C_hi, C_n,
    E1_rho, E1_lo, E1_hi, E1_n,
    E2_rho, E2_lo, E2_hi, E2_n,
    E3_rho, E3_lo, E3_hi, E3_n,
    tercile_stats, verdict, mid, T, N,
):
    md_path = os.path.join(here, "RESULTS_poscontrol.md")

    def _s(rho, lo, hi):
        if not np.isfinite(rho):
            return "nan"
        return f"{rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]"

    # Tercile table rows
    terc_rows = ""
    for ts in tercile_stats:
        terc_rows += (
            f"| {ts['group']:18s} | {ts['n']:4d} | {ts['r_true_mean']:11.3f} | "
            f"{ts['delta_acc_mean']:14.4f} | {ts['total_gain_mean']:15.4f} |\n"
        )

    text = f"""\
# Positive Control: E2/E2b Predictive-Validity Metric

**Purpose.** On real data (E2 EdNet, E2b ASSISTments) the test
Spearman(r_hat_firsthalf, delta_acc = late_acc - early_acc) returned
null to negative (-0.04, -0.08). This positive control checks whether
the metric itself is valid when a true learning rate is present, using
synthetic data with known ground-truth rates.

**Suspected confound.** A fast learner saturates in the first half of
the sequence. Their late-half accuracy change (delta_acc) is small or
negative because the curve has already flattened. A slow learner keeps
improving into the second half, yielding large delta_acc. This makes
rate NEGATIVELY correlated with delta_acc by construction, regardless
of recovery quality.

---

## Setup

| Parameter | Value |
|---|---|
| n_items | 120 |
| n_respondents | {N} |
| seq_len | {T} |
| K (GPCM categories) | 4 |
| split midpoint | {mid} |
| n_epochs | 300 |
| model | DeepIRTModel(gpcm, lstm, decouple=True) |
| seed | 0 |

---

## Results

All correlations are Spearman rho with bootstrap 95% CI (1000 resamples).

### A. Recovery (clean target)

`corr(r_hat, r_true)` -- does the model recover the true rate over the
full sequence?

**rho = {_s(A_rho, A_lo, A_hi)}  (n={A_n})**

Expected ~0.4 from prior E0 runs. A positive result here confirms that
the recovery machinery works on this synthetic dataset.

### B. The E2/E2b Predictive-Validity Metric

`corr(r_hat_firsthalf, delta_acc)` -- does the first-half recovered rate
predict late-half accuracy gain, exactly as E2/E2b tested?

**rho = {_s(B_rho, B_lo, B_hi)}  (n={B_n})**

Permuted-order control (shuffled r_hat): {_s(Bp_rho, Bp_lo, Bp_hi)}

If this is near zero or negative despite recovery working (A positive),
the metric is confounded.

### C. Metric Validity Check (the crux)

`corr(r_true, delta_acc)` -- does the GROUND-TRUTH rate predict late-half
accuracy gain?

**rho = {_s(C_rho, C_lo, C_hi)}  (n={C_n})**

This is the decisive test. If even the true rate does not predict
delta_acc, the predictive-validity metric is ill-posed regardless of
recovery quality.

### D. Tercile Diagnostic (saturation effect)

Learners binned by r_true tercile. Fast learners should show smaller
delta_acc (saturation) and larger total_gain.

| group              |    n | r_true mean | delta_acc mean | total_gain mean |
|:-------------------|-----:|------------:|---------------:|----------------:|
{terc_rows}
### E. Alternative Criteria

These criteria avoid the saturation confound.

| Criterion | rho | 95% CI | n |
|---|---|---|---|
| E1: corr(r_hat, r_true) [low/mid rate learners] | {E1_rho:+.3f} | [{E1_lo:+.3f}, {E1_hi:+.3f}] | {E1_n} |
| E2: corr(r_hat, total_acc_gain) | {E2_rho:+.3f} | [{E2_lo:+.3f}, {E2_hi:+.3f}] | {E2_n} |
| E3: corr(r_true, total_acc_gain) [sanity] | {E3_rho:+.3f} | [{E3_lo:+.3f}, {E3_hi:+.3f}] | {E3_n} |

---

## Verdict

{verdict}

### Implications

1. **Does recovery work on synthetic data (A)?**
   A rho of {A_rho:+.3f} {"confirms" if A_rho > 0.2 else "DOES NOT confirm"}
   that the encoder recovers the true rate on synthetic sequences of
   length {T}.

2. **Does the predictive-validity metric fire (B)?**
   B = {B_rho:+.3f}. {"It does not fire, matching the E2/E2b null." if B_rho < 0.1 else "It fires."}

3. **Is the metric itself valid -- does the TRUE rate predict late gain (C)?**
   C = {C_rho:+.3f}. {"No. The true rate does not predict delta_acc, confirming the saturation confound is structural." if C_rho < 0.1 else "Yes."}

**If A is positive but B and C are null/negative,** the correct
interpretation is that the predictive-validity test (Spearman of
r_hat_firsthalf vs delta_acc) is confounded by saturation and is the
wrong validation criterion. The human-front negative predictive
validity in E2/E2b is largely a metric artifact. The better validation
criteria are (a) recovery-style corr(r_hat, r_true) when ground truth
is available, or (b) corr(r_hat, total_acc_gain) over the whole
sequence when it is not.

---

*Generated by `deep_irt/traj_synth/run_poscontrol.py`.*
*Plots: `outputs/poscontrol_plots.png`. JSON: `outputs/poscontrol_results.json`.*
"""
    with open(md_path, "w") as f:
        f.write(text)
    print(f"  Wrote {md_path}")


if __name__ == "__main__":
    main()
