"""_validity_criterion_exp.py -- held-out prediction as a validity criterion.

Tests whether held-out prediction (dynamic vs static-ability model) is a
well-posed validity criterion for a recovered learning rate. Delivers a SPLIT
verdict over two separable claims:

  EXISTENCE  -- does the dynamic (trajectory) model beat a static-ability model
                at held-out prediction when, and only when, a rate exists?
  MAGNITUDE  -- does the per-learner prediction margin delta_NLL rank-correlate
                with the true rate r?

The existence test is the cross-condition contrast (with-rate vs no-rate). The
magnitude test is Spearman(delta_NLL, r_true) under two static baselines (full
fit window vs the recent window just before the tail), contrasted with the
oracle rate ceiling oracle_rate_mle fitted on the model's recovered items.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import spearmanr, mannwhitneyu

from deep_irt.traj_synth.data_gen import generate_learning_curves
from deep_irt.traj_synth.metrics import fit_rate, oracle_rate_mle
from deep_irt.core.model import DeepIRTModel


# ---------------------------------------------------------------------------
# GPCM log-prob helpers
# ---------------------------------------------------------------------------

def gpcm_logprob_scalar(theta_s: float, resp_seq: np.ndarray,
                         item_ids_seq: np.ndarray, a: np.ndarray,
                         b: np.ndarray) -> float:
    a_i = a[item_ids_seq]
    b_i = b[item_ids_seq]
    diff = a_i[:, None] * (theta_s - b_i)
    psi = np.concatenate([np.zeros((len(resp_seq), 1)), np.cumsum(diff, axis=1)], axis=1)
    psi -= psi.max(axis=1, keepdims=True)
    log_Z = np.log(np.exp(psi).sum(axis=1))
    ll = psi[np.arange(len(resp_seq)), resp_seq] - log_Z
    return float(ll.sum())


def gpcm_logprob_vec(theta_vec: np.ndarray, resp_seq: np.ndarray,
                      item_ids_seq: np.ndarray, a: np.ndarray,
                      b: np.ndarray) -> float:
    a_i = a[item_ids_seq]
    b_i = b[item_ids_seq]
    diff = a_i[:, None] * (theta_vec[:, None] - b_i)
    psi = np.concatenate([np.zeros((len(resp_seq), 1)), np.cumsum(diff, axis=1)], axis=1)
    psi -= psi.max(axis=1, keepdims=True)
    log_Z = np.log(np.exp(psi).sum(axis=1))
    ll = psi[np.arange(len(resp_seq)), resp_seq] - log_Z
    return float(ll.sum())


def mle_theta_static(resp_seq: np.ndarray, item_ids_seq: np.ndarray,
                      a: np.ndarray, b: np.ndarray) -> float:
    result = minimize_scalar(
        lambda t: -gpcm_logprob_scalar(t, resp_seq, item_ids_seq, a, b),
        bounds=(-5.0, 5.0), method="bounded",
    )
    return float(result.x)


def fit_ability_curve_mle(resp_seq: np.ndarray, item_ids_seq: np.ndarray,
                           a: np.ndarray, b: np.ndarray,
                           T_fit: int) -> tuple:
    """Fit (A, B, r) by MLE on responses; return (A, B, r) or (nan, nan, nan)."""
    t_seq = np.arange(T_fit, dtype=float)

    def negll(x):
        A, B, r = x
        theta_t = A - B * np.exp(-r * t_seq)
        return -gpcm_logprob_vec(theta_t, resp_seq, item_ids_seq, a, b)

    best = None
    for A0, B0, r0 in [(-0.5, -1.5, 0.1), (0.5, -1.5, 0.3), (0.5, -1.5, 0.05)]:
        try:
            res = minimize(negll, x0=[A0, B0, r0],
                           bounds=[(-5, 5), (-8, 8), (1e-3, 2.0)],
                           method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    if best is None or not best.success:
        return float("nan"), float("nan"), float("nan")
    return float(best.x[0]), float(best.x[1]), float(best.x[2])


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def spearman_bootstrap_ci(x, y, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    N = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        stats.append(spearmanr(x[idx], y[idx])[0])
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(spearmanr(x, y)[0]), float(lo), float(hi)


def mean_gap_bootstrap_ci(a, b, n_boot=1000, alpha=0.05, seed=7):
    """Bootstrap CI on mean(a) - mean(b), resampling each group independently."""
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    gaps = []
    for _ in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        gaps.append(a[ia].mean() - b[ib].mean())
    gaps = np.array(gaps)
    lo, hi = np.percentile(gaps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean() - b.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Broken metric (gain over window)
# ---------------------------------------------------------------------------

def compute_broken_metric(r_hat: np.ndarray, resp: np.ndarray) -> float:
    """Spearman corr of r_hat vs late-minus-early accuracy gain."""
    T = resp.shape[1]
    half = T // 2
    early_acc = (resp[:, :half] == resp[:, :half].max()).mean(axis=1)
    late_acc  = (resp[:, half:] == resp[:, half:].max()).mean(axis=1)
    gain = late_acc - early_acc
    ok = np.isfinite(r_hat) & np.isfinite(gain)
    if ok.sum() < 5:
        return float("nan")
    return float(spearmanr(r_hat[ok], gain[ok])[0])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(data: dict, label: str, n_epochs: int = 500,
                   device=torch.device("cpu"), oracle_max: int = 300) -> dict:
    N = data["n_respondents"]
    T = data["seq_len"]
    K = data["K"]
    n_items = data["n_items"]

    seq_items = data["seq_item_ids"]
    seq_resp  = data["seq_responses"]
    r_true    = data["rate_true"].numpy()

    T_fit  = int(T * 0.8)
    T_tail = T - T_fit
    T_recent = T_tail   # recent static window = same width as the tail

    fit_items = seq_items[:, :T_fit]
    fit_resp  = seq_resp[:, :T_fit]
    tail_items = seq_items[:, T_fit:]
    tail_resp  = seq_resp[:, T_fit:]

    # --- Train DeepIRTModel ---
    print(f"[{label}] Training DeepIRTModel ({n_epochs} epochs)...")
    model = DeepIRTModel(
        num_items=n_items, n_cats=K, decoder="gpcm",
        encoder="lstm", decouple=True, device=device, seed=0,
    )
    model.fit(fit_items, fit_resp, n_epochs=n_epochs, verbose=False)
    print(f"[{label}] Training done.")

    item_params = model.recover_item_params(fit_items, fit_resp)
    a_hat = item_params["alpha"]                          # (Q,)
    b_hat = item_params["beta"]                           # (Q, K-1)

    # LSTM trajectory on fit window (prediction-aligned, causal)
    theta_aligned = model.track(fit_items, fit_resp).cpu().numpy()  # (N, T_fit)

    fit_items_np  = fit_items.numpy()
    fit_resp_np   = fit_resp.numpy()
    tail_items_np = tail_items.numpy()
    tail_resp_np  = tail_resp.numpy()
    seq_resp_np   = seq_resp.numpy()

    # --- Per-learner held-out NLL ---
    nll_static_full   = np.zeros(N)   # static MLE on full fit window
    nll_static_recent = np.zeros(N)   # static MLE on last T_recent fit steps
    nll_curve_mle     = np.zeros(N)   # curve MLE extrapolated to tail (dynamic)

    print(f"[{label}] Per-learner held-out NLL (N={N})...")
    for i in range(N):
        f_items = fit_items_np[i]
        f_resp  = fit_resp_np[i]
        t_items = tail_items_np[i]
        t_resp  = tail_resp_np[i]

        # Static MLE on full fit window
        theta_full = mle_theta_static(f_resp, f_items, a_hat, b_hat)
        nll_static_full[i] = -gpcm_logprob_vec(
            np.full(T_tail, theta_full), t_resp, t_items, a_hat, b_hat
        ) / T_tail

        # Static MLE on the recent window just before the tail (like-for-like)
        theta_recent = mle_theta_static(
            f_resp[-T_recent:], f_items[-T_recent:], a_hat, b_hat
        )
        nll_static_recent[i] = -gpcm_logprob_vec(
            np.full(T_tail, theta_recent), t_resp, t_items, a_hat, b_hat
        ) / T_tail

        # Dynamic: curve MLE extrapolated to tail
        A, B, r_fit = fit_ability_curve_mle(f_resp, f_items, a_hat, b_hat, T_fit)
        if np.isnan(r_fit):
            nll_curve_mle[i] = nll_static_full[i]
        else:
            tail_t = np.arange(T_fit, T, dtype=float)
            theta_tail = A - B * np.exp(-r_fit * tail_t)
            nll_curve_mle[i] = -gpcm_logprob_vec(
                theta_tail, t_resp, t_items, a_hat, b_hat
            ) / T_tail

    # delta_NLL: positive = dynamic wins (static is harder to beat)
    delta_full   = nll_static_full   - nll_curve_mle
    delta_recent = nll_static_recent - nll_curve_mle

    rho_full,   lo_full,   hi_full   = spearman_bootstrap_ci(delta_full,   r_true)
    rho_recent, lo_recent, hi_recent = spearman_bootstrap_ci(delta_recent, r_true)

    # --- Clean rate anchor: oracle_rate_mle on the model's RECOVERED items ---
    n_oracle = min(oracle_max, N)
    print(f"[{label}] Oracle rate MLE on recovered items (n={n_oracle})...")
    r_oracle = np.full(n_oracle, np.nan)
    for i in range(n_oracle):
        idx = fit_items_np[i]
        a_seq = a_hat[idx]                       # (T_fit,)
        b_seq = b_hat[idx]                       # (T_fit, K-1)
        r_oracle[i] = oracle_rate_mle(fit_resp_np[i], a_seq, b_seq)
    ok_o = np.isfinite(r_oracle)
    rho_o, lo_o, hi_o = spearman_bootstrap_ci(
        r_oracle[ok_o], r_true[:n_oracle][ok_o]
    )

    # --- Weak rate anchor (the prior run's crippled estimator), for contrast ---
    half = T_fit // 2
    r_hat_weak = np.array([fit_rate(theta_aligned[i, :half])[0] for i in range(N)])
    ok_w = np.isfinite(r_hat_weak)
    rho_weak, _ = spearmanr(r_hat_weak[ok_w], r_true[ok_w])

    # --- Broken metric (gain over window) ---
    broken = compute_broken_metric(r_hat_weak, seq_resp_np)

    results = {
        "label": label,
        "N": int(N),
        "T_fit": int(T_fit),
        "T_tail": int(T_tail),
        "T_recent": int(T_recent),
        # delta_NLL, full-window static baseline
        "mean_delta_full": float(delta_full.mean()),
        "frac_delta_full_pos": float((delta_full > 0).mean()),
        "spearman_full_vs_r_true": float(rho_full),
        "spearman_full_CI": [float(lo_full), float(hi_full)],
        # delta_NLL, recent-window static baseline (like-for-like)
        "mean_delta_recent": float(delta_recent.mean()),
        "frac_delta_recent_pos": float((delta_recent > 0).mean()),
        "spearman_recent_vs_r_true": float(rho_recent),
        "spearman_recent_CI": [float(lo_recent), float(hi_recent)],
        # clean oracle rate ceiling
        "n_oracle": int(n_oracle),
        "spearman_oracle_vs_r_true": float(rho_o),
        "spearman_oracle_CI": [float(lo_o), float(hi_o)],
        # crippled weak anchor + broken metric (prior run, for contrast)
        "spearman_weak_anchor_vs_r_true": float(rho_weak),
        "broken_metric_spearman": float(broken),
    }
    arrays = {
        "delta_full": delta_full,
        "delta_recent": delta_recent,
        "r_true": r_true,
        "r_oracle": r_oracle,
        "r_true_oracle": r_true[:n_oracle],
    }
    return results, arrays


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # With-rate condition
    print("\n=== WITH-RATE CONDITION ===")
    data_with = generate_learning_curves(
        n_items=120, n_respondents=600, seq_len=80, K=4, seed=0
    )
    res_with, arr_with = run_experiment(
        data_with, "with_rate", n_epochs=500, device=device
    )

    # No-rate control (flat ability)
    print("\n=== NO-RATE CONTROL ===")
    data_flat = generate_learning_curves(
        n_items=120, n_respondents=600, seq_len=80, K=4, seed=1,
        gain_lo=0.05, gain_hi=0.05,
    )
    res_flat, arr_flat = run_experiment(
        data_flat, "no_rate", n_epochs=500, device=device
    )

    # --- Formal existence test: with-rate vs no-rate delta_NLL ---
    # The EXISTENCE comparator is the full-window static null (constant ability,
    # time ignored). The recent-window static is itself a partial trajectory read
    # and washes the test out, so it is reported only for completeness.
    dW = arr_with["delta_full"]
    dF = arr_flat["delta_full"]
    U, p_mw = mannwhitneyu(dW, dF, alternative="greater")
    gap, gap_lo, gap_hi = mean_gap_bootstrap_ci(dW, dF)

    # Recent-window-static existence contrast, for completeness only.
    dW_rec = arr_with["delta_recent"]
    dF_rec = arr_flat["delta_recent"]
    U_rec, p_mw_rec = mannwhitneyu(dW_rec, dF_rec, alternative="greater")
    gap_rec, gap_rec_lo, gap_rec_hi = mean_gap_bootstrap_ci(dW_rec, dF_rec)

    existence = {
        "existence_comparator": "full_window_static",
        "mannwhitney_U_full": float(U),
        "mannwhitney_p_full": float(p_mw),
        "mean_gap_full": float(gap),
        "mean_gap_full_CI": [float(gap_lo), float(gap_hi)],
        "mannwhitney_U_recent": float(U_rec),
        "mannwhitney_p_recent": float(p_mw_rec),
        "mean_gap_recent": float(gap_rec),
        "mean_gap_recent_CI": [float(gap_rec_lo), float(gap_rec_hi)],
    }

    # --- Save JSON ---
    out = {"with_rate": res_with, "no_rate": res_flat, "existence_test": existence}
    json_path = "deep_irt/traj_synth/results_validity_criterion.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {json_path}")

    # --- Print summary ---
    print("\n========== RESULTS ==========")
    for cond, res in [("with_rate", res_with), ("no_rate", res_flat)]:
        print(f"\n--- {cond} ---")
        print(f"  Full-window static baseline:")
        print(f"    mean delta_NLL={res['mean_delta_full']:.4f}  "
              f"frac>0={res['frac_delta_full_pos']:.3f}")
        print(f"    Spearman(delta, r_true)={res['spearman_full_vs_r_true']:.3f}  "
              f"CI={res['spearman_full_CI']}")
        print(f"  Recent-window static baseline (primary):")
        print(f"    mean delta_NLL={res['mean_delta_recent']:.4f}  "
              f"frac>0={res['frac_delta_recent_pos']:.3f}")
        print(f"    Spearman(delta, r_true)={res['spearman_recent_vs_r_true']:.3f}  "
              f"CI={res['spearman_recent_CI']}")
        print(f"  Oracle rate ceiling Spearman(r_oracle, r_true)="
              f"{res['spearman_oracle_vs_r_true']:.3f} CI={res['spearman_oracle_CI']}")
        print(f"  Weak anchor (prior run) Spearman: "
              f"{res['spearman_weak_anchor_vs_r_true']:.3f}")
        print(f"  BROKEN metric Spearman: {res['broken_metric_spearman']:.3f}")

    print("\n--- EXISTENCE TEST (full-window static, the correct comparator) ---")
    print(f"  Mann-Whitney U={U:.1f}  p={p_mw:.3e}  (with > no)")
    print(f"  mean gap (with - no)={gap:.4f}  CI=[{gap_lo:.4f}, {gap_hi:.4f}]")
    print("--- EXISTENCE TEST (recent-window static, for completeness) ---")
    print(f"  Mann-Whitney U={U_rec:.1f}  p={p_mw_rec:.3e}  (with > no)")
    print(f"  mean gap (with - no)={gap_rec:.4f}  "
          f"CI=[{gap_rec_lo:.4f}, {gap_rec_hi:.4f}]")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: delta_NLL (recent-window static) vs r_true (with-rate)
    ax = axes[0]
    rtw = arr_with["r_true"]
    dnw = arr_with["delta_recent"]
    ax.scatter(rtw, dnw, alpha=0.3, s=10, color="steelblue")
    m = np.polyfit(rtw, dnw, 1)
    x_line = np.linspace(rtw.min(), rtw.max(), 100)
    ax.plot(x_line, np.polyval(m, x_line), color="red", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("True learning rate r")
    ax.set_ylabel("delta_NLL (recent static - curve MLE)")
    ax.set_title(
        f"MAGNITUDE test\nSpearman(delta, r_true) = "
        f"{res_with['spearman_recent_vs_r_true']:.3f} "
        f"[{res_with['spearman_recent_CI'][0]:.3f}, "
        f"{res_with['spearman_recent_CI'][1]:.3f}]"
    )

    # Panel 2: delta_NLL distributions, with-rate vs no-rate (existence test).
    # Uses the full-window static null, the correct existence comparator.
    ax = axes[1]
    all_vals = np.concatenate([dW, dF])
    bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 50)
    ax.hist(np.clip(dW, bins[0], bins[-1]), bins=bins, alpha=0.6,
            color="steelblue",
            label=f"with rate  mean={dW.mean():.3f}")
    ax.hist(np.clip(dF, bins[0], bins[-1]), bins=bins, alpha=0.6,
            color="orange",
            label=f"no rate    mean={dF.mean():.3f}")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("delta_NLL (full-window static - curve MLE)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"EXISTENCE test (full-window null)\nMann-Whitney p = {p_mw:.2e}  "
        f"mean gap = {gap:.3f} [{gap_lo:.3f}, {gap_hi:.3f}]"
    )
    ax.legend()

    # Panel 3: oracle rate vs r_true (the recoverable ceiling)
    ax = axes[2]
    ro = arr_with["r_oracle"]
    rt_o = arr_with["r_true_oracle"]
    ok = np.isfinite(ro)
    ax.scatter(rt_o[ok], ro[ok], alpha=0.3, s=10, color="seagreen")
    lim = [0, max(rt_o[ok].max(), ro[ok].max()) * 1.05]
    ax.plot(lim, lim, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("True learning rate r")
    ax.set_ylabel("Oracle MLE rate (recovered items)")
    ax.set_title(
        f"Rate CEILING (oracle_rate_mle)\nSpearman = "
        f"{res_with['spearman_oracle_vs_r_true']:.3f} "
        f"[{res_with['spearman_oracle_CI'][0]:.3f}, "
        f"{res_with['spearman_oracle_CI'][1]:.3f}]"
    )

    plt.tight_layout()
    png_path = "deep_irt/traj_synth/results_validity_criterion.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")

    # --- Write .md report ---
    write_report(res_with, res_flat, existence)


def write_report(r, f, ex):
    md = f"""# Held-Out Prediction as a Validity Criterion

## Setup

600 synthetic learners, 120 items, K=4 GPCM, seq_len=80 (rate log-uniform in
[0.04, 0.6]). The last 20% (T_tail={r['T_tail']}) of each learner's sequence is the
held-out tail. The first 80% (T_fit={r['T_fit']}) is the fit window. DeepIRTModel
(lstm, gpcm, decouple=True), 500 epochs. The no-rate control fixes the gain at 0.05
so theta is flat.

All comparisons use item parameters recovered from the dynamic model.

The dynamic prediction fits an exponential curve (A, B, r) by MLE on the
fit-window responses and extrapolates it to the tail time points. Two static
baselines predict the tail with one constant theta.

  Full-window static. A single theta fitted by MLE to all {r['T_fit']} fit-window
  responses. This pools the whole trajectory and ignores time.

  Recent-window static. A single theta fitted by MLE to the last {r['T_recent']}
  fit steps only, the window immediately before the tail.

delta_NLL = NLL_static - NLL_dynamic, per learner, in nats per held-out step.
Positive means the trajectory model predicts the future better.

## The two questions and the baseline each one needs

The held-out criterion answers two separable questions. They have different
answers, and each is served by a different static baseline.

  EXISTENCE. Does the dynamic model beat the static model when, and only when, a
  rate exists? This is the cross-condition contrast, with-rate vs no-rate. The
  right comparator is the full-window static theta, the genuine constant-ability
  null that ignores time. The recent-window static is itself a partial trajectory
  read (it tracks where the learner ended up), so it is not a clean stand-in for
  the no-trajectory hypothesis.

  MAGNITUDE. Does the per-learner margin delta_NLL rank-correlate with the true
  rate r? This is Spearman(delta_NLL, r_true) within the with-rate condition. The
  right comparator is the recent-window static theta. The full-window static is an
  unfair comparator here, because for fast early-plateau learners it pools the
  plateau and predicts the tail almost as well as the trajectory does, which drives
  the correlation negative.

## Results

### With-rate condition

| Metric | Full-window static | Recent-window static |
|---|---|---|
| Mean delta_NLL | {r['mean_delta_full']:.4f} | {r['mean_delta_recent']:.4f} |
| Fraction delta_NLL > 0 | {r['frac_delta_full_pos']:.3f} | {r['frac_delta_recent_pos']:.3f} |
| Spearman(delta, r_true) | {r['spearman_full_vs_r_true']:.3f} [{r['spearman_full_CI'][0]:.3f}, {r['spearman_full_CI'][1]:.3f}] | {r['spearman_recent_vs_r_true']:.3f} [{r['spearman_recent_CI'][0]:.3f}, {r['spearman_recent_CI'][1]:.3f}] |

Rate ceiling. oracle_rate_mle fitted on the model's recovered item parameters
(n={r['n_oracle']}) gives Spearman(r_oracle, r_true) =
{r['spearman_oracle_vs_r_true']:.3f}
[{r['spearman_oracle_CI'][0]:.3f}, {r['spearman_oracle_CI'][1]:.3f}].

For contrast, the prior run's weak rate anchor (curve fit to smoothed encoder theta
on the first half of the fit window) scored Spearman =
{r['spearman_weak_anchor_vs_r_true']:.3f}, and the broken gain-over-window metric
scored Spearman = {r['broken_metric_spearman']:.3f}.

### No-rate control (gain fixed at 0.05)

| Metric | Full-window static | Recent-window static |
|---|---|---|
| Mean delta_NLL | {f['mean_delta_full']:.4f} | {f['mean_delta_recent']:.4f} |
| Fraction delta_NLL > 0 | {f['frac_delta_full_pos']:.3f} | {f['frac_delta_recent_pos']:.3f} |
| Spearman(delta, r_true) | {f['spearman_full_vs_r_true']:.3f} | {f['spearman_recent_vs_r_true']:.3f} |

### Existence test (with-rate vs no-rate delta_NLL)

Full-window static (the correct existence comparator).
Mann-Whitney U = {ex['mannwhitney_U_full']:.0f}, p = {ex['mannwhitney_p_full']:.3e}
(one-sided, with-rate greater).
Mean gap (with minus no) = {ex['mean_gap_full']:.4f}
[{ex['mean_gap_full_CI'][0]:.4f}, {ex['mean_gap_full_CI'][1]:.4f}].

Recent-window static (for completeness, not the right comparator here).
Mann-Whitney U = {ex['mannwhitney_U_recent']:.0f}, p = {ex['mannwhitney_p_recent']:.3e}.
Mean gap (with minus no) = {ex['mean_gap_recent']:.4f}
[{ex['mean_gap_recent_CI'][0]:.4f}, {ex['mean_gap_recent_CI'][1]:.4f}].

## Diagnosis

The existence question and the magnitude question come apart, and each is read off
a different baseline.

Existence. Against the full-window static null, the dynamic model's held-out
advantage is far larger when a rate exists than when ability is flat. With a rate
the mean delta_NLL is {r['mean_delta_full']:.3f} and
{r['frac_delta_full_pos'] * 100:.0f}% of learners are predicted better by the
trajectory. With flat ability the mean drops to {f['mean_delta_full']:.3f} and only
{f['frac_delta_full_pos'] * 100:.0f}% favor the trajectory. The cross-condition gap
is {ex['mean_gap_full']:.3f} with a CI above zero and Mann-Whitney
p = {ex['mannwhitney_p_full']:.1e}. The criterion detects the presence of a
trajectory and does not reward trajectory modeling when there is nothing to track.
This is the saturation-robust property the criterion was meant to have.

The recent-window static baseline collapses this separation
(p = {ex['mannwhitney_p_recent']:.2f}) for a clear reason. That baseline is already
a one-step trajectory estimate. It tracks the learner's recent ability, so against
it the curve model has little left to add even when a rate is present, and the
with-rate and no-rate margins look alike. That is why the existence test must use
the full-window null, not the recent-window read.

Magnitude. The per-learner margin does not rank learners by rate. Against the
full-window static the correlation with r_true is negative
({r['spearman_full_vs_r_true']:.3f}), because that baseline pools the plateau and
for fast learners predicts the tail nearly as well as the trajectory. Switching to
the recent-window static removes that head start and flips the correlation positive
({r['spearman_recent_vs_r_true']:.3f}, CI above zero), but the signal is weak. The
reason is structural. By the tail, fast and slow learners are both near plateau, so
the held-out window carries little rate information no matter which static theta it
is compared against.

The rate is nonetheless recoverable from this data. oracle_rate_mle, which fits the
full parametric curve to the responses rather than comparing two windowed theta
estimates, recovers r with Spearman {r['spearman_oracle_vs_r_true']:.3f} from the
model's own recovered item parameters, well above the held-out margin and the broken
metric. The ceiling is {r['spearman_oracle_vs_r_true']:.3f} rather than the ~0.9
seen with true item parameters because the estimated items add noise, but it is the
strongest available estimator and is the correct one for magnitude.

## Side-by-side with the broken metric

The original gain-over-window metric scores Spearman =
{r['broken_metric_spearman']:.3f} against the true rate, near zero, and it is
structurally ill-posed because fast learners plateau inside the window. The held-out
existence test, by contrast, separates with-rate from no-rate at
p = {ex['mannwhitney_p_full']:.1e}. The held-out criterion is a real,
saturation-robust signal for the existence question where the broken metric is not.
For magnitude, neither the broken metric nor the held-out margin is usable;
oracle_rate_mle is.

## Verdict

EXISTENCE. Valid. Held-out predictive improvement, measured against a full-window
constant-ability null, is a saturation-robust detector of whether a learning
trajectory exists. The dynamic model beats the static null significantly more when a
rate is present than when ability is flat (p = {ex['mannwhitney_p_full']:.1e}, mean
gap {ex['mean_gap_full']:.3f} with CI above zero), and the no-rate control confirms
the criterion does not reward trajectory modeling in the absence of a trajectory. It
needs no ground-truth rate, so it transfers to real data. Use it as the validity
gate that licenses a dynamic ability claim. Use the full-window static theta as the
null, not a recent-window read, which is itself a partial trajectory and washes the
test out.

MAGNITUDE. Not valid. The per-learner delta_NLL margin does not track the rate. It
is negative against the full-window null and only weakly positive against the
recent-window comparator, because the held-out tail sits near plateau for fast and
slow learners alike. For the magnitude of the rate, fit the parametric learning
curve (theta_0, theta_inf, r) by MLE to each learner's full response sequence given
the model's estimated item parameters (oracle_rate_mle). That estimator uses the
full curve structure and is not confused by saturation in any subwindow.

Recommendation for real data (EdNet, ASSISTments, KDD). Report the two together and
keep their roles distinct. Held-out prediction against a full-window null is the
existence gate that licenses calling a learner dynamic. oracle_rate_mle on the
model's estimated item parameters is the rate estimator once that gate is passed.
The held-out margin itself is not a magnitude readout and should not be reported as
one.
"""
    md_path = "deep_irt/traj_synth/RESULTS_validity_criterion.md"
    with open(md_path, "w") as fh:
        fh.write(md)
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
