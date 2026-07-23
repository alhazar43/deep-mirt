"""CT0, the per-edge sign-recovery POWER precondition
(`_planning/design/a1_design.md` v1.1, sections 2.1.1, 4.1, 5.1).

CT0 is the make-or-break, fail-fast test for goal G1: at the smallest
setup (D = 3), is a per-EDGE signed influence recoverable AT ALL at some
feasible (N learners, decoupling)? It reads the fitted signed matrix ``G``
from a `transfer.model.TransferModel` fit to the SYN-T-KG twin, scores its
recovered signs against the injected ``G_true`` over the off-diagonal
cells (sign-F1, section 4.1), and sweeps the per-edge effective sample via
N and decoupling to trace the power curve.

Verdict logic (section 5.1 CT0 row, section 5.4 K-T1): if per-edge sign-F1
clears the CT1 bar at SOME feasible (N, decoupling) at D = 3, G1 is
feasible and the smallest such effective sample is the frozen CT0 minimum;
if NO feasible (N, decoupling) clears it, per-edge sign is unidentifiable
and G1 dies on this design space (K-T1), the kill decided on the power
curve, not on a single failed fit.

Sign thresholding is against a matched-null band drawn from the SYN-T-NG
twin's recovered ``|G|`` (design section 4.2: fitted G carries a per-seed
additive offset, so it is NEVER compared to a bare zero), pooled across
seeds.

CT0 posture note (best-case identifiability). By default the model's gate
constants ``M`` and ``floor`` are FIXED at the generator's true values, so
the fitted gate matches the generating gate exactly. This isolates the
question CT0 asks -- is the SIGN identifiable -- from own-gain-ceiling
misfit (which is CT5/SYN-T-NS's job, a later stage). A failure under this
best case is decisive (K-T1); a pass is necessary, not sufficient, and
later stages re-test with a fitted gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import torch

from kt_mirt.growth.tracker import build_learner_batch
from kt_mirt.transfer.model import TransferConfig, TransferModel, train_transfer
from kt_mirt.transfer.synth import SignedTwin, generate_signed_twin


# ---------------------------------------------------------------------------
# CT1 bar (design section 5.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CT1Bar:
    """The pre-registered CT1 sign-recovery bar (design section 5.1),
    reused by CT0 as the threshold the power curve must reach."""

    sign_f1: float = 0.80
    sign_accuracy_true_edges: float = 0.85
    false_edge_rate: float = 0.05
    neg_half_sign_f1: float = 0.75
    baseline_margin: float = 0.15  # must beat all-zero and random-sign by this


# ---------------------------------------------------------------------------
# Sign scoring (section 4.1)
# ---------------------------------------------------------------------------


def edge_signs(G: np.ndarray, band: float) -> np.ndarray:
    """Off-diagonal recovered sign in {-1, 0, +1}: ``+1`` if ``G > band``,
    ``-1`` if ``G < -band``, else ``0``. Diagonal forced to 0."""
    G = np.asarray(G, dtype=float)
    sign = np.zeros_like(G)
    sign[G > band] = 1.0
    sign[G < -band] = -1.0
    np.fill_diagonal(sign, 0.0)
    return sign


def _f1_for_target(pred: np.ndarray, true: np.ndarray, off: np.ndarray, target_sign: Optional[int]) -> dict:
    """F1 of detecting edges of ``target_sign`` (``+1``/``-1``); if
    ``target_sign is None`` the signed-edge task (detect any nonzero edge
    WITH correct sign). A wrong-sign call on a true edge is both a false
    positive and a false negative (a sign flip is maximally penalized, the
    standard signed link-prediction convention)."""
    p = pred[off]
    t = true[off]
    if target_sign is None:
        tp = int(np.sum((t != 0) & (p == t)))
        fp = int(np.sum((p != 0) & (p != t)))
        fn = int(np.sum((t != 0) & (p != t)))
    else:
        tp = int(np.sum((t == target_sign) & (p == target_sign)))
        fp = int(np.sum((p == target_sign) & (t != target_sign)))
        fn = int(np.sum((t == target_sign) & (p != target_sign)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec), "tp": tp, "fp": fp, "fn": fn}


def sign_f1(G_hat: np.ndarray, G_true: np.ndarray, band: float) -> dict:
    """The section-4.1 sign-recovery metrics over the off-diagonal cells.

    Headline ``sign_f1`` is the signed-edge F1 (detect a true nonzero edge
    with the correct sign). Also returns ``sign_accuracy_true_edges``
    (fraction of true nonzero edges recovered with the right sign),
    ``false_edge_rate`` (fraction of true-zero cells called nonzero), the
    positive- and negative-half F1 and the negative-half recall (reported
    separately so L1 shrinkage cannot pass by silently zeroing the weaker
    negative edges, section 2.2), and the degenerate baselines (all-zero
    F1 = 0 by construction, and the expected random-sign F1)."""
    C = G_true.shape[0]
    off = ~np.eye(C, dtype=bool)
    pred = edge_signs(G_hat, band)
    true = edge_signs_true(G_true)

    overall = _f1_for_target(pred, true, off, None)
    pos = _f1_for_target(pred, true, off, +1)
    neg = _f1_for_target(pred, true, off, -1)

    t = true[off]
    p = pred[off]
    n_true_edges = int(np.sum(t != 0))
    n_zero_cells = int(np.sum(t == 0))
    sign_acc = float(np.sum((t != 0) & (p == t)) / n_true_edges) if n_true_edges else float("nan")
    false_edge_rate = float(np.sum((t == 0) & (p != 0)) / n_zero_cells) if n_zero_cells else 0.0

    return {
        "sign_f1": overall["f1"],
        "sign_accuracy_true_edges": sign_acc,
        "false_edge_rate": false_edge_rate,
        "pos_f1": pos["f1"],
        "neg_f1": neg["f1"],
        "neg_recall": neg["recall"],
        "pos_recall": pos["recall"],
        "n_true_edges": n_true_edges,
        "n_true_pos": int(np.sum(t > 0)),
        "n_true_neg": int(np.sum(t < 0)),
        "n_zero_cells": n_zero_cells,
        "baseline_allzero_f1": 0.0,
        "baseline_randsign_f1": _random_sign_f1(true, off),
    }


def edge_signs_true(G_true: np.ndarray) -> np.ndarray:
    """True off-diagonal signs in {-1, 0, +1} (no band; ground truth is
    exact)."""
    G = np.asarray(G_true, dtype=float)
    sign = np.sign(G)
    np.fill_diagonal(sign, 0.0)
    return sign


def _random_sign_f1(true: np.ndarray, off: np.ndarray, n_draws: int = 2000, seed: int = 0) -> float:
    """Expected signed-edge F1 of a predictor that calls every off-diagonal
    cell +1 or -1 uniformly at random (the section-4.1 degenerate baseline
    the recovered F1 must beat by the CT1 margin)."""
    rng = np.random.default_rng(seed)
    C = true.shape[0]
    t = true[off]
    vals = []
    for _ in range(n_draws):
        p_full = np.zeros((C, C))
        p_full[off] = rng.choice([-1.0, 1.0], size=int(off.sum()))
        vals.append(_f1_for_target(p_full, true, off, None)["f1"])
    return float(np.mean(vals))


def pooled_null_band(G_hats_ng: Sequence[np.ndarray], quantile: float = 0.95) -> float:
    """The matched-null sign band (design section 4.2): the ``quantile`` of
    the pooled off-diagonal ``|G_hat|`` across the SYN-T-NG fits (seeds).
    Recovered signals below this band are indistinguishable from the null
    and scored as zero."""
    pooled = []
    for G in G_hats_ng:
        C = G.shape[0]
        off = ~np.eye(C, dtype=bool)
        pooled.append(np.abs(np.asarray(G))[off])
    if not pooled:
        return 0.0
    return float(np.quantile(np.concatenate(pooled), quantile))


# ---------------------------------------------------------------------------
# Fitting a twin
# ---------------------------------------------------------------------------


def fit_twin(
    twin: SignedTwin, cfg: TransferConfig, model_seed: int = 0, fix_gate_at_truth: bool = True
) -> tuple[np.ndarray, TransferModel]:
    """Two-stage fit of a `TransferModel` to one twin, returning
    ``(G_hat, model)``. On CT0's clean synthetic screen stage 1 (bank
    freeze) is the generator's true difficulties, fed through ``batch.b_j``.
    ``fix_gate_at_truth`` pins the model's gate constants ``M``/``floor`` at
    the generator's true values (the best-case posture, module docstring).
    Intra-op threads are pinned to 2 for the fit (the tiny per-step tensors
    make the default pool's synchronization dominate)."""
    batch = build_learner_batch(twin.learners, b_true=twin.truth.b_true, device=cfg.device)
    cfg_fit = replace(
        cfg, seed=model_seed, floor=twin.truth.floor, m_fixed=(fix_gate_at_truth or cfg.m_fixed)
    )
    torch.manual_seed(model_seed)
    model = TransferModel(
        num_items=twin.item_bank.n_items, n_kcs=twin.n_kcs, cfg=cfg_fit, m_init=twin.truth.M_gen
    )
    n_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        train_transfer(model, batch, cfg_fit)
    finally:
        torch.set_num_threads(n_threads)
    return model.recovered_G(), model


# ---------------------------------------------------------------------------
# Power sweep
# ---------------------------------------------------------------------------


@dataclass
class CT0CellResult:
    density: str
    n_learners: int
    decoupling: float
    seeds: list[int]
    eff_sample_mean: float  # mean per-edge effective sample over true edges (and seeds)
    eff_sample_by_edge: dict  # {(c,a): mean eff sample}
    band: float
    metrics_by_seed: list[dict]
    metrics_mean: dict  # seed-mean of the sign_f1 metrics
    sign_consistent: bool  # every true edge sign-consistent across seeds
    clears_ct1: bool


def _seed_mean_metrics(metrics_by_seed: list[dict]) -> dict:
    keys = ["sign_f1", "sign_accuracy_true_edges", "false_edge_rate", "pos_f1", "neg_f1", "neg_recall",
            "baseline_allzero_f1", "baseline_randsign_f1"]
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_by_seed if np.isfinite(m.get(k, np.nan))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def clears_ct1(metrics_mean: dict, sign_consistent: bool, bar: CT1Bar = CT1Bar()) -> bool:
    """Whether a cell's seed-mean metrics clear the CT1 bar (design section
    5.1): sign-F1, sign accuracy on true edges, false-edge rate, negative-
    half F1, beating both degenerate baselines by the margin, and seed sign
    consistency."""
    m = metrics_mean
    if not np.isfinite(m.get("sign_f1", np.nan)):
        return False
    return bool(
        m["sign_f1"] >= bar.sign_f1
        and m["sign_accuracy_true_edges"] >= bar.sign_accuracy_true_edges
        and m["false_edge_rate"] <= bar.false_edge_rate
        and m["neg_f1"] >= bar.neg_half_sign_f1
        and m["sign_f1"] >= m["baseline_allzero_f1"] + bar.baseline_margin
        and m["sign_f1"] >= m["baseline_randsign_f1"] + bar.baseline_margin
        and sign_consistent
    )


def run_cell(
    density: str,
    n_learners: int,
    decoupling: float,
    seeds: Sequence[int],
    cfg: TransferConfig,
    g_pos: float = 0.05,
    g_neg: float = 0.02,
    band_quantile: float = 0.95,
    fix_gate_at_truth: bool = True,
    bar: CT1Bar = CT1Bar(),
) -> CT0CellResult:
    """Fit KG and NG for every seed at one (N, decoupling) cell, pool the
    NG band, score each KG's sign recovery, and return the aggregated cell
    result."""
    G_hats_kg, G_hats_ng, eff_samples, G_true_ref = [], [], [], None
    for s in seeds:
        kg = generate_signed_twin("syn_t_kg", density, seed=s, n_learners=n_learners,
                                   decoupling=decoupling, g_pos=g_pos, g_neg=g_neg)
        ng = generate_signed_twin("syn_t_ng", density, seed=s, n_learners=n_learners,
                                   decoupling=decoupling, g_pos=g_pos, g_neg=g_neg)
        G_true_ref = kg.truth.G_true
        Gk, _ = fit_twin(kg, cfg, model_seed=s, fix_gate_at_truth=fix_gate_at_truth)
        Gn, _ = fit_twin(ng, cfg, model_seed=s, fix_gate_at_truth=fix_gate_at_truth)
        G_hats_kg.append(Gk)
        G_hats_ng.append(Gn)
        eff_samples.append(kg.truth.eff_sample)

    band = pooled_null_band(G_hats_ng, band_quantile)
    metrics_by_seed = [sign_f1(Gk, G_true_ref, band) for Gk in G_hats_kg]
    metrics_mean = _seed_mean_metrics(metrics_by_seed)

    # Per-edge sign consistency across seeds (design section 4.9, applied to CT0).
    off = ~np.eye(G_true_ref.shape[0], dtype=bool)
    true_edges = list(zip(*np.where((G_true_ref != 0) & off)))
    sign_consistent = True
    for (c, a) in true_edges:
        signs = [np.sign(edge_signs(Gk, band)[c, a]) for Gk in G_hats_kg]
        nonzero = [sg for sg in signs if sg != 0]
        if not nonzero or len(set(nonzero)) != 1 or len(nonzero) != len(signs):
            sign_consistent = False
            break

    eff_mean_arr = np.mean(eff_samples, axis=0)
    eff_by_edge = {(int(c), int(a)): float(eff_mean_arr[c, a]) for (c, a) in true_edges}
    eff_sample_mean = float(np.mean([eff_mean_arr[c, a] for (c, a) in true_edges])) if true_edges else 0.0

    return CT0CellResult(
        density=density, n_learners=n_learners, decoupling=decoupling, seeds=list(seeds),
        eff_sample_mean=eff_sample_mean, eff_sample_by_edge=eff_by_edge, band=band,
        metrics_by_seed=metrics_by_seed, metrics_mean=metrics_mean,
        sign_consistent=sign_consistent, clears_ct1=clears_ct1(metrics_mean, sign_consistent, bar),
    )


@dataclass
class CT0Verdict:
    density: str
    cells: list[CT0CellResult]
    any_cell_clears: bool
    min_effective_sample: Optional[float]  # smallest eff sample at a clearing cell; the frozen CT0 cut
    min_cell: Optional[dict]
    kt1_fired: bool  # per-edge sign unidentifiable at D=3 (design K-T1)


def power_sweep(
    density: str = "kdd",
    n_grid: Sequence[int] = (250, 500, 1000, 2000),
    decoupling_grid: Sequence[float] = (0.9, 0.75, 0.5),
    seeds: Sequence[int] = (0, 1, 2),
    cfg: Optional[TransferConfig] = None,
    g_pos: float = 0.05,
    g_neg: float = 0.02,
    fix_gate_at_truth: bool = True,
    bar: CT1Bar = CT1Bar(),
) -> CT0Verdict:
    """Run the CT0 power sweep over (N, decoupling) at D = 3 and return the
    verdict. ``min_effective_sample`` is the smallest per-edge effective
    sample among cells that clear the CT1 bar (the frozen CT0 minimum,
    section 2.1.1 / judgment call 10); if no cell clears, K-T1 fires
    (per-edge sign unidentifiable at feasible data at D = 3, section 5.4)."""
    if cfg is None:
        cfg = TransferConfig()
    cells: list[CT0CellResult] = []
    for N in n_grid:
        for d in decoupling_grid:
            cells.append(run_cell(density, N, d, seeds, cfg, g_pos, g_neg,
                                  fix_gate_at_truth=fix_gate_at_truth, bar=bar))

    clearing = [c for c in cells if c.clears_ct1]
    any_clear = len(clearing) > 0
    min_eff, min_cell = None, None
    if any_clear:
        best = min(clearing, key=lambda c: c.eff_sample_mean)
        min_eff = best.eff_sample_mean
        min_cell = {"n_learners": best.n_learners, "decoupling": best.decoupling,
                    "eff_sample_mean": best.eff_sample_mean, "sign_f1": best.metrics_mean["sign_f1"]}
    return CT0Verdict(
        density=density, cells=cells, any_cell_clears=any_clear,
        min_effective_sample=min_eff, min_cell=min_cell, kt1_fired=not any_clear,
    )


__all__ = [
    "CT1Bar",
    "edge_signs",
    "edge_signs_true",
    "sign_f1",
    "pooled_null_band",
    "fit_twin",
    "CT0CellResult",
    "clears_ct1",
    "run_cell",
    "CT0Verdict",
    "power_sweep",
]
