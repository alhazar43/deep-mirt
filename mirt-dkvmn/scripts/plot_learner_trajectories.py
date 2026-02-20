"""Plot learner state trajectories comparing DEEP-GPCM vs DKVMN+Softmax stub.

For each of four representative students (high-ability, low-ability, mid-ability
long-sequence, ambiguous), produces a 4-panel figure with:
  - Blue solid:   DEEP-GPCM E[r_t] = sum_k k * p_k  (expected response score)
  - Orange dashed: DEEP-GPCM theta_t[:,0] linearly rescaled to [0, K-1]
  - Gray solid:   DKVMN+Softmax stub (same probs; replaced when real baseline trains)
  - Black dashed: true E[r | theta_true, alpha_ref, beta_ref] (horizontal reference)

Theory
------
E[r_t] = sum_{k=0}^{K-1} k * P(R=k | theta_t, alpha_t, beta_t)

This is the natural ordinal proxy for learner state under the GPCM: it collapses
the full categorical distribution to a scalar on [0, K-1], making it comparable
across models regardless of whether they expose IRT parameters.

The theta rescaling maps the IRT ability scale (centred on N(0,1)) to the item
response scale via the linear map:
  theta_display = (theta_1 + 3) / 6 * (K - 1)
clipped to [0, K-1].  This is a display convenience, not a model transformation.

The reference item is selected as the item j where:
  argmin_j |beta_j[0] - 0|  among items with |alpha_j[0] - mean(alpha[:,0])| < 0.1
ensuring a "typical" item near the population ability centre.

Usage
-----
PYTHONPATH=src python scripts/plot_learner_trajectories.py \\
    --config configs/run_c3_d3_20.yaml \\
    --checkpoint artifacts/run_c3_d3_20/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path bootstrap: allow running from repo root with PYTHONPATH=mirt-dkvmn/src
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT


# ---------------------------------------------------------------------------
# Student selection helpers
# ---------------------------------------------------------------------------

def select_representative_students(
    theta_true: np.ndarray,          # (N, D)
    questions_all: List[List[int]],  # len N
    responses_all: List[List[int]],  # len N
    rng: np.random.Generator,
) -> Dict[str, int]:
    """Return {label: student_index} for four representative students.

    Selection criteria
    ------------------
    high:      top 5% by mean signed theta (mean across traits) — genuinely high ability
    low:       bottom 5% by mean signed theta — genuinely low ability
    mid_long:  |mean_theta| < 0.3, sequence length > 100
               (falls back to |mean_theta| < 0.5 if no long sequences exist)
    ambiguous: mid-ability (40th-60th percentile), highest response variance
    """
    n = theta_true.shape[0]
    # Use mean trait value (signed) as the primary ability score so that
    # high = strong positive ability, low = strong negative ability.
    scores = theta_true.mean(axis=1)  # (N,) signed mean across D traits

    p5 = int(np.floor(0.05 * n))
    p95 = n - p5

    sorted_idx = np.argsort(scores)
    high_pool = sorted_idx[p95:]   # top 5% by mean theta
    low_pool = sorted_idx[:p5]     # bottom 5% by mean theta

    # High and low: pick the extreme within each pool
    high_idx = int(high_pool[np.argmax(scores[high_pool])])
    low_idx = int(low_pool[np.argmin(scores[low_pool])])

    # Mid-long: prefer sequences > 100, |mean_theta| < 0.3
    mid_long_idx: Optional[int] = None
    for threshold in [0.3, 0.5, 1.0]:
        candidates = [
            i for i in range(n)
            if abs(scores[i]) < threshold and len(questions_all[i]) > 100
        ]
        if candidates:
            # Pick longest sequence among candidates
            mid_long_idx = int(max(candidates, key=lambda i: len(questions_all[i])))
            break
    if mid_long_idx is None:
        # Absolute fallback: most central theta, longest sequence
        central = sorted_idx[n // 2]
        mid_long_idx = int(central)

    # Ambiguous: mid-ability pool (40th-60th percentile score), highest response variance
    lo40, hi60 = np.percentile(scores, [40, 60])
    mid_pool = [i for i in range(n) if lo40 <= scores[i] <= hi60]
    if not mid_pool:
        mid_pool = list(range(n))
    resp_vars = [float(np.var(responses_all[i])) for i in mid_pool]
    ambiguous_idx = int(mid_pool[int(np.argmax(resp_vars))])

    return {
        "High-ability": high_idx,
        "Low-ability": low_idx,
        "Mid-ability": mid_long_idx,
        "Ambiguous": ambiguous_idx,
    }


# ---------------------------------------------------------------------------
# Reference item selection
# ---------------------------------------------------------------------------

def select_reference_item(
    alpha_true: np.ndarray,  # (J, D)
    beta_true: np.ndarray,   # (J, K-1)
    alpha_tol: float = 0.15,
) -> int:
    """Select the reference item j for computing true E[r | theta_true, params_j].

    The reference item satisfies two criteria:
    1. alpha_j[0] is within `alpha_tol` of mean(alpha[:,0])  => "typical discrimination"
    2. beta_j[0] is closest to 0.0 among qualifying items   => "medium difficulty"

    Falls back to global closest beta_j[0] to 0 if no items pass the alpha filter.
    """
    mean_alpha0 = float(np.mean(alpha_true[:, 0]))
    alpha_diff = np.abs(alpha_true[:, 0] - mean_alpha0)

    # Widen tolerance if too restrictive
    for tol in [alpha_tol, 0.3, 1.0]:
        qualifying = np.where(alpha_diff < tol)[0]
        if len(qualifying) > 0:
            break

    beta_diff = np.abs(beta_true[qualifying, 0])
    ref_item = int(qualifying[np.argmin(beta_diff)])
    return ref_item


# ---------------------------------------------------------------------------
# True GPCM expected response computation
# ---------------------------------------------------------------------------

def gpcm_prob_true(
    theta: np.ndarray,  # (D,)
    alpha: np.ndarray,  # (D,)
    beta: np.ndarray,   # (K-1,)
) -> np.ndarray:
    """Compute P(R=k | theta, alpha, beta) using the M-GPCM formulation.

    This replicates MirtGpcmGenerator.gpcm_prob exactly:
      logit_k = sum_{h=1}^{k} (dot(theta, alpha) - beta_h * alpha_scale)
    where alpha_scale = ||alpha|| / sqrt(D).

    Returns prob vector of shape (K,).
    """
    n_cats = beta.shape[0] + 1
    dot = float(np.dot(theta, alpha))
    alpha_scale = float(np.linalg.norm(alpha) / math.sqrt(alpha.shape[0]))
    cum_logits = np.zeros(n_cats)
    for k in range(1, n_cats):
        cum_logits[k] = float(np.sum(dot - beta[:k] * alpha_scale))
    # Numerically stable softmax
    cum_logits -= np.max(cum_logits)
    exp_logits = np.exp(cum_logits)
    return exp_logits / np.sum(exp_logits)


def expected_response(probs: np.ndarray) -> float:
    """E[r] = sum_k k * p_k for 0-indexed categories."""
    k_vals = np.arange(probs.shape[0], dtype=float)
    return float(np.dot(k_vals, probs))


# ---------------------------------------------------------------------------
# Model inference for a single student
# ---------------------------------------------------------------------------

def run_single_student(
    model: DKVMNMIRT,
    questions: List[int],
    responses: List[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one forward pass for a single student.

    Returns
    -------
    theta_seq : ndarray of shape (T, D)
    probs_seq : ndarray of shape (T, K)
    """
    q_t = torch.tensor([questions], dtype=torch.long, device=device)   # (1, T)
    r_t = torch.tensor([responses], dtype=torch.long, device=device)   # (1, T)

    model.eval()
    with torch.no_grad():
        theta, beta, alpha, probs = model(q_t, r_t)

    # theta: (1, T, D), probs: (1, T, K) — squeeze batch dim
    theta_np = theta[0].cpu().numpy()    # (T, D)
    probs_np = probs[0].cpu().numpy()    # (T, K)
    return theta_np, probs_np


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_trajectories(
    config_path: str,
    checkpoint_path: str,
    output_dir: Optional[str] = None,
) -> None:
    """Load model + data, select students, and produce the 4-panel trajectory figure."""

    # ------------------------------------------------------------------
    # 1. Load config and resolve paths
    # ------------------------------------------------------------------
    config = load_config(config_path)
    device = torch.device("cpu")  # trajectory inference is lightweight

    artifact_dir = Path(config.training.output_dir)
    if output_dir is None:
        out_dir = artifact_dir / "trajectory_plots"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cats = config.model.n_cats
    K = n_cats

    # ------------------------------------------------------------------
    # 2. Validate checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("Available checkpoints:", file=sys.stderr)
        for p in sorted(artifact_dir.glob("*.pt")):
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Load dataset (all students, unshuffled)
    # ------------------------------------------------------------------
    data_root = Path(config.data.data_root)
    dataset_name = config.data.dataset_name
    dataset_dir = data_root / dataset_name

    params_path = dataset_dir / "true_irt_parameters.json"
    if not params_path.exists():
        print(f"[ERROR] Missing true IRT parameters: {params_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading IRT parameters from {params_path}")
    with params_path.open("r", encoding="utf-8") as fh:
        true_params = json.load(fh)

    theta_true = np.array(true_params["theta"])   # (N, D)
    alpha_true = np.array(true_params["alpha"])   # (J, D)
    beta_true = np.array(true_params["beta"])     # (J, K-1)

    print(f"Loading sequences from {dataset_dir}")
    loader_mgr = DataLoaderManager(dataset_name, data_root=str(data_root))

    # For text-format datasets, DataLoaderManager.load() only reads _train.txt.
    # We need all students in original order to align with theta_true[0..N-1].
    # load_splits() returns train/valid/test in order, so we concatenate them.
    splits = loader_mgr.load_splits(split_ratio=0.8, val_ratio=0.1)
    questions_all: List[List[int]] = (
        splits["train"].questions
        + splits["valid"].questions
        + splits["test"].questions
    )
    responses_all: List[List[int]] = (
        splits["train"].responses
        + splits["valid"].responses
        + splits["test"].responses
    )
    n_questions_data = splits["train"].n_questions
    n_cats_data = splits["train"].n_cats
    n_students = len(questions_all)
    print(
        f"Loaded {n_students} students "
        f"(train={len(splits['train'].questions)}, "
        f"val={len(splits['valid'].questions)}, "
        f"test={len(splits['test'].questions)}), "
        f"{n_questions_data} items, {n_cats_data} categories"
    )

    # Align: theta_true has n_students rows (all students before splitting)
    n_aligned = min(n_students, theta_true.shape[0])
    theta_aligned = theta_true[:n_aligned]
    questions_aligned = questions_all[:n_aligned]
    responses_aligned = responses_all[:n_aligned]

    # ------------------------------------------------------------------
    # 4. Build model and load checkpoint
    # ------------------------------------------------------------------
    model = DKVMNMIRT(
        n_questions=config.model.n_questions,
        n_cats=config.model.n_cats,
        n_traits=config.model.n_traits,
        memory_size=config.model.memory_size,
        key_dim=config.model.key_dim,
        value_dim=config.model.value_dim,
        summary_dim=config.model.summary_dim,
        concept_aligned_memory=config.model.concept_aligned_memory,
        theta_projection=config.model.theta_projection,
        memory_add_activation=config.model.memory_add_activation,
        theta_source=config.model.theta_source,
    ).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # 5. Select representative students
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    selected = select_representative_students(
        theta_aligned, questions_aligned, responses_aligned, rng
    )
    print("\nSelected students:")
    for label, idx in selected.items():
        seq_len = len(questions_aligned[idx])
        mean_theta = float(theta_aligned[idx].mean())
        print(f"  {label:15s}: student {idx:5d}  mean(theta)={mean_theta:.3f}  T={seq_len}")

    # ------------------------------------------------------------------
    # 6. Select reference item and compute true E[r] per student
    # ------------------------------------------------------------------
    ref_item = select_reference_item(alpha_true, beta_true)
    alpha_ref = alpha_true[ref_item]   # (D,)
    beta_ref = beta_true[ref_item]     # (K-1,)
    print(f"\nReference item: {ref_item}  alpha={alpha_ref}  beta_0={beta_ref[0]:.3f}")

    true_ref_er: Dict[str, float] = {}
    for label, idx in selected.items():
        probs_ref = gpcm_prob_true(theta_aligned[idx], alpha_ref, beta_ref)
        true_ref_er[label] = expected_response(probs_ref)

    # ------------------------------------------------------------------
    # 7. Run model inference for each selected student
    # ------------------------------------------------------------------
    k_vals = np.arange(K, dtype=float)

    student_data: Dict[str, dict] = {}
    for label, idx in selected.items():
        q_seq = questions_aligned[idx]
        r_seq = responses_aligned[idx]

        theta_seq, probs_seq = run_single_student(model, q_seq, r_seq, device)
        # theta_seq: (T, D), probs_seq: (T, K)

        # DEEP-GPCM E[r_t]
        er_deepgpcm = np.einsum("tk,k->t", probs_seq, k_vals)  # (T,)

        # DKVMN+Softmax stub: same probs (intentional; replaced when real baseline trains)
        er_stub = er_deepgpcm.copy()

        # Theta rescaling: map [-3, 3] -> [0, K-1] linearly
        theta0 = theta_seq[:, 0]                         # (T,) first trait
        theta_display = (theta0 + 3.0) / 6.0 * (K - 1)
        theta_display = np.clip(theta_display, 0.0, K - 1)

        def smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
            """Rolling mean with window w, keeping length unchanged (edge-pad)."""
            out = np.convolve(x, np.ones(w) / w, mode="same")
            # Fix edges: use smaller windows near boundaries
            for i in range(w // 2):
                out[i] = x[:2 * i + 1].mean()
                out[-(i + 1)] = x[-(2 * i + 1):].mean()
            return out

        student_data[label] = {
            "er_deepgpcm": smooth(er_deepgpcm),
            "er_stub": smooth(er_stub),
            "theta_display": smooth(theta_display),
            "T": len(q_seq),
            "true_er": true_ref_er[label],
            "mean_theta": float(theta_aligned[idx].mean()),
        }

    # ------------------------------------------------------------------
    # 8. Plot
    # ------------------------------------------------------------------
    labels_ordered = ["High-ability", "Low-ability", "Mid-ability", "Ambiguous"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    axes_flat = axes.flatten()

    for panel_idx, label in enumerate(labels_ordered):
        ax = axes_flat[panel_idx]
        d = student_data[label]
        T = d["T"]
        t_steps = np.arange(T)

        ax.plot(
            t_steps,
            d["er_deepgpcm"],
            color="tab:blue",
            linewidth=1.8,
            label="DEEP-GPCM E[r_t]",
            zorder=3,
        )
        ax.plot(
            t_steps,
            d["theta_display"],
            color="tab:orange",
            linewidth=1.5,
            linestyle="--",
            label=r"DEEP-GPCM $\theta_t^{(1)}$ (rescaled)",
            zorder=2,
        )
        ax.plot(
            t_steps,
            d["er_stub"],
            color="gray",
            linewidth=1.2,
            alpha=0.6,
            label="DKVMN+Softmax (stub)",
            zorder=1,
        )
        ax.axhline(
            d["true_er"],
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"True E[r | ref item] = {d['true_er']:.2f}",
            zorder=4,
        )

        ax.set_title(
            f"{label}  (mean θ={d['mean_theta']:.2f}, T={T})",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Interaction step t", fontsize=9)
        ax.set_ylabel("Expected response [0, K-1]", fontsize=9)
        ax.set_ylim(-0.05, K - 1 + 0.05)
        ax.set_xlim(0, max(T - 1, 1))
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.tick_params(labelsize=8)

        if panel_idx == 0:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

    fig.suptitle(
        f"Learner state trajectories — {dataset_name}\n"
        f"Reference item {ref_item}: "
        rf"$\alpha_{{ref,0}}$={alpha_ref[0]:.3f}, "
        rf"$\beta_{{ref,0}}$={beta_ref[0]:.3f}",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / "learner_trajectories.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_path}")

    # Save student metadata as JSON for provenance
    meta = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "reference_item": ref_item,
        "alpha_ref": alpha_ref.tolist(),
        "beta_ref": beta_ref.tolist(),
        "students": {
            label: {
                "index": idx,
                "seq_len": len(questions_aligned[idx]),
                "theta_true": theta_aligned[idx].tolist(),
                "true_er_ref_item": true_ref_er[label],
            }
            for label, idx in selected.items()
        },
    }
    meta_path = out_dir / "trajectory_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Metadata saved to: {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learner state trajectories for representative students."
    )
    parser.add_argument(
        "--config",
        default="configs/large_d3_opt3.yaml",
        help="Path to YAML config (default: configs/large_d3_opt3.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for plots (default: <artifact_dir>/trajectory_plots/)",
    )
    args = parser.parse_args()
    plot_trajectories(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
