"""Research-oriented plots for polytomous MIRT-DKVMN runs."""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT

try:
    from scipy.stats import rankdata as _rankdata
except ImportError:  # pragma: no cover - fallback for environments without scipy
    _rankdata = None


def _pad_sequence(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[1] == max_len:
            padded.append(t)
            continue
        pad_shape = list(t.shape)
        pad_shape[1] = max_len - t.shape[1]
        pad = torch.full(pad_shape, pad_value, dtype=t.dtype)
        padded.append(torch.cat([t, pad], dim=1))
    return torch.cat(padded, dim=0)


def _collect_outputs(
    model,
    dataloader,
    device,
    capture_memory: bool = False,
    max_students: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    model.capture_memory = capture_memory
    outputs = {
        "probs": [],
        "responses": [],
        "theta": [],
        "alpha": [],
        "beta": [],
        "attention": [],
        "mask": [],
        "questions": [],
        "read": [],
        "value_memory": [],
    }
    collected = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_students is not None and collected >= max_students:
                break
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            mask = batch["mask"].to(device)
            if max_students is not None:
                remaining = max_students - collected
                if remaining <= 0:
                    break
                questions = questions[:remaining]
                responses = responses[:remaining]
                mask = mask[:remaining]
            theta, beta, alpha, probs = model(questions, responses)
            attention = getattr(model, "last_attention", None)
            read = getattr(model, "last_read", None)
            value_memory = getattr(model, "last_value_memory", None)

            outputs["probs"].append(probs.cpu())
            outputs["responses"].append(responses.cpu())
            outputs["theta"].append(theta.cpu())
            outputs["alpha"].append(alpha.cpu())
            outputs["beta"].append(beta.cpu())
            outputs["mask"].append(mask.cpu())
            outputs["questions"].append(questions.cpu())
            if attention is not None:
                outputs["attention"].append(attention.cpu())
            if read is not None:
                outputs["read"].append(read.cpu())
            if value_memory is not None:
                outputs["value_memory"].append(value_memory.cpu())
            if max_students is not None:
                collected += questions.shape[0]

    outputs["probs"] = _pad_sequence(outputs["probs"], pad_value=0.0).numpy()
    outputs["responses"] = _pad_sequence(outputs["responses"], pad_value=0).numpy()
    outputs["theta"] = _pad_sequence(outputs["theta"], pad_value=0.0).numpy()
    outputs["alpha"] = _pad_sequence(outputs["alpha"], pad_value=0.0).numpy()
    outputs["beta"] = _pad_sequence(outputs["beta"], pad_value=0.0).numpy()
    outputs["mask"] = _pad_sequence(outputs["mask"], pad_value=0).numpy().astype(bool)
    outputs["questions"] = _pad_sequence(outputs["questions"], pad_value=0).numpy()
    if outputs["attention"]:
        outputs["attention"] = _pad_sequence(outputs["attention"], pad_value=0.0).numpy()
    else:
        outputs["attention"] = None
    if outputs["read"]:
        outputs["read"] = _pad_sequence(outputs["read"], pad_value=0.0).numpy()
    else:
        outputs["read"] = None
    if outputs["value_memory"]:
        outputs["value_memory"] = _pad_sequence(outputs["value_memory"], pad_value=0.0).numpy()
    else:
        outputs["value_memory"] = None
    return outputs


def _calibration_by_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 10,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    results = {}
    n_cats = probs.shape[-1]
    for k in range(1, n_cats):
        pred = probs[:, :, k:].sum(axis=-1)
        obs = (targets >= k).astype(float)
        pred = pred[mask]
        obs = obs[mask]
        bins = np.linspace(0, 1, n_bins + 1)
        centers = []
        accs = []
        for i in range(n_bins):
            idx = (pred >= bins[i]) & (pred < bins[i + 1])
            if idx.any():
                centers.append(pred[idx].mean())
                accs.append(obs[idx].mean())
        results[k] = (np.array(centers), np.array(accs))
    return results


def plot_calibration_thresholds(out_dir: Path, probs, targets, mask) -> None:
    data = _calibration_by_threshold(probs, targets, mask)
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, (centers, accs) in data.items():
        ax.plot(centers, accs, marker="o", label=f"P(X>= {k})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Ordinal Calibration (Thresholds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_thresholds.png", dpi=150)
    plt.close(fig)


def plot_category_proportions(out_dir: Path, probs, targets, mask) -> None:
    preds = probs.argmax(axis=-1)
    true_flat = targets[mask].reshape(-1)
    pred_flat = preds[mask].reshape(-1)
    n_cats = probs.shape[-1]
    true_counts = np.bincount(true_flat, minlength=n_cats)
    pred_counts = np.bincount(pred_flat, minlength=n_cats)
    true_pct = true_counts / max(true_counts.sum(), 1)
    pred_pct = pred_counts / max(pred_counts.sum(), 1)

    x = np.arange(n_cats)
    width = 0.35
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - width / 2, true_pct, width, label="Observed")
    ax.bar(x + width / 2, pred_pct, width, label="Predicted")
    ax.set_xlabel("Category")
    ax.set_ylabel("Proportion")
    ax.set_title("Category Proportions")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "category_proportions.png", dpi=150)
    plt.close(fig)


def plot_posterior_predictive_scores(out_dir: Path, probs, targets, mask) -> None:
    rng = np.random.default_rng(0)
    n_cats = probs.shape[-1]
    masked_probs = probs[mask]
    preds = np.array([rng.choice(n_cats, p=p) for p in masked_probs])
    true_flat = targets[mask].reshape(-1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(true_flat, bins=np.arange(n_cats + 1) - 0.5, alpha=0.6, density=True, label="Observed")
    ax.hist(preds, bins=np.arange(n_cats + 1) - 0.5, alpha=0.6, density=True, label="Replicated")
    ax.set_xlabel("Category")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_predictive.png", dpi=150)
    plt.close(fig)


def plot_theta_trajectories(out_dir: Path, theta, mask, n_students: int = 3) -> None:
    n_traits = theta.shape[-1]
    cols = min(3, n_traits)
    rows = int(np.ceil(n_traits / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for d in range(n_traits):
        r, c = divmod(d, cols)
        ax = axes[r][c]
        for s in range(min(n_students, theta.shape[0])):
            valid = mask[s]
            ax.plot(theta[s][valid, d], label=f"student {s}")
        ax.set_title(f"Theta dim {d}")
        ax.grid(True, alpha=0.3)
    for idx in range(n_traits, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "theta_trajectories.png", dpi=150)
    plt.close(fig)


def plot_theta_band(out_dir: Path, theta, mask) -> None:
    n_traits = theta.shape[-1]
    max_len = theta.shape[1]
    cols = min(3, n_traits)
    rows = int(np.ceil(n_traits / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for d in range(n_traits):
        r, c = divmod(d, cols)
        ax = axes[r][c]
        means = []
        stds = []
        for t in range(max_len):
            valid = mask[:, t]
            if not valid.any():
                means.append(np.nan)
                stds.append(np.nan)
                continue
            vals = theta[valid, t, d]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)
        xs = np.arange(max_len)
        ax.plot(xs, means, color="C0")
        ax.fill_between(xs, means - stds, means + stds, color="C0", alpha=0.25)
        ax.set_title(f"Theta dim {d}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Theta")
        ax.grid(True, alpha=0.3)
    for idx in range(n_traits, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "theta_band.png", dpi=150)
    plt.close(fig)


def plot_attention_vs_alpha(out_dir: Path, attention, alpha, mask) -> None:
    if attention is None:
        return
    attn = attention[mask]
    alpha_norm = np.linalg.norm(alpha, axis=-1)[mask]
    entropy = -np.sum(attn * np.log(attn + 1e-8), axis=-1)
    if entropy.size > 5000:
        idx = np.random.default_rng(0).choice(entropy.size, size=5000, replace=False)
        entropy = entropy[idx]
        alpha_norm = alpha_norm[idx]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(alpha_norm, entropy, alpha=0.5, s=10)
    ax.set_xlabel("Alpha norm")
    ax.set_ylabel("Attention entropy")
    ax.set_title("Attention Entropy vs Discrimination")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_vs_alpha.png", dpi=150)
    plt.close(fig)


def _project_memory_values(
    value_memory: np.ndarray,
    theta_proj: Optional[torch.nn.Linear],
) -> np.ndarray:
    if theta_proj is None:
        return value_memory
    weight = theta_proj.weight.detach().cpu().numpy()
    bias = theta_proj.bias.detach().cpu().numpy()
    projected = value_memory @ weight.T + bias
    return projected


def _rank(x: np.ndarray) -> np.ndarray:
    if _rankdata is not None:
        return _rankdata(x)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    rx = _rank(x)
    ry = _rank(y)
    if np.std(rx) < 1e-8 or np.std(ry) < 1e-8:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_attention_slot_maps(
    out_dir: Path,
    attention: np.ndarray,
    value_memory: Optional[np.ndarray],
    theta_proj: Optional[torch.nn.Linear],
    mask: np.ndarray,
    max_students: int = 5,
) -> None:
    if attention is None:
        return
    n_students = min(max_students, attention.shape[0])
    slot_dim_map = None
    if value_memory is not None:
        projected = _project_memory_values(value_memory, theta_proj)
        slot_dim_map = np.mean(np.abs(projected), axis=(0, 1))
    for s in range(n_students):
        valid = mask[s]
        attn_s = attention[s][valid]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im = axes[0].imshow(attn_s.T, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Memory slot")
        axes[0].set_title(f"Attention heatmap (student {s})")
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        if slot_dim_map is not None:
            im2 = axes[1].imshow(slot_dim_map, aspect="auto", origin="lower", cmap="magma")
            axes[1].set_xlabel("Trait dim")
            axes[1].set_ylabel("Memory slot")
            axes[1].set_title("Slot -> trait magnitude")
            fig.colorbar(im2, ax=axes[1], fraction=0.046)
        else:
            axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"attention_slot_map_student{s}.png", dpi=150)
        plt.close(fig)


def plot_theta_alignment(
    out_dir: Path,
    summary_theta: np.ndarray,
    read: Optional[np.ndarray],
    theta_proj: Optional[torch.nn.Linear],
    mask: np.ndarray,
    max_students: int = 3,
) -> None:
    if read is None:
        return
    memory_theta = _project_memory_values(read, theta_proj)
    n_traits = summary_theta.shape[-1]
    n_students = min(max_students, summary_theta.shape[0])
    for s in range(n_students):
        valid = mask[s]
        cols = min(3, n_traits)
        rows = int(np.ceil(n_traits / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        for d in range(n_traits):
            r, c = divmod(d, cols)
            ax = axes[r][c]
            ax.plot(summary_theta[s][valid, d], label="summary")
            ax.plot(memory_theta[s][valid, d], label="memory", linestyle="--")
            ax.set_title(f"Theta dim {d}")
            ax.grid(True, alpha=0.3)
        for idx in range(n_traits, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"theta_alignment_student{s}.png", dpi=150)
        plt.close(fig)


def plot_item_characteristic_curves(
    out_dir: Path,
    questions: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    mask: np.ndarray,
    n_traits: int,
    n_cats: int,
    max_items_per_dim: int = 2,
) -> None:
    q_flat = questions[mask].reshape(-1)
    alpha_flat = alpha[mask].reshape(-1, n_traits)
    beta_flat = beta[mask].reshape(-1, n_cats - 1 if n_cats > 1 else 1)
    items = np.unique(q_flat)
    if items.size == 0:
        return
    item_alpha = {}
    item_beta = {}
    for item_id in items:
        idx = q_flat == item_id
        if idx.sum() == 0:
            continue
        item_alpha[item_id] = alpha_flat[idx].mean(axis=0)
        item_beta[item_id] = beta_flat[idx].mean(axis=0)

    grid = np.linspace(-3, 3, 121)
    for d in range(n_traits):
        ranked = sorted(item_alpha.items(), key=lambda kv: kv[1][d], reverse=True)
        chosen = [item for item, _ in ranked[:max_items_per_dim]]
        if not chosen:
            continue
        fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 4), squeeze=False)
        for idx, item_id in enumerate(chosen):
            alpha_vec = item_alpha[item_id]
            beta_vec = item_beta[item_id]
            theta = np.zeros((grid.size, n_traits))
            theta[:, d] = grid
            dot = np.sum(theta * alpha_vec, axis=-1)
            alpha_scale = np.linalg.norm(alpha_vec) / max(np.sqrt(n_traits), 1.0)
            logits = np.zeros((grid.size, n_cats))
            for k in range(1, n_cats):
                logits[:, k] = np.sum(dot[:, None] - beta_vec[:k] * alpha_scale, axis=-1)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            ax = axes[0][idx]
            for k in range(n_cats):
                ax.plot(grid, probs[:, k], label=f"cat {k}")
            ax.set_title(f"Item {item_id} (dim {d})")
            ax.set_xlabel("Theta")
            ax.set_ylabel("P(category)")
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"icc_dim{d}.png", dpi=150)
        plt.close(fig)


def plot_item_slot_attention_heatmap(
    out_dir: Path,
    attention: np.ndarray,
    questions: np.ndarray,
    mask: np.ndarray,
    theta: Optional[np.ndarray] = None,
) -> None:
    if attention is None:
        return
    attn_flat = attention[mask]
    items_flat = questions[mask].reshape(-1)
    if attn_flat.size == 0:
        return
    n_items = int(items_flat.max()) + 1
    n_slots = attn_flat.shape[-1]
    item_slot = np.zeros((n_items, n_slots), dtype=np.float32)
    counts = np.zeros(n_items, dtype=np.float32)
    for idx, item_id in enumerate(items_flat):
        item_slot[item_id] += attn_flat[idx]
        counts[item_id] += 1.0
    counts = np.maximum(counts, 1.0)
    item_slot = item_slot / counts[:, None]

    dominant_slot = np.argmax(item_slot, axis=1)
    max_weight = np.max(item_slot, axis=1)
    order = np.lexsort((-max_weight, dominant_slot))
    sorted_matrix = item_slot[order]
    sorted_groups = dominant_slot[order]

    slot_dim_map = None
    if theta is not None:
        attn_flat = attention[mask]
        theta_flat = theta[mask]
        if attn_flat.size and theta_flat.size:
            slot_dim_map = np.einsum("ns,nd->sd", attn_flat, theta_flat)
            slot_dim_map = np.abs(slot_dim_map) / max(attn_flat.shape[0], 1)

    if slot_dim_map is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_item_slot = axes[0] if slot_dim_map is not None else axes[0]
    im = ax_item_slot.imshow(sorted_matrix, aspect="auto", origin="lower", cmap="viridis")
    ax_item_slot.set_ylabel("Items (grouped by dominant slot)")
    ax_item_slot.set_title("Item ↔ Slot Attention (aggregate)")
    fig.colorbar(im, ax=ax_item_slot, fraction=0.046)

    boundaries = np.where(np.diff(sorted_groups) != 0)[0]
    for b in boundaries:
        ax_item_slot.axhline(b + 0.5, color="white", linewidth=0.5, alpha=0.6)

    if slot_dim_map is not None:
        im2 = axes[1].imshow(slot_dim_map.T, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_xlabel("Memory slot")
        axes[1].set_ylabel("Latent dim")
        axes[1].set_title("Slot ↔ Dim (aggregate magnitude)")
        fig.colorbar(im2, ax=axes[1], fraction=0.046)
    else:
        ax_item_slot.set_xlabel("Memory slot")

    fig.tight_layout()
    fig.savefig(out_dir / "item_slot_attention_heatmap.png", dpi=150)
    plt.close(fig)


def plot_theta_attention_alignment(
    out_dir: Path,
    attention: np.ndarray,
    value_memory: Optional[np.ndarray],
    theta_proj: Optional[torch.nn.Linear],
    alpha: np.ndarray,
    questions: np.ndarray,
    mask: np.ndarray,
    top_m: int = 10,
    max_interactions: int = 5000,
) -> None:
    if attention is None or value_memory is None:
        return
    n_traits = alpha.shape[-1]
    slot_states = _project_memory_values(value_memory, theta_proj)
    if slot_states.shape[-1] != n_traits:
        return

    attn_flat = attention[mask]
    alpha_flat = alpha[mask]
    questions_flat = questions[mask].reshape(-1)
    slot_flat = slot_states[mask]
    total = attn_flat.shape[0]
    if total == 0:
        return

    if total > max_interactions:
        rng = np.random.default_rng(0)
        idx = rng.choice(total, size=max_interactions, replace=False)
        attn_flat = attn_flat[idx]
        alpha_flat = alpha_flat[idx]
        questions_flat = questions_flat[idx]
        slot_flat = slot_flat[idx]

    rhos = []
    rhos_null = []
    js_divs = []
    item_rhos: Dict[int, List[float]] = {}
    item_alpha_norm: Dict[int, List[float]] = {}
    item_entropy: Dict[int, List[float]] = {}

    rng = np.random.default_rng(0)
    for w, a, item_id, s_slots in zip(attn_flat, alpha_flat, questions_flat, slot_flat):
        m = min(top_m, w.shape[0])
        top_idx = np.argsort(w)[-m:]
        w_top = w[top_idx]
        s_top = s_slots[top_idx]
        slot_score = s_top @ a
        c = w_top * slot_score
        rho = _spearman(w_top, np.abs(c))
        rhos.append(rho)

        w_shuff = rng.permutation(w_top)
        c_shuff = w_shuff * slot_score
        rhos_null.append(_spearman(w_shuff, np.abs(c_shuff)))

        c_norm = np.abs(c)
        c_norm = c_norm / max(c_norm.sum(), 1e-8)
        w_norm = w_top / max(w_top.sum(), 1e-8)
        m_dist = 0.5 * (w_norm + c_norm)
        js = 0.5 * (
            np.sum(w_norm * np.log((w_norm + 1e-8) / (m_dist + 1e-8)))
            + np.sum(c_norm * np.log((c_norm + 1e-8) / (m_dist + 1e-8)))
        )
        js_divs.append(js)

        if item_id > 0 and not np.isnan(rho):
            item_rhos.setdefault(int(item_id), []).append(rho)
            item_alpha_norm.setdefault(int(item_id), []).append(float(np.linalg.norm(a)))
            entropy = -np.sum(w_norm * np.log(w_norm + 1e-8))
            item_entropy.setdefault(int(item_id), []).append(float(entropy))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([np.array(rhos), np.array(rhos_null)], bins=30, label=["observed", "shuffled"], alpha=0.7)
    axes[0].set_title("Spearman(w, |contrib|)")
    axes[0].set_xlabel("Correlation")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(js_divs, bins=30, color="C2", alpha=0.8)
    axes[1].set_title("JS(w, normalized contrib)")
    axes[1].set_xlabel("JS divergence")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "theta_attention_alignment.png", dpi=150)
    plt.close(fig)

    if item_rhos:
        item_ids = sorted(item_rhos.keys())
        rho_med = np.array([np.median(item_rhos[i]) for i in item_ids])
        alpha_norm = np.array([np.mean(item_alpha_norm[i]) for i in item_ids])
        entropy = np.array([np.mean(item_entropy[i]) for i in item_ids])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(alpha_norm, rho_med, s=12, alpha=0.6)
        axes[0].set_title("Item median rho vs |alpha|")
        axes[0].set_xlabel("Alpha norm")
        axes[0].set_ylabel("Median rho")
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(entropy, rho_med, s=12, alpha=0.6)
        axes[1].set_title("Item median rho vs attention entropy")
        axes[1].set_xlabel("Attention entropy")
        axes[1].set_ylabel("Median rho")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "theta_attention_item_scatter.png", dpi=150)
        plt.close(fig)




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.base.device if torch.cuda.is_available() else "cpu")

    loader = DataLoaderManager(config.data.dataset_name, data_root=config.data.data_root)
    dataloaders = loader.build_dataloaders(batch_size=config.training.batch_size)

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

    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])

    outputs = _collect_outputs(model, dataloaders["test"], device)
    memory_outputs = _collect_outputs(
        model,
        dataloaders["test"],
        device,
        capture_memory=True,
        max_students=200,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_calibration_thresholds(out_dir, outputs["probs"], outputs["responses"], outputs["mask"])
    plot_category_proportions(out_dir, outputs["probs"], outputs["responses"], outputs["mask"])
    plot_posterior_predictive_scores(out_dir, outputs["probs"], outputs["responses"], outputs["mask"])
    plot_theta_trajectories(out_dir, outputs["theta"], outputs["mask"])
    plot_theta_band(out_dir, outputs["theta"], outputs["mask"])
    plot_attention_vs_alpha(out_dir, outputs["attention"], outputs["alpha"], outputs["mask"])
    plot_attention_slot_maps(
        out_dir,
        memory_outputs["attention"],
        memory_outputs["value_memory"],
        model.theta_from_memory if model.theta_from_memory is not None else None,
        memory_outputs["mask"],
        max_students=5,
    )
    plot_theta_alignment(
        out_dir,
        memory_outputs["theta"],
        memory_outputs["read"],
        model.theta_from_memory if model.theta_from_memory is not None else None,
        memory_outputs["mask"],
        max_students=3,
    )
    plot_item_characteristic_curves(
        out_dir,
        outputs["questions"],
        outputs["alpha"],
        outputs["beta"],
        outputs["mask"],
        config.model.n_traits,
        config.model.n_cats,
    )
    plot_item_slot_attention_heatmap(
        out_dir,
        memory_outputs["attention"],
        memory_outputs["questions"],
        memory_outputs["mask"],
        memory_outputs["theta"],
    )
    plot_theta_attention_alignment(
        out_dir,
        memory_outputs["attention"],
        memory_outputs["value_memory"],
        model.theta_from_memory if model.theta_from_memory is not None else None,
        memory_outputs["alpha"],
        memory_outputs["questions"],
        memory_outputs["mask"],
    )


if __name__ == "__main__":
    main()
