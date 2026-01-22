"""Visualize DKVMN key/value memory state and projections."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT


def pca_2d(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    return u[:, :2] * s[:2]

def heatmap_config(values: np.ndarray) -> tuple[str, float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin < 0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        return "coolwarm", -bound, bound
    return "viridis", vmin, vmax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.base.device if torch.cuda.is_available() else "cpu")

    loader = DataLoaderManager(config.data.dataset_name, data_root=config.data.data_root)
    dataloaders = loader.build_dataloaders(batch_size=config.training.batch_size, split_ratio=0.8, val_ratio=0.1)
    dataloader = dataloaders[args.split]

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
    model.eval()

    batch = next(iter(dataloader))
    questions = batch["questions"].to(device)
    responses = batch["responses"].to(device)

    with torch.no_grad():
        _ = model(questions, responses)

    key_mem = model.memory.key_memory.detach().cpu().numpy()
    value_mem = model.memory.value_memory.detach().cpu().numpy()
    value_mean = value_mem.mean(axis=0)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap, vmin, vmax = heatmap_config(key_mem)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(key_mem, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Key memory heatmap")
    ax.set_xlabel("Key dimension")
    ax.set_ylabel("Memory slot")
    fig.colorbar(im, ax=ax, label="Value")
    fig.tight_layout()
    fig.savefig(out_dir / "key_memory_heatmap.png", dpi=150)
    plt.close(fig)

    cmap, vmin, vmax = heatmap_config(value_mean)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(value_mean, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Value memory heatmap (batch mean)")
    ax.set_xlabel("Value dimension")
    ax.set_ylabel("Memory slot")
    fig.colorbar(im, ax=ax, label="Value")
    fig.tight_layout()
    fig.savefig(out_dir / "value_memory_heatmap.png", dpi=150)
    plt.close(fig)

    key_proj = pca_2d(key_mem)
    val_proj = pca_2d(value_mean)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(key_proj[:, 0], key_proj[:, 1], s=12, alpha=0.7)
    ax.set_title("Key memory PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_dir / "key_memory_pca.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(val_proj[:, 0], val_proj[:, 1], s=12, alpha=0.7)
    ax.set_title("Value memory PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_dir / "value_memory_pca.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
