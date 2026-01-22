"""Plot per-student item influence over time as heatmaps."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT


def load_compatible_state(model, payload):
    model_state = model.state_dict()
    filtered = {}
    for key, value in payload["model_state"].items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
    model.load_state_dict(filtered, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--n_students", type=int, default=10)
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
    load_compatible_state(model, payload)
    model.eval()

    reads = []
    masks = []

    with torch.no_grad():
        for batch in dataloader:
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            mask = batch["mask"].to(device)
            _ = model(questions, responses)
            read = model.last_read
            if read is None:
                raise RuntimeError("Model did not expose last_read for heatmaps.")
            reads.append(read.abs().cpu().numpy())
            masks.append(mask.cpu().numpy())
            if sum(r.shape[0] for r in reads) >= args.n_students:
                break

    read_all = np.concatenate(reads, axis=0)[: args.n_students]
    mask_all = np.concatenate(masks, axis=0)[: args.n_students]
    max_len = read_all.shape[1]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in range(config.model.n_traits):
        values = np.full((args.n_students, max_len), np.nan, dtype=np.float32)
        for i in range(args.n_students):
            valid = mask_all[i]
            values[i, valid] = read_all[i, valid, dim]

        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Item influence over time (dim {dim})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Student")
        fig.colorbar(im, ax=ax, label="Mean |read|")
        fig.tight_layout()
        fig.savefig(out_dir / f"item_influence_heatmap_dim_{dim}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
