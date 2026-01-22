"""Plot per-item influence on each latent dimension."""

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

    sums = np.zeros((config.model.n_questions, config.model.n_traits), dtype=np.float64)
    counts = np.zeros((config.model.n_questions, config.model.n_traits), dtype=np.float64)

    with torch.no_grad():
        for batch in dataloader:
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            mask = batch["mask"].to(device)

            _ = model(questions, responses)
            reads = model.last_read
            if reads is None:
                raise RuntimeError("Model did not expose last_read for influence plots.")

            q_ids = questions.cpu().numpy()
            msk = mask.cpu().numpy()
            read_vals = reads.abs().cpu().numpy()

            for i in range(q_ids.shape[0]):
                valid = msk[i]
                q_seq = q_ids[i][valid]
                r_seq = read_vals[i][valid]
                for q_id, r_vec in zip(q_seq, r_seq):
                    idx = int(q_id) - 1
                    if idx < 0 or idx >= config.model.n_questions:
                        continue
                    sums[idx] += r_vec
                    counts[idx] += 1

    means = np.divide(sums, np.clip(counts, 1.0, None))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(config.model.n_questions)
    for dim in range(config.model.n_traits):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, means[:, dim], linewidth=1.0)
        ax.set_title(f"Item influence on dim {dim}")
        ax.set_xlabel("Item id")
        ax.set_ylabel("Mean |read|")
        fig.tight_layout()
        fig.savefig(out_dir / f"item_influence_dim_{dim}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
