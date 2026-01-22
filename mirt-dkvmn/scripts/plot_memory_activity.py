"""Plot memory attention activity for DKVMN."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--gif", action="store_true", help="Generate an attention GIF")
    parser.add_argument("--fps", type=int, default=6)
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

    all_attn = []
    with torch.no_grad():
        for batch in dataloader:
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            _ = model(questions, responses)
            if model.last_attention is not None:
                all_attn.append(model.last_attention.cpu().numpy())

    if not all_attn:
        raise RuntimeError("No attention weights collected for plotting.")

    min_len = min(arr.shape[1] for arr in all_attn)
    if min_len <= 0:
        raise RuntimeError("No valid sequence lengths for attention.")
    trimmed = [arr[:, :min_len, :] for arr in all_attn]
    attn = np.concatenate(trimmed, axis=0)  # (batch, seq, memory_size)
    mean_attn = attn.mean(axis=(0, 1))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(mean_attn.shape[0]), mean_attn)
    ax.set_title("Mean Attention per Memory Slot")
    ax.set_xlabel("Memory slot")
    ax.set_ylabel("Mean attention")
    fig.tight_layout()
    fig.savefig(out_dir / "memory_attention_bar.png", dpi=150)
    plt.close(fig)

    mean_time = attn.mean(axis=0).T  # (memory_size, seq)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(mean_time, aspect="auto", cmap="viridis")
    ax.set_title("Mean Attention Over Time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Memory slot")
    fig.colorbar(im, ax=ax, label="Attention")
    fig.tight_layout()
    fig.savefig(out_dir / "memory_attention_heatmap.png", dpi=150)
    plt.close(fig)

    if args.gif:
        mean_by_t = attn.mean(axis=0)  # (seq, memory_size)
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(np.arange(mean_by_t.shape[1]), mean_by_t[0])
        ax.set_ylim(0, mean_by_t.max() * 1.1)
        ax.set_title("Mean Attention per Slot (time 0)")
        ax.set_xlabel("Memory slot")
        ax.set_ylabel("Mean attention")

        def update(frame_idx):
            for bar, height in zip(bars, mean_by_t[frame_idx]):
                bar.set_height(height)
            ax.set_title(f"Mean Attention per Slot (time {frame_idx})")
            return bars

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=mean_by_t.shape[0],
            interval=1000 / max(args.fps, 1),
            blit=False,
        )
        gif_path = out_dir / "memory_attention.gif"
        anim.save(gif_path, writer="pillow", fps=args.fps)
        plt.close(fig)


if __name__ == "__main__":
    main()
