#!/usr/bin/env python3
"""Plot training metrics from a metrics.csv file.

Usage::

    PYTHONPATH=src python scripts/plot_metrics.py \\
        --metrics outputs/base/metrics.csv \\
        --output  outputs/base/metric_plots
"""

from __future__ import annotations
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows OpenMP conflict

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server / headless use
import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, list[float]]:
    """Load a CSV file into a dict of column_name -> list of values."""
    data: dict[str, list] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    val = v
                data.setdefault(k, []).append(val)
    return data


def plot_panel(
    ax: plt.Axes,
    epochs: list,
    series: dict[str, list],
    title: str,
    ylabel: str,
    legend_loc: str = "best",
) -> None:
    for label, values in series.items():
        ax.plot(epochs, values, label=label, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics CSV.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_csv(metrics_path)
    epochs = data.get("epoch", list(range(1, len(next(iter(data.values()))) + 1)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")

    # Loss
    plot_panel(
        axes[0, 0], epochs,
        {"train": data.get("train_loss", []), "val": data.get("val_loss", [])},
        "Loss", "Loss",
    )

    # Accuracy
    plot_panel(
        axes[0, 1], epochs,
        {
            "train acc": data.get("train_accuracy", []),
            "val cat acc": data.get("val_categorical_accuracy", []),
            "val ord acc": data.get("val_ordinal_accuracy", []),
        },
        "Accuracy", "Accuracy",
    )

    # QWK
    plot_panel(
        axes[0, 2], epochs,
        {"QWK": data.get("val_qwk", [])},
        "Quadratic Weighted Kappa", "QWK",
    )

    # MAE
    plot_panel(
        axes[1, 0], epochs,
        {"MAE": data.get("val_mae", [])},
        "Mean Absolute Error", "MAE",
    )

    # Spearman
    plot_panel(
        axes[1, 1], epochs,
        {"Spearman": data.get("val_spearman", [])},
        "Spearman Correlation", "rho",
    )

    # LR + grad norm
    ax = axes[1, 2]
    if "lr" in data:
        ax.plot(epochs, data["lr"], label="LR", color="tab:orange")
        ax.set_ylabel("Learning Rate", color="tab:orange")
    if "train_grad_norm" in data:
        ax2 = ax.twinx()
        ax2.plot(epochs, data["train_grad_norm"], label="Grad Norm",
                 color="tab:blue", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Grad Norm", color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_title("LR & Gradient Norm")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "training_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
