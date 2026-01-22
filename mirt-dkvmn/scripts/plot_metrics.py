"""Plot training metrics from metrics.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path: Path) -> dict:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {name: rows[name] for name in rows.dtype.names}


def plot_series(out_path: Path, epochs, series, title, ylabel):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(epochs, series, marker="o", markersize=3, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_metrics(metrics_path)
    epochs = data["epoch"]

    plot_series(out_dir / "loss.png", epochs, data["train_loss"], "Train loss", "Loss")
    plot_series(out_dir / "val_loss.png", epochs, data["val_loss"], "Validation loss", "Loss")
    plot_series(out_dir / "qwk.png", epochs, data["qwk"], "Validation QWK", "QWK")
    plot_series(out_dir / "cat_acc.png", epochs, data["cat_acc"], "Validation cat acc", "Accuracy")
    plot_series(out_dir / "ece.png", epochs, data["ece"], "Validation ECE", "ECE")
    plot_series(out_dir / "nll.png", epochs, data["nll"], "Validation NLL", "NLL")


if __name__ == "__main__":
    main()
