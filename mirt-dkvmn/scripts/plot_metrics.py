"""Plot training metrics from metrics.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path: Path) -> dict:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {name: rows[name] for name in rows.dtype.names}


def plot_series(ax, epochs, series, title, ylabel):
    ax.plot(epochs, series, marker="o", markersize=3, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_metrics(metrics_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_metrics(metrics_path)
    epochs = data["epoch"]
    candidates = [
        ("train_loss", "Train loss", "Loss"),
        ("val_loss", "Validation loss", "Loss"),
        ("qwk", "Validation QWK", "QWK"),
        ("cat_acc", "Validation cat acc", "Accuracy"),
        ("mae", "Validation MAE", "MAE"),
        ("balanced_acc", "Balanced acc", "Accuracy"),
        ("within_one_acc", "Within-one acc", "Accuracy"),
        ("spearman", "Spearman", "Correlation"),
        ("ece", "Validation ECE", "ECE"),
        ("nll", "Validation NLL", "NLL"),
    ]
    series = [(key, title, ylabel) for key, title, ylabel in candidates if key in data]
    if not series:
        return
    cols = 2
    rows = int(np.ceil(len(series) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), squeeze=False)
    for idx, (key, title, ylabel) in enumerate(series):
        r, c = divmod(idx, cols)
        plot_series(axes[r][c], epochs, data[key], title, ylabel)
    for idx in range(len(series), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_overview.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    args = parser.parse_args()

    plot_metrics(Path(args.metrics), Path(args.output))


if __name__ == "__main__":
    main()
