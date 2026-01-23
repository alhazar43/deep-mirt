"""Training entry point (placeholder)."""

import argparse
import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT
from mirt_dkvmn.training.losses import CombinedOrdinalLoss
from mirt_dkvmn.training.trainer import Trainer
from mirt_dkvmn.utils.checkpoint import save_checkpoint
from mirt_dkvmn.utils.logging import get_logger

from plot_metrics import plot_metrics
from plot_recovery import plot_recovery

def _metrics_header(metric_keys: list[str]) -> list[str]:
    ordered = [
        "qwk",
        "cat_acc",
        "mae",
        "balanced_acc",
        "within_one_acc",
        "spearman",
        "ece",
        "nll",
    ]
    extras = [key for key in metric_keys if key not in ordered]
    return ["epoch", "train_loss", "val_loss"] + ordered + sorted(extras)


def _read_metrics_header(metrics_path: Path) -> list[str] | None:
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
    if not first:
        return None
    return [col.strip() for col in first.split(",") if col.strip()]


def _write_confusion(confusion: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path.with_suffix(".csv"), confusion.cpu().numpy(), fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion.cpu().numpy(), cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--plot_only", action="store_true", help="Skip training and only generate plots")
    args = parser.parse_args()

    logger = get_logger("train")
    config = load_config(args.config)

    device = torch.device(config.base.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

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
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    loss_fn = CombinedOrdinalLoss(
        qwk_weight=config.training.qwk_weight,
        ordinal_weight=config.training.ordinal_weight,
    )
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=str(device),
        attention_entropy_weight=config.training.attention_entropy_weight,
        theta_norm_weight=config.training.theta_norm_weight,
        alpha_prior_weight=config.training.alpha_prior_weight,
        beta_prior_weight=config.training.beta_prior_weight,
        alpha_norm_weight=config.training.alpha_norm_weight,
        alpha_norm_target=config.training.alpha_norm_target,
        alpha_ortho_weight=config.training.alpha_ortho_weight,
    )

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    start_epoch = 1
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("step", 0)) + 1
        logger.info("Resumed from %s (start_epoch=%s)", args.resume, start_epoch)

    if not args.plot_only and start_epoch <= config.training.epochs:
        writer = None
        last_conf = None
        existing_header = _read_metrics_header(metrics_path)
        metrics_handle = metrics_path.open("a", newline="", encoding="utf-8")
        for epoch in range(start_epoch, config.training.epochs + 1):
            train_loss = trainer.train_epoch(dataloaders["train"])
            val_loss = float("nan")
            val_metrics = {}
            conf = None
            if dataloaders["valid"]:
                val_loss, val_metrics, conf = trainer.evaluate_epoch(dataloaders["valid"])
                logger.info(
                    "Epoch %s: train_loss=%.4f val_loss=%.4f qwk=%.4f cat_acc=%.4f ece=%.4f nll=%.4f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics.get("qwk", float("nan")),
                    val_metrics.get("cat_acc", float("nan")),
                    val_metrics.get("ece", float("nan")),
                    val_metrics.get("nll", float("nan")),
                )
            else:
                logger.info("Epoch %s: train_loss=%.4f val_loss=nan", epoch, train_loss)

            if writer is None:
                writer = csv.writer(metrics_handle)
                header = existing_header or _metrics_header(list(val_metrics.keys()))
                if not existing_header:
                    writer.writerow(header)

            row = [
                epoch,
                train_loss,
                val_loss,
            ]
            for key in header[3:]:
                row.append(val_metrics.get(key, float("nan")))
            writer.writerow(row)
            metrics_handle.flush()

            if conf is not None:
                _write_confusion(conf, output_dir / f"confusion_matrix_epoch_{epoch}")
                last_conf = conf

            save_checkpoint(str(output_dir / f"epoch_{epoch}.pt"), model, optimizer, step=epoch)

        metrics_handle.close()
        save_checkpoint(str(output_dir / "last.pt"), model, optimizer, step=config.training.epochs)
        logger.info("Saved checkpoint to %s", output_dir / "last.pt")
        if last_conf is not None:
            _write_confusion(last_conf, output_dir / "confusion_matrix_last")
    elif args.plot_only:
        logger.info("Plot-only mode: skipping training loop.")

    try:
        if metrics_path.exists():
            plot_metrics(metrics_path, output_dir / "metric_plots")
        else:
            logger.warning("Skipping metrics plots; missing %s", metrics_path)
        checkpoint_path = args.resume if args.resume else str(output_dir / "last.pt")
        plot_recovery(args.config, checkpoint_path, str(output_dir / "recovery_plots"))
        logger.info("Saved plots under %s", output_dir)
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    sample = next(iter(dataloaders["train"]))
    with torch.no_grad():
        theta, beta, alpha, probs = model(
            sample["questions"].to(device),
            sample["responses"].to(device),
        )
    logger.info("Output shapes: theta=%s beta=%s alpha=%s probs=%s", theta.shape, beta.shape, alpha.shape, probs.shape)


if __name__ == "__main__":
    main()
