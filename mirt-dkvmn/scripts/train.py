"""Training entry point (placeholder)."""

import argparse
import csv
from pathlib import Path

import torch

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT
from mirt_dkvmn.training.losses import CombinedOrdinalLoss
from mirt_dkvmn.training.trainer import Trainer
from mirt_dkvmn.utils.checkpoint import save_checkpoint
from mirt_dkvmn.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
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
    )

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "qwk", "cat_acc", "ece", "nll"])

        for epoch in range(1, config.training.epochs + 1):
            train_loss = trainer.train_epoch(dataloaders["train"])
            val_loss = float("nan")
            val_metrics = {}
            if dataloaders["valid"]:
                val_loss, val_metrics = trainer.evaluate_epoch(dataloaders["valid"])
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

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics.get("qwk", float("nan")),
                    val_metrics.get("cat_acc", float("nan")),
                    val_metrics.get("ece", float("nan")),
                    val_metrics.get("nll", float("nan")),
                ]
            )
            handle.flush()

            save_checkpoint(str(output_dir / f"epoch_{epoch}.pt"), model, optimizer, step=epoch)

    save_checkpoint(str(output_dir / "last.pt"), model, optimizer, step=config.training.epochs)
    logger.info("Saved checkpoint to %s", output_dir / "last.pt")

    sample = next(iter(dataloaders["train"]))
    with torch.no_grad():
        theta, beta, alpha, probs = model(
            sample["questions"].to(device),
            sample["responses"].to(device),
        )
    logger.info("Output shapes: theta=%s beta=%s alpha=%s probs=%s", theta.shape, beta.shape, alpha.shape, probs.shape)


if __name__ == "__main__":
    main()
