"""Rebuild metrics.csv from checkpoints."""

import argparse
import csv
from pathlib import Path

import torch

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT
from mirt_dkvmn.training.losses import CombinedOrdinalLoss
from mirt_dkvmn.training.trainer import Trainer
from plot_metrics import plot_metrics


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts", required=True)
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

    artifacts = Path(args.artifacts)
    checkpoints = sorted(artifacts.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {artifacts}")

    metrics_path = artifacts / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for ckpt in checkpoints:
            payload = torch.load(ckpt, map_location=device)
            model.load_state_dict(payload["model_state"])
            epoch = int(ckpt.stem.split("_")[-1])
            val_loss, val_metrics, _ = trainer.evaluate_epoch(dataloaders["valid"])
            if writer is None:
                header = _metrics_header(list(val_metrics.keys()))
                writer = csv.writer(handle)
                writer.writerow(header)
            row = [epoch, float("nan"), val_loss]
            for key in header[3:]:
                row.append(val_metrics.get(key, float("nan")))
            writer.writerow(row)

    plot_metrics(metrics_path, artifacts / "metric_plots")


if __name__ == "__main__":
    main()
