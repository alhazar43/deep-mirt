#!/usr/bin/env python3
"""Training entry point for Deep-GPCM (memirt).

Usage::

    PYTHONPATH=src python scripts/train.py --config configs/smoke.yaml
    PYTHONPATH=src python scripts/train.py --config configs/base.yaml --resume
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

# --- Package imports -------------------------------------------------------
from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule
from kt_gpcm.models.kt_gpcm import DeepGPCM
from kt_gpcm.training.losses import CombinedLoss, compute_class_weights
from kt_gpcm.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_device(cfg_device: str) -> torch.device:
    """Return a torch.device, falling back to CPU if CUDA is unavailable."""
    if cfg_device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(cfg_device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg, device: torch.device) -> DeepGPCM:
    model = DeepGPCM(**vars(cfg.model))
    return model.to(device)


def append_csv_row(path: Path, row: dict, write_header: bool) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepGPCM model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name (overrides data.dataset_name and experiment_name). "
                             "n_questions and n_categories are read from the dataset's metadata.json automatically.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from the config.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last.pt in the outputs directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.data.dataset_name = args.dataset
        cfg.base.experiment_name = args.dataset
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    # ---- Config -----------------------------------------------------------
    device = resolve_device(cfg.base.device)
    set_seed(cfg.base.seed)
    log.info("Experiment: %s | device: %s | seed: %d",
             cfg.base.experiment_name, device, cfg.base.seed)

    # ---- Artifacts directory ----------------------------------------------
    artifact_dir = Path("outputs") / cfg.base.experiment_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = artifact_dir / "metrics.csv"

    # ---- Data -------------------------------------------------------------
    data_mgr = DataModule(cfg)
    train_loader, test_loader = data_mgr.build()
    log.info(
        "Dataset: %s | train batches: %d | test batches: %d",
        cfg.data.dataset_name, len(train_loader), len(test_loader),
    )

    # ---- Class weights (from training data) -------------------------------
    if cfg.training.weighted_ordinal_weight > 0.0:
        log.info("Computing class weights from training data …")
        all_targets = data_mgr.all_train_targets()
        class_weights = compute_class_weights(
            all_targets, cfg.model.n_categories, strategy="sqrt_balanced"
        ).to(device)
        log.info("Class weights: %s", class_weights.tolist())
    else:
        class_weights = None

    # ---- Model ------------------------------------------------------------
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", n_params)

    start_epoch = 0
    if args.resume:
        ckpt_path = artifact_dir / "last.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            start_epoch = state.get("epoch", 0)
            log.info("Resumed from %s (epoch %d)", ckpt_path, start_epoch)

    # ---- Loss + optimiser + scheduler -------------------------------------
    t = cfg.training
    loss_fn = CombinedLoss(
        n_categories=cfg.model.n_categories,
        class_weights=class_weights,
        focal_weight=t.focal_weight,
        weighted_ordinal_weight=t.weighted_ordinal_weight,
        ordinal_penalty=t.ordinal_penalty,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=t.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t.lr_factor, patience=t.lr_patience
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        grad_clip=t.grad_clip,
        attention_entropy_weight=t.attention_entropy_weight,
        theta_norm_weight=t.theta_norm_weight,
        alpha_prior_weight=t.alpha_prior_weight,
        beta_prior_weight=t.beta_prior_weight,
    )

    # ---- Training loop ----------------------------------------------------
    best_qwk = -1.0
    best_epoch = 0
    header_written = metrics_csv.exists() and args.resume

    log.info(
        "%-6s %-10s %-10s %-10s %-10s %-10s %-10s",
        "Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "QWK", "LR",
    )
    log.info("-" * 70)

    for epoch in range(start_epoch, start_epoch + t.epochs):
        ep_start = time.time()

        train_stats = trainer.train_epoch(train_loader)
        val_stats = trainer.evaluate_epoch(test_loader)

        val_qwk = val_stats["metrics"]["qwk"]
        val_acc = val_stats["metrics"]["categorical_accuracy"]
        current_lr = optimizer.param_groups[0]["lr"]

        # LR scheduler step (driven by QWK)
        scheduler.step(val_qwk)

        # Best model checkpoint
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                       artifact_dir / "best.pt")

        # Per-epoch checkpoint
        torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                   artifact_dir / "last.pt")

        ep_time = time.time() - ep_start

        log.info(
            "%-6d %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.6f",
            epoch + 1,
            train_stats["loss"],
            train_stats["accuracy"],
            val_stats["loss"],
            val_acc,
            val_qwk,
            current_lr,
        )

        # CSV logging
        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "train_grad_norm": train_stats["grad_norm"],
            "val_loss": val_stats["loss"],
            "val_categorical_accuracy": val_acc,
            "val_ordinal_accuracy": val_stats["metrics"]["ordinal_accuracy"],
            "val_qwk": val_qwk,
            "val_mae": val_stats["metrics"]["mae"],
            "val_spearman": val_stats["metrics"]["spearman"],
            "lr": current_lr,
            "epoch_time_s": round(ep_time, 2),
        }
        append_csv_row(metrics_csv, row, write_header=not header_written)
        header_written = True

    log.info("Training complete. Best QWK: %.4f at epoch %d", best_qwk, best_epoch)
    log.info("Artifacts saved to: %s", artifact_dir)


if __name__ == "__main__":
    main()
