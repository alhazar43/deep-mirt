"""End-to-end smoke test for ``rl/scripts/train_ppo.py``.

Runs the full training pipeline (synthetic adapter -> frozen MAGPCM ->
item cache -> OrdRecEnv -> PPO) for two updates on a tiny config and
asserts the run completes with checkpoints, metrics.csv and
summary.json in a sane schema. This is the script that produces the
E4.5/E5 training numbers; before this test it had zero coverage.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest
import torch


pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


EXPECTED_METRIC_COLUMNS = {
    "update", "mean_return", "mean_length",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
    "n_grad_steps", "early_stop", "entropy_coef",
    "r_info", "r_cost", "r_expo", "r_voi",
    "elapsed_s",
}

_FLOAT_COLUMNS = (
    "mean_return", "policy_loss", "value_loss", "entropy",
    "approx_kl", "clipfrac", "r_info", "r_cost", "r_expo", "r_voi",
)


def test_train_ppo_end_to_end(
    train_ppo_module: Any, smoke_cfg: Dict[str, Any],
) -> None:
    summary = train_ppo_module.train(smoke_cfg)

    out_dir = (
        Path(smoke_cfg["output_dir"]) / smoke_cfg["experiment_name"]
    )
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    metrics_csv = out_dir / "metrics.csv"
    summary_json = out_dir / "summary.json"

    # 1. All four artifacts exist.
    assert best_path.exists(), "best checkpoint missing"
    assert last_path.exists(), "last checkpoint missing"
    assert metrics_csv.exists(), "metrics.csv missing"
    assert summary_json.exists(), "summary.json missing"

    # 2. metrics.csv has the expected columns and one row per update.
    with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == smoke_cfg["ppo"]["total_updates"]
    assert EXPECTED_METRIC_COLUMNS.issubset(rows[0].keys()), (
        f"missing columns: {EXPECTED_METRIC_COLUMNS - set(rows[0].keys())}"
    )
    for i, row in enumerate(rows):
        assert int(row["update"]) == i
        for col in _FLOAT_COLUMNS:
            value = float(row[col])
            assert math.isfinite(value), f"{col} not finite in row {i}"

    # 3. summary.json round-trips with the run's update history.
    with summary_json.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["experiment_name"] == smoke_cfg["experiment_name"]
    assert len(on_disk["updates"]) == smoke_cfg["ppo"]["total_updates"]
    assert summary["experiment_name"] == smoke_cfg["experiment_name"]
    assert len(summary["updates"]) == smoke_cfg["ppo"]["total_updates"]

    # 4. The checkpoint carries a loadable policy plus its dimensions.
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    for key in (
        "policy_state_dict", "observation_dim", "action_dim",
        "hidden_dim", "n_hidden_layers",
    ):
        assert key in ckpt, f"checkpoint missing key {key}"
    assert ckpt["algorithm"] == "PPO"
    assert ckpt["hidden_dim"] == (
        smoke_cfg["ppo"]["hyperparameters"]["hidden_dim"]
    )
