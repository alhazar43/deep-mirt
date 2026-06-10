"""End-to-end smoke test for ``rl/scripts/eval_policy.py``.

Trains a tiny PPO inline via ``train_ppo.train``, then drives
``eval_policy.main()`` through its CLI surface against the produced
checkpoint and asserts the markdown report carries every field the
E4.5/E5 analysis reads, mean episode return for both policies, the
per-component reward means, the exposure diagnostics and the
random-baseline comparison.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


def test_eval_policy_end_to_end(
    train_ppo_module: Any,
    eval_policy_module: Any,
    smoke_cfg: Dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Train a checkpoint inline with the same tiny config.
    train_ppo_module.train(smoke_cfg)
    ckpt = (
        Path(smoke_cfg["output_dir"])
        / smoke_cfg["experiment_name"]
        / "best.pt"
    )
    assert ckpt.exists()

    # 2. Run the eval CLI against it.
    cfg_path = tmp_path / "smoke_eval_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(smoke_cfg), encoding="utf-8")
    report_path = tmp_path / "eval_report.md"
    monkeypatch.setattr(
        sys, "argv",
        [
            "eval_policy.py",
            "--config", str(cfg_path),
            "--checkpoint", str(ckpt),
            "--n-episodes", "2",
            "--seed", "123",
            "--output", str(report_path),
        ],
    )
    exit_code = eval_policy_module.main()
    assert exit_code == 0
    assert report_path.exists(), "eval report not written"

    report = report_path.read_text(encoding="utf-8")

    # 3. Mean-return section with one row per policy and parseable
    # numbers (mean, std, n_episodes).
    assert "## Mean episode return" in report
    row_re = re.compile(
        r"^\|\s*(trained PPO|uniform random)\s*\|\s*([+-]?\d+\.\d+)\s*"
        r"\|\s*(\d+\.\d+)\s*\|\s*(\d+)\s*\|",
        re.MULTILINE,
    )
    return_rows = {m.group(1): m for m in row_re.finditer(report)}
    assert set(return_rows) == {"trained PPO", "uniform random"}, (
        f"return table incomplete: {sorted(return_rows)}"
    )
    for name, m in return_rows.items():
        n_eps = int(m.group(4))
        assert n_eps > 0, f"{name} reports zero evaluated episodes"

    # 4. Random-baseline comparison line.
    assert re.search(
        r"Delta \(trained minus random\), [+-]\d+\.\d+\.", report,
    ), "baseline delta line missing"

    # 5. Per-component means for both policies.
    assert "## Per-component reward, mean over evaluation" in report
    assert "| policy | r_info | r_cost | r_expo | r_voi |" in report
    comp_re = re.compile(
        r"^\|\s*(trained PPO|uniform random)\s*\|"
        r"(\s*[+-]\d+\.\d+\s*\|){4}",
        re.MULTILINE,
    )
    comp_rows = {m.group(1) for m in comp_re.finditer(report)}
    assert comp_rows == {"trained PPO", "uniform random"}

    # 6. Exposure diagnostics table with the quantile columns.
    assert "## Exposure diagnostics" in report
    assert "| policy | max | p99 | p95 | p50 | frac > r_max |" in report
    expo_re = re.compile(
        r"^\|\s*(trained PPO|uniform random)\s*\|"
        r"(\s*\d+\.\d+\s*\|){5}",
        re.MULTILINE,
    )
    expo_rows = {m.group(1) for m in expo_re.finditer(report)}
    assert expo_rows == {"trained PPO", "uniform random"}
