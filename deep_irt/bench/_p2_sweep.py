"""
_p2_sweep.py -- Sequential, foreground, resumable sweep driver for Phase 2.

SCAFFOLD: fill-in pending. Core loop structure is in place; error handling
and manifest enrichment are TODO(fill-in) markers.

Role:
  - Scan a configs_p2/ directory for *.yaml files (sorted, deterministic order).
  - For each config, compute its sha256 identity hash.
  - Skip cells whose results.json already carries a matching hash (resume gate).
  - Run each remaining cell sequentially via _p2_run_cell.run_cell().
  - Write / update manifest.json in the common output root after every cell.
    Partial manifests survive a killed sweep so re-runs can resume cleanly.

Usage:
  python -m deep_irt.bench._p2_sweep configs_p2/ --device cuda
  python -m deep_irt.bench._p2_sweep configs_p2/e1_smoke.yaml --device cpu
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from deep_irt.bench._p2_config import P2Config, load_config, config_sha256
from deep_irt.bench._p2_run_cell import run_cell


# ---------------------------------------------------------------------------
# Resume-gate helper
# ---------------------------------------------------------------------------

def _cell_status(cfg: P2Config, cfg_hash: str) -> str:
    """Return "pending" | "done" | "stale" | "unreadable" for a cell."""
    results_path = Path(cfg.report.output_dir) / cfg.report.cell_name / "results.json"
    if not results_path.exists():
        return "pending"
    try:
        existing = json.loads(results_path.read_text())
        if existing.get("config_sha256") == cfg_hash:
            return "done"
        return "stale"  # config changed since last run
    except Exception:
        return "unreadable"


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(
    config_path: str | Path,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> Path:
    """Run all experiment cells in config_path sequentially.

    config_path may be:
      - a directory: all *.yaml files inside are treated as cells (sorted).
      - a single .yaml file: only that cell is run.

    Args:
        config_path: directory of *.yaml configs, or a single .yaml file.
        device: torch device string ("cpu" or "cuda").
        output_dir: override the output_dir in every config's report block.
            Useful for redirecting a sweep to a different partition.

    Returns:
        Path to manifest.json (written after every cell, not just at the end).

    TODO(fill-in): add --n_parallel flag for embarrassingly parallel cell runs
    (cells with different data seeds are independent). For now, always sequential
    on one GPU.
    TODO(fill-in): wrap run_cell() in try/except and write "failed" + traceback
    into the manifest entry rather than crashing the full sweep.
    """
    config_path = Path(config_path)
    if config_path.is_dir():
        config_files: List[Path] = sorted(config_path.glob("*.yaml"))
        if not config_files:
            raise FileNotFoundError(f"no .yaml configs found in {config_path}")
    else:
        config_files = [config_path]

    # Determine manifest output root from the first config's report block
    # (or the override). Load it without running to avoid side effects.
    first_cfg = load_config(config_files[0])
    manifest_root = Path(output_dir or first_cfg.report.output_dir)
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "manifest.json"

    manifest_entries = []

    for cfg_path in config_files:
        cfg = load_config(cfg_path)
        if output_dir is not None:
            cfg.report.output_dir = output_dir

        cfg_hash = config_sha256(cfg)
        status = _cell_status(cfg, cfg_hash)
        results_path = str(
            Path(cfg.report.output_dir) / cfg.report.cell_name / "results.json"
        )
        entry = {
            "cell_name": cfg.report.cell_name,
            "config_path": str(cfg_path),
            "config_sha256": cfg_hash,
            "status": status,
            "results_path": results_path,
        }

        if status == "done":
            # Hash matches: skip silently.
            manifest_entries.append(entry)
            _write_manifest(manifest_path, manifest_entries)
            continue

        # Run the cell.
        t0 = time.time()
        # TODO(fill-in): wrap in try/except; on failure set status="failed"
        # and entry["error"] = traceback.format_exc() rather than re-raising.
        run_cell(cfg, device=device, force=(status in {"stale", "unreadable"}))
        wall = round(time.time() - t0, 1)

        entry["status"] = "done"
        entry["wall_time_s"] = wall
        manifest_entries.append(entry)

        # Write manifest after every cell so a killed sweep has a valid manifest.
        _write_manifest(manifest_path, manifest_entries)

    return manifest_path


def _write_manifest(path: Path, entries: list) -> None:
    """Write manifest.json atomically (write to .tmp, then rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"cells": entries}, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 sweep driver")
    parser.add_argument(
        "config_path",
        help="directory of *.yaml configs or a single .yaml file",
    )
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    parser.add_argument("--output_dir", default=None, help="override output root")
    args = parser.parse_args()

    manifest = run_sweep(
        args.config_path, device=args.device, output_dir=args.output_dir
    )
    print(f"Manifest written to: {manifest}")
