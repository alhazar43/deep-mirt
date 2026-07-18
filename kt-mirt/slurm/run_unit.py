"""Runs ONE A4 campaign unit (see `enumerate_units.py`) by integer id,
writing its cell JSON into a shared results store via
`kt_mirt.growth.run`'s own idempotent cell functions. Called by
`chain_runner.sbatch` (cluster) and `local_worker.sh` (local 4060); both
write into the SAME store layout, so units are idempotent across
machines -- a unit already computed by one machine is skipped (unless
--force) rather than recomputed by the other.

This script deliberately does NOT call `run.run_campaign` (which also
CERTIFIES a twin from every seed's cell): certification needs every
seed of a twin present, which is exactly what distributed execution
does not guarantee mid-campaign. Run `certify.py` after the units of
interest are done to assemble the gate verdict + report (idempotent
skip means it recomputes nothing already on disk).

Usage:
    python slurm/run_unit.py --index 17 --output-dir outputs/a4/campaign \\
        --device cuda --skip-done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from enumerate_units import (
    PRODUCTION_HYPERPARAMS,
    PROFILE_HYPERPARAMS,
    PROFILES,
    enumerate_units,
    is_done,
)

from kt_mirt.growth import run as run_mod


def run_unit_by_id(unit_id: int, output_dir: Path, device: str, force: bool,
                   kind: str | None = None) -> dict:
    """``kind=None``: ``unit_id`` is a GLOBAL id over the full coverage-first
    enumeration (local_worker.sh's semantics). ``kind`` given: ``unit_id`` is
    the POSITION within that kind's own coverage-first sublist (the cluster
    tracks' semantics -- autopilot.sh's GPU track submits array indices
    0..N_neural-1 and its CPU track 0..N_slice-1, which are positions per
    kind, NOT global ids; mapping them here keeps the two tracks disjoint
    and each track's TOTAL_UNITS bound correct)."""
    units = enumerate_units()
    if kind is not None:
        sublist = [u for u in units if u.kind == kind]
        if not (0 <= unit_id < len(sublist)):
            raise SystemExit(f"no {kind} unit at position {unit_id} (total {len(sublist)})")
        unit = sublist[unit_id]
    else:
        matches = [u for u in units if u.unit_id == unit_id]
        if not matches:
            raise SystemExit(f"no unit with id {unit_id} (total {len(units)})")
        unit = matches[0]

    if not force and is_done(unit, output_dir):
        return {"unit_id": unit_id, "kind": unit.kind, "profile": unit.profile,
                "twin": unit.twin, "seed": unit.seed, "status": "already_done"}

    profile = PROFILES[unit.profile]
    # cfg.twins/cfg.model_seeds are INERT here: run_slice_cell/run_neural_cell
    # take (cfg, twin, seed) as direct arguments, never reading cfg.twins or
    # cfg.model_seeds (those only matter to the higher-level run_campaign
    # orchestrator, which this per-unit script does not call). cfg.
    # generator_seeds[0] IS read (a neural cell's twin is always generated at
    # that seed, design 3.2's compute concession); its class default, [0,1],
    # already starts at 0 = GEN_SEEDS[0], so it is left unset below.
    # Per-profile overrides layered on the shared production hyperparameters
    # (currently only ednet_matched's tracker mini-batch size, the OOM-ruling
    # step-2 memory concession -- see enumerate_units.PROFILE_HYPERPARAMS).
    cfg = run_mod.RunConfig(
        output_dir=output_dir,
        profile=profile,
        device=device,
        force=force,
        **{**PRODUCTION_HYPERPARAMS, **PROFILE_HYPERPARAMS.get(unit.profile, {})},
    )

    t0 = time.time()
    if unit.kind == "slice":
        result, _raw = run_mod.run_slice_cell(cfg, unit.twin, unit.seed)
    else:
        result = run_mod.run_neural_cell(cfg, unit.twin, unit.seed)
    return {
        "unit_id": unit_id, "kind": unit.kind, "profile": unit.profile, "twin": unit.twin,
        "seed": unit.seed, "device": device, "status": "computed", "wall_clock_s": time.time() - t0,
    }


def _failed_path(output_dir: Path, unit_id: int) -> Path:
    d = Path(output_dir) / "_failed"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"unit_{unit_id}.json"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, required=True,
                   help="global unit id (no --kind) or position within --kind's coverage-first sublist")
    p.add_argument("--kind", choices=["slice", "neural"], default=None,
                   help="interpret --index as a position within this kind's sublist (cluster-track semantics)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--skip-done", action="store_true", help="no-op flag for parity with the kt-irt CLI convention (this is the default here); see --force")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir)
    try:
        status = run_unit_by_id(args.index, output_dir, args.device, args.force, kind=args.kind)
        print(json.dumps(status))
        return 0
    except Exception as exc:  # noqa: BLE001 -- a chain must survive one bad unit and record it
        failed = {"unit_id": args.index, "error": str(exc), "traceback": traceback.format_exc()}
        _failed_path(output_dir, args.index).write_text(json.dumps(failed, indent=2))
        print(json.dumps({"unit_id": args.index, "status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
