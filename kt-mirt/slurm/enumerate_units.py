"""Unit enumeration for the A4 synthetic certification campaign
(`_planning/design/a4_design.md` v1.1, section 8's R2 "synthetic
certification matrix"), shared by the cluster chain runner
(`slurm/chain_runner.sbatch` + `slurm/autopilot.sh`) and the local worker
(`slurm/local_worker.sh`).

No new cell granularity is introduced here: this module assigns stable
integer ids over exactly the two unit-of-work cell types
`kt_mirt.growth.run` already implements and idempotently checkpoints --
a SLICE cell (profile, twin, generator seed): bank calibration + PAS-G/
MIX-L + permutation battery; and a NEURAL cell (profile, twin, model
seed, always at generator seed 0 per design section 3.2's compute
concession): PAS-N1/PAS-N2 trackers + ACT-P0/P1 and every neural-side
battery arm. `run.py`'s own `run_slice_cell`/`run_neural_cell` are called
directly (`run_unit.py`, this directory) -- this module only orders and
numbers the grid.

Campaign grid: 2 density profiles (kdd_matched, ednet_matched, design
section 3.2's "two density profiles") x 4 twins (syn_ng, syn_kg, syn_ns,
syn_sat) x {5 generator seeds for slice cells, 3 model seeds for neural
cells} = 40 slice-cell units + 24 neural-cell units = 64 units total.

Ordering: COVERAGE-FIRST. Phase 1 assigns ids 0..15 to one unit (seed
index 0) of every (kind, profile, twin) combination (2 kinds x 2 profiles
x 4 twins = 16), so an interrupted campaign always has at least one
generator seed and one model seed of every cell computed before any cell
gets a second seed. Phase 2 (seed-deepening) fills in the remaining
seeds, again cycling breadth-first across (kind, profile, twin) rather
than exhausting one cell's seeds before starting the next, so partial
campaign coverage stays broad rather than deep in one corner of the grid.

Device routing (judgment call, stated rather than silently assumed):
NEURAL cells default to "cuda" (trackers + ACT need it for tolerable
per-fit wall time). SLICE cells default to "cpu": the design's own
compute table (section 7) states the slice-based pipeline "runs on CPU
and goes 10-50x faster on GPU", i.e. CPU is a documented-adequate
fallback, and `run.py`'s slice cell bundles bank calibration with the
CPU-native gate/rate/permutation-battery machinery into ONE atomic call --
splitting bank calibration onto its own GPU-only unit would require
re-slicing the verified (397/397) harness's cell boundaries, which this
infrastructure layer deliberately does not do. `Unit.device` is only a
DEFAULT; both `run_unit.py` and the sbatch/worker scripts accept a
`--device` override per chain, so an operator can route slice-cell chains
to GPU if a timing/smoke run shows synthetic-scale bank calibration is
CPU-bound in practice.

Production hyperparameters (PRODUCTION_HYPERPARAMS below): `RunConfig`'s
own class defaults are tuned for a fast wiring smoke test (the tiny dry
run in `outputs/a4/tiny_kdd/` used them unmodified), NOT the design's
pre-registered replicate/reshuffle/drill counts. This module overrides
exactly the numerically pre-registered ones for a real campaign run:
n_perm_bed=999, n_perm_kc=199 (section 4 arm 1), n_reshuffles=5 (arm 5),
drill_repeats=100 (arm 4, "100 times"). Epoch counts (bank/tracker/act)
are convergence-gated caps the design leaves to build-agent discretion
(section 2.1), so `RunConfig`'s own defaults are kept for those.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from kt_mirt.growth import synth as synth_mod

# KT_MIRT_UNIT_SCALE selects between the real campaign grid ("production",
# default) and a small/fast "smoke" grid used ONLY to exercise the SLURM
# plumbing (chain_runner/autopilot/local_worker/pull_results, unit
# enumeration, idempotent store writes) before the real campaign is sized
# and launched -- mirroring `run.py`'s own TINY_PROFILE precedent
# (`a4_tiny_cert.sbatch`'s dry run). Smoke-scale numbers are NEVER
# certification results (same caveat `run.py`'s tiny report already
# states); this only exists so the smoke burst (campaign-prep checklist
# item 4) can run a handful of REAL units end-to-end within a lean
# sbatch time limit, rather than gambling that one full-scale unit (999/
# 199 permutation replicates, 100-repeat contamination drills, at
# C=515/N=3000 or C=189/N=6000) fits inside chain_runner.sbatch's 40-
# minute header before that unit's own timing has ever been measured.
_SCALE = os.environ.get("KT_MIRT_UNIT_SCALE", "production")

if _SCALE == "production":
    PROFILES: dict[str, synth_mod.DensityProfile] = {
        "kdd_matched": synth_mod.KDD_MATCHED,
        "ednet_matched": synth_mod.EDNET_MATCHED,
    }
    # The design's own pre-registered numeric bars (section 4), overriding
    # RunConfig's dry-run-friendly class defaults. bank_epochs/tracker_epochs/
    # act_epochs are NOT overridden (convergence-gated caps left to build-agent
    # discretion, section 2.1) -- RunConfig's own defaults apply.
    PRODUCTION_HYPERPARAMS = {
        "n_perm_bed": 999,     # arm 1: "B = 999 for the bed-level pooled statistic"
        "n_perm_kc": 199,      # arm 1: "B = 199 replicates for per-KC empirical p"
        "n_reshuffles": 5,     # arm 5: "permute cross-KC interleaving (5 reshuffles)"
        "drill_repeats": 100,  # arm 4: "one KC practiced 100 times"
        "run_act_p1": True,
        # ACT-P0 UNMASKED (orchestrator ruling superseding the same-day
        # exclusion): the ACT trainer's stationarity-gated convergence
        # repair (active.py note 8, commit 2dc0818) removed the
        # under-training fabrication, and BOTH variants go to production
        # certification with the pre-registered gates as the arbiter.
        # run_act_p0 stays available in RunConfig as the masking switch
        # should a future ruling re-exclude it.
        "run_act_p0": True,
    }
elif _SCALE == "smoke":
    PROFILES = {
        "kdd_matched": dataclasses.replace(synth_mod.KDD_MATCHED, name="kdd_matched", n_kcs=6, n_learners=80, kcs_per_learner=4.0),
        "ednet_matched": dataclasses.replace(synth_mod.EDNET_MATCHED, name="ednet_matched", n_kcs=6, n_learners=80, kcs_per_learner=3.0),
    }
    PRODUCTION_HYPERPARAMS = {
        "n_perm_bed": 9, "n_perm_kc": 5, "n_reshuffles": 1, "drill_repeats": 5, "run_act_p1": True,
    }
else:
    raise ValueError(f"KT_MIRT_UNIT_SCALE must be 'production' or 'smoke', got {_SCALE!r}")

PROFILE_NAMES = tuple(PROFILES)
TWINS = tuple(synth_mod.TWIN_NAMES)
GEN_SEEDS = list(range(5))   # design 3.2: "5 generator seeds per config for all slice-based machinery"
MODEL_SEEDS = list(range(3))  # design 3.2: "neural arms ... run on generator seed 0 with 3 model seeds"


@dataclass(frozen=True)
class Unit:
    unit_id: int
    kind: str      # "slice" | "neural"
    profile: str   # key into PROFILES
    twin: str
    seed: int      # generator seed (slice) or model seed (neural)
    device: str    # default device for this unit's kind (see module docstring)

    def to_dict(self) -> dict:
        return asdict(self)


def _default_device(kind: str) -> str:
    return "cpu" if kind == "slice" else "cuda"


def enumerate_units() -> list[Unit]:
    """Coverage-first enumeration (module docstring); deterministic, pure
    function of the module-level constants above -- no filesystem or
    network access, so both the cluster and the local worker always agree
    on unit ids without coordination."""
    cells = [(kind, profile, twin) for kind in ("slice", "neural") for profile in PROFILE_NAMES for twin in TWINS]
    seed_lists = {"slice": GEN_SEEDS, "neural": MODEL_SEEDS}
    max_depth = max(len(seed_lists["slice"]), len(seed_lists["neural"]))

    units: list[Unit] = []
    for depth in range(max_depth):
        for kind, profile, twin in cells:
            seeds = seed_lists[kind]
            if depth >= len(seeds):
                continue
            units.append(Unit(
                unit_id=len(units), kind=kind, profile=profile, twin=twin,
                seed=seeds[depth], device=_default_device(kind),
            ))
    return units


def _cell_path(output_dir: Path, profile_name: str, twin: str, kind: str, seed: int) -> Path:
    """Mirrors `kt_mirt.growth.run`'s own `_slice_cell_path`/
    `_neural_cell_path` naming exactly (kept in sync by convention, not by
    import, since those helpers require a full `RunConfig` -- this check
    only needs the path). ``profile_name`` MUST be the `DensityProfile`
    object's own ``.name`` field (what `run.py` actually writes under),
    not necessarily a `PROFILES` dict key -- see `is_done`."""
    fname = f"slice_seed{seed}.json" if kind == "slice" else f"neural_modelseed{seed}.json"
    return Path(output_dir) / profile_name / twin / fname


def is_done(unit: Unit, output_dir: Path) -> bool:
    """Resolves the on-disk path via ``PROFILES[unit.profile].name``, NOT
    ``unit.profile`` directly: `run.py` names a cell's directory after the
    `DensityProfile` object's own ``.name`` field (`RunConfig.profile.
    name`), which happens to equal its `PROFILES` dict key in production
    but is not guaranteed to (e.g. a caller substituting a differently
    named profile object under the same key, as a test harness might for
    isolation) -- resolving through the object avoids that fragility."""
    profile_name = PROFILES[unit.profile].name
    return _cell_path(output_dir, profile_name, unit.twin, unit.kind, unit.seed).exists()


def _apply_filters(
    units: list[Unit],
    kind: Optional[str] = None,
    profile: Optional[str] = None,
    twin: Optional[str] = None,
    device: Optional[str] = None,
    mod: Optional[int] = None,
    remainder: Optional[int] = None,
) -> list[Unit]:
    out = units
    if kind is not None:
        out = [u for u in out if u.kind == kind]
    if profile is not None:
        out = [u for u in out if u.profile == profile]
    if twin is not None:
        out = [u for u in out if u.twin == twin]
    if device is not None:
        out = [u for u in out if u.device == device]
    if mod is not None:
        r = remainder if remainder is not None else 0
        out = [u for u in out if u.unit_id % mod == r]
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["slice", "neural"], default=None)
    p.add_argument("--profile", choices=PROFILE_NAMES, default=None)
    p.add_argument("--twin", choices=TWINS, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="filter by unit's DEFAULT device")
    p.add_argument("--mod", type=int, default=None, help="keep only unit_id %% MOD == REMAINDER (id-space partition)")
    p.add_argument("--remainder", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="store root, for --remaining / --done-count")
    p.add_argument("--count", action="store_true", help="print the number of matching units")
    p.add_argument("--list", action="store_true", help="print matching units as JSON")
    p.add_argument("--get", type=int, default=None, metavar="UNIT_ID", help="print one unit (by id) as JSON")
    p.add_argument("--remaining", action="store_true", help="print REMAINING=<n> not-yet-done matching units (needs --output-dir)")
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    all_units = enumerate_units()

    if args.get is not None:
        matches = [u for u in all_units if u.unit_id == args.get]
        if not matches:
            raise SystemExit(f"no unit with id {args.get} (total {len(all_units)})")
        print(json.dumps(matches[0].to_dict(), indent=2))
        return

    units = _apply_filters(
        all_units, kind=args.kind, profile=args.profile, twin=args.twin,
        device=args.device, mod=args.mod, remainder=args.remainder,
    )

    if args.remaining:
        if args.output_dir is None:
            raise SystemExit("--remaining requires --output-dir")
        remaining = sum(1 for u in units if not is_done(u, Path(args.output_dir)))
        print(f"REMAINING={remaining}")
        return

    if args.list:
        print(json.dumps([u.to_dict() for u in units], indent=2))
        return

    # default / --count
    print(len(units))


if __name__ == "__main__":
    main()
