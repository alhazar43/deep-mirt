# deep-irt portable-repo refactor — systems half

Planning only. This document owns layout, schemas, artifact management,
multi-machine sync, and migration mechanics. A separate track owns
researcher-workflow ergonomics. Nothing here changes model behavior, metric
computation, or plot rendering. Every proposal is a relocation or a
standardization, verified against a byte-identical baseline.

Scope note. The goal is a standalone `deep-irt` repo that trains on any of
three machines (Windows RTX 4060 PC, a working laptop, a SLURM compute head)
and shares results so figures and tables regenerate anywhere without
retraining.

---

## 0. What the survey found (grounding for every decision below)

**Tree and sizes (measured 2026-07-13).**

| tree | size | files | role |
|---|---|---|---|
| `outputs/` total | 2.6 G | 16,291 json / 13,118 npz / 127 pdf+png | the shared store to be |
| `outputs/p2_exposure` | 641 M | — | exposure sweep (verify figure-source) |
| `outputs/p2_v3_arm1r` | 443 M | 742 | real EdNet-NRM routed campaign (SHIPPED) |
| `outputs/p2_toggle` | 365 M | 6,401 | synthetic toggle grid (feeds fig_dd) |
| `outputs/p2_nrm_repar` | 201 M | 3,714 | synthetic arm-ladder grid (arm1r under `/arm1r`) |
| `outputs/p2_realstudy` | 174 M | 902 | real-data frozen driver + `mirt/` MML refs |
| `outputs/p2_v3_hardnrm` | 120 M | 223 | arm1h/arm1r real EdNet-NRM rerun + traj |
| `outputs/p2_v3_export` | 42 M | 794 | traj npz + thresholds + `tab_real_metrics` |
| `outputs/p2_v3_analysis` | 8.1 M | 200 | analysis JSONs + `figs/` (the rendered set) |
| ~35 other `p2_*` trees | 36 K – 160 M | — | earlier probes, mostly superseded |

**Per-artifact sizes** (drive the sync split). Realstudy scalar fold JSON
about 420 bytes; EdNet-NRM routed fold JSON about 1.1 MB (carries per-item
`alpha (Q,4)` + `beta (Q,4)` + `seen`); traj npz about 1.9 MB; analysis JSON
about 1.4 KB; figure PDF about 500 KB. The heavy mass is npz and the
NRM fold JSONs; the "regenerate-anywhere-light" mass (analysis JSONs, tables,
figures) is under about 30 MB total.

**The single choke point.** `deep_irt/bench/_p2_v3_analysis_lib.py` resolves
every artifact path through module-level globals derived from one base:

```
REPO      = <two dirs up from the file>
REALSTUDY = REPO/outputs/p2_realstudy
EXPORT    = REPO/outputs/p2_v3_export
TRAJ      = EXPORT/traj
TIMSS_THR = EXPORT/timss_gpcm_sk
MIRT      = REALSTUDY/mirt
TOGGLE    = REPO/outputs/p2_toggle
OUT       = REPO/outputs/p2_v3_analysis
FIGS      = OUT/figs
SEEDS, FOLDS = (0..4), (0..4)
```

Every analysis and figure script imports these. Re-pointing this one base is
what made all recent tree moves cheap. The producers have the mirror of this:
`_p2_realstudy.OUT_ROOT`, `_p2_nrm_repar_sweep.OUT_ROOT_BASE`,
`_p2_v3_export` traj root, and the dataset cache `CACHE = rs.OUT_ROOT/direct/_datacache`.
The plan makes all of these derive from one `RESULTS_ROOT`.

**Codex ownership boundary.** `deep_irt/core/*`, `deep_irt/bench/run_*.py`,
`deep_irt/bench/datagen.py`, `deep_irt/bench/engines.py`, and
`deep_irt/bench/_ednet_ot*.py` are owned by another agent and must not be
edited. The `_p2_*.py` scratch is the user's and may be edited or relocated.
All `_p2_*` files import Codex modules by absolute package path
(`import deep_irt.bench._p2_realstudy`, `from deep_irt.bench._p2_nrm_repar import ...`),
so absolute-package imports are position-independent under a package move.

**Tracked vs untracked.** 127 of ~233 bench files are tracked. Of 115
`_p2_*.py`, 17 are tracked (the stable p2 core: `_p2_config`, `_p2_model`,
`_p2_engine`, `_p2_datagen_realistic(+test)`, `_p2_aggregate`, `_p2_cat`,
`_p2_gpcm_alpha_key`, `_p2_nrm_channels`, `_p2_datagen_budget`,
`_p2_ordinal_ce(+test)`) and 98 are untracked scratch. The untracked 98 hold
the entire arm1r campaign. Promotion (section 7-lens below and the migration
phases) curates a load-bearing subset into tracked pipeline code.

**Dataset staging.** `data/` is 3.0 G (`data/kdd/algebra_2008_2009_train.txt`
a 3 GB TSV; `data/timss/*.csv` triplets + classical GPCM coefficients + R
build scripts; `data/slam_raw/`). `EdNet-KT1/KT1/` is roughly 780k per-user
CSV files, the expensive glob (cold load runs into minutes per process,
about 20 min for a cold campaign). `EdNet-Contents/` holds the option-tracing
answer key. All four (`EdNet-KT1`, `EdNet-Contents`, `data`, `outputs`) are
gitignored. `_p2_datacache.py` already memoizes `load_dataset` output to
`outputs/p2_realstudy/direct/_datacache/{dataset}_{decoder}_{sig}.npz`,
cross-process, byte-identical to a fresh load, keyed by a SIZE signature. This
cache is the seed of the versioned dataset artifact below.

**The R/MML component (author clarification 2026-07-13).** Classical MML
calibration runs through four R fitters (`_p2_mml_control.R` GPCM,
`_p2_mml_control_2pl.R` 2PL, `_p2_mml_nrm.R` NRM, `_p2_mml_real.R` real-data)
invoked by five Python callers (`_p2_mml_control.py`, `_p2_mml_control_2pl.py`,
`_p2_mml_nrm.py`, `_p2_mml_real.py`, `_p2_realstudy_mml.py`; plus
`_p2_coldstart_starved.py` importing the plumbing), each hardcoding
`RSCRIPT_EXE = C:\Program Files\R\R-4.5.0\bin\Rscript.exe`. All are user-owned
scratch, not Codex files, so they are editable. The calibration OUTPUT is
`outputs/p2_realstudy/mirt/<dataset>_<decoder>/{in.csv, items.csv,
reference.json, rlog*}`; the consumer side
(`_p2_v3_analysis_lib.mml_reference()` and every downstream figure/table) reads
`reference.json` with no R anywhere. R also appears in dataset staging
(`data/timss/_build_timss_gpcm.R` built the TIMSS matrices, already cached as
CSV). Author's constraint: R runs ONLY on the PC and the work laptop, never on
the SLURM remote. Consequence for the plan: MML calibration is an EXTRACTED
optional stage gated by a machine-capability flag, and its outputs are
first-class synced artifacts so R-less machines consume references from the
store (sections 1, 2, 3 below).

**Naming in use.**
- realstudy / arm1r real cell: `{enc}_{dec}_{dataset}_{klass}`, e.g.
  `lstm_nrm_ednet_separate`, `dkvmn_nrm_ednet_separate`.
- toggle synthetic cell: `tog_{enc}_{dec}_{klass}_static_N{N}_Q{Q}`, e.g.
  `tog_lstm_nrm_decoupled_static_N2000_Q200`.
- arm-ladder synthetic cell: `<arm>/<prefix>_{enc}_{dec}_{klass}_static_N{N}_Q{Q}`
  with `arm1r -> rep1r`, e.g. `arm1r/rep1r_lstm_nrm_decoupled_static_N2000_Q200`.
- unit file: `d{seed}_f{fold}.json` (+ `traj_d{s}_f{f}.npz`, `arrays_d{s}_f{f}.npz`,
  `thr_d{s}_f{f}.npz`).
- design axes in the names: `klass` in {separate=SK, shared=SH}; NRM channel in
  {decoupled, shared}; regime in {static, dynamic}; head in
  {soft/arm1, arm1h, arm1r, g0}.

---

## 1. Target tree for the portable `deep-irt` repo

Requirement 2 mandates separation of code, scripts, configs, dataset,
results, figures, and checkpoints. The layout below is a `src/` package plus
sibling top-level directories, one per artifact class. Names are chosen for
Google/Microsoft-grade legibility; the separation is the invariant.

```
deep-irt/
  pyproject.toml            # packaging: `pip install -e .` kills the PYTHONPATH hack
  README.md  LICENSE  CODEOWNERS  CONTRIBUTING.md
  .gitignore  .dvcignore  dvc.yaml (optional)  .dvc/config

  src/deep_irt/             # ALL importable library code (one package root)
    core/                   # VENDORED unedited (Codex-owned)
    bench/                  # VENDORED unedited: run_*, datagen, engines, _ednet_ot*
                            #   + the stable tracked p2 core (config/model/engine/datagen)
    pipeline/               # PROMOTED from _p2_* scratch (tracked, curated)
      producers/            # real driver, hardnrm rerun, nrm_repar head + sweep, export, datacache
      analysis/             # analysis_lib (the choke point), shsk, metrics, stability, common
      figures/              # deltadelta, reversal_bridge, ednet_case, agreement, 2in1, ...
      calibrate/            # EXTRACTED optional R/MML stage (see below): the four
                            #   .R fitters + their Python callers; torch-free;
                            #   Rscript path from the machine profile, never hardcoded
      manifest.py           # NEW: run-id, manifest.json read/write, unit enumeration
      paths.py              # NEW: RESULTS_ROOT + DATA_ROOT + machine-profile resolution

  scripts/                  # thin OS-agnostic CLI entrypoints (no logic; call src/)
    train_unit.py           # run one (cell, seed, fold) unit
    run_sweep.py            # enumerate + drive a sweep (local, sequential)
    calibrate_mml.py        # OPTIONAL stage: classical MML references (R-capable machines only)
    score.py                # post-hoc metrics/tables from artifacts
    make_figures.py         # regenerate the figure/table set from RESULTS_ROOT
    stage_data.py           # build/verify the versioned dataset cache
    launch/                 # thin per-platform launchers (see below)
      windows.ps1
      linux.sh
    slurm/                  # STUB, gated (section 3 + phase 7)
      sbatch_array.template.sh
      README.md

  configs/                  # all YAML; today's bench/configs_p2 lands here
    real/  toggle/  bench/  smoke/  _bases/
    machines/               # machine-capability profiles (see section 3)
      pc.yaml  laptop.yaml  slurm.yaml   # tracked profiles; active one picked by
                                         #   DEEP_IRT_MACHINE env or hostname match

  data/                     # gitignored payload; git-tracked registry + fetch script
    registry.json           # dataset name -> source, SIZE signature, build cmd, hash
    cache/                  # DVC-tracked built matrices (the datacache npz, versioned)
    raw/                    # gitignored; staged by scripts/stage_data.py (EdNet, KDD, TIMSS)

  results/                  # the SHARED store (section 2). git-light + DVC-heavy split
    runs/<run_id>/manifest.json
    p2_realstudy/  p2_v3_arm1r/  p2_v3_export/  p2_nrm_repar/  p2_toggle/
    p2_v3_hardnrm/  p2_v3_analysis/            # subdir names kept VERBATIM (see note)

  figures/                  # rendered pdf/png (was p2_v3_analysis/figs); git or DVC per size
  checkpoints/              # model weights, if/when persisted; DVC-tracked, gitignored
  docs/                     # this plan, schemas.md, HANDOFF, provenance
  tests/                    # relocated deep_irt/tests + schema/round-trip tests
```

**Why each separation.**
- `src/deep_irt/` is the only place Python imports resolve from. One package,
  installed editable, so no machine needs `PYTHONPATH`. This is the packaging
  fix demanded for SLURM (section 3).
- `scripts/` holds zero logic. Each entrypoint is a `python -m` friendly CLI
  that parses args and calls into `src/`. This is what a SLURM array task and a
  Windows double-click both invoke, unchanged.
- `configs/` is code-adjacent but not code. Today's `bench/configs_p2/` (real,
  toggle, bench, smoke, `_bases`) moves here with its names intact.
- `data/` splits a git-tracked `registry.json` + fetch script from a gitignored
  payload. The registry is the reproducibility contract; the payload is
  machine-local or DVC-pulled.
- `results/` keeps the internal `p2_*` subdir names verbatim (not renamed).
  This guarantees the choke-point globals and every producer resume path change
  only their ROOT, never a path suffix, so relocation is provably content- and
  layout-neutral. A cosmetic rename to `realstudy/`, `arm1r/`, etc. is a
  separate, later, non-load-bearing cleanup once the byte-identical gate has
  passed at least once.
- `figures/` and `checkpoints/` are their own classes so a laptop can pull
  figures without pulling npz or weights.
- `pipeline/calibrate/` isolates the R/MML stage as an OPTIONAL component
  (author's constraint: R exists only on the PC and the work laptop). The four
  `.R` fitters move here WITH their Python callers — R scripts are code and
  belong in `src/`, not `data/` or `scripts/`. The stage is torch-free and
  GPU-free (it always was); its only environment demand is Rscript + the mirt
  package, and that demand is declared, not assumed. The hardcoded
  `RSCRIPT_EXE` constants (five user-owned files) are replaced by one lookup in
  the machine profile; on the author's PC the resolved command line is
  character-identical to today's, so calibration outputs are unchanged. No
  Codex file is involved. `data/registry.json` likewise marks the TIMSS matrix
  build (`_build_timss_gpcm.R`) as R-requiring; since the built CSVs are cached
  artifacts, no machine ever needs to rerun it.

**One decision to coordinate: `src/` layout vs flat.** The `src/deep_irt/`
layout is the Google/MS-clean target but it changes Codex's file paths from
`deep_irt/core/*` to `src/deep_irt/core/*`. A rename is not an edit (content
byte-identical, behavior identical), so it honors the no-touch letter and
spirit, but it must be a single coordinated `git mv` at a quiescent point with
Codex paused (Codex is actively committing; see risk 1). If that coordination
is unacceptable, the fallback is a flat layout keeping `deep_irt/` at the repo
root (package dir = repo root), which leaves Codex's paths identical at the
cost of a less-clean top level. Recommendation: `src/` layout, executed as one
atomic vendor move (phase 2).

---

## 2. The run-manifest standard

**Run id.** A run is one sweep invocation. Its id is a stable slug over the
axes already present in the cell names plus the campaign axes:

```
run_id = {bed}.{decoder}.{encoder}.{design}.{head}.{scale}.{stamp}
  bed      real:<dataset> | synth            e.g. real:ednet, synth
  decoder  2pl | gpcm | nrm | bt
  encoder  lstm | transformer | dkvmn
  design   sk | sh            (klass: separate | shared)
  head     soft | arm1h | arm1r | g0     (nrm only; else 'na')
  scale    real | N{n}_Q{q}
  stamp    UTC yyyymmddThhmmss  (uniqueness; NOT used for resume)
```

Example: `real:ednet.nrm.lstm.sk.arm1r.real.20260710T2103`. A unit inside a
run is the existing `(cell, seed, fold)` triple; its artifact filename stays
`d{seed}_f{fold}.json`. The run id names the manifest and the `runs/<run_id>/`
directory; it does not rename any artifact, so resume and the choke point are
untouched.

**manifest.json (one per run).** Written next to nothing it depends on; it is
pure metadata that makes a tree reproducible and auditable.

```json
{
  "run_id": "real:ednet.nrm.lstm.sk.arm1r.real.20260710T2103",
  "created_utc": "2026-07-10T21:03:11Z",
  "code": {
    "deep_irt_sha": "<git sha of the deep-irt submodule>",
    "parent_sha":   "<git sha of deep-mirt, if run inside it>",
    "dirty": false
  },
  "env": {
    "python": "3.11.x", "torch": "2.7.1+cu126", "cuda": "12.6",
    "platform": "Windows-11 | Linux-x86_64", "gpu": "RTX 4060 Laptop 8GB",
    "hostname": "<machine label>",
    "flags": {
      "KMP_DUPLICATE_LIB_OK": "TRUE",
      "CUDA_LAUNCH_BLOCKING": "1|unset"        // Windows-driver footgun, recorded
    }
  },
  "inputs": {
    "dataset_cache": {"name": "ednet_nrm", "sha256": "<hash>", "size_sig": "..."},
    "config_sha256": "<hash of the resolved YAML>"
  },
  "spec": {
    "bed":"real:ednet","decoder":"nrm","encoder":"lstm","design":"sk",
    "head":"arm1r","scale":"real","seeds":[0,1,2,3,4],"folds":[0,1,2,3,4]
  },
  "artifacts": [
    {"path":"results/p2_v3_arm1r/lstm_nrm_ednet_separate/d0_f0.json",
     "bytes":1110336,"sha256":"..."},
    {"path":"results/p2_v3_export/traj/lstm_nrm_ednet_separate/traj_d0_f0.npz",
     "bytes":1897892,"sha256":"..."}
  ],
  "timing": {"units_done": 25, "wall_s": 1310.4, "per_unit_s_median": 52.0}
}
```

`inputs.dataset_cache.sha256` and `config_sha256` are the reproducibility key;
`code.deep_irt_sha` pins the exact library; `env.flags.CUDA_LAUNCH_BLOCKING`
records whether the Windows async-stall workaround was active (it is a driver
issue and should be unset on Linux, so recording it lets a reader tell a
Windows run from a Linux run at a glance). The manifest is generated by
`src/deep_irt/pipeline/manifest.py`, which the entrypoints call on completion;
it reads what the run already wrote, so it adds provenance without touching the
fit path.

**Mapping today's trees into the standard.** The internal layout is preserved;
only the root and the manifest are added. Old-path to new-path is a pure move:

| current | new (`results/` keeps the subdir name) | schema |
|---|---|---|
| `outputs/p2_realstudy/<cell>/d{s}_f{f}.json` | `results/p2_realstudy/<cell>/d{s}_f{f}.json` | fold-row JSON |
| `outputs/p2_realstudy/mirt/<name>/{reference.json,items.csv,in.csv,rlog*}` | `results/p2_realstudy/mirt/...` | classical MML ref (first-class, see below) |
| `outputs/p2_v3_arm1r/<cell>/d{s}_f{f}.json` | `results/p2_v3_arm1r/...` | fold-row JSON (NRM routed) |
| `outputs/p2_v3_hardnrm/<cell>/{d*.json, traj/*.npz}` | `results/p2_v3_hardnrm/...` | fold-row + traj |
| `outputs/p2_nrm_repar/<arm>/<cell>/fold_*.json + arrays_*.npz` | `results/p2_nrm_repar/...` | sweep fold + arrays |
| `outputs/p2_toggle/<cell>/d*_f*.json` | `results/p2_toggle/...` | toggle fold JSON |
| `outputs/p2_v3_export/traj/<cell>/traj_*.npz+.json` | `results/p2_v3_export/traj/...` | traj export |
| `outputs/p2_v3_export/timss_gpcm_{sk,sh}/thr_*.npz` | `results/p2_v3_export/...` | GPCM thresholds |
| `outputs/p2_v3_export/tab_*.{md,json}` | `results/p2_v3_export/...` | tables |
| `outputs/p2_v3_analysis/*.json` | `results/p2_v3_analysis/...` | analysis JSON |
| `outputs/p2_v3_analysis/figs/*` | `figures/*` | rendered figures |
| `outputs/p2_realstudy/direct/_datacache/*.npz` | `data/cache/*.npz` | versioned dataset cache |

**The three schemas, formalized without change** (documented in
`docs/schemas.md`; no code rewrites the bytes).

1. Realstudy / arm1r fold-row JSON — `_p2_realstudy_hardnrm.run_fold_hard`,
   `_p2_realstudy.run_fold`. Keys: `cell, encoder, decoder, dataset, klass,
   data_seed, fold, n_learners, seq_len, n_items, n_cats, n_train, n_val,
   seen[bool], n_seen, alpha[list], beta[list|null], acc{dict}, delta_slack,
   delta_n_items, timing_s, head, keyed, status`. `alpha`/`beta` are scalar
   per item for 2PL/GPCM, `[4]` per item for NRM. Size 420 B (scalar) to
   1.1 MB (NRM Q=200).
2. Sweep fold JSON + arrays npz — `_p2_nrm_repar_sweep._fit_one_fold_repar`,
   written to `<arm>/<cell_name>/fold_d{s}_f{f}.json` + `arrays_d{s}_f{f}.npz`.
   Row: `cell_name, group, encoder, decoder, torch_decoder, n_cats, Q, N, E,
   config, state_alpha, item_key_dim, nrm_channel, data_seed, init_seed, fold,
   fit_time_s, n_params, **pred, **theta (incl a_spearman,
   theta_spearman_lastvalid), **item, **diag, **repar_info, status`. npz:
   `theta_hat_lastvalid, theta_true, theta_track, val_rows, train_rows,
   coverage_full, item_*`.
3. Traj export npz + sidecar — `_p2_v3_export._write_traj`. npz:
   `theta_track, theta_final, lengths, items(int32), responses(int8),
   mask(bool), train_idx(int32), val_idx(int32)`. Sidecar JSON: `cell, stage,
   data_seed, fold, decoder, dataset, n_items, n_cats, n_learners, seq_len,
   npz, theta_final_mean, theta_final_sd, status`. GPCM thresholds npz:
   `alpha, beta(31,2), seen`.

4. MML calibration references — `_p2_realstudy_mml.fit_unit` and the
   `_p2_mml_*` family. Per `<dataset>_<decoder>`: `in.csv` (long triplets,
   the exact matrix the R fit saw), `items.csv` (mirt coefficients),
   `reference.json` (item-aligned parameters + `seen`, the ONLY file
   downstream code reads), `rlog*` (Rscript stdout for audit). These are
   first-class synced artifacts: produced only on R-capable machines,
   consumed everywhere through `mml_reference()` with no R dependency. They
   are small (JSON + CSV) and live in the git-light set, so even a fresh
   SLURM clone has them after `git pull`.

**Calibration runs get manifests too.** A calibration manifest uses the same
schema with `spec.stage = "calibrate-mml"`, `env.r = {version, mirt_version,
rscript_path}` in place of the torch/CUDA block, and the artifact list above.
This is what makes an R-less machine trust a reference: the manifest names the
producing machine, the R and mirt versions, and the input hash.

A `tests/test_schemas.py` pins these key sets and dtypes so a future edit that
would silently change a schema fails a gate rather than a downstream figure.

---

## 3. Results sharing across three machines + SLURM leg

**The workload.** Solo researcher, Windows-primary, three machines. The laptop
needs the light set (analysis JSONs, tables, figures; under about 30 MB). The
Windows PC and the SLURM head produce and re-score, so they need the heavy set
(traj npz, NRM fold JSONs, dataset cache; the bulk of 2.6 G, and only about
1 G once superseded probe trees are pruned). Offline tolerance and near-zero
cost are wanted.

**Options weighed.**

| option | Windows | partial sync (laptop=light) | offline | cost | solo overhead | verdict |
|---|---|---|---|---|---|---|
| (a) DVC + rclone remote (Drive/OneDrive/bucket) | first-class, pip-only | yes, `dvc pull <target>.dvc` | yes | free tier fits after prune | moderate | RECOMMENDED |
| (b) git-lfs on private GitHub | ok | no, LFS pulls all | yes | 2.6 G exceeds 1 G free, paid | low | rejected |
| (c) plain git (light) + content-addressed dir via rclone | ok | yes, manual | yes | free | you build the CAS | this is DVC by hand |
| (d) syncthing whole tree | first-class | selective/receive-only folders, coarse | yes, P2P | free | low, but no provenance/versioning | good fallback |

**Recommendation: DVC layered on git, with an rclone-backed remote.** DVC is
option (c) productized: content-addressed cache, `.dvc` pointer files in git,
remotes over rclone (so Google Drive, OneDrive, or an S3/GCS bucket all work).
It gives the exact split the machines need.

- Git tracks the light, regenerate-anywhere set: code, configs, manifests,
  `results/p2_v3_analysis/*.json`, `results/p2_v3_export/tab_*`, and `figures/`.
  Under about 30 MB, versioned inline.
- DVC tracks the heavy set: `results/**/*.npz`, the NRM fold JSONs, and
  `data/cache/`. Pointer files (`*.dvc` or a single `results.dvc` per subtree)
  live in git; blobs live in the remote.
- Partial sync is the win. Laptop does `git pull` and gets everything light,
  then optionally `dvc pull figures` for a fresh render. The PC and the head do
  `dvc pull` for the full heavy set. No machine pulls what it does not need.
- The dataset cache becomes a first-class DVC artifact keyed by its SIZE
  signature and content hash, so the EdNet 780k-file glob runs ONCE ever; every
  other machine `dvc pull`s `data/cache/ednet_nrm_*.npz` in seconds. This alone
  removes the 20-minute cold-start from two of three machines.

**Concrete setup sketch.**

```bash
pip install "dvc[gdrive]"                # or dvc[s3], dvc[all]
dvc init
dvc remote add -d store gdrive://<folder-id>      # or onedrive via rclone remote
#   rclone route:  rclone config -> onedrive/drive ;  dvc remote add -d store rclone://<remote>/deep-irt
dvc config cache.type reflink,hardlink   # avoid file copies; big on Windows w/ many small files

# track the heavy subtrees + the dataset cache
dvc add results/p2_v3_export/traj results/p2_nrm_repar results/p2_toggle \
        results/p2_v3_arm1r results/p2_v3_hardnrm data/cache
git add results/**/*.dvc data/cache.dvc .dvc/config .gitignore
git commit -m "results: DVC-track heavy artifacts + dataset cache"
dvc push                                 # blobs -> remote

# on a second machine
git clone <deep-irt> && cd deep-irt && pip install -e .
dvc pull figures results/p2_v3_analysis  # laptop: light + figures only
dvc pull                                 # PC/head: everything
```

Caveat recorded honestly: DVC re-hashes on `add`/`status`, and `results/` has
about 13k npz + 16k json, so a full `dvc status` on Windows is not instant.
Mitigations are `cache.type=reflink,hardlink`, tracking whole subtrees rather
than per-file `.dvc`, and pruning superseded probe trees first (below). If DVC
ceremony ever outweighs its value for a solo workflow, the escape hatch is
syncthing (option d) with a receive-only light folder on the laptop and a full
folder on the producers; it loses provenance and versioning, which is why it is
the fallback, not the pick.

**Prune before sync.** Of the 2.6 G, the shipped campaign is
`p2_v3_arm1r + p2_v3_export + p2_realstudy + p2_toggle + p2_nrm_repar/arm1r +
p2_v3_hardnrm + p2_v3_analysis`, roughly 1.3 G. The `p2_exposure*`, `p2_realtax`,
`p2_mml`, `p2_rescore`, and the killed-arm trees are candidates to leave out of
the shared store (or push to a cold `archive/` remote), pending the
figure-source clearance in phase 3. Do not delete; exclude from the hot remote.

**Machine-capability profiles.** Three machines, three tracked YAMLs under
`configs/machines/`, selected by `DEEP_IRT_MACHINE` env (fallback: hostname
match), loaded by `pipeline/paths.py` alongside `RESULTS_ROOT`/`DATA_ROOT`:

```yaml
# configs/machines/pc.yaml
machine: pc
capabilities: {cuda: true, r: true}
r: {rscript: 'C:\Program Files\R\R-4.5.0\bin\Rscript.exe'}
env: {KMP_DUPLICATE_LIB_OK: 'TRUE', CUDA_LAUNCH_BLOCKING: '1'}   # Windows footguns

# configs/machines/laptop.yaml
machine: laptop
capabilities: {cuda: false, r: true}
r: {rscript: '<laptop Rscript path>'}

# configs/machines/slurm.yaml
machine: slurm
capabilities: {cuda: true, r: false}     # R intentionally absent; MML refs come from the store
```

Behavioral contract: a stage declares its required capability; the entrypoint
checks the active profile and fails fast with a pointed message
(`calibrate-mml requires capabilities.r; this machine (slurm) consumes MML
references from results/p2_realstudy/mirt/ instead`). No stage probes for
`Rscript.exe` on PATH, no stage hardcodes a path. The five files carrying
`RSCRIPT_EXE` constants today are user-owned and get this one mechanical
substitution during promotion (phase 3); the resolved command on the author's
PC is character-identical, so outputs cannot drift.

**SLURM leg (Linux compute head).** The remote head runs SLURM, so every
entrypoint must be Linux-clean and array-friendly. R is deliberately NOT part
of this leg: the head never calibrates, it consumes `reference.json` from the
git-light store (already present after `git pull`; no DVC pull needed for
them).

- Entrypoints are Python CLIs with no hardcoded `C:/Users` paths and no
  Windows-only shell. Path resolution goes through `pipeline/paths.py`
  (`RESULTS_ROOT`, `DATA_ROOT` from env or a config, defaulting to repo-relative
  via `Path(__file__)`), which is already how `_p2_v3_analysis_lib.REPO` works.
- Packaging (`pyproject.toml` + `pip install -e .`) retires
  `PYTHONPATH=".;rl/src;ma-irt"`; `import deep_irt` resolves as an installed
  package on any OS.
- Thin per-platform launchers: `scripts/launch/windows.ps1` (sets
  `KMP_DUPLICATE_LIB_OK`, optional `CUDA_LAUNCH_BLOCKING`) and
  `scripts/launch/linux.sh`. They set env and call the same Python CLI; they
  hold no logic.
- The unit-of-work already maps onto a SLURM array. Today `(cell, seed, fold)`
  is atomic with `--skip-done` resume. Add
  `pipeline/manifest.enumerate_units(run_spec) -> list[Unit]` returning a
  deterministic ordered list; a `scripts/train_unit.py --run <spec> --index $I`
  selects `units[I]`, writes its atomic artifact, and is idempotent under
  requeue because `--skip-done` short-circuits a finished unit. An sbatch
  `--array=0-N%K` then fans the run across the head, one array task per unit.
- `scripts/slurm/sbatch_array.template.sh` is a PLACEHOLDER stub. It is gated on
  the author's remote instructions (partition, account, module loads, data
  staging path) and marked a stub phase (phase 7). The enumeration and the
  `--index` selector are built and unit-tested locally now; the sbatch wrapper
  is filled in only after local tests pass and the author supplies the remote
  details.

```bash
# scripts/slurm/sbatch_array.template.sh  (STUB; <<...>> filled from author's remote spec)
#SBATCH --job-name=deep-irt
#SBATCH --array=0-<<N_UNITS-1>>%<<CONCURRENCY>>
#SBATCH --partition=<<PARTITION>>
#SBATCH --gres=gpu:1
#SBATCH --time=<<HH:MM:SS>>
set -euo pipefail
module load <<CUDA/PYTHON MODULES>>
source activate research
python scripts/train_unit.py --run "<<RUN_SPEC>>" --index "${SLURM_ARRAY_TASK_ID}" --skip-done
# CUDA_LAUNCH_BLOCKING intentionally unset on Linux (Windows-driver footgun); recorded in manifest.
```

---

## 4. Import/shim strategy + parent-repo relationship

**Vendor Codex-owned files by pure relocation, no shim.** `git mv` preserves
content byte-identical (git shows a rename, R100), and because every import is
an absolute package path (`deep_irt.core.*`, `deep_irt.bench.*`), the imports
still resolve after the package root moves to `src/deep_irt/`. A shim would
leave a second copy or an indirection module, which contradicts the clean-repo
requirement and creates a stale-copy maintenance seam. A rename is not an edit;
it changes neither content nor behavior, so it honors the no-touch constraint's
letter and spirit better than a shim does. The ownership boundary becomes a
`CODEOWNERS` file listing `src/deep_irt/core/`, `src/deep_irt/bench/run_*`,
`datagen.py`, `engines.py`, `_ednet_ot*.py`, not a physical split.

The one caveat is the path change from `deep_irt/core` to `src/deep_irt/core`.
This is why the vendor move is a single atomic phase executed while Codex is
paused, and why the flat-layout fallback exists if that pause cannot be
arranged.

**Parent-repo relationship: `deep-irt` as a submodule of `deep-mirt`.** The
parent already carries three submodules (ma-irt, overleaf-sync, docs/slides),
so a fourth is idiomatic. As a submodule:

- Codex keeps editing `core/bench` inside the submodule working tree; the
  parent tracks a pinned submodule sha, exactly as it already does for the
  paper. The recent parent history is dominated by "submodule pointer" commits,
  so the workflow is proven.
- `deep-mirt` keeps `ma-irt/`, `rl/`, `docs/`, and the paper; `deep-irt` owns
  the active framework and its results. Legacy and archived trees stay in the
  parent and are never imported.
- A sibling clone (rejected) would break the parent's ability to pin a tested
  state and would fragment Codex's cross-file edits across two working copies
  with no atomic commit spanning them.

Recommendation: submodule. The DVC remote and the git remote for `deep-irt` are
its own (a private GitHub repo for the code and light results, an rclone remote
for the heavy blobs); the parent pins the sha.

---

## 5. Migration phases with verification gates

The choke point makes the master gate cheap: repoint one base, regenerate a
named golden set, diff against a baseline captured before any file moves.

**Golden set (the master-gate targets).** Chosen to exercise all three schemas
and both real and synthetic beds, and because the ledger (sections 15-18) marks
them shipped:
- Numeric, byte-identical guaranteed: `tab_real_metrics.json`,
  `tab_real_metrics_allenc.json`, `results/p2_v3_analysis/{ability_shsk,
  agreement_shsk_points,case_shsk_numbers}.json`, and the `tab:mass` NRM JSON.
- Rendered, determinized: `fig_dd` (toggle + arm1r synthetic),
  `fig_agreement_both` (arm1r real EdNet-NRM fold JSONs + traj),
  `fig_reversal_bridge` (arm1r + hardnrm), `fig_ednet_2in1`.

Honesty on "byte-identical figures." Pure relocation changes zero bytes, so a
re-render on the SAME machine with the SAME matplotlib reproduces the committed
figure exactly; that is the primary gate. Cross-machine PDF byte-identity is
NOT guaranteed by default (matplotlib embeds timestamps and subsets fonts). To
claim it, pin matplotlib and set determinism (`SOURCE_DATE_EPOCH`,
`metadata=None`, fixed font cache). Where that is not worth it, the gate is
byte-identical on the NUMERIC JSON that feeds the figure plus a pixel-diff
tolerance on the render. This distinction is stated in `docs/schemas.md` so no
one over-claims.

| phase | work | gate |
|---|---|---|
| 0. Baseline freeze | Hash the golden set in place (before any move). Record file count + sha256 of every `outputs/` artifact to be relocated. | Baseline manifest committed; golden hashes stored. |
| 1. Scaffold | Create `src/`, `scripts/`, `configs/`, `data/`, `results/`, `figures/`, `checkpoints/`, `docs/`, `tests/`. Add `pyproject.toml`, `pip install -e .`, `CODEOWNERS`, `.dvcignore`. | `python -c "import deep_irt"` works with NO `PYTHONPATH`; existing `deep_irt/tests` green. |
| 2. Vendor core/bench (Codex paused) | `git mv` `core/`, `run_*`, `datagen.py`, `engines.py`, `_ednet_ot*.py`, and the 17 tracked p2-core files into `src/deep_irt/`. No content edits. | `git diff` shows pure renames (R100), zero content change; import smoke + tests pass. |
| 3. Promote pipeline subset | Move the load-bearing `_p2_*` subset into `src/deep_irt/pipeline/{producers,analysis,figures,calibrate}`; clear each candidate against the 3-check rule (docs cite / imported-by-kept / regenerates-committed-figure); leave the rest in `archive/scratch/`. Extract the R/MML stage into `calibrate/` and replace the five `RSCRIPT_EXE` constants with the machine-profile lookup. | Every promoted script imports; its `--smoke` path runs; no committed figure is orphaned. Calibrate gate: resolved Rscript command line on the PC is character-identical to the old constant; `slurm` profile fails fast with the consume-from-store message. |
| 4. Relocate artifacts | Move `outputs/p2_*` into `results/` (subdir names verbatim); move `figs/` to `figures/`; move the datacache to `data/cache/`. Backfill `runs/<run_id>/manifest.json`. | Post-move file count + sha256 equal the phase-0 baseline for every artifact (pure move, no rewrite). |
| 5. Repoint the choke point | Change `pipeline/paths.py` + the analysis-lib base + the producer roots to derive from `RESULTS_ROOT` (default repo-relative for back-compat). Regenerate the golden set. | MASTER GATE: golden numeric JSON byte-identical to phase-0 baseline; golden figures byte-identical on the same machine (or within tolerance cross-machine per the determinism note). |
| 6. Wire DVC + remote | `dvc init`, track heavy subtrees + `data/cache`, `dvc push`. On a second machine, fresh clone + `dvc pull` + regenerate golden set. | Second-machine regeneration byte-identical to phase-0 baseline. |
| 7. SLURM leg (STUB, gated) | Build `enumerate_units` + `train_unit.py --index`; unit-test the array-index-to-unit map on Linux (no fit). Fill `sbatch_array.template.sh` only after the author supplies remote details. | Array-index enumeration matches the local unit list exactly; a `--dry-run` fit selection is correct for a spot set of indices. Held until author's remote instructions land. |

Rollback: phases 2-6 are `git mv` + DVC pointer commits, each revertible; the
gitignored `outputs/` original is retained until phase 6 passes on a second
machine, so no artifact is unrecoverable until the shared store is proven.

---

## 6. Sonnet-tier task decomposition (mechanical work per phase)

Opus plans and gates; Sonnet does the moves and the boilerplate. Sizes: S under
an hour, M a few hours, L a day. Codex-owned files are touched only by `git mv`
in phase 2 and never edited.

**Phase 0.** T0.1 (S) script to sha256 + size every `outputs/` artifact into a
baseline manifest. T0.2 (S) render the golden set once, store its hashes.

**Phase 1.** T1.1 (M) author `pyproject.toml` (package `deep_irt`, deps pinned
from the working env; mirror `rl/pyproject.toml`). T1.2 (S) `CODEOWNERS`,
`.dvcignore`, `.gitignore` for the new tree. T1.3 (S) create the empty tree +
`__init__.py` files. T1.4 (S) `pip install -e .` and confirm `import deep_irt`
with no `PYTHONPATH`.

**Phase 2.** T2.1 (M) `git mv` core/bench Codex files + the 17 tracked p2-core
files into `src/deep_irt/`; verify R100 renames. T2.2 (S) update `CODEOWNERS`
to the new paths. T2.3 (S) run tests; fix only import roots in NON-Codex callers
if any relative import breaks (absolute imports should not).

**Phase 3.** T3.1 (M) apply the 3-check clearance to the 98 untracked `_p2_*`;
emit a promote/archive ledger. T3.2 (M) `git mv` the promoted subset into
`pipeline/{producers,analysis,figures}`; add a compatibility import shim ONLY
inside the pipeline package if two promoted files import each other by the old
`deep_irt.bench._p2_x` path (rewrite those imports, they are user files).
T3.3 (S) move `configs_p2/` to `configs/`. T3.4 (M) smoke every promoted
entrypoint. T3.5 (M) extract `pipeline/calibrate/` (four `.R` fitters + five
Python callers), add the machine-profile loader in `paths.py` +
`configs/machines/{pc,laptop,slurm}.yaml`, replace the `RSCRIPT_EXE` constants
with the profile lookup, add `scripts/calibrate_mml.py` with the
capability-gate fail-fast; unit-test that the PC-profile command line equals
the old constant string.

**Phase 4.** T4.1 (M) `git mv`/move `outputs/p2_*` to `results/` verbatim; move
`figs/` to `figures/`; move datacache to `data/cache/`. T4.2 (M)
`manifest.py`: `write_manifest(run_dir)` that scans a tree and emits the schema
in section 2; backfill manifests for the shipped runs. T4.3 (S) re-hash and
diff against the phase-0 baseline.

**Phase 5.** T5.1 (M) `pipeline/paths.py` with `RESULTS_ROOT`/`DATA_ROOT`
resolution; repoint the analysis-lib base and the three producer roots to it,
default repo-relative. T5.2 (S) `scripts/make_figures.py` that regenerates the
golden set from `RESULTS_ROOT`. T5.3 (S) run the master gate diff.

**Phase 6.** T6.1 (S) `dvc init` + remote config (Drive/OneDrive via rclone).
T6.2 (S) `dvc add` the heavy subtrees + `data/cache`; commit pointers.
T6.3 (M) second-machine dry run: clone, `pip install -e .`, `dvc pull`,
regenerate golden set, diff.

**Phase 7 (gated).** T7.1 (M) `enumerate_units(run_spec)` + `train_unit.py
--index`; unit-test the map. T7.2 (S) `stage_data.py` + `data/registry.json`
(dataset -> source, SIZE sig, build cmd, hash) so the head builds or pulls the
cache. T7.3 (S) fill `sbatch_array.template.sh` from the author's remote spec
(deferred). T7.4 (S) `scripts/launch/{windows.ps1,linux.sh}` thin launchers.

---

## Executive summary

1. Target layout is `src/deep_irt/` (one installed package) plus one top-level
   directory per artifact class: `scripts/`, `configs/`, `data/`, `results/`,
   `figures/`, `checkpoints/`, `docs/`, `tests/`. Separation is the invariant;
   `results/` keeps the internal `p2_*` subdir names verbatim so relocation is
   provably content-neutral.
2. `pyproject.toml` + `pip install -e .` retires `PYTHONPATH=".;rl/src;ma-irt"`;
   `import deep_irt` resolves on any OS, which is the precondition for the Linux
   SLURM head.
3. The whole system pivots on one choke point,
   `_p2_v3_analysis_lib.py`'s path base, mirrored by three producer roots.
   Repointing that base to a single `RESULTS_ROOT` moves every artifact tree at
   once and is what makes the byte-identical gate cheap.
4. Codex-owned `core/bench` files are vendored by pure `git mv` (content
   byte-identical, absolute imports position-independent), no shims; ownership
   becomes a `CODEOWNERS` entry. `deep-irt` is a submodule of `deep-mirt`,
   matching the existing ma-irt pattern, so Codex keeps editing in place.
5. Sync is DVC layered on git with an rclone remote (Google Drive or OneDrive):
   git holds code, configs, manifests, tables, and figures (about 30 MB);
   DVC holds npz, NRM fold JSONs, and the dataset cache. Partial pull gives the
   laptop figures-only and the producers everything. The EdNet 780k-file glob
   runs once ever, then every machine pulls the cached matrix.
6. A run manifest (`manifest.json` per sweep) records code sha, env, dataset and
   config hashes, artifact hashes, timing, and the `CUDA_LAUNCH_BLOCKING`
   Windows-footgun flag. The `(cell, seed, fold)` unit maps directly onto a
   SLURM array index via `enumerate_units` + `train_unit.py --index`, with
   `--skip-done` making requeues idempotent.
7. R/MML calibration is an extracted optional stage (`pipeline/calibrate/` +
   `scripts/calibrate_mml.py`): Rscript path from a machine profile
   (`configs/machines/{pc,laptop,slurm}.yaml`, capability flag `r:
   true/false`), never hardcoded; the SLURM profile excludes R and consumes
   the small, git-tracked `mirt/*/{reference.json,items.csv,in.csv}`
   references from the store. Calibration runs carry manifests recording R and
   mirt versions.
8. Seven phases, each with a gate; the master gate is byte-identical
   regeneration of a named golden set (numeric tables + four figures) from
   relocated artifacts, first on the same machine, then after a second-machine
   `dvc pull`. The SLURM leg is a gated stub awaiting the author's remote
   instructions.
9. Byte-identity honesty: relocation changes zero bytes and numeric JSON
   regenerates exactly; PDF byte-identity across machines needs matplotlib
   pinning + determinism, else the gate falls back to the numeric intermediate
   plus a pixel tolerance.

Top three risks.
- R1. Codex is committing actively (parent history is submodule-pointer churn).
  The phase-2 vendor `git mv` changes Codex's file paths and must run at a
  quiescent point with Codex paused, or fall back to a flat `deep_irt/` layout
  that leaves paths identical. Mis-sequencing collides with in-flight edits.
- R2. Promotion misclassification. My cleanup-audit experience says
  underscore-prefixed scratch is often the sole regenerator of a shipped figure
  (for example `_p2_toggle_sweep` produces the `p2_toggle` grid behind `fig_dd`;
  `_p2_exposure_sweep` may feed `fig_stability_exposure`). Every archive
  candidate must pass the 3-check clearance in phase 3 before it is left out;
  default is promote, never silent-delete.
- R3. DVC overhead on many small files. `results/` has about 13k npz + 16k JSON,
  so `dvc status` on Windows is slow. Mitigate with `cache.type=reflink,
  hardlink`, whole-subtree tracking, and pruning superseded probe trees from the
  hot remote. If ceremony still outweighs value, syncthing (receive-only light
  folder on the laptop) is the documented fallback, at the cost of provenance.
