# Refactor plan, researcher-workflow half: pipeline I/O contracts

Scope. This document specifies the WORKFLOW CONTRACTS for making `deep_irt/`
a portable, standardized repo. A separate systems document owns physical
layout, sync, and migration. This half owns what the working researcher
actually does and precisely what the pipeline's inputs and outputs must
guarantee so those workflows keep functioning unchanged after the move.

Hard rule (author). No change to model behavior, metric computation, or
plot rendering. Formats and locations standardize. Computations do not.
Every contract below is written so a reorganization can satisfy it without
touching a training recipe, a metric definition, or a figure's look.

Portability target (author addendum). The remote compute head runs a
scheduler (SLURM as a placeholder; full remote instructions come later).
Every workflow and staging step must be OS-agnostic and headless. Three
Windows-isms in the current pipeline are called out inline as portability
debts: the shell-session chain scripts, the `;` PYTHONPATH separator, and
hardcoded `C:/Users/steph/...` roots.

Ground truth for how work actually flows is `docs/v3_results_record.md`
sections 0-19 (the just-closed v3 campaign). The workflows below are
abstracted from that record and from the drivers it ran.

---

## 0. The recurring loop (what actually happened, sections 0-19)

One pass of the campaign loop, with the concrete files:

1. Train grid cells on the GPU box through resumable chain scripts
   (`outputs/p2_v3_arm1r/_arm1r_full_chain.sh`), each stage a
   `--skip-done` slice of a driver
   (`deep_irt/bench/_p2_realstudy_hardnrm.py` for real cells,
   `deep_irt/bench/_p2_nrm_repar_sweep.py` for synthetic). Unit of work is
   one `(cell, seed, fold)` fit with atomic JSON+npz writes.
2. Score and gate against pre-registered bars
   (`deep_irt/bench/_p2_arm1r_gate_score.py` writes
   `outputs/p2_v3_arm1r/VERDICT_arm1r.{md,json}`).
3. Regenerate analysis artifacts and figures through the path-resolving
   analysis library (`deep_irt/bench/_p2_v3_analysis_lib.py`, globals
   `REPO/REALSTUDY/EXPORT/TRAJ/TIMSS_THR/MIRT/TOGGLE/OUT/FIGS/SEEDS/FOLDS`).
   Every analysis and figure script resolves paths through this lib. This
   single indirection is what let the campaign stage or re-point trees
   (compare an `arm1r` tree against the shipped `arm1` tree without
   touching shipped cells).
4. Update manuscript tables from computed values with validated machinery
   (`deep_irt/bench/_p2_v3_tabmass_nrm.py` reproduces existing manuscript
   entries within a tolerance before it is trusted to emit new ones).
5. Record in `docs/v3_results_record.md` and push figures to the paper
   repo submodule `overleaf-sync/` (via `copy_to_overleaf` into
   `overleaf-sync/figures/`).

Everything downstream of step 1 is CPU-only and reads artifacts. Steps 2-5
must run on a laptop with no GPU and no raw data, given only the artifact
store. That separation is the backbone of every contract here.

---

## 1. Workflow inventory

Each workflow names its inputs, its outputs, and the contract that must not
break across the refactor.

### W1. Train a grid (GPU box)

- Inputs: a driver + its embedded grid definition; a staged dataset (real
  datacache npz, or a synthetic bed regenerated in-process from `(cfg,
  seed)`); the fixed training recipe constants (`N_EPOCHS`, `LR`,
  `BATCH_SIZE`, `HIDDEN_DIM`, seeds).
- Outputs: per unit, one fold-record JSON + one arrays npz (W-artifacts a,
  b in section 3), written atomically; a per-run progress file.
- Contract. The UNIT OF WORK is one `(cell, seed, fold)` fit. A unit is
  self-contained: its inputs are fully determined by the cell name, the
  data seed, the fold index, and the pinned recipe constants; its outputs
  are two files at deterministic paths. No unit depends on another unit's
  in-memory state. This is the portable primitive (see W2 and the
  scheduler note).

### W2. Resume after a crash (idempotent)

- Inputs: the partially populated output tree from a prior W1 run.
- Outputs: only the missing units, filled in.
- Contract. `--skip-done` must be exactly idempotent. Re-running a
  completed grid does zero work and changes zero bytes. Done-ness is
  decided by reading the artifact store, never by shell-session state.
  Current done-detection (must be preserved in spirit, unified in the
  refactor, see the divergence note):
  - Real driver (`_p2_realstudy_hardnrm._unit_done`): fold JSON exists AND
    its `status == "done"` AND the traj npz exists. All three required, so
    a crash mid-write never counts as done.
  - Synthetic driver (`_p2_nrm_repar_sweep._fold_done`): fold JSON exists
    (presence only). Failed folds write a `status == "failed"` JSON, which
    counts as present, so `--retry-failed` must clear them before a rerun.
  DIVERGENCE TO RESOLVE: the two drivers disagree on what "done" means
  (both-artifacts-and-status vs mere-presence). Standardize on the
  stricter rule (both artifacts present AND `status == "done"`), and make
  a `failed` record never satisfy done. This is a format/robustness fix,
  not a computation change.

- Crash-safety write ORDER (must be preserved). `np.savez` refuses a
  `.tmp` staging name (it appends `.npz`, so `foo.npz.tmp` lands at
  `foo.npz.tmp.npz` and the rename fails on Windows). The pipeline
  therefore gets crash-safety from write ORDER, not from a temp-rename on
  the npz: write the npz first, then any secondary JSON, then the
  authoritative fold JSON LAST with `status == "done"` via
  tmp-write-plus-`os.replace` (atomic). Because done-ness requires the
  last file, a crash at any earlier point leaves a not-done unit that W2
  re-runs cleanly. Any refactor of the writer must keep this ordering
  invariant (npz -> sidecar json -> authoritative json last, atomic).

### W3. Chain a night of grids (scheduler-portable)

- Inputs: a list of grid slices to run in sequence, each resumable.
- Outputs: the union of all units, plus a progress trail.
- Current form (Windows-session scratch, PORTABILITY DEBT). The campaign
  used bash chain scripts (`outputs/p2_v3_arm1r/_arm1r_full_chain.sh`) that
  loop `--skip-done` slices, append human lines to a `HEARTBEAT.md`, and
  read back `REMAINING_UNITS` / `REMAINING_FOLD_UNITS` printed on stdout.
  This couples progress to a live shell session and to stdout scraping.
- Replacement contract. Progress must be DERIVABLE FROM THE ARTIFACT STORE
  ALONE. A `remaining(grid)` function that scans the output tree and counts
  not-done units is the single source of truth (both drivers already have
  `count_remaining`; promote it to the contract and drop stdout scraping).
  A run may still emit a heartbeat file, but nothing may DEPEND on it;
  killing and restarting the chain must recover state purely from the
  files on disk. The heartbeat becomes an optional human convenience, not
  a control input.
- Scheduler mapping (SLURM placeholder). Because a unit is fully specified
  by `(cell, seed, fold)`, the natural port is one scheduler array index =
  one unit. A driver invoked with a unit selector (e.g.
  `--only <cell> --seed s --fold f`, or an array-index-to-unit mapping)
  runs exactly one atomic unit, and `--skip-done` makes re-queued or
  overlapping array tasks safe. The same idempotent-resume semantics that
  work under a `for` loop today work under an array scheduler because they
  live in the artifact store, not the loop. The refactor should expose a
  clean unit-selector CLI so the SLURM wrapper is a thin adapter, and must
  keep the drivers Linux-runnable (shebang-agnostic; invoked as
  `python -u <driver> ...`, no bash-only constructs in the driver itself).

### W4. Score and gate against pre-registered bars (CPU)

- Inputs: the fold JSONs + traj npz for the cells under test; the frozen
  reference numbers and clause bars (currently literals in the gate
  script, e.g. `_p2_arm1r_gate_score.REF` and the `>= .22 / >= .625 /
  <= 1.00` clauses, mirroring `outputs/p2_v3_analysis/mathfix_round4.md`).
- Outputs: a verdict record `VERDICT_<arm>.{md,json}` with a per-seed
  table, a clause table, and PASS/PARTIAL/FAIL plus the pre-registered
  interpretive branch.
- Contract. Gating reads artifacts only, recomputes the gated quantities
  from stored params + traj (e.g. theta-vs-rawscore |rho| from
  `theta_final`, `items`, `responses`, `mask`, `key`), and never re-fits.
  The frozen bars are inputs, not results; the refactor must keep them
  legible and version-pinned (a bar is a pre-registration, so its
  provenance, the commit that froze it, matters). The verdict JSON is a
  LIGHT artifact and must always sync.

### W5. Compute a table value with validated machinery (CPU)

- Inputs: the fold JSONs of the target cells; the existing manuscript
  entries the machinery must reproduce first.
- Outputs: paste-ready table rows + a `{name}.{md,json}` pair.
- Contract (self-validation gate). Before emitting any new number, the
  machinery recomputes entries already in the manuscript and aborts unless
  it reproduces them within a stated tolerance (`_p2_v3_tabmass_nrm`:
  mean within .0015, bootstrap CI bound within .006, over the `arm1`
  trees). The bootstrap convention (dataset-clustered, resample the data
  seeds, 10k replicates, percentile interval, `default_rng(0)`) is part of
  the contract and must be reproduced bit-for-bit. The refactor may move
  where cells live but must keep the validation gate wired to a stable set
  of reference entries.

### W6. Regenerate figures on a non-GPU machine (CPU)

- Inputs: analysis/export artifacts resolved through
  `_p2_v3_analysis_lib` globals; the shared style module
  `_paperfig_style` (palette, page widths, `savefig_both`, usetex smoke
  test with mathtext fallback).
- Outputs: `{stem}.pdf` (vector) + `{stem}.png` (flat review DPI) in
  `outputs/p2_v3_analysis/figs/`, optionally a `{stem}_caption.txt`.
- Contract. Figure scripts must run with NO torch import and NO GPU (the
  style module and analysis lib are torch-free by construction). Path
  resolution goes through the analysis-lib globals so a figure can be
  re-pointed at a staged tree without editing the script. Rendering look
  is frozen (see non-goals): palette hex, marker grammar, page widths,
  `pdf.fonttype 42`, and the usetex-with-fallback behavior are unchanged.
  The refactor may standardize the figs directory location and the
  caption/provenance sidecar (section 3d) but must not alter what a figure
  looks like.

### W7. Exact held-out re-scoring from artifacts (CPU, precision-critical)

- Inputs: a traj npz (`theta_track`, `items`, `responses`, `mask`,
  `val_idx`) + the matching fold JSON (`alpha`, `beta`, `acc`).
- Outputs: per-scored-position hit/nll arrays; a pooled accuracy that
  MUST equal the on-disk `acc` to machine precision.
- Contract (the exact-rescore contract, `_p2_v3_ednet_2in1`). This is the
  load-bearing reproducibility guarantee and must survive verbatim:
  - `theta_track[:, t]` is the responsive ability AFTER answering position
    `t`. To predict position `t` the model uses ability after `t-1`, so
    the predictive theta is `theta_track[:, t-1]` (and `theta_track[:, t]`
    at `t == 0`).
  - The held-out window is the last `min(H, Ln)` VALID positions of each
    val-split learner, `H = 10` (`rs.N_HOLDOUT`); index the last valid
    steps, never the padding.
  - Scoring recomputes the head's own probability (2PL sigmoid, GPCM/NRM
    softmax with max-subtraction) from the fold JSON params at
    `theta_track[t-1]` and takes argmax for accuracy, `-log p[y]` for NLL.
  - A `_verify_rescore()` gate asserts the pooled re-score equals every
    on-disk fold `acc` to `< 1e-6`. Any refactor that changes npz field
    names, dtypes, or the meaning of `theta_track` breaks this and is
    forbidden. NRM option accuracy is invariant to the mirror gauge
    `(a_k, theta) -> (-a_k, -theta)`, so the rescore needs no sign
    correction; only direction-reading analyses use the per-fold sign key.

### W8. Cross-machine pull then plot (three-machine workflow)

- Inputs: the artifact store produced on the GPU box, transferred to a
  laptop; nothing else (no raw data, no GPU).
- Outputs: analysis artifacts, tables, figures.
- Contract. Everything W4-W7 needs must be present in the LIGHT artifact
  set (fold JSONs, analysis JSON/md, verdicts, sidecars) plus whatever
  HEAVY artifacts (traj/arrays npz) the specific analysis consumes. A pull
  is trustworthy only if each artifact carries the reproducibility stamp
  of section 4, so the laptop can confirm it is plotting the run it thinks
  it is. The refactor must make "which units are present and from which
  code state" answerable by scanning the store, not by asking the box.

---

## 2. Input side: dataset staging

### 2.1 Raw sources (find them; make their location configurable)

| dataset | raw form on disk today | loader |
|---|---|---|
| EdNet KT1 | `EdNet-KT1/KT1/u*.csv` (~784k per-learner CSV logs, scan capped at `scan_files=30000`) + `EdNet-Contents/contents/questions.csv` (answer key) at REPO ROOT | `deep_irt/bench/_ednet_reliability.py` (`build_item_bank`, `load_learners`, `load_correct_answers`), wrapped by `_p2_real.load_data`/`RealCell`, reached via `_p2_realstudy.load_dataset` |
| KDD | `data/kdd/algebra_2008_2009_train.txt` (~3.1 GB single TSV, streamed twice) | `deep_irt/bench/_kdd_reliability.py` (`KDD = ROOT/"data"/"kdd"/...`) |
| TIMSS | `data/timss/timss_g8_usa_poly_triplets.csv` (student,item,resp genuine 0/1/2) + `timss_g8_usa_gpcm_coef.csv` (classical GPCM reference). True raw is `data/timss/raw/*.sav` + `T19_G8_USA_SPSS.zip`, converted to triplets by the R scripts `data/timss/_build_timss_gpcm.R` etc. | `_p2_realstudy._load_timss` |

Everything raw and derived is gitignored: `/EdNet-KT1/`, `/EdNet-Contents/`,
`/data/`, `/outputs/`. Tracked are the loader/generator `.py` under
`deep_irt/bench/`, the `configs_p2/` YAMLs, and the TIMSS R build scripts.
So a fresh checkout has code and configs but NO data and NO cache; staging
must reconstruct both.

PORTABILITY DEBT 1 (hardcoded Windows roots). The loaders hardcode
`ROOT = Path("C:/Users/steph/documents/deep-mirt")` (`_ednet_reliability.py:39`,
`_kdd_reliability.py:28`, `_p2_real.py`). On the remote box these paths do
not exist. Staging must resolve the raw-data root from a single
configurable source (env var or a repo-root-relative default), never a
Windows absolute path, resolving with no interactive step. EdNet's ~784k-file
tree is the one heavy raw input and should be stageable independently of the
code checkout.

PORTABILITY DEBT 2 (external R toolchain, hardcoded R path). The classical
MML reference (section 2.5) shells out to Rscript at a hardcoded
`C:\Program Files\R\R-4.5.0\bin\Rscript.exe` running `_p2_mml_real.R` /
`_p2_mml_nrm.R`. The remote path must be configurable and the R
availability must be a declared, checkable staging prerequisite (headless,
no prompt), or the MML rows are unbuildable there.

PORTABILITY DEBT 3 (`;` vs `:` PYTHONPATH). The chain scripts export
`PYTHONPATH=".;rl/src;ma-irt"` (Windows). The portable form must select the
separator by platform (or set the path so it does not depend on the shell
literal).

ASSISTments is a known acquisition blocker (`_p2_realstudy.BLOCKED`): raw
response data absent. Staging must fail loudly and legibly for a blocked
dataset, not silently.

### 2.2 The expensive-scan problem, and the built-dataset cache

Cost. An EdNet cold build globs ~784k CSVs (scanning the first
`scan_files=30000`) and the KDD build streams a 3.1 GB TSV twice; each costs
minutes PER PROCESS. Roughly 20 minutes for EdNet in the campaign's usage
(the qid-join replicate quotes ~9 min for its slice). This must never be on
the critical path of a routine train/analyze loop.

What exists today. A cross-process built-dataset cache module
`deep_irt/bench/_p2_datacache.py`, writing `.npz` under
`outputs/p2_realstudy/direct/_datacache/`, one file per
`(dataset, decoder, size-signature)`. The key is
`f"{dataset}_{decoder}_{_sig(dataset)}.npz"`, where `_sig()` builds a
size-signature from the live `_p2_realstudy.SIZE[dataset]` dict (sorted
`key=value` pairs, or `"full"` for TIMSS), e.g.
`ednet_2pl_max_seq_len200_min_answers20_n_items250_n_learners2000_scan_files30000.npz`.
It stores `items, resp, mask, Q, n_cats`; atomic write via `.npz.tmp` +
`os.replace`, corrupt files fall through and rebuild. The `_p2_v3` scripts
assert byte-identity of a freshly built matrix against this datacache
(`_p2_v3_ednet_2in1.build_qid_join`), which is how neural fits and post-hoc
analyses stay aligned to the same item bank. There is also a second,
in-process memo (`_p2_realstudy.load_dataset`'s module-level `_DATA_CACHE`,
keyed `(decoder, dataset)`) that lives only for one process.

Gaps to close (make the cache a first-class versioned artifact).
- The `_sig()` key captures the SIZE dict but NOT the raw source, NOT the
  loader code, and NOT the coverage constant. So editing `MIN_NRM_COVERAGE`,
  the loader, or the raw EdNet files silently returns a stale cache unless a
  SIZE number changes. Add a content hash over (raw-source fingerprint,
  loader-code version, filter config incl. coverage) and store it IN the
  cache (an npz field or a companion `{cache}.meta.json`), so a cache
  validates against, or is invalidated by, any of those changing.
- The cache lives under `outputs/`. Promote it to a declared, versioned
  artifact class with an explicit build step (`build-datacache`) that is
  the ONLY place the 20-minute scan happens; every driver consumes the
  cache and refuses to cold-scan unless explicitly asked. This makes the
  three-machine workflow tractable (build once on the box, ship the
  cache).
- Coverage filtering is part of the identity. NRM applies
  `_coverage_filter` at `MIN_NRM_COVERAGE = 20` and remaps ids; the cache
  key must include coverage so a differently filtered bank is a different
  artifact. `_p2_realstudy` and `_p2_realstudy_mml` both call
  `load_dataset`, which is why ids stay aligned; the cache contract must
  preserve that single-source-of-truth property.

### 2.3 Synthetic beds

Three generators, all single-seed deterministic (a bed is fully
reproducible from `(cfg, data_seed)` on any machine):
- `deep_irt/bench/datagen.py::generate(cfg)` -- the clean/rectangular
  generator (`BenchDataConfig/BenchDataset/BenchGroundTruth`), seeded by
  `default_rng(cfg.seed)`.
- `deep_irt/bench/_p2_datagen_realistic.py::generate_realistic(cfg,
  data_seed)` -- the realistic administration bed used by the YAML-driven
  CV path (`_p2_run_cell`); all randomness from one `default_rng(data_seed)`.
- `deep_irt/bench/_p2_datagen_spiraled.py::generate_spiraled(cfg,
  data_seed)` -- the spiraled fixed-L bed used by the v3 toggle/repar
  sweeps (`_p2_nrm_repar_sweep`, `_p2_toggle_sweep`); one
  `default_rng(data_seed)` again. THIS is the generator behind the
  campaign's synthetic NRM cells.

Beds are NOT persisted to disk on the normal path; they are regenerated
in-process per `data_seed`. That is fine and cheap. The reproducibility
burden therefore falls entirely on capturing the `(cfg, data_seed)` that
defines a bed, which depends on the config path (see 2.4).

Synthetic seed/split conventions (DO NOT CHANGE, see non-goals): bed rng
`default_rng(data_seed)`; fold-split rng `default_rng(2000 + data_seed)`;
`init_seed = data_seed` (sweep path). The reliability loaders use
`GLOBAL_SEED = 42`.

### 2.4 Config system (two coexisting; both must reproduce their beds)

There are TWO config mechanisms, and the refactor must keep both
reproducible anywhere:

- YAML-driven (tracked, hashed). `deep_irt/bench/configs_p2/` holds ~52
  tracked YAML files consumed by `_p2_run_cell.py` through
  `_p2_config.load_config` into a `P2Config` (`data/model/train/eval/
  report` blocks). `_p2_config.config_sha256` hashes the experiment
  identity (data+model+train+eval, excluding report/description), and
  `_p2_run_cell` writes `results.json` with that hash, the full resolved
  config, and an env manifest INCLUDING the git SHA. This path already has
  the provenance section 4 asks for; it is the model to copy.
  (`configs_p2/real/` is a different schema, `RealCell` via
  `_p2_real.load_real_config`, for the real-data cells.)
- Code-embedded grids (the v3 campaign scratch drivers). The `_p2_*_sweep`
  drivers do NOT read YAML; they build each cell/`cfg` in Python
  (`_p2_toggle_sweep._cell` / `_make_cfg`, grid builders `build_grid`,
  `ARM_ENCODER_CONFIGS`, `GRID_N = [500,1000,2000,5000]`, `Q_FIXED`). A bed
  here is reproducible ONLY IF the exact code that built its cell is
  recoverable, i.e. identity = (code commit + cell name + data seed).

Contract. Whichever path built a fold, the fold record must let the bed be
reconstructed from the artifact alone. For the YAML path, keep writing the
resolved config + hash + commit (already done). For the scratch-driver
path, stamp the resolved `cfg` (or its hash) + commit into every fold
record (section 4), so the code-embedded configs stop being an
unrecoverable dependency. Do not change any seed derivation.

### 2.5 Classical MML reference staging (R dependency)

`deep_irt/bench/_p2_realstudy_mml.py` builds the classical MML calibration
that the agreement metrics compare against. It calls the SAME
`_p2_realstudy.load_dataset` the neural study uses (so item ids align),
dedups to first attempts, writes `outputs/p2_realstudy/mirt/<ds>_<dec>/
in.csv`, then shells to Rscript running `_p2_mml_real.R` (2PL) /
`_p2_mml_nrm.R` (nominal Bock), and normalizes the output into
`reference.json` (`alpha, beta/intercept, seen, n_seen, converged,
status`). TIMSS is special: no fit, it reads
`data/timss/timss_g8_usa_gpcm_coef.csv` directly into `reference.json`.
NRM restricts to estimable items (`NRM_MIN_COV=40, NRM_MAX_ITEMS=500`).

Contract. `reference.json` is a LIGHT staged input consumed by every
agreement analysis; it must be present in a pull. Its build has the R
prerequisite (PORTABILITY DEBT 2) and the datacache dependency, so on a
fresh machine the order is: stage raw -> `build-datacache` -> build MML
references -> everything else. The refactor should make that order an
explicit, headless staging sequence.

---

## 3. Output side: artifact taxonomy (formalize exactly what exists)

Six artifact classes. For each: required fields, and heavy (sync-optional)
vs light (always-synced).

### (a) Fold record (JSON) -- LIGHT

The atomic unit's authoritative record; the last file written; carries
`status == "done"`. Two schemas exist and must both be preserved (the
refactor may add fields, must not remove or rename the ones analyses
read).

Real-study fold JSON (`_p2_realstudy.run_fold`,
`_p2_realstudy_hardnrm.run_fold_hard`): `cell, encoder, decoder, dataset,
klass, data_seed, fold, n_learners, seq_len, n_items, n_cats, n_train,
n_val, seen (bool[]), n_seen, alpha (scalar[] or (Q,4)[]), beta ([] or
null), acc (dict: acc, nll, and per-decoder qwk / auc / macro-F1 /
n_scored), delta_slack, delta_n_items, timing_s{fit,total}, status`, plus
PROVENANCE `head` (arm1/arm1h/arm1r/g0/arm1s), `repar_epsilon`, `keyed`.

Synthetic fold JSON (`_p2_nrm_repar_sweep._fit_one_fold_repar`):
`cell_name, group, encoder, decoder, torch_decoder, n_cats, Q, N, E,
config, state_alpha, item_key_dim, nrm_channel, data_seed, init_seed,
fold, fit_time_s, n_params, status`, plus spread-in metric blocks
`**pred` (acc, qwk, nll, ...), `**theta` (`theta_spearman_lastvalid`,
...), `**item` (`a_spearman, b_spearman, c_spearman`), `**diag`, and
PROVENANCE `**repar_info` (`repar_arm, repar_epsilon`).

Required-field contract: any field named by an analysis loader
(`_p2_v3_analysis_lib`, gate/tabmass scripts) is load-bearing:
`alpha, beta, seen, acc.acc, acc.nll, a_spearman, c_spearman,
theta_spearman_lastvalid, data_seed, fold, status`. Provenance fields
(`head`/`repar_arm`, `repar_epsilon`, `keyed`) distinguish head variants
and must be preserved; section 4 extends them.

### (b) Arrays (npz) -- HEAVY (sync-optional)

Real traj npz (`traj/<cell>/traj_d{s}_f{f}.npz`): `theta_track (N,T),
theta_final (N,), lengths (N,), items (N,T) int32, responses (N,T) int8,
mask (N,T) bool, train_idx, val_idx`. Consumed by W7 (exact rescore), P2
theta-vs-rawscore, reversal bridge, exposure, DOA.

Synthetic arrays npz (`arrays_d{s}_f{f}.npz`): `theta_hat_lastvalid,
theta_true, theta_track, val_rows, train_rows, coverage_full`, plus
`item_*` pairs (`item_a_hat, item_a_true, item_b_hat, item_b_true, ...`).
Consumed by recovery scatters and per-item forensics.

Contract. Field names, dtypes, and the semantics of `theta_track`
(post-answer ability, so `t-1` predicts `t`) are frozen by W7. A companion
sidecar `traj_d{s}_f{f}.json` records `theta_final` summary + the npz
name; keep it as the lightweight index into the heavy file. Heavy npz may
be sync-optional, but any node running W7/recovery must fetch them.

### (c) Analysis artifacts ({name}.md + {name}.json pairs) -- LIGHT

Under `outputs/p2_v3_analysis/` (and `outputs/p2_v3_export/` for tables).
Examples: `stability_table.{md,json}`, `ednet_option_checks.{md,json}`,
`flip_forensics.{md,json}`, `reversal_bridge.{md,json}`,
`tab_real_metrics.{md,json}`, `tab_real_allenc.md`. The JSON is the
machine-readable computed values; the MD is the human/paste view. Contract:
they are always emitted as a pair, always resolved through the analysis-lib
`OUT` global, always light and synced. The JSON is the citable number; the
MD must agree with it.

### (d) Figures (pdf + png + sidecars) -- PDF ships to paper, PNG light

`savefig_both` writes `{stem}.pdf` (vector, to the paper) and `{stem}.png`
(flat `REVIEW_DPI = 150` review copy). `copy_to_overleaf(stems)` copies
both into `overleaf-sync/figures/`.

Caption sidecar: INCONSISTENT today (only 5 of 19 figures carry a
`{stem}_caption.txt`). Standardize: every figure emits a caption sidecar.

PROVENANCE RULE (new, lightweight). Every figure must name its generating
script and its input artifacts. Propose a `{stem}.prov.json` sidecar
emitted by `savefig_both` (or a thin wrapper) with fields: `stem, script
(the fig_*.py module), inputs (list of artifact paths/globs it read),
code_commit, created_utc, style_mode` (the `_paperfig_style.STYLE_MODE`
usetex-vs-fallback string, so a reviewer knows which render path made the
PDF). This makes a figure in the paper traceable back to the exact script
and artifacts without changing the figure's look. Sidecars are light and
always synced; the PDF is the paper's copy.

### (e) Verdict / gate records -- LIGHT

`VERDICT_<arm>.{md,json}` (e.g. `outputs/p2_v3_arm1r/VERDICT_arm1r.md`).
Required: the `verdict` string, the per-seed real table, the synthetic
table, the pre-registered clause table with bar/value/result, and the
mechanism block. The pre-registered bars and the protocol file they mirror
(`mathfix_round4.md`) are inputs; the verdict is the record that they were
met. Always light and synced.

### (f) The ledger -- LIGHT, human-authored

`docs/v3_results_record.md`, one running markdown, section-per-pass. It is
the human index tying artifacts to claims and to the paper. Never
machine-overwritten. Always synced.

Heavy vs light summary. Heavy (sync-optional, fetch on demand): the built
datacache npz, the traj/arrays npz, the raw EdNet tree. Light (always
synced): fold JSONs, analysis JSON/md, verdict JSON/md, caption and prov
sidecars, the ledger, PNG review copies. The design rule: everything
needed to reproduce a NUMBER or a TABLE is light; only the per-learner
arrays and raw scans are heavy.

---

## 4. Reproducibility metadata (the three-machine trust stamp)

For a build-on-box, pull-to-laptop, plot-anywhere workflow to be
trustworthy, every fold record (and, derived from it, every verdict/table/
figure) must be stampable back to the exact run. Stamp on every unit:

1. Code commit. The `git` SHA of `deep_irt/` at fit time (and dirty flag).
   The YAML-driven path already does this (`_p2_run_cell` writes a git SHA
   and `config_sha256` into `results.json`); it is the template. The v3
   scratch drivers (`_p2_realstudy*`, `_p2_nrm_repar_sweep`) do NOT stamp a
   commit into their fold JSONs; add it there. Without it, "which code
   produced this number" is unanswerable after a pull.
2. Head / arm identifier. Already present (`head` / `repar_arm`,
   `repar_epsilon`, `keyed`). Keep; this is what separates arm1 from arm1r
   from g0 cells in the same tree family.
3. Seeds. Already implicit in the filename (`d{seed}_f{fold}`) and in
   `data_seed`/`init_seed`. Make the FULL derivation explicit in the
   record: `SEED_BASE`, and the derived `split_seed = SEED_BASE + seed*100`
   and `init_seed` for real cells; `bed_seed`, `fold_split_seed = 2000 +
   seed`, `init_seed = seed` for synthetic. So the exact RNG streams are
   reconstructable from the artifact alone.
4. Environment fingerprint. Python, torch, numpy, CUDA versions; GPU model
   (RTX 4060 Laptop 8 GB on the current box); conda env name (`research`).
5. Platform + the CUDA_LAUNCH_BLOCKING footgun (author addendum). Stamp
   the OS/platform and whether `CUDA_LAUNCH_BLOCKING=1` was active. Ledger
   section 17 records a Windows-specific async CUDA scheduling stall where
   the routed transformer hung >90 min (historical pace 32s) with both
   autograd threads idle, resolved ONLY by `CUDA_LAUNCH_BLOCKING=1` (52s/
   unit, no code change). This flag changes timing, not numbers, but it is
   a run condition a future debugger needs recorded. Since the flag is
   environment-specific and the stall is Windows-specific, the stamp lets
   the remote-Linux runs be compared honestly against the box's runs.
6. Dataset-cache hash. The section-2.2 content hash of the built datacache
   the fit consumed (raw-source fingerprint + filter config + coverage).
   This closes the loop from a number back to the exact data it was fit on.

Implementation note. Items 1, 4, 5, 6 are new; 2, 3 exist and only need to
be made explicit. Emit them once per unit into the fold JSON (a
`provenance` sub-object is the clean place), and have verdict/table/figure
scripts propagate a digest so a downstream artifact names the commit and
cache-hash range of its inputs. None of this touches a computation.

---

## 5. Non-goals (must NOT change)

The refactor standardizes formats and locations. It must not touch:

1. Training recipes. `N_EPOCHS=120, LR=1e-2, BATCH_SIZE=512,
   HIDDEN_DIM=32, NRM_EPSILON=0.1, N_HOLDOUT=10`, the per-`(decoder,
   klass)` engine kwargs (`emb_dim`/`item_key_dim`/`nrm_channel`), the
   `patience=None` full-epoch training, and the head-attach seam
   (`attach_repar_arm1/_hard/_routed/g0/_shared_plane`). These define the
   fits on disk.
2. Seed conventions. Real: `split_seed = SEED_BASE + seed*100`,
   `init_seed = SEED_BASE + seed*1000 + fold + 1`, `SEED_BASE = 42`.
   Synthetic: bed `default_rng(seed)`, fold split `default_rng(2000 +
   seed)`, `init_seed = seed`. Byte-identical reproduction of shipped runs
   depends on these exactly.
3. Fold-split reproduction. `np.random.default_rng(...).permutation(N)`
   then `np.array_split(perm, N_FOLDS)` with `N_SEEDS = N_FOLDS = 5`. The
   split is part of a fit's identity.
4. Metric definitions. `prediction_metrics` (`metrics_bench` /
   `nrm_metrics`), Spearman for gauge-free params and Pearson for
   recovery-vs-truth, the truth-free two-stage `delta_slack`, the
   dataset-clustered bootstrap (10k, `default_rng(0)`, percentile), DOA,
   point-biserial, infit/outfit. Definitions and their random seeds are
   frozen.
5. Figure styling. `_paperfig_style` in full: the palette hex
   (`SHARED #E69F00`, `SEPARATE #0072B2`, `ORACLE`, `MML`, Okabe-Ito
   accents), the encoder marker grammar, the three page widths,
   `pdf.fonttype 42`, `REVIEW_DPI 150`, and the usetex-smoke-test-with-
   mathtext-fallback. Standardize WHERE figures land and add sidecars;
   never change how they look.
6. The exact-rescore contract (W7). `theta_track[t-1]` predicts `t`, the
   last-`min(H,Ln)`-valid window with `H=10`, the per-head probability
   recomputation, and the `< 1e-6` reproduction of on-disk `acc`. npz
   field names, dtypes, and `theta_track` semantics are frozen by this.
7. The analysis-lib indirection. Every analysis/figure script resolving
   paths through `_p2_v3_analysis_lib` globals is the mechanism that made
   tree-staging and re-pointing possible; preserve it (the refactor may
   change the global VALUES/locations, not the pattern of resolving
   through them).

---

## 6. What a Sonnet-tier coder must be told (do-not-break list)

Hand this list to any coder doing the mechanical migration:

- Never change a number. If a moved file makes any fold `acc`, recovery
  Spearman, or bootstrap CI differ from the ledger, you broke a
  computation. Revert.
- Preserve the write ORDER (npz first, authoritative fold JSON with
  `status=="done"` last via `os.replace`). Do not "fix" the npz to use a
  temp-rename; `np.savez` breaks that. Done-ness = both artifacts + status.
- `--skip-done` must stay exactly idempotent and read done-ness FROM THE
  ARTIFACT STORE, never from a heartbeat or stdout. A rerun of a full grid
  does zero work.
- Keep the unit selectable as one `(cell, seed, fold)` so the SLURM
  wrapper can map one array index to one unit; keep drivers Linux-runnable.
- Do not touch seed derivations, engine kwargs, training constants, metric
  code, or `_paperfig_style`. These are frozen (section 5).
- Do not hardcode `C:/Users/...`; resolve raw-data and repo roots from a
  single configurable source, and select the PYTHONPATH separator by
  platform.
- The 20-minute EdNet scan happens in exactly one place (`build-datacache`)
  and nowhere on the routine path; drivers consume the cache.
- Add provenance (commit, env, platform, CUDA_LAUNCH_BLOCKING, cache-hash)
  as NEW fields; do not remove or rename any existing field an analysis
  loader reads.
