# deep-irt portable-repo refactor — master plan

Mastermind synthesis over the two commissioned halves:
`refactor_plan_systems.md` (layout, schemas, sync, migration mechanics) and
`refactor_plan_workflows.md` (workflow contracts, artifact taxonomy,
guardrails). Those documents carry the detail; this one carries the
decisions, phases, gates, and task assignments. Status: APPROVED WITH
RULINGS (2026-07-13, section 6) -- execution mode is COPY-THEN-SWAP: build
`deep-irt-port/` alongside the live tree, pass multi-stage local + remote
tests, then retire `deep_irt/` and rename the port into place. The live
tree is never touched during the build.

## 1. Objective and hard constraints

Make `deep_irt` a standalone, portable repository with pipelined I/O, so
that training can run on any of three machines (Windows PC, work laptop,
SLURM remote) and results synchronize such that every figure and table is
regenerable anywhere without retraining.

Constraints, in force throughout:
- C1 No core-function changes: model behavior, metric computation, and
  plot rendering stay byte-identical. Relocation and standardization only.
- C2 Hierarchy separation: code / scripts / configs / data / results /
  figures / checkpoints / docs / tests.
- C3 (amended) Codex is deprecated, but the codebase stays AGENT-NEUTRAL:
  a standalone repo any agent may edit. Editing for function is allowed;
  "modifying for an agent's own convenience" is not -- copies stay
  byte-identical except where the plan requires an import-path rewrite.
- C4 Linux/SLURM compatibility for all pipeline entrypoints; the sbatch
  leg is a stub until the author supplies remote specifics.
- C5 R/MML is a PC-only capability: extracted, config-pathed, outputs
  synced as artifacts; never required on the remote.
- C6 Model economy: Fable = plan/review gates; Opus = complex legs;
  Sonnet = mechanical legs.

## 2. Target architecture (decisions)

```
deep-irt/                      (new standalone git repo; submodule of deep-mirt)
  pyproject.toml               pip install -e .  -> retires PYTHONPATH hacks
  src/deep_irt/
    core/                      vendored Codex files, byte-identical, CODEOWNERS
    bench/                     tracked stable p2 core (vendored as-is)
    pipeline/
      train/                   promoted drivers (realstudy, sweep) + unit addressing
      score/                   gate scorers, rescore machinery
      analyze/                 analysis lib (single RESULTS_ROOT choke point) + suites
      figures/                 figure builders + _paperfig_style (unchanged rendering)
      calibrate/               extracted R/MML stage (PC-only capability)
  scripts/                     thin CLIs: train_unit --index, enumerate_units,
                               score, analyze, make_figures, calibrate_mml,
                               build_datacache; slurm/ (stub)
  configs/                     cell grids as tracked configs; machines/{pc,laptop,slurm}.yaml
  data/                        gitignored; staged raw + versioned datacache (DVC)
  results/                     the shared store; existing p2_* tree names verbatim
  figures/                     regenerated outputs (png+pdf+caption+prov sidecars)
  checkpoints/                 (currently unused by the campaign; reserved)
  docs/  tests/
```

Key decisions (rationale in the systems doc):
- D1 (amended) Repo boundary: built as `deep-irt-port/` inside deep-mirt,
  its own local git repo with NO remote assumption (could become GitHub,
  HuggingFace, GitLab -- author decides later). Parent submodule wiring
  happens only at adoption time.
- D2 (amended) Migration is COPY, not move: the live `deep_irt/` keeps
  working untouched until the port passes all tests and the author swaps
  it in. Copies are byte-identical except mechanical import-path rewrites.
- D3 (amended) No DVC. The author manages heavy-data syncing themselves.
  The port's git tracks the light set only; heavy artifacts (npz, raw
  data, datacache) are gitignored with a documented layout. RESULTS
  CURATION RULE: only artifacts the CAEAI paper uses are kept in
  `results/`; everything else is archived locally (gitignored attic) as
  reference.
- D4 Unit of work is the portable primitive: one (cell, seed, fold) fit,
  atomic npz-then-JSON writes, store-derived done-ness; maps 1:1 onto a
  SLURM array index. Progress is derivable from the store alone; the
  heartbeat/REMAINING_UNITS chain-script pattern retires.
- D5 Provenance: per-unit provenance block (code sha, env, platform,
  CUDA_LAUNCH_BLOCKING flag, datacache hash) as NEW fields; per-sweep
  manifest; per-figure {stem}.prov.json + caption sidecar. Historical
  artifacts are backfilled with provenance="historical".
- D6 Frozen contracts (from the workflows doc): fold-record schema, arrays
  npz keys/dtypes, traj schema + sidecar, exact-rescore semantics
  (theta_track[t-1], last-10-valid window, reproduces acc to <1e-6),
  seed derivations, engine kwargs, metric code, figure styling.

## 3. Phases, gates, and assignments

Phase 0 — Freeze and baseline (Sonnet, S)
  Hash inventory of outputs/ + overleaf figures (byte-gate baseline); tag
  the parent repo; extract the promotion candidate list from ledger
  sections 14-19. Gate: baseline hashes recorded and committed.

Phase 1 — Skeleton and packaging (Sonnet, S)
  New repo, target tree, pyproject + editable install, machine profiles
  (configs/machines/*.yaml with capabilities.r and rscript_path), README
  stub. Gate: `pip install -e .` and `pytest deep_irt/tests` (moved) pass
  on Windows.

Phase 2 — Vendor the code (Opus lead, Sonnet mechanical, M)
  `git mv` core + tracked bench into src/deep_irt/; absolute-import
  verification; CODEOWNERS. REQUIRES Codex quiescence (ruling 6.1).
  Gate: full test suite + one tiny-cell smoke fit produce the frozen
  fold-record schema; no PYTHONPATH anywhere.

Phase 3 — Promote the campaign scratch (Opus curation, M)
  The 98 untracked `_p2_*` files through the 3-check clearance (docs-cite /
  imported-by-kept / regenerates-committed-figure); promote into
  pipeline/{train,score,analyze,figures}/ with underscore-free names;
  attic the remainder in the parent repo (never delete). R/MML extraction
  into pipeline/calibrate with profile-resolved Rscript path. Gates: every
  artifact in ledger sections 14-19 names a promoted regenerator; the
  PC-resolved Rscript command line is character-identical to today's.

Phase 4 — Relocate results + manifests (Sonnet, M)
  Move outputs/ trees into results/ verbatim; repoint the single
  RESULTS_ROOT choke point; backfill sweep manifests. MASTER GATE:
  byte-identical regeneration of the golden set from relocated artifacts —
  tab_real_metrics + the shsk analysis JSONs; fig_dd, fig_agreement_both,
  fig_reversal_bridge, fig_ednet_2in1 — against Phase-0 hashes.

Phase 5 — Entrypoints and unit addressing (Opus, M)
  scripts/train_unit.py --index + enumerate_units; score/analyze/figure
  CLIs wrapping existing functions unchanged; store-derived remaining();
  unify done-detection across drivers (the failed-JSON-counts-as-done
  divergence — the ONE approved behavior unification, ruling 6.5; changes
  no numbers). SLURM stub committed. Gate: tiny-cell local train/eval
  smoke: schema-identical records, idempotent --skip-done, rescore
  reproduces on-disk acc.

Phase 6 — Sync fabric (Sonnet, M)
  DVC init + rclone remote (ruling 6.2); track heavy set; laptop dry-run:
  clone, pull light set, regenerate golden figures. Gate: second-machine
  regeneration passes — numeric artifacts byte-identical; PDFs
  byte-identical only if matplotlib is pinned + SOURCE_DATE_EPOCH, else
  pixel-tolerance (honesty note from the systems doc).

Phase 7 — Parent integration and the remote leg (gated)
  Submodule wiring in deep-mirt; runbook; ledger record. SLURM leg
  activates only when the author supplies remote instructions (their
  stated sequencing: local train/eval tests pass first).

Rough effort: phases 0-4 one focused working day of agent execution with
review gates; 5-6 a second day; 7 gated on the author.

## 4. Verification philosophy

Relocation changes zero bytes by construction; every phase gate is a
regeneration-and-compare against the Phase-0 hash baseline, cheap because
all analysis paths resolve through one repointable base. Retraining is
never part of a gate; the only fits in the plan are tiny-cell smokes.

## 5. Risks

- R1 Codex committing during the vendor move -> schedule quiescence or
  take the flat-layout fallback.
- R2 Promotion misclassification -> default-promote, 3-check clearance,
  attic-never-delete.
- R3 DVC overhead on ~29k small files on Windows -> hardlink cache,
  subtree tracking, prune superseded trees from the hot remote
  (p2_exposure 641 MB etc. stay archived locally); syncthing fallback.
- R4 Cross-machine PDF nondeterminism -> pin matplotlib + usetex inputs or
  accept numeric-identity + pixel tolerance as the gate.

## 6. Rulings required from the author before execution

1. Codex quiescence: is Codex still actively committing? If yes, name a
   pause window for Phase 2, or choose the flat-layout fallback.
2. Sync remote: Google Drive / OneDrive / other (rclone-compatible), for
   the DVC heavy set.
3. New repo: name and visibility (assumed github.com/alhazar43/deep-irt,
   private).
4. Superseded heavy trees (p2_exposure et al., ~1.3 GB): confirm
   local-archive-only, excluded from the hot remote.
5. Approve the single behavior unification: sweep-driver done-detection
   gains the status=="done" check (bug fix, no numeric change).

## 7. Rulings received (2026-07-13) -- all resolved

1. Codex deprecated; codebase stays agent-neutral; no gratuitous rewrites.
2. Author manages dataset/heavy sync; heavy gitignored; light tracked;
   results/ keeps CAEAI-paper artifacts ONLY, rest archived locally.
3. No remote assumption for the new repo; author picks later.
4. Superseded/deprecated trees archived + ignored as local reference.
5. Done/failed semantics: done means the computation completed
   (status=="done" + required artifacts); failures of any kind are
   incomplete and never count as done.
EXECUTION AMENDMENT: build in `deep-irt-port/`, multi-stage tests (local
now, remote when the author supplies instructions), then retire
`deep_irt/` and rename the port to replace it.
