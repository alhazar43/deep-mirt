# Project Handoff (START HERE)

Last updated 2026-07-17. Read CLAUDE.md (working rules) and the memory
index first; this file is the state pointer, kept short on purpose.

Repo root `C:/Users/steph/documents/deep-mirt`, branch
`feat/prediction-loss` (origin github.com/alhazar43/deep-mirt).

## State in one paragraph

The measurement-audit paper (CAEAI-first; repo
github.com/alhazar43/JEDM-paper, name historical; draft
`overleaf-sync/main_caeai.tex`, plan of record `docs/paper_plan_v2.md`,
results ledger `docs/v3_results_record.md` through sec 22) has a COMPLETE,
frozen experimental campaign. The framework lives in the standalone
`kt-irt/` submodule (github.com/alhazar43/kt-irt), which replaced the
retired in-tree `deep_irt/` and is verified on all three machines (PC,
laptop-ready, UT HPC SLURM). The 2026-07-16 extraction closed the
campaign apparatus: the gradient-routed head (arm1r) is the one NRM head,
SH/SK (shared head / separated key) is the one live toggle, and the paper
replicates FROM SCRATCH -- runbook `kt-irt/docs/REPLICATION.md`, closure
record `kt-irt/docs/port/extraction_report.md`. Every change was gated
(pytest 139/1; byte-identical figure regeneration; zero-delta refits;
cluster bit-reproducibility 25/25).

## Working with kt-irt (the base for new work)

- Install: `pip install -e kt-irt` (imports stay `deep_irt.*`); tests
  `python -m pytest` from `kt-irt/`.
- Entry points: `deep-irt-train-unit / -enumerate / -status / -figures /
  -weights-manifest`; batch layers `kt-irt/local/train_batch.ps1`
  (Windows) and `kt-irt/slurm/autopilot.sh` (UT HPC; cluster facts in
  memory `ut-hpc-cluster`).
- Contracts: results are small (fold/verdict JSON, slim traj npz; panel
  arrays resolve from `data/cache/`); weights two-tier (`weights/*.pt`
  manifest-tracked, `checkpoints/*.pth` local debug); every
  byte-diverging edit logged in `kt-irt/docs/port/copy_edits.json`;
  `results/` artifacts and `docs/port/` records are frozen history.
- Overleaf's own git endpoint is unreachable from this environment; the
  paper syncs through the JEDM-paper GitHub repo.

## Parked lines

`ma-irt/` frozen Chapter 0 (IJAIED); `rl/` OrdRec; Q-MIRT transfer paper
(memory `qmirt-learning-transfer-paper`); thesis north star in
`docs/Thesis_overview.md` + memory `thesis-vision`.
