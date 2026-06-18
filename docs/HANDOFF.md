# Project Handoff (START HERE)

Last updated 2026-06-17. This is the orientation doc for a fresh
conversation. Read this first, then the pointers at the bottom only as
needed. Everything here is committed locally; origin is out of sync after
the 2026-06-17 history rewrite (see Repo maintenance).

## What this project is

Three stacked pieces in one repo (`C:/Users/steph/documents/deep-mirt`).

1. **ma-irt** (`ma-irt/`), a deep ordinal IRT model. DKVMN/LSTM/
   Transformer encoder feeding a GPCM decoder, recovers theta, alpha,
   beta from response sequences. This is the paper under review at
   IJAIED and is slated for public release. Now a git submodule
   (github alhazar43/ma-irt). Frozen Chapter 0; do not edit it except
   additive configs.

2. **deep_irt** (`deep_irt/`), the ACTIVE framework. DeepIRTModel with
   swappable lstm/transformer/dkvmn encoders, decoupled-alpha default,
   PREDICTION-loss training (IRT as a readout flavor, no model-wise NLL).
   Holds the RQ1-3 learning-dynamics study (docs/LEARNING_DYNAMICS_STUDY.md,
   sections 3 and 4 for RQ1/RQ2 and RQ3) and the workshop deck
   (docs/slides/workshop.tex + workshop.pdf). 139 tests pass, 3 skipped.

3. **OrdRec** (`rl/`), an ExRec-style exercise-recommendation framework
   built on a custom PPO, parked at the D1 SLAM milestone. The Duolingo
   track. Modular, adapter-based.

`overleaf-sync/` is also a git submodule (the paper draft).

## Branch map

| Branch | What it holds |
|---|---|
| **`feat/prediction-loss`** | **THE CANONICAL LIVE WORKING BRANCH.** All current work: deep_irt learning-dynamics study + OrdRec (E1-E4.7 + D1 SLAM) + workshop slides. Work here. |
| `feat/ordrec` | OrdRec mainline pointer; fully contained in prediction-loss ancestry. |
| `main` | Public-facing ma-irt release line; far behind the research branches. |

The old per-milestone branches (feat/duolingo-mini, feat/ordrec-e1..e47,
feat/ordrec-d1-slam, feat/online-step-api, feat/v2-simulator-delta-j) and
all worktree-* refs were deleted in the 2026-06-17 cleanup; their history
is preserved in the backup bundle (see Repo maintenance).

## Status, what is done

**OrdRec is fully built, 228 tests pass.** Layers, data adapters
(E1-E2), env (E3), reward (E3), PPO library (E4), plus hardening
(E4.6a/b). ~11,400 lines under `rl/`.

**The headline scientific result (E4.7).** On static synthetic data
the learned policy LOSES to random (an honest null traced to theta
saturation). On DYNAMIC data (ability drifts within a session) the
ordering flips completely, PPO > BC > max-Fisher > random with
non-overlapping CIs on both staircase and random-walk cohorts, and the
VOI reward goes from never-positive to 100 percent positive in
training. The story, adaptive ordinal item selection pays precisely
when ability moves, and a learned policy beats greedy information
maximization. This is the OrdRec paper core.

**First real-data run (D1).** SlamAdapter on the public Duolingo SLAM
2018 en_es corpus (2,593 real learners, ~960k responses), K=3 ordinal,
zero ma-irt edits. ACC 0.682, QWK 0.374, binary-collapsed AUC 0.773.
Proves deep_irt runs end-to-end on real Duolingo data.

**The Duolingo collaboration angle is verified and time-sensitive.** A
101-agent adversarial verification confirmed the published Duolingo
calibration line (AutoIRT, BanditCAT) is dichotomous-only, and their
own June 2026 paper (S2A3, arXiv 2606.07364) names the polytomous
extension as roadmap future work. The pitch fills an author-
acknowledged, still-unpublished gap, but the window is closing, so the
pitch leads with ordinal PLUS longitudinal deep tracking (the part
their roadmap does not cover). Correction adopted, the operational DET
already ingests some polytomous grades under an undocumented model
class, so the claim is "published line is binary-only," never
"Duolingo is binary-only."

## Open decisions, the user must call these

1. **Eedi download (the only hard blocker).** The OrdRec headline
   confirmation (E5) on real knowledge-tracing data needs the user to
   download Eedi NeurIPS 2020 Task 3+4 csvs locally. Then it runs
   largely unattended. E5 is PAUSED, not abandoned, behind the
   Duolingo track.
2. **Paper structure.** Open. Recommendation, two papers, OrdRec RL to
   IJAIED and the ordinal-calibration SLAM result to BEA or EDM (where
   the Duolingo team publishes). Deferred until D4/E5 results.
3. **The mixed-K item-bank feature.** Open. The only proposed change
   to the public ma-irt repo (a per-category mask, 20-50 lines).
   Recommendation, HOLD, single-K adapters cover current work.
4. **Duolingo outreach.** Open. Recommendation, plan it, gate the cold
   email on D4 so it leads with two pieces of evidence and names the
   S2A3 authors.
5. **Origin re-add and force-push.** After the history rewrite, origin
   no longer matches local. Decision pending on whether to re-add the
   remote and force-push feat/prediction-loss, or leave origin stale.

Priority is LOCKED, the Duolingo / SLAM track is the active build
priority (user decision 2026-06-11).

## Immediate next step

**D2.** Add the SLAM es_en track plus LSTM and logistic-regression
baselines, tabulate AUC and log-loss, show the real-data result is
competitive. Then D3/D4 (synthetic mixed-format generator and recovery
experiment, the IJAIED scientific core). Branch off
`feat/prediction-loss`. SLAM config defaults already taken, K=3, es_en
next. The full D-milestone ladder is in
`docs/duolingo_mini_plan.md` Section 8.

## Operating conventions (carry these into the new conversation)

- **Minimum ma-irt edits.** ma-irt is now a submodule (frozen Chapter 0).
  Extend from `deep_irt/` or `rl/`, additive configs only, see memory
  `ordrec-ma-irt-boundary`.
- **Model economy.** Subagents on sonnet, trivial tasks on haiku,
  reserve the top model for the main loop and project-level decisions,
  see memory `model-economy`.
- **Writing style (strict).** No em-dashes or en-dashes, no colons in
  flowing prose, American English. Applies to all docs and paper text.
- **Staging discipline.** Never `git add -A`. Explicit paths only.
  Never stage `__pycache__`, `outputs/`, `*/data/`, or
  `ma-irt/_plot_encoder_recovery.py` (a persistent untracked stray,
  leave it alone).
- **Attribution.** Commits and PRs carry NO Co-Authored-By and NO
  Claude/Anthropic attribution. The author is the user only.
- **Env.** `conda activate research`, then set
  `PYTHONPATH=".;rl/src;ma-irt"` (Windows semicolon separator) and
  `KMP_DUPLICATE_LIB_OK=TRUE`. Tests: deep_irt suite via
  `python -m pytest deep_irt/tests/`; OrdRec suite via
  `python -m pytest rl/src/ordrec/ rl/tests/`. CUDA is an RTX 4060
  Laptop 8 GB. A full PPO synthetic run is ~90s; a world-model train
  ~4 min.
- **Workflow pattern.** Build in an isolated worktree, verify, document,
  commit with explicit staging, push the feature branch, then the main
  loop merges with `--no-ff` and re-runs tests. Worktrees must be
  cleaned (`git worktree remove --force`) after each.

## Repo maintenance (2026-06-17)

The 2026-06-17 cleanup made the following permanent changes.

- **Legacy tree removed.** The untracked `legacy/` directory (~5.5 GB,
  predecessor projects deep-gpcm/akt/pykt/kt-mirt/mirt-dkvmn and old
  experiment outputs) was deleted. Small paper artifacts inside legacy/
  were retained.
- **Git history rewritten.** `git filter-repo` purged dead predecessor
  trees (kt-gpcm, mirt-dkvmn, kt-mirt, figures, substrate,
  sn-article-template, archive) and the pre-submodule ma-irt file blobs.
  The `.git` pack went from 1.44 GiB to 15 MiB.
- **EdNet kept.** `EdNet-KT1/` (4.1 GB public dataset, used by
  `deep_irt/ednet_sep`) was NOT removed.
- **Backup.** Full bundle at
  `C:/Users/steph/documents/deep-mirt-backup-20260617-1136.bundle`
  plus a refs snapshot
  `deep-mirt-refs-20260617-1136.txt`. All deleted branch history is
  recoverable from the bundle.
- **Origin removed.** `filter-repo` removed the origin remote. Local
  history now diverges from origin (origin still holds old branches and
  the full old history). Do NOT `git fetch` the old origin; doing so
  re-bloats the pack. Re-adding origin and force-pushing
  `feat/prediction-loss` is a pending decision (open item 5 above).

## Pointers (read only if relevant)

- `docs/duolingo_mini_plan.md`, the active track plan, Section 0 is the
  decisions log, Section 8 the D-milestones.
- `docs/ordrec_progress.md`, the full milestone change log.
- `docs/exrec_ordinal_plan.md`, the OrdRec strategic plan.
- `docs/ordrec_impl_guide.md`, the file-level implementation contract.
- `rl/README.md`, the current rl/ tree map and how to run things.
- `rl/results/E47_dynamic_dgp.md`, the headline result writeup, plots
  under `rl/results/plots/e47_*`.
- `rl/results/D1_slam_en_es.md`, the first real-data run.
- `docs/LEARNING_DYNAMICS_STUDY.md`, the consolidated RQ1-3 learning-dynamics
  study (theory appendix `docs/learning_dynamics_toy.md`).
- `docs/slides/workshop.pdf`, the workshop deck.
