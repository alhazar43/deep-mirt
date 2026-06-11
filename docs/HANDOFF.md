# OrdRec / DuoLingo Mini, Session Handoff (START HERE)

Last updated 2026-06-11. This is the orientation doc for a fresh
conversation. Read this first, then the pointers at the bottom only as
needed. Everything here is committed and pushed.

## What this project is

Two stacked pieces in one repo (`C:/Users/steph/documents/deep-mirt`).

1. **ma-irt** (`ma-irt/`), a deep ordinal IRT model. DKVMN/LSTM/
   Transformer encoder feeding a GPCM decoder, recovers theta, alpha,
   beta from response sequences. This is the paper under review at
   IJAIED and is slated for public release. It stays AS-IS. Do not
   edit it except additive configs, unless a measured major bottleneck
   forces it (one such audit found none, see pointers).

2. **OrdRec** (`rl/`), a new ExRec-style exercise-recommendation
   framework built on top of frozen ma-irt with a custom PPO. The
   active research. Modular, adapter-based, isolated from the archived
   job-recommendation work at `archive/rl_jobrec/`.

## Branch map

| Branch | Tip | What it holds |
|---|---|---|
| **`feat/duolingo-mini`** | **`bb13a93`** | **THE LIVE WORKING BRANCH.** Full OrdRec code (E1-E4.7) + D1 SLAM adapter + the DuoLingo plan + all results. Work here. |
| `feat/ordrec` | `e960be9` | OrdRec mainline, E1 through E4.7 plus D1. One commit behind the live branch (lacks the plan docs and the decision merge). |
| `feat/ordrec-e1..e47`, `-d1-slam` | various | Per-milestone branches, all merged, retained for history. Safe to ignore or delete. |
| `feat/online-step-api`, `feat/v2-simulator-delta-j` | dormant | Abandoned jobrec-era branches. Ignore. |

`main` is far behind and is the public-facing ma-irt release line; the
research lives on the feat branches.

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
Proves ma-irt runs end-to-end on real Duolingo data.

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

Priority is LOCKED, the Duolingo / SLAM track is the active build
priority (user decision 2026-06-11).

## Immediate next step

**D2.** Add the SLAM es_en track plus LSTM and logistic-regression
baselines, tabulate AUC and log-loss, show the real-data result is
competitive. Then D3/D4 (synthetic mixed-format generator and recovery
experiment, the IJAIED scientific core). Branch off
`feat/duolingo-mini`. SLAM config defaults already taken, K=3, es_en
next. The full D-milestone ladder is in
`docs/duolingo_mini_plan.md` Section 8.

## Operating conventions (carry these into the new conversation)

- **Minimum ma-irt edits.** ma-irt is the public paper repo. Extend
  from `rl/`, additive configs only, see memory `ordrec-ma-irt-boundary`.
- **Model economy.** Subagents on sonnet, trivial tasks on haiku,
  reserve the top model for the main loop and project-level decisions,
  see memory `model-economy`.
- **Writing style (strict).** No em-dashes or en-dashes, no colons in
  flowing prose, American English. Applies to all docs and paper text.
- **Staging discipline.** Never `git add -A`. Explicit paths only.
  Never stage `__pycache__`, `outputs/`, `rl/data/`, `ma-irt/data/`,
  `archive/`, or `ma-irt/_plot_encoder_recovery.py` (a persistent
  untracked stray, leave it alone).
- **Env.** `conda activate research`, then
  `PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE python ...`.
  Tests, `pytest rl/src/ordrec/ rl/tests/`. CUDA is an RTX 4060 Laptop
  8 GB. A full PPO synthetic run is ~90s; a world-model train ~4 min.
- **Workflow pattern.** Milestones were built by background workflows,
  build in an isolated worktree, verify, document, commit with explicit
  staging and `Co-Authored-By`, push the feature branch, then the main
  loop merges with `--no-ff` and re-runs tests. Worktrees must be
  cleaned (`git worktree remove --force`) after each.

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
- `docs/cleanup/_det_deep_research_report.md`, the verified Duolingo
  intelligence (gitignored working doc, local only).
- `docs/cleanup/_ma_irt_bottleneck_audit.md`, the "keep ma-irt as-is"
  audit (gitignored, local only).
- `archive/rl_jobrec/README_ARCHIVE.md` and
  `docs/archive/jobrec/README.md`, the abandoned job-rec direction and
  why it was dropped.
