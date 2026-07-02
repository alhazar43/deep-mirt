# Project Handoff (START HERE)

Last updated 2026-07-02. Orientation for a fresh conversation. The active work is the JEDM
paper **"Not All Parameters Learn Alike"** (a knowledge-tracing encoder plus an interpretable
IRT decoder). Honest state: experiments are **partially done**, the paper needs a **major
rework**, and the current writeups are below standard (some material is salvageable). Do not
treat any existing draft as final.

Repo root `C:/Users/steph/documents/deep-mirt`. Canonical branch `feat/prediction-loss`
(origin github.com/alhazar43/deep-mirt).

## 1. The paper (what it argues)
Home is knowledge tracing; IRT is the readable flavor. Never "neural KT" (banned), never a
psychometrics-theory contribution. Two things only:
- **(a) A modular tracer, benchmarked by recovery.** A swappable sequence encoder feeding a
  swappable IRT decoder; the contribution is the parameter-recovery benchmark across the
  encoder-by-decoder matrix (modularity is real, and the ordering is not one encoder's artifact).
- **(b) A finite-data law.** Trained on prediction loss alone, the parameter that MULTIPLIES
  ability recovers worst (a multiplicative scale-gauge effect), separable from a low-Fisher rate
  penalty. Two mechanisms, two fixes: DECOUPLING (the multiplicative parameter gets its own item
  key) fixes the representation / scale-gauge effect; the DYNAMIC (state_t in the readout) head
  fixes the low-Fisher rate penalty. NRM is the control that dissociates them (its slope a_k is
  multiplicative but NOT low-Fisher).

Full design `docs/paper_plan.md`; derivations + reviewer caveats `docs/theory_memo.md`; run
plan `docs/experiment_blueprint.md`; tracker `docs/paper_workflow.md`. Terminology is exact and
load-bearing: 2PL/GPCM have DISCRIMINATION (alpha) and DIFFICULTY / step thresholds (beta); NRM
has SLOPE (a_k) and INTERCEPT (c_k). Never "slope" for GPCM, never "difficulty" for NRM.
"Decoupling" = own item key for the hard parameter; "dynamic" = whether state_t enters the
readout. There is NO "coupled theta" and no "two decoupling axes" framing (both purged).

## 2. The framework + the experiment pipeline
`DeepIRTModel` (`deep_irt/core/model.py`): swappable encoder (lstm default, transformer, dkvmn)
+ swappable IRT decoder (gpcm, binary=2PL, nrm, bt), trained on a PREDICTION loss (no IRT-NLL;
GPCM uses ordinal cross-entropy, not WOL, not categorical CE). IRT params are recovered after
training from frozen decoder weights.

The paper's experiments run through the `_p2_` pipeline in `deep_irt/bench/` (all `_`-prefixed
scratch that extends deep_irt additively): `_p2_datagen_realistic.py` (the realistic bed),
`_p2_ordinal_ce.py`, `_p2_model.py`, `_p2_engine.py`, `_p2_run_cell.py`, `_p2_gpcm_alpha_key.py`
(Option-A discrimination-on-item-key), `_p2_nrm_channels.py` (the 5 NRM couplings),
`_p2_reliability.py` + `_p2_real.py` (real-data split-half), driven by
`deep_irt/bench/configs_p2/*.yaml` via `_p2_sweep.py`. **Codex owns and you must NOT edit**
`deep_irt/core/*`, `bench/run_*.py`, `datagen.py`, `engines.py`, and existing `_ednet_ot*.py`;
extend only in new `_`-prefixed files.

## 3. State of the experiments (partially done)
Raw record, config in and numbers out, no interpretation: **`docs/experiment_results.md`**.

DONE (overnight run 2026-07-01), realistic ma-irt bed (Q=200, N=2000, administration
Uniform(40,80), uniform exposure, 150 epochs, 5 data seeds x 5 folds):
- Benchmark, 11 cells: {lstm, dkvmn, transformer} x {2PL, GPCM, NRM} + a dense control + a
  random-walk drift arm.
- Toggles, 18 cells: 2PL/GPCM {shared, decoupled} x {static, dynamic} (8); all 10 NRM couplings
  x {static, dynamic}.
- Real-data reliability, 10 cells: EdNet + KDD x {2PL, GPCM, NRM}, shared vs fix, split-half
  Spearman-Brown, accuracy-guarded.
- Fix applied: NRM ability (theta) had a GLOBAL-SIGN scoring bug, sign-aligned in
  `_p2_run_cell` (item params were always correct; re-scored from saved folds, no re-train).

NOT DONE (the load-bearing controls from the blueprint, needed before the paper is defensible):
- The **multiplicative-vs-additive ablation** (hold Fisher fixed, vary only multiplicative vs
  additive entry). This turns "multiplicative" from best-explanation into necessary and is the
  single most important missing experiment.
- The finite-vs-asymptotic budget sweep (the gap should vanish with data and training budget).
- The oracle-clamp control (theta = theta*) for the co-learned-theta gap.
- The a_star / eigenmode-inversion check (below discrimination ~1 the recovery order inverts).
- ASSISTments cells (raw data absent on disk; only EdNet + KDD ran).

## 4. State of the writing (needs a major rework)
None of the current writeups meet the standard; treat them as raw material, not drafts.
- `overleaf-sync/main.tex` (acmart sigconf, a local stand-in for JEDM's acmtrans; the "Not All
  Parameters" draft) is agent-written and below standard; rework on the real results.
- `overleaf-sync/main_magpcm_ijaied.tex` is the archived MA-GPCM (IJAIED) paper. SALVAGE its
  recovery-benchmark methodology, metrics, and architecture diagram for scope (a); it is also
  the prose REGISTER exemplar. Do not discard.
- `docs/slides/workshop.tex` (XeLaTeX, Twente SimplePlusAIC theme) is the deck the paper grew
  from; salvage its structure, not its stale headline.
- Overleaf push is BLOCKED (403, account-wide; see docs/paper_workflow.md); build locally.
  GitHub backup of the draft source: `docs/paper_not_all_params_draft.tex`.

## 5. Immediate next steps
1. Run the missing blueprint controls, ablation first (it is what makes multiplicativity
   necessary), then the budget sweep, oracle-clamp, a_star.
2. Major rework of the paper prose on the real results, deck-anchored, tight (a)+(b), register
   matched to `main_magpcm_ijaied.tex`.
3. Resolve Overleaf (403) or keep local; fetch ASSISTments if its cells are wanted.

## 6. Operating conventions (carry over)
- **Env.** `source ~/anaconda3/etc/profile.d/conda.sh && conda activate research`, then
  `export PYTHONPATH=".;rl/src;ma-irt"` (Windows `;`) and `export KMP_DUPLICATE_LIB_OK=TRUE`.
  Tests `python -m pytest deep_irt/tests/`. CUDA is one RTX 4060 Laptop 8 GB (runs sequential).
- **Codex boundary.** Do NOT edit `deep_irt/core/*`, `bench/run_*.py`, `datagen.py`,
  `engines.py`, existing `_ednet_ot*.py`. Extend in new `_`-prefixed gitignored scratch.
- **Execution discipline.** Long runs go in a harness-tracked background job that writes full
  results to JSON; agents return SHORT summaries (<600 words), never per-cell log dumps (they
  crash on the 32k output limit). Single GPU, sequential.
- **Model economy.** Top model for the main loop, planning, verification; sonnet for mechanical
  work, haiku for trivial. Decompose independent work and run it in parallel.
- **Writing.** No em- or en-dashes, no colons in flowing prose, American English. Exact,
  established names, never invented labels (memory use-established-names). Match the register of
  `main_magpcm_ijaied.tex` + memory writing-style. Slides: noun-phrase titles reused as bold
  summary leads, terse bullets, grant-then-qualify.
- **Staging.** Never `git add -A`; explicit paths only. Never stage `__pycache__`, `outputs/`,
  `*/data/`, `archive/`. No Co-Authored-By / Claude attribution; author = user.
- **PSI-KT is AGPL** (relevant to the parked transfer thread): reference its design, never
  vendor its code.

## Parked / separate tracks (one-liners, do not start here)
- **Q-MIRT learning-via-transfer paper** (Thread A): a dynamic MA-GPCM that shows learning via
  cross-concept transfer, fixed-measurement / moving-state. Direction and existence achieved,
  magnitude gauge-bound; D-scaling and real KDD open. Memory [[qmirt-learning-transfer-paper]],
  full log `docs/overnight_transfer_active_campaign.md`, scratch `deep_irt/bench/_qmirt_*.py`.
- **ma-irt** (`ma-irt/`): frozen Chapter-0 deep ordinal IRT, IJAIED. Submodule, additive only.
- **OrdRec** (`rl/`): parked ExRec-style ordinal item recommendation.

## Pointers
- Paper: `docs/paper_plan.md`, `docs/theory_memo.md`, `docs/experiment_blueprint.md`,
  `docs/experiment_results.md`, `docs/paper_workflow.md`.
- Framework API: `deep_irt/README.md`.
- SUPERSEDED, do not treat as current: `docs/paper2.tex` / `.pdf`,
  `docs/LEARNING_DYNAMICS_STUDY.md`, `docs/learning_dynamics_*.md` (predate the current paper and
  the overnight results; salvageable material only).
