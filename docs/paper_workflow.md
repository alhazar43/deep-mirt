# Workflow: building "Not All Parameters Learn Alike"

**To my future self (a fresh conversation).** This is the execution workflow and state tracker for the
fused JEDM paper. Read `docs/paper_plan.md` first for the design; this file is how to build it and where
we are. Update the STATUS lines as you go. Nothing is written into the paper until the theory spine
(Phase 1) is on paper and the user has seen it.

## How to use this doc
1. Read `docs/paper_plan.md` (design) and skim `docs/slides/workshop.tex` (the deck, the current best
   statement of the science) and `overleaf-sync/main.tex` (MA-GPCM, the precursor to fold/trim).
2. Find the first phase whose STATUS is not DONE. Do it. Update STATUS.
3. Respect the constraints below. They are not optional.
4. The user switches conversations often; keep this file current so the next pick-up is cold-start clean.

## Current state (2026-07-01)
- Planning DONE. Three Opus advisors (constructive, adversarial, math-spine) and a two-part pivot-search
  (venue mechanics, exemplars/framing) have reported; their conclusions are baked into `paper_plan.md`.
- Decision DONE: fuse (MA-GPCM is a free draft, user controls it), organic single story, KT home / IRT
  flavor, target JEDM, provisional title = the workshop title.
- Nothing written into the paper yet. No experiments run yet for this paper.
- SCOPE TIGHTENED + PLAN APPROVED (2026-07-01): the paper is ONLY (a) a modular KT-encoder + IRT-decoder benchmarked by the recovery MATRIX across encoders x decoders, and (b) the finite-data law that the parameter which MULTIPLIES ability recovers worst (a scale-gauge multiplicative effect; NRM dissociates it from the low-Fisher rate penalty). See docs/paper_plan.md. Salvage the old paper's recovery benchmark; drop its ordinal-KT / separated-pathway / deployment framing. Never "neural KT".
- NEXT ACTION: Phase 4 REWRITE on the JEDM acmtrans class (two-column, ~12-15pp), deck-anchored, tight scope. Previous elsarticle draft SCRAPPED. Rigorous experiments (Phase 3, incl. the new multiplicative-vs-additive ablation E-mult) come after. Overleaf push blocked (403), preview from local builds. Phase 1 theory DONE (docs/theory_memo.md).

## Constraints (do not violate)
- **KT is home, IRT is a flavor.** Do not let the paper drift IRT-centric; that was MA-GPCM's mistake.
- **psychometric-researcher SCOPE:** verify that the IRT-flavor usage is *correct*. It must NOT add new
  IRT content or contributions. It validates, it does not expand.
- **Finite-data framing is the spine.** Always pair the recovery penalty with "vanishes asymptotically,
  bites at finite budget." Do not overclaim an asymptotic impossibility.
- **Report the mixed lever result honestly.** decoupling+static wins real-data reliability; do not spin
  the dynamic head as a clean win.
- **Writing style:** call things by exact names, no decorative or invented jargon (no "Fisher-starved",
  "doubly cursed", etc.), no em-dashes or en-dashes, no colons in flowing prose, American English,
  grant-then-qualify, claims sized to evidence. Match the workshop deck's register. NEVER use the term "neural KT" (banned), say knowledge tracing or a KT encoder with an IRT decoder. Scrub any agent
  draft for jargon before it lands.
- **Model economy:** Opus for thinking (theory, design, analysis, writing, review); Sonnet for mechanical
  (runs, file ops, git, Overleaf sync, literature fetches).
- **Runs:** foreground only, single 8 GB GPU, sequential. Scripts write full results to JSON; agents return
  short summaries (< 600 words) to avoid the 32k output crash. No detached/background training (it dies silently).
- **Git / files:** never `git add -A`; stage explicit paths; never stage `__pycache__`, `outputs/`,
  `*/data/`, `archive/`. No Claude/Anthropic attribution in commits or PRs; author is the user only.
- **Overleaf:** the deck submodule (`docs/slides`) and `overleaf-sync` push to Overleaf; never touch or
  commit an Overleaf token. `overleaf-sync` may be ahead (auto "Update on Overleaf" commits), merge, do not force.
- **Codex-owned, do not edit:** `deep_irt/core/*`, `deep_irt/bench/run_*.py`, `datagen.py`, `engines.py`,
  and existing `_ednet_ot*.py`. Extend via new `_`-prefixed scratch scripts in `deep_irt/bench/`.
- **Env:** `source ~/anaconda3/etc/profile.d/conda.sh && conda activate research &&
  export PYTHONPATH=".;rl/src;ma-irt" && export KMP_DUPLICATE_LIB_OK=TRUE` (Windows path sep is `;`).

## Phases

### Phase 1: Theory formalization  [STATUS: DONE (theory) / AWAITING USER APPROVAL -- 2026-07-01]
Reconciled memo at `docs/theory_memo.md`. No hard math errors; the spine holds. KEY REFINEMENT (both
passes): the two "decouplings" are distinct axes -- ability-item coupling = identifiability (structural,
persists), width sharing = representation/rate (finite, vanishes asymptotically); NRM adjudicates only the
representation one, E2 is the vehicle for identifiability. Other corrections folded into the memo: write
GPCM gradients in the honest binary+r_k form (not compressed scalar-r); control the NRM parameter-count
confound; frame adaptive testing as a joint (alpha,beta) rule with alpha the degraded channel. Do NOT
start Phase 2 or any writing until the user approves the spine.
Goal: the formal spine on paper, for the user to approve before any writing.
- *ml-math-researcher* (Opus): the Fisher-rate derivation for a generic readout parameter; the
  coupling/identifiability formalization (coupled vs decoupled readout, encoder-agnostic); the gauge
  argument; and the **asymptotic-vs-finite result** (prove the rate gap is a finite-budget effect and
  characterize its decay). A memory already exists at
  `~/.claude/agent-memory/ml-math-researcher/combined-paper-two-axis-spine.md`; build on it.
- *psychometric-researcher* (Opus): confirm the GPCM/NRM Fisher, recovery, reliability, and identifiability
  statements are psychometrically correct as used. SCOPE: validate the flavor, add no new IRT content.
- Output: a short theory memo (derivations + the precise claims + what is finite vs structural). I (main
  loop) reconcile and show the user.
- Done when: the user approves the spine.

### Phase 2: Framework and experiment infrastructure  [STATUS: DESIGN + SCAFFOLD DONE; PARKED for the later rigorous redesign -- 2026-07-01]
Blueprint at `docs/experiment_blueprint.md`. SCAFFOLD DONE: the _p2_ pipeline skeleton in deep_irt/bench/ (9 modules:
_p2_config [near-complete], _p2_run_cell, _p2_sweep, _p2_oracle, _p2_datagen_budget, _p2_coupled_theta, _p2_reliability,
_p2_cat, _p2_aggregate) + configs_p2/ templates, all TODO-stubbed, no runs. Fill-in order = blueprint build order 0-8.
Known fill-in blocker: beta_sigma is not yet a field in the Codex-owned datagen. REORDERING (user directive): write the
full STRUCTURED paper NOW using existing results (Phase 4, in progress), then do the code-heavy fill-in + GPU runs LATER
(needs ultracode) and swap the rigorous results in. Open config decisions deferred: E2(a) vehicle; cell size vs compute.
Goal: one clean pipeline to run every experiment consistently.
- *research-scientist* (Opus, design): spec the swappable framework and the experiment designs (E1, E2,
  E-budget, E-levers, E8) in KT terms; confirm which already exist in `deep_irt/`.
- *ml-system-architect* (Opus design, Sonnet build): a unified train/eval/recovery pipeline for the
  encoder x decoder matrix; the budget sweep harness; the adaptive-item-selection simulator; the
  consolidation harness. Foreground, JSON out, 8 GB GPU aware.
- Done when: the pipeline runs one matrix cell end to end and writes recovery JSON.

### Phase 3: Experiments  [STATUS: NOT STARTED]
Goal: the evidence. Run and interpret. See the checklist below.
- *ml-system-architect* / *research-scientist* run (Sonnet) and interpret (Opus). Foreground, JSON out,
  short summaries back.
- Done when: every checklist item is DONE or explicitly deferred with a logged reason.

### Phase 4: Writing  [STATUS: RESTART on JEDM class -- 2026-07-01]
RESTART (plan approved 2026-07-01): the previous elsarticle draft glued MA-GPCM in and is SCRAPPED. Rewrite on the JEDM
acmtrans class (two-column, ~12-15pp), TIGHT scope per docs/paper_plan.md: only (a) the modular recovery-matrix and (b) the
finite-data MULTIPLICATIVE-coupling law (the trade-off is the scale-gauge coefficient*ability effect; NRM dissociates it from
the low-Fisher rate penalty; add the multiplicative-vs-additive ablation E-mult). Deck-anchored, no "neural KT". Salvage the
old paper's recovery benchmark for the matrix; drop its ordinal-KT / separated-pathway / deployment framing. The notes below
describe the SCRAPPED elsarticle draft, retained only for the record:
Full structured combined draft written to overleaf-sync/main.tex (compiles, ~40pp elsarticle review mode / ~13pp
two-column), scrub-clean, KT-home/IRT-flavor, full prose. MA-GPCM archived as overleaf-sync/main_magpcm_ijaied.tex
(do NOT discard). Committed in overleaf-sync LOCALLY (commit e2f0088). Each experiment carries a one-line strengthening
note pointing to the rigorous rebuild (Phase 2 scaffold + Phase 3). 10 bibitems added (5 marked [VERIFY]: ma_neural_2024,
vtirt_2023, wide_deep_irt_2024, beta4irt_2023, autoirt_2024).
BLOCKER (Overleaf auth, account/service-wide): ALL Overleaf git access now 403s, including the DECK project that pushed
fine earlier THIS session, plus the user's NEW project 6a4518736a3fce49c9041e19, with both the cached credential AND the
user-provided token (olp_, used correctly as username=git / password=token). Since a previously-working project now 403s
too, it is NOT the token or the project. Most likely Overleaf RATE-LIMITING from several auth attempts in quick succession,
or the account's git integration (premium) lapsed. STOPPED further attempts to avoid deepening a rate-limit. Submodule
origin now points at the new project 6a45... (tokenless) for the retry. Paper is SAFE: local commit e2f0088 + GitHub backup
docs/paper_not_all_params_draft.tex. RETRY after a ~20-30 min cooldown: the new project has UNRELATED history so it needs a
FORCE push, `git -C overleaf-sync push --force <token-URL> master`; if it still 403s, it is account-side (check Overleaf
premium / whether the project shows a Git menu) or push from a machine with a live Overleaf session.
NOTE: this PRECEDES the rigorous experiments (Phase 3), by user directive; the rigorous results swap in later.
Goal: the fused draft in `overleaf-sync`, KT home, IRT flavor.
- I (main loop): framing, thesis, claim lines, jargon scrub, KT-centric voice.
- *research-scientist* (Opus): framework and experiment sections in the DL-KT voice.
- *psychometric-researcher* (Opus): draft and vet the decoder-math methodology (flavor-correct, not center).
- *ml-math-researcher* (Opus): draft and vet the theory section.
- Done when: draft compiles, scrub-clean, and every claim is backed by a Phase 3 result or a marked placeholder.

### Phase 5: Adversarial review  [STATUS: NOT STARTED]
Goal: survive the reviewer before submission.
- *research-scientist* adversarial pass + *ml-math* check. Fix or concede each finding.
- Done when: no unaddressed blocking finding.

## Agent roster (type / tier / job)
- ml-math-researcher / Opus / the rate + coupling + asymptotic theory.
- psychometric-researcher / Opus / validate IRT-flavor correctness ONLY (no new IRT content); downstream item-selection math.
- research-scientist / Opus design, Sonnet runs / framework, experiment design, KT framing, related work, drafting, adversarial pass.
- ml-system-architect / Opus design, Sonnet build / the unified pipeline, budget sweep, CAT simulator.
- general-purpose / Sonnet / git, Overleaf sync, file ops, literature fetches.

## Where things live
- The paper (to build): `overleaf-sync/main.tex` (currently MA-GPCM; fold + trim into the fused draft).
- The science, current best statement: `docs/slides/workshop.tex` (the deck).
- The design + this workflow: `docs/paper_plan.md`, `docs/paper_workflow.md`.
- Existing results to consolidate: `deep_irt/bench/outputs/_nrm_*.json` (NRM dissociation, EdNet),
  and the learning-dynamics study docs `docs/LEARNING_DYNAMICS_STUDY.md`, `docs/learning_dynamics_*`.
- Framework code: `deep_irt/` (swappable lstm/transformer/dkvmn encoders, gpcm/binary/nrm/bt decoders).
- MA-GPCM code (frozen, reference only): `ma-irt/`.

## Experiment checklist
- [ ] E1  swappability matrix {DKVMN, LSTM, (transformer)} x {GPCM, NRM, binary} recovery, synthetic
- [ ] E2  coupling lever (coupled vs decoupled ability readout) in DKVMN and LSTM = "why separate"; alpha vs beta
- [ ] E-budget  recovery gap vs data size and training budget (asymptotic vanishing)
- [ ] E-mult  MULTIPLICATIVE-vs-additive ablation (hold Fisher fixed, vary multiplicative vs additive entry with ability; the trade-off should track multiplicative entry, not information) -- CORE, makes the multiplicative claim necessary
- [ ] E-levers  decoupling+static vs decoupling+dynamic, synthetic recovery and real reliability, NRM as control
- [ ] E8  adaptive item selection by max Fisher info at the running ability estimate, vs oracle
- [ ] Consolidation  re-run key panels under one training protocol
- [ ] E9 (optional)  classical MML-EM cross-calibration on EdNet/KDD

## Next action
Start Phase 1: launch *ml-math-researcher* and *psychometric-researcher* in parallel (both Opus) on the
theory spine and the flavor-correctness check. Reconcile their outputs into a short theory memo and show
the user before any writing.
