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
- NEXT ACTION: Phase 2 design (rigorous protocol + pipeline) IN PROGRESS. DIRECTIVE (2026-07-01): redo the entire experiment from scratch at journal rigor, do NOT reuse workshop results as evidence. BUILD+RUN needs ultracode. Phase 1 theory DONE (memo at docs/theory_memo.md).

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
  grant-then-qualify, claims sized to evidence. Match `overleaf-sync/main.tex` register. Scrub any agent
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

### Phase 2: Framework and experiment infrastructure  [STATUS: DESIGN DONE; SCAFFOLD IN PROGRESS -- 2026-07-01]
DIRECTIVE: redo the ENTIRE experiment from scratch at journal rigor; do NOT reuse workshop results as evidence.
DESIGN DONE: reconciled blueprint at `docs/experiment_blueprint.md` (one shared protocol; experiment->config
mapping; build order; reproducibility; honest compute). USER DIRECTIVE: write the design + SCAFFOLD now, defer the
code-heavy fill-in and the GPU runs. SCAFFOLD (in progress): the _p2_ pipeline skeleton in deep_irt/bench/ (config
schema, run_cell, sweep, stubbed experiment scripts, configs_p2/) with TODO fill-in markers; NO heavy logic, NO runs.
DEFERRED to Phase 3 (needs ultracode): the code-heavy fill-in + the GPU runs (sequential on one 8 GB card). Open
config decisions (NOT blocking the scaffold, resolved at run-time): E2(a) coupling vehicle (ma_irt vs deep_irt-native);
cell size vs compute (N~800 days vs N~4000 weeks).
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

### Phase 4: Writing  [STATUS: NOT STARTED]
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
- [ ] E-levers  decoupling+static vs decoupling+dynamic, synthetic recovery and real reliability, NRM as control
- [ ] E8  adaptive item selection by max Fisher info at the running ability estimate, vs oracle
- [ ] Consolidation  re-run key panels under one training protocol
- [ ] E9 (optional)  classical MML-EM cross-calibration on EdNet/KDD

## Next action
Start Phase 1: launch *ml-math-researcher* and *psychometric-researcher* in parallel (both Opus) on the
theory spine and the flavor-correctness check. Reconcile their outputs into a short theory memo and show
the user before any writing.
