# Goal: identifiability audit of prediction-trained neural-IRT latents

> Charter document. Indexes the separate leg plans and defines done. Live status is in `docs/PROGRESS_identifiability_audit.md`.

## Objective (one sentence)
Show that prediction-trained neural IRT recovers its state READOUT (ability level, discrimination) but turns the STRUCTURE of the state (how it moves over time, how its dimensions couple) into reproducible-but-not-recovered artifacts; establish the reproduce-vs-recover gap as the diagnostic that detects this and Fisher leverage as the discriminator of which mechanism is at work; and position the result as the identifiability boundary around the interpretable-dynamic-KT program (PSI-KT included), not a competing model.

## Why this is one goal, not two
Leg A (temporal axis) and Leg B (dimensional axis) are the same failure on two axes, by two different mechanisms, caught by one diagnostic. Alone each is weak (an LSTM quirk; a small echo of Paper 2). Together they are a general boundary on what prediction-trained interpretable models can be trusted to report. The integration is the contribution.

## Non-goals (the guardrail, do not cross)
- Do NOT propose or build a gated trajectory model (that is PSI-KT's transition).
- Do NOT propose or build a learned propagation or concept-graph operator as a deliverable (that is PSI-KT's learned adjacency).
- Do NOT claim prerequisite-graph or learner-trait discovery; our coupling result is a negative.
- Do NOT turn a rescue regime into a recovery recipe; report it as a boundary (when the quantity is recoverable in principle, and at what cost).
- ELBO (PSI-KT) is touched only as the foil that proves the boundary. Never vendor its AGPL code into `deep_irt`; run it isolated if at all.

## Success criteria (done = all of)
1. One synthetic generator carrying a known ability trajectory AND a known coupling operator.
2. The three-metric diagnostic computed for every latent quantity: marginal Fisher, conditional retention eta, reproduce-vs-recover gap.
3. One figure: readout recovered, structure reproduced-not-recovered, on both the time and dimension axes, with the two mechanisms labeled.
4. Each axis's boundary characterized: whether and where the structure becomes recoverable, and at what cost.
5. The ELBO foil run on both axes: does generative targeting recover where prediction loss does not, and does even it reproduce-toward-prior on the temporal axis.
6. A writeup positioning the result as the boundary around PSI-KT and the dynamic-IRT program.

## The legs (read their plans)
- **Leg A, trajectory audit** -> `docs/research_program_1_dynamic_trajectory_plan.md`.
  Object: the rate / inertia of the ability trajectory.
  Mechanism (proven by the gates): estimator inductive bias. The rate is information-rich (conditional retention eta approx 0.45) yet the encoder's population-common smoothing prior overrides it (reproduces at 0.77, matches truth at 0.36).
  Separator: perturbation locality (learner-invariant impulse kernel = artifact).
  Rescue regime, as a boundary: spaced-repetition / repeated transients.
- **Leg B, coupling audit** -> `docs/research_program_2_latent_field_plan.md`.
  Object: the propagation operator P, the whole operator (diagonal included), not just the off-diagonal.
  Mechanism (proven by the gates): collinearity. Conditional retention eta approx 0.02 to 0.13 in the realistic regime.
  Rescue regime, as a boundary: strong anchoring (gain approx 2 to 8) crossed with a decorrelating curriculum; caps at eta approx 0.32 and is costly, it buys coupling identifiability by spending the discrimination channel's.

## Shared infrastructure
The three-metric diagnostic; the single trajectory-plus-coupling generator; the ELBO foil arm (reimplemented operator, AGPL-safe); gauge-free recovery metrics; the existence gate.

## Stop conditions (per leg)
- A leg is DONE when its boundary is characterized (recoverable or not, plus the cost), with the diagnostic and controls in place. A clean negative is a valid done.
- A leg is concluded FAILED only when every avenue in its exhaustion checklist (see the progress doc) has been tried and none separates signal from artifact, or none recovers above the null. A bottleneck is not a failure.
- Single GPU: Leg A and Leg B training stages never run concurrently. The no-GPU gates always run first.

## Autonomous run protocol (the manual trigger)
There is no `/goal` command, so this section IS the trigger. When a session starts on this goal, or a loop iteration fires, drive it like this.

Each iteration:
1. Read this goal doc and `docs/PROGRESS_identifiability_audit.md`.
2. Pick the next action: the first unblocked, unfinished item, in this order: an open gate, then the current leg's next experiment (the current leg is named in the progress doc's Current active leg line; WF-2 and WF-3 are defined there too), then the other leg, then the unified figure and writeup. Honor the single-GPU rule (never two training stages at once).
3. Execute it under the model economy: the strong model plans, synthesizes, and makes every go/no-go call; ml-math-researcher on the strong model does the Fisher and identifiability math; ml-system-architect and general-purpose on Sonnet do the build, the runs, the plots, and git; Haiku does trivial file ops. Fan out independent work via Workflow or Agent.
4. Record the outcome in the progress doc: update the status table, append a dated log line, tick the relevant exhaustion box, and commit the results note.
5. Repeat.

Completion (only when literally true, never to escape the loop):
- A leg is DONE when its boundary is characterized (recoverable or not, plus the cost). FAILED only when its exhaustion checklist is fully ticked and nothing worked.
- The whole goal is COMPLETE only when both legs are DONE-or-exhausted AND the unified figure and the writeup exist.

Pause and surface to the user (do not auto-proceed) when:
- a go/no-go is genuinely ambiguous, or
- a result forces a change to the thesis statement or to a non-goal guardrail, or
- an avenue needs a resource I cannot obtain (a dataset, a credential, GPU beyond the 8GB laptop).

Loop wiring (ralph-loop or `/loop`):
- Trigger prompt to feed the loop: "Resume the identifiability audit. Read docs/GOAL_identifiability_audit.md and docs/PROGRESS_identifiability_audit.md, do the next action per the Autonomous run protocol, update the progress doc, then continue."
- Completion promise (emit only when literally true): "Both legs of the identifiability audit are DONE or exhausted, the unified figure exists, and the writeup is drafted."
