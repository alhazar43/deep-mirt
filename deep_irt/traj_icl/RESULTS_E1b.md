# E1b results: in-context adaptation on a controllable task

The positive companion to the E1 ARC null. On ARC the models already know
the task, so demonstrations can only prime. E1b builds a task where the
mapping must be inferred in context, so a genuine adaptation curve exists
to measure. Code in `deep_irt/traj_icl/synth_remap.py`, results in
`outputs_e1b/results_e1b.json`.

## Setup

Four known categories (animal, fruit, vehicle, tool) are relabeled with
arbitrary single letters A to D by a fixed bijection that is never stated,
only implied by the demonstrations. The model sees k demonstration
(word, letter) pairs, then must label a held-out query word. Because the
mapping is arbitrary, the model cannot know it at k=0 and must infer it
from the demos. A 200-word query bank, shot counts k in
{0,1,2,4,8,16,32}, two conditions, true (the consistent mapping) and
shuffled (random per-demo letters, so nothing consistent to learn).
Scoring is the next-token log-probability over A to D, the same as E1.
Three models, Qwen2.5-Instruct 0.5B, 1.5B, 3B. A joint 2PL places all 42
(model, k, condition) examinees on one shared scale.

## Results

The task validates cleanly. At k=0 every model is at chance (accuracy
0.18 to 0.23, theta near 0), confirming the mapping is genuinely hidden.
Under shuffled labels accuracy stays at chance at every k with no trend,
confirming there is no signal to learn.

**Under true labels the ability trajectory rises sharply, and the
adaptation grows with model size.** Per-model priming-corrected gain,
theta_true(k_max) minus theta_shuffled(k_max), on the shared scale.

| model | gain | true accuracy k=0 -> k=32 |
|---|---|---|
| 0.5B | +0.67 | 0.18 -> 0.57 |
| 1.5B | +2.65 | 0.21 -> 0.83 |
| 3B | +5.42 | 0.23 -> 0.95 |

The bigger the model, the more it adapts from the same in-context
evidence, a clean model-size ordering of in-context adaptation on one
measurement scale. This is exactly what the ARC null lacked, and it is the
positive machine-front result.

**The curve shape is threshold-like, not the smooth approach of a human
learning curve.** Accuracy is flat near chance through k=8 and then jumps
at k=16 (3B theta goes from about 0 at k=8 to +2.45 at k=16 to +4.36 at
k=32). This is a phase transition in shot count, the model suddenly
infers the mapping once enough evidence accumulates, rather than the
gradual exponential approach of E0's synthetic human curves. So the
exponential rate dtheta/de, the right summary for a smooth learning curve,
is the wrong summary here, the machine adaptation is better described by
its magnitude (the priming-corrected gain) and its threshold location.
Plot in `outputs_e1b/e1b_adaptation_curves.png`.

## Reading

E1 and E1b together make the machine-front point. The measure registers no
adaptation when there is nothing to learn in context (ARC, a known task,
true equals shuffled, flat theta) and a large, size-scaled adaptation when
there is (the remapping task, true diverges from shuffled, rising theta).
The priming control does the load-bearing work, only the gap above the
shuffled baseline counts as learning.

A genuine cross-respondent finding falls out. The trajectory SHAPE differs
by respondent type. Human and synthetic learning curves approach an
asymptote smoothly, so a single rate parameter captures them (E0, E2).
LLM in-context adaptation is threshold-like, a phase transition once
enough demonstrations accumulate, so the comparable estimand on the
machine front is the adaptation magnitude and its threshold, not a smooth
rate. The trajectory is the unifying object, but its parameterization is
respondent-specific.

## Robustness

A follow-up varied the arbitrary category-to-letter mapping across three
seeds and refined the shot grid (k up to 32 with finer spacing), on the
1.5B and 3B models, results in `outputs_e1b_robust/`. The headline holds.
The priming-corrected gap is positive in all three mappings for both
models, and the 3B gap exceeds the 1.5B gap in every seed (3B max gap mean
0.78, 1.5B 0.64). The adaptation threshold, the smallest k where the gap
passes half its maximum, localizes to k about 10 plus or minus 2 for both
models regardless of the mapping (1.5B k* in {10, 10, 12}, 3B k* in
{8, 10, 10}). So the size-scaled threshold adaptation is a property of the
task and the model, not of a lucky bijection, and the phase transition
sits near ten shots.

## Limitations

A synthetic task with abstract letter labels, three model sizes of one
family, and a shot grid that brackets the threshold (8 to 16) rather than
localizing it. The shared-scale magnitudes depend on the examinee pool
(many chance-level cells anchor the low end). The result demonstrates that
in-context adaptation is recoverable as a rising trajectory with a
size-scaled magnitude, it does not yet pin the threshold or test a real
task where in-context learning helps (low-resource translation is the
real-data follow-up).
