# E1 results: in-context adaptation on a real benchmark (ARC)

Front 1 of the trajectory-recovery program. A ladder of Qwen2.5-Instruct
models answers a shared ARC item bank at increasing shot counts, a joint
2PL places every model-at-every-shot-count on one item scale, and theta(k)
is read as the in-context adaptation curve. Code in `deep_irt/traj_icl/`,
results in `outputs/results_e1.json`.

## Setup

Three models, Qwen2.5-Instruct 0.5B, 1.5B, and 3B (fp16, local, no API).
A shared 400-item bank, 200 ARC-Easy plus 200 ARC-Challenge, fixed across
all models. Shot counts k in {0, 1, 2, 4, 8, 16}. Two conditions, true
labels and shuffled labels on the demonstrations (the Min et al. 2022
priming control). Each item is scored by the model's next-token
log-probabilities over the four option letters. Each (model, k, condition)
is one examinee, 36 in all, and a joint 2PL (fit by maximum likelihood)
places them on one shared ARC scale.

## Results

The 2PL fits well (BCE 0.208) and the shared scale orders the ladder as
expected, mean ability about -1.0 for 0.5B, about +0.1 for 1.5B, about
+1.1 for 3B. This is the tinyBenchmarks and Growing Pains result, models
as examinees on a shared scale, now carrying a shot-count axis.

**The adaptation curve is flat to declining, never sustained rising.**

| model | theta(0) | theta(16) true | shape |
|---|---|---|---|
| 0.5B | -0.73 | -1.71 | declines, demos hurt |
| 1.5B | +0.28 | +0.05 | flat |
| 3B | +0.79 | +0.99 | flat, slight hump at k=4 |

The 0.5B model is actively hurt by demonstrations, its accuracy falling
from 0.59 at k=0 to 0.36 at k=16, the long context distracts a model too
small to use it. The 1.5B and 3B curves are flat.

**No genuine in-context learning, only priming.** The true-minus-shuffled
ability gap at the largest k is about zero for every model (+0.03, -0.08,
+0.04). Demonstrations with correct labels do no better than
demonstrations with shuffled labels. On a benchmark these instruct models
already know, demonstrations carry format and distribution signal, not
new task content, exactly the Min et al. 2022 finding, here quantified on
a latent ability scale across a model ladder.

## Reading

This is an honest null with a useful validity check. ARC is a task the
models already know, so demonstrations cannot teach it anything, and the
measure correctly registers the absence of adaptation rather than
inventing a curve. The rate readout is meaningful only on a genuinely
rising trajectory, so no adaptation rate is reported here, the curves do
not rise. The shared-scale calibration and the priming control both work,
which is what licenses the next step.

To measure a genuine in-context adaptation rate we need a task with
something to learn in context, where true and shuffled labels must
diverge. That is E1b, a synthetic label-remapping task (a concept the
model knows, relabeled with arbitrary tokens defined only by the demos),
the machine analog of E0's known-rate synthetic curves.

## Limitations

One benchmark, one model family at three sizes, multiple choice only. The
result bounds what a real known-answer benchmark can show (priming, not
learning) and motivates the controllable positive-adaptation task, it does
not by itself demonstrate rate recovery on the machine front.
