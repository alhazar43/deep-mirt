# Engineering Overview

Architecture and engineering philosophy. This document governs how the
framework is built. It sits beside `Thesis_overview.md` and
`Thesis_chapter.md` and takes precedence in design decisions. Citations
for the methods named here are in the thesis overview.

## Engineering philosophy

**Modularity by interface.** The latent ability state is the contract
between encoder and decoder. Each side depends only on that state, not on
the other's internals. This is what makes substitution real rather than
aspirational.

**Substitute, do not rewrite.** A stronger encoder is plugged into the
same interface. A new response format is handled by a new decoder. The
rest of the system is unchanged.

**Anchoring is a first-class operation.** Extension of the scale without
recalibration is a core capability, designed in from the start, not a
post-hoc script.

**Chapter 0 is frozen.** `ma-irt` is the groundwork attempt, Chapter 0,
and the first instance of an encoder paired with an ordinal decoder. It
is not modified beyond additive configuration, and the framework builds
beyond it rather than extending it. This protects a public, reviewed
package and keeps the starting point fixed while the framework moves.

**Extend from `rl/`.** New code lives in the research package. Edits to
the kernel are additive configs only. See the project boundary note in
memory.

**Evaluate on the right axis.** Report scale stability, transfer,
calibration, and trajectory recovery alongside accuracy and ordinal
agreement. Never report accuracy alone.

**Graded paths are first-class.** Do not concentrate development on
binary responses. A graded decoder is built and tested alongside the
binary one.

**Reproducible and explicit.** Deterministic seeds, config-driven
experiments, isolated worktrees for parallel work, explicit file
staging, tests before merge.

## Architecture

**Encoder.** Maps a response history to a latent ability state.
Interchangeable across memory, recurrent, and transformer designs.
Chosen by use.

**Latent ability state.** The interface contract. It carries the scale
and the anchors, and it is the only object the decoder and the
downstream policy may read. Encoder internals are private.

**Decoder.** Maps the latent state to a response distribution under one
response format. A binary decoder uses a two-parameter logistic form. An
ordinal decoder uses step thresholds, a generalized partial credit form.
A graded decoder handles continuous or rubric scores. Decoders are
interchangeable and selected by format.

**Anchoring module.** Holds designated item parameters or scale
references fixed while new item parameters are estimated. This
generalizes fixed-parameter linking to a setting where the scale is
defined partly by encoder weights rather than by explicit item
parameters alone. It is the central engineering problem of the thesis.

**Selection policy, later stage.** Chooses items both to measure ability
and to calibrate new content, and adapts jointly with the measurement
model rather than on top of frozen parameters.

## The latent-state contract

Define the interface explicitly and keep it small. It should expose the
current ability estimate, its uncertainty, and the active scale anchors.
It should not expose encoder hidden states or decoder parameters. Every
substitution test, swapping an encoder or swapping a decoder, is a test
against this contract. If a change requires reading across the contract,
the boundary is wrong.

## Evaluation

Define and report the following alongside standard metrics.

**Scale stability.** The drift of anchored estimates after an extension
step. Low drift is the target.

**Transfer.** The degree to which parameters estimated from one
respondent set predict another, for example language-model-derived item
parameters predicting human item behavior.

**Trajectory recovery.** On synthetic data with a known generating
curve, the error between the estimated improvement trajectory and the
true one.

**Calibration and ordinal agreement.** Reliability of predicted response
distributions, and quadratic weighted agreement for ordinal responses.

## Build order

1. The latent-state contract, with a reference encoder and reference
   decoder behind it.
2. The anchoring operation and the scale-stability metric.
3. The first-paper experiments (see `Thesis_chapter.md`), starting from
   the language-model judging entry point with graded responses.
4. Multidimensional growth with masked loadings.
5. The co-adaptive selection policy.
