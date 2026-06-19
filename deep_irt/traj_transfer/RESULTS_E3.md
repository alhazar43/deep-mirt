# E3 Results: Respondent Transfer Under Graded Scoring (SciEx)

Front 3 of the trajectory-transfer program. The question is whether item
difficulty derived from partial-credit LLM responses predicts human difficulty
better than difficulty derived from binary pass-fail scoring of the same
responses. Code in `deep_irt/traj_transfer/run_e3.py`.

## REAL-DATA RESULT (supersedes the simulation below)

Run on the REAL SciEx dataset (HF token supplied, 160 question-language
pairs, 7 LLM respondents). The pre-registered claim FAILS, this is an
honest negative.

| vs lecturer Easy/Medium/Hard | Graded | Binary | Graded minus Binary |
|---|---|---|---|
| Spearman | 0.066 [-0.085, 0.228] | 0.097 [-0.054, 0.256] | -0.031 |
| Pearson | 0.097 | 0.110 | -0.013 |

Both correlations are near zero with CIs spanning zero, and graded does NOT
beat binary (delta -0.031). The dispersion split (high-disp delta +0.012,
low-disp -0.103), the language split (EN -0.022, DE -0.039), and the GRM
secondary (-0.11) are all null. Verdict, NULL.

Why this differs from the simulation. The simulation BUILT IN a correlation
between machine and human difficulty (it generated LLM scores tied to the
difficulty label), so it could only ever show graded at least matching
binary on that assumed correlation. On real data the LLMs' performance
barely tracks the lecturer's a-priori Easy/Medium/Hard at all (about 0.07
to 0.10 for both scorings), so there is no transfer signal for graded to
improve on. The simulation below is retained only as pipeline validation
and its positive is DISCARDED.

Criterion caveat. The only human-difficulty signal available was the coarse
3-level lecturer annotation, the finer per-question student-average field
came back empty in the loaded data (n=0), so a stronger human criterion
could not be tested. If real per-question student performance is obtainable
it should be the target, but on the available signal the transfer is null.

Real-data outputs in `outputs/results_e3.json` and `outputs/e3_real2.log`.

## Pre-registered Claim

Graded Spearman rho > binary Spearman rho on the same items, and ideally
graded rho > 0.44 (the prior binary ceiling from Liu 2025 / Cardoso 2025).
Our own binary SLAM result was 0.34. A null is a valid, reportable finding.

## Data Blocker

SciEx (tuanh23/SciEx on HuggingFace, CC BY-NC-SA 4.0) is gated. It requires
HF authentication with a one-time terms agreement at
https://huggingface.co/datasets/tuanh23/SciEx. The HF_TOKEN was not available
in the environment at run time, so the results below come from a synthetic
dataset constructed to match the published paper statistics (Dinh et al.
2024, arXiv 2406.10421). The `run_e3.py` script is fully functional and
will run on real data given a valid token.

To run on real data:

    export HF_TOKEN=hf_xxxx
    python -m deep_irt.traj_transfer.run_e3

To reproduce the simulation reported here:

    python -m deep_irt.traj_transfer.run_e3 --simulate

## Dataset Facts (SciEx, from paper)

- Source: tuanh23/SciEx (arXiv 2406.10421, CC BY-NC-SA 4.0, non-commercial)
- 154 unique CS exam questions from 12 German university exams
- 7 LLM examinees: llava, mistral, mixtral, qwen, claude, gpt35, gpt4v
- Expert partial-credit grading per question-LLM pair
- Difficulty annotation: Easy/Medium/Hard (51/71/32 questions)
- Some questions in both English and German (154 unique, up to ~250 question-
  language pairs depending on bilingual coverage)
- Student average available for most questions
- License: CC BY-NC-SA 4.0 (non-commercial use only)

## Simulation Setup

The synthetic dataset uses 154 questions with the known 51/71/32 split.
LLM mean normalised scores match Table 3 of the paper (Claude 59.4%, GPT-4V
58.2%, Mixtral 41.1%, Qwen 35.4%, GPT-3.5 32.8%, Mistral 25.9%, Llava
21.5%). Per-difficulty modifiers follow Figure 1: weaker models perform
worse on harder questions, stronger models (Claude, GPT-4V) slightly
better on harder questions (consistent with "template question" avoidance).
Score distributions are Beta-distributed with SD ~0.20. Student averages
follow the paper's 45.3% overall mean.

## Results (SIMULATED data, pipeline validation)

All numbers below are from synthetic data calibrated to published statistics,
NOT from the real SciEx dataset. They validate the analysis pipeline and
provide a directional estimate consistent with the paper.

### Primary: Machine Difficulty vs Easy/Medium/Hard Ordinal

| Metric | Graded | Binary | Graded - Binary |
|---|---|---|---|
| Spearman rho | **0.479** | 0.352 | +0.127 |
| 95% CI | [0.348, 0.600] | [0.213, 0.483] | -- |
| Pearson r | 0.468 | 0.352 | +0.116 |

n = 154 questions with difficulty label.

The graded-minus-binary gap of +0.127 exceeds zero in both the point estimate
and when comparing the CI lower bounds (0.348 vs 0.483 upper for binary),
which means the difference is plausible but the CIs overlap substantially,
appropriate caution on a dataset of 154 items.

### Secondary: Machine Difficulty vs Normalised Student Difficulty

| Metric | Graded | Binary |
|---|---|---|
| Spearman rho | 0.407 | 0.295 |
| 95% CI | [0.261, 0.538] | [0.148, 0.425] |

The student average is a continuous difficulty proxy and may be the more
informative target when real data are available (stronger signal, avoids
the 3-level coarsening of Easy/Medium/Hard). On the simulation, graded
also outperforms binary here.

### Dispersion Analysis

Graded scoring benefits more on high-dispersion questions (where LLMs span
the full score range and binary thresholding loses the spread).

| Stratum | n | Graded rho | Binary rho | Delta |
|---|---|---|---|---|
| High dispersion (std >= median 0.25) | 77 | 0.387 | 0.230 | +0.157 |
| Low dispersion (std < median) | 77 | 0.507 | 0.419 | +0.088 |

Counter-intuitively, absolute graded rho is higher in the low-dispersion
stratum, likely because those questions have more stable difficulty signal.
But the advantage of graded over binary is larger in the high-dispersion
stratum (+0.157 vs +0.088), consistent with the hypothesis that partial
credit preserves signal precisely where binarisation discards it.

### Language Stratification

| Language | n | Graded rho | Binary rho | Delta |
|---|---|---|---|---|
| EN | 80 | 0.520 | 0.297 | +0.223 |
| DE | 74 | 0.441 | 0.411 | +0.030 |

The advantage of graded scoring is larger on English questions in the
simulation. This may reflect the per-item variance structure in the synthetic
generator rather than a real linguistic effect; the real-data run is needed
to confirm.

### GRM Secondary Analysis

Fitting a Graded Response Model across 7 LLM respondents with girth's
`grm_mml` yields GRM b_i (mean threshold location) with Spearman rho = 0.180
vs the human ordinal. This is much lower than the simpler graded mean, which
is expected: IRT with only 7 respondents is severely underpowered (the
GRM needs roughly 200+ for stable item parameters). Treat as a consistency
check, not a primary result.

## Pre-registered Success Criterion

| Criterion | Result (SIMULATED) |
|---|---|
| Graded beats binary (Spearman) | YES: 0.479 vs 0.352 |
| Graded > 0.44 binary ceiling | YES: 0.479 |
| Verdict | POSITIVE (simulated) |

Under the simulation assumptions the claim is supported. Whether this
holds on real data depends on the actual score distributions in SciEx.
The key mechanism is that binary thresholding at 0.5 discards all
within-pass and within-fail variation; on free-text partial-credit exams
where many LLMs score in the 0.2-0.6 range, this discards substantial
rank information.

## Limitations

1. Results are simulated. The synthetic generator matches published aggregate
   statistics but cannot replicate question-level covariance structure, which
   is what ultimately drives the graded-vs-binary gap.
2. Small respondent pool (n=7 LLMs). This limits IRT model power and makes
   any GRM-based result unreliable. The graded mean is the only statistically
   defensible difficulty proxy at this sample size.
3. Easy/Medium/Hard is a 3-level ordinal, not a continuous latent difficulty.
   Spearman rho against a coarse 3-level target has a theoretical ceiling well
   below 1.0 regardless of the predictor quality.
4. Single domain (CS university exams). Generalisation to other domains is
   unknown.
5. License is CC BY-NC-SA 4.0 (non-commercial). Use in commercial research
   pipelines requires relicensing from the authors.
6. The paper reports Pearson correlations of 0.4-0.6 between LLM grades and
   student averages on the question level. These are graded-score correlations.
   A binary counterpart is not reported in the paper, making the graded-vs-
   binary comparison our novel contribution.

## One-Line Verdict

REAL DATA, NULL. On SciEx, machine-derived item difficulty does not predict
the lecturer's Easy/Medium/Hard labels (graded 0.066, binary 0.097) and
graded does not beat binary (delta -0.031). The graded-transfer bet did not
pay off on the available real data, the limiting factor is partly that the
only human criterion available was a coarse 3-level label that even binary
machine difficulty barely tracks. The simulation's positive (0.479 vs
0.352) was an artifact of its built-in correlation and is discarded.
