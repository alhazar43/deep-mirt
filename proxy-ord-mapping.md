# Proxy-Ordinal Mapping: ASSISTments Binary to K=4

## Overview

This mapping converts binary correctness logs from the ASSISTments 2009
competition dataset into K=4 ordinal response categories. The mapping uses
five features available per interaction: `correct`, `attemptCount`,
`hintCount`, `bottomHint`, and `frIsHelpRequest`.

Only original problems (`original == 1`) are included; scaffolding
sub-problems are excluded.

## Mapping Rules

| Category | Label             | Rule                                                                                    |
|----------|-------------------|-----------------------------------------------------------------------------------------|
| 3        | Mastery           | `correct == 1`, `attemptCount == 1`, `hintCount == 0`                                  |
| 2        | Partial mastery   | `correct == 1` and (`attemptCount > 1` or `hintCount > 0`), `bottomHint == 0`          |
| 1        | Independent effort| `correct == 0`, `frIsHelpRequest == 0`, `hintCount == 0`, `attemptCount <= 2`           |
| 0        | Struggled         | `correct == 0` and any of: `frIsHelpRequest == 1`, `hintCount > 0`, `attemptCount >= 3` |

**Bottom-out override**: Any interaction with `bottomHint == 1` maps to category 0
regardless of correctness, following the Ostrow et al. (2015) convention that
bottom-out hints reveal the answer, making "correct" responses uninformative.

## Decision Logic

```
if bottomHint == 1:
    -> 0 (Struggled)
elif correct == 1:
    if attemptCount == 1 and hintCount == 0:
        -> 3 (Mastery)
    else:
        -> 2 (Partial mastery)
else:  # correct == 0
    if frIsHelpRequest == 0 and hintCount == 0 and attemptCount <= 2:
        -> 1 (Independent effort)
    else:
        -> 0 (Struggled)
```

## Distribution

On original problems (N = 249,105 interactions):

| Category | Count  | Percentage |
|----------|--------|------------|
| 0        | 55,251 | 22.2%      |
| 1        | 86,642 | 34.8%      |
| 2        | 27,760 | 11.1%      |
| 3        | 79,452 | 31.9%      |

## Construct Validity

Mean BKT knowledge estimate (Ln) increases monotonically across categories:

| Category | Mean Ln |
|----------|---------|
| 0        | 0.134   |
| 1        | 0.158   |
| 2        | 0.207   |
| 3        | 0.618   |

This confirms that the ordinal categories align with the latent knowledge
construct estimated by Bayesian Knowledge Tracing, providing evidence that
the proxy mapping captures meaningful gradations in student performance.

## Literature Basis

The correctness-first split follows Wang & Heffernan (2011, 2013), who
demonstrated that richer response categories improve knowledge tracing
accuracy. The bottom-out hint convention follows Ostrow et al. (2015).
