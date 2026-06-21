# Learning Dynamics Progress and Results

This document tracks expectation versus reality for the learning-dynamics study.
The stable research plan is in `docs/learning_dynamics_research_plan.md`.

Rule: update this file after every run, including nulls and refutations.

Decision labels:

- supports claim
- weak support
- mixed
- null
- refutes claim
- blocked
- needs rerun
- planned

## Status Dashboard

| ID | Experiment | Expected | Reality | Verdict | Next Action |
|---|---|---|---|---|---|
| E0 | Existing consolidated spine | Prior runs should already support the finite-time alpha-recovery framing | Existing evidence is consolidated in `docs/LEARNING_DYNAMICS_STUDY.md` and `deep_irt/bench/outputs` | supports claim with caveats | Preserve as historical evidence, do not overwrite |
| E1 | Representation gate | Selective or decoupled alpha should lift alpha without simply widening everything | Existing gate, swap, alpha-fix, and ablation outputs show alpha representation matters, but some older width framing was later retracted or refined | supports claim with caveats | Use refined framing only |
| E2 | Recovery trajectories | Endpoint tables should hide important dynamics | Existing trajectory outputs show shared-wide alpha can peak and then decay, while decoupled alpha holds better | supports claim | Use checkpoint curves in write-up |
| E3 | Gradient and Fisher mechanism | Alpha should be lower-information and slower than theta/beta | Existing study converged on conditioning and recovery-rate framing, not a simple population-limit law | supports revised claim | Keep the downgraded mechanism wording |
| E4 | Alpha/beta asymmetry | Dynamic alpha should help more than dynamic beta | Existing alpha/beta asymmetry table reports positive alpha-vs-K trend and near-neutral beta dynamics | supports claim | Replicate if new code changes |
| E5 | Positive-map ablation | Exp should beat generic positivity maps after fair controls, especially high-alpha items | Full K=2,4,8 runs completed. Exp does not beat the best smooth non-exp maps by a meaningful margin, and the high-alpha split is inconsistent | mixed | Reframe toward smooth positive-map geometry |
| E6 | Initialization and LR controls | Exp advantage should survive matched effective initialization and LR tuning | Matched-init LR sweeps across K=2,4,8 leave no stable exp-specific advantage. LR tuning confirms the comparison is fair enough to reject the strong exp-only version | mixed | Do not run an exp-only E7 without rewriting it |
| E7 | Geometry-matched control | Exp-equivalent alpha-space preconditioning should partly reproduce exp | No-training and direct-alpha controls completed. With true theta/beta fixed and a wide LR grid, direct alpha-space conditions mostly converge to the same recovery band, so the neural map separation is not explained by alpha-space preconditioner magnitude alone | refutes claim | Treat preconditioner-only as rejected; next test representation/optimizer interactions in the neural model |
| E7a | Neural map-isolation control | If direct alpha-space geometry is insufficient, freezing neural representation pieces and stabilizing raw/ReLU maps should identify whether the remaining effect is representation learning, item-key learning, range instability, or optimizer interaction | Full K=2,4,8 grids completed. Clipped raw/ReLU plus gradient clipping does not rescue the learned-all gap; freezing item embeddings lowers recovery and gives category-dependent raw/ReLU behavior; freezing the backbone or encoder breaks recovery | mixed | Write mechanism as neural representation interaction plus smooth-map stability; do not use range-only or exp-only wording |
| E8 | Residual diagnostics | Contextual alpha residual should carry interpretable state-dependent structure but may include null artifacts | Null, planted-signal, and scale controls completed. Residuals contain a strong static-alpha null artifact, but matched-null subtraction directionally detects planted theta-dependent discrimination; magnitude remains strongly attenuated | mixed | Use directional-not-calibrated wording; do not interpret real-data residuals without the null-artifact caveat |
| E9 | Misspecification | Alpha may absorb slope-like, local-dependence, response-style, or threshold misspecification | Local-dependence, noisy-threshold, and learner response-style controls completed. Response style is the first probe where contextual alpha clearly tracks the planted nuisance, but item recovery worsens; local dependence/noisy thresholds remain dominated by null artifacts or recovery degradation | mixed | Run threshold disorder or DIF before claiming a general taxonomy |
| E10 | GRM/NRM extension | Slope-like parameters should behave like alpha, intercepts like beta | Existing NRM probes are preliminary and mixed | mixed | Treat as thesis extension, not minimum paper |
| E11 | Real-data stability | Real data should support prediction, calibration, and stability, not true recovery | Not yet organized for this study | planned | Use only as supporting evidence |

## Experiment Template

Use this template for every new run.

```markdown
## E<ID>: <Experiment Name>

### Expectation
What was predicted before running.

### Setup
Dataset, model, scripts, seeds, hyperparameters, and outputs.

### Reality
What actually happened. Include numbers and paths.

### Interpretation
What this means for the claim.

### Decision
One of: supports claim, weak support, mixed, null, refutes claim, blocked,
needs rerun, planned.

### Next Action
The next concrete experiment, rerun, or documentation step.
```

## E0: Existing Consolidated Spine

### Expectation

The existing learning-dynamics study should provide the empirical backbone:
gate, trajectory, gradient, Fisher, K-sweep, N-sweep, and state-conditioned
alpha diagnostics.

### Setup

Primary references:

- `docs/LEARNING_DYNAMICS_STUDY.md`
- `docs/learning_dynamics_toy.md`
- `deep_irt/bench/outputs/gate_table.md`
- `deep_irt/bench/outputs/trajectory_table.md`
- `deep_irt/bench/outputs/gradient_conflict_table.md`
- `deep_irt/bench/outputs/ksweep_table.md`
- `deep_irt/bench/outputs/ndata_sweep_table.md`
- `deep_irt/bench/outputs/alpha_beta_asymmetry_table.md`

### Reality

The consolidated doc says the original strong population-limit
learning-dynamics law was downgraded. The stable contribution is finite-data
plus learned-representation recovery, with low Fisher setting recovery rate
rather than endpoint bias.

### Interpretation

This is the right backbone for the new plan. The new study should not restart
from scratch. It should add the positive-map geometry layer and keep the refined
representation claim.

### Decision

supports claim with caveats.

### Next Action

Use existing results as the evidence base and avoid citing retracted width-only
framing as a live claim.

## E1: Representation Gate

### Expectation

Selective alpha representation should improve discrimination recovery without
paying the theta tax of generic widening.

### Setup

Existing outputs:

- `deep_irt/bench/outputs/alpha_fix_table.md`
- `deep_irt/bench/outputs/gate_table.md`
- `deep_irt/bench/outputs/swap_table.md`
- `deep_irt/bench/outputs/ablation_table.md`

### Reality

The existing tables show a consistent alpha recovery lift from selective or
decoupled treatment. The swap bench reports the effect across LSTM,
Transformer, and DKVMN. However, the consolidated doc also warns that some
older width-decoupling wording was retracted or refined as a capacity artifact.

### Interpretation

The live claim should be parameter-specific representation and finite-time
recoverability, not a simplistic "separate width always dominates" story.

### Decision

supports claim with caveats.

### Next Action

When writing, use the current refined mechanism from
`docs/LEARNING_DYNAMICS_STUDY.md` and cite the historical caveat explicitly.

## E2: Recovery Trajectories

### Expectation

Endpoint recovery tables should miss important training dynamics.

### Setup

Existing output:

- `deep_irt/bench/outputs/trajectory_table.md`

### Reality

The trajectory table records that shared-wide alpha can reach a high value early
and then decay under continued training, while decoupled alpha rises and holds
better. Theta also degrades under long training in bare configurations.

### Interpretation

This supports the learning-dynamics framing. The model may visit a useful
solution and then leave it, so endpoint-only reporting is inadequate.

### Decision

supports claim.

### Next Action

Use checkpointed curves and recovery AUC in the new positive-map ablation.

## E3: Gradient and Fisher Mechanism

### Expectation

Alpha should be lower-information and more weakly conditioned than theta or beta.
The mechanism may appear as gradient conflict, Fisher asymmetry, or convergence
rate asymmetry.

### Setup

Existing references:

- `docs/LEARNING_DYNAMICS_STUDY.md`
- `docs/learning_dynamics_toy.md`
- `deep_irt/bench/outputs/gradient_conflict_table.md`
- `deep_irt/bench/outputs/fisher_ratio.json`

### Reality

The mechanism did not remain a simple gradient-capture or population-limit law.
The consolidated result says low Fisher sets the recovery rate, not the endpoint,
and that the learned encoder is important for the empirical decoupling effect.

### Interpretation

The theory should be modest and local. It should support finite-time recovery
dynamics, not claim a universal population-limit bias.

### Decision

supports revised claim.

### Next Action

Add positive-map gradient-flow propositions as a mechanism layer, while keeping
the existing Fisher-conditioning caveat.

## E4: Alpha/Beta Asymmetry

### Expectation

Making low-Fisher alpha dynamic should help more than making high-Fisher beta
dynamic. Dynamic beta should be neutral or harmful.

### Setup

Existing output:

- `deep_irt/bench/outputs/alpha_beta_asymmetry_table.md`

### Reality

The existing table reports mean `delta_alpha = +0.042` and mean
`delta_beta = +0.003`, with positive delta-alpha versus K correlation. The file verdict says
asymmetry is confirmed.

### Interpretation

This supports the parameter-specific design principle, but should be replicated
if model code or loss code changes.

### Decision

supports claim.

### Next Action

Keep as core evidence. Add positive-map ablation without changing the static
and dynamic beta negative-control logic.

## E5: Positive-Map Ablation

### Expectation

If positivity alone is insufficient, exp should outperform other positive maps
under fair controls, especially on high true-alpha items and finite training
budgets.

### Setup

Planned maps:

- unconstrained raw alpha,
- ReLU,
- softplus,
- softplus plus epsilon,
- scaled softplus,
- temperature softplus,
- sigmoid,
- scaled sigmoid,
- square plus epsilon,
- exponential,
- clipped exponential,
- direct positive projection if feasible.

Required controls:

- matched effective alpha initialization,
- per-map learning-rate sweep,
- same encoder,
- same decoder except alpha map,
- same seeds,
- same parameter budget,
- same regularization,
- monitored gradient norms and effective alpha ranges.

### Reality

Implementation support now exists for named alpha maps in the GPCM and binary
decoders, with model and bench-engine pass-through. The supported map set is:

- `identity` / `raw`,
- `relu`,
- `softplus`,
- `softplus_eps`,
- `scaled_softplus`,
- `temperature_softplus`,
- `sigmoid`,
- `scaled_sigmoid`,
- `square`,
- `exp`,
- `clipped_exp`.

The dedicated controlled runner is now implemented:

- `deep_irt/bench/run_alpha_map_bench.py`
- full outputs: `deep_irt/bench/outputs/alpha_map_results.json` and
  `deep_irt/bench/outputs/alpha_map_table.md`
- quick or smoke outputs use a separate prefix, so they do not overwrite full
  evidence.

The runner currently enforces:

- same LSTM/GPCM architecture across maps,
- `state_alpha=True` and `item_key_dim=64`,
- same dataset, seeds, parameter budget, and loss,
- matched effective alpha initialization through inverse maps,
- per-map learning-rate rows,
- best-LR summary selected by alpha Spearman, then high-alpha Spearman, then
  QWK,
- alpha-head, decoder, and total gradient-norm summaries,
- final effective-alpha range summaries,
- high true-alpha recovery metrics.

Focused tests pass:

```text
conda run -n research python -m pytest deep_irt\tests\test_alpha_link.py
20 passed

conda run -n research python -m pytest deep_irt\tests\test_decoupled_alpha.py
22 passed

conda run -n research python -m pytest deep_irt\tests\test_alpha_link.py deep_irt\tests\test_alpha_map_bench.py
23 passed
```

A tiny smoke run also passes:

```text
conda run -n research python -m deep_irt.bench.run_alpha_map_bench --quick --device cpu --maps softplus exp relu --lrs 0.01 --epochs 2 --n-learners 40 --n-items 10 --seq-len 10 --seeds 0 --K 4 --out-prefix alpha_map_smoke
```

Smoke outputs:

```text
deep_irt/bench/outputs/alpha_map_smoke_results.json
deep_irt/bench/outputs/alpha_map_smoke_table.md
```

Smoke numbers were only an integration check. They must not be interpreted as
study evidence.

The canonical K=4 full run completed:

```text
conda run -n research python -m deep_irt.bench.run_alpha_map_bench --device cuda --K 4 --seeds 0 1 2 3 4
```

Full run setup:

- mode: full,
- device: cuda,
- epochs: 150,
- `N=800`, `Q=60`, `T=60`, `K=4`,
- seeds: `0 1 2 3 4`,
- learning rates: `0.003`, `0.01`, `0.03`,
- total fits: 165.

Full outputs:

```text
deep_irt/bench/outputs/alpha_map_results.json
deep_irt/bench/outputs/alpha_map_table.md
```

Best-LR results by alpha Spearman:

| Map | Best LR | Alpha Spearman | High-alpha Spearman | Theta Spearman | Beta Spearman | QWK |
|---|---:|---:|---:|---:|---:|---:|
| exp | 0.01 | 0.928+-0.015 | 0.692+-0.151 | 0.958+-0.005 | 0.988+-0.002 | 0.599+-0.021 |
| clipped_exp | 0.01 | 0.928+-0.015 | 0.692+-0.151 | 0.958+-0.005 | 0.988+-0.002 | 0.599+-0.021 |
| softplus_eps | 0.01 | 0.926+-0.017 | 0.725+-0.133 | 0.956+-0.005 | 0.987+-0.002 | 0.596+-0.013 |
| temperature_softplus | 0.01 | 0.925+-0.014 | 0.706+-0.165 | 0.957+-0.005 | 0.987+-0.003 | 0.596+-0.011 |
| scaled_softplus | 0.01 | 0.924+-0.017 | 0.706+-0.137 | 0.957+-0.003 | 0.987+-0.002 | 0.598+-0.015 |
| softplus | 0.01 | 0.924+-0.017 | 0.716+-0.136 | 0.958+-0.004 | 0.988+-0.002 | 0.599+-0.016 |
| square | 0.01 | 0.922+-0.016 | 0.683+-0.146 | 0.954+-0.005 | 0.986+-0.003 | 0.599+-0.023 |
| scaled_sigmoid | 0.01 | 0.921+-0.019 | 0.727+-0.129 | 0.957+-0.006 | 0.987+-0.003 | 0.597+-0.018 |
| sigmoid | 0.01 | 0.911+-0.023 | 0.735+-0.148 | 0.956+-0.007 | 0.986+-0.002 | 0.593+-0.016 |
| identity | 0.01 | 0.906+-0.009 | 0.713+-0.145 | 0.960+-0.005 | 0.963+-0.022 | 0.593+-0.020 |
| relu | 0.01 | 0.901+-0.015 | 0.721+-0.159 | 0.955+-0.005 | 0.980+-0.009 | 0.587+-0.023 |

Exp-family readout:

- exp and clipped-exp are identical at K=4 under this clip range, which means
  clipping is inactive in the observed region.
- exp beats the best generic non-exp map, `softplus_eps`, by only `+0.002` on
  mean alpha Spearman.
- exp trails `softplus_eps` by `-0.033` on high-alpha Spearman.
- raw identity and ReLU are clearly weaker than the smooth positive maps.

The category extension also completed without rerunning K=4:

```text
conda run -n research python -m deep_irt.bench.run_alpha_map_bench --device cuda --K 2 8 --seeds 0 1 2 3 4 --out-prefix alpha_map_kext
```

Extension setup:

- mode: full,
- device: cuda,
- epochs: 150,
- `N=800`, `Q=60`, `T=60`,
- `K=2` and `K=8`,
- seeds: `0 1 2 3 4`,
- learning rates: `0.003`, `0.01`, `0.03`,
- total fits: 330.

Extension outputs:

```text
deep_irt/bench/outputs/alpha_map_kext_results.json
deep_irt/bench/outputs/alpha_map_kext_table.md
```

Combined best generic non-exp comparison:

| K | Exp best LR | Exp alpha Spearman | Best generic non-exp | Generic alpha Spearman | Delta | High-alpha delta |
|---:|---:|---:|---|---:|---:|---:|
| 2 | 0.003 | 0.808+-0.024 | temperature_softplus | 0.809+-0.031 | -0.001 | -0.024 |
| 4 | 0.01 | 0.928+-0.015 | softplus_eps | 0.926+-0.017 | +0.002 | -0.033 |
| 8 | 0.01 | 0.948+-0.019 | temperature_softplus | 0.948+-0.014 | -0.001 | +0.017 |

This eliminates the proposed category-count rescue for the exp-only claim. The
exp margin does not grow with `K` under the current controlled protocol.

### Interpretation

The strict K=2,4,8 controls do not support a strong claim that exponential
geometry is uniquely responsible for alpha recovery. They support a weaker
claim: smooth, non-degenerate positive maps behave similarly once effective
initialization and LR tuning are controlled, while raw/ReLU-like maps are less
stable. Exp remains among the best overall alpha-recovery maps, but the margin
is too small and too inconsistent to use as load-bearing evidence.

### Decision

mixed.

### Next Action

Reframe the result around smooth positive-map geometry, not exp optimality.
Update any write-up language and the E7 design before running an optimizer
geometry diagnostic.

## E6: Initialization and Learning-Rate Controls

### Expectation

An exp advantage that survives matched effective initialization and per-map LR
tuning is stronger evidence for geometry than for favorable initialization.

### Setup

Planned controls:

- matched effective alpha initialization,
- LR sweep per positive map,
- gradient-norm matched training,
- effective alpha range tracking,
- clipped exp,
- scaled sigmoid,
- temperature softplus.

### Reality

The alpha-map API supports the constants required for the planned controls:
`scale`, `temperature`, `epsilon`, `clip_min`, and `clip_max`. The old
`alpha_log_scale` compatibility path remains intact and still drives the default
decoupled exp configuration.

Matched effective-alpha initialization is implemented in
`deep_irt/bench/run_alpha_map_bench.py`:

- `raw_alpha_value_for_map(...)` computes the inverse-map raw bias.
- `set_matched_alpha_init(...)` zeroes alpha-head weights and sets the static
  and state-conditioned alpha biases so all maps start at the same effective
  alpha.
- The default target is `alpha_init=0.5`, which is reachable for sigmoid and the
  other bounded or positive maps.

The runner also implements the per-map LR sweep. Full mode defaults to
`lrs=[0.003, 0.01, 0.03]`; quick mode defaults to `lrs=[0.01]`. The table reports
all LR cells and separately reports the pre-declared best-LR view.

Gradient monitoring is now present through the existing `DeepIRTModel.fit`
callback. The bench wrapper passes the callback through, and the runner records
alpha-head, decoder, and total gradient-norm summaries.

The full K=4 LR sweep is now complete. The best LR selected by alpha Spearman is
`0.01` for every map. This means the comparison is not currently being driven by
one map needing a different LR from the others.

After matched effective-alpha initialization and LR tuning, the exp-specific
claim weakens across the full K extension:

- at `K=2`, exp reaches `0.808+-0.024`, while temperature_softplus reaches
  `0.809+-0.031`,
- at `K=4`, exp reaches `0.928+-0.015`, while softplus_eps reaches
  `0.926+-0.017`,
- at `K=8`, exp reaches `0.948+-0.019`, while temperature_softplus reaches
  `0.948+-0.014`,
- exp and clipped-exp remain tied, so clipping is inactive under the current
  ranges,
- sigmoid and scaled sigmoid do not collapse, but bounded alpha range changes
  high-alpha behavior,
- identity and ReLU remain weaker or less stable, especially at low `K` and high
  LR.

### Interpretation

The control did its job. It separates an exp-only story from a broader smooth
positive-map story. Across `K=2,4,8`, LR tuning does not rescue a strong
exp-specific advantage.

### Decision

mixed.

### Next Action

Before E7, rewrite the geometry diagnostic. The target is no longer "why exp
wins"; it is "why smooth positive maps cluster and why raw/ReLU-like maps are
less stable under this finite-time recovery protocol."

## E7: Geometry-Matched Control

### Expectation

If exp helps through induced preconditioning, an alpha-space update with an
`alpha^2` preconditioner should partially reproduce the exp result.

### Setup

Original planned conditions:

```text
alpha = exp(a)
direct alpha with positivity projection
direct alpha with exp-equivalent preconditioner
direct alpha with softplus-equivalent preconditioner
```

Revised setup after E5/E6:

```text
direct alpha with projection
direct alpha with exp-equivalent preconditioner
direct alpha with softplus-equivalent preconditioner
direct alpha with scaled-softplus-equivalent preconditioner
direct alpha with bounded-sigmoid-equivalent preconditioner
raw/ReLU-like low-smoothness controls
```

### Reality

It is now gated by E5/E6. The positive-map ablation did not leave a stable
exp-specific advantage, so an exp-equivalent preconditioner control would answer
the wrong question unless the experiment is rewritten.

A no-training E7 geometry diagnostic has been implemented and run:

```text
conda run -n research python -m deep_irt.bench.analyze_alpha_map_geometry
```

Code and outputs:

```text
deep_irt/bench/analyze_alpha_map_geometry.py
deep_irt/tests/test_alpha_map_geometry.py
deep_irt/bench/outputs/alpha_map_geometry_summary.json
deep_irt/bench/outputs/alpha_map_geometry_summary.md
```

The diagnostic reads the completed K=2,4,8 alpha-map outputs and computes:

- map families,
- induced preconditioner values `m_g(alpha_init)`, `m_g(alpha_p50)`, and
  `m_g(alpha_p95)`,
- best generic smooth-map comparisons,
- family summaries,
- within-K rank correlations between recovery and `m_g(alpha_p95)`,
  effective-alpha range, and alpha-head gradient norms.

Key diagnostic numbers:

| K | Exp alpha Spearman | Best generic smooth map | Generic alpha Spearman | Delta |
|---:|---:|---|---:|---:|
| 2 | 0.808 | temperature_softplus | 0.809 | -0.001 |
| 4 | 0.928 | softplus_eps | 0.926 | +0.002 |
| 8 | 0.948 | temperature_softplus | 0.948 | -0.001 |

Correlation of alpha recovery with `m_g(alpha_p95)`:

| K | Spearman |
|---:|---:|
| 2 | -0.119 |
| 4 | +0.183 |
| 8 | -0.101 |

This says preconditioner magnitude alone does not explain the recovery ordering
across maps. The direct-alpha control, if run, must test smoothness,
saturation, monotonicity, and effective-alpha range alongside preconditioning.

The direct-alpha optimizer control has now been implemented and run. This is the
stricter identifiability simplification from the plan:

```text
Use the same synthetic GPCM generator.
Freeze theta to the true learner ability used at each response.
Freeze beta to the true item thresholds.
Optimize only one scalar alpha_j per item from response-level GPCM likelihood.
Do not use parameter-recovery loss.
Use the same train split and response-level GPCM likelihood.
Initialize all alpha_j to the same effective alpha.
Compare alpha-space update rules under the same LR grid.
Score alpha recovery without latent sign alignment because theta and beta are fixed.
```

Code and outputs:

```text
deep_irt/bench/run_direct_alpha_geometry.py
deep_irt/bench/analyze_direct_alpha_geometry.py
deep_irt/tests/test_direct_alpha_geometry.py
deep_irt/bench/outputs/direct_alpha_geometry_results.json
deep_irt/bench/outputs/direct_alpha_geometry_table.md
deep_irt/bench/outputs/direct_alpha_geometry_lrext_results.json
deep_irt/bench/outputs/direct_alpha_geometry_lrext_table.md
deep_irt/bench/outputs/direct_alpha_geometry_lrext30_results.json
deep_irt/bench/outputs/direct_alpha_geometry_lrext30_table.md
deep_irt/bench/outputs/direct_alpha_geometry_summary.json
deep_irt/bench/outputs/direct_alpha_geometry_summary.md
```

Commands:

```text
conda run -n research python -m deep_irt.bench.run_direct_alpha_geometry --device cuda
conda run -n research python -m deep_irt.bench.run_direct_alpha_geometry --device cuda --lrs 3 10 --out-prefix direct_alpha_geometry_lrext
conda run -n research python -m deep_irt.bench.run_direct_alpha_geometry --device cuda --lrs 30 --out-prefix direct_alpha_geometry_lrext30
conda run -n research python -m deep_irt.bench.analyze_direct_alpha_geometry
```

Combined best results:

| K | Best direct-alpha condition | Best LR | Alpha Spearman | High-alpha Spearman | Recovery AUC | At max LR |
|---:|---|---:|---:|---:|---:|---|
| 2 | projected | 3.0 | 0.932 | 0.779 | 0.894 | no |
| 4 | exp_precond | 3.0 | 0.963 | 0.790 | 0.942 | no |
| 8 | square_precond | 30.0 | 0.969 | 0.841 | 0.966 | yes |

Family-level summary:

| K | Identity alpha-space mean | Smooth positive preconditioner mean | Square preconditioner mean | Bounded preconditioner mean |
|---:|---:|---:|---:|---:|
| 2 | 0.932 | 0.930 | 0.931 | 0.918 |
| 4 | 0.962 | 0.962 | 0.962 | 0.961 |
| 8 | 0.969 | 0.969 | 0.969 | 0.968 |

The first full run had all best learning rates at the upper grid edge, so it was
not treated as decisive. After extending the grid to `3`, `10`, and `30`, the
central direct-alpha conditions mostly converge to the same recovery band. The
bounded sigmoid condition remains lower when its range cap constrains alpha;
that is a range restriction, not an exp advantage.

### Interpretation

The direct-alpha control refutes the alpha-space preconditioner-only mechanism.
When encoder learning, beta learning, theta drift, item embeddings, and
representation capacity are removed, the smooth-map cluster from the neural
experiment does not reappear as a meaningful separation among alpha-space update
rules.

The result does not erase the neural positive-map result. It relocates the
mechanism: the weak raw/ReLU-like behavior in E5/E6 is more likely tied to
representation learning, optimizer interaction, finite-time saturation, or the
way raw-map geometry interacts with neural heads, not to scalar alpha-space
preconditioner magnitude alone.

### Decision

refutes claim.

### Next Action

Stop using the preconditioner-only explanation as a live claim. The next strict
test should stay inside the neural model and isolate representation/optimizer
interactions: fixed encoder versus learned encoder, frozen item embeddings
versus learned item embeddings, and raw/ReLU maps with matched output range and
gradient clipping.

## E7a: Neural Map-Isolation Control

### Expectation

After E7, the remaining question is whether the neural positive-map behavior is
caused by representation learning, item-key learning, raw/ReLU range instability,
or optimizer interaction. If clipped raw/ReLU plus gradient clipping closes the
gap, the earlier weakness was mostly range or optimizer instability. If frozen
item embeddings erase the map differences, the item-key pathway is central. If
freezing the sequence backbone changes little, the LSTM state is less central
than the item representation.

### Setup

Implemented controls:

```text
learned_all
frozen_backbone
frozen_item_embeddings
frozen_encoder

exp
softplus
identity
relu
clipped_identity with clip_min=-2, clip_max=2
clipped_relu with clip_min=0, clip_max=2
```

The clipped raw/ReLU maps are explicit decoder maps. Gradient clipping is
default-off in `DeepIRTModel.fit` and is enabled only for the clipped raw/ReLU
stability controls in this runner.

Code and smoke outputs:

```text
deep_irt/bench/run_neural_map_isolation.py
deep_irt/tests/test_neural_map_isolation.py
deep_irt/bench/outputs/neural_map_isolation_smoke_results.json
deep_irt/bench/outputs/neural_map_isolation_smoke_table.md
deep_irt/bench/outputs/neural_map_isolation_results.json
deep_irt/bench/outputs/neural_map_isolation_table.md
deep_irt/bench/outputs/neural_map_isolation_kext_results.json
deep_irt/bench/outputs/neural_map_isolation_kext_table.md
```

Smoke command:

```text
conda run -n research python -m deep_irt.bench.run_neural_map_isolation --quick --device cpu --epochs 3 --n-learners 30 --n-items 8 --seq-len 8 --maps softplus clipped_relu --representations learned_all frozen_item_embeddings --seeds 0 --lrs 0.01 --out-prefix neural_map_isolation_smoke
```

### Reality

The smoke run completed and wrote outputs. It is intentionally too small to
interpret mechanistically:

| representation | map | alpha Spearman | theta Spearman | beta Spearman |
|---|---|---:|---:|---:|
| learned_all | softplus | 0.167 | 0.676 | 0.808 |
| learned_all | clipped_relu | 0.204 | 0.673 | 0.827 |
| frozen_item_embeddings | softplus | 0.167 | 0.668 | 0.813 |
| frozen_item_embeddings | clipped_relu | 0.204 | 0.666 | 0.797 |

The smoke only proves that the new controls execute, write artifacts, and report
the planned grouped table. It does not support or refute the mechanism.

The focused full K=4 grid has now completed:

```text
conda run -n research python -m deep_irt.bench.run_neural_map_isolation --device cuda --K 4 --seeds 0 1 2 3 4
```

Outputs:

```text
deep_irt/bench/outputs/neural_map_isolation_results.json
deep_irt/bench/outputs/neural_map_isolation_table.md
```

Best alpha Spearman by representation:

| representation | exp | softplus | best raw/ReLU | raw/ReLU minus softplus | theta Spearman pattern |
|---|---:|---:|---|---:|---|
| learned_all | 0.928 | 0.924 | clipped_identity 0.907 | -0.017 | all near 0.955 to 0.960 |
| frozen_backbone | 0.415 | 0.390 | relu 0.134 | -0.255 | unstable, mean near 0.03 to 0.11 |
| frozen_item_embeddings | 0.844 | 0.811 | clipped_relu 0.819 | +0.009 | all near 0.957 to 0.959 |
| frozen_encoder | 0.191 | 0.197 | clipped_identity 0.160 | -0.037 | fixed poor mean near -0.109 |

Important side metrics:

| representation | map | beta Spearman | alpha p95/max | alpha gradient |
|---|---|---:|---:|---:|
| learned_all | exp | 0.988 | 1.301/1.893 | 0.033 |
| learned_all | softplus | 0.988 | 1.275/1.779 | 0.024 |
| learned_all | clipped_identity | 0.963 | 1.369/1.799 | 0.067 |
| learned_all | clipped_relu | 0.980 | 1.331/1.769 | 0.057 |
| frozen_item_embeddings | exp | 0.968 | 1.908/2.797 | 0.035 |
| frozen_item_embeddings | softplus | 0.961 | 1.684/2.275 | 0.025 |
| frozen_item_embeddings | clipped_relu | 0.913 | 1.570/1.862 | 0.060 |

The K=2/8 extension has also completed:

```text
conda run -n research python -m deep_irt.bench.run_neural_map_isolation --device cuda --K 2 8 --seeds 0 1 2 3 4 --out-prefix neural_map_isolation_kext
```

Outputs:

```text
deep_irt/bench/outputs/neural_map_isolation_kext_results.json
deep_irt/bench/outputs/neural_map_isolation_kext_table.md
```

Learned-all map deltas across category complexity:

| K | exp | softplus | best raw/ReLU | raw/ReLU minus softplus | interpretation |
|---:|---:|---:|---|---:|---|
| 2 | 0.746 | 0.753 | clipped_relu 0.652 | -0.100 | raw/ReLU still weak; softplus slightly beats exp |
| 4 | 0.928 | 0.924 | clipped_identity 0.907 | -0.017 | raw/ReLU gap persists but smaller |
| 8 | 0.948 | 0.947 | identity 0.941 | -0.006 | near-ceiling recovery; gap shrinks |

Frozen item-embedding comparison:

| K | exp | softplus | best raw/ReLU | raw/ReLU minus softplus | note |
|---:|---:|---:|---|---:|---|
| 2 | 0.683 | 0.658 | relu 0.537 | -0.121 | freezing items does not rescue raw/ReLU |
| 4 | 0.844 | 0.811 | clipped_relu 0.819 | +0.009 | raw/ReLU closes alpha gap but beta recovery is lower |
| 8 | 0.918 | 0.906 | identity 0.897 | -0.009 | near-ceiling but still below smooth maps |

Frozen-backbone and frozen-encoder controls remain weak across K. Their best
smooth-map alpha Spearman is far below learned-all recovery, and theta recovery
is unstable or poor. This confirms that the neural representation is part of the
measurement mechanism, not only nuisance capacity.

### Interpretation

This is no longer a smoke-only entry. The K=2,4,8 grids give four useful facts.

First, clipped raw/ReLU plus gradient clipping does not explain away the
learned-all gap. In the normal learned neural model, the best raw/ReLU variant
remains below softplus by `0.100` at `K=2`, `0.017` at `K=4`, and `0.006` at
`K=8`.

Second, frozen item embeddings do not give a clean rescue. They narrow the
raw/ReLU versus softplus difference at `K=4`, but lower recovery overall and
damage beta recovery for raw/ReLU. At `K=2` they leave a large raw/ReLU deficit,
and at `K=8` they remain near ceiling but still slightly below smooth maps.

Third, freezing the sequence backbone or the full encoder breaks theta and alpha
recovery badly. The neural representation is not an incidental nuisance; it is
part of the recovery mechanism.

Fourth, the gap is category-dependent. It is largest in the binary/2PL-like
setting, moderate at `K=4`, and small near the high-recovery `K=8` regime. This
does not support an exp-only claim; it supports a weaker smooth-map stability
claim that depends on the neural representation and the recovery regime.

The current mechanism wording should be: direct alpha-space preconditioning
alone is insufficient; range clipping alone is insufficient; exp is not uniquely
preferred; the remaining effect is a neural representation and optimizer
interaction, with smooth positive maps more stable than raw/ReLU maps.

### Decision

mixed. Supports the revised representation-interaction and smooth-map stability
mechanism across `K=2,4,8`; refutes the range-only rescue and exp-only wording.

### Next Action

Treat E7a as complete for the positive-map mechanism layer. Move to E8 only
after preserving the wording above in any paper or study summary:

```text
smooth positive-map stability under neural representation learning,
not exp optimality and not scalar alpha-space preconditioning alone.
```

## E8: Contextual Alpha Residual Diagnostics

### Expectation

The contextual residual `delta[j,t]` should carry some state-dependent
information, but it may also contain finite-data or Fisher-tail artifacts.

### Setup

Existing references:

- `docs/LEARNING_DYNAMICS_STUDY.md`, especially the state-conditioned
  discrimination diagnostics.
- Existing scripts mentioned there:
  - `deep_irt/bench/run_adynamic_probe.py`
  - `deep_irt/bench/run_adyn_theta_relation.py`
  - `deep_irt/bench/run_phase2_signal.py`
  - `deep_irt/bench/run_phase2_scale.py`

Structured null-control code and outputs:

```text
deep_irt/bench/run_alpha_residual_null.py
deep_irt/tests/test_alpha_residual_null.py
deep_irt/bench/outputs/alpha_residual_null_smoke_results.json
deep_irt/bench/outputs/alpha_residual_null_smoke_table.md
deep_irt/bench/outputs/alpha_residual_null_results.json
deep_irt/bench/outputs/alpha_residual_null_table.md
deep_irt/bench/outputs/phase2_signal.json
deep_irt/bench/outputs/phase2_signal_table.md
deep_irt/bench/outputs/phase2_scale.json
deep_irt/bench/outputs/phase2_scale_table.md
```

### Reality

Existing notes report directional detection, not calibrated magnitude. They also
report strong attenuation and contamination by low-ability or low-Fisher regions.

The E8 null control has now been run on static-alpha synthetic data, where the
true contextual residual is exactly zero. The fitted residual is:

```text
log alpha_hat[j,t] = a_j + delta[j,t]
E_t[delta[j,t] | j] = 0
```

Smoke command:

```text
conda run -n research python -m deep_irt.bench.run_alpha_residual_null --quick --device cpu --N 24 --seeds 0 --epochs 2 --n-items 6 --seq-len 8 --out-prefix alpha_residual_null_smoke
```

Full null command:

```text
conda run -n research python -m deep_irt.bench.run_alpha_residual_null --device cuda
```

Full null summary:

| K | N | static log-alpha Spearman | delta std | strongest null artifact | max abs corr | max cubic R2 |
|---:|---:|---:|---:|---|---:|---:|
| 4 | 200 | 0.787+-0.053 | 0.578+-0.133 | info_model | 0.599 | 0.089 |
| 4 | 800 | 0.926+-0.012 | 0.721+-0.014 | info_model | 0.698 | 0.022 |
| 4 | 3200 | 0.936+-0.016 | 0.583+-0.036 | info_model | 0.696 | 0.173 |

Key variable correlations:

| K | N | corr(delta, info_model) | corr(delta, theta_true) | corr(delta, history_pos) | corr(delta, exposure_count) |
|---:|---:|---:|---:|---:|---:|
| 4 | 200 | -0.599 | 0.126 | 0.169 | 0.008 |
| 4 | 800 | -0.698 | 0.014 | 0.202 | 0.002 |
| 4 | 3200 | -0.696 | 0.045 | 0.203 | 0.006 |

The artifact does not shrink cleanly with more learners. Static item-level
discrimination recovery improves with N, but the within-item contextual residual
still carries substantial structure in a dataset where the true residual is
zero.

The matched planted-signal detector has also been refreshed:

```text
conda run -n research python -m deep_irt.bench.run_phase2_signal --K 4 --sigmas 0.2 0.4 --seeds 0 1 2 --N 2000 --Q 60 --T 60 --epochs 150 --device cuda
conda run -n research python -m deep_irt.bench.analyze_phase2_signal
```

Signal summary:

| planted sigma | corr(planted slope, gamma) | corr(null-subtracted signal, gamma) | corr(null, gamma) | calibration k | null std |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.427 | 0.438 | -0.040 | 0.039 | 0.015 |
| 0.40 | 0.649 | 0.666 | -0.040 | 0.040 | 0.015 |

The magnitude attenuation probe has also been rerun and now writes persistent
outputs:

```text
conda run -n research python -m deep_irt.bench.run_phase2_scale
```

Magnitude summary for `sigma=0.4`:

| mean raw calibration k | mean theta scale c | mean k/c | corr(theta_hat, theta0) |
|---:|---:|---:|---:|
| 0.038 | 1.137 | 0.034 | 0.962 |

Because theta scale is near one, the small calibration slope is not explained by
latent theta compression. The residual detects the planted direction, but its
magnitude is strongly attenuated.

### Interpretation

The residual is useful as a diagnostic, but not as a calibrated measurement
without additional constraints. The null control says that learned contextual
alpha residuals can track model uncertainty or local information even when true
alpha is static. Any planted-signal or real-data analysis must therefore be
interpreted relative to this null artifact, not as raw residual structure.

The planted detector supports directional validity after matched-null
subtraction, especially at `sigma=0.4`, but calibration is far from one. This is
exactly the conservative E8 framing: directional detection, not magnitude.

### Decision

mixed.

### Next Action

Treat E8 as complete for the minimum mechanism study. Any write-up should state:

```text
Contextual alpha residuals can directionally detect planted theta-dependent
discrimination after matched-null subtraction, but static-alpha null artifacts
are substantial and magnitude is not calibrated.
```

Do not move to real-data residual interpretation until this null-artifact
paragraph is included.

## E9: Misspecification Studies

### Expectation

Under misspecification, alpha may absorb slope-like violations, local
dependence, response style, or threshold noise.

### Setup

Planned misspecifications:

- generate with GRM and fit GPCM,
- generate with GPCM and fit NRM,
- local dependence on previous response,
- learner response styles,
- item exposure imbalance,
- drifting theta,
- noisy thresholds,
- differential item functioning,
- threshold disorder.

First implemented probe:

```text
deep_irt/bench/run_misspecification_probe.py
deep_irt/tests/test_misspecification_probe.py
deep_irt/bench/outputs/misspec_localdep_smoke_results.json
deep_irt/bench/outputs/misspec_localdep_smoke_table.md
deep_irt/bench/outputs/misspec_localdep_results.json
deep_irt/bench/outputs/misspec_localdep_table.md
```

Local-dependence generator:

```text
Use the same synthetic GPCM ground truth, item sequences, train/validation
split, and response-draw uniforms across strengths.
At each step t > 0, add a logit bonus to the previous response category.
Fit a GPCM anyway, so the local-dependence channel is misspecified.
```

Model variants:

```text
static: static alpha and static beta
state_alpha: state-conditioned alpha, static beta
state_beta: static alpha, state-conditioned beta
```

Smoke command:

```text
conda run -n research python -m deep_irt.bench.run_misspecification_probe --quick --device cpu --N 24 --n-items 6 --seq-len 8 --epochs 2 --strengths 0 1 --variants static state_alpha state_beta --seeds 0 --out-prefix misspec_localdep_smoke
```

Full command:

```text
conda run -n research python -m deep_irt.bench.run_misspecification_probe --device cuda
```

Full setup:

- `K=4`, `N=800`, `Q=60`, `T=60`,
- strengths: `0.0`, `0.5`, `1.0`,
- seeds: `0`, `1`, `2`,
- epochs: `150`,
- learning rate: `0.01`,
- total fits: `27`,
- elapsed: `107.9s`.

Second implemented probe:

```text
deep_irt/bench/outputs/misspec_noisy_thresholds_smoke_results.json
deep_irt/bench/outputs/misspec_noisy_thresholds_smoke_table.md
deep_irt/bench/outputs/misspec_noisy_thresholds_results.json
deep_irt/bench/outputs/misspec_noisy_thresholds_table.md
```

Noisy-threshold generator:

```text
Use the same synthetic GPCM ground truth, item sequences, train/validation
split, and response-draw uniforms across strengths.
At each occurrence, add zero-mean Gaussian jitter to the item's thresholds,
then sort the threshold vector before sampling the response.
Fit a static-threshold GPCM anyway, so occurrence-level threshold noise is
misspecified but threshold disorder is not yet introduced.
```

Smoke command:

```text
conda run -n research python -m deep_irt.bench.run_misspecification_probe --quick --misspecification noisy_thresholds --device cpu --N 24 --n-items 6 --seq-len 8 --epochs 2 --strengths 0 0.5 --variants static state_alpha state_beta --seeds 0 --out-prefix misspec_noisy_thresholds_smoke
```

Full command:

```text
conda run -n research python -m deep_irt.bench.run_misspecification_probe --misspecification noisy_thresholds --device cuda
```

Full setup:

- `K=4`, `N=800`, `Q=60`, `T=60`,
- threshold noise sigmas: `0.0`, `0.25`, `0.5`,
- seeds: `0`, `1`, `2`,
- epochs: `150`,
- learning rate: `0.01`,
- total fits: `27`,
- elapsed: `107.5s`.

Third implemented probe:

```text
deep_irt/bench/outputs/misspec_response_style_smoke_results.json
deep_irt/bench/outputs/misspec_response_style_smoke_table.md
deep_irt/bench/outputs/misspec_response_style_results.json
deep_irt/bench/outputs/misspec_response_style_table.md
```

Learner response-style generator:

```text
Use the same synthetic GPCM ground truth, item sequences, train/validation
split, and response-draw uniforms across strengths.
Draw one stable learner style score per learner.
Positive style favors extreme categories; negative style favors middle
categories, independent of item content and true ability.
Fit a GPCM with no explicit response-style parameter.
```

Smoke command:

```text
conda run -n research python -m deep_irt.bench.run_misspecification_probe --quick --misspecification learner_response_style --device cpu --N 24 --n-items 6 --seq-len 8 --epochs 2 --strengths 0 1 --variants static state_alpha state_beta --seeds 0 --out-prefix misspec_response_style_smoke
```

Full command:

```text
conda run -n research python -m deep_irt.bench.run_misspecification_probe --misspecification learner_response_style --device cuda
```

Full setup:

- `K=4`, `N=800`, `Q=60`, `T=60`,
- learner style logit scales: `0.0`, `0.5`, `1.0`,
- seeds: `0`, `1`, `2`,
- epochs: `150`,
- learning rate: `0.01`,
- total fits: `27`,
- elapsed: `92.1s`.

### Reality

The local-dependence manipulation worked. The observed repeated-response rate
rose from `0.349` at strength `0.0`, to `0.451` at strength `0.5`, to `0.560`
at strength `1.0`.

Aggregate readout:

| strength | variant | QWK | theta Spearman | alpha Spearman | beta Spearman | alpha delta std | corr(delta, prev) | corr(delta, info) | beta delta std | corr(beta, prev) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | static | 0.543 | 0.848 | 0.927 | 0.988 | 0.000 | -0.006 | 0.010 | 0.000 | 0.002 |
| 0.0 | state_alpha | 0.574 | 0.897 | 0.913 | 0.982 | 0.884 | 0.117 | -0.757 | 0.000 | 0.000 |
| 0.0 | state_beta | 0.519 | 0.883 | 0.924 | 0.985 | 0.000 | -0.002 | 0.032 | 0.614 | -0.485 |
| 0.5 | static | 0.554 | 0.861 | 0.910 | 0.986 | 0.000 | -0.004 | -0.024 | 0.000 | 0.001 |
| 0.5 | state_alpha | 0.592 | 0.878 | 0.893 | 0.980 | 0.845 | 0.107 | -0.771 | 0.000 | 0.003 |
| 0.5 | state_beta | 0.544 | 0.891 | 0.898 | 0.984 | 0.000 | -0.003 | 0.062 | 0.597 | -0.549 |
| 1.0 | static | 0.579 | 0.839 | 0.874 | 0.977 | 0.000 | 0.004 | 0.019 | 0.000 | -0.005 |
| 1.0 | state_alpha | 0.640 | 0.871 | 0.803 | 0.961 | 0.952 | 0.105 | -0.797 | 0.000 | 0.001 |
| 1.0 | state_beta | 0.595 | 0.879 | 0.863 | 0.983 | 0.000 | -0.000 | -0.009 | 0.568 | -0.612 |

Main observations:

- State-conditioned alpha gives the best prediction under local dependence:
  QWK rises from `0.574` at the matched strength-zero control to `0.640` at
  strength `1.0`.
- The prediction gain is not a recovery gain. In the same state-alpha model,
  alpha recovery drops from `0.913` to `0.803`, and beta recovery drops from
  `0.982` to `0.961`.
- Static alpha also loses recovery as local dependence strengthens:
  alpha Spearman drops from `0.927` to `0.874`.
- The E8 artifact repeats here. In the state-alpha model, the strongest alpha
  residual correlate is still `info_model`, not previous response. The
  `corr(delta, info_model)` magnitude is large even at strength zero
  (`-0.757`) and becomes only slightly larger at strength one (`-0.797`).
- Direct previous-response correlation in alpha residuals does not strengthen
  with the manipulation. It is `0.117` at strength zero and `0.105` at strength
  one for state-alpha.
- Dynamic beta shows contextual beta variation even under the correctly
  specified strength-zero control: beta delta std is `0.614` and
  `corr(beta, prev)=-0.485`. Under strength one, this becomes beta delta std
  `0.568` and `corr(beta, prev)=-0.612`.
- Dynamic beta therefore carries a clearer previous-response signature than
  alpha, but this is also contaminated by a large strength-zero contextual beta
  artifact.

The noisy-threshold manipulation also ran to completion. Because the threshold
noise is zero mean and sorted, it does not materially change repeated-response
rate. The repeated-response rate stays near `0.34`.

Noisy-threshold aggregate readout:

| sigma | variant | QWK | theta Spearman | alpha Spearman | beta Spearman | alpha delta std | corr(delta, prev) | corr(delta, info) | beta delta std | corr(beta, prev) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | static | 0.528 | 0.828 | 0.927 | 0.988 | 0.000 | -0.006 | -0.042 | 0.000 | -0.001 |
| 0.00 | state_alpha | 0.578 | 0.888 | 0.902 | 0.985 | 0.909 | 0.079 | -0.754 | 0.000 | 0.002 |
| 0.00 | state_beta | 0.522 | 0.871 | 0.920 | 0.980 | 0.000 | -0.002 | 0.006 | 0.660 | -0.492 |
| 0.25 | static | 0.520 | 0.832 | 0.918 | 0.985 | 0.000 | 0.004 | -0.018 | 0.000 | 0.003 |
| 0.25 | state_alpha | 0.585 | 0.889 | 0.904 | 0.979 | 0.956 | 0.117 | -0.775 | 0.000 | -0.001 |
| 0.25 | state_beta | 0.533 | 0.878 | 0.926 | 0.976 | 0.000 | -0.002 | 0.023 | 0.659 | -0.488 |
| 0.50 | static | 0.509 | 0.804 | 0.895 | 0.976 | 0.000 | 0.004 | -0.088 | 0.000 | -0.003 |
| 0.50 | state_alpha | 0.572 | 0.871 | 0.877 | 0.974 | 0.912 | 0.083 | -0.773 | 0.000 | -0.002 |
| 0.50 | state_beta | 0.505 | 0.870 | 0.908 | 0.967 | 0.000 | 0.000 | -0.011 | 0.664 | -0.487 |

Main noisy-threshold observations:

- Static item recovery degrades as threshold noise grows. Alpha Spearman drops
  from `0.927` to `0.895`, and beta Spearman drops from `0.988` to `0.976`.
- State-alpha remains the best predictor, but it does not gain from the
  threshold violation. QWK is `0.578` at sigma zero and `0.572` at sigma `0.5`.
- State-alpha item recovery also worsens at sigma `0.5`: alpha Spearman drops
  from `0.902` to `0.877`, and beta Spearman drops from `0.985` to `0.974`.
- The state-alpha residual is again dominated by `info_model`: correlation is
  `-0.754` at sigma zero, `-0.775` at sigma `0.25`, and `-0.773` at sigma
  `0.5`. This is not a clean threshold-noise detector.
- Dynamic beta does not rescue the threshold-noise condition. It has large
  contextual beta variation already at sigma zero (`0.660`) and stays large at
  sigma `0.5` (`0.664`), while beta recovery drops from `0.980` to `0.967`.

The learner response-style manipulation also ran to completion. It increases
the repeated-response rate from `0.345` to `0.442` and the extreme-category
rate from `0.451` to `0.471`.

Response-style aggregate readout:

| style scale | variant | QWK | theta Spearman | alpha Spearman | beta Spearman | alpha delta std | corr(delta, style) | corr(delta, info) | beta delta std | corr(beta, style) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | static | 0.531 | 0.837 | 0.914 | 0.988 | 0.000 | -0.004 | -0.028 | 0.000 | -0.002 |
| 0.0 | state_alpha | 0.578 | 0.891 | 0.906 | 0.985 | 0.905 | 0.005 | -0.750 | 0.000 | 0.001 |
| 0.0 | state_beta | 0.533 | 0.887 | 0.929 | 0.984 | 0.000 | 0.002 | 0.084 | 0.652 | -0.017 |
| 0.5 | static | 0.516 | 0.816 | 0.913 | 0.987 | 0.000 | -0.001 | -0.026 | 0.000 | 0.001 |
| 0.5 | state_alpha | 0.571 | 0.868 | 0.879 | 0.974 | 1.002 | -0.346 | -0.779 | 0.000 | -0.002 |
| 0.5 | state_beta | 0.512 | 0.868 | 0.882 | 0.988 | 0.000 | 0.000 | -0.051 | 0.561 | -0.039 |
| 1.0 | static | 0.516 | 0.802 | 0.866 | 0.981 | 0.000 | 0.000 | 0.050 | 0.000 | -0.002 |
| 1.0 | state_alpha | 0.586 | 0.835 | 0.773 | 0.920 | 1.849 | -0.665 | -0.788 | 0.000 | 0.000 |
| 1.0 | state_beta | 0.530 | 0.878 | 0.822 | 0.986 | 0.000 | -0.003 | -0.022 | 0.456 | -0.075 |

Main response-style observations:

- This is the first E9 probe where the planted nuisance appears clearly in the
  contextual alpha residual. In the state-alpha model, `corr(delta, style)` is
  `0.005` at scale zero, `-0.346` at scale `0.5`, and `-0.665` at scale `1.0`.
- This is still not a psychometric success. In the same state-alpha model,
  alpha recovery drops from `0.906` to `0.773`, and beta recovery drops from
  `0.985` to `0.920`.
- Prediction does not show a decisive gain from the response-style violation:
  state-alpha QWK is `0.578` at scale zero and `0.586` at scale `1.0`.
- Static alpha is less expressive, but its recovery also worsens as style grows:
  alpha Spearman drops from `0.914` to `0.866`.
- Dynamic beta does not absorb the style cleanly. Its `corr(beta, style)` only
  reaches `-0.075` at scale `1.0`, while alpha recovery worsens from `0.929` to
  `0.822`.

### Interpretation

The first E9 result is a measurement-validity warning. The misspecified local
dependence makes response prediction easier because the history contains a real
Markov signal. The state-alpha model uses that extra flexibility for better
QWK, but item-parameter recovery worsens.

This does not support a clean claim that contextual alpha directly absorbs
previous-response local dependence. The alpha residual remains dominated by the
same information/uncertainty artifact found in E8, and the previous-response
correlation does not increase over the matched null. Dynamic beta shows a
stronger previous-response contextual pattern, but its strength-zero artifact is
already large.

The safe wording is:

```text
Under previous-response local dependence, state-conditioned alpha improves
prediction but degrades item recovery. The contextual alpha residual does not
cleanly isolate the local-dependence violation beyond the E8 null artifact.
Dynamic beta carries a stronger previous-response signal, but with substantial
null contextual beta noise.
```

The noisy-threshold result reinforces the same conservative conclusion from the
item-parameter side. Adding occurrence-level threshold noise mostly makes the
measurement problem noisier: recovery worsens, and prediction does not improve.
Neither contextual alpha nor dynamic beta cleanly absorbs the threshold noise
after comparing against the matched strength-zero control.

The response-style result adds an important nuance. Contextual alpha can track a
planted learner-level nuisance when the nuisance is slope/spread-like in the
ordinal categories. But this is nuisance absorption, not valid item
discrimination recovery: the style correlation strengthens exactly as alpha and
beta recovery degrade.

Together, the first three E9 probes say that flexible contextual heads can
improve prediction or express contextual variation, but those effects are not
equivalent to valid psychometric recovery under misspecification.

### Decision

mixed.

### Next Action

Keep local dependence, noisy thresholds, and learner response style as the first
three E9 controls. The next E9 probe should target threshold disorder or
differential item functioning before claiming a general misspecification
taxonomy.

## E10: GRM and NRM Extension

### Expectation

Slope-like parameters should behave like alpha. Intercepts and thresholds should
behave like beta.

### Setup

Existing NRM-related files and outputs:

- `deep_irt/core/nrm_ma_irt.py`
- `deep_irt/bench/nrm_engines.py`
- `deep_irt/bench/outputs/nrm_bench_table.md`
- `deep_irt/bench/outputs/nrm_bench_itemonly_table.md`
- temporary NRM probes in `deep_irt/bench/_nrm_*.py`

### Reality

Existing NRM evidence is preliminary and mixed. Some notes suggest
state-conditioned NRM slopes can help prediction while hurting item-parameter
recovery.

### Interpretation

NRM is valuable for the taxonomy but should not be load-bearing for the first
paper unless cleaned and replicated.

### Decision

mixed.

### Next Action

Treat as expanded thesis work. Do not let it delay the GPCM positive-map study.

## E11: Real-Data Stability

### Expectation

Real data can test prediction, calibration, stability, and agreement with
external references, but cannot prove true parameter recovery.

### Setup

Candidate datasets:

- ASSISTments,
- KDD Cup 2010,
- EdNet when appropriate,
- ordinal or partial-credit data,
- questionnaire-style data if compatible.

### Reality

Not yet organized for this specific alpha learning-dynamics study.

### Interpretation

Real data should be supporting evidence only. The core proof remains synthetic
and mechanistic.

### Decision

planned.

### Next Action

After E5 to E7, choose one dataset for split-half item-parameter stability and
calibration. Do not claim ground-truth recovery.
