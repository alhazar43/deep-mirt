# ACT-P0 fabrication defect: mechanism diagnosis

Status: DIAGNOSIS ONLY. No source file was edited. All experiments run
CPU-only, using `growth.active`/`growth.tracker` unmodified, via two
scratch scripts under `kt-mirt/scripts/probe/` (`_diag_act_p0_mechanism.py`,
`_diag_act_p0_fix_check.py`, both deletable). Context: the A4 pre-run
probe (`outputs/a4/probe_medium/probe_result.json`, `_diag`-adjacent
`scripts/probe/act_p0_fabrication_probe.py`) found ACT-P0 (`lambda_i`
pinned at 1) implying a ~0.20-0.30 population score-scale rise over
opportunities 1-10 on EVERY twin including SYN-NG (no growth), while
ACT-P1 (free amortized `lambda_i`) is correctly silent on SYN-NG
(0.00067-0.0024) and responsive on SYN-KG. Per `_planning/LEDGER.md`
(2026-07-18, "Probe verdict, campaign GO"), ACT-P0 is currently EXCLUDED
from the campaign as a pre-campaign revision, reversible via the
idempotent store; this document is the requested code-level diagnosis.

## 1. Summary verdict

**The fabrication is not a data confound, not an identifiability
problem, and not an implementation bug in the transition/gating logic.
It is a training-convergence pathology: `train_active`'s fixed 20-epoch
loop (`ActiveConfig.n_epochs` used as 20 everywhere ACT is actually run
-- `run.py`'s `RunConfig.act_epochs`, the probe's `ACT_EPOCHS`) is
roughly one to two orders of magnitude too short for Adam to move `g_c`
from its default initialization down to the near-zero region the CG1
silence bar demands, given the closed-form 10-opportunity extrapolation
that reads it out.** The same estimator code, given enough epochs from
the SAME default initialization, converges correctly in both
directions: silent on no-growth, and accurately recovering the true
gain on known-growth data. A cheap partial fix exists inside the frozen
ACT-P0 constraints (raise the Adam learning rate used for ACT training),
but it does not, by itself, robustly clear the design's per-learner p95
clause across seeds at the current epoch count; the full principled fix
is a convergence-gated stopping rule for `train_active`, mirroring the
one `bank.calibrate_bank` already uses in this same package. Section 6
gives the recommendation on repair vs. retirement.

## 2. Code trace (read-only)

Traced every place `lambda`, `g_c`, `M`, `z0` enter `active.py` /
`recognition.py` (`kt-mirt/src/kt_mirt/growth/active.py`,
`kt-mirt/src/kt_mirt/growth/recognition.py`) before any experiment was
run.

- **Transition** (`run_transition`, `active.py:185-218`): pre-update
  `Z` is gathered per tagged KC, the logit is `mean_z - b_j`, then
  `gain = lam * g_gathered * relu(M - z_tagged) * tag_mask`, scattered
  back via `scatter_add`. Padding slots (`tag_ids` clamped to 0,
  `tag_mask` False) contribute exactly zero gain and zero logit mass --
  no double counting, no index-collision leakage. Causal ordering is
  correct (state read before update). **No bug found here.**
- **`ActiveModel.forward`** (`active.py:167-182`): `lam_eff =
  torch.ones_like(u_i)` for P0 (`lam` is `None`, since
  `predict_lambda=False`), so P0's transition truly uses `lambda_i = 1`
  exactly, as pinned. `z0 = u_i + v_c`, `u_i` from
  `RecognitionNetwork`, `v_c` a free `(n_kcs,)` population parameter.
  **Matches the design's pin exactly; no bug found.**
- **`g_c = softplus(g_raw)`** (`active.py:164-165`), `g_raw =
  nn.Parameter(torch.zeros(n_kcs))` (`active.py:152`). At
  initialization, `g_c = softplus(0) = ln(2) approx 0.693` for EVERY
  KC, regardless of variant. This is the number that matters below.
- **`M` initialization** (`ceiling_init`, `active.py:103-108`): 95th
  percentile of calibrated `b_hat` plus 2, a deliberately generous
  headroom constant (design 2.3, judgment call 10). For the probe's
  scale this lands `M_init` around 3-4 (confirmed directly in the
  reproduction below).
- **`train_active`** (`active.py:242-257`): a bare `for _ in
  range(cfg.n_epochs): loss.backward(); optimizer.step()` loop. **No
  convergence check of any kind** -- no patience, no relative-loss
  threshold, no parameter-drift stop. `cfg.n_epochs` defaults to 30 in
  `ActiveConfig`, but every actual caller overrides it to 20
  (`run.py`'s `RunConfig.act_epochs = 20`, the probe's `ACT_EPOCHS =
  20`) at `lr = 0.05`.
- **Contrast**: `growth/bank.py`'s `calibrate_bank` (the OTHER neural
  fit in this same package) uses `BankModelConfig.n_epochs_max` (module
  default 300, though callers pass 40) PLUS `patience_epochs` (module
  default 3) and an explicit relative-NLL-change convergence test
  (`bank.py` docstring: "convergence = relative NLL change < 1e-4 over
  3 consecutive epochs plus a parameter-drift check", `bank.py:667-718`
  implements this). `train_active` has no analogous mechanism. This
  asymmetry, not any transition-logic bug, is the proximate
  implementation gap.
- **CG1 readout is a 10-step extrapolation, not a replay**
  (`implied_z_trajectory`/`implied_score_rise`, `active.py:276-318`,
  matching the design's stated interleaving-invariance). The transition
  is geometric: writing `gap_n = M - z_n`, the recurrence gives `gap_n
  = (1-g)^n * gap_0` (since `gain = g*gap` exactly makes the residual
  gap shrink by a factor `(1-g)` per opportunity, for `g` in the
  observed sub-1 range). At `g_c approx 0.69` (the untrained value),
  `(1-0.69)^10 approx 0.69*10^-5`, i.e. the state closes essentially
  ALL of the `M - z0` gap within 10 opportunities. This is why the
  readout is so sensitive: it needs `g_c` driven down roughly three
  orders of magnitude from its default init before the 10-step
  extrapolation stops looking like near-total saturation.

## 3. Ablation ladder (evidence)

All experiments: CPU only (`CUDA_VISIBLE_DEVICES=""`), `research` conda
env, `torch 2.7.1+cu126` installed but unused (device forced to
`"cpu"`), `KMP_DUPLICATE_LIB_OK=TRUE`. Two scripts, both under
`kt-mirt/scripts/probe/`:

- `_diag_act_p0_mechanism.py`: a minimal hand-built no-growth ("SYN-NG
  surrogate") dataset -- single item per KC (no bank-calibration
  noise), `z0_true_ic = xi_i + eta_c + noise_ic`, flat truth (every
  opportunity drawn from the SAME `z0_true_ic`), heterogeneous or
  homogeneous practice counts drawn from KDD-like anchors. Trains
  `ActiveModel(variant="act_p0")` via the UNMODIFIED `active.py`
  functions (`train_active`, `run_transition`, `implied_score_rise`),
  toggling one knob at a time against a probe-matched baseline
  (`hidden=16, emb=8, lr=0.05, epochs=20`, `n_kcs=12, n_learners=300`).
- `_diag_act_p0_fix_check.py`: adds an optional matched-family TRUE
  GROWTH channel (`z_{n+1} = z_n + g_true*(M_true - z_n)`, the SAME
  recurrence ACT itself uses, `g_true=0.15`) so growth-DETECTION can be
  checked, not just no-growth silence, plus scale-up and long-training
  runs.

### 3.1 Isolating suspects (a)/(b)/(c): none of them matter

| Arm | Knob changed vs. baseline | pop_mean_rise | p95_abs_rise | g_c (mean) | Verdict |
|---|---|---|---|---|---|
| A baseline | probe-matched: heterog. T, noise=0.3, fitted M, 20 epochs | 0.6447 | 0.8659 | 0.307 | fabricates |
| C | M FIXED at init (ceiling frozen) | 0.7081 | 0.9391 | 0.302 | fabricates, unchanged |
| D | ORACLE z0 (recognition bypassed, true `xi_i` fed directly) | 0.5677 | 0.7960 | 0.309 | fabricates, unchanged |
| E | HOMOGENEOUS practice counts (T=8 for every slice) | 0.5423 | 0.8691 | 0.306 | fabricates, unchanged |
| F | ZERO idiosyncratic noise (`z0 = xi_i + eta_c` exactly) | 0.7086 | 0.8653 | 0.307 | fabricates, unchanged |
| G | ALL FOUR friendly at once (oracle z0 + homog. T + zero noise + M fixed) | 0.6512 | 0.8712 | 0.303 | **still fabricates** |

CG1 silence bar: population mean <= 0.01, p95 <= 0.01 (KDD density).

Arm G is the decisive result: even under the single friendliest
possible condition -- the model is handed the exact true per-learner
ability, there is no idiosyncratic per-slice noise the additive `u_i +
v_c` family could fail to represent, every slice has identical practice
exposure, and the ceiling can't move -- ACT-P0 still implies a 0.65
population rise at the standard 20-epoch budget. This rules out, as
ROOT causes:

- **(a) ceiling `M` absorbing learner-intercept variance**: fixing `M`
  (arm C) changes nothing.
- **(b) recognition network `z0` bias**: replacing the amortized `u_i`
  with the true `xi_i` (arm D) changes nothing.
- **(c) practice-count/opportunity-heterogeneity identifiability
  interaction**: removing heterogeneity (arm E) and removing the
  omitted-noise channel (arm F) each change nothing, and removing all
  of them together (arm G) still fabricates at essentially the same
  magnitude as the unconstrained baseline.

### 3.2 Isolating the real driver: initialization x epoch budget

| Arm | Change | pop_mean_rise | p95_abs_rise | g_c (mean) |
|---|---|---|---|---|
| A (20 epochs, init 0) | baseline | 0.6447 | 0.8659 | 0.307 |
| B (200 epochs, init 0) | 10x epochs | 0.0464 | 0.0871 | 0.030 |
| B (1000 epochs, init 0) | 50x epochs | 0.0135 | 0.0298 | 0.0125 |
| B (2000 epochs, init 0) | 100x epochs | **0.0088** | 0.0207 | 0.0089 |
| H (20 epochs, `g_raw` init = -6) | low init, SAME 20-epoch budget | 0.0161 | 0.0212 | 0.0029 |
| I (200 epochs, `g_raw` init = -6) | low init + more epochs | 0.0121 | 0.0160 | 0.0024 |

Loss decreases monotonically at every epoch count tested (never stuck,
never diverging), and `pop_mean_rise` falls monotonically as epochs
increase, from 0.645 (20 epochs) through 0.046, 0.014, down to 0.0088 at
2000 epochs -- clearing the population-mean bar, with the p95 clause
lagging behind (0.021, still above its 0.01 bar; the design's own stated
reason for having a separate p95 clause -- "the qmirt lesson that the
population mean alone hides per-learner fabrication" -- reproduces
directly in this ladder). **This is a convergence-speed problem, not an
identifiability wall**: there is no plateau away from zero; more
training keeps helping, arbitrarily far past what the current budget
allows.

### 3.3 A cheap initialization fix creates a mirror-image failure

Lowering `g_raw`'s initialization (so the untrained gain starts small
rather than starting at `softplus(0) approx 0.69`) looks at first like
a free, principled fix confined entirely to a training hyperparameter
(no design pin touched). It is not, because the SAME fixed 20-epoch
budget is used for real-growth twins too:

| Arm | Data | `g_raw` init | Epochs | pop_mean_rise | Verdict |
|---|---|---|---|---|---|
| KG_baseline_init0_e20 | true growth (g_true=0.15) | 0 | 20 | 0.512 | ok (detects) |
| **KG_lowinit_e20** | true growth (g_true=0.15) | -6 | 20 | **0.027** | **SILENT ON REAL GROWTH** |
| KG_lowinit_e100 | true growth | -6 | 100 | 0.405 | ok (detects) |
| KG_lowinit_e200 | true growth | -6 | 200 | 0.414 | ok (detects) |

Initializing `g_raw` at -6 (a plausible-looking "fix" for the
false-positive on no-growth data) makes ACT-P0 fail to detect REAL
growth within the same 20-epoch budget it currently uses (arm
KG_lowinit_e20: 0.027, well below any reasonable firing bar), because
`softplus`'s derivative (`sigmoid(g_raw)`) is tiny in the negative tail
ACT-P0 would need to climb out of. The asymmetry runs both ways: the
CURRENT default init (`g_raw=0`, `g_c approx 0.69`) is already close
enough to "a lot of growth" that 20 epochs is enough to detect true
growth (`KG_baseline_init0_e20`: 0.512) but nowhere near close enough to
"no growth" that 20 epochs can unlearn it (`A`: 0.645). A different
fixed init just relocates which twin fails. **There is no single fixed
initialization that is compatible with a 20-epoch budget on both twins
simultaneously**: 1000 epochs from the DEFAULT init (`g_raw=0`)
correctly detects known growth too (`KG_baseline_init0_e1000`: 0.438,
`g_c_mean=0.154` against a true `g_true=0.15` -- accurate recovery, not
a coincidence of scale) and 2000 epochs still detects it while the
no-growth read is nearly silent. The fix has to touch the OPTIMIZATION
budget, not the initialization alone.

### 3.4 A cheaper lever: raise Adam's learning rate, keep 20 epochs

Since 100x more epochs is a large compute-budget ask, a higher learning
rate was tested at the SAME 20-epoch budget already used everywhere:

| lr | NG pop_mean | NG p95 | KG pop_mean (g_true=0.15) | KG g_c_mean |
|---|---|---|---|---|
| 0.05 (current default) | 0.645 | 0.866 | 0.512 | 0.334 |
| 0.1 | 0.325 | 0.472 | 0.347 | 0.215 |
| 0.2 | 0.033 | 0.074 | 0.448 | 0.223 |
| **0.3** | **0.0033** | **0.0067** | **0.549** | 0.236 |
| 0.5 | 0.00003 | 0.00009 | 0.408 | 0.174 |

`lr=0.3` at the SAME 20 epochs clears BOTH the CG1 population-mean and
p95 bars on this single seed while keeping known-growth detection
strong (0.549, if anything stronger than the current default's read).
At `lr=0.5`, `M` is driven strongly negative (`M approx -1.5`),
mechanically clipping the `relu(M-z)` gain to zero for nearly every
learner -- a DIFFERENT route to silence than "`g_c` genuinely shrinks",
and a reminder that `M`'s identifiability is loose enough that either
channel can produce the same population statistic (partial, second-order
support for suspect (a): `M` is not the PRIMARY driver, but it is an
available escape valve once the optimizer has enough freedom to reach
it).

**Robustness check across seeds** (`lr=0.3`, 20 epochs, 3 data seeds x 2
model seeds each): NG population mean is small and consistent
(0.0010-0.0041, always well under the 0.01 bar), but the **p95 clause is
seed-sensitive** (3 of 6 seeds land at 0.007-0.008, passing; 3 of 6 land
at 0.014-0.023, failing). KG detection is robust across all 6 seeds
(0.377-0.450, always confidently firing). **Conclusion: `lr=0.3` at 20
epochs is a large, cheap improvement on the population-mean statistic
and is safe for growth detection, but is not by itself a robust fix for
the per-learner p95 clause** -- the exact clause the design added
because population-level silence can hide per-learner fabrication.

## 4. Mechanism, stated plainly

ACT-P0's `g_c` and `M` start at values (`g_c(0) approx 0.69`, `M(0)
approx` a few units above typical ability) that, run through the
closed-form 10-opportunity extrapolation the CG1/RB-A statistics use,
correspond to near-total saturation of the practice-gated ceiling gap.
Gradient descent DOES point the right direction from there (loss falls
monotonically; `g_c` and `M` both move toward the correct no-growth
regime given enough steps, and toward an ACCURATE recovery of the true
rate on known-growth data given enough steps) -- but the 20-epoch
budget used everywhere ACT actually runs (the probe, `run.py`'s default
`RunConfig`) stops training roughly one to two orders of magnitude
before that convergence completes. None of the data-side suspects
(ceiling absorbing variance, recognition z0 bias, practice-count
heterogeneity, omitted per-slice noise) contributes measurably -- arm G
proves this by removing all four at once and still reproducing the
fabrication at full strength. The defect is specific to `train_active`'s
bare fixed-epoch loop, which has no convergence check at all, unlike
`bank.calibrate_bank`'s patience/relative-NLL-change stopping rule
elsewhere in the same package.

## 5. Does a principled fix exist within ACT-P0's frozen constraints?

Yes, and it requires no change to any pinned design element (`mu=0`,
`rho=1`, `lambda_i=1` for P0, no free per-learner multipliers, no new
transition term) -- every candidate below is a pure optimizer/training
hyperparameter change to `ActiveConfig`/`train_active`'s call sites,
never to `active.py`'s model or transition logic:

1. **Raise the ACT learning rate** (from 0.05 toward roughly 0.2-0.3)
   at the current epoch budget. Cheap (no extra compute), large
   improvement on the population-mean statistic, ROBUST for
   known-growth detection, but NOT reliably robust for the p95 clause
   across seeds on its own (section 3.4).
2. **Raise the epoch budget substantially** (empirically, order
   1000-2000 epochs from the default init to clear both the mean and
   p95 bars on the toy scale tested here) at the current learning rate.
   Works, monotonically, no plateau -- but is a large, uncosted compute
   increase (roughly 50-100x per ACT cell) relative to section 7's
   budget, which was priced assuming the current 20-epoch runs.
3. **Combine both** (moderately higher lr, e.g. 0.1-0.2, plus several
   hundred epochs) is the likely efficient point, but the exact
   trade-off was not swept precisely here.
4. **The actually principled fix, not just an empirically-tuned
   number**: give `train_active` a convergence-gated stopping rule,
   mirroring `bank.calibrate_bank`'s `n_epochs_max` + `patience_epochs`
   + relative-NLL-change test. A fixed epoch count, however large,
   either wastes compute on cells that converge fast or under-trains
   cells that converge slowly (the p95 seed-sensitivity in section 3.4
   is exactly this: different draws need different amounts of
   training). A convergence criterion lets the DATA decide when g_c/M
   are done moving, for every twin (NG, KG, NS, SAT) and every real-bed
   cell uniformly, without hand-tuning a new fixed number per config.

**Caveats, stated plainly**: every number above comes from a small
CPU-only, single-tag, no-bank-calibration surrogate (`n_kcs` 12-40,
`n_learners` 300-800), built to isolate the mechanism cheaply, not to
reproduce the real pipeline's scale, multi-KC structure, or bank-noise
interactions. The DIRECTION (raise lr and/or add a convergence
criterion) is solid -- it was checked across 6 independent seeds and at
a 3-4x larger scale (section 3.2's scale-up arm: `lr` unchanged but low
init at C=40/N=800 confirms the mechanism generalizes) -- but the exact
lr/epoch/patience numbers must be re-verified at the real probe scale
(`C=64, N=2000`, the actual `synth.py`/`bank.py` pipeline) before they
are adopted for the campaign. This is a second, larger diagnostic/tuning
pass, not something this CPU ladder can responsibly certify on its own.

## 6. Recommendation: repair, not retirement

ACT-P0 should be **repaired, not retired**. The evidence rules out a
structural defect: the same unmodified estimator, given adequate
optimization budget, correctly stays silent on no-growth data and
accurately recovers the true gain on known-growth data (section 3.2-3.3,
including a near-exact rate recovery, `g_c=0.154` against a true
`g_true=0.15`). The defect is confined to one function
(`train_active`'s stopping rule) and one or two `ActiveConfig`/
`RunConfig` defaults (`lr`, `n_epochs`), not to the transition, the
pins, or the recognition network. Retiring P0 with the probe as its
epitaph would be the right call only if the ladder had found a genuine
identifiability wall (arm G proves it did not) or a structural bug
(section 2 found none).

Proposed path, for whoever owns the next revision (not executed here,
per this task's read-only scope):

1. Re-run the ablation ladder's key arms (baseline vs. higher-lr vs.
   convergence-gated) at the REAL probe scale (`C=64, N=2000`, actual
   `synth.py`/`bank.py`), on GPU, to get production-faithful numbers.
2. Add a convergence-gated stopping rule to `train_active` (patience +
   relative-loss-change, reusing `bank.py`'s pattern), rather than
   hand-picking a new fixed epoch count -- this is the fix least likely
   to need re-tuning again if the KDD/EdNet real-bed scales behave
   differently from the synthetic twins.
3. Re-verify CG1 (silence), CG1a (known-growth firing), CG1b (SYN-NS
   misfit robustness), and CG1c (saturation refusal) all still pass
   under the repaired training rule before ACT-P0 rejoins the campaign.
4. Rejoin via the idempotent store per the LEDGER's own framing (the
   exclusion was logged as "a pre-campaign revision ... reversible via
   the idempotent store"): this is a pre-R2 build-phase fix, not a
   change to any pre-registered synthetic-certification threshold, so
   it does not draw on the section 5.6 revision budget (that budget
   gates revisions to the certification MATRIX itself, which has not
   yet run for ACT-P0).

## Appendix: reproduction commands

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
export KMP_DUPLICATE_LIB_OK=TRUE
CUDA_VISIBLE_DEVICES="" python kt-mirt/scripts/probe/_diag_act_p0_mechanism.py
CUDA_VISIBLE_DEVICES="" python kt-mirt/scripts/probe/_diag_act_p0_fix_check.py
```

Both scripts are CPU-only regardless of `CUDA_VISIBLE_DEVICES` (device
is hardcoded to `"cpu"` inside each), total wall clock well under 10
minutes for the core ladder (the 1000-2000 epoch confirmatory arms run
separately, about 1-3 minutes each). Both files are scratch diagnostics
(`_diag_*.py` naming per the harness convention) and may be deleted; no
other file in the repository was modified to produce this diagnosis.
