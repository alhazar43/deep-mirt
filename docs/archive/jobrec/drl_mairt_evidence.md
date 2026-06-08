# DRL-MAIRT Evidence Synthesis

Built from two parallel literature surveys run on 2026-06-04
(workflow id `wf_bdc6f245-6ea`). Raw structured outputs preserved at
`docs/cleanup/_drl_research_digest.md`.

This file collapses the literature evidence into project decisions.
It does not duplicate the design content of
[drl_mairt_synthesis.md](drl_mairt_synthesis.md), it scopes it.

## Convergent message

The research-scientist and psychometric-researcher independently
reached the same conclusion through different vocabulary. With
one-sided data the defensible v1 is **CAT-style adaptive assessment
with a deep-IRT belief state**, where:

- The reward is intrinsic psychometric efficiency, not learning gain
  (the latter is circular against any simulator trained on the same
  responses, per Codex's earlier confounding warning and the
  psychometric agent's causal-grounds caveat).
- The policy can be a strong heuristic, a bandit, or a small RL agent.
  Full PPO is not required and the workflow's PPO commitment was
  premature.
- The contribution is measurement quality plus four publishability
  hooks that the existing literature has not covered.

## The published landscape

| Family | Lead refs | Reward / objective | Status |
|---|---|---|---|
| Bilevel meta-CAT | BOBCAT (2021) | Outer-level held-out likelihood | Strong baseline |
| End-to-end RL CAT | NCAT (2022), GMOCAT (2023), MAAT (2020) | Theta error, multi-objective scalarization | Direct competitors |
| KT-sim + RL | ExRec (2025), ALPN (2023), CSEAL (2019), GEHRL (2023) | Simulated learning gain | Closest positive template |
| Classical CAT in production | BanditCAT + AutoIRT (2024), Deep-CAT (2026) | Fisher information | Bandit baseline |
| Real-world online RL | Bassen et al. (CHI 2020) | Course completion + time-on-task | Only true real-deployment evidence |
| Offline / conservative | DCQN (2023) | Cumulative simulated gain | Conservative RL slot |

Three sample-efficiency facts to internalize before writing:

- Below 10K to 100K trajectories, offline RL (CQL, IQL, BCQ, DCQN)
  loses to MFI / weakest-skill heuristics.
- At ASSISTments-09 scale (~350K interactions), neural CAT beats MFI
  by 3 to 10 percent.
- At EdNet scale (~131M interactions), the bottleneck is
  representation and simulator-real gap, not data volume.

The bandit + heuristic pathway is competitive at all scales the user
has data for. PPO becomes interesting only after the simpler ladder is
proven.

## Anchoring decisions

### Datasets

| Role | Dataset | Why |
|---|---|---|
| Primary | ASSISTments 2009 | De facto reviewer expectation, matches ma-irt's existing pipeline, mastery-gated within-skill random order gives OPE signal |
| Scale companion | EdNet KT1 | 13K items, 131M interactions, supports held-out-item generalization and sample-efficiency curves |
| Ordinal angle | Eedi NeurIPS 2020 (Task 4) | 4-option MC, natural K=4 GPCM fit, public Task-4 train/test splits |
| Skip | Statics2011, KDD Cup 2010 | Fixed curriculum, low order variation |

For framing C (implicit reward from orderings) ASSISTments 2015 is the
best candidate because the within-skill order is mastery-gated random.

### Baseline matrix (17 cells, reviewer-bulletproof)

Random, popularity, MFI, KLI, MPWI, a-stratified, Sympson-Hetter
wrapper, BC, theta-only DQN (CaRReL replication), full-state DQN,
BanditCAT, BOBCAT, NCAT, MAAT, GMOCAT, ExRec-best-variant, CCAT. The
CaRReL-stripped DQN is the negative control proving the deep state
matters (the H1 hypothesis from Codex).

### Algorithm ladder

KL-info cold start (Chang & Ying 1996) -> MFI mid-test
(Lord 1980) -> Sympson-Hetter or randomesque exposure
wrapper -> BanditCAT (Thompson on Fisher info) -> full-state DQN
-> PPO. Walk the ladder, stop when the gap to the next rung is small.
Both agents reach this independently.

### Reward composition

Per Owen's 1975 Bayesian CAT criterion, the v1 reward is:

`R_t = w_info * Fisher_info(theta_hat, item)`
`    + w_unc * ( H(theta | history) - H(theta | history + (q, r)) )`
`    + w_exp * expected_score_gain`
`    - w_repeat * repeat_penalty`

Reward components reported separately, never collapsed in analysis.
This per-component reporting discipline is itself a methodological
contribution (rare in published RL+education work).

Learning gain is OUT in v1. It is the simulator's own predicted
correctness change and is causally circular against the simulator that
trained the policy.

### Architecture extension

Add a Q-matrix-aware skill-mastery readout alongside the existing
GPCM theta/alpha/beta readouts. This is the psychometric agent's
strongest architectural recommendation. The user's DKVMN backbone is
structurally close to NeuralCDM (Wang Liu Chen 2019, AAAI). With the
extra readout, ma-irt serves both CAT-style selection AND CDM-style
remediation under one online IRT belief. This is what unlocks the
course-assignment fallback without needing recommendation outcome
logs.

### Validation without ground truth

Mandatory:
- Simulation-based calibration of the posterior (Talts et al. 2018).
- Person-fit l_z and ECI on held-out responses.
- Cross-form rank-order correlation of theta_hat.
- Sympson-Hetter item exposure rates and test-overlap.
- Marginal reliability rho_xx = 1 - E[SE(theta)^2] / Var(theta_hat).

Optional but valuable:
- Classification consistency (Livingston-Lewis 1995) if a threshold
  recommendation is reported.
- DIF / measurement invariance across simulated cohorts.

Psychometric reviewers will reject a CAT paper without exposure
reporting. Bake it in from day one.

## Four publishability hooks the literature has not covered

1. **Ordinal-reward CAT with MA-GPCM.** Every published neural CAT
   paper assumes binary correctness. K>2 categories give strictly
   more information per item (expected score, K-1 thresholds). No
   paper exploits this.
2. **Held-out item generalization.** 80/10/10 item split with the
   pointer-network scorer over the GPCM decoder. Not standard in
   NCAT, BOBCAT, MAAT, GMOCAT.
3. **Separated ability pathway as policy input.** ma-irt's
   `separate_theta=true/false` flag is a built-in ablation of "is
   the right policy state pure ability or item-conditioned
   interaction?" Nobody else has this.
4. **Cross-simulator validation.** Train policy inside DKVMN-based
   ma-irt, evaluate inside Transformer-based ma-irt trained on the
   same data with different seed. Standard mitigation for the
   simulator-real gap that nobody runs.

## What this kills from the earlier plans

- **PPO as v1 algorithm.** Out. Walk the ladder; reach PPO only if
  there is a measurable gap left.
- **Course recommendation as v1.** Out. Becomes a CDM + prerequisite
  graph extension that uses the new skill readout, defensible without
  outcome logs, but not the headline.
- **Career recommendation as v1.** Out. Becomes the RIASEC + Holland
  + O*NET Interest Profiler pipeline (public domain, fully
  theory-driven, no learnable reward). Possible separate paper, not
  v1.
- **Learning-gain reward.** Out. Causally circular against the
  simulator.

## What this preserves from the earlier plans

- IRTBridge / online step API in ma-irt.
- Sibling repo at `deep-mirt-rl/` with vendor submodule.
- Frozen ma-irt by default.
- Auto-revert guardrail if joint training is ever attempted.
- Cold-start UCB hot-mix.
- CaRReL-stripped DQN as the negative control.

## Two open forks

These are not "research questions", they are project-scoping forks
that decide what the v1 paper actually claims.

**Fork A. Ordinal-first or binary-first.**

Option A1: Eedi NeurIPS 2020 as the primary dataset. K=4
multiple-choice gives a natural GPCM target. The ordinal angle becomes
the headline ("first ordinal-reward CAT"). Cleaner methodological
story but a smaller reviewer community.

Option A2: ASSISTments 2009 as the primary, EdNet + Eedi as
companions. Binary correctness as the primary metric, ordinal angle
brought in via Eedi. Safer reviewer story (every CAT paper reports
ASSISTments) but the ordinal angle is one of four hooks, not the
single headline.

**Fork B. Paper scope.**

Option B1 (tight single paper): "CAT with deep-IRT belief state,
four hooks, beats MFI and CaRReL-stripped on three datasets." The
four hooks above. Three datasets (ASSISTments 09 + EdNet KT1 + Eedi
2020). Six months of work, IJAIED or AAAI-EDM submission.

Option B2 (stretch, two regimes): adds the CDM extension +
course-assignment fallback as a second evaluation regime, making the
contribution "a single online IRT belief that supports both CAT
selection and CDM remediation". Twelve months of work, IJAIED long
paper.

The user should call both forks before code lands.

## File reference

- [drl_mairt_background.md](drl_mairt_background.md), Codex's
  feasibility dossier.
- [drl_mairt_recommender_plan.md](drl_mairt_recommender_plan.md),
  Codex's proposal.
- [drl_mairt_synthesis.md](drl_mairt_synthesis.md), the plan-level
  synthesis (Codex vs workflow vs hybrid).
- This file, the evidence-level synthesis (literature -> decisions).
- `docs/cleanup/_drl_research_digest.md`, the raw research outputs.
- `docs/cleanup/_drl_workflow_digest.md`, the raw 7-agent workflow
  outputs.
