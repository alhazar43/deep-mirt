# E3, Env + Reward Layer, Completed 2026-06-08

Branch tip after E3, `7414fb3` on `feat/ordrec-e3`, 12 commits stacked
on `feat/ordrec` tip `f5c536e` (E2 merged).
Strategic plan, `docs/exrec_ordinal_plan.md`.
Implementation guide, `docs/ordrec_impl_guide.md` Sections 3 and 5.

## Scope

E3 lands the Gym-style environment, the four-component reward and the
three-source action mask. The reward sums probe-entropy shaping, an
ask cost, the Sympson-Hetter (1985) exposure penalty and a terminal
value-of-information anchor evaluated against the held-out probe
`H_probe`. The env wraps a `FrozenMAGPCM` and a data adapter, owns
the per-session history, the per-batch state every `K_B = 5` steps,
the per-session exposure counters, and the composed action mask. No
RL library, no training loop, no real Eedi materialisation. Those
are E4 and beyond.

The work follows the impl guide's Section 3 reward spec and Section
5.1 env spec verbatim.

## What landed

Reward package (`rl/src/ordrec/reward/`).

  - `config.py`, the frozen `RewardConfig` dataclass. Defaults
    `w_info=1.0`, `w_cost=0.05`, `w_expo=0.10`, `w_voi=5.0`,
    `probe_M=32`, `probe_H=20`, `r_max=0.20`, `c_expo=1.0`,
    `expo_ema_decay=0.99`, `K_B=5`, `T=10`, `sigma_floor=0.15`,
    `running_norm_freeze_after=1000`, plus numerical-stability
    knobs (`eps_prob`, `prior_precision_jitter`).
  - `probe_entropy.py`, `phi_entropy(theta, probe_ids, alpha_table,
    beta_table)`. Vectorised GPCM predictive entropy in nats with
    log-sum-exp stability, mirrors the head used by
    `ma-irt/models/components/irt.py::GPCMLogits`.
  - `nll_anchor.py`, `gpcm_nll` and `terminal_anchor`. Terminal
    cross-entropy on the held-out `H_probe`, activated only at the
    episode horizon.
  - `exposure.py`, `exposure_penalty` and `update_fleet_exposure`.
    Sympson-Hetter (1985) hinge above `r_max` with EMA fleet
    update, scalable jointly by `w_expo` or per-hinge-slope by
    `c_expo`.
  - `running_norm.py`, `RunningMeanStd` Welford running-mean-std
    with a freeze flag, state-dict round-trippable.
  - `ordinal_reward.py`, `OrdinalRewardCompute` callable returning
    a `RewardBreakdown` with `(r_info, r_cost, r_expo, r_voi,
    r_total, phi_t, phi_prev)`, all shaped `(B,)`. The four-component
    sum exactly equals `r_total`.

Env package (`rl/src/ordrec/envs/`).

  - `base.py`, `OrdinalEnvBase` ABC with `reset(seed)` and
    `step(action)`, plus the `OrdinalState` dataclass carrying
    `theta_t`, optional `sigma_t`, history-so-far,
    `exposure_counts`, `episode_step`, `terminated`, `truncated`,
    `action_mask` and `raw_info`. `to_tensor` flattens
    `theta_t` and a one-hot episode-step prefix into a stable
    policy observation.
  - `ordrec_env.py`, `OrdRecEnv` the concrete env. Wraps a
    `FrozenMAGPCM`, reads from an `OrdinalDatasetBase` adapter,
    samples per-episode probes `C` (size `M = 32`) and
    `H_probe` (size `H_probe_size = 20`) stratified by difficulty,
    maintains per-session history, advances the world model in
    `K_B = 5`-item batches, computes per-batch state, exposes a
    discrete action over `Q + 1` slots, composes the action mask
    from three sources, and calls the reward composer at every
    step.
  - `action_mask.py`, three-source mask composer. AND of
    `admin_mask` (env-static, pad and cold items forbidden),
    `probe_mask` (forbids any id in `C union H_probe`), and
    `no_repeat_mask` (forbids previously administered items
    within the episode). Helpers `build_admin_mask`,
    `build_probe_mask`, `update_no_repeat_mask` and
    `compose_action_mask`.

Cross-package wiring (`rl/tests/test_env_reward_wiring.py`).

  - Confirms the env's `step` returns a reward whose four-component
    sum equals `r_total` to floating-point tolerance, that the
    composed mask blocks every id in the probe through the public
    env surface, and that the fleet exposure EMA updates after an
    episode boundary.

Re-exports.

  - `rl/src/ordrec/__init__.py` now imports the `reward` subpackage
    alongside `data` and `envs`. Package version bumped to `0.0.3`.
  - `rl/src/ordrec/envs/__init__.py` re-exports `OrdinalEnvBase`,
    `OrdinalState`, `OrdRecEnv`, `build_admin_mask`,
    `build_probe_mask`, `compose_action_mask`,
    `update_no_repeat_mask`.
  - `rl/src/ordrec/reward/__init__.py` re-exports
    `OrdinalRewardCompute`, `RewardBreakdown`, `RewardConfig`,
    `RunningMeanStd`, `RunningStats`, `exposure_penalty`,
    `gpcm_nll`, `phi_entropy`, `terminal_anchor`,
    `update_fleet_exposure`.

## Reward components and their tests

Component map.

| Symbol | Source file | Sign | Active when | Cite |
| --- | --- | --- | --- | --- |
| `r_info = w_info * (phi(s_t) - phi(s_{t-K_B}))` | `probe_entropy.py`, `ordinal_reward.py` | positive when entropy drops | every step | Lindley 1956, Ng-Harada-Russell 1999, Owen 1975 |
| `r_cost = -w_cost * K_B` | `ordinal_reward.py` | negative | every step | impl guide Section 3.1 |
| `r_expo = -w_expo * c_expo * sum_{q in action} max(0, expo_q - r_max)` | `exposure.py` | non-positive | when any administered item exceeds `r_max` | Sympson and Hetter 1985 |
| `r_voi = 1[step == T // K_B] * w_voi * (nll_prior - nll_terminal)` | `nll_anchor.py` | positive when terminal NLL beats the prior | terminal step only | impl guide Section 3.5 |

Test coverage (under `rl/src/ordrec/reward/tests/`).

| File | Count | Surface |
| --- | --- | --- |
| `test_potential_shaping.py` | 2 | Two-transition telescoping `phi(s_2) - phi(s_0)`, and prev-then-current chain at the boundary. |
| `test_entropy_bounds.py` | 5 | `phi >= 0`, `phi <= log K` random, uniform limit at `alpha=0`, concentrated-limit upper bound at `K=2`, strongly concentrated below uniform at `K=4`, finite under extreme theta. |
| `test_fisher_special_case.py` | 3 | Single-item probe matches a direct entropy compute, entropy monotone in `alpha` for centred theta, uniform when `alpha=0` matches `log K` (Owen 1975 / Muraki 1993 lens). |
| `test_reward_scale.py` | 2 | Per-component contribution within `[5%, 70%]` of `|r_total|` over a tuned regime, breakdown sums to total. |
| `test_anti_gaming_mask.py` | 4 | Uniform random policy never samples a probe id, probe mask is False on every probe id, admin mask forbids pad slot, no-repeat updates remove previously administered items. |
| `test_sympson_hetter.py` | 5 | Penalty zero below threshold, linear in excess above threshold, `c_expo` scales slope, per-session aggregates over `K_B`, fleet EMA update. |
| `test_terminal_anchor.py` | 4 | VOI zero mid-episode, nonzero at horizon, sign convention (positive when terminal NLL beats prior), `gpcm_nll` matches per-row cross-entropy. |
| `test_running_norm.py` | 5 | Welford matches numpy after `n` samples, freeze at threshold is a no-op after, normalise uses current stats, batched update counts each element, state-dict round trip. |

Total reward tests, 30, all pass.

## Env state + action space contract

`OrdinalState` schema (the `step` return envelope).

  - `theta_t: Tensor (B, D)`, latest ability from the frozen world
    model, always present.
  - `sigma_t: Optional Tensor (B, D, D)`, posterior precision for
    the Laplace path; `None` in the point-estimate path used by E3.
  - `history_questions: LongTensor (B, S_so_far)`, 1-based item ids
    administered so far where `S_so_far = episode_step * K_B`.
  - `history_responses: LongTensor (B, S_so_far)`, observed ordinal
    labels.
  - `exposure_counts: LongTensor (B, Q + 1)`, per-session
    administration counts, per-row sums match `len(history_questions)`.
  - `episode_step: int`, current batch index in `[0, T // K_B]`.
  - `terminated: bool`, true iff the episode reached its horizon.
  - `truncated: bool`, reserved for variable-length episodes, always
    `False` in v1.
  - `action_mask: BoolTensor (B, Q + 1)`, the composed admin/probe/
    no-repeat mask. Allowed items have `True`.
  - `raw_info: Dict[str, Any]`, free-form. Carries `probe_C_ids`,
    `probe_H_ids`, `probe_H_resp`, `alpha_table`, `beta_table`,
    `fleet_expo`, `step_index`, `theta_0`, plus logging fields.

Action space.

  - Discrete over `Q + 1` per slot, action passed to `step` is a
    `LongTensor (B, K_B)` of 1-based item ids.
  - The pad slot (index `0`) is always disallowed in the admin mask.
  - Items in `C union H_probe` are always disallowed in the probe
    mask.
  - Items previously administered in the episode are disallowed in
    the no-repeat mask.
  - Calling `step` with any masked-out id raises `ValueError`.

Env tests (under `rl/src/ordrec/envs/tests/`).

| File | Count | Surface |
| --- | --- | --- |
| `test_action_mask.py` | 5 | Three-source AND composes correctly, probe mask disjoint with admin, no-repeat update idempotent, admin mask accepts extra forbidden ids, shape-mismatch rejection. |
| `test_ordrec_env.py` | 6 | Reset returns a well-shaped state, step advances and returns a `(B,)` reward, masked action raises, horizon terminates, observation and action dim properties, terminal-step VOI is active. |

Cross-package wiring (`rl/tests/test_env_reward_wiring.py`).

| Test | Surface |
| --- | --- |
| `test_one_step_breakdown_sums_to_total` | One env step's four-component sum exactly equals `r_total`. |
| `test_mask_blocks_every_probe_id_through_env` | Composed mask blocks every id in `C union H_probe`. |
| `test_fleet_exposure_updates_after_episode` | EMA fleet exposure changes after an episode boundary. |

Reproducer.

```
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  pytest rl/src/ordrec/ rl/tests/ -v
```

```
============================= 126 passed in 4.29s ==============================
```

All 82 pre-E3 tests (42 E1 data + 13 E2 adapter + 27 E2 envs) still
pass. The 44 new E3 tests (30 reward + 11 env + 3 wiring) pass on
the final E3 tip (`7414fb3`).

## Open issues for E4

Carried forward where still open from E1 and E2, plus new items from
E3.

  1. RL library and training loop. `RLAlgorithm` ABC, PPO, GAE,
     rollout collector, BC warmstart, sanity toy env. The whole
     `training/` subpackage belongs to E4.
  2. Per-batch sigma_t Laplace path. The state envelope carries
     an `Optional Tensor (B, D, D)` posterior precision but the
     point-estimate path is the v1 default. Laplace calibration
     wires to `RewardConfig.sigma_floor` and is deferred to a
     critic-warmstart pass in E4 or E5.
  3. Reward-normalisation activation. `OrdinalRewardCompute`
     supports a frozen Welford normaliser via the `normalise`
     constructor flag. Unit tests run un-normalised so the
     four-component sum is checkable; the headline run will
     enable it.
  4. Probe sampler stratification audit. `n_difficulty_strata = 5`
     equal-count quantiles. A small ablation over `[3, 5, 10]`
     stratifications is a candidate for E5 once the headline run
     defines a metric to optimise against.
  5. Carry-over from E2, item 1, EdNet KT4 hint-aware K=5 variant.
     E3 did not touch this. Future ablation.
  6. Carry-over from E2, item 2, per-item alpha averaging
     `n_contexts` sweep. E3 did not touch this. Future ablation.
  7. Carry-over from E2, item 3, DKVMN CUDA latency at `B=1`.
     Not blocking, rollout batches are `B >= 32`. E4 profiling
     pass.
  8. Carry-over from E2, item 4, real-Eedi pre-merge execution
     waits for csvs to land locally.
  9. Carry-over from E1, item 1, placeholder 2PL `lr` field
     cosmetic mismatch in `coercion_artefacts.json`. E3 did not
     touch this. Roll into the E4 headline-config sweep.
  10. Carry-over from E1, item 2, synthetic adapter persistence of
      `true_irt_parameters.json`. E3 did not touch this. Roll into
      the headline-pipeline pass.
  11. Open engineering questions from impl guide Section 9, items 9
      through 15 (reward) are now closed by E3. Items 1 through 8
      (data) and 16 through 23 (RL) remain open. Items 16 through 23
      belong to E4.

## File manifest

Code (E3).

```
rl/src/ordrec/envs/base.py
rl/src/ordrec/envs/ordrec_env.py
rl/src/ordrec/envs/action_mask.py
rl/src/ordrec/reward/__init__.py
rl/src/ordrec/reward/config.py
rl/src/ordrec/reward/probe_entropy.py
rl/src/ordrec/reward/nll_anchor.py
rl/src/ordrec/reward/exposure.py
rl/src/ordrec/reward/running_norm.py
rl/src/ordrec/reward/ordinal_reward.py
```

Tests (E3).

```
rl/src/ordrec/envs/tests/test_action_mask.py
rl/src/ordrec/envs/tests/test_ordrec_env.py
rl/src/ordrec/reward/tests/__init__.py
rl/src/ordrec/reward/tests/test_potential_shaping.py
rl/src/ordrec/reward/tests/test_entropy_bounds.py
rl/src/ordrec/reward/tests/test_reward_scale.py
rl/src/ordrec/reward/tests/test_anti_gaming_mask.py
rl/src/ordrec/reward/tests/test_sympson_hetter.py
rl/src/ordrec/reward/tests/test_fisher_special_case.py
rl/src/ordrec/reward/tests/test_terminal_anchor.py
rl/src/ordrec/reward/tests/test_running_norm.py
rl/tests/__init__.py
rl/tests/test_env_reward_wiring.py
```

Re-exports updated (E3).

```
rl/src/ordrec/__init__.py
rl/src/ordrec/envs/__init__.py
```

Results (E3).

```
rl/results/E3_env_reward.md
```

Total, 25 files, 2914 insertions across 12 commits stacked on
`feat/ordrec` tip `f5c536e`.
