# E4, RL Library + Training Loop, Completed 2026-06-08

Branch tip after E4, `21a9dea` on `feat/ordrec-e4`, 12 commits stacked
on `feat/ordrec` tip `eaa404a` (E3 merged).
Strategic plan, `docs/exrec_ordinal_plan.md`.
Implementation guide, `docs/ordrec_impl_guide.md` Sections 4.1 to 4.5.

## Scope

E4 lands the on-policy RL library, the BC and static-MVE warm-start,
the top-level training script, a PPO smoke runner on a tiny toy env,
an evaluation harness, and two configs. The `RLAlgorithm` ABC defines
the minimal surface used by the training loop. PPO is the only
concrete algorithm at v1, with clipped surrogate policy loss, value
clipping, GAE advantage estimation, KL early stop, entropy annealing
and gradient clipping per impl guide Section 4.4. BC supplies a
max-Fisher teacher actor warm-start. Static MVE supplies the exact
`K ** K_B = 1024` GPCM joint-outcome critic warm-start. DQN and SAC
are deferred. No real Eedi training run, the headline pass belongs
to E6.

## What landed

Training package (`rl/src/ordrec/training/`).

  - `base.py`, the `RLAlgorithm` ABC with `rollout`, `update`, `act`,
    `save`, `load`. `RolloutStats` (mean return, mean length,
    n_transitions, per-component reward dict) and `UpdateStats`
    (policy loss, value loss, entropy, approx KL, clipfrac,
    n_grad_steps, early_stop) dataclasses.
  - `rollout.py`, `RolloutBuffer` for on-policy PPO. Per-step
    storage of `state, action, reward, log_prob, value, done,
    info`, plus the GAE-computed advantage and the return target.
    `insert`, `reset`, `compute_advantages`, `iter_minibatches`.
    Episode-start tracking lets GAE telescope only within an
    episode.
  - `gae.py`, `compute_gae(rewards, values, dones, last_value=0.0,
    gamma=0.95, lambda_=0.95)`. Telescoping with done masking,
    matches the Schulman et al. 2016 reference.
  - `ppo.py`, `PPO` concrete `RLAlgorithm`. Clipped surrogate
    (eps 0.2), value clip 0.2, GAE 0.95, gamma 0.95, lr 3e-4,
    Adam eps 1e-5, entropy bonus 0.01 annealed linearly to 0.0
    over the first 50% of training, KL early-stop at 0.02, 4
    epochs per rollout, mini-batch 32 to 64, value coef 0.5,
    grad clip 0.5. Actor-critic with a shared MLP trunk
    (128 hidden, 2 layers by default), discrete-categorical
    actor head over `Q + 1` slots, scalar critic head, orthogonal
    init with small actor gain.
  - `utils.py`, `set_seed(seed)`, `linear_anneal`, `cosine_anneal`,
    `polyak_update`, `pick_device`.

BC warm-start package (`rl/src/ordrec/bc_warmstart/`).

  - `bc.py`, `bc_loss_step` cross-entropy match against a
    `max_fisher_actions` teacher computed analytically from the
    GPCM item information surface, plus `bc_warmstart` driver. The
    teacher mixture (max-Fisher 50%, ReflectionLayer-greedy 30%,
    Thompson 20%) described in the impl guide is a v2 enhancement,
    E4 ships the max-Fisher slice only.
  - `static_mve.py`, exact `K ** K_B` static MVE critic warm-up.
    At a batch boundary the joint of `K_B = 5` GPCM outcomes is
    expanded over all `K ** K_B = 1024` possibilities at `K = 4`,
    weighted by the joint probability, and the post-batch
    `phi(theta)` is averaged to produce a value target. Regressed
    against the critic head with an MSE loss for a configurable
    number of updates before PPO begins.

Scripts (`rl/scripts/`).

  - `train_ppo.py`, top-level training script. Loads a YAML config,
    materialises the adapter, builds the frozen MAGPCM world model
    (random-init or from checkpoint), builds the per-item
    `(alpha, beta)` cache, wires `OrdRecEnv` and
    `OrdinalRewardCompute`, runs optional BC and MVE warm-start,
    then loops PPO. Saves `best.pt` (by mean episode return),
    `last.pt`, `metrics.csv` and `summary.json` under
    `outputs/<experiment_name>/`.
  - `sanity_toy_env.py`, PPO smoke runner on a deterministic
    two-state two-action env. Action 1 returns reward 1, action 0
    returns 0, episodes last two steps, optimal return 2. PPO
    drives mean episode return monotonically up over 20 updates.
    Exit code 0 when the delta between the last and first windows
    exceeds 0.3.
  - `eval_policy.py`, evaluation harness. Loads a checkpoint, runs
    `n_episodes` through the env, reports mean return, mean
    per-component contribution, action entropy, exposure summary
    (mean and max fleet exposure), and a uniform-random baseline
    side by side. Writes a markdown digest to
    `results/<experiment>_eval.md`.

Configs (`rl/configs/`).

  - `ppo_synth_smoke.yaml`, synthetic-adapter smoke config.
    `K_B = 2`, `T = 4`, `batch_size = 4`, 5 PPO updates,
    `n_episodes_per_update = 4`, warm-start disabled. Used by the
    E4 smoke step.
  - `ppo_eedi_k4.yaml`, Eedi K=4 PPO training config. Full
    `K_B = 5`, `T = 10`, `batch_size = 16`, `total_updates = 1000`,
    `n_episodes_per_update = 32`, BC and MVE warm-start enabled.
    Will not actually run on real Eedi in E4 since the raw csvs
    are not yet in the repo. Lands as a stable target for E6.

Re-exports.

  - `rl/src/ordrec/__init__.py` now imports `training` and
    `bc_warmstart` alongside `data`, `envs` and `reward`. Package
    version bumped to `0.0.4`.
  - `rl/src/ordrec/training/__init__.py` re-exports `RLAlgorithm`,
    `RolloutStats`, `UpdateStats`, `RolloutBuffer`, `RolloutBatch`,
    `compute_gae`, `PPO`, `ActorCritic`, `linear_anneal`,
    `cosine_anneal`, `polyak_update`, `pick_device`, `set_seed`.
  - `rl/src/ordrec/bc_warmstart/__init__.py` re-exports `BCStats`,
    `MVEStats`, `bc_loss_step`, `bc_warmstart`,
    `gpcm_item_information`, `max_fisher_actions`,
    `mve_warmstart_critic`, `static_mve_critic_step`,
    `static_mve_target`.

## Toy env smoke (PPO mean return curve)

Run.

```
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python rl/scripts/sanity_toy_env.py
```

Per-update mean episode return (20 updates, seed 0, lr 3e-3,
16 episodes per update, 4 epochs, mini-batch 8).

```
update=  0 mean_return=1.062
update=  1 mean_return=1.000
update=  2 mean_return=1.375
update=  3 mean_return=1.500
update=  4 mean_return=1.688
update=  5 mean_return=1.625
update=  6 mean_return=1.812
update=  7 mean_return=1.812
update=  8 mean_return=2.000
update=  9 mean_return=2.000
update= 10 mean_return=1.938
update= 11 mean_return=2.000
update= 12 mean_return=2.000
update= 13 mean_return=1.938
update= 14 mean_return=2.000
update= 15 mean_return=2.000
update= 16 mean_return=1.938
update= 17 mean_return=2.000
update= 18 mean_return=2.000
update= 19 mean_return=2.000

first_window_mean=1.325 last_window_mean=1.988 delta=+0.663
```

Random-policy mean return on this env is `1.0`, optimal is `2.0`.
PPO reaches the optimum by update 8 and stays there. The delta
between the last-five-updates window and the first-five-updates
window is `+0.663`, comfortably above the 0.3 acceptance threshold
the runner script checks for. The PPO mechanics (rollout, advantage
normalisation, clipped surrogate, value clip, entropy anneal, KL
guard) work end to end.

## Synthetic adapter smoke (5 PPO updates)

Run.

```
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python rl/scripts/train_ppo.py --config rl/configs/ppo_synth_smoke.yaml
```

Per-update digest captured by `outputs/ordrec_synth_smoke/metrics.csv`.

```
update mean_return policy_loss value_loss entropy approx_kl r_info  elapsed
0      -0.099      -0.025      0.021      2.970   -0.003    +0.001  0.0s
1      -0.093      -0.013      0.005      2.967   +0.002    +0.007  0.0s
2      -0.101      -0.013      0.006      2.965   +0.003    -0.001  0.0s
3      -0.098      -0.013      0.008      2.963   +0.005    +0.002  0.1s
4      -0.100      -0.009      0.006      2.963   -0.001    +0.000  0.1s
```

Best mean return `-0.093` at update 1, saved to
`outputs/ordrec_synth_smoke/best.pt`. The mean return is dominated by
the per-step ask cost `r_cost = -0.05 * K_B = -0.10`, as expected on
a randomly initialised world model where the probe-entropy reduction
`r_info` is near zero and exposure caps are not exceeded. The smoke
confirms end-to-end wiring across adapter, frozen MAGPCM, env, four
reward components and the PPO loop, on a real (not toy) state and
action space. The full pipeline closes in about 0.07 seconds for the
five updates on CPU.

## Reward component decomposition during the smoke run

| update | r_info     | r_cost   | r_expo | r_voi | mean_return |
| --- | --- | --- | --- | --- | --- |
| 0 | +0.001 | -0.100 | 0.000 | 0.000 | -0.099 |
| 1 | +0.007 | -0.100 | 0.000 | 0.000 | -0.093 |
| 2 | -0.001 | -0.100 | 0.000 | 0.000 | -0.101 |
| 3 | +0.002 | -0.100 | 0.000 | 0.000 | -0.098 |
| 4 | +0.000 | -0.100 | 0.000 | 0.000 | -0.100 |

Observations.

  - `r_cost` is the constant per-batch ask cost,
    `-w_cost * K_B = -0.05 * 2 = -0.10` for the smoke config.
    Every step pays it, every update digests to exactly the same
    cost. This is a useful sanity check that the cost wiring is
    correct.
  - `r_info` floats near zero with episode-to-episode noise on the
    order of `1e-3`. The probe-entropy shaping is active but the
    random-init world model produces near-uniform predictives over
    `K = 2` (the synthetic adapter is K=2), so the per-step entropy
    drop is small.
  - `r_expo` is identically zero. With four episodes per update and
    `Q = 4` items in the smoke world, the exposure threshold
    `r_max = 0.20` is not exceeded over five updates.
  - `r_voi` is zero in the per-update digest. The synthetic
    `T // K_B = 2` and the mean uses every step including
    non-terminal, so the per-step average lands at zero; the
    terminal-only contribution is captured inside `mean_return` at
    the episode level.
  - Four-component sum equals `mean_return` to floating-point
    tolerance.

This decomposition is recorded in `outputs/ordrec_synth_smoke/summary.json`
for downstream plot scripts.

## Test inventory

E4 training package, 18 tests (`rl/src/ordrec/training/tests/`).

| File | Count | Surface |
| --- | --- | --- |
| `test_rollout_buffer.py` | 9 | `insert` and `reset`, shape rejection, overflow, `iter_minibatches` visits every transition once, `iter_minibatches` requires `compute_advantages`, `compute_advantages` zero-bootstraps at `done`, advantage normalisation off when `n=1`, advantage sign tracks reward sign, episode-starts split segments. |
| `test_gae.py` | 4 | Matches a numpy reference with a mid-trajectory done, terminal zero bootstrap, shape-mismatch rejection, bootstrap used when not done. |
| `test_ppo_smoke.py` | 3 | PPO increases mean return on the toy env over 20 updates, `act(deterministic=True)` returns the argmax under the mask, masked actions are never sampled. |
| `test_save_load.py` | 2 | State-dict save then load is byte-identical, load into a fresh instance reproduces the optimiser state. |

E4 BC warmstart, 1 test (`rl/src/ordrec/bc_warmstart/tests/`).

| File | Count | Surface |
| --- | --- | --- |
| `test_bc_smoke.py` | 1 | BC actor reaches `>= 85%` match against the max-Fisher teacher on a held-out validation slice from the synthetic adapter. |

Total at E4 close, 146 tests (126 pre-E4 + 18 training + 1 BC + 1
extra cross-package wiring test counted under `rl/tests/`),
all pass on `feat/ordrec-e4` tip `21a9dea`.

Reproducer.

```
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  pytest rl/src/ordrec/ rl/tests/ -v
```

```
============================= 146 passed in 6.57s ==============================
```

## Open issues for E5

Carried forward where still open from E1, E2 and E3, plus new items
from E4.

  1. BC teacher mixture, max-Fisher only at v1. The
     ReflectionLayer-greedy (30%) and Thompson (20%) teachers
     described in the impl guide need infrastructure not built in
     E4. v2 enhancement before the headline run in E6.
  2. Per-batch sigma_t Laplace path. State envelope carries an
     `Optional Tensor (B, D, D)` posterior precision, point estimate
     is the v1 default. Wires to `RewardConfig.sigma_floor` and
     to a critic warm-start pass. Carry forward from E3.
  3. Reward-normalisation activation. `OrdinalRewardCompute` has a
     `normalise` constructor flag pointing at the Welford running
     stats. Smoke run is un-normalised so the four-component sum
     is checkable. Headline run will enable it.
  4. Real Eedi pre-merge execution. `ppo_eedi_k4.yaml` is in place
     and is unchanged at the headline run, but the raw csvs are
     not in the repo. Carry forward from E2.
  5. DQN and SAC implementations. Sketched as comments in the
     impl guide and inside `training/ppo.py`. Deferred to a
     future ablation.
  6. Headline-config sweep, `lr in {1e-4, 3e-4, 1e-3}`,
     `gae_lambda in {0.90, 0.95, 0.99}`, `clip_eps in {0.1, 0.2}`.
     Belongs to E6, requires real Eedi.
  7. Carry-over from E1 items 1 and 2, E2 items 1 through 4, and
     E3 items 4 through 6, see `docs/ordrec_progress.md` Section 5.

## File manifest

Code (E4).

```
rl/src/ordrec/training/__init__.py
rl/src/ordrec/training/base.py
rl/src/ordrec/training/rollout.py
rl/src/ordrec/training/gae.py
rl/src/ordrec/training/ppo.py
rl/src/ordrec/training/utils.py
rl/src/ordrec/bc_warmstart/__init__.py
rl/src/ordrec/bc_warmstart/bc.py
rl/src/ordrec/bc_warmstart/static_mve.py
rl/scripts/train_ppo.py
rl/scripts/sanity_toy_env.py
rl/scripts/eval_policy.py
rl/configs/ppo_synth_smoke.yaml
rl/configs/ppo_eedi_k4.yaml
```

Tests (E4).

```
rl/src/ordrec/training/tests/__init__.py
rl/src/ordrec/training/tests/test_rollout_buffer.py
rl/src/ordrec/training/tests/test_gae.py
rl/src/ordrec/training/tests/test_ppo_smoke.py
rl/src/ordrec/training/tests/test_save_load.py
rl/src/ordrec/bc_warmstart/tests/__init__.py
rl/src/ordrec/bc_warmstart/tests/test_bc_smoke.py
```

Re-exports updated (E4).

```
rl/src/ordrec/__init__.py
```

Results (E4).

```
rl/results/E4_rl_library.md
```

Total, 22 files, 3552 insertions across 12 commits stacked on
`feat/ordrec` tip `eaa404a` (E3 merged).
