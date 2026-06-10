# rl/, OrdRec, Ordinal-IRT Exercise Recommendation

OrdRec wires the ma-irt deep ordinal IRT world model into a PPO-based exercise
recommendation policy whose rewards are defined in trait space. The package is
isolated from the archived job-recommendation work at `archive/rl_jobrec/`.

Strategic plan, [`docs/exrec_ordinal_plan.md`](../docs/exrec_ordinal_plan.md).
Implementation guide, [`docs/ordrec_impl_guide.md`](../docs/ordrec_impl_guide.md).
Build progress and A/B results, [`docs/ordrec_progress.md`](../docs/ordrec_progress.md).

---

## Package map

```
rl/
  pyproject.toml               pip-installable; depends on ma-irt via path
  README.md                    this file
  src/ordrec/
    __init__.py
    data/                      Module 1, data adapters (E1-E2)
      base.py                  OrdinalDatasetBase ABC, AdapterConfig
      schema.py                COMMON_RECORD_SCHEMA + validators
      split.py                 deterministic user-level splits, chunking
      eedi.py                  EediAdapter, distractor-difficulty K=4
      ednet.py                 EdNetAdapter, (correctness, time) K=4
      assist.py                AssistAdapter, K=2 identity passthrough
      synthetic.py             wraps ma-irt synthetic gen, smoke target
      placeholder_2pl.py       StaticGPCM(K=2) wrapper for ordinal recoding
      ma_irt_bridge.py         adapter -> ma-irt SequenceDataset shim
      tests/                   42 E1 + 13 E2 unit tests
    envs/                      Module 4, environment layer (E2-E3)
      base.py                  OrdinalEnvBase ABC, OrdinalState dataclass
      ordrec_env.py            Gym-style env wrapping frozen MAGPCM
      action_mask.py           three-source mask (admin + probe + no-repeat)
      item_cache.py            per-item (alpha, beta) lookup builder
      frozen_magpcm.py         FrozenMAGPCM two-line eval+no_grad wrapper
      bench_forward.py         timing harness + no_grad invariance test
      tests/                   27 E2 + 11 E3 unit tests
    reward/                    Module 2, reward computation (E3)
      config.py                RewardConfig dataclass
      ordinal_reward.py        OrdinalRewardCompute callable
      probe_entropy.py         phi_entropy(theta, probe, alpha, beta)
      nll_anchor.py            gpcm_nll, terminal NLL anchor
      exposure.py              Sympson-Hetter penalty + EMA buffer
      running_norm.py          RunningMeanStd with freeze
      gpcm_ops.py              shared gpcm_log_probs helper (B1)
      tests/                   30 E3 + 2 E4.6a unit tests
    training/                  Module 3, RL algorithm library (E4)
      base.py                  RLAlgorithm ABC, RolloutStats, UpdateStats
      rollout.py               RolloutBuffer (on-policy, GAE, one row per env-step)
      gae.py                   compute_gae(rewards, values, dones, ...)
      ppo.py                   PPO concrete implementation
      config.py                PPOConfig dataclass (B6)
      utils.py                 set_seed, schedule helpers, polyak
      tests/                   18 E4 + 2 E4.6a unit tests
    bc_warmstart/
      bc.py                    behaviour cloning warm-start for actor
      static_mve.py            exact K^{K_B} MVE warm-start for critic
      tests/                   1 E4 unit test (bc_smoke)
  scripts/
    train_ppo.py               top-level PPO training driver
    eval_policy.py             evaluation harness, four-policy comparison
    sanity_toy_env.py          PPO smoke test on toy two-state env
    prepare_eedi_csv.py        pre-merge script for real Eedi NeurIPS 2020 csvs
    run_bench_forward.py       one-off encoder latency benchmark
    eval_e46b.py               B-side eval harness (E4.6b)
    plot_e46b.py               B-side figure generator (E4.6b)
  tests/                       cross-package integration tests (E3-E4, E4.6a)
    conftest.py                shared fixtures for script smoke tests
    test_env_reward_wiring.py  one-step breakdown, mask, fleet EMA
    test_train_ppo_smoke.py    train_ppo end-to-end, artifact schema
    test_eval_policy_smoke.py  eval_policy CLI surface, report schema
  configs/
    ppo_synth_smoke.yaml       minimal smoke config (2 PPO updates)
    ppo_synth_e45.yaml         A-side headline run config
    ppo_synth_e46b.yaml        B-side headline run config (current defaults)
    ppo_eedi_k4.yaml           E5 Eedi production config (pending real csvs)
  results/
    E2_bench_forward.{json,md} measured encoder latency data
    E45_synth_headline.md      A-side: random beats all policies (E4.5)
    E46b_bside_eval.md         B-side: RC1+RC2 fixed, PPO still trails random
    E46b_ab_comparison.md      head-to-head A/B table and attribution
    E46b_R1_ablation.{json,md} R1 exposure recalibration ablation
    archive/                   E1-E4 milestone records (build history)
    plots/                     training curves and comparison figures
  data/                        gitignored
```

---

## How to run tests

Activate the research environment and set PYTHONPATH before running anything.

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research
export PYTHONPATH="rl/src:ma-irt"
export KMP_DUPLICATE_LIB_OK=TRUE
```

Run the full suite from the repository root.

```bash
python -m pytest rl/src/ordrec/ rl/tests/ -v
```

209 tests as of E4.6b. The suite runs in under 20 seconds on CPU.

---

## How to train

The training driver reads a YAML config. All paths are relative to the
repository root unless absolute.

```bash
cd <repo_root>
PYTHONPATH="rl/src:ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python rl/scripts/train_ppo.py --config rl/configs/ppo_synth_e46b.yaml
```

The script expects a frozen MA-GPCM checkpoint at the path specified under
`ma_irt.checkpoint` in the config. Train one with `ma-irt/scripts/train.py`
first. Outputs land in `rl/outputs/<run_name>/` by default.

For the minimal two-update smoke run.

```bash
PYTHONPATH="rl/src:ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python rl/scripts/train_ppo.py --config rl/configs/ppo_synth_smoke.yaml
```

---

## How to evaluate

```bash
PYTHONPATH="rl/src:ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python rl/scripts/eval_policy.py \
    --config rl/configs/ppo_synth_e46b.yaml \
    --checkpoint rl/outputs/<run_name>/best.pt \
    --n-eval-episodes 200
```

The harness evaluates four policies (trained PPO, BC-only, max-Fisher, uniform
random) against the same env-reset seed sequence and writes a per-component
return breakdown plus exposure diagnostics to stdout and a JSON report.

---

## Isolation rule

Do not import from `archive/rl_jobrec/`. That tree is frozen prior-direction
code preserved for traceability. Any useful pattern must be re-derived clean
in this tree with a note on provenance.
