# OrdRec Implementation Guide

This is the developer-facing implementation guide that pairs with the strategic plan at `docs/exrec_ordinal_plan.md`. The strategic plan answers why and what. This guide answers how, file by file, so a code-completion assistant can produce the code with minimal additional context.

The guide reflects three locked corrections from 2026-06-04. (A) Probe-based GPCM entropy reduction is the primary reward, Fisher information is a one-paragraph theoretical lens. (B) A single framework with per-dataset adapters behind one `OrdinalDatasetBase` interface, not parallel per-dataset code paths. (C) A custom RL library with a small `RLAlgorithm` ABC and one file per algorithm, no Tianshou.

Namespace note. The Phase 1 bundles wrote one reward file under `rl/src/irtrec/rewards/`; this guide unifies everything under `rl/src/ordrec/` so the top-level package is consistent. All call sites below assume the unified namespace.

## 1. Overall directory tree

```
rl/
  pyproject.toml
  src/ordrec/
    __init__.py
    data/                                  # Module 1, Agent A
      __init__.py                          # E1
      base.py                              # E1  OrdinalDatasetBase ABC, AdapterConfig
      schema.py                            # E1  COMMON_RECORD_SCHEMA + validators
      split.py                             # E1  deterministic user-level splits, chunking
      eedi.py                              # E1  EediAdapter, distractor-difficulty K=4
      ednet.py                             # E2  EdNetAdapter, (correctness, time) K=4
      assist.py                            # E2  AssistAdapter, K=2 identity passthrough
      synthetic.py                         # E1  wraps ma-irt synthetic gen, smoke target
      placeholder_2pl.py                   # E1  StaticGPCM(K=2) wrapper for Eedi step 1
      ma_irt_bridge.py                     # E1  adapter -> ma-irt SequenceDataset shim
      tests/
        test_base_contract.py              # E1
        test_schema_round_trip.py          # E1
        test_eedi_adapter.py               # E1
        test_ednet_adapter.py              # E2
        test_assist_adapter.py             # E2
        test_split_determinism.py          # E1
        test_ma_irt_bridge.py              # E1
        fixtures/{eedi_mini.csv, ednet_mini/, assist_mini.csv}
    envs/                                  # Module 4
      base.py                              # E3  OrdinalEnvBase, OrdinalState dataclass
      ordrec_env.py                        # E3  Gym-style env around frozen MAGPCM
      action_mask.py                       # E3  admin + probe + within-ep no-repeat
      item_cache.py                        # E2  per-item (alpha, beta) lookup builder
      bench_forward.py                     # E2  timing harness, no_grad regression test
    reward/                                # Module 2, Agent B
      config.py                            # E3  RewardConfig dataclass
      ordinal_reward.py                    # E3  OrdinalRewardCompute callable
      probe_entropy.py                     # E3  phi_entropy(theta, probe, alpha, beta)
      nll_anchor.py                        # E3  gpcm_nll, terminal anchor
      exposure.py                          # E3  Sympson-Hetter penalty + EMA buffer
      running_norm.py                      # E3  RunningMeanStd with freeze
      tests/{test_potential_shaping, test_entropy_bounds, test_reward_scale,
             test_anti_gaming_mask, test_sympson_hetter,
             test_fisher_special_case, test_terminal_anchor}.py  # all E3
    training/                              # Module 3, Agent C
      base.py                              # E4  RLAlgorithm ABC, RolloutStats, UpdateStats
      rollout.py                           # E4  RolloutBuffer (on-policy, GAE)
      replay.py                            # later, ReplayBuffer for DQN / SAC
      gae.py                               # E4  compute_gae(rewards, values, dones, ...)
      ppo.py                               # E4  PPO concrete implementation
      dqn.py                               # later, DQN sketch
      sac.py                               # later, SAC-discrete sketch
      utils.py                             # E4  set_seed, schedule helpers, polyak
      tests/{test_rollout_buffer, test_gae, test_ppo_smoke,
             test_action_mask, test_save_load}.py                  # all E4
    bc_warmstart/
      bc.py                                # E4  behaviour cloning warm-start for actor
      static_mve.py                        # E4  exact K^{K_B} MVE warm-start for critic
    scripts/
      train_ppo.py                         # E4  top-level train script
      eval_policy.py                       # E4  evaluation harness
      sanity_toy_env.py                    # E4  PPO smoke test runner
    configs/{ppo_eedi_k4, ppo_ednet_k4, ppo_assist_k2}.yaml         # all E4
  tests/                                   # cross-package integration tests
    test_env_reward_wiring.py              # E3
    test_train_smoke.py                    # E4
```

## 2. Module 1, data adapters

### 2.1 The `OrdinalDatasetBase` ABC

The interface lives in `rl/src/ordrec/data/base.py` and is the single contract seen by the model, training loop, reward function, and evaluation harness. Every dataset-specific quirk lives inside a concrete subclass file. The model never branches on dataset identity.

```python
# rl/src/ordrec/data/base.py
Split = Literal["train", "valid", "test"]

@dataclass(frozen=True)
class AdapterConfig:
    name: str; raw_dir: Path; out_dir: Path
    split_seed: int = 0
    test_frac: float = 0.1; valid_frac: float = 0.1
    min_seq_len: int = 5;   max_seq_len: int = 200
    chunk_long_sequences: bool = True

class OrdinalDatasetBase(ABC):
    @abstractmethod
    def materialise(self) -> None: ...
    @abstractmethod
    def load(self) -> None: ...
    def get_split(self, split: Split) -> np.ndarray:
        return np.where(self._student_split ==
                        {"train": 0, "valid": 1, "test": 2}[split])[0]
    def __len__(self): return len(self._questions)
    def __getitem__(self, idx) -> Dict[str, Tensor]:
        return {"questions":  torch.tensor(self._questions[idx], dtype=torch.long),
                "responses":  torch.tensor(self._responses[idx], dtype=torch.long),
                "student_id": idx + 1}
    def get_n_questions(self):      return int(self._metadata["n_questions"])
    def get_n_categories(self): return int(self._metadata["n_categories"])
    def get_n_kcs(self):        return int(self._metadata.get("n_kcs", 0))
    def get_q_matrix(self):     return self._q_matrix
    def get_metadata(self):     return dict(self._metadata)
```

Lifecycle. `cfg = AdapterConfig(...)`, `adapter = EediAdapter(cfg)`, `adapter.materialise()` writes the artefact, `adapter.load()` reads it back, `adapter[i]` returns one sequence, `adapter.get_split("train")` returns row indices.

### 2.2 On-disk schema (every adapter emits exactly this)

A materialised dataset lives at `<out_dir>/<adapter_name>/` with four files.

```jsonc
// sequences.json
[
  {
    "questions": [4, 7, 2, 91, ...],   // 1-based ids in [1, n_questions], 0 reserved as pad
    "responses": [0, 2, 1, 3, ...],    // ints in [0, n_categories - 1]
    "split":     "train"
  }, ...
]
```

```jsonc
// metadata.json
{
  "dataset_name": "eedi_k4_task34",
  "adapter_class": "EediAdapter",
  "n_students":     int,
  "n_questions":        int,
  "n_categories":   int,
  "n_kcs":          int,                // 0 if no KC tags
  "seq_len_range":  [int, int],
  "ordinal_coercion_method": str,       // binary | distractor_difficulty_2pl
                                        // | correctness_x_time_quadrant
                                        // | synthetic_gpcm
  "splits": {"split_seed": int, "train_frac": float, "valid_frac": float,
             "test_frac": float, "n_train": int, "n_valid": int, "n_test": int},
  "question_id_map": {"raw_id_str": int, ...},
  "coercion_artefacts_path": "coercion_artefacts.json"
}
```

`q_matrix.npz`, optional, present iff `n_kcs > 0`, shape `(n_questions, n_kcs)`, uint8, binary. `coercion_artefacts.json`, adapter-specific train-only statistics persisted so test-fold recoding is reproducible without leakage.

### 2.3 EediAdapter, K=4 distractor difficulty (Wang et al. 2020)

Eedi NeurIPS 2020 Tasks 3+4 ships four-option multiple-choice data with the chosen distractor recorded. The three distractors carry different psychometric signals about partial understanding. We order them by the mean ability of students who selected them on the train fold (Muraki 1992 polytomous intuition applied to MC data) so the K=4 scale is empirically defensible.

Algorithm.
1. Fit a placeholder 2PL on the train fold treating `y_uq = 1 iff response == CorrectAnswer`. Read off `theta_hat_u`.
2. For each question q with distractors `D_q = {d_1, d_2, d_3}`, compute `mean_theta_q_d = mean theta_hat over students who chose d on train`.
3. Rank distractors ascending by `mean_theta_q_d`, giving `sigma_q = [d_low, d_mid, d_high]`.
4. Recode every train and test response. `0` if response equals `sigma_q[0]`, `1` if `sigma_q[1]`, `2` if `sigma_q[2]`, `3` if the correct answer.
5. Persist `sigma_q` in `coercion_artefacts.json["distractor_order_per_q"]`. Do not refit on test, refitting would leak.

Edge cases. A distractor never chosen on train, lexicographic fallback so `sigma_q` is always length 3. A test response with an unseen `AnswerValue`, recode to the median wrong category 1 and log under `fallback_questions`. A test-only question, fall back to `[1, 2, 3, correct]` and log.

The placeholder 2PL fitter `fit_placeholder_2pl` in `placeholder_2pl.py` wraps `StaticGPCM(n_categories=2, n_traits=1)` from `ma-irt/models/static_gpcm.py`, trained roughly 5 epochs at lr=1e-2. The R `mirt` path stays available as an audit reference behind a `--use-r-mirt` flag.

### 2.4 EdNetAdapter, K=4 from (correctness, response_time)

EdNet (Choi et al. 2020 AIED) does not ship distractor information. We fall back to the polytomous-from-binary literature pattern (Pelanek 2017, Khajah et al. 2016), folding response time into the ordinal label.

Coercion table.

```
                       response_time <= median_q (fast)     response_time > median_q (slow)
correctness = 0        1   (incorrect, fast)                0   (incorrect, slow)
correctness = 1        3   (correct, fast)                  2   (correct, slow)
```

Ordering rationale. `incorrect slow < incorrect fast < correct slow < correct fast` on the principle that genuine struggle without success is the lowest mastery signal, fast incorrect is a guess-and-move-on, slow correct is mastery present but not automatic, fast correct is the strongest mastery signal. Monotonicity of `P(category k | theta)` in theta is enforced as a contract test on the materialised fold.

Per-question median, not global median. A global median conflates item difficulty with student speed. The median is computed on train only and persisted. Missing response time defaults to the slower bucket (`label_as_slow`). Cold items on test fall back to the global median computed across the train fold, with the count logged for the eval write-up. KT3 is the default load level. Hint usage defaults to 0 on KT3; KT4 may upgrade to K=5 later.

### 2.5 AssistAdapter, K=2 identity passthrough

ASSISTments 2009 Skill Builder. The point of this adapter is not to push K=2 prediction accuracy. It is to provide the binary ablation control against which the K=4 ordinal datasets must outperform on the same backbone, encoder, reward, eval. Coercion is identity, `y_ord = correct` clipped to `{0, 1}`. `metadata["n_categories"] = 2`, `ordinal_coercion_method = "binary"`. User-level deterministic split. The existing `ma-irt/data/assist2009_bin/` artefact can be upgraded by adding the `splits` block in a one-off script.

### 2.6 The `ma_irt_bridge` shim

`ma-irt/utils/dataloader.py` already implements `SequenceDataset` and `collate_sequences`. The adapter framework owns deterministic splits (Correction B). To avoid two split policies in conflict we do not reuse `DataModule`. Instead a single shim file wraps a materialised adapter into `SequenceDataset` per split.

```python
# rl/src/ordrec/data/ma_irt_bridge.py
def adapter_to_sequence_dataset(adapter, split, *, min_seq_len=1, max_seq_len=0,
                                id_offset=1) -> SequenceDataset:
    idx = adapter.get_split(split)
    return SequenceDataset(
        [adapter._questions[i] for i in idx],
        [adapter._responses[i] for i in idx],
        min_seq_len=min_seq_len, max_seq_len=max_seq_len, id_offset=id_offset)

def adapter_to_dataloader(adapter, split, *, batch_size, shuffle=False,
                          num_workers=0, **kwargs) -> DataLoader:
    ds = adapter_to_sequence_dataset(adapter, split, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_sequences, num_workers=num_workers)
```

### 2.7 Tests required (data)

Contract (every adapter), `emits_canonical_schema`, `item_ids_are_one_based` (0 reserved as pad), `responses_in_range`, `ordinal_monotonicity_under_placeholder_2pl` (mean label per theta quantile is monotone non-decreasing, catches sign flips), `split_determinism` (byte-identical sequences.json on rebuild), `no_test_in_train`. Eedi, `distractor_order_is_train_only`, `recodes_correct_answer_to_top_category`, `unseen_distractor_falls_back_to_median_wrong`. EdNet, `median_computed_train_only`, `quadrant_mapping_table`, `missing_time_labeled_as_slow`, `kt3_no_hint_field`. ASSISTments, `identity_passthrough`. Bridge, `emits_collate_compatible_batch` (returns the `(q_pad, r_pad, mask, sid)` 4-tuple `MAGPCM.forward` expects), `n_questions_matches_metadata`, `q_matrix_shape_and_dtype`.

## 3. Module 2, reward computation

### 3.1 `RewardConfig` dataclass

```python
# rl/src/ordrec/reward/config.py
@dataclass(frozen=True)
class RewardConfig:
    # Per-batch shaping weights
    w_info: float = 1.0; w_cost: float = 0.05; w_expo: float = 0.10; w_voi: float = 5.0
    # Probe sizes:  |C| shaping = 32,  |H_probe| terminal anchor = 20
    probe_M: int = 32; probe_H: int = 20
    # Sympson-Hetter exposure control
    r_max: float = 0.20; c_expo: float = 1.0; expo_ema_decay: float = 0.99
    # Episode geometry, plan Section 5
    K_B: int = 5; T: int = 10
    # Stratification
    n_difficulty_strata: int = 5
    # Numerical stability
    eps_prob: float = 1e-12; prior_precision_jitter: float = 1e-6
    running_norm_freeze_after: int = 1000
```

### 3.2 `OrdinalRewardCompute` callable signature

```python
# rl/src/ordrec/reward/ordinal_reward.py
class OrdinalRewardCompute:
    def __init__(self, cfg: RewardConfig, n_categories: int): ...
    def __call__(self, state: dict, action: Tensor, next_state: dict,
                 info: dict) -> tuple[Tensor, dict]: ...
```

Argument contract.
- `state`, `next_state`, MAGPCM forward output at positions `t` and `t + K_B`, dicts with `{logits, probs, theta, alpha, beta}`. Only `theta[:, -1, :]` is read.
- `action`, `LongTensor (B, K_B)`, 1-indexed item ids, 0 reserved as pad.
- `info`, probe context fixed at `env.reset`. Keys, `probe_C_ids (B, M)`, `probe_H_ids (B, H)`, `probe_H_resp (B, H)`, `alpha_table (Q+1, D)`, `beta_table (Q+1, K-1)`, `step_index (1..T/K_B)`, `theta_0 (B, D)`, `last_K_B_acts (B, K_B)`, `fleet_expo (Q+1,) EMA`, `theta_prev (B, D)` cached phi input.
- Returns, `(B,) reward`, plus a breakdown dict `{r_info, r_cost, r_expo, r_voi, r_total, phi_t, phi_prev}` each shaped `(B,)`.

### 3.3 Probe set construction (called once at `env.reset`)

The probe sets are sampled once per trajectory and frozen for the episode. This is what keeps Ng-Harada-Russell (1999) potential-based invariance intact; the probe is part of the static reward structure, not part of the state.

Recipe.
1. Difficulty stratification. Require an offline-fit 2PL `beta_em` of shape `(Q,)` produced by the R `mirt` baseline at training-set time and persisted as `data/<dataset>/em_2pl_beta.pt`. Partition the bank into `cfg.n_difficulty_strata = 5` quintiles by sorted `beta_em`. Equal-count quintiles, not equal-width.
2. Allowable pool. Per student, remove items already in `H_init` and previously sampled probe ids, giving stratum-wise pool `A_b`.
3. C draw. `ceil(M / 5) = 7` items per quintile via seeded `torch.randperm`, truncate to M. H_probe draw, 4 per quintile, disjoint from C; the union must have size `M + H = 52`.
4. Held-out responses. H_probe items are drawn from the student's held-out tail (real attempts, not simulated). If fewer than 20 stratified items are available, substitute from the nearest quintile and log to `info["probe_H_fallbacks"]`. This is the only place the design depends on real (not simulated) responses, so it is explicit.
5. Mask invariants. The policy's action mask sets all ids in `C union H_probe` to `-inf` for the entire episode. Together with the administered-items mask and the within-episode no-repeat mask, this gives a single `(B, Q+1)` ActionMask updated at batch boundaries.
6. Seed handling. Per-episode probes use a derived generator seeded by `hash((cfg.seed, student_id, episode_idx)) & 0xFFFFFFFF`.

### 3.4 Vectorised batched entropy in PyTorch

Formula, `phi(theta) = -(1/M) sum_q sum_k P_qk(theta) log P_qk(theta)`. Shapes, `theta (B, D)`, `probe_ids (B, M) long`, `alpha_table (Q+1, D)`, `beta_table (Q+1, K-1)`, returns `phi (B,)`.

```python
# rl/src/ordrec/reward/probe_entropy.py
def phi_entropy(theta, probe_ids, alpha_table, beta_table, eps=1e-12):
    alpha_q = alpha_table[probe_ids]                                # (B, M, D)
    beta_q  = beta_table[probe_ids]                                 # (B, M, K-1)
    interaction = (alpha_q * theta.unsqueeze(1)).sum(-1)            # (B, M)
    alpha_norm  = alpha_q.norm(dim=-1)                              # (B, M)
    step_values = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta_q
    cum_logits  = step_values.cumsum(-1)                            # (B, M, K-1)
    zeros       = torch.zeros(*theta.shape[:1], probe_ids.shape[1], 1,
                              device=theta.device, dtype=theta.dtype)
    logits      = torch.cat([zeros, cum_logits], dim=-1)            # (B, M, K)
    log_probs   = torch.log_softmax(logits, dim=-1)                 # (B, M, K)
    H_q         = -(log_probs.exp().clamp_min(eps) * log_probs).sum(-1)
    return H_q.mean(dim=1)                                          # (B,)
```

`(alpha, beta)` are read from the precomputed per-item table (plan Section 4), not re-encoded. At `B = 32, M = 32, K = 5` this is 5120 logits, trivial relative to one encoder forward.

### 3.5 Fisher-information grounding paragraph (one paragraph, theoretical lens only)

Our reward sits in the Bayesian sequential design family that originates with Lindley (1956, Annals of Mathematical Statistics), where the value of an experiment is the expected reduction in posterior entropy. Owen (1975, Journal of the American Statistical Association) specialised this to quantal-response item selection and gave the modern Bayesian-CAT formulation. Under a Laplace posterior approximation `N(theta_hat, I_obs^{-1})` and a one-step lookahead in which the probe set C is taken to be the singleton `{q_next}`, our per-batch entropy-reduction reward `phi(theta_t) - phi(theta_{t-K_B})` reduces exactly to the expected Kullback-Leibler gain from administering `q_next`, which equals one half `log det I(q_next, theta_hat)` plus a theta-independent constant. For the GPCM, `I(q, theta)` is Muraki's (1993, Applied Psychological Measurement) closed-form item information, `sum_k k^2 P_qk(theta)` minus the squared expectation of `k` under `P_q`. Choosing q to maximise our reward in this special case is therefore identical to maximum Fisher information item selection. The probe-based generalisation departs from Fisher information by setting C to a held-out, difficulty-stratified probe rather than the candidate item, so the reward ties the encoder's representational capacity to downstream measurement quality on items the policy did not pick. This is the property that lets us claim the deep encoder buys something Fisher information at the candidate cannot directly score.

### 3.6 Sympson-Hetter exposure penalty (1985)

Two-scale exposure tracking. Per-session counter `sess_count[q] in {0, 1}` resets at `env.reset` and increments on first administration within the episode. Fleet-wide rate `expo_rate[q]` is an EMA across completed episodes with decay 0.99 (effective horizon roughly 100 episodes, matching the rollout budget at 32 episodes per PPO update), updated once per episode after the terminal step. Stored as a buffer on the env and persisted across resets.

Per-batch penalty.

```
r_expo_b = - c_expo * sum_{q in a_b} max(0, expo_rate[q] - r_max)
```

At `r_max = 0.20` and `c_expo = 1.0`, an item at 0.30 fleet exposure incurs a 0.10 penalty per administration, comparable in magnitude to one batch's `w_info` shaping at typical phi differences. The hinge means exposure is a soft constraint that bites only after the EMA breaches `r_max`, not a hard wall that distorts early exploration.

### 3.7 Reward terms in code (the full per-batch computation)

```python
# 1. Shaping
phi_t    = phi_entropy(theta_t,    probe_C, alpha_table, beta_table)
phi_prev = phi_entropy(theta_prev, probe_C, alpha_table, beta_table)
r_info   = cfg.w_info * (phi_t - phi_prev)                          # (B,)

# 2. Ask cost
r_cost   = -cfg.w_cost * (action > 0).sum(-1).to(theta_t.dtype)     # (B,)

# 3. Exposure penalty
r_expo   = -cfg.c_expo * cfg.w_expo * \
           (fleet_expo[action] - cfg.r_max).clamp_min(0.0).sum(-1)  # (B,)

# 4. Terminal NLL anchor, only at horizon
if info["step_index"] == cfg.T // cfg.K_B:
    nll_T     = gpcm_nll(theta_t,         probe_H_ids, probe_H_resp, ...)
    nll_prior = gpcm_nll(info["theta_0"], probe_H_ids, probe_H_resp, ...)
    r_voi     = cfg.w_voi * (nll_prior - nll_T)
else:
    r_voi     = torch.zeros_like(r_info)

r_total = r_info + r_cost + r_expo + r_voi
```

All four terms preserve the batch dimension B, so the rollout buffer logs per-trajectory contributions without aggregation loss. `gpcm_nll` mirrors `phi_entropy` but indexes `log_probs` at the observed `y_j` and averages over H.

### 3.8 Numerical stability notes

Use `torch.log_softmax`, never `log(softmax(x))`. Clamp probabilities at `eps_prob = 1e-12` only on the `probs` side of the entropy product, not on the `log_probs` side. For terminal NLL, gather `log_probs` at `y_j`, do not `log(softmax(...)[y_j])`. Add `prior_precision_jitter = 1e-6 * I` to any Laplace precision matrix before inversion. Cap cumulative logits at `+/- 50` before softmax to prevent NaN when `raw_alpha` briefly explodes early. Use `RunningMeanStd` over `r_total` for the first 1000 rollouts, then freeze. Assert `phi in [0, log K]` in debug builds. Cache `phi_prev` at boundary b for reuse as `phi_t` at boundary b+1; the cache lives in `info`, not in module state.

### 3.9 Tests required (reward)

`potential_shaping_telescopes` (Ng-Harada-Russell, sum of `r_info` over two transitions equals `w_info * (phi(theta_T) - phi(theta_0))`), `entropy_bounds_at_extremes` (phi finite and in `[0, log K]` over theta and alpha extremes), `reward_scale_balance` (no single component exceeds 70% of `|r_total|` in expectation over 256 random rollouts), `per_component_logging_decomposes` (sum equals total at atol 1e-5), `probe_anti_gaming_mask` (uniform random policy never samples a probe id), `sympson_hetter_threshold_engages` (hinge zero below `r_max`, linear above), `fisher_special_case_limit` (single-item probe matches Muraki 1993 half-log-det I up to a constant), `invariance_to_probe_seed_at_fixed_reset` (bit-exact probes on identical reset seeds), `running_mean_std_freeze` (1001st update is a no-op), `terminal_anchor_only_at_horizon` (`r_voi == 0` mid-episode, nonzero at horizon).

## 4. Module 3, RL algorithm library

### 4.1 The `RLAlgorithm` ABC

Pedagogy follows CleanRL (Huang et al. 2022, JMLR) and Spinning Up (Achiam 2018). The ABC is deliberately small. The only shared surface is `rollout`, `update`, `act`, `save`, `load`. The buffer is owned by the algorithm because PPO needs an on-policy `RolloutBuffer` and DQN needs a `ReplayBuffer` with fundamentally different semantics.

```python
# rl/src/ordrec/training/base.py
@dataclass
class RolloutStats:
    mean_episode_return: float; mean_episode_length: float
    n_transitions: int; info: dict[str, float] = field(default_factory=dict)

@dataclass
class UpdateStats:
    policy_loss: float; value_loss: float; entropy: float
    approx_kl: float; clipfrac: float; n_grad_steps: int
    extras: dict[str, float] = field(default_factory=dict)

class RLAlgorithm(ABC):
    policy: nn.Module
    buffer: RolloutBuffer | ReplayBuffer
    def __init__(self, observation_dim, action_dim, device, seed=0): ...
    @abstractmethod
    def rollout(self, env, n_episodes) -> RolloutStats: ...
    @abstractmethod
    def update(self, buffer=None) -> UpdateStats: ...
    @abstractmethod
    def act(self, state, deterministic=False) -> tuple[int, dict]: ...
    def save(self, path) -> None: ...
    def load(self, path) -> None: ...
```

### 4.2 `RolloutBuffer`

Pre-allocated `(capacity, ...)` storage to avoid Python-side appends in the hot loop. Capacity equals `n_episodes_per_update * max_steps_per_episode`. For OrdRec with batched inflow, `max_steps_per_episode = T / K_B = 2`, so a PPO update over 32 episodes uses capacity 64.

```python
# rl/src/ordrec/training/rollout.py
@dataclass
class RolloutBatch:
    states; actions; old_log_probs; advantages; returns; old_values; action_masks

class RolloutBuffer:
    def __init__(self, capacity, observation_dim, n_actions, device,
                 gamma=0.95, gae_lambda=0.95): ...
    def insert(self, state, action, reward, log_prob, value, done,
               action_mask=None, episode_start=False) -> None: ...
    def reset(self) -> None: ...
    def compute_advantages(self, last_value=0.0) -> None: ...
    def iter_minibatches(self, minibatch_size, shuffle=True,
                         normalize_advantages=True): ...
```

Per-step fields, all preallocated. `states (obs_dim,)`, `actions (long)`, `rewards`, `log_probs`, `values (float)`, `dones (bool)`, `action_masks (n_actions bool)`, `episode_starts (bool)`. Sequential PPO via `iter_minibatches` with shuffle. Random uniform sampling for DQN lives in `replay.py`.

### 4.3 GAE (Schulman et al. 2016, arXiv 1506.02438)

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_hat_t = delta_t + (gamma * lambda) * (1 - done_t) * A_hat_{t+1}
R_t     = A_hat_t + V(s_t)
```

Hyperparameters. `gamma = 0.95`, `lambda = 0.95`. Tight discount because OrdRec episodes are only two transitions long. `lambda = 0.95` is standard, biasing toward TD(1) Monte Carlo.

Implementation, reverse-time loop over `t in reversed(range(T))`, mask `next_non_terminal = 0` when `dones[t]`, accumulate `gae = delta + gamma * lambda * next_non_terminal * gae`, return `(advantages, advantages + values)`. The test uses a hand-written numpy reference at T=8 with one mid-trajectory done and `last_value = 0.0`, asserting `torch.allclose` at atol 1e-6.

### 4.4 PPO with all the bells and whistles

Schulman et al. 2017, arXiv 1707.06347. File-per-algorithm in the CleanRL style. Key differences from CleanRL.
1. Discrete action over Q items (`Q ~ 948` on Eedi). Categorical with an action mask passed from the env; already-administered items get logit `-inf`.
2. State comes from an OrdRec env that wraps the frozen MAGPCM. State vector is `theta_t` plus probe-entropy summary plus exposure features plus batch index. Policy sees the state vector only, not the encoder hidden state (plan Section 5).
3. Two transitions per episode at `K_B = 5, T = 10`. Short rollouts, so we collect many episodes (32) per update and run fewer (4) epochs.

Hyperparameter defaults.

```
learning_rate = 3e-4         entropy_coef_initial = 0.01
adam_eps = 1e-5              entropy_coef_final = 0.0
clip_eps = 0.2 (policy+val)  entropy_anneal_fraction = 0.5
gamma = 0.95                 value_coef = 0.5
gae_lambda = 0.95            max_grad_norm = 0.5
n_epochs = 4                 minibatch_size = 32 (or 64)
target_kl = 0.02 (early stop at 1.5x)
n_episodes_per_update = 32   max_steps_per_episode = 2
total_updates = 1000         hidden = 128
```

Update body, the bells and whistles in one place.

```python
ratio       = torch.exp(log_probs - batch.old_log_probs)
unclipped   = ratio * batch.advantages
clipped     = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch.advantages
policy_loss = -torch.min(unclipped, clipped).mean()

# Value clipping, CleanRL style
v_clipped    = batch.old_values + torch.clamp(values - batch.old_values, -clip_eps, clip_eps)
value_loss   = 0.5 * torch.max((values - batch.returns).pow(2),
                                (v_clipped - batch.returns).pow(2)).mean()

loss = policy_loss + value_coef * value_loss - ent_coef * entropy
optimizer.zero_grad(); loss.backward()
nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
optimizer.step()

approx_kl = (batch.old_log_probs - log_probs).mean().item()
if approx_kl > 1.5 * target_kl: early_stop = True; break
```

Entropy anneal, `frac = min(1.0, update_count / int(entropy_anneal_fraction * total_updates))`, `ent_coef = (1 - frac) * ent0 + frac * ent_final`. KL early stop terminates the epoch inner loop.

### 4.5 DQN sketch (future-fillable)

Mnih et al. 2015 Nature, with Double DQN (van Hasselt et al. 2016 AAAI). Off-policy using `ReplayBuffer` in `training/replay.py`. Target network `target_q_net` is a frozen lagged copy, hard update every `target_update_interval = 1000` steps or soft polyak at `tau = 0.005`. Epsilon-greedy from 1.0 to 0.05 linearly over the first 10% of training. Double-DQN argmax with `Q_online`, value with `Q_target`. SmoothL1 loss. Adam lr 1e-4 (lower than PPO's 3e-4), grad clip 0.5. Action masking via `masked_fill` on the Q head.

### 4.6 SAC-discrete sketch (future-fillable)

Christodoulou 2019, arXiv 1910.07207. Twin Qs `Q1, Q2` with polyak targets at `tau = 0.005`. Stochastic categorical policy. Auto-tuned temperature with target entropy `0.98 * log(|A|)`. Critic target uses `sum_a pi(a|s') * (min(Q1_t, Q2_t)(s', a) - alpha log pi(a|s'))`. Actor minimises `sum_a pi(a|s) * (alpha log pi(a|s) - min(Q1, Q2)(s, a))`. Same `ReplayBuffer` as DQN. Roughly 350 lines once filled.

### 4.7 Tests required (RL)

`rollout_buffer_insert_reset`, `rollout_buffer_minibatch_iteration` (covers every transition exactly once), `replay_buffer_circular_overwrite`, `gae_matches_numpy_reference`, `gae_terminal_zero_bootstrap`, `ppo_act_deterministic_argmax`, `ppo_action_mask_respected` (1000 samples against a mask permitting one action, all equal it), `ppo_smoke_on_toy_env` (`mean(last_5) > mean(first_5) + 0.5` over 20 updates), `ppo_kl_early_stop_triggers`, `ppo_entropy_anneal_schedule`, `algorithm_swap_deterministic_env` (compare env-side `(s, r, done)` sequence, not policy decisions), `save_load_round_trip`, `advantage_normalization_off_when_n_eq_1`, `value_clipping_active`, `rollout_advantage_sign_matches_reward`.

## 5. Module 4, env and training loop

### 5.1 `OrdinalEnvBase` and `OrdRecEnv`

The env wraps the frozen MAGPCM (the world model) and the reward function. It owns the per-item `(alpha, beta)` cache, the probe sampler, and the fleet exposure EMA buffer. Gym-style API.

```python
# rl/src/ordrec/envs/base.py
@dataclass
class OrdinalState:
    theta_t: Tensor           # (D,)        most recent ability
    probe_summary: Tensor     # (M,)        per-probe entropy summary
    exposure_feat: Tensor     # (F,)        fleet exposure statistics
    batch_idx: int            # 0..T/K_B
    action_mask: Tensor       # (Q+1,) bool
    raw_info: dict            # passthrough for reward
    def to_tensor(self, device) -> Tensor: ...   # concat then to device

class OrdinalEnvBase(ABC):
    @abstractmethod
    def reset(self) -> OrdinalState: ...
    @abstractmethod
    def step(self, action) -> tuple[OrdinalState, float, bool, dict]: ...
    observation_dim: int      # @property
    action_dim: int           # @property
```

`OrdRecEnv` implements this against MAGPCM. Inside `step`, items are administered in batches of `K_B = 5`. The frozen world model is forwarded under `torch.no_grad()` over the extended history. The reward function is called with `(state, action, next_state, info)` and returns `(B,) reward + breakdown`. Episode terminates when `batch_idx == T // K_B`.

### 5.2 Top-level training script

Pseudocode for `rl/src/ordrec/scripts/train_ppo.py`. Seven steps, sequential.

```python
def main():
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"]); device = pick_device(cfg)

    # 1. Frozen world model (MAGPCM, eval mode, requires_grad_(False))
    world_model = load_frozen_magpcm(cfg["ma_irt"], device)

    # 2. Adapter (Agent A) and reward (Agent B) factories
    adapter   = build_adapter(cfg["adapter"])
    reward_fn = build_reward(cfg["reward"], world_model=world_model)

    # 3. Env, owns frozen world model, item (alpha, beta) cache, probe sampler,
    #    fleet exposure EMA, action mask machinery
    env = OrdRecEnv(world_model=world_model, adapter=adapter, reward_fn=reward_fn,
                    K_B=cfg["env"]["batch_size_K_B"], T=cfg["env"]["horizon_T"],
                    probe_size=cfg["env"]["probe_size"],
                    held_out_probe_size=cfg["env"]["held_out_probe_size"],
                    device=device)

    # 4. PPO
    ppo = PPO(observation_dim=env.observation_dim, action_dim=env.action_dim,
              device=device, total_updates=cfg["ppo"]["total_updates"],
              **cfg["ppo"].get("hyperparameters", {}))

    # 5. Optional warm-start (BC for actor + exact K^{K_B} static MVE for critic)
    if cfg.get("warmstart", {}).get("enabled", True):
        bc_warmstart(ppo, env, n_updates=cfg["warmstart"]["bc_updates"])
        mve_warmstart_critic(ppo, env, n_updates=cfg["warmstart"]["mve_updates"])

    # 6. PPO loop, save best by mean episode return
    for it in range(cfg["ppo"]["total_updates"]):
        rstats = ppo.rollout(env, n_episodes=ppo.n_episodes_per_update)
        ustats = ppo.update()
        log_metrics(log_path, it, rstats, ustats)
        maybe_save_best(ppo, rstats, cfg["output_dir"])
```

## 6. Cross-references between modules

The contract surface between modules is tight. `OrdinalEnvBase.reset()` returns an `OrdinalState` whose `to_tensor(device)` yields the policy's observation; the state also carries the per-step `action_mask` and a `raw_info` dict. `OrdinalEnvBase.step(action)` returns `(next_state, reward_scalar, done, info)`. The reward module is called inside `env.step()` with `(state_dict, action, next_state_dict, info)` where `state_dict` is the five-key MAGPCM forward output `{logits, probs, theta, alpha, beta}` and `info` carries the probe context fixed at reset. The reward returns `(B,) reward` plus a breakdown dict; the env aggregates to a scalar for the buffer and surfaces the breakdown via `info` for logging. `RLAlgorithm.rollout(env, n_episodes)` drives `env.reset()` and `env.step()` and writes into `self.buffer`. `RLAlgorithm.update(buffer)` operates purely on the buffer and never touches the world model. The adapter is only touched by the env (probe sampling at reset) and the `bc_warmstart` step. The world model is owned by the env, the env exposes `item_alpha` and `item_beta` buffers built once at construction; the reward and any MVE expansion read those rather than re-forwarding.

The PPO file does not know about Eedi, EdNet, or ASSISTments. The reward file does not know about the dataset either. Only the adapter does. This is the Correction B invariant in practice.

## 7. Implementation order for E1 through E4

E1, data adapters. `data/{base, schema, split, synthetic, placeholder_2pl, ma_irt_bridge, eedi}.py` plus tests (`base_contract`, `schema_round_trip`, `split_determinism`, `eedi_adapter`, `ma_irt_bridge`) and the 50-row Eedi fixture, plus `configs/ordrec_eedi_k4.yaml`. No RL code in E1. Deliverable, materialise Eedi from fixture, materialise the synthetic adapter, train MAGPCM on the synthetic adapter to convergence, confirm prediction metrics match the existing baseline.

E2, per-item lookup, freeze wrapper, bench. `envs/item_cache.py` (sweep each `q in [1..Q]` through one frozen forward to build `(Q+1, D)` alpha and `(Q+1, K-1)` beta tables), `envs/bench_forward.py` (timing harness asserting frozen forward speed and the no_grad regression), plus `data/ednet.py`, `data/assist.py` and tests. Also a thin `class FrozenMAGPCM(nn.Module)` freeze wrapper exposing `forward_no_grad(...)`. Test, identical `(q, theta)` always yields identical logits across calls.

E3, env, reward, wiring. The whole `envs/` (`base.py`, `ordrec_env.py`, `action_mask.py`) and `reward/` packages, plus the reward test suite, plus `tests/test_env_reward_wiring.py` confirming a single `env.step(action)` returns a reward whose four-component breakdown sums to `r_total` and the action mask blocks every probe id.

E4, RL library, training loop, smoke. `training/{base, rollout, gae, ppo, utils}.py` plus tests, plus `bc_warmstart/{bc, static_mve}.py`, plus `scripts/{train_ppo, sanity_toy_env, eval_policy}.py`, plus `configs/ppo_eedi_k4.yaml`. Smoke run, PPO on a toy two-state two-action env shows mean episode return strictly increasing over 20 updates; PPO on the real env runs end-to-end for 5 updates on the synthetic adapter and produces a saved `best.pt`. DQN and SAC stay as sketches; fill when an off-policy ablation is requested.

## 8. First PR scope (E1 only)

The PR creates the `rl/` package skeleton and lands the data layer plus the Eedi MAGPCM training config. Nothing else.

- `rl/pyproject.toml`, single-package layout, depends on `ma-irt` via path or editable install.
- `rl/src/ordrec/__init__.py`, empty marker.
- `rl/src/ordrec/data/__init__.py`, re-exports `OrdinalDatasetBase`, `AdapterConfig`, `EediAdapter`, `SyntheticAdapter`, `build_adapter`.
- `rl/src/ordrec/data/base.py`, ABC and `AdapterConfig` dataclass.
- `rl/src/ordrec/data/schema.py`, `COMMON_RECORD_SCHEMA` constants and JSON validators.
- `rl/src/ordrec/data/split.py`, `make_split`, `stratified_split`, `_chunk_sequences`.
- `rl/src/ordrec/data/synthetic.py`, wraps the existing `ma-irt/scripts/data_gen.py` outputs and adds the missing `splits` block.
- `rl/src/ordrec/data/placeholder_2pl.py`, `fit_placeholder_2pl` around `StaticGPCM(K=2)`.
- `rl/src/ordrec/data/eedi.py`, `EediAdapter` with the K=4 distractor-difficulty algorithm.
- `rl/src/ordrec/data/ma_irt_bridge.py`, `adapter_to_sequence_dataset`, `adapter_to_dataloader`.
- `rl/src/ordrec/data/tests/`, contract, schema round-trip, split determinism, Eedi, bridge, plus fixtures.
- `ma-irt/configs/ordrec_eedi_k4.yaml`, MAGPCM training config pointing at the materialised Eedi artefact.

Out of scope, no env, no reward, no RL library, no training loop.

## 9. Open engineering questions, aggregated

Data.
1. Eedi placeholder 2PL, R `mirt` (~30 min, calibrated) vs `StaticGPCM(K=2)` SGD (~2 min). Recommend the SGD path with a `--use-r-mirt` audit flag.
2. EdNet KT3 vs KT4. K=4 from `(correctness, time)` is the lock; K=5 with a hint axis is a follow-up KT4 sub-experiment.
3. Per-user vs per-sequence split granularity. Per-user is the lock; per-sequence may be needed for within-user CV.
4. Cold-item handling on EdNet test, global-median fallback vs dropping cold-item responses.
5. Train-fold `theta_hat` caching, prefer an out-of-band cache keyed by `(raw_csv_md5, fit_seed)`, not embedded in `coercion_artefacts.json`.
6. Q-matrix for Eedi, multi-hot (~388 KCs) vs single-tag hierarchy collapse (pyKT convention).
7. ASSISTments quality, passthrough vs filter (~30% loss).
8. Multi-trait extension, `n_traits = 1` for the headline; Q-matrix to D-trait mapping needs a separate design pass.

Reward.
9. Synchronous vs per-trajectory `fleet_expo` updates.
10. EM 2PL beta refit, one-time canonical vs per-experiment.
11. Held-out probe shortfall, nearest-quintile substitution, shrink H, or skip the trajectory.
12. `RunningMeanStd` per-transition vs per-episode normalisation.
13. Prior precision jitter, config constant vs Hessian-spectral-norm auto-tune; the latter complicates the Ng-Harada-Russell argument.
14. `K_B` scaling beyond 5, refit default weights vs learnable `w_info` with a Lagrangian floor.
15. Laplace special-case caveat, paper-level discussion vs code-level Hessian condition-number diagnostic.

RL.
16. Per-episode bootstrap, hardcode `0.0` for v1 (always reach done); revisit if T becomes variable.
17. Per-item `(alpha, beta)` cache lives in the env, not the world model or the reward.
18. Action mask is emitted on `OrdinalState`, passed through by the algorithm, stored per transition in the buffer.
19. Episode batching in `rollout()`, sequential v1; vectorised v2 if throughput needs it.
20. BC warm-start, `bc_warmstart(ppo, env, n_updates)` imports the actor from the PPO instance.
21. Static MVE warm-start, `5^5 = 3125` evals per state per update; vectorise via `(3125, K_B)` category tensor against `(probe, K)` probs.
22. Seeding, `training.utils.set_seed(seed)` covers python, numpy, torch, torch.cuda; each `RLAlgorithm` reseeds in `__init__`.
23. Logging, CSV plus print in v1; wandb behind a config flag in v2.
24. Entropy anneal per-update vs per-env-step. Current, per-update.
