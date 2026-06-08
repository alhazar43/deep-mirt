# DRL-MAIRT v1 Implementation Plan

**Date.** 2026-06-04.
**Status.** Spec locked, ready for M0.

A realtime interactive job recommender built on top of ma-irt's deep
IRT belief tracker. The pipeline harvests theta_t from a student's
questionnaire responses in continuous time, matches against a
job-pool-agnostic retriever, supports student-initiated job ratings,
and decides when to ask another question vs surface the current top-K.

This document is the canonical record. It supersedes the earlier
[plan synthesis](drl_mairt_synthesis.md) and the
[evidence synthesis](drl_mairt_evidence.md) on points where they
differ from the eight decisions locked on 2026-06-04 (Section 3).

---

## 1. Project framing

### 1.1 What v1 is

A systems contribution. A methodological paper. A controlled, fully
synthetic evaluation of a deep-IRT-backed interactive recommender
under three honest constraints. A defensible publication target is a
methodology venue (RecSys, AIED, applied psychometrics) rather than a
deployment-results venue.

### 1.2 What v1 is not

Not a deployment claim. Not a learning-gain claim. Not "our DRL
recommender beats heuristics on real users." Not a DRL paper in the
narrow sense; the decision layer is heuristic in v1, the policy
learning sits in the retrieval head (UserTower) which is supervised.

### 1.3 The three constraints and how the design honors each

**Constraint 1, questionnaire-agnostic.** The pipeline only cares
about extracting theta_t and the encoder hidden h_t from any
ma-irt-compatible response stream. Solved by ma-irt's online step API
exposing a frozen StepState. The 1D theta limit is mitigated by
concatenating theta_t (scaled by a learnable scalar s_theta) with the
high-D encoder hidden h_t before projecting into the retrieval space.

**Constraint 2, job-pool-agnostic.** Any job pool plugs in without
retraining the user tower. Solved by an ID-free ItemTower over
features and frozen text embeddings (Volkovs et al. 2021 ZESRec; Yi
et al. 2019 RecSys). The user side never sees a job ID.

**Constraint 3, real-time continuous interactive stream.** Students
respond at their own pace, recommendations regenerate in the
background, students can stop, rate, or continue at any moment.
Solved by async re-rank with debounce, a closed-form ReflectionLayer
that updates the query vector from in-session likes and dislikes
without touching ma-irt, and a heuristic DecisionController that
arbitrates ask vs recommend vs terminate.

### 1.4 What got dropped from earlier framings

DRL as the headline. Full MIRT (rotation indeterminacy). fdata as the
primary dataset (too small, too coarse, no real value over a
controlled synthetic). Path B (theory-driven mapping) as a v1 target.
Each is preserved as v2+ work in Section 10.

---

## 2. The eight locked decisions

| # | Decision | v1 choice | Deferred to |
|---|---|---|---|
| D1 | Repo placement | Subdirectory `deep-mirt/rl/` (no submodule, no new remote) | n/a |
| D2 | Theta dimensionality | 1D inside ma-irt | Section 10.5 (2D MIRT path) |
| D3 | Job pool source | O*NET 2024 (synthetic on both ends) | Section 10.7 (real data) |
| D4 | Questionnaire text | Textless items (IDs only) | n/a |
| D5 | Rating fidelity | Binary like/dislike | n/a |
| D6 | DecisionController | Heuristic rules with config thresholds | Section 10.1 (bandit) |
| D7 | Evaluation simulator | Replay over held-out synthetic users | Sections 10.2 (SNIPS), 10.3 (iEvaLM LLM) |
| D8 | Preference model | Option A, 1D distance, sigmoid on theta - delta_j | Section 10.4 (Option B RIASEC) |

---

## 3. System architecture

### 3.1 System diagram

Lifted from [`docs/cleanup/_drl_interactive_design.md`](cleanup/_drl_interactive_design.md).

```
                            +----------------------------------+
                            |          FRONT-END UI            |
                            |  (any client, web or mobile)     |
                            +----------------------------------+
                                |   ^         |   ^      |   ^
       respond(item, resp)      |   |  next_q |   | top-K|   | rate(job, liked)
                                v   |         v   |      v   |
                            +----------------------------------+
                            |   Service API (FastAPI)          |
                            |   session router + debounce      |
                            +----------------------------------+
                                |                ^
                          event |                | snapshot, top-K, confidence
                                v                |
                            +----------------------------------+
                            |       DecisionController         |
                            |  (heuristic rules: ask /         |
                            |   recommend / terminate)         |
                            +----------------------------------+
                              |          |              |
                              | ask?     | recommend?   | rate?
                              v          v              v
            +-------------------+   +---------------+   +-----------------+
            | FisherItemSelector|   |  BeliefTracker|   | ReflectionLayer |
            |  (next item from  |   |  cache + emit |   |  (q_t -> q_t')  |
            |  ma-irt alpha,beta|   |   snapshot    |   |  in-session     |
            +-------------------+   +---------------+   +-----------------+
                       |                  ^                      |
                       | step(item,resp)  | StepState            | adjusted q_t
                       v                  |                      v
                 +---------------------------------+   +------------------+
                 |      ma-irt (frozen at deploy)  |   |    UserTower     |
                 |  EncoderDecoderModel.step()     |-->|  W_user MLP +    |
                 |  -> theta_t (1D), Sigma_t,      |   |  L2 norm,        |
                 |     h_t (d_h)                   |   |  s_theta scaling |
                 +---------------------------------+   +------------------+
                                                                |
                                                          q_t (in R^d)
                                                                v
                                                       +-------------------+
                                                       |  RetrievalIndex   |
                                                       |  cosine over {v_j}|
                                                       |  (numpy or HNSW)  |
                                                       +-------------------+
                                                                ^
                                                            v_j |
                                                       +-------------------+
                                                       |    ItemTower      |
                                                       |  frozen BGE text  |
                                                       |  + W_item head    |
                                                       |  (pool-swappable) |
                                                       +-------------------+
                                                                ^
                                                                | features
                                                       +-------------------+
                                                       |    O*NET 2024     |
                                                       |  occupation bank  |
                                                       +-------------------+
```

### 3.2 Component table

| Component | Responsibility | ML problem inside | Research anchor | Tech choice | Location |
|---|---|---|---|---|---|
| ma-irt step API | Streaming GPCM forward; emit (theta_t, sigma_t, h_t) per response | Streaming Bayesian / IRT inference of 1D theta from an ordered stream | Yeung 2019 (Deep-IRT); Mislevy 2018 (validity); Lord 1980 (information) | `EncoderDecoderModel.step` + `StepState` dataclass + observed Fisher info for sigma_t | `ma-irt/models/base.py`, `ma-irt/models/components/irt.py` |
| BeliefTracker | Per-session adapter, buffers (q, r), calls `ma_irt.step`, emits `belief.snapshot()` | None new; debounce policy | EAR-style "asynchronous re-rank with debounce" (Lei et al. 2020) | Plain Python adapter (~150 LOC), threadsafe asyncio lock | `rl/src/irtrec/belief/tracker.py` |
| UserTower | Map (theta_t, h_t) into a d-dim query vector q_t | Two-tower retrieval, user side | Covington et al. 2016 (YouTubeDNN); Yi et al. 2019 (sampling-bias correction) | MLP `Linear(1 + d_h, 4d) + GELU + Dropout + Linear(4d, d) + L2norm` with learnable s_theta initialized at sqrt(d_h) | `rl/src/irtrec/retrieval/user_tower.py` |
| ItemTower | Encode each O*NET occupation into a d-dim vector v_j from features only | Pool-agnostic item representation (cold start) | Volkovs et al. 2021 (ZESRec); Reimers & Gurevych 2019 (SBERT); Penha 2025 (RecSys) | Frozen BGE-small-en-v1.5 text branch + Linear head on O*NET structured features | `rl/src/irtrec/retrieval/item_tower.py` |
| RetrievalIndex | Hold {v_j} for the active pool, serve top-K cosine | ANN search (no learning) | Faiss / HNSWlib OSS | numpy matmul for ~900 occupations; HNSWlib if pool grows | `rl/src/irtrec/retrieval/index.py` |
| ReflectionLayer | Adjust q_t in session from like/dislike events | Online preference learning from binary feedback | Lei et al. 2020 WSDM (EAR) | Closed-form: q_t' = normalize(q_t + eta * (sum_liked v_j - sum_disliked v_j)); max cosine shift 0.2 | `rl/src/irtrec/policy/reflection.py` |
| FisherItemSelector | Pick next questionnaire item to ask | Classical CAT item selection (no learning) | Lord 1980 (MFI); Chang & Ying 1996 (KL cold-start) | KL info for t < 5; MFI mid-test; randomesque exposure mask | `rl/src/irtrec/policy/fisher_selector.py` |
| DecisionController | Arbitrate ask vs recommend vs terminate per event | Mixed-action decision under uncertainty | Heuristic family (per the literature for systems shipping without dialogue logs); SCPR (Lei et al. 2020 KDD) as the design precedent | Rules-based controller with config thresholds | `rl/src/irtrec/controller/heuristic.py` |
| FastAPI Service | HTTP surface; async re-rank scheduling | None | drl_mairt_synthesis.md H10 | FastAPI + uvicorn, per-session asyncio task queue | `rl/src/irtrec/service/app.py` |

### 3.3 The one architectural rule

**Ratings never flow into ma-irt.** A like or dislike on a recommended
job is not a response to a calibrated item and would corrupt theta_t
if it leaked into the encoder. Encoded as a runtime assertion in
BeliefTracker.

```python
class BeliefTracker:
    def on_rate(self, job_id: int, liked: bool) -> None:
        # Rating flows to ReflectionLayer only. Never to ma-irt.
        assert not _calls_ma_irt_step()
        self._rating_log.append((job_id, liked))
```

A test in `rl/tests/test_belief_tracker.py` asserts ma-irt's parameter
tensors do not change in response to /rate events.

---

## 4. Realtime data flow

### 4.1 EVENT A. Student answers question (q_i, r_i)

1. Front-end POSTs `/session/{id}/respond` with `(q_i, r_i)`.
2. Service appends to per-session log, schedules an async re-rank
   task, returns 202 immediately.
3. Async task. BeliefTracker calls `ma_irt.step(q_i, r_i)` returning
   `StepState{theta_t, sigma_t, h_t, t}`.
4. Debounce. If theta_t shifted < epsilon AND h_t cosine drift <
   delta, return early and reuse cached top-K.
5. UserTower forward. `q_t = L2norm(W_user([s_theta * theta_t, h_t]))`.
6. ReflectionLayer applies prior session likes / dislikes with the
   0.2 cosine shift cap.
7. RetrievalIndex returns `top_K' = argtopK cosine(q_t', v_j)` with
   `K' = 50` for oversample, post-filtered (no-repeat, exposure cap)
   down to `K = 10`.
8. Service commits the new top-K as the session snapshot.
9. DecisionController, on the same event, calls
   `FisherItemSelector.next()` against the updated theta_hat_t and
   caches the "next question to ask" field.

### 4.2 EVENT B. Student rates a recommended job (j, liked)

1. Front-end POSTs `/session/{id}/rate` with `(j, liked)`.
2. Service appends to the per-session like-log, schedules async
   re-rank.
3. **ma-irt is NOT called.** The rule of Section 3.3.
4. ReflectionLayer pulls the cached q_t from the last response event,
   recomputes q_t' with the new (j, liked) included.
5. RetrievalIndex re-ranks. Service commits the new top-K.
6. DecisionController increments a "low confidence" counter if the
   rating contradicts a top-K head. After three contradictions in a
   session, the controller flips an "offer more questions" flag on
   the next /recommendations read so the UI can prompt the student.

### 4.3 EVENT C. Student requests recommendations (stop or read)

1. Front-end GETs `/session/{id}/recommendations?k=10`.
2. Service returns the latest committed top-K from session cache. No
   model call.
3. Three attached fields: `confidence` (high or low, from
   DecisionController), `reliability_estimate = 1 - sigma_t/var_prior`,
   `items_asked = t`.
4. If `stop=true`, DecisionController flips the session to terminal
   state, emits a final snapshot, persists the session log for
   offline evaluation.
5. If on the boundary (low confidence AND `t < max_items` AND
   reliability < threshold), DecisionController surfaces an "ask
   more?" affordance with the next question already chosen by
   FisherItemSelector.

---

## 5. Synthetic data generator (v1 scope)

### 5.1 User side

Mixed item bank, ma-irt-compatible.

- N synthetic users. Default N = 5000 for the recovery preset, N =
  500 for the dev preset.
- True theta drawn from N(0, 1). 1D, locked.
- Questionnaire bank. 50 items split as follows:
  - 25 items, 2PL binary (K = 2). alpha ~ LogNormal(0, 0.4),
    b ~ N(0, 1).
  - 15 items, GPCM K = 3. alpha ~ LogNormal(0, 0.4), thresholds
    sorted N(b_mean, 0.5) with b_mean ~ N(0, 0.7).
  - 8 items, GPCM K = 5. Same priors.
  - 2 items, GPCM K = 6. Same priors.
- Response generation per (u, q) follows the standard 2PL or GPCM
  kernel (Muraki 1992, *A Generalized Partial Credit Model*, Applied
  Psychological Measurement 16(2)).
- Engagement mixture (from the [fdata-shaped simulator design](cleanup/_drl_simulator_design.md)).
  - Rejecter class, 40 percent of users. Deterministically dislike
    every job. Modeled as a latent class indicator `z_u in {0, 1}`
    sampled per user with P(z_u = 1) = 0.40.
  - Engaged class, 60 percent of users. Sample likes from the
    Option A preference model.
  - No "low engagement" intermediate class in v1 (the fdata
    three-class split is preserved in [Section 10.4](#104-option-b-preference-model-RIASEC)).

### 5.2 Job side, O*NET 2024

Public-domain occupation database from
[onetcenter.org](https://www.onetcenter.org/). Roughly 900
occupations under the 2024 release.

Per-occupation feature schema:

| Field | Type | Source O*NET file | Used by |
|---|---|---|---|
| `occupation_code` | string | `Occupation Data.txt` | join key |
| `title` | string | `Occupation Data.txt` | ItemTower text branch |
| `description` | string | `Occupation Data.txt` | ItemTower text branch |
| `tasks` | list[string] | `Task Statements.txt` | ItemTower text branch (concatenated) |
| `work_activities` | list[float] | `Work Activities.txt` | optional structured feature |
| `knowledge` | list[float] | `Knowledge.txt` | optional structured feature |
| `skills` | list[float] | `Skills.txt` | optional structured feature |
| `abilities` | list[float] | `Abilities.txt` | optional structured feature |
| `riasec_code` | string (3 letters) | `Interests.txt` | Section 10.4 (Option B) |
| `work_zone` | int 1 to 5 | `Job Zones.txt` | **delta_j source in v1** |
| `education` | float | `Education, Training, and Experience.txt` | optional structured feature |

JobPoolSpec dataclass at `rl/src/irtrec/retrieval/pool.py`:

```python
@dataclass
class JobPoolSpec:
    occupation_codes: List[str]
    titles: List[str]
    descriptions: List[str]
    tasks_concat: List[str]            # ItemTower text input
    structured_features: np.ndarray    # shape (n_jobs, d_struct)
    delta_j: np.ndarray                # shape (n_jobs,) v1 scalar from work_zone
    riasec_code: List[str]             # reserved for Option B
```

### 5.3 Preference model (Option A, locked)

Each occupation has a scalar **delta_j**, the z-scored
`work_zone` field. work_zone is an ordinal 1 to 5 that captures the
training and complexity demand of the occupation. Higher work_zone
means higher demand. We z-score across the pool so delta_j ~ N(0, 1)
approximately, matching the theta scale.

Per-user, per-job like probability:

```
P(IsLiked = 1 | u, j) =
    0                            if engagement_class(u) = rejecter
    sigmoid(lambda * (theta_u - delta_j) - bias)   if engagement_class(u) = engaged
```

`lambda` and `bias` are calibrated by bisection to hit a target
overall like rate of 0.20 (default), so the binary classifier has
reasonable signal in either direction. Default search range
`lambda in [0.5, 3.0]`, `bias in [-2.0, 2.0]`.

Three alternative delta_j sources are configurable as sensitivity
analysis in M6 (the "O*NET attribute choice" risk, Section 14).

```yaml
preference:
  delta_source: work_zone        # MVP
  delta_candidates: [work_zone, education_zscore, complexity_composite]
  target_like_rate: 0.20
```

### 5.4 Output files

Written under `data/<dataset_name>/`.

| File | Contents | Schema notes |
|---|---|---|
| `sequences.json` | List of `{questions: [int], responses: [int]}` per user | Matches existing ma-irt loaders unchanged |
| `jobs.json` | List of JobPoolSpec entries (one per occupation) | New schema, consumed only by rl/ |
| `likes.json` | Long-form list of `{user_id, job_id, IsLiked, prob}` records | `prob` is the synthetic ground-truth probability, used in M6 calibration |
| `true_irt_parameters.json` | `{theta, alpha, beta}`. theta is 1D | Matches existing schema |
| `true_preference_parameters.json` | `{delta_j, lambda, bias, engagement_class}` | New; v1 ground truth for the preference model |
| `metadata.json` | seed, N, mixture rates, schema flags, dataset version | New `schema_flag: "drlmairt_sim_v1"` |

### 5.5 Sanity checks (the generator passes these BEFORE training anything)

| Check | Tolerance |
|---|---|
| ma-irt fits theta with `corr(theta_hat, theta_true) > 0.85` at the recovery preset (N=5000) | hard pass |
| Overall like rate within +/-0.02 of target | hard pass |
| Engagement class shares within +/-0.02 of priors | hard pass |
| Unique work_zone values present in [1, 5], no NaNs | hard pass |
| Per-user response count >= 30 (full bank) | hard pass |
| Item-type counts match config (`k_distribution`) within multinomial chi-square p > 0.05 | warn |
| Two runs with the same seed produce byte-identical outputs | hard pass |

---

## 6. ML components in detail

### 6.1 ma-irt online step API

**Input contract.** `step(item_id: int, response: int, state:
Optional[StepState]) -> StepState`.

**StepState.**

```python
@dataclass
class StepState:
    theta_t: float                # 1D MA-GPCM ability estimate
    sigma_t: float                # posterior SD via observed Fisher info
    h_t: torch.Tensor             # encoder hidden, shape (d_h,)
    item_log: List[Tuple[int, int]]
    t: int                        # number of responses so far
```

**Forward computation.** Per-encoder `forward_with_state` advances
the encoder by one step. Per-decoder `compute_logits_from_state`
produces logits given the new hidden. theta_t is the IRT readout.
sigma_t is computed from the observed information

```
I(theta_t) = sum_{q in item_log} alpha_q^2 * Var(Y_q | theta_t)
sigma_t = 1 / sqrt(I(theta_t) + 1/sigma_prior^2)
```

with a jitter floor (1e-6) to avoid zero division at t = 0.

**Tests (parity).** Iterated `step(q_1, r_1), step(q_2, r_2), ...`
must match `forward(full_sequence)` on logits, probs, theta, alpha,
beta to atol = 1e-5. Microbenchmark CI test asserts per-step CPU
latency under 20 ms (DKVMN), 10 ms (LSTM), 40 ms (Transformer at
t = 200).

**Reference.** This is the H1 milestone surface from
[`drl_mairt_synthesis.md`](drl_mairt_synthesis.md). It lives entirely
in ma-irt and is reusable beyond this project.

### 6.2 BeliefTracker

**Purpose.** Thin wrapper around ma-irt held per session. Threadsafe.

**API.**

```python
class BeliefTracker:
    def __init__(self, ma_irt_model, *, eps_theta=0.02, delta_h=0.05): ...
    def step(self, item_id: int, response: int) -> bool:
        """Calls ma_irt.step. Returns True if snapshot changed
        (theta moved > eps OR h cosine drift > delta)."""
    def snapshot(self) -> dict:
        """Returns frozen dict {theta_t, sigma_t, h_t, item_log, t}."""
    def on_rate(self, job_id: int, liked: bool) -> None:
        """Logs rating. Never calls ma_irt.step."""
```

**Tests.** Assert ma-irt's `state_dict` byte-equal before and after
N `on_rate` calls. Debounce test with a contrived sequence.

### 6.3 UserTower

**Purpose.** Project (theta_t, h_t) into the shared retrieval space.

**Forward.**

```
inp = concat([s_theta * theta_t, h_t])    # shape (1 + d_h,)
hidden = GELU(Dropout(Linear(1 + d_h, 4 * d)(inp)))
q_t = L2_normalize(Linear(4 * d, d)(hidden))
```

`s_theta` is a single learnable scalar initialized at sqrt(d_h).
Initialization fights the drown risk (Section 14).

**Loss.** Sampled-softmax with logQ correction (Yi et al. 2019,
RecSys) over training likes (user, liked_job, in-batch negatives).
Auxiliary BPR loss on (liked > randomly-sampled-unliked) for the
same user with weight 0.3.

**Training data.** `likes.json` from the synthetic generator. 80/20
user split (training users vs held-out users for M6 evaluation).

**Defaults.** `d = 64`, `d_h = 32` (matches ma-irt DKVMN default),
batch 256, AdamW lr 1e-3, weight decay 1e-4, 20 epochs.

**Tests.** Gradient flow check; output L2 norm exactly 1; theta
ablation drops Hit@10 measurably; full inputs beat theta-only on
held-out users.

### 6.4 ItemTower

**Purpose.** Encode each O*NET occupation into v_j from features
only. Pool-swappable.

**Forward.**

```
e_text  = BGE_small_en_v1_5(title + " " + description + " " + tasks_concat)   # 384 dims, frozen
e_struct = Linear(d_struct, d_text_branch)(structured_features)
v_j     = L2_normalize(Linear(384 + d_text_branch, d)(concat([e_text, e_struct])))
```

For pools without text, drop the text branch and pass only
structured features. For pools without structured features, drop the
structured branch. The pool registration helper handles either case
via `JobPoolSpec`.

**Training.** Frozen text encoder. Only the head `Linear(384 +
d_text_branch, d)` and the structured branch projection are
trainable. Trained jointly with the UserTower in M4.

**Caveats.** Anisotropy of frozen text encoders biases cosine scores.
L2-normalize after the item-tower head, not on the raw text
embedding. An optional whitening step measured on the registered
pool is in the v2 backlog.

### 6.5 RetrievalIndex

**Purpose.** Top-K cosine over the precomputed {v_j} for the active
pool.

**Defaults.** numpy matmul for the ~900-occupation O*NET pool (sub-ms
on CPU). HNSWlib for pools above 10K items. Faiss only above 1M
items.

**API.**

```python
class RetrievalIndex:
    def fit(self, v_j: np.ndarray, job_ids: List[int]) -> None: ...
    def topk(self, q_t: np.ndarray, k: int, mask: Optional[np.ndarray]) -> Tuple[List[int], np.ndarray]:
        """Returns (job_ids_top, scores) of length k."""
```

**Tests.** Cosine correctness on toy data, mask correctness, top-K
ordering deterministic, retrieval over a fixed pool reproducible.

### 6.6 ReflectionLayer

**Purpose.** Per-session adjustment of q_t from like/dislike events.
Closed-form, no learning.

**Update.**

```
delta = eta * (sum_{j in liked} v_j - sum_{j in disliked} v_j)
q_raw = q_t + delta
q_t'  = L2_normalize(q_raw)

# Cap cosine shift
if cosine_similarity(q_t', q_t) < 1 - max_shift:
    q_t' = slerp(q_t, q_t', alpha=max_shift_to_alpha(max_shift))
```

**Defaults.** `eta = 0.1`, `max_shift = 0.2`. Session-scoped, reset
on new session.

**Tests.** Adversarial trajectory where the user dislikes every
recommendation: assert q_t' does not collapse to the last v_j; assert
cosine shift cap fires; assert reset works.

**Reference.** Lei et al. 2020 WSDM (EAR), Estimation-Action-Reflection
formulation. The cosine-shift cap is the explicit fix for the CaRReL
collapse mode (Section 14).

### 6.7 FisherItemSelector

**Purpose.** Pick the next questionnaire item from the remaining
bank.

**Logic.**

```
if t < 5:
    item = argmax_q KL_info(q, theta_t, neighborhood_radius=0.5)   # Chang & Ying 1996
else:
    item = argmax_q Fisher_info_GPCM(q, theta_t)                   # Lord 1980 + Muraki 1993
```

For GPCM polytomous items the Fisher information is

```
I_q(theta) = alpha_q^2 * (E[Y_q^2 | theta] - E[Y_q | theta]^2)
```

(Muraki 1993, *Information functions of the generalized partial
credit model*, Applied Psychological Measurement).

**Exposure mask.** Randomesque wrapper (Kingsbury & Zara 1989) picks
uniformly from the top n (default n = 5) most informative items, so
the deterministic argmax does not over-expose a small set. Sympson-
Hetter (1985) is a v2 upgrade.

**Tests.** Toy IRT bank with known optimal items, assert MFI picks
them. Cold-start KL fallback fires for t < 5.

---

## 7. The DecisionController (heuristic v1)

The controller arbitrates on every event. All thresholds in a YAML
config.

```yaml
controller:
  rho_high: 0.85               # confidence threshold for "recommend"
  rho_terminate: 0.95          # auto-stop threshold
  topk_jaccard_window: 3       # number of recent updates for stability
  topk_jaccard_floor: 0.7      # required stability for high-confidence
  max_items: 30                # hard cap
  cold_start_min: 5            # ask at least this many before allowing recommend
  contradict_threshold: 3      # "offer more questions" after this many
```

**Rule set.**

1. On each response event, ask `FisherItemSelector.next()` for the
   next item to ask, and trigger a background re-rank.
2. On a `/recommendations` read, return the latest committed top-K
   plus a confidence flag computed as

   ```
   high  if reliability_estimate >= rho_high AND
           topk_jaccard(last topk_jaccard_window updates) >= topk_jaccard_floor AND
           t >= cold_start_min
   low   otherwise
   ```

3. Terminate when one of: user stop, `t >= max_items`, or
   `reliability_estimate >= rho_terminate`.

4. After `contradict_threshold` rating contradictions in a session
   (the user said dislike on something the model scored highly),
   flip a `offer_more_questions = true` flag for the next
   `/recommendations` read.

Sensitivity sweep on the five thresholds is part of M6 (Section 11).

---

## 8. Service layer

FastAPI service at `rl/src/irtrec/service/app.py`. Async re-rank
scheduling via `asyncio.create_task`.

### 8.1 Endpoints

| Method | Path | Payload | Response | Notes |
|---|---|---|---|---|
| POST | `/session` | `{ma_irt_checkpoint, pool_id}` | `{session_id}` | Creates session, loads frozen models |
| POST | `/session/{id}/respond` | `{item_id: int, response: int}` | `202` | Schedules async re-rank; returns immediately |
| GET | `/session/{id}/next_question` | none | `{item_id}` | Returns cached `FisherItemSelector.next()` |
| GET | `/session/{id}/recommendations?k=10` | none | `{top_k: [job_ids], confidence, reliability, items_asked, offer_more}` | Reads latest committed snapshot |
| POST | `/session/{id}/rate` | `{job_id: int, liked: bool}` | `202` | Schedules async re-rank via ReflectionLayer |
| POST | `/session/{id}/stop` | none | `{final_top_k, session_log_id}` | Flips terminal state, persists log |

### 8.2 Async scheduling and debounce

Each session holds an `asyncio.Queue` of events. A single per-session
task drains the queue, applies debounce, performs the re-rank, and
commits the snapshot. The GET endpoints never block on the model;
they read the latest committed snapshot from session cache.

### 8.3 Latency budget

| Path | Budget |
|---|---|
| Background re-rank end-to-end | < 100 ms p95 (DKVMN encoder) |
| GET endpoints | < 5 ms p95 (cache read) |
| ma-irt.step alone | < 20 ms p95 |

Microbenchmark CI test in M1 enforces the step API budget.

### 8.4 Session store

In-memory dict keyed by `session_id`. Optional Redis adapter sketched
at `rl/src/irtrec/service/session_store.py` for a future multi-node
deployment, but v1 ships single-node only.

---

## 9. Evaluation strategy (v1)

Three buckets, all anchored on synthetic ground truth. Reported on
held-out users.

### 9.1 Bucket 1, theta recovery

**Headline psychometric claim.** Generate N = 5000 synthetic users at
the recovery preset, fit ma-irt, measure recovery on a 80/20 user
split.

**Metrics.**
- `RMSE(theta_hat_t, theta_true)` vs t under three policies (random,
  Fisher-only, full controller).
- `Pearson r(theta_hat_T, theta_true)` at session end.
- Marginal reliability `rho_marginal = 1 - E[sigma_t^2] / Var(theta_hat)`.

**Reporting.** Mean and 95 percent bootstrap CI over 5 seeds. Headline
plot: RMSE-vs-t curves for the three policies on one set of axes.

### 9.2 Bucket 2, Hit@K and NDCG@K

**Headline recommender claim.** Held-out synthetic users, run through
the full system to t = T_max (or terminate naturally), take the final
q_T, rank the O*NET pool. Compare top-K against the user's synthetic
ground-truth like probabilities.

**Metrics.**
- `Hit@K` at K in {5, 10, 20}.
- `NDCG@K` at K in {5, 10, 20} using the ground-truth probability as
  the relevance score.
- `MRR`.

**Baselines.**
- Random.
- Popularity (most-liked occupations from training users).
- Theta-only retrieval (UserTower with h_t zeroed).
- Cosine-only (no UserTower, theta_T as the 1D query against the
  z-scored delta_j).
- CaRReL-stripped replication (theta_t cosine against the 2D
  fdata-style feature space, for the negative control story).
- ReflectionLayer ablation (use likes 1 to k-1 to adjust q_t,
  evaluate against like k, with and without the layer).

### 9.3 Bucket 3, dialogue efficiency

**Headline systems claim.** Items asked to reach
`reliability_estimate >= 0.85` under three policies.

**Metrics.**
- `items_to_reliability_0.85` median, IQR.
- `top_K_stability_at_termination` (Jaccard over last 5 updates).
- `exposure_rate_max` and exposure Gini.

### 9.4 Sanity checks

The generator passes Section 5.5. The trained models pass.

| Check | Hard pass? |
|---|---|
| UserTower output is unit-norm | yes |
| Retrieval is deterministic at fixed q_t | yes |
| BeliefTracker drift assertion: ma-irt params unchanged after 100 rate events | yes |
| Reflection collapse adversarial trajectory: top-K does not collapse to last like | yes |
| Cross-seed robustness: train with seed A, evaluate with seed B; disagreement bounded | warn |

### 9.5 Reporting

`rl/results/v1/` directory. One markdown report
`rl/results/v1/RESULTS.md` summarizing all buckets with embedded
PGF figures. Per-figure raw data at `rl/results/v1/data/`.

---

## 10. Future work / v2 (documented, not built)

### 10.1 Bandit DecisionController

**Goal.** Replace the heuristic ask-vs-recommend-vs-terminate rules
with a contextual bandit that learns from logged dialogue data.

**Context features.**
- `theta_hat_t, sigma_t, h_t_summary` (8-dim PCA of h_t).
- `t, items_asked, n_likes_so_far, n_dislikes_so_far`.
- `top_K_jaccard_stability` over last 3 updates.
- `predicted_top_score, predicted_top_margin`.

**Arms.** Three. Ask the next Fisher-optimal question. Surface the
current top-K. Terminate.

**Posterior parameterization.**
- LinUCB (Li et al. 2010, *A Contextual-Bandit Approach to
  Personalized News Article Recommendation*, WWW). Closed-form
  ridge regression per arm.
- Thompson sampling (Russo & Van Roy 2016, *An Information-Theoretic
  Analysis of Thompson Sampling*) with Gaussian posterior.

**Training data needed.** Logged dialogue sessions
`(state_t, action_taken, reward_t)` where reward is a session-end
proxy like "user accepted top-K = 1" or "user clicked through to a
recommendation." None of this exists for v1. The upgrade is gated on
a pilot collecting at least a few hundred logged sessions.

**Tradeoff vs heuristic.** Bandit learns thresholds the heuristic
hand-picks. Expected gain is small (2 to 5 percent on simulated
dialogue efficiency) but the bandit replaces the sensitivity sweep
on five hyperparameters with one learned policy.

**Upgrade path.** Implement at `rl/src/irtrec/controller/bandit.py`
behind a config flag. The heuristic remains the v1 default and the
fallback when no logged data is available.

### 10.2 SNIPS off-policy evaluation

**Goal.** Robust Hit@K under the system policy estimated from logged
likes, when those exist.

**Method.** Self-normalized importance sampling (Saito et al. 2021
RecSys tutorial), clipped propensity ratios.

**When tractable.** When a behavior policy's action probabilities are
known or can be estimated. fdata does not log these so SNIPS would
have high variance there. A pilot deployment that logs the system's
top-K and propensity per session unlocks this.

**Why deferred from v1.** Synthetic data already gives ground-truth
preferences. SNIPS is a robustness check for real-data scenarios.

### 10.3 iEvaLM LLM simulator

**Goal.** Exercise the rating channel that the v1 replay simulator
cannot. The replay simulator only replays response sequences; it
cannot generate session-mid rating events because the synthetic
generator emits them all up front.

**Method.** Use an LLM (e.g., Claude or GPT-4-class) to play
synthetic students with assigned profiles. The LLM answers
questionnaire items, rates recommendations, asks for more questions
if unsatisfied. Cite Wang et al. 2023 EMNLP (*iEvaLM, A New Way to
Evaluate Conversational Recommender Systems*).

**Cost estimate.** Roughly 1500 to 2500 tokens per simulated session.
At 500 simulated sessions for a credible result, around 1M tokens
per evaluation pass, which is single-digit USD on current pricing.

**Risks.**
- The LLM is itself a model, could be unrealistic.
- LLM responses to GPCM items need to be coerced into the K-category
  ordinal scale, which is non-trivial.
- The LLM's "rating" of a job is a judgment, not data. Adds another
  layer of simulator-real gap.

**v1 fallback.** The synthetic generator emits all ratings up front.
The replay simulator replays them in order, including ratings. This
covers the ratings channel functionally but not the "user changes
mind mid-session" dynamic that iEvaLM would exercise.

### 10.4 Option B preference model, RIASEC

**Goal.** Replace Option A's 1D distance with a 6D RIASEC interest
model where each user has a hidden 6D profile and likes are a
function of 6D cosine.

**User-side.**
- Each user has a hidden 6D RIASEC profile `u ~ N(mu_RIASEC,
  Sigma_RIASEC)` with realistic correlations from Tracey & Rounds
  1995 (six-dimensional spherical model).
- The questionnaire is RIASEC-shaped (alternating R, I, A, S, E, C
  items). Responses come from a multi-trait 2PL kernel.
- ma-irt fits 1D theta which captures the dominant RIASEC component
  (typically the user's strongest interest).

**Job-side.** O*NET occupations have a 3-letter Holland code (e.g.,
"RIA"). Convert to a continuous 6D occupation vector by giving the
first code weight 0.6, second 0.3, third 0.1.

**Preference.** `P(IsLiked | u, j) = sigmoid(lambda * cosine(u_6D,
v_6D))`.

**What this honestly shows.** The 1D theta degradation. ma-irt
captures whichever dimension of the 6D profile is most informative
about the user's questionnaire responses; the recommender then has
to recover the rest of the RIASEC profile from h_t. This is the
hardest test for the deep encoder hidden state as a sufficient
statistic.

**Why deferred from v1.** Demonstrating the pipeline first under the
easier Option A is the v1 contribution. Option B is the followup
that addresses "but real vocational matching is 6D, not 1D."

### 10.5 2D MIRT path

**Goal.** Validated 2D MA-GPCM in ma-irt, with rotation handling.

**Plan.**
- Anchor-item identifiability. Designate K items per dimension as
  pure-loading (alpha non-zero on one dimension only). This is
  confirmatory MIRT.
- Procrustes alignment of recovered 2D theta against the
  ground-truth basis.
- Recovery sweep on synthetic 2D data with three correlation
  structures (orthogonal, hexagonal, near-collinear).

**Pre-conditions.** Successful 1D recovery in v1. Test infrastructure
in `ma-irt/tests/test_mirt_recovery.py`.

**Reference.** Reckase 2009, *Multidimensional Item Response Theory*,
Springer. Lord 1980 for the identifiability conditions.

### 10.6 Cross-simulator robustness

**Goal.** Standard mitigation for the simulator-real gap. Train
inside one simulator family, evaluate inside another. Bound the
disagreement.

**Plan.**
- Train UserTower with ma-irt seed A using DKVMN encoder.
- Evaluate inside ma-irt seed B with Transformer encoder. The
  generator (Section 5) is rerun with the alternative encoder family
  to produce a parallel synthetic universe.
- Report Hit@K disagreement, RMSE disagreement, dialogue efficiency
  disagreement.

**Reference.** Liu et al. 2025 (AdvKT, ECML-PKDD), single-step-train
vs multi-step-inference framing.

### 10.7 Real data integration

**Goal.** Path from fully synthetic to a small real-data pilot.

**Components needed.**
- IRB review for human-subjects research.
- Front-end UI for a vocational questionnaire (start with O*NET
  Interest Profiler short form, 60 items, public domain).
- Session-level logging schema. At minimum
  `(user_id_hash, item_id, response, t_event)` and
  `(user_id_hash, action, top_K, scores, t_event)` for every
  controller event.
- Minimum N for a credible held-out user claim. ASSISTments-style
  back-of-envelope is roughly 200 users with at least 30 responses
  each for a 80/20 split that supports significance testing on
  Hit@10 vs the heuristic baseline.

**Ethical caveats.** Vocational guidance recommendation has documented
gender, race, and SES bias when grounded in observed occupational
distributions. DIF analysis and measurement invariance across
subgroups is mandatory. Cite AERA, APA, NCME 2014 Standards.

**v1 stance.** Out of scope. v1 is a methodology paper on synthetic
data.

### 10.8 LLM-as-orchestrator

**Goal.** Replace the DecisionController with an LLM that reads the
session state and decides actions in natural language.

**Sketch.** Prompt an LLM with `{belief, history, current_top_K,
user_preferences}` per event, ask it to choose ask / recommend /
terminate / offer-more, plus optionally generate a natural language
explanation for the user.

**Risks.** LLM latency. LLM hallucination. The interpretability gain
is real but the policy quality is bounded by the LLM's reasoning,
which is hard to evaluate.

**v3 territory.** After bandit v2 is shown to add value over
heuristic v1.

---

## 11. Milestones

Each milestone has a definition of done (DoD) crisp enough that a
test passes or fails. Tasks 157 to 163 in the in-conversation tracker
mirror these.

### M0. Spec lock + O\*NET data prep

**Scope.** Lock the eight decisions in `rl/docs/spec.md`. Download
O\*NET 2024, parse into the per-occupation feature table at
`rl/artifacts/onet_v1.parquet`. Validate `work_zone` distribution.

**Deliverables.**
- `rl/docs/spec.md` (this plan compressed to a one-page committed
  contract).
- `rl/scripts/build_onet_pool.py`.
- `rl/artifacts/onet_v1.parquet` with all columns from Section 5.2.

**DoD.** `spec.md` committed. All occupations in the parquet have
non-null `title`, `description`, `work_zone in [1, 5]`. At least 800
occupations present after filtering.

**Cost.** 2 to 3 agent rounds. 0 GPU hours.

### M1. ma-irt online step API PR

**Scope.** The H1 surface from
[`drl_mairt_synthesis.md`](drl_mairt_synthesis.md). PR against
`ma-irt` main.

**Deliverables.**
- `ma-irt/models/base.py` (the `EncoderDecoderModel.step` method).
- Per-encoder `forward_with_state` on DKVMN, LSTM, Transformer.
- Per-decoder `compute_logits_from_state` on GPCM, Rasch, Binary.
- `StepState` dataclass.
- `freeze_irt(flag)` helper.
- `ma-irt/tests/test_step_api.py` (parity + microbenchmark).
- `ma-irt/docs/step_api.md`.

**DoD.** Iterated `step` matches batched `forward` to atol = 1e-5 on
logits, probs, theta, alpha, beta. CPU step latency under 20 ms
(DKVMN), 10 ms (LSTM), 40 ms (Transformer at t = 200).

**Cost.** 8 to 10 agent rounds. 2 to 4 GPU hours.

**Critical path.** Single load-bearing prerequisite for M4 onward.

### M2. `rl/` skeleton + O\*NET pool registration + ItemTower precomputation

**Scope.** Build the rl/ directory tree (Section 12). Implement
`ItemTower`, `RetrievalIndex`, `pool.py`. Precompute v_j for the
O\*NET pool.

**Deliverables.**
- `rl/src/irtrec/retrieval/{item_tower,index,pool}.py`.
- `rl/scripts/register_pool.py`.
- `rl/artifacts/onet_v1_embed.npy` (precomputed v_j).
- `rl/tests/test_retrieval.py`.

**DoD.** Cosine retrieval correctness test passes. Pool swap test
swaps in a fake 100-occupation pool and asserts the system still
returns sane top-K with no retraining of ItemTower's head.

**Cost.** 4 to 6 agent rounds. 0 GPU hours.

### M3. Synthetic data generator (Option A)

**Scope.** Section 5 generator, end to end. Two presets (dev N = 500,
recovery N = 5000).

**Deliverables.**
- `rl/src/irtrec/datagen/{synth_users,synth_likes,onet_pool_attach}.py`.
- `rl/configs/sim_v1_dev.yaml`, `rl/configs/sim_v1_recovery.yaml`.
- `rl/scripts/build_synthetic_dataset.py`.
- `rl/tests/test_synth_generator.py`.

**DoD.** All sanity checks in Section 5.5 pass at the recovery
preset. Reproducibility: two runs with the same seed produce
byte-identical files.

**Cost.** 6 to 8 agent rounds. 1 to 2 GPU hours (the ma-irt-fit
sanity check is the GPU cost).

### M4. UserTower + BeliefTracker + Trained Retrieval

**Scope.** Train UserTower on synthetic likes. Wire BeliefTracker.

**Deliverables.**
- `rl/src/irtrec/belief/tracker.py`.
- `rl/src/irtrec/retrieval/user_tower.py`.
- `rl/scripts/train_user_tower.py`.
- `rl/artifacts/user_tower_v1.pt`.

**DoD.** Trained user tower beats random and theta-only retrieval on
held-out synthetic Hit@10 by a non-trivial margin (target +20% over
theta-only). Belief tracker test asserts ma-irt params unchanged
after 100 `on_rate` events.

**Cost.** 6 to 8 agent rounds. 4 to 8 GPU hours.

### M5. Policy components + Service + E2E smoke

**Scope.** FisherItemSelector, ReflectionLayer, heuristic
DecisionController, FastAPI Service. End-to-end smoke test.

**Deliverables.**
- `rl/src/irtrec/policy/{fisher_selector,reflection}.py`.
- `rl/src/irtrec/controller/heuristic.py`.
- `rl/src/irtrec/service/app.py`.
- `rl/tests/test_{fisher_selector,reflection_cap,controller,e2e}.py`.

**DoD.** E2E smoke test drives one held-out synthetic student through
the full loop (responses, ratings, stop) and produces a top-K with
the expected confidence flag. Reflection adversarial collapse test
passes.

**Cost.** 4 to 6 agent rounds. 0 GPU hours.

### M6. Evaluation harness + headline plots

**Scope.** Three buckets from Section 9. Sensitivity sweep on the
five DecisionController thresholds.

**Deliverables.**
- `rl/scripts/eval_{recovery,retrieval,dialogue}.py`.
- `rl/results/v1/RESULTS.md`.
- Headline plots at `rl/results/v1/plots/`.

**DoD.** Three buckets reported with bootstrap CI over 5 seeds. The
publishable claim from Section 11 (positioning) is supported or
honestly refuted.

**Cost.** 6 to 8 agent rounds. 4 to 8 GPU hours.

### M7, M8 (deferred, in Section 10)

LLM simulator (M7), bandit DecisionController (M8). Not in v1.

---

## 12. File layout

```
deep-mirt/
├── ma-irt/                                       (existing, light touch in M1)
│   ├── models/
│   │   ├── base.py                               (M1, +step method)
│   │   ├── components/irt.py                     (M1, +sigma_t helpers)
│   │   ├── encoders/                             (M1, +forward_with_state)
│   │   └── decoders/                             (M1, +compute_logits_from_state)
│   ├── tests/test_step_api.py                    (M1, new)
│   └── docs/step_api.md                          (M1, new)
├── rl/                                           (new in v1)
│   ├── pyproject.toml
│   ├── README.md
│   ├── docs/
│   │   └── spec.md                               (M0)
│   ├── src/
│   │   └── irtrec/
│   │       ├── __init__.py
│   │       ├── belief/
│   │       │   ├── __init__.py
│   │       │   └── tracker.py                    (M4)
│   │       ├── retrieval/
│   │       │   ├── __init__.py
│   │       │   ├── user_tower.py                 (M4)
│   │       │   ├── item_tower.py                 (M2)
│   │       │   ├── index.py                      (M2)
│   │       │   └── pool.py                       (M2)
│   │       ├── policy/
│   │       │   ├── __init__.py
│   │       │   ├── fisher_selector.py            (M5)
│   │       │   └── reflection.py                 (M5)
│   │       ├── controller/
│   │       │   ├── __init__.py
│   │       │   ├── heuristic.py                  (M5)
│   │       │   └── bandit.py                     (deferred, Section 10.1)
│   │       ├── service/
│   │       │   ├── __init__.py
│   │       │   ├── app.py                        (M5)
│   │       │   └── session_store.py              (M5)
│   │       ├── sim/
│   │       │   ├── __init__.py
│   │       │   ├── replay.py                     (M6)
│   │       │   └── llm_simulator.py              (deferred, Section 10.3)
│   │       └── datagen/
│   │           ├── __init__.py
│   │           ├── onet_pool.py                  (M0)
│   │           ├── synth_users.py                (M3)
│   │           └── synth_likes.py                (M3)
│   ├── configs/
│   │   ├── sim_v1_dev.yaml                       (M3)
│   │   └── sim_v1_recovery.yaml                  (M3)
│   ├── scripts/
│   │   ├── build_onet_pool.py                    (M0)
│   │   ├── register_pool.py                      (M2)
│   │   ├── build_synthetic_dataset.py            (M3)
│   │   ├── train_user_tower.py                   (M4)
│   │   ├── serve.py                              (M5)
│   │   ├── eval_recovery.py                      (M6)
│   │   ├── eval_retrieval.py                     (M6)
│   │   └── eval_dialogue.py                      (M6)
│   ├── tests/
│   │   ├── test_belief_tracker.py                (M4)
│   │   ├── test_retrieval.py                     (M2)
│   │   ├── test_reflection_cap.py                (M5)
│   │   ├── test_fisher_selector.py               (M5)
│   │   ├── test_controller.py                    (M5)
│   │   ├── test_e2e.py                           (M5)
│   │   └── test_synth_generator.py               (M3)
│   ├── artifacts/                                (gitignored)
│   │   ├── onet_v1.parquet                       (M0)
│   │   ├── onet_v1_embed.npy                     (M2)
│   │   └── user_tower_v1.pt                      (M4)
│   └── results/                                  (gitignored except v1/)
│       └── v1/
│           ├── RESULTS.md                        (M6)
│           ├── plots/                            (M6)
│           └── data/                             (M6)
└── docs/                                         (existing, shared)
    └── drl_mairt_plan_v1.md                      (THIS FILE)
```

Gitignore additions go in `deep-mirt/.gitignore` at M0:

```
rl/artifacts/
rl/results/*/data/
!rl/results/v1/plots/
```

---

## 13. Risks with mitigations

Lifted from
[`docs/cleanup/_drl_interactive_design.md`](cleanup/_drl_interactive_design.md)
and extended with three new risks from the synthetic-only choice.

| # | Risk | Mitigation |
|---|---|---|
| R1 | 1D theta drowned by high-D h_t in the user tower | Learnable scalar `s_theta` initialized at `sqrt(d_h)`. M6 ablation removes h_t entirely and removes theta entirely to quantify each channel. |
| R2 | ReflectionLayer collapse to the user's most recent like (the CaRReL failure mode) | Hard cosine-shift cap at 0.2 per step. Per-session reset. Adversarial test in M5. |
| R3 | Like/dislike signal leaking into ma-irt | Runtime assertion in BeliefTracker. Test in M4 that asserts ma-irt's parameter tensors are byte-equal after 100 rate events. |
| R4 | Frozen BGE anisotropy biasing cosine scores | L2-normalize after the item-tower head, not on raw text embeddings. Optional whitening step measured on the registered pool, deferred. |
| R5 | Heuristic controller thresholds arbitrary | Config-driven. Sensitivity sweep on 5 thresholds in M6. |
| R6 | Pool swap fails on a new structured schema | JobPoolSpec contract; new structured fields require a head retrain (not a full retrain). M2 includes a pool-swap smoke test. |
| R7 | Encoder retraining drift makes a trained UserTower stale | Pin a specific ma-irt checkpoint in `rl/artifacts/`. Retraining is a versioned operation that retriggers UserTower training. Documented in M1's `step_api.md`. |
| R8 | The 1D-theta constraint conflicts with realistic vocational matching (which is 6D RIASEC) | Locked for v1. Option B in Section 10.4 addresses this. The contribution claim is bounded accordingly in Section 11. |
| R9 | Cross-encoder retraining drift | Pin checkpoint version. Documented. |
| R10 | Sympson-Hetter exposure missing in v1 | Randomesque is used as the v1 fallback. Sympson-Hetter is a v2 add when actual exposure rates become a publication blocker. |
| **R11 (new)** | **No real-data anchor at all in v1** | Frame as a methodology paper. Cite the published precedents (ExRec 2025, ALPN 2023) that did simulator-only. Add Section 10.7 as the followup. |
| **R12 (new)** | **The synthetic preference model is "too easy" for any retriever, inflating Hit@K** | Add a noise floor (per-user random noise on the sigmoid utility, default sigma_noise = 0.5). Include decoy items in the pool. Report Hit@K both with and without the noise. |
| **R13 (new)** | **O*NET attribute choice for delta_j is a single point of failure** | Sensitivity analysis across three candidate delta_j sources (`work_zone`, `education_zscore`, `complexity_composite`) in M6. Report the headline number with all three. |

---

## 14. Active task tracking

Mirror of the in-conversation tracker. The TaskList tooling is the
source of truth; this checklist is the human-readable copy.

- [ ] **M0**, spec lock + O\*NET data prep (task 157)
- [ ] **M1**, ma-irt online step API PR (task 158, blocked by M0)
- [ ] **M2**, rl/ skeleton + ItemTower (task 159, blocked by M0)
- [ ] **M3**, synthetic data generator (task 160, blocked by M0)
- [ ] **M4**, UserTower + BeliefTracker + Retrieval training (task 161, blocked by M1, M2, M3)
- [ ] **M5**, Policy + Service + E2E (task 162, blocked by M4)
- [ ] **M6**, Evaluation harness + plots (task 163, blocked by M5)

M0 has no blockers; M1 / M2 / M3 can run in parallel after M0
completes; M4 needs all three; M5 after M4; M6 after M5. M7 (LLM
simulator) and M8 (bandit) are deferred and live in Section 10.

---

## 15. The first concrete step

Begin M0.

1. Create `deep-mirt/rl/` with `pyproject.toml`, `README.md`,
   directory tree per Section 12.
2. Write `rl/docs/spec.md` as a one-page committed contract of the
   eight decisions, formatted as YAML.
3. Download O*NET 2024 from the public-domain endpoint at
   [onetcenter.org/database.html](https://www.onetcenter.org/database.html).
   Implement `rl/scripts/build_onet_pool.py` to parse the relevant
   tab-separated files into the per-occupation feature table at
   `rl/artifacts/onet_v1.parquet`.
4. Validate `work_zone` distribution. Confirm at least 800
   occupations have all required fields.
5. Open the M1 branch `feat/online-step-api` against ma-irt.

This unblocks M1, M2, M3 in parallel. M4 follows once all three
complete. M5 follows M4. M6 closes the v1 cycle.

---

## 16. References

- Chang, H.-H., & Ying, Z. (1996). A global information approach to
  computerized adaptive testing. *Applied Psychological Measurement*,
  20(3), 213–229.
- Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks
  for YouTube recommendations. *RecSys 2016*.
- Khajah, M., Lindsey, R. V., & Mozer, M. C. (2016). How deep is
  knowledge tracing? *EDM 2016*, arXiv:1604.02336.
- Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting
  items for computerized adaptive tests. *Applied Measurement in
  Education*, 2(4), 359–375.
- Lei, W. et al. (2020). Estimation-Action-Reflection: Towards Deep
  Interaction Between Conversational and Recommender Systems.
  *WSDM 2020*.
- Li, L. et al. (2010). A contextual-bandit approach to personalized
  news article recommendation. *WWW 2010*.
- Lord, F. M. (1980). *Applications of Item Response Theory to
  Practical Testing Problems.* Erlbaum.
- Mislevy, R. J. (2018). *Sociocognitive Foundations of Educational
  Measurement.* Routledge.
- Muraki, E. (1992). A generalized partial credit model. *Applied
  Psychological Measurement*, 16(2), 159–176.
- Muraki, E. (1993). Information functions of the generalized partial
  credit model. *Applied Psychological Measurement*, 17(4), 351–363.
- Ozyurt, A. et al. (2025). ExRec, Personalized Exercise
  Recommendation with Semantically-Grounded Knowledge Tracing.
  arXiv:2507.11060.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory.*
  Springer.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP-IJCNLP*.
- Russo, D., & Van Roy, B. (2016). An information-theoretic analysis
  of Thompson sampling. *JMLR* 17.
- Saito, Y. et al. (2021). Counterfactual Learning and Evaluation for
  Recommender Systems. *RecSys 2021* (tutorial).
- Sympson, J. B., & Hetter, R. D. (1985). Controlling item-exposure
  rates in computerized adaptive testing. *Proceedings of the 27th
  annual meeting of the Military Testing Association*.
- Talts, S. et al. (2018). Validating Bayesian inference algorithms
  with simulation-based calibration. arXiv:1804.06788.
- Volkovs, M. et al. (2021). ZESRec, Zero-Shot Recommender Systems.
  arXiv:2105.08318.
- Wang, X. et al. (2023). iEvaLM, A New Way to Evaluate
  Conversational Recommender Systems. *EMNLP 2023*.
- Yeung, C. K. (2019). Deep-IRT, Make Deep Learning Based Knowledge
  Tracing Explainable Using Item Response Theory. arXiv:1904.11738.
- Yi, X. et al. (2019). Sampling-Bias-Corrected Neural Modeling for
  Large Corpus Item Recommendations. *RecSys 2019*.
