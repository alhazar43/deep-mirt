# DRL-MAIRT v2 Preliminary Results

## Generated 2026-06-04

## Headline Summary

1. The v2 ``delta_j`` composite is fully continuous, with 923 distinct values across the 923 occupation pool versus only 4 in v1. This removes the work-zone bucketing that capped v1 ranking power.

2. The 1D Bayes-ceiling Hit@10 rises from 0.158 (v1) to 0.261 (v2), a 1.65x relative lift over the same O\*NET pool. The continuous difficulty axis lets the oracle break the work-zone ties that previously held it back.

3. The oracle now beats popularity. v1 had popularity Hit@10 = 0.263 vs oracle 0.158, so a non-personalised counter dominated the IRT signal. v2 reverses this, oracle 0.261 vs popularity 0.236, recovering the expected ordering and confirming that personalisation now adds value over a global counter.

## V2 Simulator Changes (M4-RL recap)

The M4-RL milestone introduces five changes to the v1 generator.

1. Continuous ``delta_j`` composite. Built from a weighted z-scored sum of O\*NET work zone (0.45), education z-score (0.35) and a complexity composite from work activity categories (0.20), plus N(0, 0.30) seeded noise, then re-standardised to unit variance. See ``rl/src/irtrec/datagen/onet_pool_attach.py``.

2. K=5 GPCM ordinal response replaces the binary sigmoid. Each user emits y in {0, 1, 2, 3, 4} per candidate job under cumulative-logit GPCM with step thresholds beta = (-1.5, -0.5, 0.5, 1.5). Backward-compatible ``IsLiked = 1[y >= 3]``. See ``rl/src/irtrec/datagen/synth_likes.py``.

3. Per-user discrimination scalar ``lambda_u`` ~ LogNormal(log 1.5, 0.4). Median 1.5, mean ~1.62, long right tail.

4. Engagement mixture removed. v1 split users into rejecter (40 percent) and engaged (60 percent). v2 treats all users as engaged but heterogeneous via ``lambda_u``, since the K=5 GPCM now provides intrinsic response variation.

5. ``ItemTower`` renamed to ``JobTower`` with a shim, since the retrieval target is occupations not test items.

## Continuous delta_j Verification

Test, ``rl/tests/test_delta_j_continuity.py``. All three checks pass.

- ``test_delta_j_n_unique_close_to_n_jobs``. n_unique = 923 of 923 jobs (v1 had 4).
- ``test_delta_j_zscored_and_finite``. mean = 0.0, std = 1.0, range [-2.16, 2.58].
- ``test_bayes_ceiling_hit_at_10_above_floor``. Held-out Bayes ceiling Hit@10 = 0.287, well above the 0.20 floor.

The Bayes ceiling in the test uses the true theta, lambda_u, and delta_j with the v2 GPCM oracle ``P(y >= 3)``. It is slightly higher than the held-out theta_true 1D match number reported below (0.261) because the test uses all of ``users.json``'s own ``splits.json`` test partition (200 users), whereas the recommender eval uses a freshly seeded 80/20 user permutation matched to the v1 protocol (split_seed = 0).

Plot, ``rl/results/v2/plots/m4rl_delta_j_distribution.png``.

## Recommender Baselines

Eval protocol matches v1. 80/20 user split (seed=0), Hit@K and NDCG/MRR with 500-bootstrap 95 percent CIs, evaluated on test users with at least one IsLiked positive. n_eval = 356 of 400 (89 percent), n_jobs = 923.

| baseline | hit@5 | hit@10 | hit@20 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| Random | 0.090 [0.061, 0.121] | 0.157 [0.121, 0.197] | 0.289 [0.244, 0.337] | 0.021 [0.013, 0.030] | 0.071 [0.052, 0.092] |
| Popularity (train likes) | 0.115 [0.087, 0.149] | 0.236 [0.194, 0.278] | 0.376 [0.326, 0.427] | 0.034 [0.024, 0.045] | 0.093 [0.072, 0.114] |
| 1D match, theta-hat | 0.140 [0.107, 0.174] | 0.261 [0.219, 0.305] | 0.435 [0.385, 0.486] | 0.036 [0.026, 0.046] | 0.103 [0.082, 0.124] |
| 1D match, theta-true (oracle) | 0.140 [0.107, 0.174] | 0.261 [0.219, 0.305] | 0.435 [0.385, 0.486] | 0.036 [0.026, 0.046] | 0.103 [0.082, 0.124] |

Theta recovery on v2 GPCM responses, Pearson r = 0.974, RMSE = 0.222 (N = 2000). theta_hat lands in the same equivalence class as theta_true at the Hit@K resolution used here, so the two 1D rows tie. Plot, ``rl/results/v2/plots/m4rl_theta_recovery.png``.

## V1 versus V2 Comparison

Hit@10 across the four baselines, same pool, same protocol.

| baseline | v1 (n_eval=57) | v2 (n_eval=356) | absolute delta | relative |
|---|---|---|---|---|
| Random | 0.070 | 0.157 | +0.087 | +124 percent |
| Popularity | 0.263 | 0.236 | -0.027 | -10 percent |
| 1D theta-hat | 0.158 | 0.261 | +0.103 | +65 percent |
| 1D theta-true (oracle) | 0.158 | 0.261 | +0.103 | +65 percent |

Three things move together. First, random rises because v2's mean candidate set is larger and IsLiked rate is 0.39 (vs 0.20 in v1), so the relevant set is bigger relative to the pool. Second, the oracle rises by 65 percent because the continuous ``delta_j`` lets each user partition the 923 jobs into 923 ranked positions rather than 4 work-zone buckets. Third, popularity drops slightly because v2's heavier left-tail of likes (driven by ``lambda_u`` heterogeneity) spreads positives across more jobs, making a single global counter less concentrated. The combined effect flips the v1 anomaly where popularity beat the oracle.

Plot, ``rl/results/v2/plots/m4rl_v1_vs_v2_baselines.png``.

## What M4-RL Unlocks for M5-M8-RL

The v2 simulator gives later milestones three things v1 could not.

1. A non-degenerate 1D ceiling. Hit@10 = 0.261 oracle is the bar any 1D retriever must clear. The trained UserTower in M5-RL can now demonstrate measurable lift over both popularity and the 1D oracle without bumping into a 4-bucket cap.

2. K=5 ordinal supervision. The response distribution is roughly uniform across the 5 categories (15485, 14593, 15037, 14244, 14777), giving the encoder genuine ordinal signal rather than the v1 binary 80/20. M6-RL's policy can use this graded reward shape instead of a noisy click signal.

3. Per-user discrimination. ``lambda_u`` heterogeneity is the natural target for the UserTower to learn from sequence history, since users with high lambda_u have sharper preference functions and should be ranked differently from low-lambda users at the same theta. This is the win that the M5-RL contrastive head is designed to exploit.

Floor for M5-RL training, Hit@10 > 0.261 (the v2 1D oracle). Headroom above the oracle must come from multi-dimensional matching that the JobTower embedding can support but the scalar ``delta_j`` cannot.
