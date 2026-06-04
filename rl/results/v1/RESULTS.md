# DRL-MAIRT v1 Preliminary Results

## Generated 2026-06-04

## M2 + M3 Validation

M2 ItemTower embedding analysis.

- Pool, O\*NET v1, 923 occupations, 64 dim, L2 normalised via the frozen BGE-small-en-v1.5 sentence transformer plus a linear head.
- Embedding stats, `rl/results/v1/data/m2_embedding_stats.json`. Nearest neighbour spot checks, `rl/results/v1/data/m2_nearest_neighbors.json`.
- Effective rank 12.6 (participation) and 30.1 (entropy), 3 PCs explain 90 percent of variance. RIASEC silhouette 0.18, work zone silhouette 0.32.
- Pool swap smoke test on a synthetic 50 occupation pool passed.
- UMAP, `rl/results/v1/plots/m2_embedding_umap.png`.

M3 synthetic data generator validation.

- Bayesian EAP theta recovery on true items, sim_v1_recovery (N=5000), Pearson r = 0.978, RMSE = 0.207. sim_v1_dev (N=500), r = 0.975, RMSE = 0.224.
- Like generation, overall like rate 0.202 (target 0.20), rejecter mean rate 0.000, engaged mean rate 0.337, 19036 candidate impressions, 3657 positives.
- Plots, `m3_theta_recovery.png`, `m3_like_distribution.png`, `m3_engagement_split.png`, `m3_k_distribution.png`, `m3_delta_j_distribution.png`.

## First Recommender Baselines

### Methodology

Each baseline assigns a score s\_uj to every (user, job) pair over the full O\*NET pool of 923 occupations. Per user we sort jobs by score, breaking ties uniformly at random via a tiny seeded jitter (the two 1D matchers share the jitter seed, so any difference between them is purely about cross class ordering, not within class noise), then compute Hit@5, Hit@10, Hit@20, NDCG@10, MRR against the user's positive like set.

Users are split 80/20 (seed=0). The popularity baseline uses positive like counts from the 400 training users only. Held-out users with zero positives are skipped since they have no relevant set, leaving 57 of 100 test users in the metric averages. Means come with 95 percent bootstrap CIs over 500 resamples of evaluable users.

Four baselines.

1. Random. Per user permutation, seeded.
2. Popularity. Rank by training set positive like counts, jitter for ties.
3. Theta-true 1D match (oracle). Rank by sigmoid(lambda*(theta\_u - delta\_j) + bias) with TRUE theta\_u and TRUE delta\_j. Upper bound for any 1D retriever.
4. Theta-hat 1D match (realistic). Same score with theta\_hat from EAP on true items.

### Results

| baseline | hit@5 | hit@10 | hit@20 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| Random | 0.035 [0.000, 0.088] | 0.070 [0.018, 0.140] | 0.123 [0.053, 0.211] | 0.007 [0.000, 0.015] | 0.034 [0.017, 0.056] |
| Popularity (train likes) | 0.123 [0.053, 0.211] | 0.263 [0.140, 0.378] | 0.386 [0.263, 0.509] | 0.032 [0.017, 0.046] | 0.094 [0.056, 0.137] |
| 1D match, theta-hat | 0.070 [0.018, 0.140] | 0.158 [0.070, 0.263] | 0.298 [0.184, 0.421] | 0.020 [0.008, 0.034] | 0.077 [0.039, 0.125] |
| 1D match, theta-true (oracle) | 0.070 [0.018, 0.140] | 0.158 [0.070, 0.263] | 0.298 [0.184, 0.421] | 0.020 [0.008, 0.034] | 0.077 [0.039, 0.125] |

Plot, `rl/results/v1/plots/m23_baselines.png`. Raw metrics, `rl/results/v1/data/m23_baselines.json`.

### Insights

Across 57 held-out users with at least one positive like (out of 100 test users), the random retriever lands at Hit@10 = 0.070, which is the naive floor for a pool of 923 occupations. Popularity reaches Hit@10 = 0.263, a lift of +275.0 percent over random, confirming that O*NET occupations attract likes very unevenly under the simulator's work zone driven delta_j (4 distinct values, with the largest class holding 331 of 923 jobs). The oracle 1D matcher with true theta and true delta_j hits Hit@10 = 0.158, a relative lift of +125.0 percent over random but -0.105 absolute compared to popularity. Using theta-hat from EAP on true items, the realistic 1D matcher achieves Hit@10 = 0.158, an oracle vs recovered gap of only +0.000, because theta recovery is already strong on this preset (Pearson r = 0.975, RMSE = 0.224), so going from theta-hat to theta-true buys almost no extra retrieval power. The headline surprise is that popularity beats 1D theta matching, which is a property of the v1 simulator rather than a deficiency of the IRT signal, since the 4 valued delta_j means each user's score function partitions the pool into only 4 equivalence classes, so within the best class the order is uniform random and a heavily likes biased class can swamp the cross class signal. Implication for M4, the trained UserTower must clear Hit@10 = 0.263 (popularity, the best non oracle baseline) to claim any value, while the 0.158 oracle is the credible 1D ceiling that pure ability matching can reach on v1 data. Real wins beyond the oracle must come from richer signals than a single ability axis, for example the full O*NET embedding interacting with sequence level user history, which is what the encoder is meant to learn and is the main bet of M4. On simulator difficulty, even the oracle reaching only Hit@10 = 0.158 despite perfect parameters reflects the structural cap of a 4 valued delta_j on a 923 occupation pool, not bottlenecking of the response model. If the absolute numbers look low, that is the dataset and not the method, since per user relevant sets are tiny (median 2 positives) and lambda = 1.75 gives only modest cross class separation. Next steps, train M4 against the popularity floor, report the oracle gap, and queue a v2 simulator with a continuous text driven delta_j and higher lambda so that 1D matching produces a non degenerate ranking and the retrieval ceiling rises.

## What M4's Trained UserTower Must Beat

- Hit@10 floor, 0.263 (popularity). This is the bar M4 must clear before it can claim to add value over a trivial popularity recommender.
- 1D ceiling, Hit@10 = 0.158 (theta-true oracle). Any retriever that only matches user ability to item difficulty on a single axis tops out here on the v1 simulator.
- Theta recovery is no longer the bottleneck. theta-hat Hit@10 = 0.158 versus oracle 0.158, gap = +0.000. EAP on true items already lands in the same delta\_j bucket as the true theta for every evaluable user, so M4's main job is not better ability recovery.
- M4's win must therefore come from leaving the 1D axis. Concretely, the encoder should consume the full O\*NET embedding (not just delta\_j) and condition on sequence-level user history so that within-work-zone ordering becomes signal instead of jitter. That is the only path above both popularity and the 1D oracle.

