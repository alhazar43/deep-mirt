# Retrain Plan: Separate Theta Extraction Pathway

## The Change
In `models/kt_gpcm.py`, the current forward loop uses a SHARED summary:
```python
summary_input = torch.cat([read_t, q_t], dim=-1)  # (B, value_dim + key_dim)
summary_t = self.summary(summary_input)             # (B, summary_dim)
theta_t = self.irt.ability_network(summary_t)        # theta from shared summary
alpha_t = self.irt.discrimination_network(cat([summary_t, q_t]))  # alpha from shared + q
```

The fix: theta should come from read_t ONLY (student state), not from [read_t; q_embed]:
```python
theta_t = self.ability_network(read_t)               # theta from memory state only
summary_input = torch.cat([read_t, q_t], dim=-1)
summary_t = self.summary(summary_input)
alpha_t, beta_t = self.irt(summary_t, q_t)           # alpha/beta still use item info
```

## Evidence
Linear probe on read_t alone recovers theta at r=0.876 (pooled) vs model's extracted theta at r=0.656.
The q_embed contamination in the summary adds ~46% item-level variance to theta.

## Experiments That Need Retraining

### Priority 1: Validate the change
- [ ] K=4, Q=200, seed=42 only (static theta) — compare r_theta before/after
- [ ] K=4, Q=200, dynamic theta (50k students) — compare per-timestep r

### Priority 2: Main results (Table 3 - recovery)
- [ ] K=3, Q=200, 5 seeds
- [ ] K=4, Q=200, 5 seeds
- [ ] K=5, Q=200, 5 seeds
- [ ] K=6, Q=200, 5 seeds
Total: 20 runs × ~15 min = ~5 hours

### Priority 3: Prediction table (Table 1)
- [ ] K=3,4,5,6 × 5 seeds = 20 runs
(Same models as Priority 2 — prediction and recovery come from same checkpoint)

### Priority 4: Binary compatibility (Table 2)
- [ ] K=2, Q=200, 5 seeds
- [ ] ASSIST2015, Synthetic-5

### Priority 5: Ablations
- [ ] Monotonicity: K=4, 5 seeds × 2 variants = 10 runs
- [ ] Imbalance: K=4, 4 conditions × 5 seeds = 20 runs
- [ ] Item representation: Q=200,500,1000,2000 × 5 seeds × 3 encodings = 60 runs

### Priority 6: Dynamic theta
- [ ] N=50k, K=4, new architecture

## Total: ~130 training runs
At ~15 min each on RTX 4060: ~32 hours
Can parallelize with run_all_experiments.sh

## Expected Impact
- r_theta should improve (read_t probe shows r=0.876 vs current 0.656 per-timestep)
- r_alpha may change (discrimination network still gets full info)
- r_beta should be similar (threshold network still gets q_embed)
- Prediction metrics (QWK, ACC) may slightly decrease (theta has less item info)
- Dynamic theta recovery should improve significantly

## Validation Result: PASSED
K=4 s42 static theta, 9 epochs only:
- Per-timestep pooled r_theta: 0.656 → 0.900 (+37%)
- Per-student mean r_theta: 0.917 → 0.953 (+4%)
- Exceeds the linear probe ceiling (0.876) on per-timestep recovery
- Proceed with full retrain.
