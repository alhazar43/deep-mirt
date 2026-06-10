# OrdRec E4.5 four-policy evaluation

Config: `rl\configs\ppo_synth_e45.yaml`.
PPO checkpoint: `outputs\ordrec_synth_e45\best.pt`.
BC checkpoint: `outputs\ordrec_synth_e45\bc_warmstart.pt`.

## Notes

- Probe sampler is uniform (stratified lands in E4.6b).
- K_B credit assignment: full reward on first sub-step (known-suboptimal, fixed in E4.6b). This is side A of the E4.6b A/B comparison.
- Evaluated on test split, 200 episodes per policy.
- PPO checkpoint: outputs\ordrec_synth_e45\best.pt.
- BC checkpoint: outputs\ordrec_synth_e45\bc_warmstart.pt.

## Mean episode return

| policy | mean return | std | n_episodes |
| --- | --- | --- | --- |
| trained PPO | -0.7295 | 0.3419 | 6400 |
| BC-only | -0.7338 | 0.3263 | 6400 |
| max-Fisher | -0.7450 | 0.3222 | 6400 |
| uniform random | -0.5304 | 0.2422 | 6400 |

## Per-component reward means

| policy | r_info | r_cost | r_expo | r_voi |
| --- | --- | --- | --- | --- |
| trained PPO | +0.0013 | -0.2500 | -0.0940 | -0.0221 |
| BC-only | +0.0012 | -0.2500 | -0.0968 | -0.0213 |
| max-Fisher | +0.0011 | -0.2500 | -0.1026 | -0.0211 |
| uniform random | +0.0008 | -0.2500 | +0.0000 | -0.0160 |

## Exposure diagnostics (fleet EMA at end of policy eval)

| policy | max | p99 | p95 | p50 | frac > r_max |
| --- | --- | --- | --- | --- | --- |
| trained PPO | 0.6610 | 0.6451 | 0.5663 | 0.0233 | 0.0746 |
| BC-only | 0.6610 | 0.6451 | 0.5798 | 0.0232 | 0.0697 |
| max-Fisher | 0.6610 | 0.6451 | 0.6178 | 0.0214 | 0.0746 |
| uniform random | 0.0873 | 0.0825 | 0.0792 | 0.0647 | 0.0000 |
