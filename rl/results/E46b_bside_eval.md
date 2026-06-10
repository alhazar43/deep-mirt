# OrdRec E4.5 four-policy evaluation

Config: `rl\configs\ppo_synth_e46b.yaml`.
PPO checkpoint: `outputs\ordrec_synth_e46b\best.pt`.
BC checkpoint: `outputs\ordrec_synth_e46b\bc_warmstart.pt`.

## Notes

- B-side of E4.5/E4.6b A/B comparison.
- Buffer fix (RC1): one entry per env-step per row; terminal r_voi enters training.
- Reward recalibration (RC2): w_expo=0.02, r_max=0.40 (was 0.10, 0.20 in E4.5).
- Stratified probe sampler (B5): difficulty-stratified, 5 strata.
- BC teacher uses top-5 soft target (R3).
- Evaluated on test split, 200 episodes per policy.
- PPO checkpoint: outputs\ordrec_synth_e46b\best.pt.
- BC checkpoint: outputs\ordrec_synth_e46b\bc_warmstart.pt.

## Mean episode return

| policy | mean return | std | n_episodes |
| --- | --- | --- | --- |
| trained PPO | -0.5699 | 0.4005 | 6400 |
| BC-only | -0.5609 | 0.3513 | 6400 |
| max-Fisher | -0.5638 | 0.3406 | 6400 |
| uniform random | -0.5368 | 0.2835 | 6400 |

## Per-component reward means

| policy | r_info | r_cost | r_expo | r_voi |
| --- | --- | --- | --- | --- |
| trained PPO | +0.0020 | -0.2500 | -0.0048 | -0.0322 |
| BC-only | +0.0016 | -0.2500 | -0.0052 | -0.0269 |
| max-Fisher | +0.0015 | -0.2500 | -0.0074 | -0.0260 |
| uniform random | +0.0011 | -0.2500 | +0.0000 | -0.0195 |

## Exposure diagnostics (fleet EMA at end of policy eval)

| policy | max | p99 | p95 | p50 | frac > r_max |
| --- | --- | --- | --- | --- | --- |
| trained PPO | 0.6492 | 0.6464 | 0.4201 | 0.0235 | 0.0597 |
| BC-only | 0.6484 | 0.6396 | 0.4944 | 0.0220 | 0.0647 |
| max-Fisher | 0.6575 | 0.6502 | 0.6199 | 0.0216 | 0.0647 |
| uniform random | 0.0872 | 0.0834 | 0.0765 | 0.0646 | 0.0000 |
