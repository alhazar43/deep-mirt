# MT results: real-task in-context adaptation (English to Dinka, FLORES-200)

The real-task companion to the synthetic E1b. The goal is to see whether a
language model's in-context adaptation, shown on a synthetic label-remapping
task in E1b, also appears on a genuine task. A Qwen2.5 model translates
English into Dinka (dik_Latn, a genuinely low-resource language) at
increasing shot counts k, scored by chrF. True demonstrations use correct
parallel pairs, the shuffled control uses mismatched pairs (no translation
signal), the analog of E1's shuffled-label priming control. Code in
`deep_irt/traj_mt/generate_mt.py`.

## Result (self-gate, Qwen2.5-1.5B-Instruct, 93 FLORES sentences)

| k | true chrF | shuffled chrF |
|---|---|---|
| 0 | 12.4 | 12.4 |
| 16 | 18.0 | 10.8 |

- Gain k=0 to k=16 (true): **+5.6 chrF**.
- True minus shuffled at k=16: **+7.1 chrF**.
- Self-gate: **PASS**.

## Reading

In-context adaptation appears on a real task. Translation quality rises
with shot count under correct demonstrations and does NOT rise under
shuffled demonstrations, so the gain is genuine in-context learning of the
English-to-Dinka mapping, not format priming. This is the real-task analog
of E1b (synthetic label-remapping), and together they show the machine
front's in-context adaptation is not an artifact of the synthetic setup, it
holds on a real low-resource translation task with the same
learning-versus-priming separation.

## Status and scope

The full cross-model IRT ladder (0.5B, 1.5B, 3B over the full k grid, with
theta(k) placed on a shared scale and the cross-model magnitude scaling, via
`run_e1 --dir`) is compute-bound on the 8 GB laptop GPU. Full autoregressive
decoding per sentence is slow, and a three-model run is multi-hour, so it is
deferred rather than re-run for a marginal strengthening of an already clear
positive. The single-model gate establishes the headline. To complete the
ladder, run `python -m deep_irt.traj_mt.generate_mt` to produce the response
files, then `python -m deep_irt.traj_icl.run_e1 --dir deep_irt/traj_mt/outputs --tag mt`.

## Limitations

One language pair, one model size for the reported curve, and a two-point
gate (k=0 and k=16) for the headline (intermediate shot counts were started
but the run is compute-bound). chrF is a surface metric and a binarized chrF
would be the IRT response. The result establishes the existence of real-task
in-context adaptation with a clean priming control, not its precise rate or
its cross-model scaling, which await the compute-bound full ladder.
