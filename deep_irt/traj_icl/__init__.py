"""traj_icl -- in-context adaptation curves for language models (E1).

Front 1 of the trajectory-recovery program (docs/trajectory_program.md).
A ladder of open-weight models answers a shared multiple-choice item bank
at increasing in-context shot counts k. Each (model, k, condition) is a
static examinee. A 2PL IRT calibration places every examinee on one shared
item scale, the per-model ability theta(k) is the in-context adaptation
curve, and its rate is the comparable estimand. A shuffled-label control
(Min et al. 2022) separates genuine in-context learning from format
priming.

Data flow. generate.py runs the models and writes per-model response
files; irt_fit.py assembles the examinee-by-item matrix and fits the 2PL;
run_e1.py reads theta(k), fits the adaptation rate (reusing traj_synth
metrics), compares true against shuffled, and writes results and plots.
"""
