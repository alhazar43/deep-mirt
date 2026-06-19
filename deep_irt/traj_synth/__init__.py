"""traj_synth -- synthetic ability-trajectory recovery (experiment E0).

The RQ0 precondition study for the trajectory-recovery program
(docs/trajectory_program.md). Respondents follow a parametric learning
curve theta(t) = theta_inf - (theta_inf - theta_0) exp(-r t) with a known
per-respondent rate r. Responses are generated through the GPCM/binary
decoder, the encoder-decoder is trained under prediction loss, the
per-step ability is read back with model.track(), a curve is fit to the
recovered trajectory, and the recovered rate is scored against the truth.

Key identifiability fact. The encoder's theta scale is arbitrary (IRT has
a location and scale indeterminacy), but the rate r is invariant under any
affine transform of theta, so it is recoverable without fixing the scale.
The absolute ability is not. This is why the rate is the estimand.
"""
