"""A1 signed cross-KC influence (goal G1): the ``transfer/`` subpackage on
top of the A4 ``growth/`` substrate (`_planning/design/a1_design.md` v1.1).

This stage builds ONLY the CT0 machinery (section 2.1.1, the make-or-break
per-edge sign-recovery power precondition):

- `model`  -- the signed-influence model (ACT own-gain + fitted zero-
  diagonal signed G, practice-gated and sign-asymmetric-gated) and its
  two-stage fit.
- `synth`  -- the D=3 signed-edge generator (SYN-T-KG / SYN-T-NG twins,
  N-learners and decoupling knobs) injecting a known ``G_true``.
- `ct0`    -- per-edge sign-F1 and the (N, decoupling) power sweep, with
  the CT1 bar and the K-T1 kill decided on the power curve.

Imports run DOWNWARD from `growth/` and `core/` only (never sideways, no
runtime import from `deep_irt`).
"""

from kt_mirt.transfer import ct0, model, synth

__all__ = ["model", "synth", "ct0"]
