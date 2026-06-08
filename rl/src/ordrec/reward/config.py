"""``RewardConfig`` dataclass for the OrdRec reward composer.

Centralises every numeric knob the four-component reward uses so the
ordinal reward callable can be constructed from a single config object
and serialised to YAML without scattering magic numbers across the
codebase.

See ``docs/ordrec_impl_guide.md`` Section 3.1 for the strategic
defaults and the rationale behind each weight.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Frozen configuration for ``OrdinalRewardCompute``.

    Attributes:
        w_info: Shaping weight on ``phi(theta_t) - phi(theta_prev)``.
            Ng-Harada-Russell (1999) potential-based shaping term.
        w_cost: Per-administered-item ask cost. Discourages probing
            with very long batches when shorter ones suffice.
        w_expo: Sympson-Hetter (1985) exposure penalty weight.
        w_voi: Value-of-information weight on the terminal NLL anchor
            evaluated against the held-out probe ``H_probe``.
        probe_M: Number of items in the per-batch entropy probe
            ``C``.
        probe_H: Number of items in the held-out probe ``H_probe``
            used by the terminal NLL anchor.
        r_max: Sympson-Hetter hinge threshold. Items exceeding this
            fleet exposure rate are penalised linearly above it.
        c_expo: Scalar coefficient applied to the hinge. ``w_expo``
            and ``c_expo`` are kept separate so that the penalty can
            be scaled jointly (``w_expo``) or per-hinge-slope
            (``c_expo``) for ablations.
        expo_ema_decay: EMA decay used by the fleet-wide exposure
            rate buffer. ``0.99`` gives an effective horizon of
            roughly 100 episodes.
        K_B: Batch size of items administered between reward
            evaluations. Plan Section 5.
        T: Episode horizon in administered items. ``T / K_B`` is the
            number of reward boundaries per episode.
        n_difficulty_strata: Number of equal-count quantiles used by
            the probe sampler. Five gives a balanced stratification
            across difficulty.
        eps_prob: Lower clamp for probabilities when computing the
            entropy product ``p * log p``.
        prior_precision_jitter: Diagonal jitter added to any Laplace
            precision before inversion.
        running_norm_freeze_after: Number of running-mean-std
            updates after which the normaliser freezes its buffers.
        sigma_floor: Posterior-precision floor for the Laplace
            calibration step. Kept here for future critic warmstart;
            unused by the per-batch reward composer.
    """

    # Per-batch shaping weights
    w_info: float = 1.0
    w_cost: float = 0.05
    w_expo: float = 0.10
    w_voi: float = 5.0

    # Probe sizes
    probe_M: int = 32
    probe_H: int = 20

    # Sympson-Hetter exposure control
    r_max: float = 0.20
    c_expo: float = 1.0
    expo_ema_decay: float = 0.99

    # Episode geometry
    K_B: int = 5
    T: int = 10

    # Stratification
    n_difficulty_strata: int = 5

    # Numerical stability
    eps_prob: float = 1e-12
    prior_precision_jitter: float = 1e-6
    running_norm_freeze_after: int = 1000

    # Laplace floor (future use)
    sigma_floor: float = 0.15


__all__ = ["RewardConfig"]
