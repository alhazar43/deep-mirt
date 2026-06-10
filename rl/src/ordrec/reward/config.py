"""``RewardConfig`` dataclass for the OrdRec reward composer.

Centralises every numeric knob the four-component reward uses so the
ordinal reward callable can be constructed from a single config object
and serialised to YAML without scattering magic numbers across the
codebase.

See ``docs/ordrec_impl_guide.md`` Section 3.1 for the strategic
defaults and the rationale behind each weight.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RewardConfig:
    """Frozen configuration for ``OrdinalRewardCompute``.

    Attributes:
        w_info: Shaping weight on ``phi(theta_t) - phi(theta_prev)``.
            Ng-Harada-Russell (1999) potential-based shaping term.
        w_cost: Per-administered-item ask cost. Discourages probing
            with very long batches when shorter ones suffice.
        w_expo: Sympson-Hetter (1985) exposure penalty weight.
            Recalibrated in E4.6b R1 from 0.10 to 0.02 after the
            ablation showed the original weight made random selection
            optimal by construction (RC2). At 0.02 the penalty is
            small enough not to dominate r_info while still providing
            a soft exposure bound.
        w_voi: Value-of-information weight on the terminal NLL anchor
            evaluated against the held-out probe ``H_probe``.
        probe_M: Number of items in the per-batch entropy probe
            ``C``.
        probe_H: Number of items in the held-out probe ``H_probe``
            used by the terminal NLL anchor.
        r_max: Sympson-Hetter hinge threshold. Items exceeding this
            fleet exposure rate are penalised linearly above it.
            Recalibrated in E4.6b R1 from 0.20 to 0.40. With Q=200
            and T=10 items per episode, max-Fisher concentrates on a
            small fraction of items whose fleet EMA settles near 0.65
            under long training; r_max=0.40 allows up to 40% exposure
            before the hinge fires, giving information-seeking policies
            room to operate.
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
        probe_sampler: Probe-set sampling strategy. One of
            ``"stratified"`` (default) or ``"uniform"``.
            ``"stratified"`` distributes items across difficulty
            quantile bins; ``"uniform"`` draws uniformly at random.
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
    w_expo: float = 0.02
    w_voi: float = 5.0

    # Probe sizes
    probe_M: int = 32
    probe_H: int = 20

    # Sympson-Hetter exposure control
    r_max: float = 0.40
    c_expo: float = 1.0
    expo_ema_decay: float = 0.99

    # Episode geometry
    K_B: int = 5
    T: int = 10

    # Stratification
    n_difficulty_strata: int = 5
    probe_sampler: str = "stratified"

    # Numerical stability
    eps_prob: float = 1e-12
    prior_precision_jitter: float = 1e-6
    running_norm_freeze_after: int = 1000

    # Laplace floor (future use)
    sigma_floor: float = 0.15

    # Dynamic-world probe resampling (E4.7).
    # When True the terminal NLL anchor re-samples probe_H responses
    # from the frozen world model conditioned on the full simulated
    # history at the episode horizon, instead of using the real
    # historical-tail responses captured at reset time. Set to True
    # when training on staircase or random-walk cohorts so the anchor
    # measures whether the policy helped the model track within-session
    # drift, not whether it predicted stale past responses.
    resample_probe_at_terminal: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        """Construct from a plain dict, ignoring unknown keys.

        Only the keys that match :class:`RewardConfig` field names are
        forwarded; extra keys (from a broader YAML section) are silently
        dropped. Missing keys fall back to the dataclass defaults.

        Args:
            d: Mapping from field name to value. Typically the ``reward``
               sub-dict loaded from a YAML config file.

        Returns:
            A frozen :class:`RewardConfig` instance.

        Example::

            cfg = RewardConfig.from_dict(yaml_data.get("reward", {}))
        """
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


__all__ = ["RewardConfig"]
