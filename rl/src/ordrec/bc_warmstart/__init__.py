"""Behaviour-cloning and static-MVE warm-start for the PPO actor and critic.

E4 ships the v1 surface: a max-Fisher teacher BC loss for the actor
and the exact ``K ** K_B`` MVE warm-start for the critic. The
ReflectionLayer-greedy and Thompson teachers are documented in the
impl guide but live behind infrastructure that has not been built;
their addition is a v2 enhancement.

See ``docs/ordrec_impl_guide.md`` Section 4.5.
"""

from __future__ import annotations

from .bc import (
    BCStats,
    bc_loss_step,
    bc_warmstart,
    gpcm_item_information,
    max_fisher_actions,
)
from .static_mve import (
    MVEStats,
    mve_warmstart_critic,
    static_mve_critic_step,
    static_mve_target,
)


__all__ = [
    "BCStats",
    "MVEStats",
    "bc_loss_step",
    "bc_warmstart",
    "gpcm_item_information",
    "max_fisher_actions",
    "mve_warmstart_critic",
    "static_mve_critic_step",
    "static_mve_target",
]
