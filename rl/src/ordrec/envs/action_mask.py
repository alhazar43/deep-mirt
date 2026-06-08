"""Three-source action mask composer.

The OrdRec policy faces a categorical over ``Q + 1`` slots (``0``
reserved as padding). The valid-action mask is the boolean AND of
three sources.

1. Administrative mask. Items the env wants to forbid for system
   reasons. Typically the pad slot and any items the dataset cannot
   simulate (cold items lacking ``(alpha, beta)`` entries). This is
   carried in the env's static state.
2. Probe mask. Items in ``C union H_probe`` are reserved for the
   reward computation and must never be administered. This prevents
   the policy from gaming the entropy reduction by directly observing
   probe items.
3. No-repeat-within-episode mask. Items already administered in the
   current episode cannot be re-administered. Tracked through the
   exposure-counts buffer the env maintains.

See ``docs/ordrec_impl_guide.md`` Section 3.3 ("Mask invariants")
and Section 5.1 for the contract.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def compose_action_mask(
    admin_mask: Tensor,
    probe_mask: Tensor,
    no_repeat_mask: Tensor,
) -> Tensor:
    """AND of the three Boolean mask sources, shape ``(B, Q + 1)``.

    Args:
        admin_mask: ``BoolTensor (Q + 1,)`` or ``(B, Q + 1)``. ``True``
            for items the env allows administratively. The pad slot
            (index ``0``) should be ``False``.
        probe_mask: ``BoolTensor (B, Q + 1)``. ``True`` outside the
            per-episode probe ``C union H_probe``; ``False`` for any
            probe id.
        no_repeat_mask: ``BoolTensor (B, Q + 1)``. ``True`` for any
            item not yet administered in this episode.

    Returns:
        ``BoolTensor (B, Q + 1)`` allowing only items that satisfy all
        three sources simultaneously.
    """
    if probe_mask.ndim != 2 or no_repeat_mask.ndim != 2:
        raise ValueError(
            "probe_mask and no_repeat_mask must be 2D (B, Q + 1)."
        )
    if probe_mask.shape != no_repeat_mask.shape:
        raise ValueError(
            f"probe_mask {tuple(probe_mask.shape)} and no_repeat_mask "
            f"{tuple(no_repeat_mask.shape)} must share shape."
        )

    if admin_mask.ndim == 1:
        admin_b = admin_mask.unsqueeze(0).expand_as(probe_mask)
    else:
        admin_b = admin_mask
        if admin_b.shape != probe_mask.shape:
            raise ValueError(
                f"admin_mask shape {tuple(admin_b.shape)} must broadcast to "
                f"{tuple(probe_mask.shape)}."
            )

    return admin_b & probe_mask & no_repeat_mask


def build_probe_mask(
    probe_C_ids: Tensor,
    probe_H_ids: Tensor,
    n_questions: int,
) -> Tensor:
    """Return ``(B, Q + 1)`` mask blocking probe ids.

    The probe ids are scattered across a Boolean buffer; the pad slot
    is left unmasked here and is forbidden by ``admin_mask`` upstream.

    Args:
        probe_C_ids: ``LongTensor (B, M)``.
        probe_H_ids: ``LongTensor (B, H)``.
        n_questions: ``Q``.

    Returns:
        ``BoolTensor (B, Q + 1)`` with ``True`` outside the probe
        union and ``False`` on every probe id.
    """
    if probe_C_ids.ndim != 2 or probe_H_ids.ndim != 2:
        raise ValueError(
            "probe_C_ids and probe_H_ids must both be 2D (B, *)."
        )
    if probe_C_ids.shape[0] != probe_H_ids.shape[0]:
        raise ValueError(
            f"probe batch dims differ, {probe_C_ids.shape[0]} vs "
            f"{probe_H_ids.shape[0]}."
        )

    B = probe_C_ids.shape[0]
    Qp1 = int(n_questions) + 1
    mask = torch.ones(B, Qp1, dtype=torch.bool, device=probe_C_ids.device)
    # ``scatter_`` writes ``False`` into the probe ids per row. The
    # source tensor must share shape with ``index``.
    false_C = torch.zeros_like(probe_C_ids, dtype=torch.bool)
    false_H = torch.zeros_like(probe_H_ids, dtype=torch.bool)
    mask.scatter_(dim=1, index=probe_C_ids.long(), src=false_C)
    mask.scatter_(dim=1, index=probe_H_ids.long(), src=false_H)
    return mask


def update_no_repeat_mask(
    no_repeat_mask: Tensor,
    action: Tensor,
) -> Tensor:
    """In-place set ``False`` at every (b, action[b, k]) slot.

    Args:
        no_repeat_mask: ``BoolTensor (B, Q + 1)``, updated in-place.
        action: ``LongTensor (B, K_B)``. Pad ids (``0``) are tolerated;
            the resulting ``False`` write at column 0 is harmless
            because the admin mask already disallows the pad slot.

    Returns:
        ``no_repeat_mask`` after the in-place update.
    """
    if no_repeat_mask.ndim != 2 or action.ndim != 2:
        raise ValueError(
            "no_repeat_mask must be 2D (B, Q+1) and action must be 2D (B, K_B)."
        )
    if no_repeat_mask.shape[0] != action.shape[0]:
        raise ValueError(
            f"batch dims differ, no_repeat_mask {no_repeat_mask.shape[0]} "
            f"vs action {action.shape[0]}."
        )
    false_src = torch.zeros_like(action, dtype=torch.bool)
    no_repeat_mask.scatter_(dim=1, index=action.long(), src=false_src)
    return no_repeat_mask


def build_admin_mask(
    n_questions: int,
    *,
    device: Optional[torch.device] = None,
    forbidden_ids: Optional[Tensor] = None,
) -> Tensor:
    """Static admin mask, ``(Q + 1,)`` Boolean, pad slot forbidden.

    Args:
        n_questions: ``Q``.
        device: Output device. Defaults to ``cpu``.
        forbidden_ids: Optional ``LongTensor`` of additional ids to
            forbid (cold items, banned content). Pad ``0`` is always
            forbidden regardless of this argument.

    Returns:
        ``BoolTensor (Q + 1,)``.
    """
    Qp1 = int(n_questions) + 1
    mask = torch.ones(Qp1, dtype=torch.bool, device=device)
    mask[0] = False
    if forbidden_ids is not None and forbidden_ids.numel() > 0:
        mask[forbidden_ids.long().to(device=mask.device)] = False
    return mask


__all__ = [
    "build_admin_mask",
    "build_probe_mask",
    "compose_action_mask",
    "update_no_repeat_mask",
]
