"""Three-source action mask composition.

The composite mask is the boolean AND of the admin mask, the probe
mask and the no-repeat mask. Each source is responsible for one
class of constraint and the composer must respect all three
simultaneously.
"""

from __future__ import annotations

import torch

from ordrec.envs.action_mask import (
    build_admin_mask,
    build_probe_mask,
    compose_action_mask,
    update_no_repeat_mask,
)


def test_three_source_and() -> None:
    """Each source can independently forbid an item."""
    Q, B = 12, 2
    # Admin forbids pad and item 1.
    admin = build_admin_mask(Q, forbidden_ids=torch.tensor([1]))
    # Probe forbids ids 5, 6.
    probe_C = torch.tensor([[5], [5]])
    probe_H = torch.tensor([[6], [6]])
    probe_mask = build_probe_mask(probe_C, probe_H, Q)
    # No-repeat: item 8 is already administered in batch row 1.
    no_repeat = torch.ones(B, Q + 1, dtype=torch.bool)
    no_repeat[1, 8] = False

    mask = compose_action_mask(admin, probe_mask, no_repeat)
    assert mask.shape == (B, Q + 1)
    # Pad is forbidden everywhere.
    assert not mask[:, 0].any()
    # Item 1 forbidden everywhere via admin.
    assert not mask[:, 1].any()
    # Items 5, 6 forbidden everywhere via probe.
    assert not mask[:, 5].any()
    assert not mask[:, 6].any()
    # Item 8 forbidden only in batch row 1.
    assert mask[0, 8].item() is True
    assert mask[1, 8].item() is False
    # Other items are allowed in both rows.
    for q in [2, 3, 4, 7, 9, 10, 11, 12]:
        assert mask[0, q].item() is True
        assert mask[1, q].item() is True


def test_probe_mask_is_disjoint_with_admin() -> None:
    """The composition order is commutative for set membership."""
    Q, B = 8, 1
    admin = build_admin_mask(Q)
    probe_C = torch.tensor([[3, 5]])
    probe_H = torch.tensor([[7]])
    probe_mask = build_probe_mask(probe_C, probe_H, Q)
    no_repeat = torch.ones(B, Q + 1, dtype=torch.bool)
    m_abc = compose_action_mask(admin, probe_mask, no_repeat)
    m_bac = compose_action_mask(admin, probe_mask, no_repeat)
    assert torch.equal(m_abc, m_bac)


def test_no_repeat_mask_update_idempotent() -> None:
    """Applying ``update_no_repeat_mask`` with the same action twice
    leaves the buffer unchanged after the first application."""
    B, Q = 2, 8
    mask = torch.ones(B, Q + 1, dtype=torch.bool)
    action = torch.tensor([[3, 4], [1, 5]])
    update_no_repeat_mask(mask, action)
    snapshot = mask.clone()
    update_no_repeat_mask(mask, action)
    assert torch.equal(mask, snapshot)


def test_admin_mask_accepts_extra_forbidden_ids() -> None:
    mask = build_admin_mask(10, forbidden_ids=torch.tensor([2, 5, 9]))
    assert not mask[0].item()
    for q in [2, 5, 9]:
        assert not mask[q].item()
    for q in [1, 3, 4, 6, 7, 8, 10]:
        assert mask[q].item()


def test_compose_rejects_shape_mismatch() -> None:
    """Probe and no-repeat must share shape."""
    B, Q = 2, 6
    admin = build_admin_mask(Q)
    probe = torch.ones(B, Q + 1, dtype=torch.bool)
    no_rep = torch.ones(B + 1, Q + 1, dtype=torch.bool)
    try:
        compose_action_mask(admin, probe, no_rep)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for shape mismatch")
