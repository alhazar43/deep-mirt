"""Probe-leakage assertion.

The policy must never administer any probe id ``C union H_probe``.
This is enforced by the env-side action mask. The reward function
itself does not police the action set, but it is the consumer most
sensitive to leakage, so we still test that the env layer's mask
excludes every probe id.
"""

from __future__ import annotations

import torch

from ordrec.envs.action_mask import (
    build_admin_mask,
    build_probe_mask,
    compose_action_mask,
    update_no_repeat_mask,
)


def test_uniform_random_policy_never_samples_probe() -> None:
    """Sampling under the composed mask never hits a probe id."""
    torch.manual_seed(0)
    Q, M, H, B = 32, 8, 5, 4
    probe_C = torch.zeros(B, M, dtype=torch.long)
    probe_H = torch.zeros(B, H, dtype=torch.long)
    for b in range(B):
        ids = torch.randperm(Q)[: M + H] + 1  # 1-based
        probe_C[b] = ids[:M]
        probe_H[b] = ids[M:]

    admin = build_admin_mask(Q)
    probe_mask = build_probe_mask(probe_C, probe_H, Q)
    no_repeat = torch.ones(B, Q + 1, dtype=torch.bool)
    mask = compose_action_mask(admin, probe_mask, no_repeat)

    # Sample 1000 uniform actions under the mask; every action must
    # respect the mask (be True at its column).
    for _ in range(1000):
        # Convert bool mask to a probability distribution and sample.
        prob = mask.to(torch.float32)
        prob = prob / prob.sum(dim=-1, keepdim=True)
        a = torch.multinomial(prob, num_samples=1).squeeze(-1)  # (B,)
        for b in range(B):
            picked = int(a[b].item())
            forbidden = set(probe_C[b].tolist()) | set(probe_H[b].tolist())
            assert picked not in forbidden, (
                f"policy sampled probe id {picked} in batch row {b}"
            )


def test_probe_mask_is_false_on_every_probe_id() -> None:
    """Direct structural check."""
    Q, M, H, B = 16, 4, 3, 2
    probe_C = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    probe_H = torch.tensor([[9, 10, 11], [12, 13, 14]])
    mask = build_probe_mask(probe_C, probe_H, Q)
    assert mask.shape == (B, Q + 1)
    for b in range(B):
        for pid in probe_C[b].tolist():
            assert mask[b, pid].item() is False
        for pid in probe_H[b].tolist():
            assert mask[b, pid].item() is False


def test_admin_mask_forbids_pad_slot() -> None:
    mask = build_admin_mask(10)
    assert mask.shape == (11,)
    assert mask[0].item() is False
    for q in range(1, 11):
        assert mask[q].item() is True


def test_update_no_repeat_removes_previously_administered() -> None:
    B, Q = 3, 8
    mask = torch.ones(B, Q + 1, dtype=torch.bool)
    action = torch.tensor([
        [3, 4],
        [1, 6],
        [2, 7],
    ])
    update_no_repeat_mask(mask, action)
    for b in range(B):
        for q in action[b].tolist():
            assert mask[b, q].item() is False
