"""Tests for `kt_mirt.transfer.synth`: the D=3 signed-edge generator
(`_planning/design/a1_design.md` v1.1, sections 3.1, 3.2) -- the reference
signed edge set, the known-G round-trip, the matched KG/NG twins, and the
decoupling knob's effect on the per-edge effective sample."""

from __future__ import annotations

import numpy as np
import pytest

from kt_mirt.transfer.synth import (
    default_G_true,
    generate_signed_twin,
    realized_transfer_ok,
)


# ---------------------------------------------------------------------------
# The reference signed edge set
# ---------------------------------------------------------------------------


def test_default_G_true_structure_at_d3():
    """One + edge, one - edge, one zero-row control KC, zero diagonal."""
    G = default_G_true(3, g_pos=0.05, g_neg=0.02)
    assert np.allclose(np.diag(G), 0.0)
    assert G[0, 1] == pytest.approx(0.05)  # KC1 -> KC0 facilitation
    assert G[1, 2] == pytest.approx(-0.02)  # KC2 -> KC1 interference
    assert np.all(G[2, :] == 0.0)  # KC2 a zero-row control (no incoming edge)
    assert int(np.sum(G > 0)) == 1
    assert int(np.sum(G < 0)) == 1


def test_default_G_true_requires_three_kcs():
    with pytest.raises(ValueError):
        default_G_true(2)


# ---------------------------------------------------------------------------
# Known-G round-trip (design section 3.1's two new acceptance checks)
# ---------------------------------------------------------------------------


def test_generator_roundtrips_known_G_signs():
    """On SYN-T-KG the realized target displacement attributable to source
    practice is sign-correct on every true edge and non-trivial."""
    kg = generate_signed_twin("syn_t_kg", "kdd", seed=0, n_learners=300, decoupling=0.9)
    disp = kg.truth.realized_displacement
    G = kg.truth.G_true
    assert np.sign(disp[0, 1]) == np.sign(G[0, 1])  # positive edge -> positive displacement
    assert np.sign(disp[1, 2]) == np.sign(G[1, 2])  # negative edge -> negative displacement
    assert abs(disp[0, 1]) > 0.0 and abs(disp[1, 2]) > 0.0
    assert realized_transfer_ok(kg)["passed"]


def test_generator_ng_twin_has_exactly_zero_transfer():
    """SYN-T-NG: own-growth present, transfer OFF -> zero realized
    displacement to generator precision (the honest null)."""
    ng = generate_signed_twin("syn_t_ng", "kdd", seed=0, n_learners=300, decoupling=0.9)
    assert np.all(ng.truth.realized_displacement == 0.0)
    assert np.all(ng.truth.G_true == 0.0)
    assert realized_transfer_ok(ng)["passed"]


def test_kg_ng_twins_share_schedule_but_differ_in_responses():
    """Matched-twin discipline: identical practice schedules (item id
    sequences per learner), responses differ where transfer moved the
    target's success probability."""
    kg = generate_signed_twin("syn_t_kg", "kdd", seed=1, n_learners=300, decoupling=0.9)
    ng = generate_signed_twin("syn_t_ng", "kdd", seed=1, n_learners=300, decoupling=0.9)
    assert all(np.array_equal(kg.learners[i].item_ids, ng.learners[i].item_ids) for i in range(300))
    diff = sum(int(not np.array_equal(kg.learners[i].responses, ng.learners[i].responses)) for i in range(300))
    assert diff > 0  # transfer changed at least some outcomes


def test_kg_ng_effective_sample_matches():
    """The per-edge effective sample is a SCHEDULE property, so it is
    identical on the matched twins (NG shares KG's schedule)."""
    kg = generate_signed_twin("syn_t_kg", "kdd", seed=0, n_learners=300, decoupling=0.9)
    ng = generate_signed_twin("syn_t_ng", "kdd", seed=0, n_learners=300, decoupling=0.9)
    assert np.allclose(kg.truth.eff_sample, ng.truth.eff_sample)


def test_generator_reproducible_by_seed():
    a = generate_signed_twin("syn_t_kg", "kdd", seed=3, n_learners=200, decoupling=0.8)
    b = generate_signed_twin("syn_t_kg", "kdd", seed=3, n_learners=200, decoupling=0.8)
    assert all(np.array_equal(a.learners[i].responses, b.learners[i].responses) for i in range(200))


# ---------------------------------------------------------------------------
# The decoupling knob -> effective sample (the CT0 power axis)
# ---------------------------------------------------------------------------


def test_decoupling_raises_effective_sample():
    """Higher decoupling -> more decoupled source-before-target
    observations per edge (the CT0 power axis, section 2.1.1). Monotone in
    decoupling at fixed N."""
    eff = {}
    for d in (0.2, 0.5, 0.9):
        kg = generate_signed_twin("syn_t_kg", "kdd", seed=0, n_learners=300, decoupling=d)
        eff[d] = kg.truth.eff_sample[0, 1]  # the positive edge's effective sample
    assert eff[0.2] < eff[0.5] < eff[0.9]


def test_effective_sample_grows_with_n():
    """At fixed decoupling the effective sample grows with N (the other
    CT0 power axis)."""
    e_small = generate_signed_twin("syn_t_kg", "kdd", seed=0, n_learners=150, decoupling=0.8).truth.eff_sample[0, 1]
    e_large = generate_signed_twin("syn_t_kg", "kdd", seed=0, n_learners=600, decoupling=0.8).truth.eff_sample[0, 1]
    assert e_large > e_small


def test_generator_rejects_unknown_twin():
    with pytest.raises(ValueError):
        generate_signed_twin("syn_t_xx", "kdd", seed=0, n_learners=100)
