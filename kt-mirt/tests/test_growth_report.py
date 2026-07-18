"""Tests for `kt_mirt.growth.report`: the restructured posture-
disagreement matrix, the Tier 1/2/3 claim ladder, the BY-sensitivity
report, and machine-readable verdict JSON assembly
(`_planning/design/a4_design.md` v1.1, section 5.5)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kt_mirt.growth import battery, report


# =============================================================================
# Posture-disagreement matrix
# =============================================================================


def test_act_fires_gate_flat_is_fabrication_suspect():
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=False, tracker_fired=False, act_fires=True)
    )
    assert "act_fires_gate_flat" in v.patterns
    assert "Fabrication suspect" in v.readings[v.patterns.index("act_fires_gate_flat")]


def test_gate_fires_act_flat_is_gain_family_too_rigid():
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=True, tracker_fired=True, act_fires=False)
    )
    assert "gate_fires_act_flat" in v.patterns


def test_gate_fires_tracker_flat_is_tracker_insensitivity():
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=True, tracker_fired=False, act_fires=True)
    )
    assert "gate_fires_tracker_flat" in v.patterns


def test_tracker_fires_gate_flat_is_reconstruction_artifact():
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=False, tracker_fired=True, act_fires=False)
    )
    assert "tracker_fires_gate_flat" in v.patterns


def test_gate_fires_rate_unreliable_is_existence_only():
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=True, tracker_fired=True, act_fires=True, rate_reliable=False)
    )
    assert "gate_fires_rate_unreliable" in v.patterns


def test_gate_and_rate_fire_shape_disagree():
    v = report.classify_posture_disagreement(
        report.PostureInputs(
            gate_fired=True, tracker_fired=True, act_fires=True, rate_reliable=True, misfit_fired=True
        )
    )
    assert "gate_and_rate_fire_shape_disagree" in v.patterns


def test_all_three_agree_is_strongest_claim():
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=True, tracker_fired=True, act_fires=True)
    )
    assert "all_three_agree" in v.patterns


def test_no_signal_when_everything_flat():
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=False, tracker_fired=False, act_fires=False)
    )
    assert v.patterns == ["no_signal"]


def test_multiple_rows_can_match_simultaneously():
    """"gate fires, tracker flat" and "gate fires, rate unreliable" both
    condition on the gate-fired side and can co-occur (module docstring
    note 2)."""
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=True, tracker_fired=False, act_fires=True, rate_reliable=False)
    )
    assert "gate_fires_tracker_flat" in v.patterns
    assert "gate_fires_rate_unreliable" in v.patterns


def test_decline_sign_never_fabricates_an_act_row():
    """Section 0's Tier-3 sign partition: ACT abstains by construction on
    decline slices (act_fires=None); no ACT-conditioned row should ever
    fire there (interpretation note 3)."""
    v = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=True, tracker_fired=True, act_fires=None, sign="decline")
    )
    assert "act_fires_gate_flat" not in v.patterns
    assert "gate_fires_act_flat" not in v.patterns
    assert "all_three_agree" in v.patterns  # PAS/MIX-only agreement still recognized


def test_posture_inputs_has_no_separate_mixed_existence_field():
    """Structural enforcement of the "impossible cell" (interpretation
    note 1): there is no field on `PostureInputs` a caller could even use
    to assert a disagreement between "gate" and "mixed" at existence."""
    fields = {f.name for f in report.PostureInputs.__dataclass_fields__.values()}
    assert "mixed_fired" not in fields
    assert fields == {"gate_fired", "tracker_fired", "act_fires", "rate_reliable", "misfit_fired", "sign"}


# =============================================================================
# BY-sensitivity report
# =============================================================================


def test_by_sensitivity_report_flags_bh_only_discoveries():
    bh = np.array([True, True, False, True])
    by = np.array([True, False, False, True])
    out = report.by_sensitivity_report(bh, by, kc_ids=["a", "b", "c", "d"])
    assert out["n_bh_discoveries"] == 3
    assert out["n_by_discoveries"] == 2
    assert out["n_dependence_sensitive"] == 1
    assert out["dependence_sensitive_kc_ids"] == ["b"]


# =============================================================================
# Tier claim assembly
# =============================================================================


def test_assemble_tier_claim_quarantine_short_circuits():
    assert report.assemble_tier_claim(rb0_passed=False, rb1_passed=True, rb3_passed=True, rb4_passed=True, rb5_passed=True) == "quarantined"
    assert report.assemble_tier_claim(rb0_passed=True, rb1_passed=False, rb3_passed=True) == "quarantined"


def test_assemble_tier_claim_ladder():
    assert report.assemble_tier_claim(True, True, False) == "none"
    assert report.assemble_tier_claim(True, True, True) == "tier1"
    assert report.assemble_tier_claim(True, True, True, rb4_passed=True) == "tier2"
    assert report.assemble_tier_claim(True, True, True, rb4_passed=True, rb5_passed=True) == "tier3"
    # RB4 fails -> capped at tier1 even if (inconsistently) rb5 were True
    assert report.assemble_tier_claim(True, True, True, rb4_passed=False, rb5_passed=True) == "tier1"


# =============================================================================
# Machine-readable verdict JSON
# =============================================================================


def test_clean_converts_numpy_and_nan_to_native():
    payload = {
        "a": np.float64(1.5),
        "b": np.int64(3),
        "c": np.bool_(True),
        "d": np.array([1.0, 2.0]),
        "e": float("nan"),
        "f": float("inf"),
        (1, 2): "tuple_key",
    }
    cleaned = report._clean(payload)
    assert cleaned["a"] == 1.5 and isinstance(cleaned["a"], float)
    assert cleaned["b"] == 3 and isinstance(cleaned["b"], int)
    assert cleaned["c"] is True
    assert cleaned["d"] == [1.0, 2.0]
    assert cleaned["e"] is None
    assert cleaned["f"] is None
    assert cleaned["(1, 2)"] == "tuple_key"


def test_clean_converts_dataclasses_recursively():
    gv = battery.GateVerdict(name="CG1", passed=True, detail={"x": np.float64(0.5)})
    cleaned = report._clean(gv)
    assert cleaned == {"name": "CG1", "passed": True, "detail": {"x": 0.5}}


def test_assemble_verdict_json_and_dumps_round_trip():
    gates = {
        "CG1": battery.cg1_active_silence(0.005, 0.005, "kdd_matched"),
        "CG2": battery.cg2_gate_false_positives(0.5, np.zeros(10, dtype=bool)),
    }
    kills = {"K1": battery.k1_posture_machinery(True, True, True, True, True, revision_budget_exhausted=False)}
    disagreement = report.classify_posture_disagreement(
        report.PostureInputs(gate_fired=True, tracker_fired=True, act_fires=True)
    )
    payload = report.assemble_verdict_json(gates, kills, disagreement=disagreement, tier_claim="tier1")
    assert payload["gates"]["CG1"]["passed"] is True
    assert payload["kills"]["K1"]["triggered"] is False
    assert payload["tier_claim"] == "tier1"

    text = report.dumps_verdict(gates, kills, disagreement=disagreement, tier_claim="tier1")
    round_tripped = json.loads(text)  # must be strict-JSON parseable
    assert round_tripped == payload


def test_dumps_verdict_has_no_nan_literal_in_output():
    gates = {"CG1c": battery.cg1c_active_saturation_refusal(float("nan"), 0.0, 0.0, 0.0, False)}
    text = report.dumps_verdict(gates)
    assert "NaN" not in text
    parsed = json.loads(text)
    assert parsed["gates"]["CG1c"]["detail"]["pop_ok"] is False
