"""Verdict assembly: the restructured posture-disagreement matrix (design
section 5.5, `_planning/design/a4_design.md` v1.1), the Tier 1/2/3 claim
ladder (section 1's G2 ladder gated by RB0/RB1/RB3/RB4/RB5), the
Benjamini-Yekutieli dependence-sensitivity report (section 2.2), and
machine-readable verdict JSON assembly for every gate this package can
produce (`battery.py`'s `GateVerdict`/`KillVerdict`, plus `active.py`'s
`FiringVerdict` for RB-A).

Interpretation notes (recorded per the harness instructions):

1. **The "impossible cell" is a structural, not a runtime, invariant.**
   Section 5.5's preamble states "'Passive fires, mixed flat at
   existence' is impossible by construction" because PAS-G IS MIX-L's
   stage 1 (one shared implementation, one result, `rate.py`'s own
   module docstring). Consequently `PostureInputs` below carries only
   ONE existence-tier gate boolean (``gate_fired``), never a separate
   "mixed fired" field -- there is nothing for a caller to even
   accidentally disagree with, so no runtime assertion is needed to rule
   the impossible cell out; its absence from `PostureInputs`'s own shape
   is the enforcement.
2. **The seven named rows of 5.5's table are matched independently**,
   not as an exhaustive partition of one big switch statement: a given
   (KC, bed) can match more than one row at once (e.g. "gate fires,
   tracker flat" and "gate fires, rate unreliable" are both about the
   gate-fired side and can co-occur), so `classify_posture_disagreement`
   returns a LIST of every matched pattern plus its reading, rather than
   a single label. A "no independent-input evidence at all" cell (every
   input flat) is not named in the design's table; it is reported here
   under a synthetic ``"no_signal"`` label rather than silently dropped.
3. **Tier-3 sign partitioning (section 0, R1-I9)** is exposed as a
   caller-supplied ``sign`` field on `PostureInputs` ("gain" | "decline");
   on "decline" slices `act_fires` is expected to always be ``None``
   (ACT abstains by construction, rho=1 pin) -- `classify_posture_
   disagreement` never fabricates an ACT vote on those slices, it simply
   omits every ACT-conditioned row when ``act_fires is None``, exactly
   matching section 0's "decline diagnostics are two-posture (PAS/MIX)
   reads."
4. **`assemble_tier_claim`'s quarantine check (K6) short-circuits before
   any tier is considered**, mirroring section 5.6: "RB0 or RB1 fails on
   a real bed = every readout on that bed is quarantined until diagnosed;
   no claim of any tier" -- this is checked first and unconditionally,
   before RB3/RB4/RB5 are even read.
5. **JSON assembly converts NaN to ``null`` and all numpy scalar/array
   types to native Python** (`_clean`), since strict JSON (unlike
   Python's permissive `json.dumps` default) has no NaN literal --
   "machine-readable verdict JSON" is read as JSON a non-Python consumer
   can parse, not merely a Python `repr`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from kt_mirt.growth import battery as battery_mod
from kt_mirt.growth.active import FiringVerdict


# =============================================================================
# Posture-disagreement matrix (design 5.5)
# =============================================================================


@dataclass
class PostureInputs:
    """The THREE actually-independent existence-tier inputs (section 5.5's
    restructuring), plus the rate-tier reliability/misfit reads needed for
    the two rate-conditioned rows. ``sign`` partitions Tier-3 reads
    (section 0); on "decline" slices ``act_fires`` must be ``None``
    (interpretation note 3)."""

    gate_fired: Optional[bool]  # PAS-G / MIX-L stage 1, shared, ONE vote
    tracker_fired: Optional[bool]  # PAS-N (E-P3 through the battery), None if quarantined/not run
    act_fires: Optional[bool]  # RB-A, None if not certified / abstaining on a decline slice
    rate_reliable: Optional[bool] = None  # RB4/RB5, meaningful only if gate_fired
    misfit_fired: Optional[bool] = None  # E-M3, meaningful only if gate_fired and rate fit
    sign: str = "gain"  # "gain" | "decline" (section 0)


@dataclass
class DisagreementVerdict:
    patterns: list[str]
    readings: list[str]


_ROW_TEXT = {
    "act_fires_gate_flat": "Fabrication suspect; active quarantined; recheck CG1/CG1b config parity.",
    "gate_fires_act_flat": "Gain family or gating too rigid; consult E-M3 misfit flags.",
    "gate_fires_tracker_flat": "Tracker insensitivity; consult CG7 margin and the contamination probe before blaming the data.",
    "tracker_fires_gate_flat": "Reconstruction/contamination artifact suspect; direction audit and CG8 consulted; tracker read quarantined.",
    "gate_fires_rate_unreliable": "Existence-only claim (Tier 1); the E2c pattern.",
    "gate_and_rate_fire_shape_disagree": "Growth real, family wrong; report the blockwise profile, withhold r.",
    "all_three_agree": "Strongest claim tier available at this density.",
    "no_signal": "No existence signal from any independent input.",
}


def classify_posture_disagreement(inputs: PostureInputs) -> DisagreementVerdict:
    """Section 5.5's seven-row table, restructured over the three
    actually-independent inputs (interpretation notes 1-3): matches every
    applicable row (not mutually exclusive) and returns each row's
    pre-registered reading."""
    gate, tracker, act = inputs.gate_fired, inputs.tracker_fired, inputs.act_fires
    patterns: list[str] = []

    if inputs.sign == "gain" and act is True and gate is False:
        patterns.append("act_fires_gate_flat")
    if inputs.sign == "gain" and gate is True and act is False:
        patterns.append("gate_fires_act_flat")
    if gate is True and tracker is False:
        patterns.append("gate_fires_tracker_flat")
    if tracker is True and gate is False:
        patterns.append("tracker_fires_gate_flat")
    if gate is True and inputs.rate_reliable is False:
        patterns.append("gate_fires_rate_unreliable")
    if gate is True and inputs.rate_reliable is True and inputs.misfit_fired is True:
        patterns.append("gate_and_rate_fire_shape_disagree")
    if gate is True and tracker is True and (act is True or inputs.sign == "decline"):
        patterns.append("all_three_agree")

    if not patterns:
        patterns.append("no_signal")

    return DisagreementVerdict(patterns=patterns, readings=[_ROW_TEXT[p] for p in patterns])


def by_sensitivity_report(bh_reject: np.ndarray, by_reject: np.ndarray, kc_ids: Optional[Sequence] = None) -> dict:
    """Section 2.2's dual discovery-control report: which per-KC BH
    discoveries vanish under the arbitrary-dependence-valid Benjamini-
    Yekutieli correction (report content, not a gate)."""
    bh_reject = np.asarray(bh_reject, dtype=bool)
    by_reject = np.asarray(by_reject, dtype=bool)
    sensitive = bh_reject & (~by_reject)
    ids = list(range(len(bh_reject))) if kc_ids is None else list(kc_ids)
    return {
        "n_bh_discoveries": int(bh_reject.sum()),
        "n_by_discoveries": int(by_reject.sum()),
        "n_dependence_sensitive": int(sensitive.sum()),
        "dependence_sensitive_kc_ids": [ids[i] for i in np.where(sensitive)[0]],
    }


# =============================================================================
# Tier claim assembly (section 1's G2 ladder; section 5.6's K6 quarantine)
# =============================================================================


def assemble_tier_claim(
    rb0_passed: bool,
    rb1_passed: bool,
    rb3_passed: bool,
    rb4_passed: Optional[bool] = None,
    rb5_passed: Optional[bool] = None,
) -> str:
    """"none" | "quarantined" | "tier1" | "tier2" | "tier3" (section 1's
    claim ladder; K6's quarantine short-circuits first, interpretation
    note 4)."""
    if not (rb0_passed and rb1_passed):
        return "quarantined"
    if not rb3_passed:
        return "none"
    tier = "tier1"
    if rb4_passed:
        tier = "tier2"
        if rb5_passed:
            tier = "tier3"
    return tier


# =============================================================================
# Machine-readable verdict JSON
# =============================================================================


def _clean(obj):
    """Recursively converts dataclasses, numpy scalars/arrays, and NaN
    floats into strict-JSON-safe native Python (interpretation note 5)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _clean(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if np.isnan(f) or np.isinf(f) else f
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def assemble_verdict_json(
    gates: dict,
    kills: Optional[dict] = None,
    disagreement: Optional[dict] = None,
    tier_claim: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Assembles one machine-readable verdict record. ``gates`` maps a
    name (e.g. "CG1", "RB3") to a `battery.GateVerdict`, an
    `active.FiringVerdict` (RB-A), or any plain dict/dataclass; ``kills``
    maps a code (e.g. "K1") to a `battery.KillVerdict`; ``disagreement``
    and ``extra`` are arbitrary caller-supplied JSON-safe payloads (e.g. a
    `DisagreementVerdict` per KC, or `by_sensitivity_report`'s output)."""
    out: dict = {
        "gates": {name: _clean(v) for name, v in gates.items()},
        "kills": {code: _clean(v) for code, v in (kills or {}).items()},
    }
    if disagreement is not None:
        out["disagreement"] = _clean(disagreement)
    if tier_claim is not None:
        out["tier_claim"] = tier_claim
    if extra is not None:
        out["extra"] = _clean(extra)
    return out


def dumps_verdict(*args, **kwargs) -> str:
    """`assemble_verdict_json` immediately serialized to a JSON string
    (``json.dumps`` over the already-cleaned dict is always strict-JSON
    valid, since `_clean` has already removed every NaN/inf and numpy
    type)."""
    return json.dumps(assemble_verdict_json(*args, **kwargs), indent=2, sort_keys=True)


__all__ = [
    "PostureInputs",
    "DisagreementVerdict",
    "classify_posture_disagreement",
    "by_sensitivity_report",
    "assemble_tier_claim",
    "assemble_verdict_json",
    "dumps_verdict",
]
