"""Tests for `kt_mirt.growth.run`: the A4 campaign harness (stage 5/5,
`_planning/design/a4_design.md` v1.1, section 8's run order).

Scale note: every test uses tiny, hand-picked sizes (a handful of KCs and
learners, a handful of epochs/replicates/reshuffles) so the whole file
runs in well under a minute on CPU -- these tests exercise the harness's
WIRING (unit-of-work idempotency, deterministic seeding, cell-to-
certification-to-report assembly, CPU-only enforcement), not the
pre-registered full-scale replicate counts (a later, compute-heavy
campaign this build stage does not execute; see `run.py`'s own module
docstring).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kt_mirt.growth import run


def _tiny_cfg(tmp_path, **overrides):
    profile = run.make_profile("pytest_tiny", n_kcs=2, n_learners=12, kcs_per_learner=2.0)
    defaults = dict(
        output_dir=tmp_path,
        profile=profile,
        twins=["syn_ng", "syn_kg"],
        generator_seeds=[0],
        model_seeds=[0],
        n_perm_bed=4,
        n_perm_kc=3,
        bank_epochs=4,
        tracker_epochs=3,
        act_epochs=3,
        n_reshuffles=1,
        drill_repeats=5,
        run_act_p1=False,
    )
    defaults.update(overrides)
    return run.RunConfig(**defaults)


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------


def test_derive_seed_is_deterministic():
    a = run.derive_seed("twin", "syn_ng", 0)
    b = run.derive_seed("twin", "syn_ng", 0)
    assert a == b


def test_derive_seed_is_sensitive_to_every_part():
    base = run.derive_seed("a", "b", 0)
    assert run.derive_seed("a", "b", 1) != base
    assert run.derive_seed("a", "c", 0) != base
    assert run.derive_seed("x", "b", 0) != base


def test_derive_seed_returns_bounded_nonnegative_int():
    for parts in [("x",), ("a", "b", 3), (1, 2, 3, 4)]:
        s = run.derive_seed(*parts)
        assert isinstance(s, int)
        assert 0 <= s < 2**31


# ---------------------------------------------------------------------------
# Density profile helper
# ---------------------------------------------------------------------------


def test_make_profile_overrides_scale_keeps_shape():
    p = run.make_profile("tiny_test", n_kcs=3, n_learners=30, kcs_per_learner=2.0)
    assert p.n_kcs == 3
    assert p.n_learners == 30
    assert p.kcs_per_learner == 2.0
    # Shape (density/rate quantiles, item arity) inherited from the KDD-matched base.
    assert p.opp_median == run.synth_mod.KDD_MATCHED.opp_median
    assert p.rate_median == run.synth_mod.KDD_MATCHED.rate_median
    assert p.item_arity_mean == run.synth_mod.KDD_MATCHED.item_arity_mean


def test_tiny_profile_is_small():
    assert run.TINY_PROFILE.n_kcs == 8
    assert run.TINY_PROFILE.n_learners == 200


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------


def test_jsonable_converts_numpy_and_nan_and_roundtrips():
    payload = {
        "a": np.array([1.0, np.nan, np.inf]),
        "b": np.int64(3),
        "c": np.bool_(True),
        "d": (1, 2),
    }
    cleaned = run._jsonable(payload)
    text = json.dumps(cleaned)  # must not raise on strict json.dumps
    back = json.loads(text)
    assert back["a"] == [1.0, None, None]
    assert back["b"] == 3
    assert back["c"] is True
    assert back["d"] == [1, 2]


# ---------------------------------------------------------------------------
# Device guard: config/CLI choice, defaulting to CPU, validated against
# torch.cuda.is_available() (the render-ban guard's replacement -- LEDGER
# 2026-07-18 "P3 build EXECUTED" smell (5): "RunConfig still hard-pins CPU
# from the render ban -- to be lifted deliberately with a test").
# ---------------------------------------------------------------------------


def test_run_config_defaults_to_cpu(tmp_path):
    cfg = run.RunConfig(output_dir=tmp_path)
    assert cfg.device == "cpu"


def test_run_config_rejects_unknown_device_string(tmp_path):
    with pytest.raises(ValueError):
        run.RunConfig(output_dir=tmp_path, device="tpu")


def test_run_config_rejects_cuda_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        run.RunConfig(output_dir=tmp_path, device="cuda")


def test_run_config_accepts_cuda_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    cfg = run.RunConfig(output_dir=tmp_path, device="cuda")
    assert cfg.device == "cuda"


def test_module_does_not_set_cuda_visible_devices():
    # This module used to hard-pin CUDA_VISIBLE_DEVICES="" at import time
    # (a stale program-wide render-ban guard); it must no longer touch the
    # variable at all (via os.environ or otherwise), so a caller's own
    # environment (or lack thereof) is respected. The module docstring may
    # still mention the variable name in prose describing this history, so
    # this checks for the assignment mechanism, not the name.
    import inspect

    source = inspect.getsource(run)
    assert "os.environ" not in source


# ---------------------------------------------------------------------------
# Slice cell: idempotency + determinism
# ---------------------------------------------------------------------------


def test_run_slice_cell_writes_file_and_returns_raw_nulls_when_fresh(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    result, raw = run.run_slice_cell(cfg, "syn_ng", 0)
    path = run._slice_cell_path(cfg, "syn_ng", 0)
    assert path.exists()
    assert raw is not None
    assert "bed_null" in raw
    assert result["twin"] == "syn_ng"
    assert result["n_kcs"] == 2
    assert 0.0 <= result["gate"]["bed_pvalue"] <= 1.0


def test_run_slice_cell_second_call_is_idempotent_skip(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    result1, raw1 = run.run_slice_cell(cfg, "syn_ng", 0)
    result2, raw2 = run.run_slice_cell(cfg, "syn_ng", 0)
    assert raw1 is not None
    assert raw2 is None  # loaded from the persisted cell file, not recomputed
    assert result1 == result2


def test_run_slice_cell_force_recomputes(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    run.run_slice_cell(cfg, "syn_ng", 0)
    cfg_force = _tiny_cfg(tmp_path, force=True)
    _, raw = run.run_slice_cell(cfg_force, "syn_ng", 0)
    assert raw is not None  # force bypasses the cached skip


def test_run_slice_cell_is_reproducible_given_same_key(tmp_path):
    cfg_a = _tiny_cfg(tmp_path / "a")
    cfg_b = _tiny_cfg(tmp_path / "b")
    result_a, _ = run.run_slice_cell(cfg_a, "syn_kg", 0)
    result_b, _ = run.run_slice_cell(cfg_b, "syn_kg", 0)
    assert result_a["gate"]["bed_stat"] == pytest.approx(result_b["gate"]["bed_stat"])
    assert result_a["bank_recovery"]["rank_corr"] == pytest.approx(result_b["bank_recovery"]["rank_corr"])


# ---------------------------------------------------------------------------
# Neural cell: idempotency + shape
# ---------------------------------------------------------------------------


def test_run_neural_cell_writes_file_and_has_expected_shape(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    result = run.run_neural_cell(cfg, "syn_kg", 0)
    path = run._neural_cell_path(cfg, "syn_kg", 0)
    assert path.exists()
    assert "cg7" in result["tracker"]
    assert "cg8" in result["tracker"]
    assert "cg9" in result["tracker"]
    assert "act_p0" in result["act"]
    assert "act_p1" not in result["act"]  # run_act_p1=False in _tiny_cfg
    kc_rise = result["act"]["act_p0"]["kc_rise"]
    assert len(kc_rise) == cfg.profile.n_kcs


def test_run_neural_cell_second_call_is_idempotent(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    r1 = run.run_neural_cell(cfg, "syn_ng", 0)
    r2 = run.run_neural_cell(cfg, "syn_ng", 0)
    assert r1 == r2


def test_run_neural_cell_act_p1_adds_lambda(tmp_path):
    cfg = _tiny_cfg(tmp_path, run_act_p1=True)
    result = run.run_neural_cell(cfg, "syn_ng", 0)
    assert "act_p1" in result["act"]
    assert result["act"]["act_p1"]["cg9_act"]["lambda_median_corr"] is not None


# ---------------------------------------------------------------------------
# Twin certification
# ---------------------------------------------------------------------------


def test_certify_twin_syn_ng_has_cg1_and_cg2(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    slice_pairs = [run.run_slice_cell(cfg, "syn_ng", s) for s in cfg.generator_seeds]
    neural_cells = [run.run_neural_cell(cfg, "syn_ng", ms) for ms in cfg.model_seeds]
    cert = run.certify_twin(cfg, "syn_ng", slice_pairs, neural_cells)
    assert cert["twin"] == "syn_ng"
    assert "CG2" in cert["gates"]
    assert "CG1[act_p0]" in cert["gates"]
    assert "CG7" in cert["gates"]
    for g in cert["gates"].values():
        assert isinstance(g.passed, (bool, np.bool_))


def test_certify_twin_syn_kg_uses_ng_reference(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    ng_slices = [run.run_slice_cell(cfg, "syn_ng", s) for s in cfg.generator_seeds]
    ng_neural = [run.run_neural_cell(cfg, "syn_ng", ms) for ms in cfg.model_seeds]
    ng_cert = run.certify_twin(cfg, "syn_ng", ng_slices, ng_neural)

    kg_slices = [run.run_slice_cell(cfg, "syn_kg", s) for s in cfg.generator_seeds]
    kg_neural = [run.run_neural_cell(cfg, "syn_kg", ms) for ms in cfg.model_seeds]
    cert_with_ref = run.certify_twin(cfg, "syn_kg", kg_slices, kg_neural, ng_ref=ng_cert["raw"])
    cert_without_ref = run.certify_twin(cfg, "syn_kg", kg_slices, kg_neural, ng_ref=None)
    assert "CG1a[act_p0]" in cert_with_ref["gates"]
    assert "CG1a[act_p0]" in cert_without_ref["gates"]
    assert "CG3" in cert_with_ref["gates"]


# ---------------------------------------------------------------------------
# Full campaign: cell -> certify -> verdict JSON + markdown
# ---------------------------------------------------------------------------


def test_run_campaign_writes_verdict_and_report(tmp_path):
    cfg = _tiny_cfg(tmp_path, twins=["syn_ng", "syn_kg", "syn_ns", "syn_sat"])
    out = run.run_campaign(cfg)
    assert "verdict" in out
    verdict = out["verdict"]
    assert "gates" in verdict and "kills" in verdict
    assert any(k.startswith("syn_ng.") for k in verdict["gates"])
    assert any(k.startswith("syn_kg.") for k in verdict["gates"])
    assert any(k.startswith("syn_ns.") for k in verdict["gates"])
    assert any(k.startswith("syn_sat.") for k in verdict["gates"])
    assert "K1" in verdict["kills"] and "K5" in verdict["kills"]

    json_path = run.Path(out["json_path"])
    md_path = run.Path(out["md_path"])
    assert json_path.exists()
    assert md_path.exists()
    reloaded = json.loads(json_path.read_text())
    assert reloaded == verdict
    assert "gate report" in md_path.read_text().lower()


def test_run_campaign_is_idempotent_across_cells(tmp_path):
    cfg = _tiny_cfg(tmp_path, twins=["syn_ng", "syn_kg"])
    out1 = run.run_campaign(cfg)
    # A second campaign call over the same output dir must skip every cell
    # (unit-of-work idempotency) and still assemble a verdict.
    out2 = run.run_campaign(cfg)
    assert out1["verdict"]["gates"].keys() == out2["verdict"]["gates"].keys()


def test_render_markdown_has_gate_table_and_kill_section():
    verdict = {
        "gates": {"syn_ng.CG2": {"name": "CG2", "passed": True, "detail": {"bed_pvalue": 0.2}}},
        "kills": {"K1": {"triggered": False}},
        "extra": {"profile": "p", "twins": ["syn_ng"], "generator_seeds": [0], "model_seeds": [0], "wall_clock_s": 1.23},
    }
    md = run.render_markdown(verdict)
    assert "syn_ng.CG2" in md
    assert "PASS" in md
    assert "K1" in md
    assert "gate report" in md.lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_build_arg_parser_defaults():
    p = run.build_arg_parser()
    args = p.parse_args([])
    assert args.profile == "tiny"
    assert args.generator_seeds == [0, 1]
    assert args.model_seeds == [0, 1]
    assert args.device == "cpu"


def test_build_arg_parser_rejects_unknown_device():
    p = run.build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--device", "tpu"])


def test_main_threads_device_into_run_config(tmp_path, monkeypatch):
    # Intercept run_campaign to inspect the RunConfig it was given, rather
    # than actually executing a (possibly cuda) campaign.
    seen = {}

    def _fake_run_campaign(cfg):
        seen["device"] = cfg.device
        out_dir = Path(cfg.output_dir) / cfg.profile.name
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "gate_verdict.json"
        md_path = out_dir / "gate_report.md"
        json_path.write_text("{}")
        md_path.write_text("stub")
        return {"verdict": {}, "json_path": str(json_path), "md_path": str(md_path)}

    monkeypatch.setattr(run, "run_campaign", _fake_run_campaign)
    argv = ["--output-dir", str(tmp_path), "--profile", "tiny", "--device", "cpu"]
    run.main(argv)
    assert seen["device"] == "cpu"


def test_main_runs_end_to_end_tiny(tmp_path, monkeypatch):
    argv = [
        "--output-dir", str(tmp_path),
        "--profile", "tiny",
        "--n-kcs", "2",
        "--n-learners", "12",
        "--twins", "syn_ng", "syn_kg",
        "--generator-seeds", "0",
        "--model-seeds", "0",
        "--n-perm-bed", "4",
        "--n-perm-kc", "3",
        "--bank-epochs", "4",
        "--tracker-epochs", "3",
        "--act-epochs", "3",
        "--n-reshuffles", "1",
        "--drill-repeats", "5",
        "--no-act-p1",
    ]
    result = run.main(argv)
    assert run.Path(result["json_path"]).exists()
    assert run.Path(result["md_path"]).exists()
