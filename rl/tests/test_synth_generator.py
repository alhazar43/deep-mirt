"""Tests for the M3 synthetic data generator on the dev preset.

These tests cover the contracts in the v1 plan, Section 5.4 and the
sanity checks of Section 5.5 at the dev preset (N=500). The heavier
recovery-preset checks (corr(theta_hat, theta_true) > 0.85 after
ma-irt fit) live downstream as part of the M3 analysis agent and are
not exercised here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "rl" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PARQUET_PATH = REPO_ROOT / "rl" / "artifacts" / "onet_v1.parquet"
CONFIG_PATH = REPO_ROOT / "rl" / "configs" / "sim_v1_dev.yaml"
BUILD_SCRIPT = REPO_ROOT / "rl" / "scripts" / "build_synthetic_dataset.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_synthetic_dataset", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_module():
    if not PARQUET_PATH.exists():
        pytest.skip("O*NET parquet not built; run rl/scripts/build_onet_pool.py first")
    return _load_build_module()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, build_module) -> Path:
    out = tmp_path_factory.mktemp("sim_v1_dev_a")
    return build_module.build_dataset(CONFIG_PATH, output_dir_override=out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------
# Schema contracts.
# ---------------------------------------------------------------------


def test_schema_files_exist(dataset: Path) -> None:
    expected = {
        "sequences.json",
        "jobs.json",
        "likes.json",
        "true_irt_parameters.json",
        "true_preference_parameters.json",
        "metadata.json",
    }
    actual = {p.name for p in dataset.iterdir() if p.is_file()}
    assert expected.issubset(actual), f"missing files: {expected - actual}"


def test_sequences_schema(dataset: Path) -> None:
    sequences = _read_json(dataset / "sequences.json")
    assert isinstance(sequences, list) and len(sequences) == 500
    for rec in sequences:
        assert set(rec.keys()) == {"questions", "responses"}
        assert len(rec["questions"]) == len(rec["responses"])
        # 1-based question IDs.
        assert min(rec["questions"]) >= 1
        # Responses are non-negative ints.
        assert all(isinstance(r, int) and r >= 0 for r in rec["responses"])


def test_jobs_schema(dataset: Path) -> None:
    jobs = _read_json(dataset / "jobs.json")
    assert isinstance(jobs, list) and len(jobs) > 0
    required = {
        "job_id",
        "occupation_code",
        "title",
        "description",
        "tasks_concat",
        "work_zone",
        "delta_j",
    }
    for rec in jobs:
        assert required.issubset(rec.keys()), required - set(rec.keys())
        assert np.isfinite(rec["delta_j"])
        assert 1 <= int(rec["work_zone"]) <= 5


def test_likes_schema(dataset: Path) -> None:
    likes = _read_json(dataset / "likes.json")
    assert isinstance(likes, list) and len(likes) > 0
    for rec in likes:
        assert set(rec.keys()) == {"user_id", "job_id", "IsLiked", "prob"}
        assert rec["IsLiked"] in (0, 1)
        assert 0.0 <= rec["prob"] <= 1.0


def test_true_irt_parameters_schema(dataset: Path) -> None:
    irt = _read_json(dataset / "true_irt_parameters.json")
    required = {"theta", "alpha", "beta", "n_categories", "n_students", "n_questions"}
    assert required.issubset(irt.keys())
    assert len(irt["theta"]) == irt["n_students"] == 500
    assert len(irt["alpha"]) == irt["n_questions"]
    assert len(irt["beta"]) == irt["n_questions"]
    # Beta is variable-K per item.
    for i, b in enumerate(irt["beta"]):
        K_i = int(irt["n_categories"][i])
        assert len(b) == K_i - 1
        # Sorted ascending.
        assert list(b) == sorted(b)


def test_true_preference_parameters_schema(dataset: Path) -> None:
    pref = _read_json(dataset / "true_preference_parameters.json")
    required = {"delta_j", "lambda", "bias", "engagement_class", "delta_source"}
    assert required.issubset(pref.keys())
    assert pref["delta_source"] == "work_zone"
    assert all(np.isfinite(d) for d in pref["delta_j"])
    assert all(z in (0, 1) for z in pref["engagement_class"])


def test_metadata_schema(dataset: Path) -> None:
    meta = _read_json(dataset / "metadata.json")
    assert meta["schema_flag"] == "drlmairt_sim_v1"
    assert meta["n_users"] == 500
    assert "k_distribution" in meta
    assert "engagement" in meta and "preference" in meta


# ---------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------


def test_reproducibility_byte_identical(tmp_path: Path, build_module) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    build_module.build_dataset(CONFIG_PATH, output_dir_override=out_a)
    build_module.build_dataset(CONFIG_PATH, output_dir_override=out_b)
    for name in ("sequences.json", "likes.json", "true_irt_parameters.json"):
        assert _sha256(out_a / name) == _sha256(out_b / name), name


# ---------------------------------------------------------------------
# Statistical sanity checks.
# ---------------------------------------------------------------------


def test_like_rate_within_tolerance(dataset: Path) -> None:
    meta = _read_json(dataset / "metadata.json")
    target = meta["preference"]["target_like_rate"]
    achieved = meta["preference"]["achieved_overall_like_rate"]
    assert abs(achieved - target) < 0.02, (achieved, target)


def test_engagement_shares_within_tolerance(dataset: Path) -> None:
    meta = _read_json(dataset / "metadata.json")
    n = meta["n_users"]
    eng = meta["engagement"]
    rejecter_share = eng["rejecter_count"] / n
    engaged_share = eng["engaged_count"] / n
    assert abs(rejecter_share - eng["rejecter_share_target"]) < 0.03
    assert abs(engaged_share - eng["engaged_share_target"]) < 0.03


def test_k_distribution_matches_config(dataset: Path) -> None:
    meta = _read_json(dataset / "metadata.json")
    # The dev YAML composition is 25/15/8/2 at K=2/3/5/6.
    expected = {2: 25, 3: 15, 5: 8, 6: 2}
    # metadata.json stores K keys as ints when dumped, but JSON
    # converts them to strings on roundtrip.
    actual = {int(k): int(v) for k, v in meta["k_distribution"].items()}
    assert actual == expected, actual


def test_per_user_response_count_at_least_30(dataset: Path) -> None:
    sequences = _read_json(dataset / "sequences.json")
    counts = [len(rec["responses"]) for rec in sequences]
    assert min(counts) >= 30, f"min count {min(counts)}"


def test_delta_j_finite_no_nans(dataset: Path) -> None:
    pref = _read_json(dataset / "true_preference_parameters.json")
    deltas = np.asarray(pref["delta_j"], dtype=np.float64)
    assert np.isfinite(deltas).all()


def test_rejecter_users_emit_only_zero_likes(dataset: Path) -> None:
    pref = _read_json(dataset / "true_preference_parameters.json")
    likes = _read_json(dataset / "likes.json")
    eng = np.asarray(pref["engagement_class"], dtype=np.int64)
    for rec in likes:
        if eng[rec["user_id"]] == 0:
            assert rec["IsLiked"] == 0 and rec["prob"] == 0.0


def test_per_user_candidate_set_sizes_in_clip(dataset: Path) -> None:
    pref = _read_json(dataset / "true_preference_parameters.json")
    sizes = np.asarray(pref["candidate_set_sizes"], dtype=np.int64)
    assert sizes.min() >= 1
    assert sizes.max() <= 200
