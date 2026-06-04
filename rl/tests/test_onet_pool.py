r"""Schema and content tests for ``rl/artifacts/onet_v1.parquet``.

The tests verify the v1 plan, Section 5.2 contract. They run against
the parquet that ``rl/scripts/build_onet_pool.py`` produces and against
a tiny synthetic fixture for the loader, so they pass with or without
the real O\*NET data on disk.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = REPO_ROOT / "rl" / "artifacts" / "onet_v1.parquet"
EXPECTED_COLUMNS = [
    "occupation_code",
    "title",
    "description",
    "tasks_concat",
    "work_activities_summary",
    "riasec_code",
    "work_zone",
    "education_zscore",
]
VALID_WORK_ZONES = {1, 2, 3, 4, 5}
RIASEC_LETTERS = set("RIASEC")


@pytest.fixture(scope="module")
def pool_df() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        pytest.skip(f"{PARQUET_PATH} not built")
    return pd.read_parquet(PARQUET_PATH)


def test_schema_columns_match(pool_df: pd.DataFrame) -> None:
    assert list(pool_df.columns) == EXPECTED_COLUMNS


def test_minimum_row_count(pool_df: pd.DataFrame) -> None:
    # The plan requires at least 800 occupations from the real release.
    assert len(pool_df) >= 800, f"only {len(pool_df)} occupations"


def test_no_null_keys_or_text(pool_df: pd.DataFrame) -> None:
    for col in ("occupation_code", "title", "description", "tasks_concat"):
        assert pool_df[col].notna().all(), col
        assert (pool_df[col].astype(str).str.len() > 0).all(), col


def test_work_zone_in_valid_range(pool_df: pd.DataFrame) -> None:
    assert pool_df["work_zone"].notna().all()
    values = set(pool_df["work_zone"].astype(int).unique().tolist())
    assert values.issubset(VALID_WORK_ZONES), values


def test_occupation_code_unique(pool_df: pd.DataFrame) -> None:
    assert pool_df["occupation_code"].is_unique


def test_riasec_codes_are_well_formed(pool_df: pd.DataFrame) -> None:
    codes = pool_df["riasec_code"].fillna("")
    populated = codes[codes.str.len() > 0]
    # When a RIASEC code is present it must be exactly three RIASEC
    # letters with no duplicates inside the triple.
    for code in populated:
        assert len(code) == 3, code
        assert set(code).issubset(RIASEC_LETTERS), code
        assert len(set(code)) == 3, code


def test_education_zscore_centred(pool_df: pd.DataFrame) -> None:
    z = pool_df["education_zscore"].dropna()
    # Pool-level z-score should sit near zero mean and unit variance.
    assert len(z) > 0
    assert abs(z.mean()) < 0.05
    assert abs(z.std(ddof=0) - 1.0) < 0.05


def test_work_zone_distribution_reported(pool_df: pd.DataFrame) -> None:
    counts = pool_df["work_zone"].value_counts().sort_index().to_dict()
    # The distribution must be non-degenerate (at least three zones
    # represented). The real O\*NET release covers zones 2..5 in v30.2.
    assert len([k for k, v in counts.items() if v > 0]) >= 3, counts


# ---------------------------------------------------------------------
# Loader unit tests on a small synthetic O\*NET-shaped tree.
# ---------------------------------------------------------------------

import importlib.util


def _load_build_module():
    path = REPO_ROOT / "rl" / "scripts" / "build_onet_pool.py"
    spec = importlib.util.spec_from_file_location("build_onet_pool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_onet(tmp_path: Path) -> Path:
    occ = pd.DataFrame(
        {
            "O*NET-SOC Code": ["00-0000.00", "00-0001.00"],
            "Title": ["Test One", "Test Two"],
            "Description": ["Desc one.", "Desc two."],
        }
    )
    occ.to_csv(tmp_path / "Occupation Data.txt", sep="\t", index=False)

    zones = pd.DataFrame(
        {
            "O*NET-SOC Code": ["00-0000.00", "00-0001.00"],
            "Job Zone": [3, 5],
            "Date": ["02/2026", "02/2026"],
            "Domain Source": ["Synthetic", "Synthetic"],
        }
    )
    zones.to_csv(tmp_path / "Job Zones.txt", sep="\t", index=False)

    tasks = pd.DataFrame(
        {
            "O*NET-SOC Code": ["00-0000.00", "00-0000.00", "00-0001.00"],
            "Task ID": ["1", "2", "3"],
            "Task": ["Do A.", "Do B.", "Do C."],
            "Task Type": ["Core", "Supplemental", "Core"],
            "Incumbents Responding": ["10", "10", "10"],
            "Date": ["02/2026"] * 3,
            "Domain Source": ["Synthetic"] * 3,
        }
    )
    tasks.to_csv(tmp_path / "Task Statements.txt", sep="\t", index=False)

    activities_rows = []
    activity_names = ["Act1", "Act2", "Act3", "Act4", "Act5", "Act6"]
    importances = [4.5, 4.2, 4.0, 3.8, 3.5, 3.0]
    for code in ["00-0000.00", "00-0001.00"]:
        for name, value in zip(activity_names, importances):
            activities_rows.append(
                {
                    "O*NET-SOC Code": code,
                    "Element ID": f"id-{name}",
                    "Element Name": name,
                    "Scale ID": "IM",
                    "Data Value": value,
                    "N": "10",
                    "Standard Error": "0.1",
                    "Lower CI Bound": "0",
                    "Upper CI Bound": "5",
                    "Recommend Suppress": "N",
                    "Not Relevant": "n/a",
                    "Date": "02/2026",
                    "Domain Source": "Synthetic",
                }
            )
    pd.DataFrame(activities_rows).to_csv(
        tmp_path / "Work Activities.txt", sep="\t", index=False
    )

    # Interests with explicit high points: occupation 1 = ECA, occupation 2 = RIA.
    interests_rows = []
    for code, hp in [("00-0000.00", [5, 6, 3]), ("00-0001.00", [1, 2, 3])]:
        for slot, value in zip(
            ["First Interest High-Point", "Second Interest High-Point", "Third Interest High-Point"],
            hp,
        ):
            interests_rows.append(
                {
                    "O*NET-SOC Code": code,
                    "Element ID": "1.B.1",
                    "Element Name": slot,
                    "Scale ID": "IH",
                    "Data Value": value,
                    "Date": "02/2026",
                    "Domain Source": "Synthetic",
                }
            )
    pd.DataFrame(interests_rows).to_csv(
        tmp_path / "Interests.txt", sep="\t", index=False
    )

    # Required Level of Education rows for 2.D.1 / RL.
    education_rows = []
    # Occupation 1 sits at category 6, occupation 2 at category 10.
    for code, peak in [("00-0000.00", 6), ("00-0001.00", 10)]:
        for cat in range(1, 13):
            value = 90.0 if cat == peak else 1.0
            education_rows.append(
                {
                    "O*NET-SOC Code": code,
                    "Element ID": "2.D.1",
                    "Element Name": "Required Level of Education",
                    "Scale ID": "RL",
                    "Category": cat,
                    "Data Value": value,
                    "N": "10",
                    "Standard Error": "0.1",
                    "Lower CI Bound": "0",
                    "Upper CI Bound": "100",
                    "Recommend Suppress": "N",
                    "Date": "02/2026",
                    "Domain Source": "Synthetic",
                }
            )
    pd.DataFrame(education_rows).to_csv(
        tmp_path / "Education, Training, and Experience.txt",
        sep="\t",
        index=False,
    )
    return tmp_path


def test_build_pool_on_synthetic_tree(tiny_onet: Path) -> None:
    module = _load_build_module()
    df = module.build_pool(tiny_onet)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 2
    assert df["work_zone"].tolist() == [3, 5]
    # Task ordering must put Core ahead of Supplemental.
    row0 = df.iloc[0]
    assert row0["tasks_concat"] == "Do A. | Do B."
    # The synthetic high-point letters resolve to ECA and RIA.
    codes = set(df["riasec_code"].tolist())
    assert codes == {"ECA", "RIA"}
    # Education z-score should be symmetric around zero with two rows.
    z = df["education_zscore"].to_numpy()
    assert math.isclose(z.sum(), 0.0, abs_tol=1e-6)
    assert math.isclose(abs(z[0]), 1.0, abs_tol=1e-6)
