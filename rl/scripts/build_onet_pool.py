"""Build the per-occupation feature parquet from the O*NET 2024+ release.

Per the v1 plan, Section 5.2, this script parses the tab-separated text
release of the O*NET database and emits a single parquet at
``rl/artifacts/onet_v1.parquet`` with the columns required by the
DRL-MAIRT pipeline.

Output schema
-------------
occupation_code : str        join key (O*NET-SOC Code)
title : str                  occupation title
description : str            occupation description (long form)
tasks_concat : str           all task statements joined by " | "
work_activities_summary : str  top work activities by importance (text)
riasec_code : str            3-letter Holland code (e.g. "ESC"). Empty
                             string when no interest data is present.
work_zone : int              1..5 (delta_j source in v1)
education_zscore : float     z-scored expected required education level
                             across the pool, NaN when missing

Usage
-----
The script accepts an already-extracted O*NET text directory via
``--onet-dir``.  If ``--download`` is passed and ``--onet-dir`` is
absent, it pulls the latest known release directly from the O*NET
public CDN. The default output path is
``rl/artifacts/onet_v1.parquet`` relative to the deep-mirt repo root.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ONET_VERSION_DEFAULT = "db_30_2_text"
ONET_URL_TEMPLATE = "https://www.onetcenter.org/dl_files/database/{version}.zip"

# Standard RIASEC letter ordering used by O*NET high-point indices.
# Index 1 corresponds to Realistic, 6 to Conventional. Index 0 means
# the slot is unused.
RIASEC_LETTERS = {
    1: "R",
    2: "I",
    3: "A",
    4: "S",
    5: "E",
    6: "C",
}

RIASEC_NAME_TO_LETTER = {
    "Realistic": "R",
    "Investigative": "I",
    "Artistic": "A",
    "Social": "S",
    "Enterprising": "E",
    "Conventional": "C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onet-dir",
        type=Path,
        default=None,
        help="Path to an already-extracted O*NET text directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="If set and --onet-dir is missing, download the release.",
    )
    parser.add_argument(
        "--onet-version",
        type=str,
        default=ONET_VERSION_DEFAULT,
        help="O*NET release version slug, e.g. db_30_2_text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/artifacts/onet_v1.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=800,
        help="Hard lower bound on the number of occupations kept.",
    )
    return parser.parse_args()


def _download_release(version: str, target_dir: Path) -> Path:
    url = ONET_URL_TEMPLATE.format(version=version)
    print(f"[onet] downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(target_dir)
    extracted = target_dir / version
    if not extracted.exists():
        candidates = [p for p in target_dir.iterdir() if p.is_dir()]
        if not candidates:
            raise RuntimeError("unzip produced no directories")
        extracted = candidates[0]
    return extracted


def _resolve_onet_dir(args: argparse.Namespace) -> Path:
    if args.onet_dir is not None:
        if not args.onet_dir.exists():
            raise FileNotFoundError(args.onet_dir)
        return args.onet_dir
    if not args.download:
        raise SystemExit(
            "either --onet-dir or --download must be provided"
        )
    tmpdir = Path(tempfile.mkdtemp(prefix="onet_"))
    return _download_release(args.onet_version, tmpdir)


def _load_occupations(onet_dir: Path) -> pd.DataFrame:
    path = onet_dir / "Occupation Data.txt"
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    df = df.rename(
        columns={
            "O*NET-SOC Code": "occupation_code",
            "Title": "title",
            "Description": "description",
        }
    )
    return df[["occupation_code", "title", "description"]].copy()


def _load_job_zones(onet_dir: Path) -> pd.DataFrame:
    path = onet_dir / "Job Zones.txt"
    df = pd.read_csv(path, sep="\t", dtype={"Job Zone": "Int64"}, na_filter=True)
    df = df.rename(columns={"O*NET-SOC Code": "occupation_code", "Job Zone": "work_zone"})
    return df[["occupation_code", "work_zone"]].copy()


def _load_tasks(onet_dir: Path) -> pd.DataFrame:
    path = onet_dir / "Task Statements.txt"
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    df = df.rename(columns={"O*NET-SOC Code": "occupation_code", "Task": "task"})
    # Prefer Core tasks first, then Supplemental, then anything else.
    task_type_order = {"Core": 0, "Supplemental": 1}
    df["__order"] = df["Task Type"].map(task_type_order).fillna(2).astype(int)
    df = df.sort_values(["occupation_code", "__order"]).reset_index(drop=True)
    grouped = (
        df.groupby("occupation_code")["task"]
        .apply(lambda s: " | ".join(s.tolist()))
        .reset_index()
        .rename(columns={"task": "tasks_concat"})
    )
    return grouped


def _load_work_activities(onet_dir: Path) -> pd.DataFrame:
    """Top-importance work activities, summarised as a short string.

    Only IM (Importance) rows are kept; the top 5 activities per
    occupation by Data Value are concatenated.
    """

    path = onet_dir / "Work Activities.txt"
    cols = [
        "O*NET-SOC Code",
        "Element Name",
        "Scale ID",
        "Data Value",
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str, na_filter=False)
    df = df[df["Scale ID"] == "IM"].copy()
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")
    df = df.dropna(subset=["Data Value"])
    df = df.sort_values(["O*NET-SOC Code", "Data Value"], ascending=[True, False])
    df = df.groupby("O*NET-SOC Code").head(5)
    grouped = (
        df.groupby("O*NET-SOC Code")["Element Name"]
        .apply(lambda s: "; ".join(s.tolist()))
        .reset_index()
        .rename(
            columns={
                "O*NET-SOC Code": "occupation_code",
                "Element Name": "work_activities_summary",
            }
        )
    )
    return grouped


def _load_riasec(onet_dir: Path) -> pd.DataFrame:
    """RIASEC three-letter code per occupation.

    Prefer the explicit High-Point fields (Scale ID == "IH"). If those
    are missing or zero we fall back to ranking the six OI scores.
    """

    path = onet_dir / "Interests.txt"
    cols = [
        "O*NET-SOC Code",
        "Element Name",
        "Scale ID",
        "Data Value",
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str, na_filter=False)
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")

    # High-point branch.
    hp = df[df["Scale ID"] == "IH"].copy()
    hp_order = {
        "First Interest High-Point": 0,
        "Second Interest High-Point": 1,
        "Third Interest High-Point": 2,
    }
    hp["__order"] = hp["Element Name"].map(hp_order)
    hp = hp.dropna(subset=["__order"])
    hp = hp.sort_values(["O*NET-SOC Code", "__order"])
    hp["__letter"] = (
        hp["Data Value"].fillna(0).round().astype(int).map(RIASEC_LETTERS).fillna("")
    )
    hp_codes = (
        hp.groupby("O*NET-SOC Code")["__letter"]
        .apply(lambda s: "".join(letter for letter in s if letter))
        .reset_index()
        .rename(
            columns={"O*NET-SOC Code": "occupation_code", "__letter": "riasec_code"}
        )
    )

    # Fallback branch from OI scores.
    oi = df[(df["Scale ID"] == "OI") & df["Element Name"].isin(RIASEC_NAME_TO_LETTER)].copy()
    oi["__letter"] = oi["Element Name"].map(RIASEC_NAME_TO_LETTER)
    oi = oi.dropna(subset=["Data Value"])
    oi = oi.sort_values(["O*NET-SOC Code", "Data Value"], ascending=[True, False])
    oi_codes = (
        oi.groupby("O*NET-SOC Code")["__letter"]
        .apply(lambda s: "".join(s.tolist()[:3]))
        .reset_index()
        .rename(
            columns={"O*NET-SOC Code": "occupation_code", "__letter": "riasec_code"}
        )
    )

    merged = hp_codes.merge(
        oi_codes,
        on="occupation_code",
        how="outer",
        suffixes=("_hp", "_oi"),
    )

    def _pick(row: pd.Series) -> str:
        hp_raw = row.get("riasec_code_hp")
        oi_raw = row.get("riasec_code_oi")
        hp_val = "" if not isinstance(hp_raw, str) else hp_raw.strip()
        oi_val = "" if not isinstance(oi_raw, str) else oi_raw.strip()
        if len(hp_val) == 3:
            return hp_val
        return oi_val[:3]

    merged["riasec_code"] = merged.apply(_pick, axis=1)
    return merged[["occupation_code", "riasec_code"]].copy()


def _load_education(onet_dir: Path) -> pd.DataFrame:
    """Expected required education level on the 1..12 ordinal scale.

    Source rows live in ``Education, Training, and Experience.txt`` under
    Element ID ``2.D.1`` (Required Level of Education) with Scale ID
    ``RL``. The Category column is the ordinal level and the Data Value
    is the percent of incumbents at that level. We compute the
    importance-weighted expected level.
    """

    path = onet_dir / "Education, Training, and Experience.txt"
    cols = [
        "O*NET-SOC Code",
        "Element ID",
        "Scale ID",
        "Category",
        "Data Value",
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str, na_filter=False)
    df = df[(df["Element ID"] == "2.D.1") & (df["Scale ID"] == "RL")].copy()
    df["Category"] = pd.to_numeric(df["Category"], errors="coerce")
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")
    df = df.dropna(subset=["Category", "Data Value"])

    def _expected(group: pd.DataFrame) -> float:
        total = group["Data Value"].sum()
        if total <= 0:
            return float("nan")
        return float((group["Category"] * group["Data Value"]).sum() / total)

    expected = (
        df.groupby("O*NET-SOC Code")
        .apply(_expected, include_groups=False)
        .reset_index()
        .rename(columns={"O*NET-SOC Code": "occupation_code", 0: "education_level"})
    )

    levels = expected["education_level"].to_numpy(dtype=float)
    mask = ~np.isnan(levels)
    if mask.sum() > 1:
        mean = levels[mask].mean()
        std = levels[mask].std(ddof=0)
        if std > 0:
            z = (levels - mean) / std
        else:
            z = np.zeros_like(levels)
    else:
        z = np.full_like(levels, np.nan)
    expected["education_zscore"] = z
    return expected[["occupation_code", "education_zscore"]].copy()


def build_pool(onet_dir: Path) -> pd.DataFrame:
    occ = _load_occupations(onet_dir)
    zones = _load_job_zones(onet_dir)
    tasks = _load_tasks(onet_dir)
    activities = _load_work_activities(onet_dir)
    riasec = _load_riasec(onet_dir)
    education = _load_education(onet_dir)

    df = occ.merge(zones, on="occupation_code", how="left")
    df = df.merge(tasks, on="occupation_code", how="left")
    df = df.merge(activities, on="occupation_code", how="left")
    df = df.merge(riasec, on="occupation_code", how="left")
    df = df.merge(education, on="occupation_code", how="left")

    df["tasks_concat"] = df["tasks_concat"].fillna("")
    df["work_activities_summary"] = df["work_activities_summary"].fillna("")
    df["riasec_code"] = df["riasec_code"].fillna("")
    # Drop rows missing the v1 delta_j source.
    df = df.dropna(subset=["work_zone"]).copy()
    df["work_zone"] = df["work_zone"].astype("int64")
    df = df[(df["work_zone"] >= 1) & (df["work_zone"] <= 5)]
    # Reorder columns to match the documented schema.
    return df[
        [
            "occupation_code",
            "title",
            "description",
            "tasks_concat",
            "work_activities_summary",
            "riasec_code",
            "work_zone",
            "education_zscore",
        ]
    ].reset_index(drop=True)


def main() -> int:
    args = parse_args()
    onet_dir = _resolve_onet_dir(args)
    df = build_pool(onet_dir)
    if len(df) < args.min_rows:
        raise SystemExit(
            f"only {len(df)} rows after filtering, expected >= {args.min_rows}"
        )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    print(f"[onet] wrote {output_path} with {len(df)} rows", flush=True)
    print("[onet] work_zone distribution:")
    counts = df["work_zone"].value_counts().sort_index()
    for level, count in counts.items():
        print(f"  zone {int(level)}: {int(count)}")
    print(
        "[onet] education_zscore non-null:",
        int(df["education_zscore"].notna().sum()),
    )
    print(
        "[onet] riasec_code populated:",
        int((df["riasec_code"].str.len() == 3).sum()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
