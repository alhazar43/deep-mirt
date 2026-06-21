"""Render the E8 Phase-2 planted-signal JSON as a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"


def _fmt(value: Any, p: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{value:.{p}f}"


def render(data: dict[str, Any]) -> str:
    meta = data["meta"]
    lines = ["# Phase-2 Contextual Alpha Signal\n"]
    lines.append(
        f"K={meta['K']} N={meta['N']} Q={meta['Q']} T={meta['T']} "
        f"epochs={meta['epochs']} seeds={meta['seeds']} "
        f"sigmas={meta['sigmas']} device={meta['device']}\n"
    )
    lines.append("## Summary\n")
    lines.append(
        "| sigma | corr(plant,gamma) | corr(signal,gamma) | "
        "corr(null,gamma) | calib k | null std |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for sigma in meta["sigmas"]:
        row = data["summary"][str(sigma)]
        lines.append(
            f"| {sigma:.2f} | {_fmt(row['corr_plant'])} | "
            f"{_fmt(row['corr_signal'])} | {_fmt(row['corr_null'])} | "
            f"{_fmt(row['calib_k'])} | {_fmt(row['null_std'])} |"
        )

    lines.append("\n## Per Seed\n")
    lines.append(
        "| seed | sigma | corr(plant,gamma) | corr(signal,gamma) | "
        "corr(null,gamma) | calib k | null std |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in data["records"]:
        lines.append(
            f"| {row['seed']} | {row['sigma']:.2f} | "
            f"{_fmt(row['corr_plant'])} | {_fmt(row['corr_signal'])} | "
            f"{_fmt(row['corr_null'])} | {_fmt(row['calib_k'])} | "
            f"{_fmt(row['null_std'])} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "The matched-null-subtracted signal detects the planted direction, but "
        "the calibration slope is much smaller than one.  Treat the residual as "
        "directional unless a separate calibration control justifies magnitude."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=OUT / "phase2_signal.json")
    ap.add_argument("--output", type=Path, default=OUT / "phase2_signal_table.md")
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.write_text(render(data), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
