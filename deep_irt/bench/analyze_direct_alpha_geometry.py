"""analyze_direct_alpha_geometry.py -- combine direct-alpha geometry runs.

The direct-alpha protocol was run in several LR chunks.  This script combines
those outputs and reports the best LR per condition, plus whether a condition is
still sitting on the top of the explored LR grid.

Outputs:
  deep_irt/bench/outputs/direct_alpha_geometry_summary.json
  deep_irt/bench/outputs/direct_alpha_geometry_summary.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"

DEFAULT_INPUTS = [
    OUT / "direct_alpha_geometry_results.json",
    OUT / "direct_alpha_geometry_lrext_results.json",
    OUT / "direct_alpha_geometry_lrext30_results.json",
]


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, p: int = 3) -> str:
    value = _f(value)
    return "-" if value != value else f"{value:.{p}f}"


def load_agg_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for row in data.get("agg", []):
            key = (int(row["K"]), row["condition"], float(row["lr"]))
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["source_file"] = str(path)
            rows.append(out)
    rows.sort(key=lambda r: (int(r["K"]), r["condition"], float(r["lr"])))
    return rows, sources


def best_by_condition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["K"]), row["condition"])].append(row)
    out = []
    for (k, condition), rs in sorted(groups.items()):
        lrs = sorted({_f(r["lr"]) for r in rs})
        max_lr = max(lrs)
        chosen = max(
            rs,
            key=lambda r: (
                _score(r.get("a_spearman_mean")),
                _score(r.get("recovery_auc_mean")),
                -_f(r.get("lr")),
            ),
        )
        best = dict(chosen)
        best["max_lr_seen"] = max_lr
        best["best_at_max_lr"] = abs(_f(best["lr"]) - max_lr) <= 1e-12
        low = [r for r in rs if abs(_f(r["lr"]) - 10.0) <= 1e-12]
        high = [r for r in rs if abs(_f(r["lr"]) - 30.0) <= 1e-12]
        if low and high:
            best["delta_lr30_minus_lr10"] = (
                _f(high[0].get("a_spearman_mean")) -
                _f(low[0].get("a_spearman_mean"))
            )
        else:
            best["delta_lr30_minus_lr10"] = float("nan")
        out.append(best)
    return out


def _score(value: Any) -> float:
    value = _f(value)
    return value if value == value else -1.0e12


def best_overall_by_k(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        by_k[int(row["K"])].append(row)
    out = []
    for k, rs in sorted(by_k.items()):
        chosen = max(
            rs,
            key=lambda r: (
                _score(r.get("a_spearman_mean")),
                _score(r.get("recovery_auc_mean")),
                -_f(r.get("lr")),
            ),
        )
        out.append(dict(chosen))
    return out


def family_label(condition: str) -> str:
    if condition in ("unconstrained", "projected"):
        return "identity_alpha_space"
    if condition in ("exp_precond", "softplus_precond", "scaled_softplus_precond",
                     "temperature_softplus_precond"):
        return "smooth_positive_preconditioner"
    if condition in ("sigmoid_precond", "scaled_sigmoid_precond"):
        return "bounded_preconditioner"
    if condition == "square_precond":
        return "square_preconditioner"
    return "other"


def family_summary(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        groups[(int(row["K"]), family_label(row["condition"]))].append(row)
    out = []
    for (k, family), rs in sorted(groups.items()):
        vals = np.array([_f(r.get("a_spearman_mean")) for r in rs], dtype=float)
        vals = vals[np.isfinite(vals)]
        high = np.array([_f(r.get("a_high_spearman_mean")) for r in rs], dtype=float)
        high = high[np.isfinite(high)]
        out.append({
            "K": k,
            "family": family,
            "n_conditions": len(rs),
            "alpha_spearman_mean": float(np.mean(vals)) if vals.size else float("nan"),
            "alpha_spearman_std": float(np.std(vals)) if vals.size else float("nan"),
            "high_alpha_spearman_mean": (
                float(np.mean(high)) if high.size else float("nan")
            ),
        })
    return out


def render_markdown(
    sources: list[str],
    best_rows: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> str:
    lines = ["# Direct Alpha Geometry Summary\n"]
    lines.append("Source files:\n")
    for source in sources:
        lines.append(f"- `{source}`")

    lines.append("\n## Overall Best By K\n")
    lines.append("| K | condition | best lr | a_sp | high-a_sp | rec AUC | at max LR |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    for row in overall:
        lines.append(
            f"| {row['K']} | {row['condition']} | {_fmt(row['lr'], 4)} | "
            f"{_fmt(row['a_spearman_mean'])} | "
            f"{_fmt(row['a_high_spearman_mean'])} | "
            f"{_fmt(row['recovery_auc_mean'])} | "
            f"{'yes' if row['best_at_max_lr'] else 'no'} |"
        )

    lines.append("\n## Best LR Per Condition\n")
    lines.append(
        "| K | condition | family | best lr | a_sp | high-a_sp | rec AUC | "
        "delta 30-10 | at max LR |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
    for row in best_rows:
        lines.append(
            f"| {row['K']} | {row['condition']} | "
            f"{family_label(row['condition'])} | {_fmt(row['lr'], 4)} | "
            f"{_fmt(row['a_spearman_mean'])} | "
            f"{_fmt(row['a_high_spearman_mean'])} | "
            f"{_fmt(row['recovery_auc_mean'])} | "
            f"{_fmt(row['delta_lr30_minus_lr10'])} | "
            f"{'yes' if row['best_at_max_lr'] else 'no'} |"
        )

    lines.append("\n## Family Summary\n")
    lines.append("| K | family | n | mean a_sp | std a_sp | mean high-a_sp |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in families:
        lines.append(
            f"| {row['K']} | {row['family']} | {row['n_conditions']} | "
            f"{_fmt(row['alpha_spearman_mean'])} | "
            f"{_fmt(row['alpha_spearman_std'])} | "
            f"{_fmt(row['high_alpha_spearman_mean'])} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "- With a wide LR grid, direct alpha-space conditions mostly converge to "
        "the same recovery band when theta and beta are fixed."
    )
    lines.append(
        "- This does not reproduce the neural-map separation where raw/ReLU-like "
        "maps are weaker.  The neural result is therefore not explained by "
        "alpha-space preconditioner magnitude alone."
    )
    lines.append(
        "- Bounded sigmoid remains lower when the true or fitted alpha range "
        "exceeds its cap; that is a range restriction, not an exp advantage."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-prefix", default="direct_alpha_geometry_summary")
    args = ap.parse_args()

    rows, sources = load_agg_rows(args.inputs)
    if not rows:
        raise SystemExit("No aggregate rows found. Run direct-alpha geometry first.")
    best_rows = best_by_condition(rows)
    overall = best_overall_by_k(best_rows)
    families = family_summary(best_rows)

    blob = {
        "sources": sources,
        "best_by_condition": best_rows,
        "best_overall_by_k": overall,
        "family_summary": families,
    }
    out_json = OUT / f"{args.out_prefix}.json"
    out_md = OUT / f"{args.out_prefix}.md"
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    out_md.write_text(
        render_markdown(sources, best_rows, overall, families),
        encoding="utf-8",
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
