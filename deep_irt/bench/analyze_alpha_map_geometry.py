"""analyze_alpha_map_geometry.py -- E7 diagnostic over alpha-map outputs.

This is a no-training analysis step.  It reads the controlled alpha-map
ablation outputs and summarizes what the next geometry experiment should explain:

  * exp is not separated from smooth non-exp maps after matched init and LR tuning,
  * smooth positive maps cluster,
  * raw, ReLU, and non-monotone maps are less stable,
  * bounded maps can be competitive but change the high-alpha range.

Outputs:
  deep_irt/bench/outputs/alpha_map_geometry_summary.json
  deep_irt/bench/outputs/alpha_map_geometry_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"

DEFAULT_INPUTS = [
    OUT / "alpha_map_results.json",
    OUT / "alpha_map_kext_results.json",
]


def _normalise_map_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan")
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][0] == indexed[pos][0]:
            end += 1
        rank = (pos + end - 1) / 2.0 + 1.0
        for _, original_idx in indexed[pos:end]:
            ranks[original_idx] = rank
        pos = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    if len(pairs) < 3:
        return float("nan")
    xr = _ranks([p[0] for p in pairs])
    yr = _ranks([p[1] for p in pairs])
    mx = _mean(xr)
    my = _mean(yr)
    sx = math.sqrt(sum((v - mx) ** 2 for v in xr))
    sy = math.sqrt(sum((v - my) ** 2 for v in yr))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(xr, yr)) / (sx * sy)


def map_family(name: str) -> str:
    name = _normalise_map_name(name)
    if name in ("exp", "clipped_exp"):
        return "exp_family"
    if name in ("softplus", "softplus_eps", "scaled_softplus",
                "temperature_softplus"):
        return "smooth_softplus_family"
    if name in ("sigmoid", "scaled_sigmoid"):
        return "bounded_sigmoid_family"
    if name in ("identity", "raw"):
        return "unconstrained_raw"
    if name == "relu":
        return "nonsmooth_relu"
    if name == "square":
        return "nonmonotone_square"
    return "other"


def map_notes(name: str) -> str:
    family = map_family(name)
    notes = {
        "exp_family": "smooth positive, multiplicative geometry",
        "smooth_softplus_family": "smooth positive, exp-like without unbounded derivative",
        "bounded_sigmoid_family": "smooth positive but range-limited",
        "unconstrained_raw": "not positive-constrained",
        "nonsmooth_relu": "positive but has a kink and inactive side",
        "nonmonotone_square": "positive but non-monotone with zero slope near zero",
        "other": "unclassified",
    }
    return notes[family]


def preconditioner_value(name: str, alpha: float, kwargs: dict[str, Any]) -> float:
    """Return m_g(alpha) = [g'(g^{-1}(alpha))]^2 for the map."""
    name = _normalise_map_name(name)
    if not math.isfinite(alpha):
        return float("nan")
    if name in ("identity", "raw"):
        return 1.0
    if name == "relu":
        return 1.0 if alpha > 0.0 else 0.0
    if name == "softplus":
        if alpha <= 0.0:
            return float("nan")
        return (1.0 - math.exp(-alpha)) ** 2
    if name == "softplus_eps":
        epsilon = float(kwargs.get("epsilon", 1e-6))
        z = alpha - epsilon
        if z <= 0.0:
            return float("nan")
        return (1.0 - math.exp(-z)) ** 2
    if name == "scaled_softplus":
        scale = float(kwargs.get("scale", 1.0))
        z = alpha / scale
        if z <= 0.0:
            return float("nan")
        return (scale * (1.0 - math.exp(-z))) ** 2
    if name == "temperature_softplus":
        temperature = float(kwargs.get("temperature", 1.0))
        if alpha <= 0.0:
            return float("nan")
        return (temperature * (1.0 - math.exp(-alpha))) ** 2
    if name == "sigmoid":
        if not 0.0 < alpha < 1.0:
            return float("nan")
        return (alpha * (1.0 - alpha)) ** 2
    if name == "scaled_sigmoid":
        scale = float(kwargs.get("scale", 1.0))
        if not 0.0 < alpha < scale:
            return float("nan")
        return (alpha * (1.0 - alpha / scale)) ** 2
    if name == "square":
        epsilon = float(kwargs.get("epsilon", 1e-6))
        if alpha <= epsilon:
            return float("nan")
        return 4.0 * (alpha - epsilon)
    if name == "exp":
        return alpha ** 2
    if name == "clipped_exp":
        clip_min = float(kwargs.get("clip_min", -20.0))
        clip_max = float(kwargs.get("clip_max", 20.0))
        if alpha <= 0.0:
            return float("nan")
        log_alpha = math.log(alpha)
        if log_alpha <= clip_min or log_alpha >= clip_max:
            return 0.0
        return alpha ** 2
    return float("nan")


def load_best_rows(paths: list[Path]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    sources: list[str] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for row in data.get("best_lr", []):
            key = (int(row["K"]), row["map"])
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["source_file"] = str(path)
            rows.append(out)
    rows.sort(key=lambda r: (int(r["K"]), map_family(r["map"]), r["map"]))
    return rows, sources


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        kwargs = dict(row.get("map_kwargs") or {})
        alpha_init = _f(row.get("alpha_init"))
        alpha_p50 = _f(row.get("alpha_p50_mean"))
        alpha_p95 = _f(row.get("alpha_p95_mean"))
        out = dict(row)
        out["family"] = map_family(row["map"])
        out["notes"] = map_notes(row["map"])
        out["m_init"] = preconditioner_value(row["map"], alpha_init, kwargs)
        out["m_p50"] = preconditioner_value(row["map"], alpha_p50, kwargs)
        out["m_p95"] = preconditioner_value(row["map"], alpha_p95, kwargs)
        enriched.append(out)
    return enriched


def best_generic_by_k(rows: list[dict]) -> list[dict]:
    out = []
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_k[int(row["K"])].append(row)
    for k in sorted(by_k):
        k_rows = by_k[k]
        exp_rows = [r for r in k_rows if r["map"] == "exp"]
        generic = [r for r in k_rows if r["family"] != "exp_family"]
        if not exp_rows or not generic:
            continue
        exp = exp_rows[0]
        best = max(generic, key=lambda r: _f(r.get("a_spearman_mean")))
        out.append({
            "K": k,
            "exp_alpha_spearman": _f(exp.get("a_spearman_mean")),
            "exp_alpha_high_spearman": _f(exp.get("a_high_spearman_mean")),
            "exp_lr": _f(exp.get("lr")),
            "best_generic_map": best["map"],
            "best_generic_family": best["family"],
            "best_generic_alpha_spearman": _f(best.get("a_spearman_mean")),
            "best_generic_alpha_high_spearman": _f(best.get("a_high_spearman_mean")),
            "best_generic_lr": _f(best.get("lr")),
            "delta_alpha_spearman": (
                _f(exp.get("a_spearman_mean")) - _f(best.get("a_spearman_mean"))
            ),
            "delta_high_alpha_spearman": (
                _f(exp.get("a_high_spearman_mean")) -
                _f(best.get("a_high_spearman_mean"))
            ),
        })
    return out


def family_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["K"]), row["family"])].append(row)
    out = []
    for (k, family), rs in sorted(groups.items()):
        out.append({
            "K": k,
            "family": family,
            "n_maps": len(rs),
            "alpha_spearman_mean": _mean([_f(r.get("a_spearman_mean")) for r in rs]),
            "alpha_spearman_std": _std([_f(r.get("a_spearman_mean")) for r in rs]),
            "high_alpha_spearman_mean": _mean(
                [_f(r.get("a_high_spearman_mean")) for r in rs]
            ),
            "m_p95_mean": _mean([_f(r.get("m_p95")) for r in rs]),
            "grad_mean": _mean([_f(r.get("alpha_grad_norm_mean_mean")) for r in rs]),
        })
    return out


def correlations_by_k(rows: list[dict]) -> list[dict]:
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_k[int(row["K"])].append(row)
    out = []
    for k in sorted(by_k):
        rs = by_k[k]
        y = [_f(r.get("a_spearman_mean")) for r in rs]
        out.append({
            "K": k,
            "spearman_alpha_recovery_vs_m_p95": spearman(
                [_f(r.get("m_p95")) for r in rs], y
            ),
            "spearman_alpha_recovery_vs_alpha_p95": spearman(
                [_f(r.get("alpha_p95_mean")) for r in rs], y
            ),
            "spearman_alpha_recovery_vs_alpha_grad_norm": spearman(
                [_f(r.get("alpha_grad_norm_mean_mean")) for r in rs], y
            ),
        })
    return out


def _fmt(value: float, p: int = 3) -> str:
    value = _f(value)
    if value != value:
        return "-"
    return f"{value:.{p}f}"


def render_markdown(
    sources: list[str],
    comparison: list[dict],
    rows: list[dict],
    families: list[dict],
    correlations: list[dict],
) -> str:
    lines = ["# Alpha-Map Geometry Summary\n"]
    lines.append("Source files:\n")
    for source in sources:
        lines.append(f"- `{source}`")
    lines.append(
        "\nThis analysis uses the already completed controlled alpha-map runs. "
        "It does not train new models.\n"
    )

    lines.append("## Exp Versus Best Generic Smooth Map\n")
    lines.append("| K | exp lr | exp a_sp | best generic | generic lr | generic a_sp | delta | high-alpha delta |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|---:|")
    for row in comparison:
        lines.append(
            f"| {row['K']} | {_fmt(row['exp_lr'], 4)} | "
            f"{_fmt(row['exp_alpha_spearman'])} | "
            f"{row['best_generic_map']} | {_fmt(row['best_generic_lr'], 4)} | "
            f"{_fmt(row['best_generic_alpha_spearman'])} | "
            f"{_fmt(row['delta_alpha_spearman'])} | "
            f"{_fmt(row['delta_high_alpha_spearman'])} |"
        )

    lines.append("\n## Best-LR Map Geometry\n")
    lines.append("| K | map | family | lr | a_sp | high-a_sp | alpha p95 | m(init) | m(p95) | grad |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (int(r["K"]), r["map"])):
        lines.append(
            f"| {int(row['K'])} | {row['map']} | {row['family']} | "
            f"{_fmt(row.get('lr'), 4)} | {_fmt(row.get('a_spearman_mean'))} | "
            f"{_fmt(row.get('a_high_spearman_mean'))} | "
            f"{_fmt(row.get('alpha_p95_mean'))} | {_fmt(row.get('m_init'))} | "
            f"{_fmt(row.get('m_p95'))} | "
            f"{_fmt(row.get('alpha_grad_norm_mean_mean'))} |"
        )

    lines.append("\n## Family Summary\n")
    lines.append("| K | family | n | mean a_sp | std a_sp | mean high-a_sp | mean m(p95) | mean grad |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in families:
        lines.append(
            f"| {row['K']} | {row['family']} | {row['n_maps']} | "
            f"{_fmt(row['alpha_spearman_mean'])} | "
            f"{_fmt(row['alpha_spearman_std'])} | "
            f"{_fmt(row['high_alpha_spearman_mean'])} | "
            f"{_fmt(row['m_p95_mean'])} | {_fmt(row['grad_mean'])} |"
        )

    lines.append("\n## Correlations\n")
    lines.append("| K | recovery vs m(p95) | recovery vs alpha p95 | recovery vs alpha grad |")
    lines.append("|---:|---:|---:|---:|")
    for row in correlations:
        lines.append(
            f"| {row['K']} | "
            f"{_fmt(row['spearman_alpha_recovery_vs_m_p95'])} | "
            f"{_fmt(row['spearman_alpha_recovery_vs_alpha_p95'])} | "
            f"{_fmt(row['spearman_alpha_recovery_vs_alpha_grad_norm'])} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "- The exp family does not separate from the best generic smooth maps at "
        "`K=2,4,8` after matched initialization and LR tuning."
    )
    lines.append(
        "- The next geometry control should explain the smooth-map cluster and "
        "the weaker raw/ReLU/non-monotone behavior, not prove exp optimality."
    )
    lines.append(
        "- Induced preconditioning should be treated as one diagnostic alongside "
        "smoothness, saturation, monotonicity, and effective-alpha range."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument(
        "--out-prefix",
        default="alpha_map_geometry_summary",
        help="Output prefix under deep_irt/bench/outputs.",
    )
    args = ap.parse_args()

    rows, sources = load_best_rows(args.inputs)
    if not rows:
        raise SystemExit("No best_lr rows found. Run alpha-map benches first.")
    enriched = enrich_rows(rows)
    comparison = best_generic_by_k(enriched)
    families = family_summary(enriched)
    correlations = correlations_by_k(enriched)

    blob = {
        "sources": sources,
        "comparison": comparison,
        "rows": enriched,
        "family_summary": families,
        "correlations": correlations,
    }
    out_json = OUT / f"{args.out_prefix}.json"
    out_md = OUT / f"{args.out_prefix}.md"
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    out_md.write_text(
        render_markdown(sources, comparison, enriched, families, correlations),
        encoding="utf-8",
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
