"""M2 ItemTower embedding analysis.

Loads the embeddings produced by the M2 ItemTower run and produces
visualization + diagnostic artifacts under ``rl/results/v1/``.

Outputs
-------
- ``rl/results/v1/plots/m2_embedding_umap.png``
- ``rl/results/v1/data/m2_nearest_neighbors.json``
- ``rl/results/v1/data/m2_embedding_stats.json``

The pool swap smoke test is run inline and its outcome is folded into
the embedding stats JSON.

Run from repo root with::

    PYTHONPATH=ma-irt KMP_DUPLICATE_LIB_OK=TRUE \
        python rl/scripts/analyze_m2_embeddings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


REPO = Path(__file__).resolve().parents[2]
ART = REPO / "rl" / "artifacts"
OUT_PLOTS = REPO / "rl" / "results" / "v1" / "plots"
OUT_DATA = REPO / "rl" / "results" / "v1" / "data"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)


RIASEC_LETTERS = ["R", "I", "A", "S", "E", "C"]
RIASEC_COLORS = {
    "R": "#1f77b4",  # Realistic
    "I": "#2ca02c",  # Investigative
    "A": "#d62728",  # Artistic
    "S": "#ff7f0e",  # Social
    "E": "#9467bd",  # Enterprising
    "C": "#8c564b",  # Conventional
    "?": "#7f7f7f",  # missing
}
WORK_ZONE_COLORS = {
    1: "#0571b0",
    2: "#92c5de",
    3: "#f7f7f7",
    4: "#f4a582",
    5: "#ca0020",
}


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


def load_artifacts() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    emb_path = ART / "onet_v1_embed.npy"
    ids_path = ART / "onet_v1_jobids.json"
    par_path = ART / "onet_v1.parquet"

    embed = np.load(emb_path).astype(np.float32)
    job_ids = json.loads(ids_path.read_text())
    df = pd.read_parquet(par_path)
    df = df.set_index("occupation_code").loc[job_ids].reset_index()
    assert df["occupation_code"].tolist() == job_ids, "id ordering mismatch"
    assert embed.shape[0] == len(job_ids), "embed/id shape mismatch"
    return embed, job_ids, df


def primary_riasec(code: str) -> str:
    if isinstance(code, str) and code and code[0] in RIASEC_LETTERS:
        return code[0]
    return "?"


# ----------------------------------------------------------------------
# UMAP plot
# ----------------------------------------------------------------------


def make_umap_plot(
    embed: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
) -> dict:
    info: dict = {}
    primary = df["riasec_code"].fillna("").map(primary_riasec).tolist()

    method = "umap"
    coords = None
    try:
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embed)
    except Exception as exc:
        info["umap_error"] = repr(exc)
        method = "pca"
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(embed)
    info["method"] = method

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    # subplot 1: RIASEC
    ax = axes[0]
    for letter in RIASEC_LETTERS + ["?"]:
        mask = np.array([p == letter for p in primary], dtype=bool)
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.7,
            label=letter,
            c=RIASEC_COLORS[letter],
            edgecolors="none",
        )
    ax.set_title(f"O*NET ItemTower v_j ({method.upper()} 2D) by primary RIASEC")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.grid(True, alpha=0.3)
    ax.legend(title="RIASEC", loc="best", fontsize=8)

    # subplot 2: work zone
    ax = axes[1]
    wz = df["work_zone"].astype(int).to_numpy()
    for z in sorted(np.unique(wz)):
        mask = wz == z
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.7,
            label=f"WZ {z}",
            c=WORK_ZONE_COLORS.get(int(z), "#000000"),
            edgecolors="none",
        )
    ax.set_title(f"Same projection coloured by work zone")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.grid(True, alpha=0.3)
    ax.legend(title="work zone", loc="best", fontsize=8)

    fig.suptitle(
        f"M2 ItemTower embedding diagnostics (n={embed.shape[0]}, d={embed.shape[1]})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    info["plot_path"] = str(out_path)
    info["primary_riasec_counts"] = (
        pd.Series(primary).value_counts().to_dict()
    )
    return info


# ----------------------------------------------------------------------
# Nearest neighbour sanity
# ----------------------------------------------------------------------


QUERIES = {
    "Software Developers": "Software Developers",
    "Registered Nurses": "Registered Nurses",
    "Carpenters": "Carpenters",
    "Elementary School Teachers": "Elementary School Teachers, Except Special Education",
    "Financial Analysts": "Financial and Investment Analysts",
    "Graphic Designers": "Graphic Designers",
}


def resolve_query(df: pd.DataFrame, label: str, preferred_title: str) -> int:
    titles = df["title"].tolist()
    # exact match first
    if preferred_title in titles:
        return titles.index(preferred_title)
    # otherwise pick closest by case-insensitive contains
    lowered = [t.lower() for t in titles]
    for needle in (preferred_title.lower(), label.lower()):
        for i, t in enumerate(lowered):
            if needle in t:
                return i
    raise KeyError(f"could not resolve {label}")


def nearest_neighbours(
    embed: np.ndarray, df: pd.DataFrame, k: int = 5
) -> dict:
    # embeddings are L2-normalised so v_j @ q_j == cosine similarity
    sims = embed @ embed.T
    np.fill_diagonal(sims, -np.inf)  # exclude self

    out: dict = {"queries": []}
    for label, preferred in QUERIES.items():
        try:
            idx = resolve_query(df, label, preferred)
        except KeyError as exc:
            out["queries"].append(
                {
                    "query_label": label,
                    "resolution_error": str(exc),
                }
            )
            continue
        scores = sims[idx]
        top = np.argsort(-scores, kind="stable")[:k]
        neighbours = [
            {
                "rank": int(r + 1),
                "occupation_code": df["occupation_code"].iat[int(j)],
                "title": df["title"].iat[int(j)],
                "riasec_code": df["riasec_code"].iat[int(j)],
                "work_zone": int(df["work_zone"].iat[int(j)]),
                "cosine": float(scores[j]),
            }
            for r, j in enumerate(top)
        ]
        out["queries"].append(
            {
                "query_label": label,
                "resolved_title": df["title"].iat[idx],
                "resolved_code": df["occupation_code"].iat[idx],
                "resolved_riasec": df["riasec_code"].iat[idx],
                "resolved_work_zone": int(df["work_zone"].iat[idx]),
                "neighbours": neighbours,
            }
        )
    return out


# ----------------------------------------------------------------------
# Embedding statistics
# ----------------------------------------------------------------------


def embedding_stats(embed: np.ndarray, df: pd.DataFrame) -> dict:
    stats: dict = {}
    norms = np.linalg.norm(embed, axis=1)
    stats["n_jobs"] = int(embed.shape[0])
    stats["d"] = int(embed.shape[1])
    stats["norm"] = {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }

    # Pairwise cosine. Since embeddings are L2-normalised, sims = E E^T.
    # Drop self-similarities by masking the diagonal.
    sims = embed @ embed.T
    n = sims.shape[0]
    upper = sims[np.triu_indices(n, k=1)].astype(np.float32)
    pct = np.percentile(upper, [5, 25, 50, 75, 95])
    stats["pairwise_cosine"] = {
        "count": int(upper.size),
        "mean": float(upper.mean()),
        "std": float(upper.std()),
        "min": float(upper.min()),
        "max": float(upper.max()),
        "p05": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
    }

    # Effective rank using stable rank and entropy of singular values.
    _u, sv, _vt = np.linalg.svd(embed, full_matrices=False)
    sv = sv.astype(np.float64)
    sv_norm = sv / sv.sum()
    # Participation ratio (Roy and Vetterli)
    eff_rank_pr = float((sv.sum() ** 2) / np.sum(sv ** 2))
    # Entropy effective rank
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    eff_rank_entropy = float(np.exp(entropy))
    cumulative = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    components_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    stats["singular_values"] = {
        "values": [float(s) for s in sv],
        "effective_rank_participation": eff_rank_pr,
        "effective_rank_entropy": eff_rank_entropy,
        "components_for_90pct_variance": components_90,
        "components_for_95pct_variance": components_95,
    }

    # Per-RIASEC silhouette using cosine distance.
    primary = df["riasec_code"].fillna("").map(primary_riasec).to_numpy()
    keep = primary != "?"
    if keep.sum() > 1 and len(set(primary[keep])) > 1:
        score = float(
            silhouette_score(
                embed[keep],
                primary[keep],
                metric="cosine",
                random_state=42,
            )
        )
    else:
        score = float("nan")
    counts = pd.Series(primary).value_counts().to_dict()
    stats["riasec_silhouette"] = {
        "score_cosine": score,
        "n_clusters": int(len(set(primary[keep]))),
        "counts": {k: int(v) for k, v in counts.items()},
    }

    # Per-work-zone silhouette as a quick second view.
    wz = df["work_zone"].astype(int).to_numpy()
    if len(set(wz)) > 1:
        wz_score = float(
            silhouette_score(embed, wz, metric="cosine", random_state=42)
        )
    else:
        wz_score = float("nan")
    stats["work_zone_silhouette"] = {
        "score_cosine": wz_score,
        "counts": {int(k): int(v) for k, v in pd.Series(wz).value_counts().to_dict().items()},
    }
    return stats


# ----------------------------------------------------------------------
# Pool swap smoke
# ----------------------------------------------------------------------


def pool_swap_smoke() -> dict:
    """Build a fake JobPoolSpec of 50 occupations, embed it, retrieve.

    Uses the forced-fallback text encoder so the test is fast and
    network-independent.
    """

    sys.path.insert(0, str(REPO / "rl" / "src"))
    from irtrec.retrieval.job_tower import JobTower  # type: ignore
    from irtrec.retrieval.pool import JobPoolSpec  # type: ignore
    from irtrec.retrieval.index import RetrievalIndex  # type: ignore

    rng = np.random.default_rng(7)
    n = 50
    titles = [f"Fake Occupation {i:02d}" for i in range(n)]
    descriptions = [f"A synthetic occupation {i} for smoke testing." for i in range(n)]
    tasks = ["Do thing A | Do thing B | Do thing C" for _ in range(n)]
    riasec = rng.choice(["RIA", "IAS", "SEC", "ECR", "CRI", "AIS"], size=n).tolist()
    work_zones_raw = rng.integers(1, 6, size=n)
    work_zone_scaled = (work_zones_raw.astype(np.float32) - 3.0) / 2.0
    edu = rng.standard_normal(n).astype(np.float32)
    edu_mask = np.ones(n, dtype=np.float32)
    structured = np.zeros((n, 8), dtype=np.float32)
    structured[:, 0] = work_zone_scaled
    structured[:, 1] = edu
    structured[:, 2] = edu_mask
    # quick R,I,A,S,E weights from riasec
    letter_to_col = {"R": 0, "I": 1, "A": 2, "S": 3, "E": 4}
    weights = (0.6, 0.3, 0.1)
    for i, code in enumerate(riasec):
        for slot, letter in enumerate(code[:3]):
            col = letter_to_col.get(letter)
            if col is not None:
                structured[i, 3 + col] += weights[slot]
    wz_mean = work_zones_raw.mean()
    wz_std = work_zones_raw.std()
    delta = ((work_zones_raw - wz_mean) / max(wz_std, 1e-6)).astype(np.float32)

    spec = JobPoolSpec(
        occupation_codes=[f"99-{i:04d}.00" for i in range(n)],
        titles=titles,
        descriptions=descriptions,
        tasks_concat=tasks,
        structured_features=structured,
        riasec_codes=riasec,
        delta_j=delta,
        work_zones_raw=work_zones_raw,
        education_zscores_raw=edu.astype(np.float64),
        source="synthetic:pool_swap_smoke",
    )

    tower = JobTower(d=64, force_fallback=True)
    import torch

    with torch.no_grad():
        v = tower(spec).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(v, axis=1)
    index = RetrievalIndex().fit(v, spec.occupation_codes)
    q = v[0]
    ids, scores = index.topk(q, k=5)
    return {
        "ok": bool(np.allclose(norms, 1.0, atol=1e-5)),
        "n_jobs": int(n),
        "norms_min": float(norms.min()),
        "norms_max": float(norms.max()),
        "top1_self": bool(ids[0] == spec.occupation_codes[0]),
        "top5_ids": list(ids),
        "top5_scores": [float(s) for s in scores],
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main() -> None:
    embed, job_ids, df = load_artifacts()
    print(f"loaded embedding {embed.shape} for {len(job_ids)} jobs", flush=True)

    # 1. UMAP plot
    umap_info = make_umap_plot(
        embed, df, OUT_PLOTS / "m2_embedding_umap.png"
    )
    print(f"saved {umap_info['plot_path']} (method={umap_info['method']})", flush=True)

    # 2. Nearest neighbours
    nn_data = nearest_neighbours(embed, df, k=5)
    nn_path = OUT_DATA / "m2_nearest_neighbors.json"
    nn_path.write_text(json.dumps(nn_data, indent=2))
    print(f"saved {nn_path}", flush=True)

    # 3. Embedding stats + 4. Pool swap smoke
    stats = embedding_stats(embed, df)
    swap = pool_swap_smoke()
    stats["pool_swap_test"] = swap
    stats["umap_info"] = {
        "method": umap_info["method"],
        "primary_riasec_counts": umap_info["primary_riasec_counts"],
    }
    if "umap_error" in umap_info:
        stats["umap_info"]["umap_error"] = umap_info["umap_error"]
    stats_path = OUT_DATA / "m2_embedding_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"saved {stats_path}", flush=True)


if __name__ == "__main__":
    main()
