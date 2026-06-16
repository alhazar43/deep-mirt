"""essay_data.py -- ASAP essay loading, sampling, partitioning, content features.

RQ1 pilot: format-agnostic shared quality scale on REAL graded essays.

The "items" are essays, the latent is essay QUALITY. Two formats observe it:
  - GRADED   : human holistic score (domain1_score), ordinal.
  - PAIRWISE : local-LLM "which essay is better" judgments (Path A, model-generated).

Items are partitioned into three groups:
  - GRADED-ONLY  : human score observed, never appears in a comparison.
  - PAIRWISE-ONLY: appears in comparisons, human score HELD OUT for eval.
  - BOTH         : graded AND compared (the bridge that links the two scales).

The cross-format transfer test reads a PAIRWISE-ONLY essay's position on the
GRADED scale (vs its held-out human score) and vice versa.

CRITICAL DESIGN (avoids the "both recover truth" trap)
------------------------------------------------------
Essay quality d_i is predicted from a FROZEN content representation of the
essay text, NOT from a free per-essay scalar. So a pairwise-only essay's
position is informed by the content->quality structure the shared extractor
learned from the graded essays. The frozen features are a sentence embedding
(all-MiniLM-L6-v2) if sentence-transformers is available, else handcrafted
surface features (length, type-token ratio, mean word length, punctuation).

ASAP encoding is WINDOWS-1252 (read with latin-1).
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ASAP_TSV = Path("rl/data/asap/training_set_rel3.tsv")

# Per-set score range (max raw domain1_score) for reference / K computation.
SET_SCORE_RANGE = {
    1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
    5: (0, 4), 6: (0, 4), 7: (2, 24), 8: (10, 60),
}


@dataclass
class EssaySplit:
    """Holds the sampled essays, their features, scores, and partition."""
    essay_ids: np.ndarray          # (n,) original ASAP essay_id
    texts: list                    # (n,) raw essay text
    scores: np.ndarray             # (n,) human domain1_score (ordinal-encoded 0..K-1)
    raw_scores: np.ndarray         # (n,) original human score (un-binned)
    features: torch.Tensor         # (n, feat_dim) FROZEN content features
    feat_kind: str                 # "sentence-transformer" or "handcrafted"
    K: int                         # number of ordinal categories
    graded_only_idx: np.ndarray
    pairwise_only_idx: np.ndarray
    both_idx: np.ndarray
    essay_set: int

    @property
    def n(self) -> int:
        return len(self.texts)

    @property
    def graded_idx(self) -> np.ndarray:
        """Essays the graded head trains on (GRADED-ONLY + BOTH)."""
        return np.sort(np.concatenate([self.graded_only_idx, self.both_idx]))

    @property
    def pairwise_idx(self) -> np.ndarray:
        """Essays eligible for comparisons (PAIRWISE-ONLY + BOTH)."""
        return np.sort(np.concatenate([self.pairwise_only_idx, self.both_idx]))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_asap_set(essay_set: int, tsv_path: Path = ASAP_TSV) -> list:
    """Load all essays for one essay_set. Returns list of dicts.

    Reads with latin-1 (ASAP is Windows-1252). Skips rows with missing score.
    """
    out = []
    with open(tsv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["essay_set"]) != essay_set:
                continue
            sc = row.get("domain1_score", "")
            if sc is None or sc == "":
                continue
            try:
                score = float(sc)
            except ValueError:
                continue
            out.append({
                "essay_id": int(row["essay_id"]),
                "essay": row["essay"],
                "score": score,
            })
    return out


# ---------------------------------------------------------------------------
# Content features (frozen)
# ---------------------------------------------------------------------------

def _handcrafted_features(texts: list) -> np.ndarray:
    """Robust surface features; standardized. Shape (n, 9)."""
    feats = []
    for t in texts:
        words = re.findall(r"[A-Za-z']+", t)
        n_words = max(len(words), 1)
        n_chars = max(len(t), 1)
        types = set(w.lower() for w in words)
        sents = re.split(r"[.!?]+", t)
        sents = [s for s in sents if s.strip()]
        n_sents = max(len(sents), 1)
        n_commas = t.count(",")
        n_long = sum(1 for w in words if len(w) >= 7)
        feats.append([
            np.log1p(n_words),                       # essay length (log)
            len(types) / n_words,                    # type-token ratio
            np.mean([len(w) for w in words]) if words else 0.0,  # mean word len
            n_words / n_sents,                       # mean sentence length
            n_commas / n_words,                      # punctuation density
            n_long / n_words,                        # long-word fraction
            np.log1p(n_sents),                       # n sentences (log)
            t.count("\"") / n_chars * 1000,          # quote density
            np.log1p(n_chars),                       # char length (log)
        ])
    X = np.asarray(feats, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def compute_features(texts: list, prefer_st: bool = True) -> tuple:
    """Compute FROZEN content features. Returns (features (n,d) float32, kind)."""
    if prefer_st:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(
                texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            return emb.astype(np.float32), "sentence-transformer:all-MiniLM-L6-v2"
        except Exception as e:  # pragma: no cover - depends on env
            print(f"  [features] sentence-transformers unavailable ({e}); "
                  f"falling back to handcrafted.", file=sys.stderr)
    return _handcrafted_features(texts), "handcrafted-surface-9d"


# ---------------------------------------------------------------------------
# Sampling + partition
# ---------------------------------------------------------------------------

def build_split(
    essay_set: int,
    n_essays: int,
    frac_graded_only: float = 0.35,
    frac_pairwise_only: float = 0.35,
    seed: int = 0,
    tsv_path: Path = ASAP_TSV,
    prefer_st: bool = True,
    max_chars: Optional[int] = None,
) -> EssaySplit:
    """Sample n_essays from one set, stratified across the score range, and
    partition into graded-only / pairwise-only / both.

    Stratified sampling ensures the pilot spans the full quality range rather
    than collapsing onto the modal score.
    """
    rng = np.random.default_rng(seed)
    rows = load_asap_set(essay_set, tsv_path)
    if not rows:
        raise ValueError(f"No essays found for essay_set={essay_set}")

    scores = np.array([r["score"] for r in rows])
    uniq = np.unique(scores)

    # Stratified sample across distinct score levels (roughly balanced),
    # so the quality range is well represented.
    per_level = max(1, n_essays // len(uniq))
    chosen_idx = []
    for s in uniq:
        pool = np.where(scores == s)[0]
        take = min(per_level, len(pool))
        chosen_idx.extend(rng.choice(pool, size=take, replace=False).tolist())
    # top up to n_essays if rounding left us short
    if len(chosen_idx) < n_essays:
        remaining = [i for i in range(len(rows)) if i not in set(chosen_idx)]
        extra = rng.choice(remaining, size=min(n_essays - len(chosen_idx), len(remaining)),
                           replace=False).tolist()
        chosen_idx.extend(extra)
    chosen_idx = np.array(chosen_idx[:n_essays])
    rng.shuffle(chosen_idx)

    sel = [rows[i] for i in chosen_idx]
    texts = [r["essay"] for r in sel]
    if max_chars is not None:
        texts = [t[:max_chars] for t in texts]
    raw_scores = np.array([r["score"] for r in sel])
    essay_ids = np.array([r["essay_id"] for r in sel])

    # Ordinal-encode scores to 0..K-1 (dense rank over observed levels in the
    # FULL set, so the scale is the set's native scale).
    lo, hi = SET_SCORE_RANGE[essay_set]
    score_levels = np.arange(lo, hi + 1)
    # Map each raw score to its index in the contiguous level range.
    score_to_k = {float(v): k for k, v in enumerate(score_levels)}
    ordinal = np.array([score_to_k[float(s)] for s in raw_scores], dtype=np.int64)
    K = len(score_levels)

    # Frozen content features
    features_np, feat_kind = compute_features(texts, prefer_st=prefer_st)
    features = torch.from_numpy(features_np)

    # Partition
    n = len(sel)
    perm = rng.permutation(n)
    n_go = int(round(frac_graded_only * n))
    n_po = int(round(frac_pairwise_only * n))
    graded_only_idx = np.sort(perm[:n_go])
    pairwise_only_idx = np.sort(perm[n_go:n_go + n_po])
    both_idx = np.sort(perm[n_go + n_po:])

    return EssaySplit(
        essay_ids=essay_ids,
        texts=texts,
        scores=ordinal,
        raw_scores=raw_scores,
        features=features,
        feat_kind=feat_kind,
        K=K,
        graded_only_idx=graded_only_idx,
        pairwise_only_idx=pairwise_only_idx,
        both_idx=both_idx,
        essay_set=essay_set,
    )
