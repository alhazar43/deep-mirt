"""helpsteer2_data.py -- HelpSteer2 loading, partition, frozen content features.

RQ1 on REAL human dual-format data. The "items" are LLM RESPONSES; the latent is
response HELPFULNESS quality. Two HUMAN formats observe it:
  - GRADED   : human helpfulness rating 0-4 (ordinal, K=5).
  - PAIRWISE : REAL signed human preference between two responses to the same
               prompt (preference_strength in [-3,3]; sign gives the winner).
               NO model judge -- these are human annotations.

Items are partitioned at the RESPONSE-POOL level into:
  - GRADED-ONLY  : helpfulness observed, its preferences HELD OUT.
  - PAIRWISE-ONLY: appears in comparisons, helpfulness HELD OUT for eval.
  - BOTH         : graded AND compared (the bridge linking the two scales).

A comparison is usable iff BOTH of its responses are pairwise-eligible
(PAIRWISE-ONLY or BOTH). The held-out-format transfer test then asks whether a
PAIRWISE-ONLY response lands correctly on the GRADED helpfulness scale (vs its
held-out human rating) and vice versa.

CRITICAL DESIGN (avoids the both-recover-truth trap)
----------------------------------------------------
Response quality d_i is predicted from a FROZEN sentence-embedding of the
(prompt, response) text, NOT a free per-item scalar. A pairwise-only response's
position is informed by the content->quality map the shared extractor learned
from the graded responses. Frozen features = all-MiniLM-L6-v2 (384d).
"""

from __future__ import annotations

import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA = Path("rl/data/helpsteer2")
K_HELP = 5  # helpfulness is 0..4


@dataclass
class HS2Split:
    """HelpSteer2 split: responses, features, helpfulness, partition, and the
    REAL human comparisons (in response-index space)."""
    response_keys: list            # (n,) (prompt, response) tuples
    texts: list                    # (n,) prompt+response text used for features
    scores: np.ndarray             # (n,) helpfulness ordinal 0..K-1
    raw_scores: np.ndarray         # (n,) helpfulness rating (== scores here)
    features: torch.Tensor         # (n, feat_dim) FROZEN content features
    feat_kind: str
    K: int
    graded_only_idx: np.ndarray
    pairwise_only_idx: np.ndarray
    both_idx: np.ndarray
    comparisons: list              # (i, j, outcome): outcome=1 => i beats j
    essay_set: str = "helpsteer2"  # label reused by the report formatter

    @property
    def n(self) -> int:
        return len(self.texts)

    @property
    def graded_idx(self) -> np.ndarray:
        return np.sort(np.concatenate([self.graded_only_idx, self.both_idx]))

    @property
    def pairwise_idx(self) -> np.ndarray:
        return np.sort(np.concatenate([self.pairwise_only_idx, self.both_idx]))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_base_help():
    """(prompt, response) -> mean helpfulness rating (float)."""
    from collections import defaultdict
    acc = defaultdict(list)
    for fn in ["train.jsonl.gz", "validation.jsonl.gz"]:
        with gzip.open(DATA / fn, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                acc[(r["prompt"], r["response"])].append(r["helpfulness"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _load_pref():
    rows = []
    with gzip.open(DATA / "preference" / "preference.jsonl.gz", "rt",
                   encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Content features (frozen)
# ---------------------------------------------------------------------------

def compute_features(texts: list, prefer_st: bool = True,
                     max_chars: int = 2000) -> tuple:
    """FROZEN content features. (n,d) float32 + kind label. all-MiniLM-L6-v2."""
    clipped = [t[:max_chars] for t in texts]
    if prefer_st:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(
                clipped, batch_size=64, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            return emb.astype(np.float32), "sentence-transformer:all-MiniLM-L6-v2"
        except Exception as e:  # pragma: no cover
            print(f"  [features] sentence-transformers unavailable ({e}); "
                  f"falling back to handcrafted.", file=sys.stderr)
    # minimal surface fallback
    import re
    feats = []
    for t in clipped:
        words = re.findall(r"[A-Za-z']+", t)
        nw = max(len(words), 1)
        feats.append([
            np.log1p(len(words)),
            len(set(w.lower() for w in words)) / nw,
            np.mean([len(w) for w in words]) if words else 0.0,
            np.log1p(len(t)),
        ])
    X = np.asarray(feats, np.float64)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    return X.astype(np.float32), "handcrafted-surface-4d"


# ---------------------------------------------------------------------------
# Build split
# ---------------------------------------------------------------------------

def build_split(
    n_pairs: int = 4000,
    frac_graded_only: float = 0.35,
    frac_pairwise_only: float = 0.35,
    seed: int = 0,
    prefer_st: bool = True,
    max_chars: int = 2000,
    drop_ties: bool = True,
) -> HS2Split:
    """Build a HelpSteer2 split.

    Strategy
    --------
    1. Load base helpfulness ratings and preference pairs.
    2. Keep preference pairs whose BOTH responses join to a base rating, and
       (if drop_ties) whose preference is non-tied (|strength| >= 1).
    3. Sample n_pairs of these (uniform; pair choice ignores helpfulness ->
       no leakage). Collect the unique responses appearing in sampled pairs;
       these are the comparable pool. Build the item index over them.
    4. Partition the responses (item pool) into graded-only / pairwise-only /
       both at the response level. A response in pairwise-eligible (PO/BOTH)
       keeps its comparisons; a graded-eligible (GO/BOTH) keeps its rating.
    5. A sampled comparison is RETAINED iff both its responses are
       pairwise-eligible (so PAIRWISE-ONLY responses are evaluated purely from
       comparisons, GRADED-ONLY purely from rating).
    """
    rng = np.random.default_rng(seed)
    help_mean = _load_base_help()
    prefs = _load_pref()

    # Valid joined, optionally non-tied
    valid = []
    for r in prefs:
        k1 = (r["prompt"], r["response_1"])
        k2 = (r["prompt"], r["response_2"])
        if k1 not in help_mean or k2 not in help_mean:
            continue
        s = r["preference_strength"]
        if drop_ties and s == 0:
            continue
        valid.append((k1, k2, s))
    rng.shuffle(valid)
    valid = valid[:n_pairs]
    if not valid:
        raise ValueError("No valid HelpSteer2 preference pairs after join/tie filter.")

    # Item pool = unique responses appearing in the sampled pairs
    key_to_idx = {}
    keys = []
    for k1, k2, _ in valid:
        for k in (k1, k2):
            if k not in key_to_idx:
                key_to_idx[k] = len(keys)
                keys.append(k)
    n = len(keys)

    raw_scores = np.array([help_mean[k] for k in keys], dtype=np.float64)
    # Ordinal-encode helpfulness: round mean to nearest int 0..4 for the GPCM
    # category target (the graded head needs integer categories). raw_scores
    # keeps the (possibly fractional) mean for correlation eval.
    ordinal = np.clip(np.round(raw_scores), 0, K_HELP - 1).astype(np.int64)

    texts = [k[0] + "\n\n" + k[1] for k in keys]  # prompt + response

    features_np, feat_kind = compute_features(texts, prefer_st=prefer_st,
                                              max_chars=max_chars)
    features = torch.from_numpy(features_np)

    # Partition responses into GO / PO / BOTH
    perm = rng.permutation(n)
    n_go = int(round(frac_graded_only * n))
    n_po = int(round(frac_pairwise_only * n))
    graded_only_idx = np.sort(perm[:n_go])
    pairwise_only_idx = np.sort(perm[n_go:n_go + n_po])
    both_idx = np.sort(perm[n_go + n_po:])

    pairwise_eligible = set(pairwise_only_idx.tolist()) | set(both_idx.tolist())

    # Build comparisons in item-index space. outcome=1 => i beats j.
    # preference_strength>0 => response_2 (k2) better; <0 => response_1 (k1).
    comparisons = []
    for k1, k2, s in valid:
        i = key_to_idx[k1]; j = key_to_idx[k2]
        if i not in pairwise_eligible or j not in pairwise_eligible:
            continue
        if s > 0:           # k2 (j) wins
            comparisons.append((j, i, 1))
        elif s < 0:         # k1 (i) wins
            comparisons.append((i, j, 1))
        # s == 0 already dropped if drop_ties

    return HS2Split(
        response_keys=keys,
        texts=texts,
        scores=ordinal,
        raw_scores=raw_scores,
        features=features,
        feat_kind=feat_kind,
        K=K_HELP,
        graded_only_idx=graded_only_idx,
        pairwise_only_idx=pairwise_only_idx,
        both_idx=both_idx,
        comparisons=comparisons,
    )
