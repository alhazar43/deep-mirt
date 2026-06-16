"""slam_content.py -- join SLAM item TEXT to ids, difficulty, and embeddings.

The content channel needs, per SLAM item: (a) its text, (b) a frozen
sentence-transformer embedding of that text, and (c) an OBSERVED difficulty to
correlate cold-start against.  This module assembles all three and aligns them
with the deep_irt's 0-based local item ids.

Sources
-------
- SLAM sequences + ``question_id_map`` (hash -> 1-based id) come from the
  materialised SlamAdapter artefact (reused via the slam_extend loaders).
- Item text (``format`` + word sequence) and an OBSERVED human difficulty
  (token-level mistake rate, pooled over all learners) come from
  ``deep_irt/slam_pilot/outputs/human_difficulty.csv``.

Item text for embedding
-----------------------
We embed a short natural-language rendering of the exercise:
``"<format>: <word sequence>"`` e.g. ``"reverse_translate: do n't bite ..."``.
The format token matters -- the three SLAM formats (reverse_translate,
reverse_tap, listen) differ systematically in difficulty -- so it is part of
the item's content.

Observed difficulty
-------------------
We use the human token-level mistake rate as the ground-truth item location.
This is construct-capped (exact-match grading -- the RQ3 limit), so even a
perfect content channel cannot fully recover it.  The honest bar is BEATING the
negative-control / noise floor, not matching the ceiling.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SlamContentBank:
    local_ids: np.ndarray       # (M,) 0-based deep_irt ids for items WITH text
    global_ids: np.ndarray      # (M,) 1-based SLAM named ids
    texts: List[str]            # (M,) "<format>: <words>" strings
    text_features: np.ndarray   # (M, text_dim) frozen ST embeddings
    obs_difficulty: np.ndarray  # (M,) observed human mistake-rate difficulty
    feat_kind: str


def load_id_text_difficulty(
    repo_root: Path,
    artefact_dir: Path,
) -> Dict[int, Dict[str, object]]:
    """Build ``{1-based named id: {"text": str, "difficulty": float}}``.

    Joins ``question_id_map`` (hash -> id) with the human_difficulty.csv
    (hash -> words_sample, format, human_difficulty).  Only items present in
    BOTH are returned.
    """
    meta_path = artefact_dir / "slam_en_es_mc10" / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    hash_to_id = {h: int(v) for h, v in meta["question_id_map"].items()}

    csv_path = repo_root / "deep_irt" / "slam_pilot" / "outputs" / "human_difficulty.csv"
    out: Dict[int, Dict[str, object]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row["item_hash"]
            if h not in hash_to_id:
                continue
            fmt = row["format"]
            words = row["words_sample"]
            text = f"{fmt}: {words}"
            out[hash_to_id[h]] = {
                "text": text,
                "difficulty": float(row["human_difficulty"]),
            }
    return out


def _embed_texts(texts: List[str], prefer_st: bool = True):
    """Frozen content features for a list of strings.  Reuses rq1_essay's
    all-MiniLM-L6-v2 embedder with the same handcrafted fallback."""
    from deep_irt.rq1_essay.essay_data import compute_features
    feats, kind = compute_features(texts, prefer_st=prefer_st)
    return feats, kind


def build_bank(
    repo_root: Path,
    artefact_dir: Path,
    keep_global_ids: Optional[np.ndarray] = None,
    prefer_st: bool = True,
) -> SlamContentBank:
    """Assemble the content bank for SLAM named items.

    Parameters
    ----------
    keep_global_ids : optional (subset of 1-based named ids).  When given, only
        these items are embedded and returned (e.g. items that actually appear
        in the filtered sequences).  When None, all items with text are used.
    """
    id_info = load_id_text_difficulty(repo_root, artefact_dir)

    if keep_global_ids is not None:
        keep = set(int(g) for g in keep_global_ids)
        id_info = {g: v for g, v in id_info.items() if g in keep}

    global_ids = np.array(sorted(id_info.keys()), dtype=np.int64)
    texts = [str(id_info[int(g)]["text"]) for g in global_ids]
    obs_diff = np.array([float(id_info[int(g)]["difficulty"]) for g in global_ids])

    feats, kind = _embed_texts(texts, prefer_st=prefer_st)

    return SlamContentBank(
        local_ids=np.arange(len(global_ids), dtype=np.int64),
        global_ids=global_ids,
        texts=texts,
        text_features=feats,
        obs_difficulty=obs_diff,
        feat_kind=kind,
    )
