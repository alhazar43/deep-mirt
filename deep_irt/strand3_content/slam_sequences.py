"""slam_sequences.py -- load SLAM learner sequences over a chosen item set.

Thin reuse of the SlamAdapter materialised artefact.  Given a set of 1-based
SLAM named ids to KEEP (e.g. the content bank's items), this returns per-learner
sequences remapped to 0-based LOCAL ids that index directly into the content
bank rows -- so ``encoder.item_val_emb`` row j and ``text_features[j]`` are the
same item.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from ordrec.data.slam import SlamAdapter
from ordrec.data.base import AdapterConfig


def load_adapter(slam_raw_dir: Path, artefact_dir: Path) -> SlamAdapter:
    cfg = AdapterConfig(
        name="slam_en_es_mc10",
        raw_dir=slam_raw_dir,
        out_dir=artefact_dir,
        min_seq_len=5,
        max_seq_len=0,
        chunk_long_sequences=False,
    )
    adapter = SlamAdapter(cfg, min_count=10)
    seq_path = artefact_dir / "slam_en_es_mc10" / "sequences.json"
    if not seq_path.exists():
        adapter.materialise()
    adapter.load()
    return adapter


def named_global_ids(adapter: SlamAdapter) -> np.ndarray:
    """1-based ids of NAMED items (catch-all buckets excluded)."""
    n_named = int(adapter._coercion["item_selection"]["n_named_items"])
    return np.arange(1, n_named + 1, dtype=np.int64)


def sequences_over_items(
    adapter: SlamAdapter,
    keep_global_ids: np.ndarray,
    global_to_local: Dict[int, int],
    splits=("train", "valid", "test"),
    min_events: int = 3,
) -> List[Dict]:
    """Per-learner sequences restricted to ``keep_global_ids``, remapped local.

    Returns dicts {student_id, questions (0-based LOCAL), responses}.  Pools all
    requested splits (item placement, not temporal generalisation, is the
    question here -- the same pooling slam_pilot uses for difficulty).
    """
    keep = set(int(g) for g in keep_global_ids)
    records: List[Dict] = []
    for split in splits:
        for i in adapter.get_split(split):
            item = adapter[i]
            qs = item["questions"].tolist()   # 1-based
            rs = item["responses"].tolist()
            pairs = [
                (global_to_local[q], r)
                for q, r in zip(qs, rs)
                if q in keep
            ]
            if len(pairs) < min_events:
                continue
            lq, lr = zip(*pairs)
            records.append({
                "student_id": int(item["student_id"]),
                "questions": torch.tensor(list(lq), dtype=torch.long),  # 0-based
                "responses": torch.tensor(list(lr), dtype=torch.long),
            })
    return records


def to_collate_records(records_local: List[Dict]) -> List[Dict]:
    """collate_adapter_items subtracts 1 (1-based -> 0-based); our ids are
    already 0-based local, so hand it (local + 1)."""
    return [
        {
            "student_id": rec["student_id"],
            "questions": rec["questions"] + 1,
            "responses": rec["responses"],
        }
        for rec in records_local
    ]
