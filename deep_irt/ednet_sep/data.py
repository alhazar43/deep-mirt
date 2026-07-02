"""
data.py -- EdNet data plumbing for the separability study.

Loads the materialised ``EdNetAdapter``, attaches the per-interaction TOEIC
``part`` recovered by :mod:`partmap`, subsamples learners for tractability on
an 8 GB GPU, and exposes helpers to:

  * build a full (multi-part) padded batch for the JOINT encoder fit, and
  * extract per-learner *part-restricted subsequences* (the subsequence of a
    learner's history containing only items from a single part), which is how
    we read a per-part ability on the shared learned scale.

Why part-restricted subsequences.  Reading theta at the timesteps where part-p
items happen to fall in the full interleaved sequence contaminates the part-p
estimate with the other-part items the encoder has just consumed.  Running the
frozen encoder over ONLY the part-p events isolates "ability as measured by
part p", which is the quantity the separability questions are about.

Public API
----------
``LearnerRecord``          -- one learner: student_id, questions(1-based), responses, parts.
``load_records(...)``      -- load + subsample -> (list[LearnerRecord], n_questions, part_of_local).
``full_batch(records)``    -- CollatedBatch over full sequences (for joint fit).
``part_subsequences(records, part, min_events)``
                           -- list of {student_id, questions, responses} for one part.
``split_half(record, part, seed)``
                           -- deterministically split a learner's part-p events into two halves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ordrec.data.ednet import EdNetAdapter
from ordrec.data.base import AdapterConfig

from deep_irt.core.realdata import collate_adapter_items, CollatedBatch
from deep_irt.ednet_sep.partmap import build_local_item_to_part, PARTS


@dataclass
class LearnerRecord:
    """One learner's full multi-part history.

    Attributes
    ----------
    student_id : int            -- 1-based learner id (stable across the study)
    questions  : (T,) long      -- 1-based local item ids (adapter convention)
    responses  : (T,) long      -- ordinal responses in {0,..,K-1}
    parts      : (T,) long      -- TOEIC part (1..7) of each event
    """
    student_id: int
    questions: Tensor
    responses: Tensor
    parts: Tensor

    def part_count(self, part: int) -> int:
        return int((self.parts == part).item() if self.parts.ndim == 0
                   else (self.parts == part).sum().item())


def load_records(
    repo_root: Path,
    artefact_root: Path,
    raw_csv: Path,
    n_learners: int,
    min_part_resp: int = 10,
    min_parts: int = 2,
    max_seq_len: int = 400,
    seed: int = 0,
    split: str = "train",
) -> Tuple[List[LearnerRecord], int, np.ndarray]:
    """Load EdNet, attach parts, subsample multi-part learners.

    Selection rule: keep learners who have at least ``min_part_resp`` responses
    on at least ``min_parts`` distinct parts (so the multi-part analyses are
    well posed), then subsample ``n_learners`` of them at random (seeded).
    Long sequences are tail-truncated to ``max_seq_len`` events to bound GPU
    memory; truncation keeps the MOST RECENT events (the encoder is causal and
    the final-state ability is what we read).

    Returns
    -------
    records       : list[LearnerRecord]
    n_questions   : int
    part_of_local : (Q+1,) int64 lookup
    """
    cfg = AdapterConfig(
        name="ednet_mini",
        raw_dir=raw_csv,
        out_dir=artefact_root,
        min_seq_len=5,
        max_seq_len=0,
        chunk_long_sequences=False,
    )
    adapter = EdNetAdapter(cfg)
    adapter.load()

    meta_path = artefact_root / "ednet_mini" / "metadata.json"
    part_of_local = build_local_item_to_part(raw_csv, meta_path)
    n_questions = adapter.get_n_questions()

    split_idx = adapter.get_split(split)
    rng = np.random.default_rng(seed)

    # First pass: find eligible learners (>= min_parts parts with >= min_part_resp).
    eligible: List[int] = []
    for i in split_idx:
        qs = np.asarray(adapter._questions[i], dtype=np.int64)
        pr = part_of_local[qs]
        counts = np.array([(pr == p).sum() for p in PARTS])
        if int((counts >= min_part_resp).sum()) >= min_parts:
            eligible.append(int(i))

    rng.shuffle(eligible)
    chosen = eligible[:n_learners]

    records: List[LearnerRecord] = []
    for i in chosen:
        qs = np.asarray(adapter._questions[i], dtype=np.int64)
        rs = np.asarray(adapter._responses[i], dtype=np.int64)
        pr = part_of_local[qs]
        if qs.shape[0] > max_seq_len:
            qs = qs[-max_seq_len:]
            rs = rs[-max_seq_len:]
            pr = pr[-max_seq_len:]
        records.append(
            LearnerRecord(
                student_id=i + 1,  # 1-based, stable
                questions=torch.tensor(qs, dtype=torch.long),
                responses=torch.tensor(rs, dtype=torch.long),
                parts=torch.tensor(pr, dtype=torch.long),
            )
        )
    return records, n_questions, part_of_local


def full_batch(records: List[LearnerRecord]) -> CollatedBatch:
    """Collate full (multi-part) learner sequences for the joint encoder fit."""
    items = [
        {
            "student_id": r.student_id,
            "questions": r.questions,
            "responses": r.responses,
        }
        for r in records
    ]
    return collate_adapter_items(items)


def part_subsequences(
    records: List[LearnerRecord],
    part: int,
    min_events: int = 1,
) -> List[Dict]:
    """Extract each learner's part-restricted subsequence for ``part``.

    Returns a list of dicts ``{student_id, questions(1-based), responses}``
    containing only the events whose part equals ``part``.  Learners with
    fewer than ``min_events`` such events are dropped.
    """
    out: List[Dict] = []
    for r in records:
        sel = (r.parts == part)
        if int(sel.sum()) < min_events:
            continue
        out.append(
            {
                "student_id": r.student_id,
                "questions": r.questions[sel].clone(),
                "responses": r.responses[sel].clone(),
            }
        )
    return out


def split_half(
    record: LearnerRecord,
    part: int,
    seed: int,
) -> Optional[Tuple[Dict, Dict]]:
    """Deterministically split one learner's part-p events into two halves.

    Used to estimate per-part split-half reliability (the attenuation
    correction denominators).  The split is on the learner's part-p event
    indices, shuffled with a per-learner seed so the two halves are
    interleaved (not first-half / second-half, which would confound with the
    learning trajectory).

    Returns ``(half_a, half_b)`` dicts in the same shape as
    ``part_subsequences`` items, or ``None`` if the learner has fewer than 4
    part-p events (too few to split meaningfully).
    """
    sel = (record.parts == part).nonzero(as_tuple=True)[0]
    n = int(sel.numel())
    if n < 4:
        return None
    rng = np.random.default_rng(seed + record.student_id)
    order = rng.permutation(n)
    cut = n // 2
    idx_a = sel[order[:cut]]
    idx_b = sel[order[cut:]]
    # Preserve chronological order within each half (encoder is causal).
    idx_a, _ = torch.sort(idx_a)
    idx_b, _ = torch.sort(idx_b)

    def _mk(idx: Tensor) -> Dict:
        return {
            "student_id": record.student_id,
            "questions": record.questions[idx].clone(),
            "responses": record.responses[idx].clone(),
        }

    return _mk(idx_a), _mk(idx_b)
