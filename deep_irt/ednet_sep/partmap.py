"""
partmap.py -- Recover the per-item TOEIC ``part`` (instrument label) for EdNet.

The materialised ``EdNetAdapter`` artefact stores only ``questions`` and
``responses`` per learner; the raw ``part`` column (the TOEIC section, the
instrument axis of this study) is dropped during materialisation.  We recover
it from the raw csv: every EdNet question maps to exactly one part (verified,
0 of 12,261 questions map to more than one part), so a ``question_id -> part``
lookup is unambiguous.  We then compose it with the adapter's
``question_id_map`` (raw ``qN`` string -> 1-based local id) to obtain a dense
``local_item_id -> part`` array.

TOEIC section structure (the construct-distance axis used by RQ4):

    LISTENING : parts 1, 2, 3, 4
    READING   : parts 5, 6, 7

Public API
----------
``build_local_item_to_part(raw_csv, metadata_json) -> np.ndarray``
    Returns ``part_of_local`` of shape ``(Q + 1,)``; ``part_of_local[i]`` is
    the part (1..7) of the 1-based local item id ``i``.  Index 0 is the pad
    slot and is set to 0.

``part_section(part) -> str``  -> ``"listening"`` or ``"reading"``.

``same_section(a, b) -> bool``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PARTS = (1, 2, 3, 4, 5, 6, 7)
LISTENING_PARTS = (1, 2, 3, 4)
READING_PARTS = (5, 6, 7)


def part_section(part: int) -> str:
    """Return the TOEIC section name for a part (1..7)."""
    if part in LISTENING_PARTS:
        return "listening"
    if part in READING_PARTS:
        return "reading"
    raise ValueError(f"unknown TOEIC part {part}")


def same_section(part_a: int, part_b: int) -> bool:
    """True iff two parts belong to the same TOEIC section."""
    return part_section(part_a) == part_section(part_b)


def build_local_item_to_part(
    raw_csv: Path,
    metadata_json: Path,
) -> np.ndarray:
    """Build a dense ``local_item_id -> part`` lookup.

    Parameters
    ----------
    raw_csv       : path to the raw ``ednet.csv`` (needs ``question_id`` and
                    ``part`` columns).
    metadata_json : path to the materialised adapter ``metadata.json`` (needs
                    ``question_id_map`` and ``n_questions``).

    Returns
    -------
    part_of_local : (Q + 1,) int64.  ``part_of_local[i]`` = part of 1-based
                    local item id ``i``; index 0 (pad) = 0.

    Raises
    ------
    ValueError if any local id in the map cannot be assigned a part.
    """
    meta = json.loads(Path(metadata_json).read_text())
    qid_map: Dict[str, int] = meta["question_id_map"]  # raw 'qN' -> 1-based local
    n_questions = int(meta["n_questions"])

    raw = pd.read_csv(raw_csv, usecols=["question_id", "part"]).drop_duplicates(
        "question_id"
    )
    raw_q2part = dict(zip(raw["question_id"], raw["part"].astype(int)))

    part_of_local = np.zeros(n_questions + 1, dtype=np.int64)
    missing = []
    for raw_q, local_id in qid_map.items():
        part = raw_q2part.get(raw_q)
        if part is None:
            missing.append(raw_q)
            continue
        part_of_local[int(local_id)] = int(part)

    if missing:
        raise ValueError(
            f"{len(missing)} questions in the adapter map had no part in the raw "
            f"csv (examples: {missing[:5]})."
        )
    unassigned = np.where(part_of_local[1:] == 0)[0]
    if unassigned.size:
        raise ValueError(
            f"{unassigned.size} local item ids were not assigned a part."
        )
    return part_of_local
