"""
realdata.py -- Bridge between ordrec adapter dicts and deep_irt tensors.

Adapters (``OrdinalDatasetBase`` subclasses, or any iterable that yields
dicts with the same contract) return variable-length sequences per learner.
This module provides a single function that collates a list of such dicts
into padded rectangular tensors suitable for ``DeepIRTModel.fit()`` and
``DeepIRTModel.track()``.

Adapter ``__getitem__`` contract (from ``rl/src/ordrec/data/base.py``)::

    {
        "student_id": int,                  # 1-based
        "questions":  torch.LongTensor,     # 1-D, 1-based item ids in [1, Q]
        "responses":  torch.LongTensor,     # 1-D, responses in [0, K-1]
    }

Public API
----------
``collate_adapter_items(items, pad_id=0) -> CollatedBatch``

    items   : list of dicts matching the adapter contract, or an adapter
              supporting ``__getitem__`` / ``__len__`` (the full dataset is
              used in that case).
    pad_id  : int, value placed in padded positions of ``item_ids``
              (responses are padded with 0; neither is used in the NLL
              because those positions are masked out).

Returns a ``CollatedBatch`` namedtuple::

    item_ids     : (N, T_max) long  -- 0-based item ids; padded with pad_id
    responses    : (N, T_max) long  -- responses; padded with 0
    mask         : (N, T_max) bool  -- True for real positions
    student_ids  : (N,) long        -- original 1-based student ids
    seq_lens     : (N,) long        -- actual sequence length per learner

The ``item_ids`` are converted from 1-based (adapter contract) to 0-based
(deep_irt contract) by subtracting 1 from every real position.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Union

import torch
from torch import Tensor


class CollatedBatch(NamedTuple):
    """Output of ``collate_adapter_items``."""
    item_ids: Tensor     # (N, T_max) long, 0-based
    responses: Tensor    # (N, T_max) long
    mask: Tensor         # (N, T_max) bool, True = valid position
    student_ids: Tensor  # (N,) long, 1-based
    seq_lens: Tensor     # (N,) long


def collate_adapter_items(
    items: Union[List[Dict[str, Any]], Any],
    pad_id: int = 0,
) -> CollatedBatch:
    """Collate a list of adapter dicts into padded deep_irt tensors.

    Parameters
    ----------
    items : list of dicts (or a dataset/adapter supporting __len__ /
            __getitem__) where each element follows the adapter contract::

                {
                    "student_id": int,
                    "questions":  1-D LongTensor  (1-based item ids)
                    "responses":  1-D LongTensor  (0-based responses)
                }

    pad_id : fill value for padded positions in ``item_ids`` (default 0).
             Padded positions in ``responses`` are always filled with 0.
             These values are irrelevant to the loss because ``mask``
             selects them out.

    Returns
    -------
    CollatedBatch with fields:
        item_ids    : (N, T_max) long  -- 0-based; padded positions = pad_id
        responses   : (N, T_max) long  -- padded positions = 0
        mask        : (N, T_max) bool  -- True for real positions
        student_ids : (N,) long        -- original 1-based student ids
        seq_lens    : (N,) long        -- per-learner sequence length
    """
    # Support adapter objects with __len__ / __getitem__ directly.
    if not isinstance(items, list):
        items = [items[i] for i in range(len(items))]

    N = len(items)
    if N == 0:
        raise ValueError("items must be non-empty")

    questions_list: List[Tensor] = []
    responses_list: List[Tensor] = []
    student_ids_list: List[int] = []
    seq_lens: List[int] = []

    for rec in items:
        q = rec["questions"]
        r = rec["responses"]
        if not isinstance(q, Tensor):
            q = torch.tensor(q, dtype=torch.long)
        if not isinstance(r, Tensor):
            r = torch.tensor(r, dtype=torch.long)
        # Ensure 1-D
        q = q.view(-1).long()
        r = r.view(-1).long()
        questions_list.append(q)
        responses_list.append(r)
        student_ids_list.append(int(rec["student_id"]))
        seq_lens.append(q.size(0))

    T_max = max(seq_lens)

    item_ids_out = torch.full((N, T_max), fill_value=pad_id, dtype=torch.long)
    responses_out = torch.zeros(N, T_max, dtype=torch.long)
    mask_out = torch.zeros(N, T_max, dtype=torch.bool)

    for i, (q, r, L) in enumerate(zip(questions_list, responses_list, seq_lens)):
        # Convert 1-based question ids to 0-based for deep_irt.
        item_ids_out[i, :L] = q - 1
        responses_out[i, :L] = r
        mask_out[i, :L] = True

    return CollatedBatch(
        item_ids=item_ids_out,
        responses=responses_out,
        mask=mask_out,
        student_ids=torch.tensor(student_ids_list, dtype=torch.long),
        seq_lens=torch.tensor(seq_lens, dtype=torch.long),
    )
