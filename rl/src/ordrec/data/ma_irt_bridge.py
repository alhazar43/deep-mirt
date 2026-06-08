"""Shim between materialised adapters and ma-irt's ``SequenceDataset``.

The adapter framework owns the split policy. ``DataModule`` in
``ma-irt/utils/dataloader.py`` also owns a split policy. To avoid two
sources of truth we never call ``DataModule.build``; instead we wrap a
single materialised adapter into ``SequenceDataset`` per split and feed
those into ``DataLoader`` with the existing ``collate_sequences``
collation. The resulting batch is identical to what MA-GPCM.forward
already expects.

Importing ``SequenceDataset`` requires ``ma-irt`` on ``PYTHONPATH``.
The harness usually exports ``PYTHONPATH=ma-irt`` at the repo root; if
not, the imports here will raise a clean ``ModuleNotFoundError``
identifying the missing module rather than failing deep inside.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import DataLoader


# ``ma-irt`` is consumed via PYTHONPATH; the imports happen at function
# call time so importing this module from a context that does not need
# the bridge (e.g. running only unit tests on adapter logic) does not
# fail.

def _import_ma_irt():
    try:
        from utils.dataloader import (  # type: ignore[import-not-found]
            SequenceDataset,
            collate_sequences,
        )
    except ImportError as e:
        raise ImportError(
            "ma_irt_bridge requires ma-irt on PYTHONPATH. Run with "
            "`PYTHONPATH=ma-irt python ...` from the repo root."
        ) from e
    return SequenceDataset, collate_sequences


def adapter_to_sequence_dataset(
    adapter,
    split: str,
    *,
    min_seq_len: int = 1,
    max_seq_len: int = 0,
    id_offset: int = 1,
):
    """Build a ``SequenceDataset`` from a materialised adapter.

    Args:
        adapter: A loaded ``OrdinalDatasetBase`` instance.
        split: ``"train"``, ``"valid"`` or ``"test"``.
        min_seq_len: Forwarded to ``SequenceDataset``; sequences shorter
            than this are dropped at dataset construction time.
        max_seq_len: Forwarded to ``SequenceDataset``; ``0`` disables
            the cap. Used by chunked configurations downstream of the
            adapter when chunking happened at materialisation time.
        id_offset: Forwarded; ``1`` keeps id ``0`` reserved for padding.

    Returns:
        ``SequenceDataset``.
    """
    SequenceDataset, _ = _import_ma_irt()

    idx = adapter.get_split(split)
    # ``adapter._questions`` is a plain Python list, indexing with a
    # numpy array uses object semantics. Materialise to plain Python
    # lists so SequenceDataset's tensorisation is straightforward.
    questions = [adapter._questions[int(i)] for i in np.asarray(idx, dtype=int)]
    responses = [adapter._responses[int(i)] for i in np.asarray(idx, dtype=int)]
    return SequenceDataset(
        questions,
        responses,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        id_offset=id_offset,
    )


def adapter_to_dataloader(
    adapter,
    split: str,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    min_seq_len: int = 1,
    max_seq_len: int = 0,
    id_offset: int = 1,
) -> DataLoader:
    """Build a ``DataLoader`` from a materialised adapter.

    The collation is ``collate_sequences`` from ma-irt, so the emitted
    batch is the ``(q_pad, r_pad, mask, student_ids)`` 4-tuple that
    MAGPCM.forward consumes.
    """
    _, collate_sequences = _import_ma_irt()
    ds = adapter_to_sequence_dataset(
        adapter, split,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len, id_offset=id_offset,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sequences,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
