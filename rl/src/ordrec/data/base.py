"""``OrdinalDatasetBase`` ABC and ``AdapterConfig``.

The base contract every dataset adapter satisfies. The training loop,
reward function and evaluation harness see only this interface; per
dataset quirks live entirely inside concrete subclasses. See the design
guide at ``docs/ordrec_impl_guide.md`` Section 2.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from torch import Tensor


Split = Literal["train", "valid", "test"]

# Numeric encoding of split labels used inside ``_student_split``. The keys
# are the public ``Split`` literals and the values are the codes stored on
# disk.
_SPLIT_CODES: Dict[Split, int] = {"train": 0, "valid": 1, "test": 2}


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration shared by every adapter.

    Attributes:
        name: Adapter-instance name. Used as the on-disk subdirectory of
            ``out_dir`` and inside the metadata file.
        raw_dir: Path to the unprocessed source data (csv, parquet,
            ma-irt synthetic directory, etc.).
        out_dir: Root directory that receives the materialised artefact.
            The adapter writes to ``out_dir / name``.
        split_seed: Seed for the user-level train/valid/test split.
            Reusing the same seed across rebuilds yields byte-identical
            ``sequences.json``.
        test_frac: Fraction of students routed to the test fold.
        valid_frac: Fraction of students routed to the valid fold.
        min_seq_len: Sequences shorter than this are dropped during
            materialisation.
        max_seq_len: Hard cap on sequence length. ``0`` disables the cap.
        chunk_long_sequences: When ``True`` together with a positive
            ``max_seq_len``, sequences longer than the cap are split into
            non-overlapping chunks rather than truncated. Each chunk is
            kept under the originating student's split label.
    """

    name: str
    raw_dir: Path
    out_dir: Path
    split_seed: int = 0
    test_frac: float = 0.1
    valid_frac: float = 0.1
    min_seq_len: int = 5
    max_seq_len: int = 200
    chunk_long_sequences: bool = True


class OrdinalDatasetBase(ABC):
    """Abstract base for every ordinal recommendation dataset adapter.

    Lifecycle::

        cfg = AdapterConfig(...)
        adapter = ConcreteAdapter(cfg)
        adapter.materialise()       # writes sequences.json + metadata.json
        adapter.load()              # reads the artefact back into memory
        n = len(adapter)            # number of usable sequences
        sample = adapter[i]         # one sequence as a dict of tensors
        train_idx = adapter.get_split("train")

    Concrete subclasses must implement ``materialise()`` and ``load()``.
    All other methods rely on five conventional attributes set inside
    ``load()``:

        ``_questions``      list of per-student question id lists (1-based)
        ``_responses``      list of per-student response int lists
        ``_student_split``  ``np.ndarray[int]`` per-row split code
        ``_metadata``       dict matching ``COMMON_RECORD_SCHEMA``
        ``_q_matrix``       optional ``np.ndarray (Q, n_kcs) uint8`` or ``None``
    """

    # Sub-class-populated attributes. Declared here so static checkers do
    # not balk at access through the base class.
    _questions: list  # list[list[int]]
    _responses: list  # list[list[int]]
    _student_split: np.ndarray
    _metadata: Dict[str, Any]
    _q_matrix: Optional[np.ndarray]
    cfg: AdapterConfig

    def __init__(self, cfg: AdapterConfig) -> None:
        if not isinstance(cfg, AdapterConfig):
            raise TypeError(
                f"cfg must be an AdapterConfig instance, got {type(cfg).__name__}"
            )
        self.cfg = cfg
        # Attributes filled by ``load``. Initialised here so that calling
        # ``__len__`` before ``load`` raises a clear error message rather
        # than ``AttributeError`` deep inside numpy.
        self._questions = []
        self._responses = []
        self._student_split = np.zeros(0, dtype=np.int8)
        self._metadata = {}
        self._q_matrix = None

    # ------------------------------------------------------------------
    # Abstract surface
    # ------------------------------------------------------------------

    @abstractmethod
    def materialise(self) -> None:
        """Read raw input from ``cfg.raw_dir`` and write the canonical
        artefact under ``cfg.out_dir / cfg.name``.

        The artefact must follow ``COMMON_RECORD_SCHEMA``. Implementors
        are responsible for deterministic splits and for persisting any
        adapter-specific train-only statistics to ``coercion_artefacts.json``.
        """

    @abstractmethod
    def load(self) -> None:
        """Populate ``_questions``, ``_responses``, ``_student_split``,
        ``_metadata`` and ``_q_matrix`` from the materialised artefact.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_split(self, split: Split) -> np.ndarray:
        """Return row indices belonging to ``split``.

        Args:
            split: One of ``"train"``, ``"valid"``, ``"test"``.

        Returns:
            ``np.ndarray[int]`` of indices into ``self._questions``.
        """
        if split not in _SPLIT_CODES:
            raise ValueError(
                f"split must be one of {list(_SPLIT_CODES)}, got {split!r}"
            )
        return np.where(self._student_split == _SPLIT_CODES[split])[0]

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sequence as a dict of tensors plus a student id.

        The ``student_id`` is set to ``idx + 1`` so that 0 stays reserved
        for padding in MA-GPCM. Matches ``SequenceDataset.__getitem__`` in
        ``ma-irt/utils/dataloader.py``.
        """
        return {
            "questions": torch.tensor(self._questions[idx], dtype=torch.long),
            "responses": torch.tensor(self._responses[idx], dtype=torch.long),
            "student_id": idx + 1,
        }

    # ------------------------------------------------------------------
    # Metadata accessors (used by env, reward, eval)
    # ------------------------------------------------------------------

    def get_n_questions(self) -> int:
        return int(self._metadata["n_questions"])

    def get_n_categories(self) -> int:
        return int(self._metadata["n_categories"])

    def get_n_kcs(self) -> int:
        return int(self._metadata.get("n_kcs", 0))

    def get_q_matrix(self) -> Optional[np.ndarray]:
        return self._q_matrix

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Convenience for the on-disk path
    # ------------------------------------------------------------------

    @property
    def artefact_dir(self) -> Path:
        """The materialised-artefact directory ``cfg.out_dir / cfg.name``."""
        return Path(self.cfg.out_dir) / self.cfg.name
