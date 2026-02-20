"""Dataset, collation, and DataLoader management for kt_gpcm.

Data format
-----------
Each dataset lives in a directory:

    <data_dir>/<dataset_name>/
        sequences.json         — list of {questions, responses} dicts
        metadata.json          — {n_questions, n_categories, ...}
        true_irt_parameters.json  — (optional) ground-truth IRT params

``sequences.json`` schema::

    [
        {"questions": [4, 7, 2, ...], "responses": [0, 2, 1, ...]},
        ...
    ]

Padding convention
------------------
Variable-length sequences are padded with 0s to the batch maximum length.
The returned ``mask`` tensor marks valid positions as ``True``.  Item IDs
use 1-based indexing inside the model (ID 0 = padding / unknown).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config.types import Config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping student response sequences.

    Args:
        questions: List of question-ID sequences (variable length).
        responses: List of response-category sequences (same lengths).
        min_seq_len: Sequences shorter than this are silently dropped.
    """

    def __init__(
        self,
        questions: List[List[int]],
        responses: List[List[int]],
        min_seq_len: int = 1,
    ) -> None:
        assert len(questions) == len(responses), (
            "questions and responses must have the same number of sequences"
        )
        # Filter short sequences
        pairs = [
            (q, r)
            for q, r in zip(questions, responses)
            if len(q) >= min_seq_len
        ]
        self._questions = [p[0] for p in pairs]
        self._responses = [p[1] for p in pairs]

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "questions": torch.tensor(self._questions[idx], dtype=torch.long),
            "responses": torch.tensor(self._responses[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_sequences(batch: List[dict]) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad a list of variable-length sequences to the batch maximum length.

    Args:
        batch: List of dicts with keys ``"questions"`` and ``"responses"``.

    Returns:
        Tuple ``(questions, responses, mask)`` each of shape ``(B, S_max)``
        where ``S_max`` is the longest sequence in the batch.
        ``mask[b, t]`` is ``True`` when position *t* of sequence *b* is
        valid (not padding).
    """
    B = len(batch)
    max_len = max(item["questions"].shape[0] for item in batch)

    q_pad = torch.zeros(B, max_len, dtype=torch.long)
    r_pad = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        s = item["questions"].shape[0]
        q_pad[i, :s] = item["questions"]
        r_pad[i, :s] = item["responses"]
        mask[i, :s] = True

    return q_pad, r_pad, mask


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class DataModule:
    """Builds train / test DataLoaders from a dataset directory.

    Args:
        cfg: Full ``Config``; uses ``cfg.data`` and ``cfg.training``.
        base_dir: Root directory that contains the dataset folder.
            Defaults to the ``data_dir`` field of ``cfg.data``.

    After calling :meth:`build`, use :attr:`train_loader` and
    :attr:`test_loader`.
    """

    def __init__(self, cfg: Config, base_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.data_dir = Path(base_dir or cfg.data.data_dir)
        self.dataset_dir = self.data_dir / cfg.data.dataset_name

        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        # Populated after build()
        self.n_questions: int = 0
        self.n_categories: int = cfg.model.n_categories
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self) -> Tuple[DataLoader, DataLoader]:
        """Load data and create train / test DataLoaders.

        Returns:
            ``(train_loader, test_loader)``

        Raises:
            FileNotFoundError: If the dataset directory or
                ``sequences.json`` does not exist.
        """
        sequences_path = self.dataset_dir / "sequences.json"
        metadata_path = self.dataset_dir / "metadata.json"

        if not sequences_path.exists():
            raise FileNotFoundError(
                f"sequences.json not found in {self.dataset_dir}. "
                "Run scripts/data_gen.py first."
            )

        # Load sequences
        with sequences_path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)

        questions_all = [rec["questions"] for rec in records]
        responses_all = [rec["responses"] for rec in records]

        # Load metadata if present and sync model dims from it
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)
            self.n_questions = self.metadata.get("n_questions", self.cfg.model.n_questions)
            self.n_categories = self.metadata.get("n_categories", self.cfg.model.n_categories)
            # Keep cfg in sync so build_model() always sees the correct values
            self.cfg.model.n_questions = self.n_questions
            self.cfg.model.n_categories = self.n_categories

        # Train / test split
        n_total = len(questions_all)
        n_train = int(n_total * self.cfg.data.train_split)

        train_q, test_q = questions_all[:n_train], questions_all[n_train:]
        train_r, test_r = responses_all[:n_train], responses_all[n_train:]

        min_len = self.cfg.data.min_seq_len
        train_ds = SequenceDataset(train_q, train_r, min_seq_len=min_len)
        test_ds = SequenceDataset(test_q, test_r, min_seq_len=min_len)

        bs = self.cfg.training.batch_size

        self.train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            collate_fn=collate_sequences,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_sequences,
            drop_last=False,
        )

        return self.train_loader, self.test_loader

    def all_train_targets(self) -> Tensor:
        """Return a flat tensor of all training response labels.

        Used to compute class weights before training starts.
        """
        if self.train_loader is None:
            raise RuntimeError("Call build() first.")
        parts = []
        for _, responses, mask in self.train_loader:
            valid = responses.view(-1)[mask.view(-1)]
            parts.append(valid)
        return torch.cat(parts)
