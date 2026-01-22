"""Data loader skeleton for MIRT-DKVMN."""

import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .schemas import DatasetBundle


class SequenceDataset(Dataset):
    """Sequence dataset."""

    def __init__(self, questions: List[List[int]], responses: List[List[int]]) -> None:
        self.questions = questions
        self.responses = responses

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict:
        questions = torch.tensor(self.questions[idx], dtype=torch.long)
        responses = torch.tensor(self.responses[idx], dtype=torch.long)
        return {"questions": questions, "responses": responses, "student_id": idx}


def collate_sequences(batch: List[dict]) -> dict:
    """Pad variable-length sequences and build a mask."""
    questions = [item["questions"] for item in batch]
    responses = [item["responses"] for item in batch]
    student_ids = [item.get("student_id", -1) for item in batch]

    batch_size = len(batch)
    max_len = max(seq.size(0) for seq in questions)

    q_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    r_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for idx, (q_seq, r_seq) in enumerate(zip(questions, responses)):
        length = q_seq.size(0)
        q_padded[idx, :length] = q_seq
        r_padded[idx, :length] = r_seq
        mask[idx, :length] = True
    return {"questions": q_padded, "responses": r_padded, "mask": mask, "student_ids": torch.tensor(student_ids)}


class DataLoaderManager:
    """Loader supporting JSON and DKVMN-style text datasets."""

    def __init__(self, dataset_name: str, data_root: str = "data") -> None:
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)

    def load(self) -> DatasetBundle:
        """Load dataset sequences and KC metadata if present."""
        dataset_dir = self.data_root / self.dataset_name
        metadata_path = dataset_dir / "metadata.json"
        sequences_path = dataset_dir / "sequences.json"
        text_path = dataset_dir / f"{self.dataset_name}_train.txt"

        if sequences_path.exists():
            questions, responses, n_questions, n_cats = self._load_json_sequences(sequences_path, metadata_path)
        elif text_path.exists():
            questions, responses, n_questions, n_cats = self._load_text_sequences(text_path, metadata_path)
        else:
            raise FileNotFoundError(f"No sequences.json or *_train.txt found in {dataset_dir}")

        return DatasetBundle(
            questions=questions,
            responses=responses,
            n_questions=n_questions,
            n_cats=n_cats,
        )

    def load_splits(
        self,
        split_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> dict:
        """Load train/val/test splits with legacy text compatibility."""
        dataset_dir = self.data_root / self.dataset_name
        metadata_path = dataset_dir / "metadata.json"

        train_path = dataset_dir / f"{self.dataset_name}_train.txt"
        valid_path = dataset_dir / f"{self.dataset_name}_valid.txt"
        test_path = dataset_dir / f"{self.dataset_name}_test.txt"

        if train_path.exists():
            train_q, train_r, n_questions, n_cats = self._load_text_sequences(train_path, metadata_path)
            valid_q = valid_r = []
            test_q = test_r = []

            if valid_path.exists():
                valid_q, valid_r, _, _ = self._load_text_sequences(valid_path, metadata_path)
            if test_path.exists():
                test_q, test_r, _, _ = self._load_text_sequences(test_path, metadata_path)

            if not valid_q and not test_q:
                split_idx = int(len(train_q) * split_ratio)
                val_count = int(len(train_q) * val_ratio)
                valid_q = train_q[split_idx : split_idx + val_count]
                valid_r = train_r[split_idx : split_idx + val_count]
                test_q = train_q[split_idx + val_count :]
                test_r = train_r[split_idx + val_count :]
                train_q = train_q[:split_idx]
                train_r = train_r[:split_idx]

            return {
                "train": DatasetBundle(train_q, train_r, n_questions, n_cats),
                "valid": DatasetBundle(valid_q, valid_r, n_questions, n_cats),
                "test": DatasetBundle(test_q, test_r, n_questions, n_cats),
            }

        bundle = self.load()
        split_idx = int(len(bundle.questions) * split_ratio)
        val_count = int(len(bundle.questions) * val_ratio)
        train_q = bundle.questions[:split_idx]
        train_r = bundle.responses[:split_idx]
        valid_q = bundle.questions[split_idx : split_idx + val_count]
        valid_r = bundle.responses[split_idx : split_idx + val_count]
        test_q = bundle.questions[split_idx + val_count :]
        test_r = bundle.responses[split_idx + val_count :]

        return {
            "train": DatasetBundle(train_q, train_r, bundle.n_questions, bundle.n_cats),
            "valid": DatasetBundle(valid_q, valid_r, bundle.n_questions, bundle.n_cats),
            "test": DatasetBundle(test_q, test_r, bundle.n_questions, bundle.n_cats),
        }

    def build_dataloader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Build a PyTorch DataLoader with padding."""
        bundle = self.load()
        dataset = SequenceDataset(bundle.questions, bundle.responses)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sequences)

    def build_dataloaders(
        self,
        batch_size: int = 64,
        split_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> dict:
        """Build train/val/test DataLoaders."""
        splits = self.load_splits(split_ratio=split_ratio, val_ratio=val_ratio)
        loaders = {}
        for split_name, bundle in splits.items():
            dataset = SequenceDataset(bundle.questions, bundle.responses)
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split_name == "train",
                collate_fn=collate_sequences,
            )
        return loaders

    def _load_json_sequences(
        self, sequences_path: Path, metadata_path: Path
    ) -> Tuple[List[List[int]], List[List[int]], int, int]:
        with sequences_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        questions: List[List[int]] = payload["questions"]
        responses: List[List[int]] = payload["responses"]

        n_questions = payload.get("n_questions")
        n_cats = payload.get("n_cats")

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            n_questions = metadata.get("n_questions", n_questions)
            n_cats = metadata.get("n_cats", n_cats)

        if n_questions is None or n_cats is None:
            n_questions = max(max(seq) for seq in questions)
            n_cats = max(max(seq) for seq in responses) + 1

        return questions, responses, n_questions, n_cats

    def _load_text_sequences(
        self, text_path: Path, metadata_path: Path
    ) -> Tuple[List[List[int]], List[List[int]], int, int]:
        lines = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        questions: List[List[int]] = []
        responses: List[List[int]] = []

        idx = 0
        while idx + 2 < len(lines):
            length = int(lines[idx])
            q_seq = [int(x) for x in lines[idx + 1].split(",") if x]
            r_seq = [int(x) for x in lines[idx + 2].split(",") if x]
            idx += 3

            if length != len(q_seq) or length != len(r_seq):
                raise ValueError(f"Sequence length mismatch at line {idx - 3}")

            questions.append(q_seq)
            responses.append(r_seq)

        n_questions = max(max(seq) for seq in questions)
        n_cats = max(max(seq) for seq in responses) + 1

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            n_questions = metadata.get("n_questions", n_questions)
            n_cats = metadata.get("n_cats", n_cats)

        return questions, responses, n_questions, n_cats
