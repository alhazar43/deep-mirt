"""Synthetic data generator for MIRT-GPCM with balanced categories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import math
import re

import numpy as np


@dataclass
class SyntheticConfig:
    n_students: int
    n_questions: int
    n_cats: int
    n_traits: int
    seq_len_range: Tuple[int, int]
    seed: int = 42
    balance: bool = True
    allow_negative_alpha: bool = False
    screening_theta_samples: int = 2000
    screening_min_pct: float = 0.03
    screening_max_pct: float = 0.5
    screening_max_attempts: int = 30


class MirtGpcmGenerator:
    """Generate synthetic MIRT-GPCM sequences in DKVMN text format."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.theta = self._sample_theta()
        self.alpha = self._sample_alpha()
        self.beta = self._sample_beta()

    def _sample_theta(self) -> np.ndarray:
        cfg = self.config
        rho = 0.2
        cov = (1 - rho) * np.eye(cfg.n_traits) + rho * np.ones((cfg.n_traits, cfg.n_traits))
        mean = np.zeros(cfg.n_traits)
        return self.rng.multivariate_normal(mean, cov, size=cfg.n_students)

    def _sample_alpha(self) -> np.ndarray:
        cfg = self.config
        alpha = self.rng.lognormal(mean=0.0, sigma=0.2, size=(cfg.n_questions, cfg.n_traits))
        alpha = np.clip(alpha, 0.6, 1.8)
        if cfg.allow_negative_alpha:
            signs = self.rng.choice([-1.0, 1.0], size=alpha.shape)
            alpha = alpha * signs
        return alpha

    def _sample_beta(self) -> np.ndarray:
        cfg = self.config
        beta = np.zeros((cfg.n_questions, cfg.n_cats - 1))
        if cfg.n_cats <= 2:
            base_offsets = np.array([0.0])
        elif cfg.n_cats == 3:
            base_offsets = np.array([-0.7, 0.7])
        elif cfg.n_cats == 4:
            base_offsets = np.array([-1.0, 0.0, 1.0])
        else:
            base_offsets = np.array([-1.5, -0.5, 0.5, 1.5])
        for q in range(cfg.n_questions):
            b = self.rng.normal(0.0, 1.2)
            noise = self.rng.normal(0.0, 0.2, size=cfg.n_cats - 1)
            offsets = np.sort(base_offsets[: cfg.n_cats - 1] + noise)
            beta[q] = b + offsets
        return beta

    def _estimate_dot_mean(self) -> float:
        cfg = self.config
        sample_students = min(cfg.n_students, 200)
        sample_items = min(cfg.n_questions, 200)
        theta_sample = self.theta[:sample_students]
        alpha_sample = self.alpha[:sample_items]
        dots = theta_sample @ alpha_sample.T
        return float(np.mean(dots))

    def gpcm_prob(self, theta: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        n_cats = beta.shape[0] + 1
        dot = float(np.dot(theta, alpha))
        alpha_scale = float(np.linalg.norm(alpha) / math.sqrt(alpha.shape[0]))
        cum_logits = np.zeros(n_cats)
        for k in range(1, n_cats):
            cum_logits[k] = np.sum(dot - beta[:k] * alpha_scale)
        exp_logits = np.exp(cum_logits - np.max(cum_logits))
        return exp_logits / np.sum(exp_logits)

    def generate_sequences(self) -> List[dict]:
        cfg = self.config
        sequences = []
        min_len, max_len = cfg.seq_len_range
        center = (min_len + max_len) / 2.0
        std_dev = max((max_len - min_len) / 6.0, 1.0)

        for student_id in range(cfg.n_students):
            seq_len = int(self.rng.normal(center, std_dev))
            seq_len = int(np.clip(seq_len, min_len, max_len))
            q_seq = self.rng.integers(0, cfg.n_questions, size=seq_len)
            r_seq = []
            for q_id in q_seq:
                probs = self.gpcm_prob(self.theta[student_id], self.alpha[q_id], self.beta[q_id])
                r_seq.append(int(self.rng.choice(cfg.n_cats, p=probs)))
            sequences.append(
                {
                    "student_id": student_id,
                    "seq_len": seq_len,
                    "questions": q_seq.tolist(),
                    "responses": r_seq,
                }
            )
        return sequences

    def save_text_format(self, sequences: List[dict], out_dir: Path, prefix: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{prefix}.txt").open("w", encoding="utf-8") as handle:
            for seq in sequences:
                handle.write(f"{seq['seq_len']}\n")
                handle.write(",".join(map(str, seq["questions"])) + "\n")
                handle.write(",".join(map(str, seq["responses"])) + "\n")

    def save_metadata(self, out_dir: Path, dataset_name: str, n_train: int, n_test: int) -> None:
        cfg = self.config
        metadata = {
            "dataset_name": dataset_name,
            "n_students": cfg.n_students,
            "n_questions": cfg.n_questions,
            "n_cats": cfg.n_cats,
            "n_traits": cfg.n_traits,
            "seq_len_range": list(cfg.seq_len_range),
            "train_students": n_train,
            "test_students": n_test,
            "format": "dkvmn_text",
            "response_type": "ordered_categorical",
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def save_true_params(self, out_dir: Path) -> None:
        params = {
            "theta": self.theta.tolist(),
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "n_traits": self.config.n_traits,
            "n_cats": self.config.n_cats,
        }
        with (out_dir / "true_irt_parameters.json").open("w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2)

    def generate_and_save(
        self,
        base_dir: str,
        dataset_name: str,
        split_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Path:
        sequences = self._generate_with_item_screening()
        n_train = int(len(sequences) * split_ratio)
        n_val = int(len(sequences) * val_ratio)
        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train : n_train + n_val]
        test_seqs = sequences[n_train + n_val :]

        out_dir = Path(base_dir) / dataset_name
        self.save_text_format(train_seqs, out_dir, f"{dataset_name}_train")
        self.save_text_format(val_seqs, out_dir, f"{dataset_name}_valid")
        self.save_text_format(test_seqs, out_dir, f"{dataset_name}_test")
        self.save_metadata(out_dir, dataset_name, n_train, len(test_seqs))
        self.save_true_params(out_dir)
        return out_dir

    def _estimate_item_marginals(self, theta: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        probs = self._gpcm_prob_batch(theta, alpha, beta)
        return np.mean(probs, axis=0)

    def _gpcm_prob_batch(self, theta: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        n_cats = beta.shape[0] + 1
        dot = theta @ alpha
        alpha_scale = float(np.linalg.norm(alpha) / math.sqrt(alpha.shape[0]))
        cum_beta = np.cumsum(beta) * alpha_scale
        logits = np.zeros((theta.shape[0], n_cats))
        for k in range(1, n_cats):
            logits[:, k] = k * dot - cum_beta[k - 1]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _generate_with_item_screening(self) -> List[dict]:
        cfg = self.config
        sampled_theta = self.rng.normal(0.0, 1.0, size=(cfg.screening_theta_samples, cfg.n_traits))
        alpha = np.zeros((cfg.n_questions, cfg.n_traits))
        beta = np.zeros((cfg.n_questions, cfg.n_cats - 1))
        min_pct = cfg.screening_min_pct
        max_pct = cfg.screening_max_pct
        if cfg.n_cats <= 2:
            min_pct = max(min_pct, 0.1)
            max_pct = max(max_pct, 0.9)
        for q in range(cfg.n_questions):
            attempts = 0
            while True:
                attempts += 1
                cand_alpha = self._sample_alpha()[0]
                cand_beta = self._sample_beta()[0]
                marginals = self._estimate_item_marginals(sampled_theta, cand_alpha, cand_beta)
                if marginals.max() < max_pct and marginals.min() > min_pct:
                    alpha[q] = cand_alpha
                    beta[q] = cand_beta
                    break
                if attempts >= cfg.screening_max_attempts:
                    alpha[q] = cand_alpha
                    beta[q] = cand_beta
                    break
        self.alpha = alpha
        self.beta = beta
        return self.generate_sequences()


def parse_dataset_name(name: str) -> Tuple[int, int, int, int]:
    match = re.match(r"synthetic_(\d+)_(\d+)_(\d+)(?:_d(\d+))?$", name)
    if not match:
        raise ValueError(f"Invalid dataset name: {name}")
    n_traits = int(match.group(4)) if match.group(4) else -1
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), n_traits


def build_default_config(dataset_name: str, n_traits: int, min_seq: int, max_seq: int, seed: int) -> SyntheticConfig:
    n_students, n_questions, n_cats, name_traits = parse_dataset_name(dataset_name)
    if name_traits > 0:
        n_traits = name_traits
    return SyntheticConfig(
        n_students=n_students,
        n_questions=n_questions,
        n_cats=n_cats,
        n_traits=n_traits,
        seq_len_range=(min_seq, max_seq),
        seed=seed,
    )
