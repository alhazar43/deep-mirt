#!/usr/bin/env python3
"""Generate synthetic GPCM student response data.

The data generator implements the standard Generalized Partial Credit
Model (GPCM, Muraki 1992) with D = 1 latent trait.

Usage::

    PYTHONPATH=src python scripts/data_gen.py \\
        --name smoke_test \\
        --n_questions 20 \\
        --n_cats 4 \\
        --n_students 200 \\
        --output_dir data

Output (inside ``<output_dir>/<name>/``):
    sequences.json          — list of {questions, responses} dicts
    metadata.json           — dataset parameters
    true_irt_parameters.json — ground-truth theta, alpha, beta

GPCM probability formula
-------------------------
    theta ~ N(0, 1)                    student ability
    alpha ~ logNormal(0, 0.3)          item discrimination
    beta_q ~ sorted(N(0, 1), K-1)     ordered thresholds per item q

    phi_k = sum_{h=0}^{k-1} alpha * (theta - beta_h)
    P(Y = k | theta, alpha, beta) = exp(phi_k) / sum_j exp(phi_j)

This uses the standard scalar GPCM formulation (Muraki 1992), which
matches GPCMLogits.forward() in the model (‖α‖·β scaling for D=1
reduces to α·β, giving φ_k = Σ α·(θ − β_h)).  Alpha is identifiable
because scaling α by c changes the logit magnitudes without a
compensating reparameterisation of θ (once θ is anchored to N(0,1)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class GPCMGenerator:
    """Synthetic GPCM data generator.

    Args:
        n_students: Number of student sequences to generate.
        n_questions: Item bank size Q.
        n_categories: Number of ordinal response categories K.
        seq_len_range: ``(min_len, max_len)`` — each student's sequence
            length is drawn uniformly from this range.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_students: int = 500,
        n_questions: int = 100,
        n_categories: int = 5,
        seq_len_range: tuple[int, int] = (10, 50),
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_students = n_students
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.seq_len_range = seq_len_range

        # Ground-truth IRT parameters
        self.theta = rng.standard_normal(n_students).astype(np.float64)
        self.alpha = rng.lognormal(mean=0.0, sigma=0.3, size=n_questions).astype(np.float64)

        # Ordered thresholds: for each item, draw K-1 values from N(base, 0.5)
        # and sort to enforce monotonicity.
        self.beta = np.zeros((n_questions, n_categories - 1), dtype=np.float64)
        for q in range(n_questions):
            base = rng.standard_normal()
            raw = rng.normal(base, 0.5, size=n_categories - 1)
            self.beta[q] = np.sort(raw)

        self._rng = rng

    # ------------------------------------------------------------------

    def _gpcm_prob(self, theta: float, alpha: float, betas: np.ndarray) -> np.ndarray:
        """GPCM response probabilities for a single (student, item) pair.

        Uses the standard GPCM cumulative logit formula (Muraki 1992)::

            φ_k = Σ_{h=0}^{k-1} α·(θ − β_h)

        This matches GPCMLogits.forward() for D = 1 where ‖α‖ = α, making
        ground-truth β directly recoverable.  α provides an identifiable
        scale because the β term is scaled by α, breaking the α–θ
        rotational ambiguity present in the M-GPCM formulation (αθ − β).
        """
        K = len(betas) + 1
        cum_logits = np.zeros(K)
        for k in range(1, K):
            # Standard GPCM: sum of α*(θ - β_h) for h=0..k-1
            cum_logits[k] = np.sum(alpha * (theta - betas[:k]))
        # Numerically stable softmax
        cum_logits -= cum_logits.max()
        exp_logits = np.exp(cum_logits)
        return exp_logits / exp_logits.sum()

    def generate(self) -> list[dict]:
        """Generate all student sequences.

        Returns:
            List of dicts, one per student::

                {"questions": [4, 2, ...], "responses": [0, 2, ...]}
        """
        sequences = []
        lo, hi = self.seq_len_range

        for sid in range(self.n_students):
            seq_len = int(self._rng.integers(lo, hi + 1))
            q_seq = self._rng.integers(0, self.n_questions, size=seq_len).tolist()

            r_seq = []
            for qid in q_seq:
                probs = self._gpcm_prob(
                    self.theta[sid], self.alpha[qid], self.beta[qid]
                )
                response = int(self._rng.choice(self.n_categories, p=probs))
                r_seq.append(response)

            # Shift question IDs to 1-based (0 = padding in the model)
            sequences.append({"questions": [q + 1 for q in q_seq], "responses": r_seq})

        return sequences

    def save(self, output_dir: str | Path, name: str) -> None:
        """Generate data and write all output files.

        Args:
            output_dir: Root directory for datasets.
            name: Dataset name (subdirectory will be ``output_dir/name/``).
        """
        out = Path(output_dir) / name
        out.mkdir(parents=True, exist_ok=True)

        print(f"Generating GPCM data: {self.n_students} students, "
              f"{self.n_questions} questions, {self.n_categories} categories")

        sequences = self.generate()

        # sequences.json
        seq_path = out / "sequences.json"
        with seq_path.open("w", encoding="utf-8") as fh:
            json.dump(sequences, fh)
        print(f"  Wrote {len(sequences)} sequences -> {seq_path}")

        # metadata.json
        meta = {
            "dataset_name": name,
            "n_students": self.n_students,
            "n_questions": self.n_questions,
            "n_categories": self.n_categories,
            "seq_len_range": list(self.seq_len_range),
            "model_type": "GPCM",
        }
        meta_path = out / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        # true_irt_parameters.json
        irt_params = {
            "theta": self.theta.tolist(),
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "n_students": self.n_students,
            "n_questions": self.n_questions,
            "n_categories": self.n_categories,
            "theta_stats": {
                "mean": float(self.theta.mean()),
                "std": float(self.theta.std()),
            },
            "alpha_stats": {
                "mean": float(self.alpha.mean()),
                "std": float(self.alpha.std()),
            },
        }
        irt_path = out / "true_irt_parameters.json"
        with irt_path.open("w", encoding="utf-8") as fh:
            json.dump(irt_params, fh, indent=2)

        print(f"  Wrote metadata       -> {meta_path}")
        print(f"  Wrote IRT parameters -> {irt_path}")
        print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GPCM student response data."
    )
    parser.add_argument(
        "--name", required=True,
        help="Dataset name (used as subdirectory under --output_dir).",
    )
    parser.add_argument("--n_students", type=int, default=500)
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--n_cats", type=int, default=5,
                        help="Number of ordinal response categories (K).")
    parser.add_argument("--min_seq", type=int, default=10)
    parser.add_argument("--max_seq", type=int, default=50)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    gen = GPCMGenerator(
        n_students=args.n_students,
        n_questions=args.n_questions,
        n_categories=args.n_cats,
        seq_len_range=(args.min_seq, args.max_seq),
        seed=args.seed,
    )
    gen.save(args.output_dir, args.name)


if __name__ == "__main__":
    main()
