#!/usr/bin/env python3
"""Generate synthetic GPCM data with controlled class imbalance.

This extends the standard GPCMGenerator to support realistic class imbalance
by adjusting the distribution of student abilities (theta) and item difficulties
(beta) to achieve target response distributions.

Usage::

    PYTHONPATH=src python scripts/data_gen_imbalanced.py \\
        --name large_q200_k4_mild_imbalance \\
        --n_questions 200 \\
        --n_cats 4 \\
        --n_students 5000 \\
        --target_dist 0.10 0.20 0.30 0.40 \\
        --output_dir data

The target_dist parameter specifies the desired proportion for each category.
The generator will iteratively adjust theta and beta distributions to achieve
this target while maintaining GPCM validity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np


class ImbalancedGPCMGenerator:
    """GPCM data generator with controlled class imbalance.

    Args:
        n_students: Number of student sequences to generate.
        n_questions: Item bank size Q.
        n_categories: Number of ordinal response categories K.
        seq_len_range: (min_len, max_len) for sequence lengths.
        target_dist: Target distribution for each category (must sum to 1.0).
            If None, uses balanced distribution.
        seed: Random seed for reproducibility.
        max_iterations: Maximum iterations for distribution adjustment.
        tolerance: Convergence tolerance for target distribution.
    """

    def __init__(
        self,
        n_students: int = 5000,
        n_questions: int = 200,
        n_categories: int = 4,
        seq_len_range: tuple[int, int] = (20, 80),
        target_dist: list[float] | None = None,
        seed: int = 42,
        max_iterations: int = 10,
        tolerance: float = 0.02,
    ) -> None:
        self.n_students = n_students
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.seq_len_range = seq_len_range
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._rng = np.random.default_rng(seed)

        # Set target distribution
        if target_dist is None:
            self.target_dist = np.ones(n_categories) / n_categories
        else:
            self.target_dist = np.array(target_dist, dtype=np.float64)
            assert len(self.target_dist) == n_categories
            assert np.abs(self.target_dist.sum() - 1.0) < 1e-6

        # Initialize IRT parameters with adjustment for target distribution
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize theta, alpha, beta to approximate target distribution."""
        # Start with standard priors
        self.alpha = self._rng.lognormal(mean=0.0, sigma=0.3, size=self.n_questions).astype(np.float64)

        # Adjust theta distribution based on target
        # If target skews high (more category 3), shift theta upward
        # If target skews low (more category 0), shift theta downward
        target_mean_category = np.sum(np.arange(self.n_categories) * self.target_dist)
        balanced_mean = (self.n_categories - 1) / 2.0
        theta_shift = (target_mean_category - balanced_mean) * 0.5

        self.theta = self._rng.standard_normal(self.n_students).astype(np.float64) + theta_shift

        # Initialize beta with adjustment
        # For skewed distributions, adjust the spacing between thresholds
        self.beta = np.zeros((self.n_questions, self.n_categories - 1), dtype=np.float64)
        for q in range(self.n_questions):
            base = self._rng.standard_normal() - theta_shift  # Counter-shift beta
            raw = self._rng.normal(base, 0.5, size=self.n_categories - 1)
            self.beta[q] = np.sort(raw)

    def _gpcm_prob(self, theta: float, alpha: float, betas: np.ndarray) -> np.ndarray:
        """GPCM response probabilities for a single (student, item) pair."""
        K = len(betas) + 1
        cum_logits = np.zeros(K)
        for k in range(1, K):
            cum_logits[k] = np.sum(alpha * (theta - betas[:k]))
        # Numerically stable softmax
        cum_logits -= cum_logits.max()
        exp_logits = np.exp(cum_logits)
        return exp_logits / exp_logits.sum()

    def _compute_empirical_distribution(self, sequences: list[dict]) -> np.ndarray:
        """Compute empirical response distribution from sequences."""
        all_responses = []
        for seq in sequences:
            all_responses.extend(seq['responses'])

        counts = Counter(all_responses)
        total = len(all_responses)
        empirical = np.zeros(self.n_categories)
        for k in range(self.n_categories):
            empirical[k] = counts.get(k, 0) / total
        return empirical

    def _adjust_parameters(self, empirical_dist: np.ndarray, iteration: int) -> None:
        """Adjust theta and beta to move empirical distribution toward target."""
        # Compute error
        error = self.target_dist - empirical_dist

        # Adjust theta: if we need more high responses, increase theta
        # Use weighted error by category value with adaptive learning rate
        learning_rate = 0.5 / np.sqrt(iteration + 1)
        theta_adjustment = np.sum(error * np.arange(self.n_categories)) * learning_rate
        self.theta += theta_adjustment

        # Adjust beta: shift in opposite direction to theta
        beta_adjustment = -theta_adjustment * 0.7
        self.beta += beta_adjustment

    def generate(self) -> list[dict]:
        """Generate student sequences with iterative distribution adjustment."""
        lo, hi = self.seq_len_range

        for iteration in range(self.max_iterations):
            sequences = []

            # Generate sequences with current parameters
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

                sequences.append({"questions": [q + 1 for q in q_seq], "responses": r_seq})

            # Check convergence
            empirical_dist = self._compute_empirical_distribution(sequences)
            max_error = np.abs(empirical_dist - self.target_dist).max()

            print(f"  Iteration {iteration + 1}/{self.max_iterations}:")
            print(f"    Target:    {' '.join(f'{p:.3f}' for p in self.target_dist)}")
            print(f"    Empirical: {' '.join(f'{p:.3f}' for p in empirical_dist)}")
            print(f"    Max error: {max_error:.4f}")

            if max_error < self.tolerance:
                print(f"  Converged after {iteration + 1} iterations!")
                break

            if iteration < self.max_iterations - 1:
                self._adjust_parameters(empirical_dist, iteration)

        return sequences

    def save(self, output_dir: str | Path, name: str) -> None:
        """Generate data and write all output files."""
        out = Path(output_dir) / name
        out.mkdir(parents=True, exist_ok=True)

        print(f"Generating imbalanced GPCM data: {self.n_students} students, "
              f"{self.n_questions} questions, {self.n_categories} categories")
        print(f"Target distribution: {self.target_dist}")

        sequences = self.generate()

        # Compute final statistics
        empirical_dist = self._compute_empirical_distribution(sequences)
        all_responses = []
        for seq in sequences:
            all_responses.extend(seq['responses'])
        total_responses = len(all_responses)

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
            "target_distribution": self.target_dist.tolist(),
            "empirical_distribution": empirical_dist.tolist(),
            "total_responses": total_responses,
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
                "min": float(self.theta.min()),
                "max": float(self.theta.max()),
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
        print("\nFinal class distribution:")
        for k in range(self.n_categories):
            count = int(empirical_dist[k] * total_responses)
            print(f"  Category {k}: {count:6d} ({100*empirical_dist[k]:5.2f}%) "
                  f"[target: {100*self.target_dist[k]:5.2f}%]")
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GPCM data with controlled class imbalance."
    )
    parser.add_argument(
        "--name", required=True,
        help="Dataset name (used as subdirectory under --output_dir).",
    )
    parser.add_argument("--n_students", type=int, default=5000)
    parser.add_argument("--n_questions", type=int, default=200)
    parser.add_argument("--n_cats", type=int, default=4,
                        help="Number of ordinal response categories (K).")
    parser.add_argument("--min_seq", type=int, default=20)
    parser.add_argument("--max_seq", type=int, default=80)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target_dist", type=float, nargs="+", default=None,
        help="Target distribution for each category (must sum to 1.0). "
             "Example: --target_dist 0.10 0.20 0.30 0.40"
    )
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.02)

    args = parser.parse_args()

    gen = ImbalancedGPCMGenerator(
        n_students=args.n_students,
        n_questions=args.n_questions,
        n_categories=args.n_cats,
        seq_len_range=(args.min_seq, args.max_seq),
        target_dist=args.target_dist,
        seed=args.seed,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    gen.save(args.output_dir, args.name)


if __name__ == "__main__":
    main()

