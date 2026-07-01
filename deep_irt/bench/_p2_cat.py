"""
_p2_cat.py -- Computerised Adaptive Testing simulation for Phase 2 (E8).

SCAFFOLD: fill-in pending. Class and function signatures wired; bodies raise
NotImplementedError with explicit TODO(fill-in) notes.

Role:
  Fisher-max item selection under a JOINT (alpha, beta) GPCM rule using
  recovered IRT parameters. Three strategies compared:
    "fisher"  -- recovered params -> argmax I(theta_est) item selection
    "oracle"  -- true params -> argmax I(theta_true) item selection (ceiling)
    "random"  -- random item selection (floor)
  Primary metric: ability RMSE at each fixed test length, averaged over learners.

  Blueprint: E8 CAT.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from deep_irt.bench.datagen import BenchDataset, _gpcm_probs


# ---------------------------------------------------------------------------
# Item information function
# ---------------------------------------------------------------------------

def gpcm_fisher_information(theta: float, a: float, b: np.ndarray) -> float:
    """Fisher information I(theta) for one GPCM item at ability theta.

    I(theta) = a^2 * Var[X | theta]
             = a^2 * sum_k [ k^2 * P_k - (sum_k k * P_k)^2 ]

    This is the ordinal GPCM information; it is the correct weighting for
    selecting items to maximise information about the latent trait.

    TODO(fill-in): call _gpcm_probs(theta, a, b) (imported from datagen),
      compute E[X] and E[X^2], return a**2 * (E[X^2] - E[X]**2).
    """
    raise NotImplementedError("gpcm_fisher_information not yet implemented -- fill-in pending")


# ---------------------------------------------------------------------------
# Item selection
# ---------------------------------------------------------------------------

def select_max_fisher_item(
    theta_est: float,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    administered: List[int],
) -> int:
    """Return the item with highest Fisher information at theta_est.

    Items in `administered` are excluded. Ties broken by item index.

    TODO(fill-in): compute gpcm_fisher_information(theta_est, a_hat[q], b_hat[q])
      for all q not in administered; return argmax.
    TODO(fill-in): handle edge case where all items have been administered
      (should not happen in a well-sized item bank).
    """
    raise NotImplementedError("select_max_fisher_item not yet implemented -- fill-in pending")


# ---------------------------------------------------------------------------
# MLE theta update
# ---------------------------------------------------------------------------

def update_theta_mle(
    theta_prev: float,
    responses_so_far: List[int],
    items_so_far: List[int],
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    bounds: tuple = (-4.0, 4.0),
) -> float:
    """MLE estimate of theta after observing responses_so_far.

    Uses scipy.optimize.minimize_scalar on the negative log-likelihood over
    administered (item, response) pairs. Bounded to `bounds` to avoid runaway
    estimates at the start of testing.

    TODO(fill-in): implement as sum of log P(r_t | theta, a_j, b_j) over
      administered items; use scipy.optimize.minimize_scalar with method="bounded".
    TODO(fill-in): use theta_prev as a warm-start if minimise_scalar supports it
      (it does not directly; use a dense grid search as fallback if needed).
    TODO(fill-in): return theta_prev unchanged if fewer than 2 responses have
      been given (insufficient information for stable MLE).
    """
    raise NotImplementedError("update_theta_mle not yet implemented -- fill-in pending")


# ---------------------------------------------------------------------------
# CAT simulator
# ---------------------------------------------------------------------------

class CATSimulator:
    """Simulate a fixed-length adaptive test for one learner.

    Usage:
        sim = CATSimulator(a_hat, b_hat, true_theta)
        results = sim.run(test_length=20, strategy="fisher")

    The simulator draws simulated responses from the TRUE GPCM at true_theta
    (using _gpcm_probs), then updates its theta estimate and selects the next
    item. This is the standard simulated CAT procedure.

    TODO(fill-in): implement run() for all three strategy branches.
    """

    def __init__(
        self,
        a_hat: np.ndarray,
        b_hat: np.ndarray,
        true_theta: float,
        a_true: Optional[np.ndarray] = None,
        b_true: Optional[np.ndarray] = None,
        theta_init: float = 0.0,
        seed: int = 0,
    ):
        self.a_hat = a_hat
        self.b_hat = b_hat
        self.true_theta = true_theta
        # Oracle uses true params for item selection; recovered used otherwise.
        self.a_true = a_true if a_true is not None else a_hat
        self.b_true = b_true if b_true is not None else b_hat
        self.theta_est = theta_init
        self.rng = np.random.default_rng(seed)

    def run(self, test_length: int = 20, strategy: str = "fisher") -> Dict:
        """Simulate the adaptive test and return RMSE trajectory.

        Args:
            test_length: number of items to administer.
            strategy: "fisher" | "oracle" | "random".

        Returns:
            Dict with:
              "theta_est_trajectory": list of theta_est after each item (length test_length).
              "rmse_trajectory": list of |theta_est - true_theta| after each item.
              "items_administered": list of item indices in order.
              "responses": simulated responses.

        TODO(fill-in): implement the adaptive loop:
          1. administered = [], responses = []
          2. For t in range(test_length):
               a. Select item: fisher -> select_max_fisher_item(theta_est, a_hat, b_hat, administered)
                                oracle -> select_max_fisher_item(theta_est, a_true, b_true, administered)
                                random -> rng.choice(unseen_items)
               b. Simulate response: r = rng.choice(K, p=_gpcm_probs(true_theta, a_true[q], b_true[q]))
               c. Append (q, r) to administered / responses.
               d. Update theta_est: update_theta_mle(theta_est, responses, administered, a_hat, b_hat)
               e. Record theta_est and RMSE.
        """
        raise NotImplementedError(
            f"CATSimulator.run() not yet implemented (strategy={strategy!r}) -- fill-in pending"
        )


# ---------------------------------------------------------------------------
# Experiment-level runner
# ---------------------------------------------------------------------------

def run_cat_experiment(
    ds: BenchDataset,
    recovered: Dict,
    test_length: int = 20,
    n_learners: Optional[int] = None,
    seed: int = 0,
) -> Dict:
    """Run E8 CAT over a sample of held-out learners and aggregate RMSE.

    Args:
        ds: BenchDataset with ground truth a, b, theta0.
        recovered: dict from engine.recover() with keys "a" (Q,) and "b" (Q,K-1).
        test_length: number of items per simulated test.
        n_learners: how many val learners to simulate (None = all val learners).
        seed: RNG seed for response simulation.

    Returns:
        Dict with per-strategy RMSE at each test length step (mean over learners),
        plus final-step summary:
          {"fisher_rmse_curve": [...], "oracle_rmse_curve": [...],
           "random_rmse_curve": [...],
           "fisher_final_rmse": float, "oracle_final_rmse": float,
           "random_final_rmse": float,
           "relative_efficiency": fisher_rmse / oracle_rmse at final step}

    TODO(fill-in): select n_learners from ds.val_idx (or a random subset).
    TODO(fill-in): loop over learners, run CATSimulator per strategy,
      stack RMSE trajectories, compute mean across learners.
    TODO(fill-in): report relative efficiency = fisher_rmse / oracle_rmse
      at final step (should be < 1 if recovered params are useful, > 1 if worse
      than oracle, likely 1 - epsilon for good recovery).
    """
    raise NotImplementedError("run_cat_experiment not yet implemented -- fill-in pending")
