"""
Mode B recovery: fit a Bradley-Terry model to pairwise comparison data.

The BT model assigns a strength score s_i to each item.
P(i beats j) = sigmoid(s_i - s_j).

Identifiability: the overall mean is not identified from pairwise differences alone.
We fix item 0 as the reference by anchoring its strength to 0 via a strong prior,
or equivalently by subtracting the mean at the end. In practice we use a small
L2 regulariser (which soft-anchors the mean around 0) and post-hoc mean-center.

Fit via gradient descent on BT NLL (binary cross-entropy on pair outcomes).
"""

import torch
import torch.nn.functional as F


class BradleyTerryModel(torch.nn.Module):
    """Per-item strength parameters for Bradley-Terry."""

    def __init__(self, n_items: int, seed: int = 0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        # Initialise near zero
        self.strength = torch.nn.Parameter(torch.randn(n_items, generator=g) * 0.01)
        self.n_items = n_items

    def forward(
        self,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-probability that item i beats item j.

        Returns
        -------
        log_prob_i_wins : (N,)
        """
        delta = self.strength[item_i] - self.strength[item_j]  # (N,)
        return F.logsigmoid(delta)

    def nll(
        self,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        outcome: torch.Tensor,
        reg_strength: float = 0.01,
    ) -> torch.Tensor:
        """
        Binary cross-entropy NLL.

        outcome == 1: item i beats j  (log prob = log sigmoid(s_i - s_j))
        outcome == 0: item j beats i  (log prob = log sigmoid(s_j - s_i))
        """
        delta = self.strength[item_i] - self.strength[item_j]  # (N,)
        # BCE: -[y * log sigma(delta) + (1-y) * log sigma(-delta)]
        bce = F.binary_cross_entropy_with_logits(delta, outcome.float(), reduction="mean")
        reg = reg_strength * self.strength.pow(2).mean()
        return bce + reg


def fit_bt(
    mode_b_data: dict,
    ground_truth: dict,
    n_epochs: int = 500,
    lr: float = 0.05,
    batch_size: int = 8192,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Fit Bradley-Terry model to Mode-B pairwise comparisons.

    Returns
    -------
    dict with:
        strength : (n_items,) recovered per-item strength (mean-centered)
        final_nll : float
    """
    torch.manual_seed(seed)

    n_items = ground_truth["n_items"]
    item_i = mode_b_data["item_i"]
    item_j = mode_b_data["item_j"]
    outcome = mode_b_data["outcome"]

    model = BradleyTerryModel(n_items=n_items, seed=seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    N = len(outcome)
    best_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()

        if batch_size >= N:
            loss = model.nll(item_i, item_j, outcome)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            epoch_loss = loss.item()
        else:
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                loss = model.nll(item_i[idx], item_j[idx], outcome[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            epoch_loss /= n_batches

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  [BT]   Epoch {epoch+1:>3d}/{n_epochs}  BCE={epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        strength = model.strength.detach().clone()
        # Mean-center for identifiability (soft anchor -> post-hoc correction)
        strength = strength - strength.mean()

    return {
        "strength": strength,
        "final_nll": best_loss,
    }
