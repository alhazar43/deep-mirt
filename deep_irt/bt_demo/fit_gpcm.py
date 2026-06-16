"""
Mode A recovery: fit a GPCM model to direct ordinal responses.

Parameters estimated (jointly, via gradient descent on GPCM NLL):
    - Per-item difficulty d_i  (scalar; difficulty centroid, analogous to s_i)
    - Per-respondent ability theta_p (scalar)
    - Per-item discrimination a_i (via log_a_i, softplus-constrained > 0)
    - Per-item step offsets delta_i (K-2 values; betas_ik = d_i + delta_ik)

Identifiability anchor: mean(theta) = 0, mean(d) = 0 enforced softly via
L2 regularisation, which is standard for GPCM. The scale is fixed by
initialising theta ~ N(0,1) and d ~ N(0,1), and the regulariser keeps
them there on average.

The key output is d_i (per-item difficulty), which should be monotonically
related to s_i (ground truth strength). Higher d_i = harder item.
"""

import torch
import torch.nn.functional as F


class GPCMModel(torch.nn.Module):
    """Jointly optimised GPCM parameters."""

    def __init__(self, n_items: int, n_respondents: int, K: int, seed: int = 0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)

        # Per-item difficulty (centroid of thresholds)
        self.log_difficulty = torch.nn.Parameter(torch.randn(n_items, generator=g) * 0.1)
        # Per-item log-discrimination (exp -> positive)
        self.log_a = torch.nn.Parameter(torch.zeros(n_items))
        # Per-item step offsets: K-2 values (first offset fixed at 0 for identifiability)
        # Full offsets: [0, delta_1, delta_2] so we have K-2 free params
        self.raw_offsets = torch.nn.Parameter(torch.zeros(n_items, K - 2))
        # Per-respondent ability
        self.theta = torch.nn.Parameter(torch.randn(n_respondents, generator=g) * 0.1)

        self.K = K
        self.n_items = n_items
        self.n_respondents = n_respondents

    def get_betas(self) -> torch.Tensor:
        """Compute (n_items, K-1) step thresholds from difficulty + offsets."""
        d = self.log_difficulty                                    # (n_items,)
        # Offsets: [0, raw_offsets columns]
        zero_col = torch.zeros(self.n_items, 1, device=d.device)
        offsets = torch.cat([zero_col, self.raw_offsets], dim=1)  # (n_items, K-2)
        # Actually K-1 thresholds: difficulty + cumulative offsets
        # beta_k = d_i + offset_k, offset_0 = 0
        # We need K-1 thresholds. Let offset_k be free for k=0..K-2
        # So offsets shape must be (n_items, K-1).
        # Fix: use raw_offsets as (n_items, K-2), prepend a fixed 0, giving K-1 thresholds.
        # That means: beta_1 = d, beta_2 = d + raw_offset_0, beta_3 = d + raw_offset_1
        betas = d.unsqueeze(1) + offsets  # (n_items, K-1)
        return betas

    def forward(
        self,
        respondent_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for all observations.

        Parameters
        ----------
        respondent_ids : (N,)
        item_ids       : (N,)

        Returns
        -------
        log_probs : (N, K)
        """
        theta = self.theta[respondent_ids]         # (N,)
        a = F.softplus(self.log_a[item_ids])       # (N,)  always positive
        betas = self.get_betas()[item_ids]          # (N, K-1)

        step_logits = a.unsqueeze(-1) * (theta.unsqueeze(-1) - betas)  # (N, K-1)
        cumsum = torch.cumsum(step_logits, dim=-1)                       # (N, K-1)
        log_num = torch.cat(
            [torch.zeros_like(cumsum[..., :1]), cumsum], dim=-1
        )  # (N, K)

        log_probs = log_num - torch.logsumexp(log_num, dim=-1, keepdim=True)
        return log_probs

    def nll(
        self,
        respondent_ids: torch.Tensor,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        reg_strength: float = 0.01,
    ) -> torch.Tensor:
        """Negative log-likelihood + weak L2 regulariser on theta and d."""
        log_probs = self.forward(respondent_ids, item_ids)  # (N, K)
        nll = F.nll_loss(log_probs, responses)

        # L2 regulariser: pulls mean toward 0 (soft anchor)
        reg = reg_strength * (
            self.theta.pow(2).mean() + self.log_difficulty.pow(2).mean()
        )
        return nll + reg


def fit_gpcm(
    mode_a_data: dict,
    ground_truth: dict,
    n_epochs: int = 300,
    lr: float = 0.05,
    batch_size: int = 8192,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Fit GPCM to Mode-A observations.

    Returns
    -------
    dict with:
        difficulty : (n_items,) recovered per-item difficulty (higher = harder)
        a_recovered : (n_items,) recovered discrimination
        theta_recovered : (n_respondents,) recovered ability
    """
    torch.manual_seed(seed)

    n_items = ground_truth["n_items"]
    K = ground_truth["K"]
    n_respondents = mode_a_data["n_respondents"]

    respondent_ids = mode_a_data["respondent_ids"]
    item_ids = mode_a_data["item_ids"]
    responses = mode_a_data["responses"]

    model = GPCMModel(n_items=n_items, n_respondents=n_respondents, K=K, seed=seed)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    N = len(responses)
    best_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()

        if batch_size >= N:
            # Full batch
            loss = model.nll(respondent_ids, item_ids, responses)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            epoch_loss = loss.item()
        else:
            # Mini-batch
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                loss = model.nll(respondent_ids[idx], item_ids[idx], responses[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            epoch_loss /= n_batches

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  [GPCM] Epoch {epoch+1:>3d}/{n_epochs}  NLL={epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        difficulty = model.log_difficulty.detach().clone()
        a_recovered = F.softplus(model.log_a).detach().clone()
        theta_recovered = model.theta.detach().clone()

    return {
        "difficulty": difficulty,
        "a_recovered": a_recovered,
        "theta_recovered": theta_recovered,
        "final_nll": best_loss,
    }
