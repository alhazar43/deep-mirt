"""
models.py -- Shared-embedding joint model + independent baselines for the
binary-vs-NRM cross-format experiment.

The shared scalar d_i (DifficultyExtractor output) is the ONE scale both
formats read and write:

  Binary2PLHead : P(correct) = sigmoid(a_i * (theta - d_i)).  Higher d_i = harder.
  NRMHead       : the CORRECT option's location is d_i.  Its predictor is
                  alpha_corr_i * (theta - d_i); distractor options get free
                  per-option (alpha_k, gamma_k) from the embedding.  The chosen
                  option is softmax over all K predictors.

Because the correct option's NRM location is forced to equal d_i, the NRM head
and the 2PL head literally share the difficulty parameter.  A pure 2PL would
only see right/wrong; the NRM head additionally fits the distractor pattern, but
both pin the SAME d_i, so a binary-only item can be placed on the NRM scale and
a nrm-only item on the binary scale.

Sign convention is consistent (higher d_i = harder for both heads), so there is
no sign-flip across the shared extractor (unlike the GPCM-vs-BT pair, which
needed a reframe -- see memory project_rq1_essay_jointfmt).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared difficulty extractor (same design as jointfmt)
# ---------------------------------------------------------------------------

class DifficultyExtractor(nn.Module):
    """emb -> scalar difficulty d_i (small MLP, unconstrained sign)."""

    def __init__(self, emb_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Binary 2PL head (reads shared d_i)
# ---------------------------------------------------------------------------

class Binary2PLHead(nn.Module):
    """P(correct) = sigmoid(a_i * (theta - d_i)); a_i from embedding."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.fc_a = nn.Linear(emb_dim, 1)

    def nll(self, theta, d, emb, correct):
        a = F.softplus(self.fc_a(emb)).squeeze(-1)        # (N,) >0
        logit = a * (theta - d)
        return F.binary_cross_entropy_with_logits(
            logit, correct.float(), reduction="mean"
        )


# ---------------------------------------------------------------------------
# NRM head (reads shared d_i as the correct-option location)
# ---------------------------------------------------------------------------

class NRMHead(nn.Module):
    """
    Bock NRM where the CORRECT option's location is the shared d_i.

    Correct option predictor : alpha_corr_i * (theta - d_i)
    Distractor option k       : alpha_k_i * theta + gamma_k_i  (free from emb)

    All K predictors are mean-centered (sum-to-zero) before the softmax for
    identifiability; centering is likelihood-neutral.
    """

    def __init__(self, emb_dim: int, K: int, correct_option: int) -> None:
        super().__init__()
        self.K = K
        self.correct_option = correct_option
        self.fc_alpha_corr = nn.Linear(emb_dim, 1)        # correct-option slope >0
        self.fc_alpha = nn.Linear(emb_dim, K)             # raw per-option slopes
        self.fc_gamma = nn.Linear(emb_dim, K)             # raw per-option intercepts

    def _logits(self, theta, d, emb):
        N = emb.size(0)
        alpha_corr = F.softplus(self.fc_alpha_corr(emb)).squeeze(-1)   # (N,) >0
        alpha = self.fc_alpha(emb)                                     # (N,K)
        gamma = self.fc_gamma(emb)                                     # (N,K)

        # Overwrite the correct option's slope/intercept with the shared-d form:
        #   predictor_corr = alpha_corr * (theta - d) = alpha_corr*theta - alpha_corr*d
        co = self.correct_option
        alpha = alpha.clone()
        gamma = gamma.clone()
        alpha[:, co] = alpha_corr
        gamma[:, co] = -alpha_corr * d

        logits = alpha * theta.unsqueeze(-1) + gamma                  # (N,K)
        logits = logits - logits.mean(dim=-1, keepdim=True)           # sum-to-zero
        return logits

    def nll(self, theta, d, emb, chosen):
        logits = self._logits(theta, d, emb)
        log_p = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_p, chosen)


# ---------------------------------------------------------------------------
# Joint model
# ---------------------------------------------------------------------------

class JointModel(nn.Module):
    """Shared embedding + shared d_i, read by a 2PL head and an NRM head."""

    def __init__(self, n_items, n_resp_a, n_resp_b, emb_dim=16, K=4,
                 correct_option=0, seed=0):
        super().__init__()
        g = torch.Generator(); g.manual_seed(seed)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.difficulty_extractor = DifficultyExtractor(emb_dim, hidden_dim=8)
        self.theta_a = nn.Parameter(torch.randn(n_resp_a, generator=g) * 0.1)
        self.theta_b = nn.Parameter(torch.randn(n_resp_b, generator=g) * 0.1)
        self.bin_head = Binary2PLHead(emb_dim)
        self.nrm_head = NRMHead(emb_dim, K, correct_option)
        self.n_items = n_items

    def _d(self, item_ids):
        emb = self.item_emb(item_ids)
        return emb, self.difficulty_extractor(emb)

    def binary_nll(self, resp_ids, item_ids, correct, reg=0.01):
        theta = self.theta_a[resp_ids]
        emb, d = self._d(item_ids)
        nll = self.bin_head.nll(theta, d, emb, correct)
        return nll + reg * self.theta_a.pow(2).mean()

    def nrm_nll(self, resp_ids, item_ids, chosen, reg=0.01):
        theta = self.theta_b[resp_ids]
        emb, d = self._d(item_ids)
        nll = self.nrm_head.nll(theta, d, emb, chosen)
        return nll + reg * self.theta_b.pow(2).mean()

    @torch.no_grad()
    def get_item_difficulty(self):
        idx = torch.arange(self.n_items)
        d = self.difficulty_extractor(self.item_emb(idx))
        return d - d.mean()


# ---------------------------------------------------------------------------
# Independent baselines (separate embedding tables; no sharing)
# ---------------------------------------------------------------------------

class IndepBinary(nn.Module):
    """2PL with its own embedding.  NRM-only items never trained -> noise."""

    def __init__(self, n_items, n_resp, emb_dim=16, seed=0):
        super().__init__()
        g = torch.Generator(); g.manual_seed(seed + 10)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.difficulty_extractor = DifficultyExtractor(emb_dim, hidden_dim=8)
        self.theta = nn.Parameter(torch.randn(n_resp, generator=g) * 0.1)
        self.head = Binary2PLHead(emb_dim)
        self.n_items = n_items

    def nll(self, resp_ids, item_ids, correct, reg=0.01):
        theta = self.theta[resp_ids]
        emb = self.item_emb(item_ids)
        d = self.difficulty_extractor(emb)
        nll = self.head.nll(theta, d, emb, correct)
        return nll + reg * self.theta.pow(2).mean()

    @torch.no_grad()
    def get_difficulty(self):
        idx = torch.arange(self.n_items)
        d = self.difficulty_extractor(self.item_emb(idx))
        return d - d.mean()


class IndepNRM(nn.Module):
    """NRM with its own embedding.  Binary-only items never trained -> noise."""

    def __init__(self, n_items, n_resp, emb_dim=16, K=4, correct_option=0, seed=0):
        super().__init__()
        g = torch.Generator(); g.manual_seed(seed + 20)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.difficulty_extractor = DifficultyExtractor(emb_dim, hidden_dim=8)
        self.theta = nn.Parameter(torch.randn(n_resp, generator=g) * 0.1)
        self.head = NRMHead(emb_dim, K, correct_option)
        self.n_items = n_items

    def nll(self, resp_ids, item_ids, chosen, reg=0.01):
        theta = self.theta[resp_ids]
        emb = self.item_emb(item_ids)
        d = self.difficulty_extractor(emb)
        nll = self.head.nll(theta, d, emb, chosen)
        return nll + reg * self.theta.pow(2).mean()

    @torch.no_grad()
    def get_difficulty(self):
        idx = torch.arange(self.n_items)
        d = self.difficulty_extractor(self.item_emb(idx))
        return d - d.mean()
