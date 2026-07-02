"""model.py -- Shared ma-irt encoder + 2PL head + item-only NRM head.

The unifying mechanism is a SINGLE shared item difficulty d_i, read by BOTH a
2PL (right/wrong) head and a Bock-NRM (chosen-option) head, on top of ONE shared
ma-irt LSTM encoder (separate_theta=True, so theta is read from the item-blind
student_summary).

This is the ENCODER / sequential analog of deep_irt/nrmfmt (which used static
per-respondent theta parameters).  Here theta is a per-STEP function of the
learner's response history, produced by the frozen-design ma-irt encoder.

Shared scale
------------
A shared ``DifficultyExtractor`` maps the encoder's item embedding q_embed(item)
to a scalar d_i.  Both heads pin d_i:

    2PL head : P(correct) = sigmoid(a_i * (theta - d_i)).  Higher d_i = harder.
    NRM head : the CORRECT option's location IS d_i (distractors free, item-only
               per-option slopes a_k and intercepts c_k, Bock sum-to-zero).
               correct option is index 0 by construction (see data.py).

Because the correct option's NRM location is forced to equal d_i, the two heads
literally share the difficulty.  The per-option NRM slopes are ITEM-ONLY (a pure
function of q_embed), the decided best-of-both NRM head from the bench
(project_maitrt_nrm_itemonly): item recovery as clean as deep_irt while keeping
ma-irt's item-blind theta.

ma-irt is FROZEN.  This module imports the encoder unchanged and bolts on heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.encoders.lstm import LSTMEncoder


# ---------------------------------------------------------------------------
# Shared difficulty extractor (same design as nrmfmt / jointfmt)
# ---------------------------------------------------------------------------

class DifficultyExtractor(nn.Module):
    """q_embed -> scalar difficulty d_i (small MLP, unconstrained sign)."""

    def __init__(self, emb_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb: Tensor) -> Tensor:
        return self.net(emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class Binary2PLHead(nn.Module):
    """P(correct) = sigmoid(a_i * (theta - d_i)); a_i item-only from q_embed."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.fc_a = nn.Linear(emb_dim, 1)

    def logit(self, theta: Tensor, d: Tensor, emb: Tensor) -> Tensor:
        """theta (B,S), d (B,S), emb (B,S,emb_dim) -> logit (B,S)."""
        a = F.softplus(self.fc_a(emb)).squeeze(-1)        # (B,S) > 0
        return a * (theta - d)


class NRMItemOnlyHead(nn.Module):
    """Bock NRM, item-only slopes/intercepts, correct option location == d_i.

    Correct option (index 0) predictor : a_corr_i * (theta - d_i)
    Distractor option k                 : a_k_i * theta + c_k_i  (item-only)

    All K predictors mean-centered (sum-to-zero) before softmax; centering is
    likelihood-neutral.  a_k / c_k are pure functions of q_embed (item-only),
    matching the decided best-of-both NRM head.
    """

    def __init__(self, emb_dim: int, K: int, correct_option: int = 0) -> None:
        super().__init__()
        self.K = K
        self.correct_option = correct_option
        self.fc_alpha_corr = nn.Linear(emb_dim, 1)   # correct-option slope > 0
        self.fc_alpha = nn.Linear(emb_dim, K)        # raw per-option slopes
        self.fc_gamma = nn.Linear(emb_dim, K)        # raw per-option intercepts

    def logits(self, theta: Tensor, d: Tensor, emb: Tensor) -> Tensor:
        """theta (B,S), d (B,S), emb (B,S,emb_dim) -> logits (B,S,K)."""
        alpha_corr = F.softplus(self.fc_alpha_corr(emb)).squeeze(-1)   # (B,S) > 0
        alpha = self.fc_alpha(emb)                                     # (B,S,K)
        gamma = self.fc_gamma(emb)                                     # (B,S,K)

        co = self.correct_option
        alpha = alpha.clone()
        gamma = gamma.clone()
        alpha[..., co] = alpha_corr
        gamma[..., co] = -alpha_corr * d

        logits = alpha * theta.unsqueeze(-1) + gamma                  # (B,S,K)
        logits = logits - logits.mean(dim=-1, keepdim=True)           # sum-to-zero
        return logits


# ---------------------------------------------------------------------------
# Encoder wrapper (shared, separate_theta ability pathway)
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Wraps the ma-irt LSTMEncoder + an item-blind ability projection.

    theta_t is read from the item-blind ``student_summary`` (ma-irt's separated
    ability pathway), projected to a scalar with shift-by-one so theta at step t
    is the ability BEFORE answering step t.
    """

    def __init__(self, n_questions: int, K: int, d_model: int, key_dim: int,
                 value_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(
            n_questions=n_questions,
            n_categories=K,
            d_model=d_model,
            n_layers=1,
            dropout_rate=dropout,
            key_dim=key_dim,
            value_dim=value_dim,
            embedding_type="learned",
            separate_theta=True,
        )
        self.ability_net = nn.Linear(d_model, 1)
        self.key_dim = key_dim

    def forward(self, questions: Tensor, value_responses: Tensor):
        """Run encoder; return (theta (B,S), q_embed (B,S,key_dim)).

        ``value_responses`` is the response stream fed to the value side of the
        encoder.  We use the NOMINAL option index here (0..K-1) so the encoder's
        history is the richest available signal; padded positions carry 0.
        """
        enc = self.encoder(questions, value_responses)
        ability = enc.student_summary                       # (B,S,d_model), item-blind
        theta = self.ability_net(ability).squeeze(-1)       # (B,S)
        # shift-by-one: theta_in[t] is ability BEFORE step t
        B = theta.size(0)
        zeros = torch.zeros(B, 1, device=theta.device, dtype=theta.dtype)
        theta_in = torch.cat([zeros, theta[:, :-1]], dim=1)  # (B,S)
        return theta_in, enc.item_embed                      # item_embed = q_embed


# ---------------------------------------------------------------------------
# Joint model: shared encoder + shared d_i + 2PL head + NRM head
# ---------------------------------------------------------------------------

class JointModel(nn.Module):
    """Shared ma-irt encoder, shared d_i, read by a 2PL head and an NRM head."""

    def __init__(self, n_questions: int, K: int = 4, d_model: int = 64,
                 key_dim: int = 64, value_dim: int = 64, correct_option: int = 0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.n_questions = n_questions
        self.K = K
        self.enc = _Encoder(n_questions, K, d_model, key_dim, value_dim, dropout)
        self.difficulty_extractor = DifficultyExtractor(key_dim, hidden_dim=8)
        self.bin_head = Binary2PLHead(key_dim)
        self.nrm_head = NRMItemOnlyHead(key_dim, K, correct_option)

    def forward(self, questions: Tensor, value_responses: Tensor):
        theta, q_emb = self.enc(questions, value_responses)   # (B,S), (B,S,key_dim)
        d = self.difficulty_extractor(q_emb)                  # (B,S)
        return theta, q_emb, d

    def losses(self, questions, value_responses, binary, nominal,
               bin_mask, nrm_mask):
        """Masked binary + NRM NLL.

        questions       : (B,S) long, 1-based item ids (0 = pad)
        value_responses : (B,S) long, nominal option index fed to encoder
        binary          : (B,S) long/float, 0/1 correctness
        nominal         : (B,S) long, 0..K-1 chosen option
        bin_mask        : (B,S) bool, binary-visible AND non-pad
        nrm_mask        : (B,S) bool, nrm-visible AND non-pad
        """
        theta, q_emb, d = self.forward(questions, value_responses)

        bin_logit = self.bin_head.logit(theta, d, q_emb)      # (B,S)
        bce = F.binary_cross_entropy_with_logits(
            bin_logit, binary.float(), reduction="none")      # (B,S)
        bm = bin_mask.float()
        bin_nll = (bce * bm).sum() / bm.sum().clamp(min=1)

        nrm_logits = self.nrm_head.logits(theta, d, q_emb)    # (B,S,K)
        B, S, K = nrm_logits.shape
        ce = F.cross_entropy(nrm_logits.reshape(B * S, K),
                             nominal.reshape(B * S), reduction="none").reshape(B, S)
        nm = nrm_mask.float()
        nrm_nll = (ce * nm).sum() / nm.sum().clamp(min=1)
        return bin_nll, nrm_nll

    @torch.no_grad()
    def item_difficulty(self) -> Tensor:
        """Shared d_i for every item, mean-centered. (n_questions,)."""
        idx = torch.arange(1, self.n_questions + 1,
                           device=self.difficulty_extractor.net[0].weight.device)
        q_emb = self.enc.encoder.q_embed(idx)                 # (n_questions, key_dim)
        d = self.difficulty_extractor(q_emb)
        return d - d.mean()


# ---------------------------------------------------------------------------
# Independent single-format baselines (separate encoder + embedding)
# ---------------------------------------------------------------------------

class IndepBinary(nn.Module):
    """ma-irt encoder + 2PL head only.  NRM-only items never trained -> noise."""

    def __init__(self, n_questions, K=4, d_model=64, key_dim=64, value_dim=64,
                 dropout=0.0):
        super().__init__()
        self.n_questions = n_questions
        self.enc = _Encoder(n_questions, K, d_model, key_dim, value_dim, dropout)
        self.difficulty_extractor = DifficultyExtractor(key_dim, hidden_dim=8)
        self.bin_head = Binary2PLHead(key_dim)

    def loss(self, questions, value_responses, binary, bin_mask):
        theta, q_emb = self.enc(questions, value_responses)
        d = self.difficulty_extractor(q_emb)
        bin_logit = self.bin_head.logit(theta, d, q_emb)
        bce = F.binary_cross_entropy_with_logits(
            bin_logit, binary.float(), reduction="none")
        bm = bin_mask.float()
        return (bce * bm).sum() / bm.sum().clamp(min=1)

    @torch.no_grad()
    def item_difficulty(self):
        idx = torch.arange(1, self.n_questions + 1,
                           device=self.difficulty_extractor.net[0].weight.device)
        q_emb = self.enc.encoder.q_embed(idx)
        d = self.difficulty_extractor(q_emb)
        return d - d.mean()


class IndepNRM(nn.Module):
    """ma-irt encoder + NRM head only.  Binary-only items never trained -> noise."""

    def __init__(self, n_questions, K=4, d_model=64, key_dim=64, value_dim=64,
                 correct_option=0, dropout=0.0):
        super().__init__()
        self.n_questions = n_questions
        self.K = K
        self.enc = _Encoder(n_questions, K, d_model, key_dim, value_dim, dropout)
        self.difficulty_extractor = DifficultyExtractor(key_dim, hidden_dim=8)
        self.nrm_head = NRMItemOnlyHead(key_dim, K, correct_option)

    def loss(self, questions, value_responses, nominal, nrm_mask):
        theta, q_emb = self.enc(questions, value_responses)
        d = self.difficulty_extractor(q_emb)
        nrm_logits = self.nrm_head.logits(theta, d, q_emb)
        B, S, K = nrm_logits.shape
        ce = F.cross_entropy(nrm_logits.reshape(B * S, K),
                             nominal.reshape(B * S), reduction="none").reshape(B, S)
        nm = nrm_mask.float()
        return (ce * nm).sum() / nm.sum().clamp(min=1)

    @torch.no_grad()
    def item_difficulty(self):
        idx = torch.arange(1, self.n_questions + 1,
                           device=self.difficulty_extractor.net[0].weight.device)
        q_emb = self.enc.encoder.q_embed(idx)
        d = self.difficulty_extractor(q_emb)
        return d - d.mean()
