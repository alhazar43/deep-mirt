"""model.py -- Shared ma-irt encoder + 2PL + item-only NRM + log-normal RT heads.

The unifying mechanism is a SINGLE shared item location d_i, read by THREE heads
on top of ONE shared ma-irt LSTM encoder (separate_theta=True).  This extends
``deep_irt/rq1_ednet_nrm/model.py`` (2PL + NRM) with a third, genuinely
independent observable: response time, via van der Linden's (2007) log-normal
speed model.

Shared scale
------------
A shared ``DifficultyExtractor`` maps the encoder's item embedding q_embed(item)
to a scalar location d_i.  All three heads pin d_i:

    2PL head : P(correct) = sigmoid(a_i * (theta - d_i)).  Higher d_i = harder.
    NRM head : the CORRECT option's location IS d_i (distractors free, item-only
               per-option slopes/intercepts, Bock sum-to-zero).  Higher d_i = harder.
    RT head  : log(RT) ~ Normal(tau + g * d_i + b, sigma).  d_i enters as the item
               TIME-INTENSITY through a single GLOBAL gain g and bias b.

Person latents: ability vs speed
--------------------------------
The 2PL and NRM heads read ABILITY theta from the item-blind student_summary
(ma-irt's separated pathway).  The RT head reads a SEPARATE person SPEED latent
tau from the SAME summary via its own projection: a fast person need not be a
high-ability person -- the speed-accuracy tradeoff is exactly why RT is an
independent observable, so reusing theta as speed would conflate the two.  What
is SHARED across the three formats is the ITEM scale d_i, not the person latent.

Why a free global gain g on d_i (not a hard equality)
-----------------------------------------------------
The 2PL/NRM heads force the item LOCATION to literally equal d_i because both
are difficulty-like on the same metric.  Time-intensity is NOT guaranteed to sit
on the same metric: harder items tend to take longer (positive link) but the
scale and even the sign are an empirical question.  The RT head therefore reads
d_i through ONE learnable scalar gain g and offset b -- it has NO free per-item
RT parameter, so the only item information the RT head can use is the shared d_i.
If RT and accuracy share a scale, training drives g > 0 and RT-only items'
d_i aligns with full-info difficulty.  If they do not, g -> 0, the RT head
ignores d_i, and RT-only items collapse to the noise floor.  This tests the
hypothesis honestly instead of forcing a PASS by hard-coupling.

ma-irt is FROZEN.  This module imports the encoder unchanged and bolts on heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.encoders.lstm import LSTMEncoder

_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Shared difficulty / location extractor
# ---------------------------------------------------------------------------

class DifficultyExtractor(nn.Module):
    """q_embed -> scalar location d_i (small MLP, unconstrained sign)."""

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
        a = F.softplus(self.fc_a(emb)).squeeze(-1)        # (B,S) > 0
        return a * (theta - d)


class NRMItemOnlyHead(nn.Module):
    """Bock NRM, item-only slopes/intercepts, correct-option location == d_i."""

    def __init__(self, emb_dim: int, K: int, correct_option: int = 0) -> None:
        super().__init__()
        self.K = K
        self.correct_option = correct_option
        self.fc_alpha_corr = nn.Linear(emb_dim, 1)
        self.fc_alpha = nn.Linear(emb_dim, K)
        self.fc_gamma = nn.Linear(emb_dim, K)

    def logits(self, theta: Tensor, d: Tensor, emb: Tensor) -> Tensor:
        alpha_corr = F.softplus(self.fc_alpha_corr(emb)).squeeze(-1)   # (B,S) > 0
        alpha = self.fc_alpha(emb).clone()                            # (B,S,K)
        gamma = self.fc_gamma(emb).clone()                            # (B,S,K)
        co = self.correct_option
        alpha[..., co] = alpha_corr
        gamma[..., co] = -alpha_corr * d
        logits = alpha * theta.unsqueeze(-1) + gamma                  # (B,S,K)
        logits = logits - logits.mean(dim=-1, keepdim=True)           # sum-to-zero
        return logits


class LogNormalRTHead(nn.Module):
    """log(RT) ~ Normal(tau + g*d_i + b, sigma).

    Reads its OWN person speed latent tau from the encoder summary.  The item's
    time-intensity is the shared location d_i passed through ONE global gain g and
    offset b; there is NO free per-item RT parameter, so the shared d_i is the
    only item information available to the RT head.  sigma is a single shared
    residual scale on log-time (log_sigma free parameter).
    """

    def __init__(self, summary_dim: int) -> None:
        super().__init__()
        self.speed_net = nn.Linear(summary_dim, 1)        # person speed tau
        self.gain = nn.Parameter(torch.tensor(0.0))       # global d_i gain g
        self.offset = nn.Parameter(torch.tensor(0.0))     # global offset b
        self.log_sigma = nn.Parameter(torch.zeros(1))     # shared residual scale

    def tau(self, summary: Tensor) -> Tensor:
        """Per-step speed latent (B,S) from the item-blind summary."""
        return self.speed_net(summary).squeeze(-1)

    def nll(self, tau: Tensor, d: Tensor, log_rt: Tensor,
            mask: Tensor) -> Tensor:
        """Masked mean NLL of observed standardized log-RT.

        tau    : (B,S)   person speed
        d      : (B,S)   shared item location
        log_rt : (B,S)   observed standardized log response time
        mask   : (B,S)   bool, rt-visible AND non-pad
        """
        mean = tau + self.gain * d + self.offset          # (B,S)
        sigma = self.log_sigma.exp()
        z = (log_rt - mean) / sigma
        ld = -0.5 * _LOG_2PI - self.log_sigma - 0.5 * z * z   # (B,S) log-density
        m = mask.float()
        return -(ld * m).sum() / m.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Encoder wrapper (shared, separate_theta ability pathway)
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """ma-irt LSTMEncoder + item-blind ability projection.

    Returns theta (ability, shift-by-one), the item-blind summary (for the RT
    head's own speed projection), and q_embed.
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
        self.d_model = d_model

    @staticmethod
    def _shift1(x: Tensor) -> Tensor:
        """Shift along time (dim 1) so value at t reflects state BEFORE step t.

        Rank-agnostic: works for (B,S) ability scalars and (B,S,D) summaries.
        """
        zeros = torch.zeros_like(x[:, :1])
        return torch.cat([zeros, x[:, :-1]], dim=1)

    def forward(self, questions: Tensor, value_responses: Tensor):
        """Return theta (B,S), summary_in (B,S,d_model), q_embed (B,S,key_dim).

        ``summary_in`` is the item-blind student_summary shifted by one so the RT
        head's speed projection at step t also reflects state BEFORE step t,
        matching the ability theta's shift.
        """
        enc = self.encoder(questions, value_responses)
        ability = enc.student_summary                       # (B,S,d_model)
        theta = self.ability_net(ability).squeeze(-1)       # (B,S)
        theta_in = self._shift1(theta)                      # (B,S)
        summary_in = self._shift1(ability)                  # (B,S,d_model)
        return theta_in, summary_in, enc.item_embed


# ---------------------------------------------------------------------------
# Joint model: shared encoder + shared d_i + three heads
# ---------------------------------------------------------------------------

class JointModel(nn.Module):
    """Shared ma-irt encoder, shared d_i, read by 2PL + NRM + RT heads."""

    def __init__(self, n_questions: int, K: int = 4, d_model: int = 64,
                 key_dim: int = 64, value_dim: int = 64, correct_option: int = 0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.n_questions = n_questions
        self.K = K
        self.enc = _Encoder(n_questions, K, d_model, key_dim, value_dim, dropout)
        self.difficulty_extractor = DifficultyExtractor(key_dim, hidden_dim=8)
        self.acc_head = Binary2PLHead(key_dim)
        self.cho_head = NRMItemOnlyHead(key_dim, K, correct_option)
        self.rt_head = LogNormalRTHead(d_model)

    def forward(self, questions: Tensor, value_responses: Tensor):
        theta, summary, q_emb = self.enc(questions, value_responses)
        d = self.difficulty_extractor(q_emb)                 # (B,S)
        return theta, summary, q_emb, d

    def losses(self, questions, value_responses, accuracy, choice, log_rt,
               acc_mask, cho_mask, rt_mask):
        """Masked accuracy + choice + RT NLL (each at its visible positions)."""
        theta, summary, q_emb, d = self.forward(questions, value_responses)

        acc_logit = self.acc_head.logit(theta, d, q_emb)
        bce = F.binary_cross_entropy_with_logits(
            acc_logit, accuracy.float(), reduction="none")
        am = acc_mask.float()
        acc_nll = (bce * am).sum() / am.sum().clamp(min=1)

        cho_logits = self.cho_head.logits(theta, d, q_emb)
        B, S, K = cho_logits.shape
        ce = F.cross_entropy(cho_logits.reshape(B * S, K),
                             choice.reshape(B * S), reduction="none").reshape(B, S)
        cm = cho_mask.float()
        cho_nll = (ce * cm).sum() / cm.sum().clamp(min=1)

        tau = self.rt_head.tau(summary)
        rt_nll = self.rt_head.nll(tau, d, log_rt, rt_mask)
        return acc_nll, cho_nll, rt_nll

    @torch.no_grad()
    def item_location(self) -> Tensor:
        """Shared d_i for every item, mean-centered. (n_questions,)."""
        idx = torch.arange(1, self.n_questions + 1,
                           device=self.difficulty_extractor.net[0].weight.device)
        q_emb = self.enc.encoder.q_embed(idx)
        d = self.difficulty_extractor(q_emb)
        return d - d.mean()


# ---------------------------------------------------------------------------
# Independent single-format baselines (separate encoder + embedding)
# ---------------------------------------------------------------------------

class IndepAccuracy(nn.Module):
    """ma-irt encoder + 2PL head only.  Other-format items never trained -> noise."""

    def __init__(self, n_questions, K=4, d_model=64, key_dim=64, value_dim=64,
                 dropout=0.0):
        super().__init__()
        self.n_questions = n_questions
        self.enc = _Encoder(n_questions, K, d_model, key_dim, value_dim, dropout)
        self.difficulty_extractor = DifficultyExtractor(key_dim, hidden_dim=8)
        self.acc_head = Binary2PLHead(key_dim)

    def loss(self, questions, value_responses, accuracy, acc_mask):
        theta, _summary, q_emb = self.enc(questions, value_responses)
        d = self.difficulty_extractor(q_emb)
        logit = self.acc_head.logit(theta, d, q_emb)
        bce = F.binary_cross_entropy_with_logits(
            logit, accuracy.float(), reduction="none")
        m = acc_mask.float()
        return (bce * m).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def item_location(self):
        idx = torch.arange(1, self.n_questions + 1,
                           device=self.difficulty_extractor.net[0].weight.device)
        q_emb = self.enc.encoder.q_embed(idx)
        d = self.difficulty_extractor(q_emb)
        return d - d.mean()


class IndepChoice(nn.Module):
    """ma-irt encoder + NRM head only."""

    def __init__(self, n_questions, K=4, d_model=64, key_dim=64, value_dim=64,
                 correct_option=0, dropout=0.0):
        super().__init__()
        self.n_questions = n_questions
        self.K = K
        self.enc = _Encoder(n_questions, K, d_model, key_dim, value_dim, dropout)
        self.difficulty_extractor = DifficultyExtractor(key_dim, hidden_dim=8)
        self.cho_head = NRMItemOnlyHead(key_dim, K, correct_option)

    def loss(self, questions, value_responses, choice, cho_mask):
        theta, _summary, q_emb = self.enc(questions, value_responses)
        d = self.difficulty_extractor(q_emb)
        logits = self.cho_head.logits(theta, d, q_emb)
        B, S, K = logits.shape
        ce = F.cross_entropy(logits.reshape(B * S, K),
                             choice.reshape(B * S), reduction="none").reshape(B, S)
        m = cho_mask.float()
        return (ce * m).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def item_location(self):
        idx = torch.arange(1, self.n_questions + 1,
                           device=self.difficulty_extractor.net[0].weight.device)
        q_emb = self.enc.encoder.q_embed(idx)
        d = self.difficulty_extractor(q_emb)
        return d - d.mean()


class IndepRT(nn.Module):
    """ma-irt encoder + RT head only.

    NOTE: the joint RT head reads d_i only through a global gain g, so a separate
    RT model whose item location is ONLY the shared-extractor d_i cannot place
    held-out items either -- it would have the SAME single-gain bottleneck.  For
    a fair noise floor the independent RT baseline is given a FREE per-item
    time-intensity (its own real item parameter), i.e. the best a pure RT model
    can do.  Its recovered per-item time-intensity is then the RT-format estimate;
    items never seen by any other head still get placed by RT alone.  The point
    of the noise floor is: an item this baseline NEVER trained on (because it is
    accuracy-only or choice-only) lands at noise on the RT scale.
    """

    def __init__(self, n_questions, K=4, d_model=64, key_dim=64, value_dim=64,
                 dropout=0.0):
        super().__init__()
        self.n_questions = n_questions
        self.enc = _Encoder(n_questions, K, d_model, key_dim, value_dim, dropout)
        # Free per-item time-intensity (own real RT item parameter).
        self.time_intensity = nn.Linear(key_dim, 1)
        self.speed_net = nn.Linear(d_model, 1)
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def _beta(self, q_emb):
        return self.time_intensity(q_emb).squeeze(-1)         # (B,S)

    def loss(self, questions, value_responses, log_rt, rt_mask):
        _theta, summary, q_emb = self.enc(questions, value_responses)
        tau = self.speed_net(summary).squeeze(-1)
        beta = self._beta(q_emb)
        mean = tau + beta
        sigma = self.log_sigma.exp()
        z = (log_rt - mean) / sigma
        ld = -0.5 * _LOG_2PI - self.log_sigma - 0.5 * z * z
        m = rt_mask.float()
        return -(ld * m).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def item_location(self):
        idx = torch.arange(1, self.n_questions + 1,
                           device=self.time_intensity.weight.device)
        q_emb = self.enc.encoder.q_embed(idx)
        beta = self.time_intensity(q_emb).squeeze(-1)
        return beta - beta.mean()
