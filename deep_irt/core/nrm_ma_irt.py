"""nrm_ma_irt.py -- ma-irt encoder + a Bock-NRM head (the bench's ma-irt-NRM).

ma-irt is FROZEN.  This module does NOT edit ma-irt; it WRAPS ma-irt's encoder
(``LSTMEncoder`` or ``DKVMNEncoder``) and bolts on a nominal-response-model head
built to mirror ma-irt's own GPCM head, so the head-to-head bench actually tests
ma-irt's design advantage on the NRM decoder.

Faithful mirror of ma-irt's GPCM head (``models/decoders/gpcm.py`` +
``models/components/irt.py``)
-----------------------------------------------------------------------------
ma-irt's GPCM head reads three signals from the encoder output:

    theta = ability_network(student_summary)            # item-BLIND ability
    alpha = exp(disc_net([joint_summary, item_embed]))  # STATE-CONDITIONED
    b     = threshold(item_embed)                        # item-only thresholds (GPCM)

The two structural tricks that are exactly ma-irt's edge:
  1. theta read from the item-BLIND ``student_summary`` (the separated ability
     pathway: encoder run twice with shared weights, ability stream sees no
     current-step item key).  This is why ma-irt wins GPCM ability recovery.
  2. discrimination STATE-CONDITIONED on ``[joint_summary, item_embed]`` rather
     than item-only.

The NRM head mirrors both:

    theta = ability_net(student_summary)                 # (B,S,1) item-blind
    a_k   = center( slope_net([joint_summary, item_embed]) )   # (B,S,K) state-cond
    c_k   = center( intercept_net(item_embed) )                # (B,S,K) item-only

    P(option k | theta) = softmax_k( a_k * theta + c_k )       # Bock 1972

``center`` is the SAME mean-centering identifiability convention as deep_irt's
``NRMDecoder`` (sum_k a_k = 0, sum_k c_k = 0), which is likelihood-neutral
(softmax shift-invariant) so the recovered a_k / c_k are directly comparable
across the two engines.

Contrast with deep_irt's NRM head (``decoders.NRMDecoder``): there a_k and c_k
are BOTH item-only linear maps of the item embedding.  Here a_k is conditioned
on the running student state, which is the lever this bench exists to test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ma-irt encoders (frozen, imported not edited).
from models.encoders.lstm import LSTMEncoder
from models.encoders.dkvmn import DKVMNEncoder


# ---------------------------------------------------------------------------
# NRM head (mirrors ma-irt's IRTParameterExtractor structure)
# ---------------------------------------------------------------------------

class NRMHead(nn.Module):
    """Per-option (a_k, c_k) from encoder summaries, Bock-centered.

    Mirrors ma-irt's GPCM ``IRTParameterExtractor``:

    - ``ability_net``   : ``input_dim -> 1``   for theta (item-blind summary).
    - ``slope_net``     : ``input_dim + question_dim -> K`` for raw a_k
                          (state-conditioned: joint summary + item embedding).
    - ``intercept_net`` : ``question_dim -> K`` for raw c_k (item-only).

    The slope/intercept raw outputs are mean-centered across the option axis to
    impose the Bock sum-to-zero constraints (likelihood-neutral).

    The ``slopes_mode`` flag selects the conditioning of the per-option slopes
    a_k (and, for symmetry, keeps the intercepts c_k item-only in both modes):

    - ``"state_conditioned"`` (default, the original ma-irt edge): a_k is a
      linear map of ``[joint_summary, item_embed]`` -- per-step, student-state
      dependent.  This is the lever the prior bench found wins prediction but
      hurts NRM item-parameter recovery.
    - ``"item_only"``: a_k is a linear map of ``item_embed`` alone -- a pure
      item constant, exactly like deep_irt's NRMDecoder.  theta stays item-
      blind (separate_theta) so only the slope conditioning changes.

    Args:
        input_dim: width of the encoder summaries (student / joint).
        question_dim: width of the encoder item embedding.
        n_options: K, number of unordered options.
        correct_option: index of the designated-correct option.
        ability_scale: multiplier on raw theta (matches ma-irt's ability_scale).
        slopes_mode: ``"state_conditioned"`` | ``"item_only"`` (see above).
    """

    def __init__(
        self,
        input_dim: int,
        question_dim: int,
        n_options: int,
        correct_option: int = 0,
        ability_scale: float = 1.0,
        slopes_mode: str = "state_conditioned",
    ) -> None:
        super().__init__()
        if n_options < 2:
            raise ValueError("n_options must be >= 2")
        if not (0 <= correct_option < n_options):
            raise ValueError(
                f"correct_option must be in [0, {n_options}), got {correct_option}"
            )
        if slopes_mode not in ("state_conditioned", "item_only"):
            raise ValueError(
                f"slopes_mode must be 'state_conditioned' or 'item_only', "
                f"got {slopes_mode!r}"
            )
        self.input_dim = input_dim
        self.question_dim = question_dim
        self.n_options = n_options
        self.correct_option = correct_option
        self.ability_scale = ability_scale
        self.slopes_mode = slopes_mode

        # theta: item-blind ability summary -> scalar (D=1).
        self.ability_net = nn.Linear(input_dim, 1)
        # a_k: slope source depends on slopes_mode.
        #   state_conditioned: [joint_summary, item_embed] -> K raw slopes.
        #   item_only:         item_embed only            -> K raw slopes.
        slope_in = (input_dim + question_dim
                    if slopes_mode == "state_conditioned" else question_dim)
        self.slope_net = nn.Linear(slope_in, n_options)
        # c_k: item-only -> K raw intercepts (both modes).
        self.intercept_net = nn.Linear(question_dim, n_options)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.ability_net.weight)
        nn.init.constant_(self.ability_net.bias, 0.0)
        nn.init.normal_(self.slope_net.weight, std=0.1)
        nn.init.constant_(self.slope_net.bias, 0.0)
        nn.init.normal_(self.intercept_net.weight, std=0.05)
        nn.init.constant_(self.intercept_net.bias, 0.0)

    @staticmethod
    def _center(x: Tensor) -> Tensor:
        """Bock sum-to-zero projection across the option axis (last dim)."""
        return x - x.mean(dim=-1, keepdim=True)

    def params(
        self, ability: Tensor, joint: Tensor, item: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Map summaries to (theta, a, c).

        Args:
            ability : (B, S, input_dim)   item-blind ability summary.
            joint   : (B, S, input_dim)   item-conditioned joint summary.
            item    : (B, S, question_dim) item embedding.

        Returns:
            theta : (B, S, 1)
            a     : (B, S, K)  Bock-centered per-option slopes.
            c     : (B, S, K)  Bock-centered per-option intercepts.
        """
        theta = self.ability_net(ability) * self.ability_scale          # (B,S,1)
        if self.slopes_mode == "state_conditioned":
            a_raw = self.slope_net(torch.cat([joint, item], dim=-1))    # (B,S,K)
        else:  # item_only: slopes are a pure function of the item embedding
            a_raw = self.slope_net(item)                                # (B,S,K)
        c_raw = self.intercept_net(item)                                # (B,S,K)
        a = self._center(a_raw)
        c = self._center(c_raw)
        return theta, a, c

    def logits(self, theta: Tensor, a: Tensor, c: Tensor) -> Tensor:
        """NRM logits a_k * theta + c_k -> (B, S, K)."""
        return a * theta + c                                            # broadcast (B,S,1)*(B,S,K)


# ---------------------------------------------------------------------------
# ma-irt encoder + NRM head model
# ---------------------------------------------------------------------------

class MaIrtNRM(nn.Module):
    """ma-irt encoder (LSTM | DKVMN) + Bock NRM head.

    The encoder is configured for the SEPARATED ability pathway so theta is read
    from the item-blind ``student_summary`` -- ma-irt's ability trick.  The head
    state-conditions the per-option slopes on ``[joint_summary, item_embed]``.

    Args:
        n_questions: item bank size Q (items are 1-indexed; 0 = padding).
        n_options: K, number of unordered options.
        encoder: ``"lstm"`` or ``"dkvmn"``.
        correct_option: designated-correct option index.
        d_model / summary_dim: encoder hidden width.
        key_dim: item-key (question) embedding width.
        value_dim: value-side (q, r) -> v embedding width.
        memory_size: DKVMN slot count (DKVMN only).
        ability_scale: multiplier on raw theta.
        slopes_mode: ``"state_conditioned"`` (a_k from [joint, item]) or
            ``"item_only"`` (a_k from item only).  See ``NRMHead``.
    """

    def __init__(
        self,
        n_questions: int,
        n_options: int,
        encoder: str = "lstm",
        correct_option: int = 0,
        d_model: int = 32,
        summary_dim: int = 32,
        key_dim: int = 32,
        value_dim: int = 32,
        memory_size: int = 20,
        dropout_rate: float = 0.0,
        ability_scale: float = 1.0,
        item_conditioned: bool = False,
        slopes_mode: str = "state_conditioned",
    ) -> None:
        super().__init__()
        self.n_questions = n_questions
        self.n_options = n_options
        self.encoder_name = encoder
        self.correct_option = correct_option

        if encoder == "lstm":
            self.encoder = LSTMEncoder(
                n_questions=n_questions,
                n_categories=n_options,
                d_model=d_model,
                n_layers=1,
                dropout_rate=dropout_rate,
                key_dim=key_dim,
                value_dim=value_dim,
                embedding_type="learned",
                separate_theta=True,            # item-blind ability stream
            )
            input_dim = d_model
        elif encoder == "dkvmn":
            self.encoder = DKVMNEncoder(
                n_questions=n_questions,
                n_categories=n_options,
                memory_size=memory_size,
                key_dim=key_dim,
                value_dim=value_dim,
                summary_dim=summary_dim,
                dropout_rate=dropout_rate,
                init_value_memory=True,
                embedding_type="learned",
                produce_ability_summary=True,    # item-blind ability stream
                produce_joint_summary=True,
                item_conditioned=item_conditioned,
            )
            input_dim = summary_dim
        else:
            raise ValueError(f"unknown encoder {encoder!r}")

        self.head = NRMHead(
            input_dim=input_dim,
            question_dim=key_dim,
            n_options=n_options,
            correct_option=correct_option,
            ability_scale=ability_scale,
            slopes_mode=slopes_mode,
        )

    def forward(self, questions: Tensor, responses: Tensor) -> dict:
        """Run encoder then NRM head.

        Args:
            questions: (B, S) long, 1-indexed item ids (0 = padding).
            responses: (B, S) long, option indices in [0, K-1].

        Returns:
            dict with logits (B,S,K), probs (B,S,K), theta (B,S,1),
            a (B,S,K), c (B,S,K).
        """
        enc = self.encoder(questions, responses)
        ability = enc.student_summary                          # item-blind
        joint = enc.joint_summary if enc.joint_summary is not None else ability
        item = enc.item_embed
        theta, a, c = self.head.params(ability, joint, item)
        logits = self.head.logits(theta, a, c)
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "theta": theta, "a": a, "c": c}
