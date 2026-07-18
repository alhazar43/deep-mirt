"""
encoder.py -- sequence encoders: response history -> per-step latent ability (theta).

The encoder is the shared backbone.  It consumes a sequence of (item_id, response)
pairs and emits a scalar theta at each timestep.  This per-step theta is the
interface contract between the encoder and any decoder.

Two item embeddings, opposite needs (the key/value framing)
-----------------------------------------------------------
An item carries two embeddings with opposite needs.  The VALUE embedding
(``item_val_emb``, width ``emb_dim``) is thin and feeds the encoder that builds
the dynamic ability theta.  The KEY embedding (``item_key_emb``, width
``item_key_dim``) is wide and feeds the decoder readout of the static item
parameters alpha and beta.  Width follows inverse Fisher: alpha is
low-information and capacity-hungry so its key is wide, while theta is
well-determined and is HURT by width (a fat encoder input lets the LSTM memorize
items into the ability state), so the value stays thin.  Both are trained
nn.Embeddings; "static" / "dynamic" describes behavior over the sequence (item
params are occurrence-invariant, ability evolves), not training.

Swappable backbone
------------------
``BaseSeqEncoder`` owns everything the decoder consumes -- the item value /
response / item key embedding tables, the ``theta_proj`` head, the single-shift
causal alignment, and the public accessors -- and delegates the actual sequence
modelling to a single per-step hidden producer a subclass implements:

    _direct_hidden(item_ids, responses)     -> h (B,T,hidden)  h_t sees (q_<=t, r_<=t)

``LSTMEncoder`` is the default backbone (a plain ``nn.LSTM``).  A Transformer or
DKVMN backbone is just another subclass implementing the same producer; the wide
item key (``item_key_emb``) lives in the base, so it composes with ANY backbone
with no decoder change.  This is the swappability contract.

Interface
---------
encoder.encode(item_ids, responses)               -> theta  (B, T)    raw responsive
encoder.theta_for_prediction(item_ids, responses) -> theta  (B, T)    causal-aligned
encoder.state_for_prediction(item_ids, responses) -> state  (B, T, H) causal-aligned
encoder.get_final_theta(item_ids, responses)      -> theta  (B,)

The item value table lives inside the encoder so the same embedding weights are
shared between encoding and decoding (both read from the same item
representations).  The AnchoredExtension in anchor.py exploits this: it appends
new embedding rows while keeping the encoder weights frozen.

ONE causal convention (the alignment contract)
----------------------------------------------
For EVERY backbone, the ability/state used to predict the response at step t is a
function of the interaction history STRICTLY BEFORE t -- that is, of
(q_{<t}, r_{<t}) INCLUDING the immediately preceding interaction (q_{t-1},
r_{t-1}) -- and NEVER of the current response r_t.  This is the same convention
ma-irt's encoders use (``ma-irt/models/encoders/lstm.py``): the hidden is
right-shifted by exactly ONE position and the decoder consumes the resulting
hidden directly.  Exactly one shift, total.

``_direct_hidden`` returns h_t that sees (q_{<=t}, r_{<=t}).  ``encode`` returns
``theta_proj(h)`` -- the RAW, responsive per-step theta.  To PREDICT step t we
need the hidden AFTER history 0..t-1, so ``aligned_theta_and_state`` applies ONE
right-shift: position t carries h_{t-1} = function of (q_{<t}, r_{<t}).  The
shift drops r_t (causal) AND means the predicting state's most recent item is
q_{t-1}, not q_t -- so it is item-blind for the current step by construction.
The decoder still reads the CURRENT item embedding q_t directly for alpha/beta,
so ability and current-item identity are separated ORGANICALLY in one pass.

Every training / evaluation / recovery path uses ONLY the ``aligned_theta_and_state``
accessors, so callers never reason about how many shifts to apply.

State-conditioned discrimination (the organic alpha feature)
------------------------------------------------------------
``state_for_prediction`` returns the SAME causal hidden that produces the
prediction-aligned theta.  A decoder may condition discrimination on
``[state, item_val_emb]`` (ma-irt's IRTParameterExtractor recipe) by reading that
state.  In the DECOUPLED variant (``item_key_dim`` set) the alpha head instead
reads ``[state, item_key_emb]`` -- a SEPARATE wide item key that feeds only the
decoder's static alpha and beta readouts, never the backbone input -- so alpha's
item capacity is decoupled from theta's.  This is a clean DECODER feature, one
backbone pass, one shift; it is selected at the DeepIRTModel level via
``state_alpha`` / ``item_key_dim`` (see deep_irt.py).
"""

import torch
import torch.nn as nn


class BaseSeqEncoder(nn.Module):
    """Backbone-agnostic IRT head shared by every sequence encoder.

    A subclass builds the embedding tables + ``theta_proj`` in ``__init__`` and
    implements ``_direct_hidden``.  Everything the decoder consumes -- the
    single-shift alignment, the raw responsive ``encode``, the final-step
    readout -- lives here and is backbone-agnostic.

    Required subclass attributes
    ----------------------------
    self.theta_proj    : nn.Linear(hidden_dim, 1)
    self.item_val_emb  : nn.Embedding(num_items, emb_dim)  -- thin value table
                         feeding the encoder/theta
    self.item_key_emb  : nn.Embedding(num_items, item_key_dim)  -- wide key table
                         feeding the static alpha/beta readouts, only when
                         ``item_key_dim`` is set (the DECOUPLED variant)

    Required subclass methods
    -------------------------
    _direct_hidden(item_ids, responses)     -> (B, T, hidden)  h_t sees q_<=t, r_<=t
    """

    # ------------------------------------------------------------------
    # Per-step hidden producer (backbone-specific; subclass MUST override)
    # ------------------------------------------------------------------

    def _direct_hidden(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Raw responsive hidden: h_t is a function of (q_{<=t}, r_{<=t})."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface (backbone-agnostic)
    # ------------------------------------------------------------------

    def encode(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Encode response sequences into per-step latent ability.

        Returns the RAW responsive theta (h_t sees q_t, r_t).  Use
        ``theta_for_prediction`` for the causally-aligned theta.

        Returns
        -------
        theta : (batch, seq_len)  float -- scalar ability at each timestep
        """
        h = self._direct_hidden(item_ids, responses)
        return self.theta_proj(h).squeeze(-1)

    def aligned_theta_and_state(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prediction-aligned (theta, state) from ONE backbone pass.

        Both come from a single hidden computation (one pass, one shift), so a
        state-conditioned decoder reads theta and the alpha-state from the same
        causal hidden without a second stream.

        Raw h_t sees (q_t, r_t).  state = right-shift(h), so state[t] = h_{t-1}
        (history < t).  theta = right-shift of the RAW theta, so theta[0] = 0
        (the legacy prior) and theta[t>=1] = theta_proj(h_{t-1}) =
        theta_proj(state[t]).

        At t=0 both priors mean "no history": theta is 0 and state is the zero
        vector -- mutually consistent.
        """
        h = self._direct_hidden(item_ids, responses)           # h_t sees q_t,r_t
        state = self._shift(h)                                  # h_{t-1}
        theta = self._shift(self.theta_proj(h).squeeze(-1))     # legacy 0 prior
        return theta, state

    def state_for_prediction(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Per-step hidden used to PREDICT step t (the alignment contract).

        state[t] is a function of the history strictly before t, including
        (q_{t-1}, r_{t-1}), never r_t.  Exactly one shift, total.
        """
        return self.aligned_theta_and_state(item_ids, responses)[1]

    def theta_for_prediction(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Per-step theta used to PREDICT step t (the alignment contract).

        theta_in[t] is the ability estimate BEFORE answering item at position t,
        so the model predicts response[t] from theta_in[t] and item_id[t].
        """
        return self.aligned_theta_and_state(item_ids, responses)[0]

    def get_final_theta(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Return the final-timestep theta for each sequence.  (batch,)"""
        return self.encode(item_ids, responses)[:, -1]

    # ------------------------------------------------------------------
    # Causal shift helper
    # ------------------------------------------------------------------

    @staticmethod
    def _shift(x: torch.Tensor) -> torch.Tensor:
        """Causal right-shift by one along the time axis.

        Prepends a zero prior at t=0 and drops the last step.  Works for theta
        (B, T) and per-step state (B, T, H) alike.
        """
        zeros = torch.zeros_like(x[:, :1])
        return torch.cat([zeros, x[:, :-1]], dim=1)


class LSTMEncoder(BaseSeqEncoder):
    """LSTM backbone mapping (item_ids, responses) sequences to per-step theta.

    Parameters
    ----------
    num_items : int
        Total number of items in the embedding table (base B, or extended B+E).
    emb_dim : int
        Dimensionality of both the thin item value and response embeddings.
    hidden_dim : int
        LSTM hidden state size.
    n_cats : int
        Number of ordered response categories K (responses in {0, ..., K-1}).
    item_key_dim : int or None
        When set, build a SEPARATE wide item KEY table
        (``item_key_emb``, ``num_items x item_key_dim``) that feeds ONLY the
        decoder's static alpha and beta readouts, NOT the LSTM input.  This
        DECOUPLES alpha's item capacity from theta's: the theta-encoder stays at
        the cheap ``emb_dim`` / ``hidden_dim`` (so static theta and dynamic drift
        are unchanged) while the static item params read a wide item key.  None
        (default) builds no extra table and is bit-for-bit identical to the
        original encoder.  The table is constructed AFTER all default-path
        modules so the default-path RNG draws are untouched.
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int = 8,
        hidden_dim: int = 32,
        n_cats: int = 4,
        item_key_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_cats = n_cats
        self.item_key_dim = item_key_dim

        self.item_val_emb = nn.Embedding(num_items, emb_dim)
        self.resp_emb = nn.Embedding(n_cats, emb_dim)

        self.lstm = nn.LSTM(emb_dim + emb_dim, hidden_dim, batch_first=True)
        self.theta_proj = nn.Linear(hidden_dim, 1)

        # Wide item key for the DECOUPLED static readouts.  Built last and only
        # when requested, so it never perturbs the default-path init.  It feeds
        # the decoder's alpha and beta heads, never the LSTM input.
        if item_key_dim is not None:
            self.item_key_emb = nn.Embedding(num_items, item_key_dim)

    # ------------------------------------------------------------------
    # Per-step hidden state (the raw responsive stream)
    # ------------------------------------------------------------------

    def _direct_hidden(
        self, item_ids: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Raw responsive hidden: h_t is a function of (q_{<=t}, r_{<=t}).

        The LSTM sees the current interaction (q_t, r_t) at step t.

        Returns
        -------
        h : (batch, seq_len, hidden_dim)
        """
        ie = self.item_val_emb(item_ids)          # (B, T, emb_dim)
        re = self.resp_emb(responses)             # (B, T, emb_dim)
        x = torch.cat([ie, re], dim=-1)           # (B, T, 2*emb_dim)
        h, _ = self.lstm(x)                       # (B, T, hidden_dim)
        return h
