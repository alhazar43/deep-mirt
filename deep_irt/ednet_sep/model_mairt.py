"""
model_mairt.py -- ma-irt joint fit + item-param recovery for the separability
study (the engine-swap of ``model.py``).

This is the ONLY engine-specific piece of the ma-irt port.  Everything
downstream -- the anchored fixed-parameter per-part theta MAP estimator
(``anchored_readout.py``) and the RQ4 / RQ5b analyses (``analysis.py``) -- is
engine-agnostic given a fixed per-item ``(a, b)`` table, so it is reused
verbatim.  See the module note in ``run_experiment_mairt.py``.

Engine.  We fit ONE joint ``LSTMGPCM`` (ma-irt's LSTM encoder + GPCM decoder)
with ``separate_theta=True`` -- the audited-correct, item-blind ability pathway
(memory project-maitrt-theta-read) -- on the SAME full multi-part EdNet
sequences the deep_irt run used, with K=4 (EdNet's ordinal coercion).  This
matches the deep_irt ``train_joint`` in data, K, learner subset and epoch
budget; only the encoder/decoder engine changes.

GPCM parameterization equivalence (the reason the readout is reused unchanged).
For D=1, ma-irt's ``GPCMLogits`` step value is ``alpha*theta - ||alpha||*beta =
alpha*(theta - beta)`` (alpha>0 so ||alpha||=alpha), and the anchored readout's
likelihood is ``cumsum(a*(theta - b))``.  They are identical with ``a=alpha`` and
``b=beta``; the MAP estimator is the exact inverse of the ma-irt GPCM likelihood.

Item-param recovery (mirrors slam_extend ``recover_full``, NOT the deep_irt).
ma-irt's beta is item-only (``threshold(item_embed)``) but its alpha is
STATE-CONDITIONED (``alpha = exp(disc_net([joint_summary, item_embed]))``).  So
per-item ``a`` is the AVERAGE over all (learner, step) occurrences of the item in
the joint fit -- exactly the bench/slam convention -- and ``b`` is the item-only
threshold read from the embedding (UNSORTED, as the decoder produces it, so the
fixed-parameter likelihood matches the trained model exactly).

Public API
----------
``train_joint_mairt(records, n_questions, ...) -> (LSTMGPCM, info)``
``recover_fixed_item_params_mairt(model, records, n_questions, device) -> (a, b)``
    -> per-item (a (Q,), b (Q, K-1)) indexed by 0-based local id, drop-in for
       ``anchored_readout.recover_fixed_item_params``'s output.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from deep_irt.core.realdata import collate_adapter_items
from deep_irt.ednet_sep.data import LearnerRecord

from models.lstm_gpcm import LSTMGPCM
from models.components.irt import alpha_from_raw


# ---------------------------------------------------------------------------
# Collation: 1-based question ids with pad=0 (ma-irt padding_idx convention)
# ---------------------------------------------------------------------------

def _q1_batch_from_records(records: List[LearnerRecord]):
    """Collate ``LearnerRecord`` (1-based questions) into ma-irt CPU tensors.

    ``collate_adapter_items`` subtracts 1 (1-based -> 0-based); ma-irt wants
    1-based with pad id 0 (the q_embed padding_idx), so we add 1 back on the
    masked positions and leave padded positions at 0.

    Returns (q (N,S) long pad=0, r (N,S) long, mask (N,S) bool, seq_lens (N,),
    student_ids (N,)).  Tensors stay on CPU; callers move minibatches to device.
    """
    items = [
        {
            "student_id": r.student_id,
            "questions": r.questions,   # already 1-based (adapter convention)
            "responses": r.responses,
        }
        for r in records
    ]
    b = collate_adapter_items(items)
    q = torch.where(b.mask, b.item_ids + 1, torch.zeros_like(b.item_ids))
    return q, b.responses, b.mask, b.seq_lens, b.student_ids


# ---------------------------------------------------------------------------
# Joint fit
# ---------------------------------------------------------------------------

def train_joint_mairt(
    records: List[LearnerRecord],
    n_questions: int,
    n_cats: int = 4,
    key_dim: int = 64,
    d_model: int = 64,
    value_dim: int = 64,
    n_epochs: int = 60,
    lr: float = 0.01,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[LSTMGPCM, dict]:
    """Fit the joint ma-irt LSTMGPCM (separate_theta=True) on full sequences.

    Minimises masked cross-entropy over all valid positions, minibatched for the
    8 GB GPU.  Returns the fitted model and an info dict (final NLL, time, params).
    """
    q, r, mask, seq_lens, sids = _q1_batch_from_records(records)
    N = q.size(0)

    torch.manual_seed(seed)
    model = LSTMGPCM(
        n_questions=n_questions,
        n_categories=n_cats,
        d_model=d_model,
        key_dim=key_dim,
        value_dim=value_dim,
        dropout_rate=0.0,
        separate_theta=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 1)

    t0 = time.time()
    model.train()
    final = float("nan")
    for ep in range(1, n_epochs + 1):
        perm = torch.randperm(N, generator=gen)
        tot_loss = 0.0
        tot_tok = 0
        for s in range(0, N, batch_size):
            bi = perm[s:s + batch_size]
            qb = q[bi].to(device)
            rb = r[bi].to(device)
            mb = mask[bi].to(device)
            opt.zero_grad()
            logits = model(qb, rb)["logits"]            # (B,S,K)
            B, S, Kk = logits.shape
            ce = F.cross_entropy(
                logits.reshape(B * S, Kk), rb.reshape(B * S), reduction="none"
            )
            mflat = mb.reshape(B * S).float()
            loss = (ce * mflat).sum() / mflat.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.item()) * int(mflat.sum().item())
            tot_tok += int(mflat.sum().item())
        final = tot_loss / max(tot_tok, 1)
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  [joint-mairt] epoch {ep:4d}  NLL={final:.4f}", flush=True)

    wall = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    info = {"final_nll": final, "wall_clock": wall, "n_params": n_params}
    return model, info


# ---------------------------------------------------------------------------
# Fixed item-parameter recovery (drop-in for recover_fixed_item_params)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _population_mean_joint_summary(
    model: LSTMGPCM,
    records: List[LearnerRecord],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Population-mean joint_summary over all valid (learner, step) positions.

    Runs the frozen encoder over every learner and averages the per-step
    ``joint_summary`` (the LSTM hidden ``h_t`` of the q-signal stream) over all
    non-padded positions.  Returns a single ``(input_dim,)`` canonical state --
    in-distribution for the LSTM by construction (it is the mean of the states
    the LSTM actually visited), and the principled choice for evaluating a
    state-conditioned quantity (alpha) at one representative point.  Preferred
    over a ZERO state, which is off the LSTM's hidden manifold (no real step has
    an all-zero hidden after LayerNorm) and would be off-distribution.
    """
    q, r, mask, seq_lens, sids = _q1_batch_from_records(records)
    N = q.size(0)
    input_dim = model.decoder.input_dim
    acc = torch.zeros(input_dim, dtype=torch.float64, device=device)
    cnt = 0
    for s in range(0, N, batch_size):
        sl = slice(s, s + batch_size)
        qb = q[sl].to(device)
        rb = r[sl].to(device)
        mb = mask[sl].to(device)
        enc_out = model.encoder(qb, rb)
        joint = enc_out.joint_summary                      # (b, S, input_dim)
        m = mb.unsqueeze(-1)                               # (b, S, 1)
        acc += (joint.double() * m).sum(dim=(0, 1))
        cnt += int(mb.sum().item())
    return (acc / max(cnt, 1)).to(torch.float32)           # (input_dim,)


@torch.no_grad()
def recover_fixed_item_params_mairt(
    model: LSTMGPCM,
    records: List[LearnerRecord],
    n_questions: int,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    alpha_mode: str = "occurrence_avg",
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover ma-irt's per-item GPCM (a, b), indexed by 0-based local id.

    Returns ``(a (Q,), b (Q, K-1))`` -- a drop-in for
    ``anchored_readout.recover_fixed_item_params``.  The feed convention matches
    the ma-irt GPCM likelihood exactly.

    ``alpha_mode`` selects how the single FROZEN per-item discrimination is read
    from ma-irt's STATE-CONDITIONED alpha (``alpha = exp(log_scale *
    disc_net([joint_summary, item_embed]))``):

    * ``"occurrence_avg"`` (default, the original convention): per-item ``a`` is
      the MEAN of ``alpha`` over every (learner, step) occurrence of the item in
      the joint fit (the bench/slam ``recover_full`` convention).  This averages
      a per-STEP quantity over occurrences, so the point estimate carries
      occurrence-sampling NOISE -- the diagnosed cause of the RQ5b sep-index dip.
      Unseen items fall back to alpha at a zero joint state.

    * ``"canonical_item_only"`` (the fix under test): evaluate alpha ONCE per
      item at a single CANONICAL state ``s_canonical`` = the population-mean
      joint_summary (mean over all learner-steps; in-distribution).  This yields
      a deterministic, occurrence-noise-free per-item alpha that is a pure
      function of the item embedding given the fixed canonical state -- the
      "item-only readout for a frozen estimate" design principle (NRM item-only
      head, RQ2 difficulty).  ``b`` is identical in both modes.

    ``b`` : item-only ``threshold(item_embed)``, UNSORTED (the decoder does not
    sort), so the fixed-parameter likelihood reproduces the trained model.
    """
    if alpha_mode not in ("occurrence_avg", "canonical_item_only"):
        raise ValueError(
            f"alpha_mode must be 'occurrence_avg' or 'canonical_item_only', "
            f"got {alpha_mode!r}."
        )
    model.eval()
    irt = model.decoder.irt
    log_scale = model.decoder.alpha_log_scale

    # ---- beta: item-only, one row per item from its q_embed -----------------
    all_ids = torch.arange(1, n_questions + 1, device=device)   # 1-based ids
    emb_all = model.encoder.q_embed(all_ids)                    # (Q, key_dim)
    b_table = irt.threshold(emb_all).cpu().numpy().astype(np.float64)  # (Q, K-1)
    if b_table.ndim == 1:
        b_table = b_table.reshape(-1, 1)

    if alpha_mode == "canonical_item_only":
        # ---- alpha: deterministic item-only read at the canonical state -----
        # One clean alpha per item: exp(log_scale * disc_net([s_canonical,
        # item_embed])).  No occurrence-averaging, so no occurrence-sampling
        # noise; alpha is a pure function of the item embedding given s_canonical.
        s_can = _population_mean_joint_summary(model, records, batch_size, device)
        s_rep = s_can.unsqueeze(0).expand(emb_all.size(0), -1)  # (Q, input_dim)
        raw = irt.discrimination_network(torch.cat([s_rep, emb_all], dim=-1))
        a_table = alpha_from_raw(raw, log_scale).squeeze(-1).cpu().numpy()
        return a_table.astype(np.float64), b_table

    # ---- alpha (occurrence_avg): average state-conditioned alpha over occ ----
    q, r, mask, seq_lens, sids = _q1_batch_from_records(records)
    N = q.size(0)
    a_sum = np.zeros(n_questions, dtype=np.float64)
    a_cnt = np.zeros(n_questions, dtype=np.float64)
    for s in range(0, N, batch_size):
        sl = slice(s, s + batch_size)
        qb = q[sl].to(device)
        rb = r[sl].to(device)
        mb = mask[sl]
        out = model(qb, rb)
        alpha = out["alpha"].squeeze(-1).cpu().numpy()          # (b,S)
        ids0 = (qb.cpu().numpy() - 1)                           # (b,S) 0-based; pad=-1
        m_np = mb.numpy()
        fi = ids0[m_np]
        fa = alpha[m_np]
        np.add.at(a_sum, fi, fa)
        np.add.at(a_cnt, fi, 1.0)
    seen = a_cnt > 0
    a_table = np.where(seen, a_sum / np.maximum(a_cnt, 1.0), 0.0)

    # Fallback alpha for unseen items: state-conditioned alpha at a zero joint
    # state (the disc_net input is [joint=0, item_embed]).  Keeps every id dense;
    # these ids are not scored by the per-part readout (no learner answered them).
    if (~seen).any():
        zero_joint = torch.zeros(emb_all.size(0), model.decoder.input_dim, device=device)
        raw = irt.discrimination_network(torch.cat([zero_joint, emb_all], dim=-1))
        a_fallback = alpha_from_raw(raw, log_scale).squeeze(-1).cpu().numpy()
        a_table = np.where(seen, a_table, a_fallback.astype(np.float64))

    return a_table.astype(np.float64), b_table
