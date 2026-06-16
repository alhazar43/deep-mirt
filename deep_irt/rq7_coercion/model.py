"""model.py -- RQ7 ma-irt engine: independent per-coercion fit + frozen readouts.

The engine is DECIDED, not re-litigated: ma-irt's ``LSTMGPCM`` with
``separate_theta=True`` (the audited-correct item-blind ability pathway, memory
project-maitrt-theta-read).  For the FROZEN single-item difficulty estimate and
the per-learner ability estimate we use DETERMINISTIC / ITEM-ONLY readouts, per
the design principle confirmed three times in this codebase (memory
project-ednet-sep-mairt, project-maitrt-nrm-itemonly): state-condition the model
for the JOINT fit, but read any frozen single parameter from an item-only
(difficulty) or final-state (ability) deterministic path -- not by averaging a
per-step state-conditioned quantity, which injects occurrence-sampling noise.

Per coercion we fit ONE LSTMGPCM at the coercion's K, then read:

    item difficulty  : GPCM location  mean_k beta_k(q_embed)   -- item-only,
                       deterministic (``threshold`` is a pure function of the
                       q_embed).  This is the frozen single-item parameter.
    learner ability  : theta at the final valid step  =
                       ability_network(student_summary_final)  -- item-blind,
                       deterministic.  One ability per learner on the coercion's
                       own scale.

Both readouts are scale-free downstream because RQ7's primary metric is Spearman
of RANKINGS across coercions; no explicit linking is needed.

Public API
----------
``fit_coercion(coerced, n_items, ...) -> (model, info)``
``item_difficulty(model, n_items, ...) -> (b_loc (Q,), a_disc (Q,))``
``learner_ability(model, coerced, ...) -> (sids, theta (N,))``
``item_seen_counts(coerced, n_items) -> (Q,) int``  per-item response count.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from deep_irt.core.realdata import collate_adapter_items
from deep_irt.rq7_coercion.data import CoercedData

from models.lstm_gpcm import LSTMGPCM
from models.components.irt import alpha_from_raw


def _q1_batch(records: List[Dict]):
    """Collate per-learner records (1-based local ``questions`` + ``responses``)
    into ma-irt CPU tensors (1-based ids, pad id 0).

    ``collate_adapter_items`` subtracts 1 (1-based -> 0-based); ma-irt wants
    1-based with pad id 0, so we add 1 back on masked positions.
    """
    items = [
        {"student_id": r["student_id"],
         "questions": torch.as_tensor(r["questions"], dtype=torch.long),
         "responses": torch.as_tensor(r["responses"], dtype=torch.long)}
        for r in records
    ]
    b = collate_adapter_items(items)
    q = torch.where(b.mask, b.item_ids + 1, torch.zeros_like(b.item_ids))
    return q, b.responses, b.mask, b.seq_lens, b.student_ids


def fit_coercion(
    coerced: CoercedData,
    n_items: int,
    d_model: int = 64,
    key_dim: int = 64,
    value_dim: int = 64,
    n_epochs: int = 80,
    lr: float = 0.005,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[LSTMGPCM, dict]:
    """Independently fit one ma-irt LSTMGPCM at the coercion's K.

    Minimises masked cross-entropy over all valid positions, minibatched for the
    8 GB GPU.  A DISTINCT ``seed`` per coercion (different q_embed init) ensures
    the cross-coercion Spearman cannot be inflated by a shared initialisation --
    each scale is learned from scratch from the coercion's labels alone.
    """
    q, r, mask, seq_lens, sids = _q1_batch(coerced.records)
    N = q.size(0)

    torch.manual_seed(seed)
    model = LSTMGPCM(
        n_questions=n_items,
        n_categories=coerced.K,
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
            logits = model(qb, rb)["logits"]
            B, S, Kk = logits.shape
            ce = F.cross_entropy(logits.reshape(B * S, Kk),
                                 rb.reshape(B * S), reduction="none")
            mflat = mb.reshape(B * S).float()
            loss = (ce * mflat).sum() / mflat.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.item()) * int(mflat.sum().item())
            tot_tok += int(mflat.sum().item())
        final = tot_loss / max(tot_tok, 1)
        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"  [{coerced.label}] epoch {ep:4d}  NLL={final:.4f}", flush=True)
    wall = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    info = {"final_nll": final, "wall_clock": wall, "n_params": n_params,
            "K": coerced.K, "label": coerced.label}
    return model, info


@torch.no_grad()
def item_difficulty(
    model: LSTMGPCM,
    n_items: int,
    coerced: CoercedData | None = None,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-item GPCM location (the frozen single-item difficulty) + canonical
    item-only discrimination.

    location  : mean over k of beta_k = threshold(q_embed)  -- item-only,
                deterministic.  For K=2 this is the single 2PL threshold b.
    a_disc    : canonical item-only alpha = exp(log_scale *
                disc_net([s_canonical, q_embed])) evaluated ONCE per item at the
                population-mean joint_summary ``s_canonical`` (deterministic,
                occurrence-noise-free; the project-ednet-sep-mairt
                "canonical_item_only" readout).  Reported for context only;
                difficulty is the primary metric.

    Returns ``(b_loc (Q,), a_disc (Q,))`` indexed by 0-based local id.
    """
    model.eval()
    irt = model.decoder.irt
    log_scale = model.decoder.alpha_log_scale

    all_ids = torch.arange(1, n_items + 1, device=device)
    emb_all = model.encoder.q_embed(all_ids)                  # (Q, key_dim)
    beta = irt.threshold(emb_all).cpu().numpy().astype(np.float64)  # (Q, K-1)
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
    b_loc = beta.mean(axis=1)                                 # (Q,)

    a_disc = np.full(n_items, np.nan)
    if coerced is not None:
        s_can = _population_mean_joint(model, coerced, batch_size, device)
        s_rep = s_can.unsqueeze(0).expand(emb_all.size(0), -1)
        raw = irt.discrimination_network(torch.cat([s_rep, emb_all], dim=-1))
        a_disc = alpha_from_raw(raw, log_scale).squeeze(-1).cpu().numpy().astype(np.float64)
    return b_loc, a_disc


@torch.no_grad()
def _population_mean_joint(
    model: LSTMGPCM,
    coerced: CoercedData,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Population-mean joint_summary over all valid (learner, step) positions."""
    q, r, mask, seq_lens, sids = _q1_batch(coerced.records)
    N = q.size(0)
    input_dim = model.decoder.input_dim
    acc = torch.zeros(input_dim, dtype=torch.float64, device=device)
    cnt = 0
    for s in range(0, N, batch_size):
        sl = slice(s, s + batch_size)
        qb = q[sl].to(device)
        rb = r[sl].to(device)
        mb = mask[sl].to(device)
        enc = model.encoder(qb, rb)
        m = mb.unsqueeze(-1)
        acc += (enc.joint_summary.double() * m).sum(dim=(0, 1))
        cnt += int(mb.sum().item())
    return (acc / max(cnt, 1)).to(torch.float32)


@torch.no_grad()
def learner_ability(
    model: LSTMGPCM,
    coerced: CoercedData,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-learner ability = theta at the final valid step (item-blind).

    Reads ``ability_network(student_summary_final) * ability_scale`` for D=1.
    Returns ``(student_ids (N,), theta (N,))`` in the order learners appear in
    ``coerced.records``.
    """
    model.eval()
    irt = model.decoder.irt
    q, r, mask, seq_lens, sids = _q1_batch(coerced.records)
    N = q.size(0)
    thetas = []
    for s in range(0, N, batch_size):
        sl = slice(s, s + batch_size)
        qb = q[sl].to(device)
        rb = r[sl].to(device)
        enc = model.encoder(qb, rb)
        idx = (seq_lens[sl].to(device) - 1).clamp(min=0)
        rows = torch.arange(qb.size(0), device=device)
        sfin = enc.student_summary[rows, idx]                # (b, input_dim)
        th = (irt.ability_network(sfin) * irt.ability_scale).squeeze(-1)  # (b,)
        thetas.append(th.cpu().numpy())
    theta = np.concatenate(thetas).astype(np.float64)
    return sids.numpy().astype(np.int64), theta


def item_seen_counts(coerced: CoercedData, n_items: int) -> np.ndarray:
    """Per-item total response count over the coercion (0-based local index)."""
    cnt = np.zeros(n_items, dtype=np.int64)
    for r in coerced.records:
        q0 = np.asarray(r["questions"]) - 1
        np.add.at(cnt, q0, 1)
    return cnt
