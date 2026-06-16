"""model.py -- RQ8 ma-irt engine: independent per-subgroup K=2 fit + frozen
item-only difficulty readout.

The engine is DECIDED, not re-litigated (memory project-maitrt-theta-read,
project-ednet-sep-mairt, project-rq7-coercion): ma-irt's ``LSTMGPCM`` with
``separate_theta=True``, and the FROZEN single-item difficulty read from the
DETERMINISTIC ITEM-ONLY path -- the GPCM location ``threshold(q_embed)``, a pure
function of the item embedding, with NO occurrence-averaging of any
state-conditioned quantity.  For K=2 this location IS the single 2PL difficulty
``b`` (one value per item).

Per subgroup we fit ONE LSTMGPCM at K=2 (binary correctness), then read:

    item difficulty : b_i = threshold(q_embed_i)  -- item-only, deterministic.
                      The frozen single-item parameter compared across subgroups.

DIF is then a comparison of these per-item ``b`` between two subgroup fits, AFTER
a mean/sigma linking onto one metric (the linking lives in the runner; this layer
just produces the per-group ``b`` table and per-item response counts so the
runner can choose the common anchor set).

A DISTINCT seed per subgroup (different q_embed init) ensures the cross-group
agreement cannot be inflated by a shared initialisation -- each subgroup's scale
is learned from scratch from its own learners' labels alone.

Public API
----------
``fit_group(records, n_items, K, ...) -> (model, info)``
``item_difficulty_b(model, n_items, ...) -> b (Q,)``  item-only 2PL difficulty.
``item_seen_counts(records, n_items) -> (Q,) int``    per-item response count.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from deep_irt.core.realdata import collate_adapter_items

from models.lstm_gpcm import LSTMGPCM


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


def fit_group(
    records: List[Dict],
    n_items: int,
    K: int = 2,
    d_model: int = 64,
    key_dim: int = 64,
    value_dim: int = 64,
    n_epochs: int = 80,
    lr: float = 0.005,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    label: str = "group",
    verbose: bool = True,
) -> Tuple[LSTMGPCM, dict]:
    """Independently fit one ma-irt LSTMGPCM at K on one subgroup's records.

    Minimises masked cross-entropy over all valid positions, minibatched for the
    8 GB GPU.  A DISTINCT ``seed`` per subgroup ensures the cross-group b
    agreement is not a shared-init artifact.
    """
    q, r, mask, seq_lens, sids = _q1_batch(records)
    N = q.size(0)

    torch.manual_seed(seed)
    model = LSTMGPCM(
        n_questions=n_items,
        n_categories=K,
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
            print(f"  [{label}] epoch {ep:4d}  NLL={final:.4f}", flush=True)
    wall = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    info = {"final_nll": final, "wall_clock": wall, "n_params": n_params,
            "K": K, "label": label, "n_learners": int(N)}
    return model, info


@torch.no_grad()
def item_difficulty_b(
    model: LSTMGPCM,
    n_items: int,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Per-item 2PL difficulty ``b`` = item-only GPCM location.

    ``b_i = threshold(q_embed_i)`` -- item-only, deterministic.  For K=2 the
    threshold has shape (Q, 1); we squeeze to (Q,).  For K>2 we take the mean
    over the K-1 thresholds (the GPCM location), but RQ8 uses K=2 so this is the
    single 2PL difficulty.  UNSORTED, as the decoder produces it.

    Returns ``b (Q,)`` indexed by 0-based local id.
    """
    model.eval()
    irt = model.decoder.irt
    all_ids = torch.arange(1, n_items + 1, device=device)
    emb_all = model.encoder.q_embed(all_ids)               # (Q, key_dim)
    beta = irt.threshold(emb_all).cpu().numpy().astype(np.float64)  # (Q, K-1)
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
    return beta.mean(axis=1)                               # (Q,)


def item_seen_counts(records: List[Dict], n_items: int) -> np.ndarray:
    """Per-item response count over a subgroup's records (0-based local index)."""
    cnt = np.zeros(n_items, dtype=np.int64)
    for r in records:
        q0 = np.asarray(r["questions"]) - 1
        np.add.at(cnt, q0, 1)
    return cnt
