"""
model.py -- Joint encoder fit and per-part ability readout for the study.

Design decision (the key one).  We fit ONE joint ``DeepIRTModel`` (LSTM
encoder + GPCM decoder, K=4) on full multi-part EdNet sequences with a SHARED
item-embedding table.  Per-part ability is then read by running the FROZEN
joint encoder over a learner's part-restricted subsequence.  This places every
per-part ability on a single learned scale by construction, which is the
precondition for asking whether that scale is separable across parts.  Fitting
per-part separately would put abilities on different scales and confound the
separability question with linking error; that is rejected for the main fit.

We use the K=4 GPCM decoder (EdNet's native ordinal coercion) rather than
collapsing to binary: the ordinal signal is the scale we built and is what the
thesis claims to recover.

Public API
----------
``train_joint(...)``    -- fit the joint model, return (DeepIRTModel, info).
``part_ability(model, subseq_items)`` -- final-step theta per learner for a
                          list of part-restricted subsequence dicts.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from deep_irt.core.model import DeepIRTModel
from deep_irt.core.realdata import collate_adapter_items
from deep_irt.ednet_sep.data import LearnerRecord, full_batch


def train_joint(
    records: List[LearnerRecord],
    n_questions: int,
    n_cats: int = 4,
    emb_dim: int = 16,
    hidden_dim: int = 64,
    n_epochs: int = 60,
    lr: float = 0.01,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[DeepIRTModel, dict]:
    """Fit the joint encoder+GPCM decoder on full multi-part sequences.

    Returns the fitted ``DeepIRTModel`` and an info dict (final NLL, time,
    param count).
    """
    batch = full_batch(records)
    model = DeepIRTModel(
        num_items=n_questions,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_cats=n_cats,
        decoder="gpcm",
        device=device,
        seed=seed,
    )
    t0 = time.time()
    info = model.fit(
        item_ids=batch.item_ids,
        responses=batch.responses,
        n_epochs=n_epochs,
        lr=lr,
        verbose=verbose,
        mask=batch.mask,
        batch_size=batch_size,
    )
    info["wall_clock"] = time.time() - t0
    info["n_params"] = (
        sum(p.numel() for p in model.encoder.parameters())
        + sum(p.numel() for p in model.decoder.parameters())
    )
    return model, info


def part_ability(
    model: DeepIRTModel,
    subseq_items: List[Dict],
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Read final-step theta for a list of part-restricted subsequences.

    Parameters
    ----------
    model        : a fitted DeepIRTModel (encoder frozen for the readout).
    subseq_items : list of dicts {student_id, questions(1-based), responses}.

    Returns
    -------
    theta   : (M,) float -- final-step ability on the shared learned scale.
    sids    : (M,) int   -- student ids, aligned with ``theta``.
    """
    if len(subseq_items) == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    batch = collate_adapter_items(subseq_items)
    traj = model.track(batch.item_ids.to(device), batch.responses.to(device))
    seq_lens = batch.seq_lens
    theta_final = traj[torch.arange(len(seq_lens)), seq_lens - 1]
    return (
        theta_final.detach().cpu().numpy(),
        batch.student_ids.cpu().numpy().astype(np.int64),
    )


def part_representation(
    model: DeepIRTModel,
    subseq_items: List[Dict],
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the final-step LSTM hidden state (the rich representation).

    The leakage probe (RQ5a) needs more than the scalar theta: it asks whether
    the *representation* the encoder builds for a part-restricted history
    carries instrument identity.  We read the final-step LSTM hidden vector
    (dimension ``hidden_dim``), which the scalar theta is a linear projection
    of, so the probe sees everything the ability readout could and more.

    Returns
    -------
    repr_  : (M, hidden_dim) final hidden state per learner.
    theta  : (M,) scalar ability (the linear projection of the hidden state).
    sids   : (M,) student ids.
    """
    if len(subseq_items) == 0:
        h = model.encoder.hidden_dim
        return np.zeros((0, h)), np.zeros(0), np.zeros(0, dtype=np.int64)
    batch = collate_adapter_items(subseq_items)
    enc = model.encoder
    enc.eval()
    item_ids = batch.item_ids.to(device)
    responses = batch.responses.to(device)
    seq_lens = batch.seq_lens
    with torch.no_grad():
        ie = enc.item_val_emb(item_ids)
        re = enc.resp_emb(responses)
        x = torch.cat([ie, re], dim=-1)
        h_seq, _ = enc.lstm(x)                       # (M, T, hidden_dim)
        idx = (seq_lens - 1).to(device)
        h_final = h_seq[torch.arange(h_seq.size(0)), idx]   # (M, hidden_dim)
        theta_final = enc.theta_proj(h_final).squeeze(-1)   # (M,)
    return (
        h_final.detach().cpu().numpy(),
        theta_final.detach().cpu().numpy(),
        batch.student_ids.cpu().numpy().astype(np.int64),
    )
