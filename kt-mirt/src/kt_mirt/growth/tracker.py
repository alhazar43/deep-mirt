"""PAS-N, the PASSIVE neural trackers (`_planning/design/a4_design.md`
v1.1, section 2.2): PAS-N1 (shared-state tracker, the field-representative
config Ding and Larson predict will fail contamination) and PAS-N2
(factorized per-KC tracker, contamination-proof by construction), both
built on the vendored core (`kt_mirt.core.encoder`), trained on
prediction NLL against the FROZEN bank scale (only the ability path
trains -- section 2.2: "the decoder consumes frozen b_j; only the ability
path trains"), under the neural 80/20-learner-split forecast-NLL
validation regime (section 2.2 / R1-I11).

Interpretation notes (recorded per the harness instructions):

1. **No `Binary2PLDecoder` in the loss.** Section 2.5's attachment-point
   table says the frozen bank "feeds `Binary2PLDecoder` unchanged", but
   `Binary2PLDecoder.item_params` by default LEARNS alpha/beta from the
   item's own embedding -- exactly the free per-item structure section
   2.1 says v1 does not have ("no per-KC alpha/beta structure in v1").
   Consistent with 1PL-only and "difficulties frozen, ability path free",
   this module bypasses the decoder's own learned item-parameter head
   entirely and computes the binary logit directly as ``theta - b_j``
   with ``b_j`` supplied externally from a `FrozenBank` (or a synthetic
   twin's true difficulties, for CG7's frozen-encoder-margin check on
   synthetic data). The ENCODER's own item value embedding
   (`item_val_emb`) remains a normal trainable encoder-side parameter (it
   builds the LSTM/DKVMN's item representation, a distinct concern from
   the decoder's static difficulty readout) -- this is exactly "only the
   ability path trains."
2. **PAS-N2's "same shared LSTM cell applied independently to each
   slice's subsequence"** is implemented literally: one `LSTMEncoder` (or
   `DKVMNEncoder`) instance, with a plain scalar `Linear(hidden_dim, 1)`
   ability head reading `state_for_prediction`, is trained with each
   (learner, KC) `Slice` (from `slices.py`) treated as its OWN
   independent training sequence (batched over slices exactly like a
   normal single-KC DeepIRT fit). Order-invariance to cross-KC
   interleaving (CG9) holds by construction because each slice's forward
   pass never sees any other KC's interactions at all.
3. **PAS-N1's multi-tag gather** averages the per-KC ability head's
   output over the CURRENT interaction's tagged KC slots (section 2.5:
   "PAS-N1's per-KC gather averages over the item's tagged slots"), via
   `torch.gather` + a masked mean; padding KC id ``-1`` is clamped to a
   valid index before the gather and then zeroed out by the mask, so it
   never contributes.
4. **Frozen-encoder null (battery arm 3 / CG7 / RB2).** `build_tracker`
   accepts `train_encoder=False`, which freezes every encoder parameter
   (`requires_grad_(False)`) and trains only the ability head -- the
   "untrained/frozen-encoder null" both CG7 (synthetic) and RB2 (real
   bed) compare the trained tracker against. This module exposes the
   frozen-vs-trained model construction and the forecast-NLL evaluator;
   the actual CG7/RB2 margin computation (trained-vs-frozen profile
   correlation, the decertification rule) is `battery.py`'s job (a later
   stage), since it also needs the contamination and order-invariance
   probes (arms 4, 5) to interpret jointly.
5. **E-P3 (tracker displacement)** is exposed as a plain post-hoc
   function (`displacement`) over a already-computed per-step theta
   series, not baked into training -- "the passive growth read from
   either tracker is E-P3 (displacement), judged only through the
   battery" (section 2.2), i.e. `battery.py` decides which reads count,
   this module only computes the quantity.
6. **Mini-batched training (`TrackerConfig.batch_size`), an
   IMPLEMENTATION-LEVEL memory concession.** The frozen design fixes
   estimators and thresholds, not batch sizes. At the EdNet-matched
   production profile (N=6000, ~25 KCs/learner) the whole-cohort
   `train_pas_n2` backward pass allocates >50 GiB and OOMs every 44-48
   GiB card on the cluster (the a100 pool is 40 GB, smaller still), so
   `batch_size` mini-batches BOTH trainers over the row dimension
   (learners for PAS-N1, slices for PAS-N2), mirroring the vendored
   core's own `DeepIRTModel.fit(batch_size=...)` semantics: ``None``
   (the default) or ``batch_size >= n_rows`` is the EXACT pre-existing
   whole-cohort loop, byte-unchanged, so every KDD-profile cell trains
   identically to before this parameter existed; a smaller value
   shuffles rows each epoch with a generator seeded ``cfg.seed + 1``
   (never the global torch RNG) and records the per-epoch MEAN loss in
   the trace. Only the EdNet-profile campaign units set it
   (`slurm/enumerate_units.py`'s per-profile overrides).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kt_mirt.core.dkvmn_encoder import DKVMNEncoder
from kt_mirt.core.encoder import BaseSeqEncoder, LSTMEncoder
from kt_mirt.growth.bank import FrozenBank
from kt_mirt.growth.synth import LearnerLog


@dataclass
class TrackerConfig:
    encoder: str = "lstm"  # "lstm" (primary) | "dkvmn" (pre-registered alternate)
    emb_dim: int = 8
    hidden_dim: int = 32
    lr: float = 1e-2
    n_epochs: int = 30
    seed: int = 0
    device: str = "cpu"
    train_encoder: bool = True  # False = frozen-encoder null (arm 3 / CG7 / RB2)
    # None = whole-cohort training, the exact pre-existing loop (module
    # docstring note 6); an int mini-batches rows per epoch (EdNet OOM fix).
    batch_size: Optional[int] = None


def _build_encoder(num_items: int, cfg: TrackerConfig) -> BaseSeqEncoder:
    if cfg.encoder == "lstm":
        return LSTMEncoder(num_items, emb_dim=cfg.emb_dim, hidden_dim=cfg.hidden_dim, n_cats=2)
    if cfg.encoder == "dkvmn":
        return DKVMNEncoder(num_items, emb_dim=cfg.emb_dim, hidden_dim=cfg.hidden_dim, n_cats=2)
    raise ValueError(f"unknown encoder {cfg.encoder!r}, expected 'lstm' or 'dkvmn'")


# ---------------------------------------------------------------------------
# PAS-N1: shared-state tracker
# ---------------------------------------------------------------------------


class PASN1Model(nn.Module):
    """Stock encoder + per-KC ability head `Linear(hidden_dim, C)` reading
    `state_for_prediction`, gathered by the current item's tagged KC ids
    (mean over valid tag slots, module docstring note 3)."""

    def __init__(self, num_items: int, n_kcs: int, cfg: TrackerConfig):
        super().__init__()
        self.encoder = _build_encoder(num_items, cfg)
        self.ability_head = nn.Linear(cfg.hidden_dim, n_kcs)
        self.n_kcs = n_kcs

    def forward(
        self, item_ids: torch.Tensor, responses: torch.Tensor, kc_ids: torch.Tensor, kc_mask: torch.Tensor
    ) -> torch.Tensor:
        """``kc_ids``/``kc_mask``: (B, T, A). Returns per-position gathered
        theta (B, T)."""
        state = self.encoder.state_for_prediction(item_ids, responses)  # (B, T, H)
        ability = self.ability_head(state)  # (B, T, C)
        kc_clamped = kc_ids.clamp(min=0)
        gathered = torch.gather(ability, 2, kc_clamped)  # (B, T, A)
        mask_f = kc_mask.to(gathered.dtype)
        denom = mask_f.sum(dim=-1).clamp(min=1.0)
        return (gathered * mask_f).sum(dim=-1) / denom


# ---------------------------------------------------------------------------
# PAS-N2: factorized per-KC tracker
# ---------------------------------------------------------------------------


class PASN2Model(nn.Module):
    """The SAME shared encoder cell, applied independently to each
    (learner, KC) slice's own subsequence, with a plain scalar ability
    head -- contamination-free and order-invariant by construction
    (module docstring note 2)."""

    def __init__(self, num_items: int, cfg: TrackerConfig):
        super().__init__()
        self.encoder = _build_encoder(num_items, cfg)
        self.ability_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, item_ids: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        state = self.encoder.state_for_prediction(item_ids, responses)  # (B, T, H), B = n_slices
        return self.ability_head(state).squeeze(-1)


def freeze_encoder(model: nn.Module) -> None:
    """Battery-arm-3 frozen-encoder null: freeze every encoder parameter
    (seeded random init), train only the ability head."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)


def build_tracker(kind: str, num_items: int, n_kcs: int, cfg: TrackerConfig) -> nn.Module:
    torch.manual_seed(cfg.seed)
    if kind == "pas_n1":
        model = PASN1Model(num_items, n_kcs, cfg)
    elif kind == "pas_n2":
        model = PASN2Model(num_items, cfg)
    else:
        raise ValueError(f"unknown tracker kind {kind!r}, expected 'pas_n1' or 'pas_n2'")
    if not cfg.train_encoder:
        freeze_encoder(model)
    return model.to(cfg.device)


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


@dataclass
class LearnerBatch:
    item_ids: torch.Tensor  # (B, T)
    responses: torch.Tensor  # (B, T) long
    kc_ids: torch.Tensor  # (B, T, A)
    kc_mask: torch.Tensor  # (B, T, A)
    seq_mask: torch.Tensor  # (B, T) bool
    b_j: torch.Tensor  # (B, T)


def _difficulty(item_ids: np.ndarray, bank: Optional[FrozenBank], b_true: Optional[np.ndarray]) -> np.ndarray:
    if bank is not None:
        return bank.difficulty(item_ids)
    if b_true is not None:
        return np.asarray(b_true)[item_ids]
    return np.zeros(len(item_ids))


def build_learner_batch(
    learners: Sequence[LearnerLog], bank: Optional[FrozenBank] = None, b_true: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> LearnerBatch:
    """PAS-N1's batch shape: full interleaved multi-KC learner sequences."""
    B = len(learners)
    if B == 0:
        return LearnerBatch(*(torch.zeros(0) for _ in range(6)))
    T_max = max(len(log.item_ids) for log in learners)
    A = max(log.tag_ids.shape[1] for log in learners)
    item_ids = torch.zeros(B, T_max, dtype=torch.long)
    responses = torch.zeros(B, T_max, dtype=torch.long)
    kc_ids = torch.zeros(B, T_max, A, dtype=torch.long)
    kc_mask = torch.zeros(B, T_max, A, dtype=torch.bool)
    seq_mask = torch.zeros(B, T_max, dtype=torch.bool)
    b_j = torch.zeros(B, T_max)
    for i, log in enumerate(learners):
        t = len(log.item_ids)
        a = log.tag_ids.shape[1]
        item_ids[i, :t] = torch.as_tensor(log.item_ids, dtype=torch.long)
        responses[i, :t] = torch.as_tensor(log.responses, dtype=torch.long)
        kc_ids[i, :t, :a] = torch.as_tensor(log.tag_ids, dtype=torch.long)
        kc_mask[i, :t, :a] = torch.as_tensor(log.tag_mask, dtype=torch.bool)
        seq_mask[i, :t] = True
        b_j[i, :t] = torch.as_tensor(_difficulty(log.item_ids, bank, b_true), dtype=torch.float32)
    return LearnerBatch(
        item_ids=item_ids.to(device), responses=responses.to(device), kc_ids=kc_ids.to(device),
        kc_mask=kc_mask.to(device), seq_mask=seq_mask.to(device), b_j=b_j.to(device),
    )


@dataclass
class SliceBatch:
    item_ids: torch.Tensor
    responses: torch.Tensor
    mask: torch.Tensor
    b_j: torch.Tensor


def build_slice_batch(
    slices: Sequence, bank: Optional[FrozenBank] = None, b_true: Optional[np.ndarray] = None, device: str = "cpu"
) -> SliceBatch:
    """PAS-N2's batch shape: one row per (learner, KC) `Slice`."""
    S = len(slices)
    if S == 0:
        return SliceBatch(*(torch.zeros(0) for _ in range(4)))
    T_max = max(sl.T for sl in slices)
    item_ids = torch.zeros(S, T_max, dtype=torch.long)
    responses = torch.zeros(S, T_max, dtype=torch.long)
    mask = torch.zeros(S, T_max, dtype=torch.bool)
    b_j = torch.zeros(S, T_max)
    for i, sl in enumerate(slices):
        t = sl.T
        item_ids[i, :t] = torch.as_tensor(sl.item_id, dtype=torch.long)
        responses[i, :t] = torch.as_tensor(sl.response, dtype=torch.long)
        mask[i, :t] = True
        b_j[i, :t] = torch.as_tensor(_difficulty(sl.item_id, bank, b_true), dtype=torch.float32)
    return SliceBatch(
        item_ids=item_ids.to(device), responses=responses.to(device), mask=mask.to(device), b_j=b_j.to(device)
    )


# ---------------------------------------------------------------------------
# Loss / training / forecast-NLL evaluation
# ---------------------------------------------------------------------------


def pas_n1_loss(model: PASN1Model, batch: LearnerBatch) -> torch.Tensor:
    theta = model(batch.item_ids, batch.responses, batch.kc_ids, batch.kc_mask)
    logits = theta - batch.b_j
    y = batch.responses.to(logits.dtype)
    per_pos = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = batch.seq_mask.to(per_pos.dtype)
    return (per_pos * mask).sum() / mask.sum().clamp(min=1.0)


def pas_n2_loss(model: PASN2Model, batch: SliceBatch) -> torch.Tensor:
    theta = model(batch.item_ids, batch.responses)
    logits = theta - batch.b_j
    y = batch.responses.to(logits.dtype)
    per_pos = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = batch.mask.to(per_pos.dtype)
    return (per_pos * mask).sum() / mask.sum().clamp(min=1.0)


def _index_learner_batch(batch: LearnerBatch, idx: torch.Tensor) -> LearnerBatch:
    return LearnerBatch(
        item_ids=batch.item_ids[idx], responses=batch.responses[idx], kc_ids=batch.kc_ids[idx],
        kc_mask=batch.kc_mask[idx], seq_mask=batch.seq_mask[idx], b_j=batch.b_j[idx],
    )


def _index_slice_batch(batch: SliceBatch, idx: torch.Tensor) -> SliceBatch:
    return SliceBatch(
        item_ids=batch.item_ids[idx], responses=batch.responses[idx],
        mask=batch.mask[idx], b_j=batch.b_j[idx],
    )


def _train(model: nn.Module, batch, cfg: TrackerConfig, loss_fn, index_fn) -> list[float]:
    """Shared training loop for both trackers. ``cfg.batch_size=None`` (or
    ``>= n_rows``) is the exact pre-existing whole-cohort loop; a smaller
    value mini-batches over rows (module docstring note 6)."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    trace: list[float] = []
    n_rows = int(batch.item_ids.size(0))
    full_batch = cfg.batch_size is None or cfg.batch_size >= n_rows

    if full_batch:
        for _ in range(cfg.n_epochs):
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            trace.append(float(loss.item()))
        return trace

    # Seeded shuffle generator, separate from the global torch RNG so the
    # whole-cohort path (and everything downstream of it) is numerically
    # unaffected by this branch existing (mirrors core DeepIRTModel.fit).
    rng = torch.Generator(device="cpu")
    rng.manual_seed(cfg.seed + 1)
    for _ in range(cfg.n_epochs):
        perm = torch.randperm(n_rows, generator=rng)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_rows, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size].to(batch.item_ids.device)
            optimizer.zero_grad()
            loss = loss_fn(model, index_fn(batch, idx))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        trace.append(epoch_loss / n_batches)
    return trace


def train_pas_n1(model: PASN1Model, batch: LearnerBatch, cfg: TrackerConfig) -> list[float]:
    return _train(model, batch, cfg, pas_n1_loss, _index_learner_batch)


def train_pas_n2(model: PASN2Model, batch: SliceBatch, cfg: TrackerConfig) -> list[float]:
    return _train(model, batch, cfg, pas_n2_loss, _index_slice_batch)


@torch.no_grad()
def forecast_nll(model: nn.Module, batch, kind: str) -> float:
    """Causal next-step forecast NLL on a held-out batch (design's neural
    80/20 regime: "causal next-step forecast NLL on the held-out 20% of
    learners" -- no re-fitting happens here, `state_for_prediction`'s
    causal alignment already makes every position a genuine forecast)."""
    if kind == "pas_n1":
        return float(pas_n1_loss(model, batch).item())
    if kind == "pas_n2":
        return float(pas_n2_loss(model, batch).item())
    raise ValueError(f"unknown tracker kind {kind!r}")


# ---------------------------------------------------------------------------
# E-P3: tracker displacement
# ---------------------------------------------------------------------------


def displacement(theta_series: np.ndarray) -> float:
    """E-P3 (design section 1): mean of the tracker's causal theta over
    the last quarter of the slice minus the mean over the first quarter.
    Free-signed, shape-free. Requires at least 4 positions (the
    population-curve floor); returns ``nan`` otherwise."""
    T = len(theta_series)
    if T < 4:
        return float("nan")
    q = max(1, T // 4)
    return float(np.mean(theta_series[-q:]) - np.mean(theta_series[:q]))


__all__ = [
    "TrackerConfig",
    "PASN1Model",
    "PASN2Model",
    "freeze_encoder",
    "build_tracker",
    "LearnerBatch",
    "build_learner_batch",
    "SliceBatch",
    "build_slice_batch",
    "pas_n1_loss",
    "pas_n2_loss",
    "train_pas_n1",
    "train_pas_n2",
    "forecast_nll",
    "displacement",
]
