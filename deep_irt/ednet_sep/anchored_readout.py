"""
anchored_readout.py -- Alternative per-part ability readout for the verification.

The headline of the main study (``run_experiment.py``) reads a per-part ability
by running the FROZEN LSTM encoder over each learner's part-restricted
SUBSEQUENCE.  That subsequence is slightly OUT OF DISTRIBUTION: the encoder was
trained on full multi-part sequences, so a part-only history is a regime it
never saw, which can SYSTEMATICALLY UNDERSTATE the true cross-part consistency.
That single confound is what this module rules out.

Alternative readout (anchored / fixed-parameter person estimation).  Instead of
the encoder, we FIX the item parameters (the joint model's recovered per-item
GPCM ``a``/``b``) and estimate each learner's per-part ability by direct
fixed-parameter person estimation: theta_hat = argmax_theta  log P(responses to
this part's items | theta, fixed a, b)  -- with an optional N(0, prior_sd^2)
prior (MAP) for numerical stability when a learner saturates a part.  This is
exactly the inverse of ``deep_irt.core.anchor.anchored_extend`` (which fixes
theta and fits item embeddings); here we fix the items and fit theta.  It uses
ALL of a learner's part-p responses and never runs the encoder OOD.

Why this is the thesis's anchoring mechanism.  Anchoring (RQ2 of the deep_irt
pipeline) stabilises the scale by holding one side fixed.  Reading per-part
ability against fixed, jointly-calibrated item parameters is the anchored
estimate of a learner's standing on each instrument; if anchoring also improves
cross-instrument consistency, that is direct evidence the identifiability
mechanism does double duty.

Public API
----------
``recover_fixed_item_params(model, n_questions, device)``
    -> (a, b) numpy arrays indexed by 0-based local item id, from the joint model.
``map_theta_estimate(responses, a, b, prior_sd, ...)``
    -> scalar MAP/MLE theta for one response set under fixed (a, b).
``anchored_part_ability(records, a, b, part, min_events, prior_sd, device)``
    -> (theta, student_ids) over all learners with >= min_events part-p events.
``anchored_split_half_reliability(records, a, b, part, prior_sd, seed, device)``
    -> split-half reliability of the ANCHORED readout (diagnostic only).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from deep_irt.ednet_sep.data import LearnerRecord, split_half
from deep_irt.ednet_sep.partmap import PARTS


# ---------------------------------------------------------------------------
# Fixed item parameters from the joint model
# ---------------------------------------------------------------------------

def recover_fixed_item_params(
    model,
    n_questions: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover the joint model's per-item GPCM (a, b), indexed by 0-based id.

    The joint ``DeepIRTModel`` maps each item embedding to ``a = softplus(...)``
    (discrimination) and ``b`` (K-1 step thresholds), sorted.  We materialise the
    full table once so the per-part MAP estimator can index it directly.

    Parameters
    ----------
    model       : fitted DeepIRTModel (GPCM decoder).
    n_questions : number of items B (table will have B rows).
    device      : device for the recovery forward pass.

    Returns
    -------
    a : (B,)      discrimination per 0-based local item id.
    b : (B, K-1)  sorted step thresholds per 0-based local item id.
    """
    item_ids = torch.arange(n_questions, device=device)
    params = model.recover_item_params(item_ids=item_ids)  # sorted b, a>0
    a = np.asarray(params["a"], dtype=np.float64).reshape(-1)
    b = np.asarray(params["b"], dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    return a, b


# ---------------------------------------------------------------------------
# GPCM log-likelihood as a function of theta (fixed item params)
# ---------------------------------------------------------------------------

def _gpcm_logprob_theta(
    theta: Tensor,         # (P,)        candidate thetas (per learner)
    a: Tensor,             # (P, n_i)    discrimination of each answered item
    b: Tensor,             # (P, n_i, K-1) thresholds of each answered item
    responses: Tensor,     # (P, n_i)    observed responses {0..K-1}
    item_mask: Tensor,     # (P, n_i)    True where the item is real (not pad)
) -> Tensor:
    """Per-learner GPCM log-likelihood summed over that learner's items.

    Replicates ``GPCMDecoder.log_probs`` (psi_0 = 0, psi_k = cumsum a*(theta-b))
    but vectorised over a batch of learners that each have a different number of
    items (padded to ``n_i`` and masked).  Returns the summed log P(responses |
    theta) for each learner; theta is the only free quantity.
    """
    K = b.shape[-1] + 1
    th = theta.view(-1, 1, 1)                       # (P, 1, 1)
    diff = a.unsqueeze(-1) * (th - b)               # (P, n_i, K-1)
    psi_pos = torch.cumsum(diff, dim=-1)            # (P, n_i, K-1)
    zeros = torch.zeros(*psi_pos.shape[:-1], 1, device=theta.device, dtype=psi_pos.dtype)
    psi = torch.cat([zeros, psi_pos], dim=-1)       # (P, n_i, K)
    logp = torch.log_softmax(psi, dim=-1)           # (P, n_i, K)
    # Gather the log-prob of the observed response for each item.
    r = responses.clamp(0, K - 1).unsqueeze(-1)     # (P, n_i, 1)
    ll_item = logp.gather(-1, r).squeeze(-1)        # (P, n_i)
    ll_item = ll_item * item_mask                   # zero out padded items
    return ll_item.sum(dim=-1)                      # (P,)


def map_theta_batch(
    a: Tensor,             # (P, n_i)
    b: Tensor,             # (P, n_i, K-1)
    responses: Tensor,     # (P, n_i)
    item_mask: Tensor,     # (P, n_i)
    prior_sd: float = 1.0,
    n_steps: int = 200,
    lr: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """MAP theta for a batch of learners under fixed item params (Adam).

    Objective per learner::

        theta_hat = argmax_theta  [ sum_i log P(r_i | theta, a_i, b_i)
                                    - theta^2 / (2 prior_sd^2) ]

    The N(0, prior_sd^2) prior is the standard population anchor for fixed-item
    person estimation (EAP/MAP); it keeps theta finite for learners who answer a
    part's items all-correct or all-incorrect (the MLE would diverge to +/-inf).
    Set ``prior_sd=inf`` to recover the pure MLE.

    All learners share one Adam optimiser over the (P,) theta vector; the per-
    learner objectives are independent so this is just a vectorised 1-D
    optimisation, which converges in well under 200 steps for GPCM.

    Returns
    -------
    theta_hat : (P,) MAP ability estimates (detached, on cpu-or-device tensor).
    """
    P = a.shape[0]
    theta = torch.zeros(P, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    use_prior = np.isfinite(prior_sd) and prior_sd > 0
    inv_2var = 1.0 / (2.0 * prior_sd * prior_sd) if use_prior else 0.0

    for _ in range(n_steps):
        opt.zero_grad()
        ll = _gpcm_logprob_theta(theta, a, b, responses, item_mask)  # (P,)
        neg = -ll.sum()
        if use_prior:
            neg = neg + inv_2var * (theta * theta).sum()
        neg.backward()
        opt.step()
    return theta.detach()


# ---------------------------------------------------------------------------
# Pad a list of per-learner response sets into a batch
# ---------------------------------------------------------------------------

def _pad_part_responses(
    items: List[Dict],
    a_table: np.ndarray,
    b_table: np.ndarray,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray]:
    """Pad a list of {questions(1-based), responses} into batched fixed-param tensors.

    Looks up each item's fixed (a, b) from the joint-model tables and packs them
    into rectangular (P, n_i_max, ...) tensors with a validity mask.

    Returns (a, b, responses, item_mask, student_ids).
    """
    P = len(items)
    lens = [len(it["questions"]) for it in items]
    n_max = max(lens)
    K1 = b_table.shape[1]

    a = np.zeros((P, n_max), dtype=np.float64)
    b = np.zeros((P, n_max, K1), dtype=np.float64)
    resp = np.zeros((P, n_max), dtype=np.int64)
    mask = np.zeros((P, n_max), dtype=np.float64)
    sids = np.zeros(P, dtype=np.int64)

    for p, it in enumerate(items):
        q0 = np.asarray(it["questions"]).astype(np.int64) - 1   # 1-based -> 0-based
        r = np.asarray(it["responses"]).astype(np.int64)
        L = q0.shape[0]
        a[p, :L] = a_table[q0]
        b[p, :L, :] = b_table[q0]
        resp[p, :L] = r
        mask[p, :L] = 1.0
        sids[p] = int(it["student_id"])

    return (
        torch.tensor(a, dtype=torch.float32, device=device),
        torch.tensor(b, dtype=torch.float32, device=device),
        torch.tensor(resp, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        sids,
    )


# ---------------------------------------------------------------------------
# Per-part anchored ability over all learners
# ---------------------------------------------------------------------------

def anchored_part_ability(
    records: List[LearnerRecord],
    a_table: np.ndarray,
    b_table: np.ndarray,
    part: int,
    min_events: int = 10,
    prior_sd: float = 1.0,
    n_steps: int = 200,
    lr: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Anchored (fixed-parameter MAP) per-part ability for every eligible learner.

    For each learner with >= ``min_events`` events on ``part``, estimate theta by
    MAP against the fixed item params of THAT learner's part-p items.  Uses all
    of the learner's part-p responses; never invokes the encoder.

    Returns
    -------
    theta : (M,) anchored ability on the fixed-item scale.
    sids  : (M,) student ids aligned with theta.
    """
    items: List[Dict] = []
    for r in records:
        sel = (r.parts == part)
        if int(sel.sum()) < min_events:
            continue
        items.append({
            "student_id": r.student_id,
            "questions": r.questions[sel].clone(),
            "responses": r.responses[sel].clone(),
        })
    if not items:
        return np.zeros(0), np.zeros(0, dtype=np.int64)

    a, b, resp, mask, sids = _pad_part_responses(items, a_table, b_table, device)
    theta = map_theta_batch(
        a, b, resp, mask, prior_sd=prior_sd, n_steps=n_steps, lr=lr, device=device
    )
    return theta.cpu().numpy().astype(np.float64), sids


def build_anchored_ability_table(
    records: List[LearnerRecord],
    a_table: np.ndarray,
    b_table: np.ndarray,
    min_events: int = 10,
    prior_sd: float = 1.0,
    n_steps: int = 200,
    lr: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, Dict[int, float]]:
    """Build ``ability[student_id][part] = anchored_theta`` over all parts.

    Mirrors ``run_experiment.build_ability_tables`` but with the anchored
    fixed-parameter readout instead of the encoder.  Returns the same nested-dict
    shape so the existing RQ4 / RQ5b analyses can consume it unchanged.
    """
    ability: Dict[int, Dict[int, float]] = {}
    for part in PARTS:
        theta, sids = anchored_part_ability(
            records, a_table, b_table, part,
            min_events=min_events, prior_sd=prior_sd,
            n_steps=n_steps, lr=lr, device=device,
        )
        for j, sid in enumerate(sids):
            ability.setdefault(int(sid), {})[part] = float(theta[j])
    return ability


# ---------------------------------------------------------------------------
# Split-half reliability of the ANCHORED readout (diagnostic)
# ---------------------------------------------------------------------------

def anchored_split_half_reliability(
    records: List[LearnerRecord],
    a_table: np.ndarray,
    b_table: np.ndarray,
    prior_sd: float = 1.0,
    n_steps: int = 200,
    lr: float = 0.2,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    min_pairs: int = 30,
) -> Dict[int, dict]:
    """Per-part split-half reliability under the ANCHORED readout.

    Reported as a diagnostic only: the primary verification holds the
    attenuation ceilings FIXED to the encoder-readout reliabilities so the two
    readouts are corrected against the same denominator (a fair comparison).
    This anchored reliability tells us whether the fixed-parameter estimate is
    itself noisier or cleaner than the encoder estimate.
    """
    from scipy.stats import pearsonr
    from deep_irt.ednet_sep.analysis import spearman_brown

    out: Dict[int, dict] = {}
    for part in PARTS:
        halves_a: List[Dict] = []
        halves_b: List[Dict] = []
        for r in records:
            split = split_half(r, part, seed=seed)
            if split is None:
                continue
            ha, hb = split
            if len(ha["questions"]) == 0 or len(hb["questions"]) == 0:
                continue
            halves_a.append(ha)
            halves_b.append(hb)
        if len(halves_a) < min_pairs:
            out[part] = {"n": len(halves_a), "split_half_r": float("nan"),
                         "reliability": float("nan")}
            continue
        a_a, b_a, r_a, m_a, sid_a = _pad_part_responses(halves_a, a_table, b_table, device)
        a_b, b_b, r_b, m_b, sid_b = _pad_part_responses(halves_b, a_table, b_table, device)
        th_a = map_theta_batch(a_a, b_a, r_a, m_a, prior_sd=prior_sd,
                               n_steps=n_steps, lr=lr, device=device).cpu().numpy()
        th_b = map_theta_batch(a_b, b_b, r_b, m_b, prior_sd=prior_sd,
                               n_steps=n_steps, lr=lr, device=device).cpu().numpy()
        map_b = {int(s): th_b[i] for i, s in enumerate(sid_b)}
        xs, ys = [], []
        for i, s in enumerate(sid_a):
            if int(s) in map_b:
                xs.append(th_a[i]); ys.append(map_b[int(s)])
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < min_pairs or np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            r_half = float("nan")
        else:
            r_half = float(pearsonr(xs, ys)[0])
        out[part] = {
            "n": len(xs),
            "split_half_r": r_half,
            "reliability": float(spearman_brown(r_half)),
        }
    return out
