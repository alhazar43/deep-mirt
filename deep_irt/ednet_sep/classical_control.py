"""classical_control.py -- Encoder-free classical 2PL control for RQ4/RQ5.

The lethal reviewer attack on the cross-instrument result (RQ5/RQ4 in
``run_experiment.py`` + ``anchored_readout.py``) is CIRCULARITY: the per-part
ability that shows ~0.72 cross-part consistency and a within>cross construct
gradient is read against FIXED item parameters that themselves come from the
SAME joint encoder fit.  A psychometrics reviewer rejects the result because the
"independent" item calibration is not independent of the encoder.

This module supplies a control with NO neural encoder anywhere.  On the SAME
EdNet subset (the records produced by ``data.load_records`` with the study's
seed), it fits a classical 2PL PER PART, INDEPENDENTLY -- each part calibrated
on its own binary-correctness responses only, no cross-part information, no
encoder.  Item difficulty/discrimination AND per-learner ability come from a
hand-rolled vectorised 2PL JOINT-MLE on CPU (the brief's sanctioned hand-roll):
girth 0.8.0's ``twopl_mml`` stalls and ``twopl_jml`` returns NaN on EdNet's large
sparse per-part banks (e.g. part 5 = 4233 items x 2705 persons at ~2% density),
where its full-information quadrature backend is intractable.  girth remains
available as a packaged cross-check on the small parts (``method='girth_mml'``).

Binary correctness.  The materialised K=4 ordinal code already encodes
correctness exactly: codes {2,3} are correct (slow/fast), {0,1} incorrect.  So
``correct = (code >= 2)`` recovers the raw binary correctness on the IDENTICAL
(learner, item) events the encoder saw -- no CSV re-read, no re-alignment.  This
binary signal is also INDEPENDENT of the response-time coercion the encoder's
K=4 GPCM consumed, which makes the control's item calibration doubly independent.

Cross-part consistency is INVARIANT to the per-part scale identification.  Each
2PL fit identifies its own scale by an arbitrary mean/SD (the standard N(0,1)
person anchor), but Pearson/Spearman correlation is invariant to a linear
transform of either variable, so NO linking is needed for the consistency check
itself -- the across-learner correlation of (part-A ability, part-B ability) is
already well defined.  Linking is only needed for the SECONDARY item-difficulty
agreement (handled by mean/SD alignment on shared items).

Public API
----------
``build_part_response_matrix(records, part, min_events, min_item_resp)``
    -> (matrix[items x persons], item_local_ids, person_sids) binary 0/1/missing.
``fit_classical_part(matrix)``
    -> dict with item discrimination/difficulty + per-person ability (joint-MLE).
``build_classical_ability_table(records, ...)``
    -> ability[sid][part] = classical theta, plus per-part item-param tables.
``classical_split_half_reliability(records, ...)``
    -> per-part split-half reliability of the classical readout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from deep_irt.ednet_sep.data import LearnerRecord, split_half
from deep_irt.ednet_sep.partmap import PARTS


MISSING_FILL = -99999  # missing-cell marker (joint-MLE mask + girth tag sentinel)


# ---------------------------------------------------------------------------
# Binary correctness from the K=4 ordinal code
# ---------------------------------------------------------------------------

def binary_correct(responses: np.ndarray) -> np.ndarray:
    """Recover raw binary correctness from the K=4 ordinal code.

    The EdNet adapter coercion is ``{0: incorrect_slow, 1: incorrect_fast,
    2: correct_slow, 3: correct_fast}`` so correctness is exactly ``code >= 2``.
    This is the IDENTICAL correctness signal the raw CSV carries, recovered on
    the exact (learner, item) events the encoder consumed, and it does not
    depend on the response-time quadrant used to build the ordinal label.
    """
    return (np.asarray(responses) >= 2).astype(np.int64)


# ---------------------------------------------------------------------------
# Per-part [items x persons] binary response matrix
# ---------------------------------------------------------------------------

def build_part_response_matrix(
    records: List[LearnerRecord],
    part: int,
    min_events: int = 10,
    min_item_resp: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the [n_items x n_persons] binary matrix for one part.

    Only learners with >= ``min_events`` part-p events are columns (matching the
    encoder/anchored eligibility), and only items answered by >= ``min_item_resp``
    of those learners are rows (a 2PL item needs enough responses to calibrate).
    Cells a learner did not answer are filled with ``MISSING_FILL`` for girth's
    ``tag_missing_data``.

    Returns
    -------
    matrix      : (n_items, n_persons) int -- 0/1 valid, MISSING_FILL elsewhere.
    item_ids    : (n_items,) 1-based local item ids (rows).
    person_sids : (n_persons,) student ids (columns).
    """
    # Collect each eligible learner's part-p (item, correct) events.
    per_person: Dict[int, Dict[int, int]] = {}
    item_resp_count: Dict[int, int] = {}
    for r in records:
        sel = (r.parts == part)
        if int(sel.sum()) < min_events:
            continue
        q = r.questions[sel].numpy().astype(np.int64)        # 1-based local ids
        c = binary_correct(r.responses[sel].numpy())
        # If a learner answered the same item twice in this part, keep the LAST
        # (most recent) response -- consistent with the causal final-state read.
        d: Dict[int, int] = {}
        for qi, ci in zip(q.tolist(), c.tolist()):
            d[qi] = ci
        per_person[r.student_id] = d
        for qi in d:
            item_resp_count[qi] = item_resp_count.get(qi, 0) + 1

    keep_items = sorted(i for i, n in item_resp_count.items() if n >= min_item_resp)
    person_sids = np.array(sorted(per_person.keys()), dtype=np.int64)
    if not keep_items or person_sids.size == 0:
        return (np.zeros((0, 0), dtype=np.int64),
                np.zeros(0, dtype=np.int64), person_sids)

    item_ids = np.array(keep_items, dtype=np.int64)
    item_col = {it: j for j, it in enumerate(item_ids)}
    n_items, n_persons = item_ids.size, person_sids.size
    matrix = np.full((n_items, n_persons), MISSING_FILL, dtype=np.int64)
    for col, sid in enumerate(person_sids):
        for qi, ci in per_person[int(sid)].items():
            row = item_col.get(qi)
            if row is not None:
                matrix[row, col] = ci

    # Drop persons left with no kept-item responses (all-missing columns).
    valid_col = (matrix != MISSING_FILL).any(axis=0)
    if not valid_col.all():
        matrix = matrix[:, valid_col]
        person_sids = person_sids[valid_col]
    return matrix, item_ids, person_sids


# ---------------------------------------------------------------------------
# Hand-rolled vectorised 2PL joint-MLE on CPU (the robust primary engine)
# ---------------------------------------------------------------------------

def _twopl_joint_mle(
    matrix: np.ndarray,
    n_steps: int = 400,
    lr: float = 0.05,
    theta_l2: float = 1e-3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hand-rolled vectorised 2PL joint maximum-likelihood on a sparse matrix.

    girth's marginal-MLE backend integrates the full GRM likelihood over a
    quadrature grid and builds dense [items x persons x quad] arrays, which
    stalls on EdNet's large sparse per-part banks (e.g. 4233 items x 2705
    persons at ~2% density).  The brief explicitly sanctions a hand-rolled CPU
    2PL joint-MLE; this is it, entirely classical and encoder-free.

    Model: P(correct) = sigmoid(a_i * (theta_p - b_i)), a_i = softplus(raw_a_i)
    (kept positive, the standard 2PL constraint).  We maximise the masked
    Bernoulli log-likelihood over (raw_a, b, theta) with Adam, applying the
    standard scale identification (theta standardised to mean 0, sd 1 after each
    step) so the otherwise-unidentified (a, theta, b) orbit is pinned.  A tiny L2
    on theta keeps it finite for all-correct / all-incorrect persons (the JML
    analogue of the anchored MAP prior).

    Parameters
    ----------
    matrix : (n_items, n_persons) int, 0/1 valid and ``MISSING_FILL`` elsewhere.

    Returns
    -------
    a     : (n_items,) discrimination (> 0).
    b     : (n_items,) difficulty on the standardised person scale.
    theta : (n_persons,) ability, standardised (mean 0, sd 1).
    """
    import torch

    mask_np = (matrix != MISSING_FILL).astype(np.float64)
    resp_np = np.where(matrix == MISSING_FILL, 0, matrix).astype(np.float64)
    n_items, n_persons = matrix.shape

    torch.manual_seed(seed)
    resp = torch.tensor(resp_np, dtype=torch.float64)
    mask = torch.tensor(mask_np, dtype=torch.float64)

    # Sensible inits: b at the item logit, theta at the person logit.
    item_p = (resp * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    person_p = (resp * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    b0 = torch.log(item_p.clamp(0.05, 0.95) / (1 - item_p.clamp(0.05, 0.95)))
    th0 = torch.log(person_p.clamp(0.05, 0.95) / (1 - person_p.clamp(0.05, 0.95)))

    raw_a = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)  # softplus(0)~0.69
    b = (-b0).clone().detach().requires_grad_(True)   # higher item_p -> lower difficulty
    theta = ((th0 - th0.mean()) / th0.std().clamp_min(1e-3)).detach().requires_grad_(True)

    opt = torch.optim.Adam([raw_a, b, theta], lr=lr)
    softplus = torch.nn.functional.softplus
    for _ in range(n_steps):
        opt.zero_grad()
        a = softplus(raw_a)                                  # (n_items,)
        z = a.unsqueeze(1) * (theta.unsqueeze(0) - b.unsqueeze(1))  # (I, P)
        # masked Bernoulli NLL (numerically stable via logsigmoid)
        log_p = torch.nn.functional.logsigmoid(z)
        log_q = torch.nn.functional.logsigmoid(-z)
        ll = resp * log_p + (1 - resp) * log_q
        nll = -(ll * mask).sum()
        nll = nll + theta_l2 * (theta * theta).sum()
        nll.backward()
        opt.step()
        # Scale identification: standardise theta after the step.
        with torch.no_grad():
            mu, sd = theta.mean(), theta.std().clamp_min(1e-6)
            theta.sub_(mu).div_(sd)

    a = softplus(raw_a).detach().numpy().astype(np.float64)
    b_out = b.detach().numpy().astype(np.float64)
    theta_out = theta.detach().numpy().astype(np.float64)
    return a, b_out, theta_out


# ---------------------------------------------------------------------------
# Classical 2PL fit + ability
# ---------------------------------------------------------------------------

def fit_classical_part(
    matrix: np.ndarray,
    item_ids: np.ndarray,
    person_sids: np.ndarray,
    max_iteration: int = 200,
    method: str = "joint_mle",
) -> dict:
    """Fit a classical 2PL on one part's binary matrix and estimate abilities.

    ``method='joint_mle'`` (default) uses the hand-rolled vectorised CPU 2PL
    joint-MLE -- fast and robust on EdNet's large sparse per-part banks, where
    girth's marginal-MLE backend stalls.  ``method='girth_mml'`` uses girth's
    ``twopl_mml`` + EAP (a packaged cross-check, viable only on the small parts).
    Both are fully classical and never touch the encoder.

    Returns a dict with item ``a``/``b`` (aligned to ``item_ids``), per-person
    ``theta`` (aligned to ``person_sids``), and fit metadata.  Returns NaNs if the
    matrix is too small to calibrate.
    """
    n_items, n_persons = matrix.shape
    out: dict = {
        "n_items": int(n_items),
        "n_persons": int(n_persons),
        "item_ids": item_ids,
        "person_sids": person_sids,
        "method": method,
    }
    if n_items < 3 or n_persons < 20:
        out.update(a=np.full(n_items, np.nan), b=np.full(n_items, np.nan),
                   theta=np.full(n_persons, np.nan), ok=False)
        return out

    if method == "girth_mml":
        from girth import twopl_mml, ability_eap, tag_missing_data
        data = tag_missing_data(matrix.astype(float), [0, 1]).astype(int)
        est = twopl_mml(data, options={"max_iteration": max_iteration})
        a = np.asarray(est["Discrimination"], dtype=np.float64)
        b = np.asarray(est["Difficulty"], dtype=np.float64)
        theta = np.asarray(ability_eap(data, b, a), dtype=np.float64)
    else:
        a, b, theta = _twopl_joint_mle(matrix)
    out.update(a=a, b=b, theta=theta, ok=True)
    return out


# ---------------------------------------------------------------------------
# Per-part ability table over all parts (the encoder-free analogue)
# ---------------------------------------------------------------------------

def build_classical_ability_table(
    records: List[LearnerRecord],
    min_events: int = 10,
    min_item_resp: int = 10,
    parts: Tuple[int, ...] = PARTS,
    max_iteration: int = 200,
    verbose: bool = False,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, dict]]:
    """Independent classical 2PL per part -> ability[sid][part] + item tables.

    Each part is calibrated on its OWN responses only (no cross-part info, no
    encoder).  Returns the same nested-dict ability shape the existing RQ4/RQ5b
    analyses consume, plus a per-part dict of fitted item params for the
    secondary difficulty-agreement check.
    """
    ability: Dict[int, Dict[int, float]] = {}
    item_tables: Dict[int, dict] = {}
    for part in parts:
        matrix, item_ids, sids = build_part_response_matrix(
            records, part, min_events=min_events, min_item_resp=min_item_resp
        )
        fit = fit_classical_part(matrix, item_ids, sids, max_iteration=max_iteration)
        item_tables[part] = fit
        if not fit.get("ok", False):
            if verbose:
                print(f"  part {part}: too small to calibrate "
                      f"({fit['n_items']} items x {fit['n_persons']} persons)")
            continue
        theta = fit["theta"]
        for j, sid in enumerate(fit["person_sids"]):
            if np.isfinite(theta[j]):
                ability.setdefault(int(sid), {})[part] = float(theta[j])
        if verbose:
            print(f"  part {part}: {fit['n_items']} items x {fit['n_persons']} "
                  f"persons calibrated")
    return ability, item_tables


# ---------------------------------------------------------------------------
# Classical split-half reliability (the attenuation ceiling, encoder-free)
# ---------------------------------------------------------------------------

def _fixed_item_theta(
    resp_rows: np.ndarray,      # (P, n_items) 0/1/MISSING_FILL
    a: np.ndarray,              # (n_items,)
    b: np.ndarray,              # (n_items,)
    prior_sd: float = 1.0,
    n_steps: int = 200,
    lr: float = 0.2,
) -> np.ndarray:
    """MAP theta per person under FIXED 2PL item params (vectorised, CPU).

    The per-person 2PL log-likelihood is maximised over scalar theta with an
    N(0, prior_sd^2) prior (the standard bounded person estimate; keeps theta
    finite for all-correct / all-incorrect halves).  Used for the split-half
    reliability, where the item params are held fixed at the part's calibrated
    values and only the responses change between halves.
    """
    import torch

    mask = torch.tensor((resp_rows != MISSING_FILL).astype(np.float64))
    resp = torch.tensor(np.where(resp_rows == MISSING_FILL, 0, resp_rows).astype(np.float64))
    a_t = torch.tensor(np.asarray(a, dtype=np.float64))
    b_t = torch.tensor(np.asarray(b, dtype=np.float64))
    P = resp.shape[0]
    theta = torch.zeros(P, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    inv_2var = 1.0 / (2.0 * prior_sd * prior_sd)
    for _ in range(n_steps):
        opt.zero_grad()
        z = a_t.unsqueeze(0) * (theta.unsqueeze(1) - b_t.unsqueeze(0))   # (P, I)
        log_p = torch.nn.functional.logsigmoid(z)
        log_q = torch.nn.functional.logsigmoid(-z)
        ll = (resp * log_p + (1 - resp) * log_q) * mask
        nll = -ll.sum() + inv_2var * (theta * theta).sum()
        nll.backward()
        opt.step()
    return theta.detach().numpy().astype(np.float64)


def classical_split_half_reliability(
    records: List[LearnerRecord],
    item_tables: Dict[int, dict],
    min_events: int = 10,
    seed: int = 0,
    parts: Tuple[int, ...] = PARTS,
    min_pairs: int = 30,
) -> Dict[int, dict]:
    """Per-part split-half reliability of the CLASSICAL readout.

    Splits each eligible learner's part-p events into two interleaved halves
    (the same ``split_half`` used by the encoder/anchored reliabilities),
    estimates each half's ability by fixed-item MAP against the SAME per-part
    calibrated 2PL item params (so the only change between halves is which
    responses are seen), correlates the two ability vectors across learners, and
    Spearman-Browns to full length.  This is the classical attenuation ceiling,
    computed exactly as the encoder/anchored ceilings are.
    """
    from scipy.stats import pearsonr
    from deep_irt.ednet_sep.analysis import spearman_brown

    out: Dict[int, dict] = {}
    for part in parts:
        fit = item_tables.get(part, {})
        if not fit.get("ok", False):
            out[part] = {"n": 0, "split_half_r": float("nan"),
                         "reliability": float("nan")}
            continue
        item_ids = fit["item_ids"]
        a, b = fit["a"], fit["b"]
        item_col = {int(it): j for j, it in enumerate(item_ids)}
        n_items = item_ids.size

        # Build per-learner half-A / half-B response vectors over the part's
        # calibrated items.
        sids: List[int] = []
        rows_a: List[np.ndarray] = []
        rows_b: List[np.ndarray] = []
        for r in records:
            if int((r.parts == part).sum()) < min_events:
                continue
            split = split_half(r, part, seed=seed)
            if split is None:
                continue
            ha, hb = split

            def _vec(half: dict) -> Optional[np.ndarray]:
                v = np.full(n_items, MISSING_FILL, dtype=np.int64)
                q = np.asarray(half["questions"]).astype(np.int64)
                c = binary_correct(np.asarray(half["responses"]))
                any_hit = False
                # Last response wins if an item recurs within the half.
                for qi, ci in zip(q.tolist(), c.tolist()):
                    col = item_col.get(int(qi))
                    if col is not None:
                        v[col] = ci
                        any_hit = True
                return v if any_hit else None

            va, vb = _vec(ha), _vec(hb)
            if va is None or vb is None:
                continue
            sids.append(r.student_id)
            rows_a.append(va)
            rows_b.append(vb)

        if len(sids) < min_pairs:
            out[part] = {"n": len(sids), "split_half_r": float("nan"),
                         "reliability": float("nan")}
            continue

        th_a = _fixed_item_theta(np.array(rows_a), a, b)
        th_b = _fixed_item_theta(np.array(rows_b), a, b)

        finite = np.isfinite(th_a) & np.isfinite(th_b)
        th_a, th_b = th_a[finite], th_b[finite]
        if th_a.size < min_pairs or np.std(th_a) < 1e-9 or np.std(th_b) < 1e-9:
            r_half = float("nan")
        else:
            r_half = float(pearsonr(th_a, th_b)[0])
        out[part] = {
            "n": int(th_a.size),
            "split_half_r": r_half,
            "reliability": float(spearman_brown(r_half)),
        }
    return out


# ---------------------------------------------------------------------------
# Secondary: classical-vs-encoder item-difficulty agreement (shared items)
# ---------------------------------------------------------------------------

def classical_vs_encoder_difficulty(
    item_tables: Dict[int, dict],
    enc_b_table: np.ndarray,
    parts: Tuple[int, ...] = PARTS,
    min_shared: int = 20,
) -> dict:
    """Correlate classical per-part item difficulty with the encoder's anchored b.

    The encoder/anchored difficulty ``enc_b_table`` is the joint model's
    per-item GPCM step thresholds (B, K-1); its scalar "difficulty" is the mean
    threshold (the location).  We link per part by aligning the classical b to
    the encoder b on the shared items via mean/SD (the difficulties live on
    arbitrary per-fit scales), then report Pearson/Spearman.  r > ~0.8 means the
    encoder-free and encoder calibrations place items at the same difficulties --
    independent corroboration that the scales agree.

    ``enc_b_table`` is indexed by 0-based local item id (the anchored readout
    convention); classical ``item_ids`` are 1-based, so we subtract 1.
    """
    from scipy.stats import pearsonr, spearmanr

    enc_loc = enc_b_table.mean(axis=1) if enc_b_table.ndim == 2 else enc_b_table
    per_part: Dict[int, dict] = {}
    all_x: List[float] = []
    all_y: List[float] = []
    for part in parts:
        fit = item_tables.get(part, {})
        if not fit.get("ok", False):
            continue
        item0 = fit["item_ids"].astype(np.int64) - 1   # 1-based -> 0-based
        cb = np.asarray(fit["b"], dtype=np.float64)
        eb = enc_loc[item0]
        good = np.isfinite(cb) & np.isfinite(eb)
        if int(good.sum()) < min_shared:
            continue
        cbg, ebg = cb[good], eb[good]
        # Standardise classical b onto the encoder b scale (mean/SD linking).
        if np.std(cbg) > 1e-9:
            cb_lk = (cbg - cbg.mean()) / cbg.std() * ebg.std() + ebg.mean()
        else:
            cb_lk = cbg
        r_p = float(pearsonr(cbg, ebg)[0])
        r_s = float(spearmanr(cbg, ebg)[0])
        per_part[part] = {
            "n_shared": int(good.sum()),
            "pearson": r_p,
            "spearman": r_s,
        }
        all_x.extend(cb_lk.tolist())
        all_y.extend(ebg.tolist())

    pooled = {}
    if len(all_x) >= min_shared:
        pooled = {
            "n_items": len(all_x),
            "pearson": float(pearsonr(all_x, all_y)[0]),
            "spearman": float(spearmanr(all_x, all_y)[0]),
        }
    return {"per_part": per_part, "pooled_linked": pooled}
