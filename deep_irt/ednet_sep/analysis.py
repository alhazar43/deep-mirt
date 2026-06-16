"""
analysis.py -- Metrics for the three separability research questions.

All metrics are RANK / VARIANCE-SHARE quantities, not prediction accuracy
(the thesis is judged on invariance, not AUC).

RQ5b  Variance decomposition (G-theory).  Decompose per-(person, part)
      ability into person, part(instrument) and person x part components.
      Separability index = person-variance share.  Computed on a BALANCED
      person x part grid (ANOVA expected-mean-square estimators) for a valid
      decomposition; an unbalanced robustness value is reported alongside.

RQ5a  Instrument-leakage probe.  After residualising the per-part learner
      representation on its scalar ability (ability-matching), can a classifier
      still predict which part it came from?  Report macro one-vs-rest AUC vs
      the ability-matched chance level (0.5 per OVR split).

RQ4   Cross-part alignment.  For each part pair, Pearson/Spearman of ability_A
      vs ability_B on shared learners, attenuation-corrected by the per-part
      split-half reliabilities.  Tested against the construct-distance gradient
      (within-section vs cross-section TOEIC pairs).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from deep_irt.ednet_sep.partmap import PARTS, same_section, part_section


# ---------------------------------------------------------------------------
# RQ5b -- variance decomposition (G-theory, two-way random effects)
# ---------------------------------------------------------------------------

def variance_decomposition(
    ability: Dict[int, Dict[int, float]],
    parts: Tuple[int, ...] = PARTS,
) -> dict:
    """Two-way random-effects variance decomposition of ability.

    Parameters
    ----------
    ability : nested dict ``ability[student_id][part] = theta``.
    parts   : the parts (instruments) to include.

    Returns
    -------
    dict with balanced-grid variance components (person, part, interaction),
    the separability index (person share), and an unbalanced robustness value.

    Method (balanced grid).  Select the largest set of learners present on ALL
    chosen parts, then a balanced person x part table is decomposed by the
    standard random-effects ANOVA expected mean squares::

        sigma2_pi = MS_E
        sigma2_p  = (MS_person - MS_E) / n_parts
        sigma2_i  = (MS_part   - MS_E) / n_persons

    Negative estimates are floored at 0 (standard practice).  The separability
    index is sigma2_p / (sigma2_p + sigma2_i + sigma2_pi).
    """
    # Choose the part set that maximises the balanced cell count
    # (persons present on all parts) -- try the full set, then drop the
    # sparsest part if the grid is too thin.
    best = _balanced_grid(ability, list(parts))
    grid, used_parts, used_persons = best

    n_p = len(used_persons)
    n_i = len(used_parts)
    result: dict = {
        "balanced_n_persons": n_p,
        "balanced_n_parts": n_i,
        "balanced_parts": used_parts,
    }
    if n_p < 5 or n_i < 2:
        result.update(
            person_var=float("nan"), part_var=float("nan"),
            interaction_var=float("nan"), separability_index=float("nan"),
            grand_mean=float("nan"),
        )
        return result

    # grid: (n_p, n_i) ability matrix
    grand = grid.mean()
    row_means = grid.mean(axis=1, keepdims=True)   # per person
    col_means = grid.mean(axis=0, keepdims=True)   # per part

    ss_person = n_i * ((row_means - grand) ** 2).sum()
    ss_part = n_p * ((col_means - grand) ** 2).sum()
    ss_total = ((grid - grand) ** 2).sum()
    ss_error = ss_total - ss_person - ss_part

    df_person = n_p - 1
    df_part = n_i - 1
    df_error = df_person * df_part

    ms_person = ss_person / df_person
    ms_part = ss_part / df_part
    ms_error = ss_error / df_error if df_error > 0 else float("nan")

    sigma2_pi = max(ms_error, 0.0)
    sigma2_p = max((ms_person - ms_error) / n_i, 0.0)
    sigma2_i = max((ms_part - ms_error) / n_p, 0.0)
    denom = sigma2_p + sigma2_i + sigma2_pi
    sep_index = sigma2_p / denom if denom > 0 else float("nan")

    result.update(
        grand_mean=float(grand),
        person_var=float(sigma2_p),
        part_var=float(sigma2_i),
        interaction_var=float(sigma2_pi),
        person_var_share=float(sigma2_p / denom) if denom > 0 else float("nan"),
        part_var_share=float(sigma2_i / denom) if denom > 0 else float("nan"),
        interaction_var_share=float(sigma2_pi / denom) if denom > 0 else float("nan"),
        separability_index=float(sep_index),
        ms_person=float(ms_person),
        ms_part=float(ms_part),
        ms_error=float(ms_error),
    )

    # Unbalanced robustness: variance of per-person mean vs variance of
    # per-part mean over ALL available cells (descriptive cross-check).
    result["unbalanced"] = _unbalanced_shares(ability, list(parts))
    return result


def _balanced_grid(
    ability: Dict[int, Dict[int, float]],
    parts: List[int],
    person_floor: int = 300,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Find a balanced person x part grid, preferring MORE parts.

    A decomposition that spans more parts (and ideally both TOEIC sections)
    estimates the instrument main effect and person x part interaction with
    more degrees of freedom and is the stronger separability test, so we prefer
    the widest part set whose balanced grid still has at least ``person_floor``
    learners.  Among part sets of equal width we take the one with the most
    covering persons.  We fall back to fewer parts (then to the cell-maximising
    grid) if the floor cannot be met.

    Greedy over widths: at each width drop the part covered by the fewest
    persons.  Returns (grid, parts, persons).
    """
    parts = sorted(parts)

    def _persons_for(subset: List[int]) -> List[int]:
        return sorted(sid for sid, d in ability.items() if all(p in d for p in subset))

    # Build the greedy chain: full set, drop sparsest, ... down to 2.
    chain: List[List[int]] = []
    candidate = list(parts)
    while len(candidate) >= 2:
        chain.append(list(candidate))
        coverage = {p: sum(1 for d in ability.values() if p in d) for p in candidate}
        candidate.remove(min(coverage, key=coverage.get))

    # Prefer the widest grid meeting the person floor.
    chosen: List[int] = []
    for subset in chain:  # widest first
        persons = _persons_for(subset)
        if len(persons) >= person_floor:
            chosen = subset
            break

    # Fallback: maximise cells (persons x parts).
    if not chosen:
        best_cells = -1
        for subset in chain:
            persons = _persons_for(subset)
            cells = len(persons) * len(subset)
            if len(persons) >= 5 and cells > best_cells:
                best_cells = cells
                chosen = subset

    if not chosen:
        return np.zeros((0, 0)), [], []

    persons = _persons_for(chosen)
    grid = np.array(
        [[ability[s][p] for p in chosen] for s in persons], dtype=np.float64
    )
    return grid, chosen, persons


def _unbalanced_shares(
    ability: Dict[int, Dict[int, float]],
    parts: List[int],
) -> dict:
    """Descriptive variance shares over all available (person, part) cells.

    Not an unbiased EMS decomposition (cells are unbalanced); reported as a
    robustness cross-check on the balanced estimate.
    """
    rows = []
    for sid, d in ability.items():
        for p, v in d.items():
            if p in parts:
                rows.append((sid, p, v))
    if len(rows) < 10:
        return {"n_cells": len(rows)}
    arr = np.array([r[2] for r in rows])
    grand = arr.mean()
    # Person effect
    by_person: Dict[int, List[float]] = {}
    by_part: Dict[int, List[float]] = {}
    for sid, p, v in rows:
        by_person.setdefault(sid, []).append(v)
        by_part.setdefault(p, []).append(v)
    person_means = np.array([np.mean(v) for v in by_person.values()])
    part_means = np.array([np.mean(v) for v in by_part.values()])
    var_total = float(arr.var(ddof=1))
    var_person_means = float(person_means.var(ddof=1)) if len(person_means) > 1 else float("nan")
    var_part_means = float(part_means.var(ddof=1)) if len(part_means) > 1 else float("nan")
    return {
        "n_cells": len(rows),
        "n_persons": len(by_person),
        "var_total": var_total,
        "var_person_means": var_person_means,
        "var_part_means": var_part_means,
        "person_to_part_var_ratio": (
            var_person_means / var_part_means
            if var_part_means and var_part_means > 0 else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# RQ5a -- instrument-leakage probe
# ---------------------------------------------------------------------------

def leakage_probe(
    features: np.ndarray,
    ability: np.ndarray,
    part_labels: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
) -> dict:
    """Can part identity be predicted from the representation BEYOND ability?

    Two probes, both cross-validated multinomial logistic regression:
      * RAW       : predict part from the full representation.
      * RESIDUAL  : predict part from the representation after regressing out
                    the scalar ability (column-wise OLS residuals), i.e. the
                    ability-matched probe.  Above-chance here = the scale leaks
                    instrument identity that ability alone does not explain.

    Parameters
    ----------
    features    : (N, D) per-(learner, part) representation rows.
    ability     : (N,)  scalar ability for each row (the matching covariate).
    part_labels : (N,)  part id (1..7) for each row.
    seed        : RNG seed for CV folds.
    n_splits    : stratified CV folds.

    Returns
    -------
    dict with raw / residual macro-OVR AUC, chance (0.5), accuracy, and the
    majority-class baseline accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    N = features.shape[0]
    classes, counts = np.unique(part_labels, return_counts=True)
    n_classes = len(classes)
    majority_acc = float(counts.max() / N)

    # Residualise each feature column on ability (ability-matching).
    abil = ability.reshape(-1, 1).astype(np.float64)
    abil_design = np.hstack([np.ones((N, 1)), abil])  # intercept + ability
    # OLS hat: beta = (X'X)^-1 X'F ; residual = F - X beta
    XtX_inv = np.linalg.pinv(abil_design.T @ abil_design)
    beta = XtX_inv @ (abil_design.T @ features)
    feat_resid = features - abil_design @ beta

    def _cv_auc(X: np.ndarray) -> Tuple[float, float]:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = part_labels
        oof_proba = np.zeros((N, n_classes))
        oof_pred = np.zeros(N, dtype=np.int64)
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            proba = clf.predict_proba(Xte)
            # Align proba columns to the global class order.
            col = {c: j for j, c in enumerate(clf.classes_)}
            for j, c in enumerate(classes):
                if c in col:
                    oof_proba[te, j] = proba[:, col[c]]
            oof_pred[te] = clf.predict(Xte)
        try:
            auc = roc_auc_score(
                y, oof_proba, multi_class="ovr", average="macro", labels=classes
            )
        except ValueError:
            auc = float("nan")
        acc = accuracy_score(y, oof_pred)
        return float(auc), float(acc)

    raw_auc, raw_acc = _cv_auc(features)
    res_auc, res_acc = _cv_auc(feat_resid)

    return {
        "n_rows": int(N),
        "n_classes": int(n_classes),
        "classes": [int(c) for c in classes],
        "chance_auc": 0.5,
        "majority_acc": majority_acc,
        "raw_auc": raw_auc,
        "raw_acc": raw_acc,
        "residual_auc": res_auc,
        "residual_acc": res_acc,
        "leakage_above_chance": float(res_auc - 0.5),
    }


# ---------------------------------------------------------------------------
# RQ4 -- cross-part alignment with construct-distance gradient
# ---------------------------------------------------------------------------

def spearman_brown(half_corr: float) -> float:
    """Spearman-Brown prophecy: full-length reliability from a split-half corr."""
    if not np.isfinite(half_corr) or half_corr <= -1.0:
        return float("nan")
    return 2.0 * half_corr / (1.0 + half_corr)


def cross_part_alignment(
    ability: Dict[int, Dict[int, float]],
    reliability: Dict[int, float],
    parts: Tuple[int, ...] = PARTS,
    min_pairs: int = 30,
) -> dict:
    """Build the cross-part alignment matrix and the construct-distance gradient.

    Parameters
    ----------
    ability     : ability[student_id][part] = theta.
    reliability : reliability[part] = per-part split-half full reliability.
    parts       : parts to include.
    min_pairs   : minimum shared learners required to report a pair.

    Returns
    -------
    dict with the per-pair raw and attenuation-corrected correlations and the
    within-section vs cross-section means (the gradient test).
    """
    parts = sorted(parts)
    pair_results: List[dict] = []
    for a_i in range(len(parts)):
        for b_i in range(a_i + 1, len(parts)):
            A, B = parts[a_i], parts[b_i]
            shared = [s for s, d in ability.items() if A in d and B in d]
            if len(shared) < min_pairs:
                continue
            x = np.array([ability[s][A] for s in shared])
            y = np.array([ability[s][B] for s in shared])
            r_pear = float(pearsonr(x, y)[0])
            r_spear = float(spearmanr(x, y)[0])
            rel_a = reliability.get(A, float("nan"))
            rel_b = reliability.get(B, float("nan"))
            denom = np.sqrt(rel_a * rel_b) if (
                np.isfinite(rel_a) and np.isfinite(rel_b)
                and rel_a > 0 and rel_b > 0
            ) else float("nan")
            corrected = r_pear / denom if np.isfinite(denom) and denom > 0 else float("nan")
            pair_results.append({
                "part_a": A,
                "part_b": B,
                "n_shared": len(shared),
                "section_a": part_section(A),
                "section_b": part_section(B),
                "same_section": same_section(A, B),
                "pearson": r_pear,
                "spearman": r_spear,
                "rel_a": float(rel_a),
                "rel_b": float(rel_b),
                "attenuation_ceiling": float(denom) if np.isfinite(denom) else float("nan"),
                "corrected": float(corrected) if np.isfinite(corrected) else float("nan"),
            })

    within = [p["pearson"] for p in pair_results if p["same_section"]]
    cross = [p["pearson"] for p in pair_results if not p["same_section"]]
    within_c = [p["corrected"] for p in pair_results
                if p["same_section"] and np.isfinite(p["corrected"])]
    cross_c = [p["corrected"] for p in pair_results
               if not p["same_section"] and np.isfinite(p["corrected"])]

    gradient = {
        "within_section_mean_pearson": float(np.mean(within)) if within else float("nan"),
        "cross_section_mean_pearson": float(np.mean(cross)) if cross else float("nan"),
        "within_section_mean_corrected": float(np.mean(within_c)) if within_c else float("nan"),
        "cross_section_mean_corrected": float(np.mean(cross_c)) if cross_c else float("nan"),
        "gradient_holds_raw": (
            bool(np.mean(within) > np.mean(cross)) if within and cross else None
        ),
        "gradient_gap_raw": (
            float(np.mean(within) - np.mean(cross)) if within and cross else float("nan")
        ),
        "n_within_pairs": len(within),
        "n_cross_pairs": len(cross),
    }
    return {"pairs": pair_results, "gradient": gradient}
