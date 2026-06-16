"""check_judgmentbench.py -- Precondition: inter-format agreement on JudgmentBench.

Does the RUBRIC (graded) score agree with the COMPARATIVE-JUDGMENT (CJ, pairwise)
ranking on the items that have BOTH? If they disagree, JudgmentBench cannot give a
clean positive for the format-agnostic transfer claim (one format is unreliable),
and we record that honestly and move on.

Procedure
---------
1. Load rubric annotations  -> per-output rubric fraction = total_points / max_points.
   Average over repeated rubric annotations of the same output.
2. Load CJ annotations -> pairwise outcomes (winner) among outputs.
3. Fit a Bradley-Terry strength over outputs from the CJ pairs (simple logistic
   MLE via torch on the comparison graph's largest component).
4. The BOTH set = outputs with a rubric fraction AND a BT strength.
   Report Spearman(rubric_fraction, BT_strength) on BOTH, plus N.
5. Sanity references: rubric-fraction vs quality_level_order; CJ win-rate vs
   quality_level_order (the brief's reported 0.048 / 0.379).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

DATA = Path("rl/data/judgmentbench")


def load_rubric():
    """output_id -> (mean rubric fraction, quality_level_order)."""
    frac_by_out = defaultdict(list)
    qlo_by_out = {}
    with open(DATA / "human" / "annotations_rubric.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oid = row["output_id"]
            try:
                tot = float(row["rubric_total_points"])
                mx = float(row["rubric_max_points"])
            except (ValueError, KeyError):
                continue
            if mx <= 0:
                continue
            frac_by_out[oid].append(tot / mx)
            qlo = row.get("output_quality_level_order", "")
            if qlo != "":
                qlo_by_out[oid] = int(qlo)
    rubric = {oid: float(np.mean(v)) for oid, v in frac_by_out.items()}
    return rubric, qlo_by_out


def load_cj():
    """Return (pairs, qlo_by_out). pairs = list of (a_id, b_id, winner_id)."""
    pairs = []
    qlo_by_out = {}
    with open(DATA / "human" / "annotations_comparative_judgment.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = row["option_a_output_id"]
            b = row["option_b_output_id"]
            w = row["preferred_output_id"]
            if not a or not b or not w:
                continue
            if w not in (a, b):
                continue
            pairs.append((a, b, w))
            for oid, key in [(a, "option_a_quality_level_order"),
                             (b, "option_b_quality_level_order")]:
                v = row.get(key, "")
                if v != "":
                    qlo_by_out[oid] = int(v)
    return pairs, qlo_by_out


def largest_component(pairs):
    """Restrict CJ pairs to the largest connected component of the comparison
    graph (BT strength is only identified within a connected component)."""
    adj = defaultdict(set)
    for a, b, _ in pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    best = set()
    for node in adj:
        if node in seen:
            continue
        stack = [node]
        comp = set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            seen.add(u)
            stack.extend(adj[u] - comp)
        if len(comp) > len(best):
            best = comp
    kept = [(a, b, w) for a, b, w in pairs if a in best and b in best]
    return kept, sorted(best)


def fit_bt(pairs, outputs, n_iter=2000, lr=0.1):
    """Bradley-Terry MLE strengths via torch. outputs = ordered id list.
    Returns dict output_id -> strength (mean-centered)."""
    idx = {oid: k for k, oid in enumerate(outputs)}
    a_idx = torch.tensor([idx[a] for a, b, w in pairs], dtype=torch.long)
    b_idx = torch.tensor([idx[b] for a, b, w in pairs], dtype=torch.long)
    # outcome 1 if a wins
    y = torch.tensor([1.0 if w == a else 0.0 for a, b, w in pairs])
    s = torch.zeros(len(outputs), requires_grad=True)
    opt = torch.optim.Adam([s], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        logit = s[a_idx] - s[b_idx]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        loss = loss + 1e-3 * (s ** 2).mean()  # tiny ridge for identifiability
        loss.backward()
        opt.step()
    s = s.detach()
    s = s - s.mean()
    return {oid: float(s[k]) for k, oid in enumerate(outputs)}


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(sp_stats.spearmanr(x, y)[0])


def cj_winrate_vs_qlo(pairs, qlo):
    """Fraction of non-tied-by-quality CJ pairs whose winner has the higher
    quality_level_order. The brief reports ~0.379."""
    agree = total = 0
    for a, b, w in pairs:
        if a not in qlo or b not in qlo:
            continue
        if qlo[a] == qlo[b]:
            continue
        higher = a if qlo[a] > qlo[b] else b
        agree += int(w == higher)
        total += 1
    return (agree / total if total else float("nan")), total


def per_output_winrate(pairs):
    """output_id -> (wins / appearances). Defined for ANY output in CJ, no
    connectivity requirement. The brief's CJ win-rate metric."""
    wins = defaultdict(int)
    appear = defaultdict(int)
    for a, b, w in pairs:
        appear[a] += 1
        appear[b] += 1
        wins[w] += 1
    return {o: wins[o] / appear[o] for o in appear}


def main():
    rubric, qlo_r = load_rubric()
    pairs, qlo_cj = load_cj()
    qlo = {**qlo_cj, **qlo_r}

    print(f"rubric outputs: {len(rubric)}")
    print(f"CJ pairs (valid): {len(pairs)}")

    kept, comp_outputs = largest_component(pairs)
    print(f"CJ largest component: {len(comp_outputs)} outputs, {len(kept)} pairs")

    # --- Approach 1: global BT on the largest connected component ---
    bt = fit_bt(kept, comp_outputs)
    both_bt = sorted(set(rubric) & set(bt))
    rho_bt = spearman([rubric[o] for o in both_bt], [bt[o] for o in both_bt])

    # --- Approach 2: per-output CJ win-rate on the FULL overlap (the brief's
    #     357 BOTH; no connectivity requirement). This is the honest direct
    #     agreement when the comparison graph is fragmented. ---
    wr = per_output_winrate(pairs)
    both_wr = sorted(set(rubric) & set(wr))
    rub_v = [rubric[o] for o in both_wr]
    wr_v = [wr[o] for o in both_wr]
    rho_wr = spearman(rub_v, wr_v)

    # Sanity refs on the win-rate overlap
    qlo_both = [qlo.get(o, np.nan) for o in both_wr]
    has = [i for i, q in enumerate(qlo_both) if not np.isnan(q)]
    rho_rub_qlo = spearman([rub_v[i] for i in has], [qlo_both[i] for i in has])
    rho_wr_qlo = spearman([wr_v[i] for i in has], [qlo_both[i] for i in has])
    cj_wr, cj_n = cj_winrate_vs_qlo(pairs, qlo)

    print("\n" + "=" * 64)
    print("JUDGMENTBENCH INTER-FORMAT AGREEMENT (the precondition)")
    print("=" * 64)
    print(f"  BOTH overlap (rubric & CJ win-rate): {len(both_wr)} outputs  <- the ~357")
    print(f"  Spearman(rubric_frac, CJ win-rate):  {rho_wr:+.4f}   <- KEY (direct)")
    print(f"  [global-BT, largest comp only N={len(both_bt)}]: "
          f"Spearman={rho_bt:+.4f}")
    print(f"  -- sanity refs (on BOTH overlap) --")
    print(f"  Spearman(rubric_frac, quality_lvl):  {rho_rub_qlo:+.4f}  (brief ~0.048)")
    print(f"  Spearman(CJ win-rate, quality_lvl):  {rho_wr_qlo:+.4f}")
    print(f"  CJ winrate vs quality_lvl (pairs):   {cj_wr:.4f} (n={cj_n}, brief ~0.379)")
    print("=" * 64)
    key = rho_wr
    verdict = ("AGREE -> usable" if (not np.isnan(key) and key >= 0.4)
               else "DISAGREE -> NOT a clean positive venue")
    print(f"  VERDICT: rubric vs CJ {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
