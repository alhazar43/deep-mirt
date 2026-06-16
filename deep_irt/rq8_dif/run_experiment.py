"""run_experiment.py -- RQ8 INVARIANCE / DIF, with a built-in placebo control.

Question (real_data_agenda.md:82): do anchor-linked item parameters REPLICATE
across learner subpopulations?  I.e. is the learned item scale SUBPOPULATION-
INVARIANT (no differential item functioning, DIF), or are item difficulties
population-specific artifacts?  This is a standard psychometric validity check
that inoculates against "your item parameters are population-specific."

Design (one raw EdNet behaviour, binary K=2 / 2PL, shared item set).
 * Split the LEARNERS three ways and fit item difficulty INDEPENDENTLY per group:
     1. RANDOM-HALF PLACEBO (the control, run FIRST): 50/50 random.  Item params
        SHOULD replicate -- the NO-DIF baseline AND the reliability floor.
     2. ABILITY: top vs bottom tercile by mean correctness (middle dropped).
     3. ACTIVITY: top vs bottom tercile by history length.
 * Each group is a DISTINCT-seed ma-irt LSTMGPCM (separate_theta=True); item
   difficulty read from the deterministic item-only GPCM location (the frozen
   single-item param, 2PL b for K=2).
 * LINK both groups onto one metric via mean/sigma linking on the COMMON ANCHOR
   SET (items answered by enough learners in BOTH groups).  Report per split:
     (a) Spearman of item-difficulty rankings (link-FREE),
     (b) mean |Delta b| after linking (DIF magnitude),
     (c) count + identity of the largest-DIF items.

Validation logic (the "validate after").
 The RANDOM-HALF PLACEBO is the CONTROL.  A meaningful split shows REAL DIF only
 if its divergence is MATERIALLY LARGER than the placebo's (which captures pure
 sampling + estimation noise).  Always interpret meaningful splits AGAINST the
 placebo, never in absolute terms:
   DIF(meaningful) ~= DIF(placebo)  -> item params subpopulation-INVARIANT (PASS).
   DIF(meaningful) >> DIF(placebo)  -> real DIF (population-specific on those items).

Synthetic controls (so the real numbers are interpretable):
 * NO-DIF synthetic: one true 2PL latent, same b for everyone.  Ability split
   SHOULD look like the placebo (the metric is not trivially high).
 * DIF synthetic: known DIF injected on a subset of items along the ability axis.
   The ability split MUST flag those items (the metric can DETECT non-invariance).

Run from repo root:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH="rl/src;ma-irt"
    python deep_irt/rq8_dif/run_experiment.py

Env knobs (bounded runs): RQ8_N_LEARNERS, RQ8_EPOCHS, RQ8_MIN_ITEM_RESP,
RQ8_MAX_SEQ, RQ8_MIN_SEQ, RQ8_LR, RQ8_BATCH, RQ8_DMODEL, RQ8_SEED,
RQ8_ANCHOR_MIN (min responses in BOTH groups to enter the anchor set),
RQ8_SYN_LEARNERS, RQ8_SYN_ITEMS, RQ8_SKIP_SYN, RQ8_SKIP_PART.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rl" / "src", REPO_ROOT / "ma-irt"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from deep_irt.rq8_dif.data import (
    load_raw_binary, split_random_half, split_ability, split_activity,
    split_part, matched_random_split, make_synthetic_nodif, make_synthetic_dif,
    RawBinary, GroupSplit,
)
from deep_irt.rq8_dif.model import (
    fit_group, item_difficulty_b, item_seen_counts,
)

OUTPUT_DIR = Path(os.environ.get(
    "RQ8_OUTPUT_DIR", str(Path(__file__).resolve().parent / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_LEARNERS = int(os.environ.get("RQ8_N_LEARNERS", "4000"))
EPOCHS = int(os.environ.get("RQ8_EPOCHS", "80"))
MIN_ITEM_RESP = int(os.environ.get("RQ8_MIN_ITEM_RESP", "50"))
MAX_SEQ = int(os.environ.get("RQ8_MAX_SEQ", "200"))
MIN_SEQ = int(os.environ.get("RQ8_MIN_SEQ", "20"))
LR = float(os.environ.get("RQ8_LR", "0.005"))
BATCH = int(os.environ.get("RQ8_BATCH", "256"))
DMODEL = int(os.environ.get("RQ8_DMODEL", "64"))
SEED = int(os.environ.get("RQ8_SEED", "0"))
ANCHOR_MIN = int(os.environ.get("RQ8_ANCHOR_MIN", "20"))  # min resp in BOTH groups
SYN_LEARNERS = int(os.environ.get("RQ8_SYN_LEARNERS", "3000"))
SYN_ITEMS = int(os.environ.get("RQ8_SYN_ITEMS", "500"))
SKIP_SYN = os.environ.get("RQ8_SKIP_SYN", "0") == "1"
SKIP_PART = os.environ.get("RQ8_SKIP_PART", "0") == "1"
N_LARGE_DIF = int(os.environ.get("RQ8_N_LARGE_DIF", "10"))  # how many to name
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Distinct seed per (split, group) so cross-group agreement is not a shared-init
# artifact.  Offsets keyed by split name and group letter.
GROUP_SEED = {
    "placebo": {"a": 1, "b": 2},
    "ability": {"a": 11, "b": 12},
    "activity": {"a": 21, "b": 22},
    "part": {"a": 31, "b": 32},
}


def tprint(*a, **k):
    print(*a, **k, flush=True)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(spearmanr(x, y).correlation)


# ---------------------------------------------------------------------------
# Mean/sigma linking on the common anchor set
# ---------------------------------------------------------------------------

def link_mean_sigma(
    b_a: np.ndarray, b_b: np.ndarray, anchor: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Mean/sigma-link group B's difficulties onto group A's metric.

    Uses the COMMON ANCHOR items to estimate the linear transform
    ``b_b_linked = slope * b_b + intercept`` so the anchor difficulties share a
    mean and SD across groups.  For a single difficulty parameter this is the
    standard mean/sigma (Marco) linking; it removes the arbitrary scale + origin
    of the two independent fits, leaving only genuine item-by-item DIF.

    Parameters
    ----------
    b_a, b_b : (Q,) per-item difficulty from group A and group B (full item set).
    anchor   : (Q,) bool mask of the common anchor items.

    Returns
    -------
    (b_b_linked (Q,), slope, intercept)
    """
    xa = b_a[anchor]
    xb = b_b[anchor]
    sa, sb = float(np.std(xa)), float(np.std(xb))
    slope = sa / sb if sb > 1e-9 else 1.0
    intercept = float(np.mean(xa)) - slope * float(np.mean(xb))
    return slope * b_b + intercept, slope, intercept


def link_mean_only(
    b_a: np.ndarray, b_b: np.ndarray, anchor: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Mean-only (intercept) linking: ``b_b_linked = b_b + shift`` so the anchor
    MEANS match but the SDs are left alone (slope == 1).

    Why we also report this.  Mean/sigma linking rescales group B's whole anchor
    SD onto group A's.  When the grouping variable is correlated with ability (the
    ability split), each group's b-range is COMPRESSED differently and DIF that
    shifts the group-level dispersion gets partly ABSORBED into the slope -- so
    mean/sigma can UNDERSTATE DIF.  Mean-only linking does not rescale, so a true
    group-level difficulty spread survives into mean|delta b|.  The two together
    bracket the DIF magnitude; the link-FREE Spearman is the arbiter.
    """
    xa = b_a[anchor]
    xb = b_b[anchor]
    shift = float(np.mean(xa)) - float(np.mean(xb))
    return b_b + shift, shift


# ---------------------------------------------------------------------------
# Fit a split, link, and score DIF
# ---------------------------------------------------------------------------

def fit_and_read_group(
    records: List[Dict], n_items: int, seed: int, label: str,
) -> Tuple[np.ndarray, dict]:
    """Fit one subgroup, return its item-only difficulty ``b (Q,)`` + info."""
    model, info = fit_group(
        records, n_items, K=2, d_model=DMODEL, key_dim=DMODEL, value_dim=DMODEL,
        n_epochs=EPOCHS, lr=LR, batch_size=BATCH, device=DEVICE, seed=seed,
        label=label, verbose=True,
    )
    b = item_difficulty_b(model, n_items, device=DEVICE)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return b, info


def score_split(
    split: GroupSplit, n_items: int, item_meta: Optional[Dict] = None,
) -> Dict:
    """Fit both groups of a split, build the common anchor set, link, score DIF.

    Returns a dict with Spearman (link-free), mean|Delta b| after linking, the
    anchor size, and the largest-DIF items (0-based local id + |Delta b|).
    """
    seeds = GROUP_SEED.get(split.name, {"a": 1, "b": 2})
    tprint(f"\n{'='*68}\nSPLIT [{split.name}]  A={split.label_a}  B={split.label_b}")
    tprint(f"  {split.note}")
    tprint(f"  group A learners: {len(split.records_a)}  "
           f"group B learners: {len(split.records_b)}")

    cnt_a = item_seen_counts(split.records_a, n_items)
    cnt_b = item_seen_counts(split.records_b, n_items)
    anchor = (cnt_a >= ANCHOR_MIN) & (cnt_b >= ANCHOR_MIN)
    n_anchor = int(anchor.sum())
    tprint(f"  common anchor items (>= {ANCHOR_MIN} resp in BOTH groups): "
           f"{n_anchor}/{n_items}")
    if n_anchor < 3:
        tprint("  [WARN] anchor set too small to score; skipping split.")
        return {"split": split.name, "n_anchor": n_anchor, "skipped": True}

    b_a, info_a = fit_and_read_group(
        split.records_a, n_items, seed=SEED + seeds["a"],
        label=f"{split.name}/{split.label_a}")
    b_b, info_b = fit_and_read_group(
        split.records_b, n_items, seed=SEED + seeds["b"],
        label=f"{split.name}/{split.label_b}")

    # Link B onto A's metric two ways (see link_mean_only docstring for why both).
    b_b_ms, slope, intercept = link_mean_sigma(b_a, b_b, anchor)   # mean/sigma
    b_b_mo, shift = link_mean_only(b_a, b_b, anchor)               # mean-only

    # Metrics over the anchor set.  Spearman is the LINK-FREE arbiter; the two
    # mean|delta b| values bracket the DIF magnitude (mean/sigma can understate
    # DIF when the grouping variable is ability-correlated, mean-only does not
    # rescale so it is the conservative-high bound).
    spear = _spearman(b_a[anchor], b_b[anchor])  # link-free
    delta_ms = b_a[anchor] - b_b_ms[anchor]
    delta_mo = b_a[anchor] - b_b_mo[anchor]
    mean_abs_db = float(np.mean(np.abs(delta_ms)))      # primary (mean/sigma)
    mean_abs_db_mo = float(np.mean(np.abs(delta_mo)))   # mean-only bracket
    rmse_db = float(np.sqrt(np.mean(delta_ms ** 2)))

    # Largest-DIF items ranked by |Delta b| under mean-only linking (the
    # non-rescaling view, so a genuine group-level shift is not absorbed).
    anchor_ids = np.nonzero(anchor)[0]
    order = np.argsort(-np.abs(delta_mo))
    top = order[:N_LARGE_DIF]
    largest = [
        {"local_id": int(anchor_ids[i]),
         "delta_b": float(delta_mo[i]),
         "abs_delta_b": float(abs(delta_mo[i])),
         "b_a": float(b_a[anchor][i]),
         "b_b_linked": float(b_b_mo[anchor][i]),
         "n_a": int(cnt_a[anchor_ids[i]]),
         "n_b": int(cnt_b[anchor_ids[i]])}
        for i in top
    ]
    # Attach global question_id if a meta map is available.
    if item_meta is not None and "local2global" in item_meta:
        l2g = item_meta["local2global"]
        for d in largest:
            d["question_id"] = l2g.get(d["local_id"], None)

    tprint(f"  link mean/sigma: b_B -> {slope:.3f} * b_B + {intercept:.3f}  "
           f"| mean-only shift={shift:.3f}")
    tprint(f"  Spearman(b_A, b_B) [link-free, ARBITER] = {spear:.4f}")
    tprint(f"  mean|delta b|: mean/sigma={mean_abs_db:.4f}  "
           f"mean-only={mean_abs_db_mo:.4f}  rmse(ms)={rmse_db:.4f}")
    tprint(f"  largest |delta b| (mean-only): " +
           ", ".join(f"id{d['local_id']}={d['abs_delta_b']:.2f}" for d in largest[:5]))

    return {
        "split": split.name,
        "label_a": split.label_a, "label_b": split.label_b,
        "note": split.note,
        "n_learners_a": len(split.records_a),
        "n_learners_b": len(split.records_b),
        "n_anchor": n_anchor,
        "link_slope": slope, "link_intercept": intercept, "link_shift": shift,
        "spearman": spear,
        "mean_abs_delta_b": mean_abs_db,
        "mean_abs_delta_b_meanonly": mean_abs_db_mo,
        "rmse_delta_b": rmse_db,
        "largest_dif": largest,
        "nll_a": info_a["final_nll"], "nll_b": info_b["final_nll"],
        # arrays kept for the synthetic DIF-recovery AUC and the report's flags.
        "_b_a": b_a, "_b_b_linked": b_b_mo, "_anchor": anchor,
        "_delta": (b_a - b_b_mo),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def write_report(real: Dict[str, Dict], syn: Dict[str, Dict]):
    lines: List[str] = []
    A = lines.append
    A("=" * 72)
    A("RQ8  INVARIANCE / DIF  --  do item parameters replicate across subpops?")
    A("=" * 72)
    A("Engine: ma-irt LSTMGPCM (separate_theta=True), K=2 binary, item-only 2PL")
    A("difficulty (threshold(q_embed)).  Independent distinct-seed fit per group;")
    A("mean/sigma linking on the common anchor set; metrics over anchor items.")
    A(f"Config: N_LEARNERS={N_LEARNERS} EPOCHS={EPOCHS} MIN_ITEM_RESP={MIN_ITEM_RESP}")
    A(f"        MIN_SEQ={MIN_SEQ} MAX_SEQ={MAX_SEQ} ANCHOR_MIN={ANCHOR_MIN}")
    A(f"        LR={LR} D_MODEL={DMODEL} DEVICE={DEVICE}")
    A("")

    placebo = real.get("placebo", {})
    pl_db = placebo.get("mean_abs_delta_b", float("nan"))          # mean/sigma
    pl_db_mo = placebo.get("mean_abs_delta_b_meanonly", float("nan"))
    pl_sp = placebo.get("spearman", float("nan"))

    A("PRIMARY ARBITER is the LINK-FREE Spearman of item difficulty (immune to the")
    A("linking confound).  mean|db| is shown under BOTH linkings: mean/sigma (ms,")
    A("can understate DIF when the grouping is ability-correlated) and mean-only")
    A("(mo, does not rescale -> the conservative-high magnitude).  The placebo row")
    A("is the noise floor for ALL columns.")
    A("")
    A("-" * 72)
    A("DIF TABLE (real EdNet) -- read every row AGAINST its matched-floor")
    A("-" * 72)
    A(f"{'split':<12}{'n_anc':>6}{'Spearman':>10}{'mFloor':>9}{'sp-gap':>8}"
      f"{'db(mo)':>9}{'db/mFl':>8}{'nLrg':>6}")
    order = ["placebo", "ability", "activity", "part"]
    for name in order:
        r = real.get(name)
        if not r or r.get("skipped"):
            continue
        db_mo = r.get("mean_abs_delta_b_meanonly", float("nan"))
        sp = r.get("spearman", float("nan"))
        # matched floor: meaningful splits carry their own; placebo is its own floor.
        m_sp = r.get("matched_placebo_sp", pl_sp)
        m_db = r.get("matched_placebo_db_mo", pl_db_mo)
        sp_gap = (m_sp - sp) if (np.isfinite(sp) and np.isfinite(m_sp)) else float("nan")
        ratio_mo = db_mo / m_db if (np.isfinite(db_mo) and np.isfinite(m_db) and m_db > 1e-9) else float("nan")
        # "nLrg" = anchor items whose |db(mo)| exceeds 3x the matched-floor mean|db(mo)|.
        thr = 3.0 * m_db if np.isfinite(m_db) else float("inf")
        delta = r.get("_delta")           # mean-only delta
        anchor = r.get("_anchor")
        if delta is not None and anchor is not None:
            n_large = int((np.abs(delta[anchor]) > thr).sum())
        else:
            n_large = -1
        A(f"{name:<12}{r.get('n_anchor', 0):>6}{sp:>10.4f}{m_sp:>9.4f}{sp_gap:>8.3f}"
          f"{db_mo:>9.4f}{ratio_mo:>8.2f}{n_large:>6}")
    A("")
    A(f"  'mFloor' = the split's SIZE-MATCHED random-placebo Spearman (its correct")
    A(f"  noise floor); 'sp-gap' = mFloor - Spearman (>0 => possible DIF).  The")
    A(f"  global random-half placebo Spearman = {pl_sp:.4f} (overall reliability).")
    A(f"  db(mo) = mean|delta b| under mean-only linking; db/mFl its matched ratio.")
    A(f"  'nLrg' = anchor items with |db(mo)| > 3x the matched-floor mean|db(mo)|.")
    A("")

    # Largest-DIF items per meaningful split.
    for name in ["ability", "activity", "part"]:
        r = real.get(name)
        if not r or r.get("skipped"):
            continue
        A("-" * 72)
        A(f"LARGEST-DIF ITEMS -- {name} ({r['label_a']} vs {r['label_b']})")
        A("-" * 72)
        A(f"  {'local_id':>9}{'question_id':>14}{'delta_b':>10}{'b_A':>9}"
          f"{'b_B_link':>10}{'n_A':>7}{'n_B':>7}")
        for d in r.get("largest_dif", [])[:N_LARGE_DIF]:
            qid = d.get("question_id", "")
            A(f"  {d['local_id']:>9}{str(qid):>14}{d['delta_b']:>10.3f}"
              f"{d['b_a']:>9.3f}{d['b_b_linked']:>10.3f}{d['n_a']:>7}{d['n_b']:>7}")
        A("  (delta_b > 0 => item is HARDER for group A than group B; the sign")
        A("   tells you which subpopulation finds the item relatively harder.)")
        A("")

    # Synthetic controls.
    if syn:
        A("-" * 72)
        A("SYNTHETIC CONTROLS (validate the metric end-to-end)")
        A("-" * 72)
        nd = syn.get("nodif", {})
        if nd:
            A(f"  NO-DIF latent  : ability Spearman={nd.get('spearman', float('nan')):.4f} "
              f"(placebo {nd.get('placebo_sp', float('nan')):.4f})  sp-gap="
              f"{nd.get('placebo_sp', float('nan')) - nd.get('spearman', float('nan')):.3f}")
            A(f"                   -> ability ~ placebo (NO DIF) expected: small sp-gap.")
        di = syn.get("dif", {})
        if di:
            A(f"  KNOWN-DIF latent: ability Spearman={di.get('spearman', float('nan')):.4f} "
              f"(placebo {di.get('placebo_sp', float('nan')):.4f})  sp-gap="
              f"{di.get('placebo_sp', float('nan')) - di.get('spearman', float('nan')):.3f}")
            A(f"                   DIF-item detection AUC={di.get('dif_auc', float('nan')):.3f} "
              f"(|db| ranks injected DIF items; 0.5=chance, 1.0=perfect).")
            A(f"                   -> ability << placebo Spearman (DIF) expected: large")
            A(f"                      sp-gap AND AUC well above 0.5.  NOTE: under an")
            A(f"                      ability-correlated split, mean/sigma linking can")
            A(f"                      ABSORB DIF into the slope, so mean|db(ms)| may NOT")
            A(f"                      rise above placebo even with real DIF -- which is")
            A(f"                      exactly why Spearman + AUC are the arbiters here.")
        A("")

    # Verdict.  Primary criterion: Spearman-gap to placebo (link-free, immune to
    # the linking confound).  mean|db(mo)| ratio corroborates magnitude.
    A("=" * 72)
    A("VERDICT (candid)")
    A("=" * 72)
    A(f"  Primary criterion: Spearman-gap to the SIZE-MATCHED random placebo")
    A(f"  (the correct floor for each split's group sizes).  A split shows DIF")
    A(f"  only if its difficulty ranking falls MATERIALLY below its matched floor.")
    A("")
    for name in ["ability", "activity", "part"]:
        r = real.get(name)
        if not r or r.get("skipped"):
            continue
        sp = r.get("spearman", float("nan"))
        db_mo = r.get("mean_abs_delta_b_meanonly", float("nan"))
        m_sp = r.get("matched_placebo_sp", pl_sp)        # matched floor (fallback global)
        m_db = r.get("matched_placebo_db_mo", pl_db_mo)
        sp_gap = (m_sp - sp) if (np.isfinite(sp) and np.isfinite(m_sp)) else float("nan")
        ratio_mo = db_mo / m_db if (np.isfinite(db_mo) and np.isfinite(m_db) and m_db > 1e-9) else float("nan")
        if not np.isfinite(sp_gap):
            verdict = "INCONCLUSIVE (matched floor not estimable)"
        elif sp_gap < 0.03:
            verdict = "INVARIANT (ranking ~ matched placebo; population-robust, NO meaningful DIF)"
        elif sp_gap < 0.08:
            verdict = "MILD DIF (ranking modestly below matched placebo; inspect named items)"
        else:
            verdict = "REAL DIF (ranking well below matched placebo; population-specific on flagged items)"
        line = (f"  [{name:<9}] Spearman={sp:.4f}  matchedFloor={m_sp:.4f}  "
                f"sp-gap={sp_gap:+.3f}  db(mo)/matched={ratio_mo:.2f}  -> {verdict}")
        A(line)
        if name == "ability":
            A("              ^ ABILITY caveat: ability grouping induces RANGE")
            A("                RESTRICTION (high-ability learners give little")
            A("                difficulty info on easy items), which inflates the")
            A("                sp-gap even with ZERO DIF (see the synthetic NO-DIF")
            A("                block, whose ability sp-gap is large by construction).")
            A("                Treat the ability aggregate as a confounded UPPER")
            A("                bound; the per-item largest-|db| list is the usable")
            A("                DIF-candidate signal (synthetic AUC validates it).")
    A("")
    A("  Gating caveat: if the placebo itself fails to replicate (its Spearman far")
    A("  below the synthetic NO-DIF ceiling), recovery is too noisy to claim")
    A("  invariance; read all rows as upper bounds in that case.")
    A("  Linking caveat: mean|db(ms)| (mean/sigma) can UNDERSTATE DIF under an")
    A("  ability-correlated split (the slope absorbs it); the verdict is driven by")
    A("  the link-free Spearman vs the matched floor, corroborated by mean-only db.")

    report = "\n".join(lines)
    (OUTPUT_DIR / "report.txt").write_text(report)
    tprint("\n" + report)


# ---------------------------------------------------------------------------
# Synthetic-control blocks
# ---------------------------------------------------------------------------

def run_synthetic_controls() -> Dict[str, Dict]:
    """NO-DIF and KNOWN-DIF synthetic blocks; both use the ABILITY split, both
    report their own placebo for an in-block reference."""
    out: Dict[str, Dict] = {}

    # ---- NO-DIF ----
    tprint("\n##### SYNTHETIC NO-DIF (one true 2PL latent) #####")
    nd_raw = make_synthetic_nodif(n_learners=SYN_LEARNERS, n_items=SYN_ITEMS,
                                  seq_len=120, seed=SEED)
    nd_pl = score_split(split_random_half(nd_raw, seed=0), nd_raw.n_items)
    nd_ab = score_split(split_ability(nd_raw), nd_raw.n_items)
    out["nodif"] = {
        "mean_abs_delta_b": nd_ab["mean_abs_delta_b"],
        "spearman": nd_ab["spearman"],
        "placebo_db": nd_pl["mean_abs_delta_b"],
        "placebo_sp": nd_pl["spearman"],
    }

    # ---- KNOWN-DIF ----
    tprint("\n##### SYNTHETIC KNOWN-DIF (injected on the ability axis) #####")
    di_raw, dif_mask = make_synthetic_dif(
        n_learners=SYN_LEARNERS, n_items=SYN_ITEMS, seq_len=120, seed=SEED)
    di_pl = score_split(split_random_half(di_raw, seed=0), di_raw.n_items)
    di_ab = score_split(split_ability(di_raw), di_raw.n_items)
    # DIF-item detection AUC: does |delta b| rank the injected DIF items above the
    # non-DIF items?  AUC of |delta b| as a score for the dif_mask label, over the
    # anchor set.
    anchor = di_ab["_anchor"]
    score = np.abs(di_ab["_delta"])[anchor]
    label = dif_mask[anchor].astype(int)
    auc = _auc(score, label)
    out["dif"] = {
        "mean_abs_delta_b": di_ab["mean_abs_delta_b"],
        "spearman": di_ab["spearman"],
        "placebo_db": di_pl["mean_abs_delta_b"],
        "placebo_sp": di_pl["spearman"],
        "dif_auc": auc,
    }
    return out


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC-AUC of ``score`` against binary ``label`` (rank-based, ties averaged)."""
    pos = label == 1
    neg = label == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(cnt))
    np.add.at(sum_ranks, inv, ranks)
    avg = sum_ranks / cnt
    ranks = avg[inv]
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = OUTPUT_DIR / "run.log"
    log_fh = open(log_path, "w", buffering=1)
    orig_stdout = sys.stdout
    sys.stdout = log_fh
    try:
        t0 = time.time()
        tprint("=" * 68)
        tprint("RQ8 INVARIANCE / DIF")
        tprint("=" * 68)
        tprint(f"Config: N_LEARNERS={N_LEARNERS} EPOCHS={EPOCHS} "
               f"MIN_ITEM_RESP={MIN_ITEM_RESP} MIN_SEQ={MIN_SEQ} MAX_SEQ={MAX_SEQ} "
               f"ANCHOR_MIN={ANCHOR_MIN} LR={LR} D_MODEL={DMODEL} SEED={SEED} "
               f"DEVICE={DEVICE}")

        # ---- SYNTHETIC CONTROLS (first, cheap, make real numbers interpretable) ----
        syn: Dict[str, Dict] = {}
        if not SKIP_SYN:
            syn = run_synthetic_controls()

        # ---- REAL EdNet ----
        tprint("\n##### REAL EdNet (binary K=2) #####")
        raw = load_raw_binary(
            n_learners=N_LEARNERS, min_item_responses=MIN_ITEM_RESP,
            min_seq_len=MIN_SEQ, max_seq_len=MAX_SEQ, seed=SEED,
        )
        tprint(f"[Data] n_items={raw.n_items} learners={len(raw.records)}")

        real: Dict[str, Dict] = {}
        # 1. PLACEBO FIRST (the control / global noise floor).
        real["placebo"] = score_split(split_random_half(raw, seed=0), raw.n_items)
        # 2. ABILITY.
        ab = split_ability(raw)
        real["ability"] = score_split(ab, raw.n_items)
        # 3. ACTIVITY.
        act = split_activity(raw)
        real["activity"] = score_split(act, raw.n_items)
        # 3b. PART-engagement alternative (optional; the two largest parts).
        part = None
        if not SKIP_PART:
            part = split_part(raw, 5, 2)
            real["part"] = score_split(part, raw.n_items)

        # ---- SIZE-MATCHED RANDOM PLACEBOS (the correct floor per split's group
        # sizes; the RIGHT reference for activity/part, a size/sampling bound for
        # ability which additionally suffers range restriction). ----
        matched: Dict[str, Dict] = {}
        for nm, sp in (("ability", ab), ("activity", act), ("part", part)):
            if sp is None or real.get(nm, {}).get("skipped"):
                continue
            na, nb = len(sp.records_a), len(sp.records_b)
            mp = matched_random_split(raw, na, nb, name=f"matched_{nm}", seed=7)
            matched[nm] = score_split(mp, raw.n_items)
        # Attach each split's matched-placebo Spearman + db for the report.
        for nm, mres in matched.items():
            if real.get(nm) and not mres.get("skipped"):
                real[nm]["matched_placebo_sp"] = mres["spearman"]
                real[nm]["matched_placebo_db_mo"] = mres["mean_abs_delta_b_meanonly"]

        wall = time.time() - t0
        results = {"real": _clean(real), "synthetic": _clean(syn),
                   "wall_clock_total": wall,
                   "config": {"N_LEARNERS": N_LEARNERS, "EPOCHS": EPOCHS,
                              "MIN_ITEM_RESP": MIN_ITEM_RESP, "MIN_SEQ": MIN_SEQ,
                              "MAX_SEQ": MAX_SEQ, "ANCHOR_MIN": ANCHOR_MIN,
                              "LR": LR, "D_MODEL": DMODEL, "SEED": SEED}}
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        write_report(real, syn)
        tprint(f"\nTotal wall clock: {wall:.1f}s")
    finally:
        sys.stdout = orig_stdout
        log_fh.close()
    print((OUTPUT_DIR / "report.txt").read_text())


if __name__ == "__main__":
    main()
