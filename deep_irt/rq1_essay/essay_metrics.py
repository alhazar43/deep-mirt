"""essay_metrics.py -- Held-out-format transfer metrics for the essay RQ1 pilot.

The latent target is the HUMAN holistic score (raw_scores). We test whether an
essay seen ONLY in one format lands correctly on the OTHER format's scale.

Two contrasts, both reported:

  (1) Independent baseline (PI-named): joint d_i vs the per-format extractor's
      content-only readout. On REAL data with frozen features this baseline is
      NOT noise -- it is content-based generalization. The gap = what
      cross-format sharing adds.

  (2) The "both-recover-truth" trap detector: for PAIRWISE-ONLY essays, does the
      joint model's d_i predict the HELD-OUT human score better than the
      indep-graded content-only readout on the SAME essays? If joint == indep,
      the apparent transfer is just content predicting quality (both formats
      independently learn the same map), NOT unification. The gap is the genuine
      cross-format transfer signal.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    r, _ = sp_stats.spearmanr(x, y)
    return float(r)


def align_sign(rec, ref):
    rec = np.asarray(rec, float); ref = np.asarray(ref, float)
    if np.std(rec) < 1e-12 or np.std(ref) < 1e-12:
        return rec
    r, _ = sp_stats.pearsonr(rec, ref)
    return -rec if r < 0 else rec


def pairwise_score_consistency(idx, comparisons, raw_scores) -> float:
    """For a set of essays, the fraction of comparisons among them whose LLM
    winner agrees with the higher human score (ties excluded). This is the
    'held-out comparisons' target for GRADED-ONLY essays -- but those essays
    have NO comparisons by construction, so we instead measure, for each essay,
    whether its quality rank vs comparison agreement is consistent. Returned as
    a diagnostic on BOTH/pairwise essays.
    """
    idxset = set(int(i) for i in idx)
    agree = total = 0
    for i, j, o in comparisons:
        if i in idxset and j in idxset:
            si, sj = raw_scores[i], raw_scores[j]
            if si == sj:
                continue
            human_winner_i = si > sj
            llm_winner_i = (o == 1)
            agree += int(human_winner_i == llm_winner_i)
            total += 1
    return agree / total if total else float("nan")


def compute_metrics(split, comparisons, joint_q, indep_graded_q, indep_bt_q) -> dict:
    """All quality vectors are (n_essays,), mean-centered, in essay-index space.

    Returns named Spearman correlations vs the HUMAN raw score.
    """
    raw = split.raw_scores
    go = split.graded_only_idx
    po = split.pairwise_only_idx
    both = split.both_idx

    # Sign-align every readout to the human score globally before slicing.
    jq = align_sign(joint_q, raw)
    igq = align_sign(indep_graded_q, raw)
    ibq = align_sign(indep_bt_q, raw)

    m = {}

    # ---- Overall recovery vs human score (all essays) ----
    m["joint_overall"] = spearman(jq, raw)
    m["indep_graded_overall"] = spearman(igq, raw)
    m["indep_bt_overall"] = spearman(ibq, raw)

    # ================================================================
    # [A] PAIRWISE-ONLY essays: human score is HELD OUT.
    #     Joint reads them onto the graded scale via shared extractor + BT.
    #     Contrast 1 (PI baseline): indep-BT readout (pairwise extractor only).
    #     Contrast 2 (trap detector): indep-GRADED content-only readout.
    # ================================================================
    m["joint_on_pairwise_only"] = spearman(jq[po], raw[po])
    m["indep_bt_on_pairwise_only"] = spearman(ibq[po], raw[po])        # PI baseline
    m["indep_graded_on_pairwise_only"] = spearman(igq[po], raw[po])    # trap detector
    m["pairwise_only_gap_vs_indepBT"] = (
        m["joint_on_pairwise_only"] - m["indep_bt_on_pairwise_only"]
    )
    m["pairwise_only_gap_vs_content"] = (
        m["joint_on_pairwise_only"] - m["indep_graded_on_pairwise_only"]
    )

    # ================================================================
    # [B] GRADED-ONLY essays: their comparisons are HELD OUT (none exist).
    #     Joint reads them onto the pairwise scale via shared extractor + graded.
    #     Contrast 1 (PI baseline): indep-GRADED readout (graded extractor only).
    #     Target = human score (the ground-truth quality the pairwise scale
    #     should agree with).
    #     Contrast 2 (trap detector): indep-BT content-only readout.
    # ================================================================
    m["joint_on_graded_only"] = spearman(jq[go], raw[go])
    m["indep_graded_on_graded_only"] = spearman(igq[go], raw[go])      # PI baseline (own data)
    m["indep_bt_on_graded_only"] = spearman(ibq[go], raw[go])          # trap detector
    m["graded_only_gap_vs_indepGraded"] = (
        m["joint_on_graded_only"] - m["indep_graded_on_graded_only"]
    )
    m["graded_only_gap_vs_content"] = (
        m["joint_on_graded_only"] - m["indep_bt_on_graded_only"]
    )

    # ---- BOTH essays (bridge; both heads trained) ----
    m["joint_on_both"] = spearman(jq[both], raw[both])

    # ---- Diagnostics ----
    m["llm_human_pair_agreement_both_po"] = pairwise_score_consistency(
        np.concatenate([po, both]), comparisons, raw
    )
    return m


def format_report(split, comparisons, m, params) -> str:
    n = split.n
    L = [
        "=" * 70,
        "RQ1 ESSAY PILOT -- Format-agnostic shared QUALITY scale on REAL essays",
        "Graded (human holistic score) + Pairwise (local-LLM judgments)",
        "PATH A: pairwise comparisons are MODEL-GENERATED (caveat applies).",
        "=" * 70,
        "",
        "DESIGN",
        f"  ASAP essay_set:       {split.essay_set}  (score range "
        f"{int(split.raw_scores.min())}-{int(split.raw_scores.max())}, K={split.K})",
        f"  Total essays:         {n}",
        f"  GRADED-ONLY:          {len(split.graded_only_idx)}  "
        f"(human score only; comparisons held out)",
        f"  PAIRWISE-ONLY:        {len(split.pairwise_only_idx)}  "
        f"(comparisons only; human score held out)",
        f"  BOTH (bridge):        {len(split.both_idx)}",
        f"  Content features:     {split.feat_kind} (FROZEN, dim={split.features.shape[1]})",
        f"  Judge model:          {params.get('judge_model')}",
        f"  Comparisons (parsed): {len(comparisons)}",
        f"  Epochs:               {params.get('n_epochs')}",
        "",
        "OVERALL RECOVERY vs HUMAN SCORE  (Spearman, all essays)",
        f"  Joint (shared extractor):        {m['joint_overall']:+.4f}",
        f"  Indep graded (content-only):     {m['indep_graded_overall']:+.4f}",
        f"  Indep BT     (content-only):     {m['indep_bt_overall']:+.4f}",
        "",
        "HELD-OUT-FORMAT TRANSFER  (the RQ1 test)",
        "",
        "  [A] PAIRWISE-ONLY essays (human score HELD OUT):",
        f"    Joint d_i vs human score:              {m['joint_on_pairwise_only']:+.4f}  <- transfer",
        f"    Indep-BT readout (PI baseline):        {m['indep_bt_on_pairwise_only']:+.4f}",
        f"    Indep-GRADED content-only (trap chk):  {m['indep_graded_on_pairwise_only']:+.4f}",
        f"      gap vs indep-BT baseline:            {m['pairwise_only_gap_vs_indepBT']:+.4f}",
        f"      gap vs content-only (genuine xfer):  {m['pairwise_only_gap_vs_content']:+.4f}",
        "",
        "  [B] GRADED-ONLY essays (comparisons HELD OUT):",
        f"    Joint d_i vs human score:              {m['joint_on_graded_only']:+.4f}  <- transfer",
        f"    Indep-GRADED readout (PI baseline):    {m['indep_graded_on_graded_only']:+.4f}",
        f"    Indep-BT content-only (trap chk):      {m['indep_bt_on_graded_only']:+.4f}",
        f"      gap vs indep-graded baseline:        {m['graded_only_gap_vs_indepGraded']:+.4f}",
        f"      gap vs content-only (genuine xfer):  {m['graded_only_gap_vs_content']:+.4f}",
        "",
        "BRIDGE / DIAGNOSTICS",
        f"  Joint on BOTH essays vs human:           {m['joint_on_both']:+.4f}",
        f"  LLM-vs-human pair agreement (po+both):    {m['llm_human_pair_agreement_both_po']:.4f}",
        "",
    ]

    # Verdict: genuine transfer = joint beats content-only on the held-out
    # format. The trap is joint == content-only.
    po_xfer = m["pairwise_only_gap_vs_content"]
    po_joint = m["joint_on_pairwise_only"]
    L.append("VERDICT (genuine content-mediated transfer = joint beats content-only)")
    if np.isnan(po_joint):
        L.append("  INDETERMINATE: pairwise-only transfer Spearman is undefined "
                 "(too few essays or degenerate variance). Scale N.")
    elif po_joint >= 0.30 and po_xfer >= 0.05:
        L.append(f"  GENUINE TRANSFER (pilot-scale). Pairwise-only essays land on the "
                 f"graded scale (rho={po_joint:+.3f}) ABOVE the content-only readout "
                 f"(gap={po_xfer:+.3f}). The cross-format sharing adds signal beyond "
                 f"content alone -> not the both-recover-truth trap.")
    elif po_joint >= 0.30 and po_xfer < 0.05:
        L.append(f"  TRAP / INCONCLUSIVE. Pairwise-only essays DO land on the graded "
                 f"scale (rho={po_joint:+.3f}), but the content-only readout does just as "
                 f"well (gap={po_xfer:+.3f}). The apparent transfer is content predicting "
                 f"quality, not the model UNIFYING formats. The pilot validates the "
                 f"PIPELINE but does not yet demonstrate genuine cross-format transfer.")
    else:
        L.append(f"  WEAK/NULL at pilot N. Pairwise-only transfer rho={po_joint:+.3f}. "
                 f"Either the judge signal or N is insufficient; see what the full run needs.")
    L.append("")
    L.append("=" * 70)
    return "\n".join(L)
