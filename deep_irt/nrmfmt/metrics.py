"""
metrics.py -- Cross-format transfer metrics for binary-vs-NRM.

Mirrors deep_irt/jointfmt/metrics.py.  The held-out-format transfer test:

  NRM-ONLY items (never in binary data):
    NRM training shaped their embedding.  The shared d_i can then be read as a
    2PL difficulty even though the binary head never saw them.
    Transfer signal: joint d_i[nrm_only] vs s_true.
    Baseline: indep_binary[nrm_only] is noise (never trained on these).

  BINARY-ONLY items (never in NRM data):
    Binary training shaped their embedding.  The shared d_i can then be read as
    the NRM correct-option location even though the NRM head never saw them.
    Transfer signal: joint d_i[binary_only] vs s_true.
    Baseline: indep_nrm[binary_only] is noise.

Because both heads write to the SAME shared scalar d_i, the joint readout is a
single d_i per item -- the held-out-format placement is just d_i sliced on the
held-out partition.
"""

import torch
from scipy import stats as sp_stats


def spearman(x, y):
    r, _ = sp_stats.spearmanr(
        x.detach().float().numpy(), y.detach().float().numpy()
    )
    return float(r)


def align_sign(recovered, reference):
    r, _ = sp_stats.pearsonr(
        recovered.detach().float().numpy(), reference.detach().float().numpy()
    )
    return -recovered if r < 0 else recovered


def compute_metrics(gt, joint_d, indep_binary_d, indep_nrm_d):
    s = gt["s"]
    bo = gt["binary_only_idx"]
    no = gt["nrm_only_idx"]
    both = gt["both_idx"]

    jd = align_sign(joint_d, s)
    ibd = align_sign(indep_binary_d, s)
    ind = align_sign(indep_nrm_d, s)

    # Overall recovery
    joint_overall = spearman(jd, s)
    indep_binary_overall = spearman(ibd, s)
    indep_nrm_overall = spearman(ind, s)

    # ---- Cross-format transfer ----
    # NRM-only: shared d_i read as 2PL difficulty (binary head never saw these)
    joint_on_nrm_only = spearman(jd[no], s[no])
    indep_binary_nrm_only = spearman(ibd[no], s[no])           # noise baseline
    nrm_only_gap = joint_on_nrm_only - indep_binary_nrm_only

    # Binary-only: shared d_i read as NRM location (NRM head never saw these)
    joint_on_binary_only = spearman(jd[bo], s[bo])
    indep_nrm_binary_only = spearman(ind[bo], s[bo])           # noise baseline
    binary_only_gap = joint_on_binary_only - indep_nrm_binary_only

    # Sanity: each indep model on its OWN items
    indep_binary_on_binary_only = spearman(ibd[bo], s[bo])
    indep_nrm_on_nrm_only = spearman(ind[no], s[no])

    # Joint on BOTH (both heads trained on these)
    joint_both = spearman(jd[both], s[both])

    return {
        "joint_overall": joint_overall,
        "indep_binary_overall": indep_binary_overall,
        "indep_nrm_overall": indep_nrm_overall,
        "joint_on_nrm_only": joint_on_nrm_only,
        "indep_binary_nrm_only": indep_binary_nrm_only,
        "nrm_only_transfer_gap": nrm_only_gap,
        "joint_on_binary_only": joint_on_binary_only,
        "indep_nrm_binary_only": indep_nrm_binary_only,
        "binary_only_transfer_gap": binary_only_gap,
        "indep_binary_on_binary_only": indep_binary_on_binary_only,
        "indep_nrm_on_nrm_only": indep_nrm_on_nrm_only,
        "joint_both": joint_both,
    }
