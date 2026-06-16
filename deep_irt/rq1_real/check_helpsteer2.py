"""check_helpsteer2.py -- Precondition: inter-format agreement on HelpSteer2.

Do the human HELPFULNESS ratings (graded, 0-4) agree with the human PAIRWISE
preferences (signed strength -3..+3) on the SAME response pairs?

Base ratings live in train/validation jsonl (prompt, response, helpfulness,...).
Preferences live in preference.jsonl (prompt, response_1, response_2,
preference_strength in [-3,3]; +ve => response_2 better, -ve => response_1).
Join key is (prompt, response) text. Some pairs will not join (a response not
present in the rated base, or rated multiple times); we keep pairs where BOTH
sides join uniquely.

Agreement metrics on joined pairs (ties in EITHER signal excluded):
  - sign agreement: sign(help_2 - help_1) == sign(pref_strength).
  - Spearman(help_diff, pref_strength) over non-tied helpfulness pairs.
This is the precondition: if helpfulness and preference disagree, HelpSteer2 is
not a clean shared-latent venue. (Both are human helpfulness judgments, so we
expect agreement.)
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

DATA = Path("rl/data/helpsteer2")


def load_base():
    """(prompt, response) -> list of helpfulness ratings.  Also returns the
    full per-attribute dict for the unique-keyed responses."""
    help_by_key = defaultdict(list)
    rows_by_key = defaultdict(list)
    for fn in ["train.jsonl.gz", "validation.jsonl.gz"]:
        with gzip.open(DATA / fn, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r["prompt"], r["response"])
                help_by_key[key].append(r["helpfulness"])
                rows_by_key[key].append(r)
    return help_by_key, rows_by_key


def load_pref():
    rows = []
    with gzip.open(DATA / "preference" / "preference.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(sp_stats.spearmanr(x, y)[0])


def main():
    help_by_key, _ = load_base()
    prefs = load_pref()
    print(f"base unique (prompt,response) keys: {len(help_by_key)}")
    print(f"preference pairs: {len(prefs)}")

    # mean helpfulness per key (multiple annotators -> average rating)
    help_mean = {k: float(np.mean(v)) for k, v in help_by_key.items()}

    joined = 0
    help_diffs = []          # help_2 - help_1
    pref_strengths = []      # signed strength
    sign_agree = sign_total = 0
    for r in prefs:
        k1 = (r["prompt"], r["response_1"])
        k2 = (r["prompt"], r["response_2"])
        if k1 not in help_mean or k2 not in help_mean:
            continue
        joined += 1
        h1 = help_mean[k1]; h2 = help_mean[k2]
        ps = r["preference_strength"]
        help_diffs.append(h2 - h1)
        pref_strengths.append(ps)
        # sign agreement: exclude ties in either signal
        if ps == 0 or (h2 - h1) == 0:
            continue
        sign_agree += int(np.sign(h2 - h1) == np.sign(ps))
        sign_total += 1

    help_diffs = np.array(help_diffs)
    pref_strengths = np.array(pref_strengths)

    rho_all = spearman(help_diffs, pref_strengths)
    # restrict to non-tied preference (|strength|>=1)
    nz = pref_strengths != 0
    rho_nztie = spearman(help_diffs[nz], pref_strengths[nz])
    sa = sign_agree / sign_total if sign_total else float("nan")

    print("\n" + "=" * 64)
    print("HELPSTEER2 INTER-FORMAT AGREEMENT (the precondition)")
    print("=" * 64)
    print(f"  preference pairs joined to base ratings: {joined} / {len(prefs)}")
    print(f"  Spearman(help_diff, pref_strength) ALL:  {rho_all:+.4f}  <- KEY")
    print(f"  Spearman, non-tied pref (|s|>=1):        {rho_nztie:+.4f}")
    print(f"  sign agreement (both non-tied):          {sa:.4f} "
          f"(n={sign_total})")
    print("=" * 64)
    verdict = ("AGREE -> USABLE venue" if (not np.isnan(rho_all) and rho_all >= 0.4)
               else "DISAGREE -> not a clean venue")
    print(f"  VERDICT: helpfulness vs preference {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
