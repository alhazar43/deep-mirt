# A4 / G2 certification verdict — KDD profile (seed-clustered)

Goal G2: per-KC growth beyond noise. This is the honest, paper-grade
verdict over the completed KDD-matched synthetic certification: 4 twins x
5 slice seeds (existence gate + MIX + split-half + bank recovery + ground
truth) and 4 twins x 3 model seeds (active posture + CG7-CG10 audit).
Verdict is seed-clustered per the frozen design. Scope and caveats at the
end. Do not read any single line as certified beyond what the gate below
supports.

## 1. Headline — the passive existence gate

**Certifies as a twin-level (pooled) growth detector. Does NOT certify as
a per-KC detector.** Split the claim, because G2 asks for per-KC growth.

- **Pooled detection is robust and seed-clean (HIGH confidence).** Direct
  read of all 25 slice cells confirms the gate is genuinely silent on the
  null twin and genuinely firing on both growth twins, in every seed
  independently, with no seed fragility and no single seed carrying the
  result. syn_ng bed_pvalue [.268,.640,.128,.973,.743], all clear of any
  0.01-0.05 bar; syn_kg and syn_ns sit at p=0.001 in all 5 seeds. The
  bed_stat distributions do not overlap: the largest ng stat (915) is
  ~2.6x below the smallest kg/ns stat (2375). CG2 (null holds) satisfied
  on syn_ng.
- **Caveat on strength.** The kg/ns p=0.001 is a permutation floor
  (1/(999+1)), so the test cannot rank "very" vs "even more" significant.
  This limits detection-strength precision, not the robustness claim.
- **Per-KC certification FAILS on the positive control (syn_kg).** CG3
  requires >=60% of KCs discovered under BH and bank recovery >=0.90.
  Per-KC BH discoveries are 0/515 in all 5 seeds (min kc_pvalue 0.005 vs
  BH threshold 9.7e-5); bank recovery is 0.73 x5. The gate sees the twin
  as a whole but resolves no individual growing KC. syn_ns is worse and
  erratic: BH discoveries [65,0,55,0,0], power in only 2 of 5 seeds, BY 0
  everywhere (all discoveries dependence-sensitive). CG5 fails.

Bottom line for G2: the existence gate is a validated **coarse** growth
detector on KDD-matched data — right sign, right twins, seed-stable — but
it is **not** a certified per-KC instrument. The per-KC resolution that G2
targets is not yet earned.

## 2. Saturation finding (syn_sat) — real limitation, not a seed fluke

CG6 (the pre-registered null-under-saturation: the gate should FAIL to
detect real growth once saturation destroys observability) is **inverted**:
the gate fires hard, bed_pvalue 0.001 x5, bed_stat ~10950.

Per CHECK 2, this fire is **seed-consistent** (stat spread ~5% across the
5 seeds) and is **NOT** genuine see-through-saturation detection. syn_sat's
true_rise_per_kc is byte-identical to syn_kg's, yet the stat is ~4x larger
— the opposite of what a genuine, saturation-compressed signal would do.
The inflation is distributional (median kc_stat 10.4 vs kg 3.1; only
8/515 KCs negative vs kg's 100/515), not a few blown-up outliers.
Independent corroboration of numerical degeneracy in the same near-ceiling
regime: the MIX-rate module returns r_c_se pinned to the 1e-6 ridge floor
(z ~ 1.2e5) on saturated KCs, a different module on the same data.

Verdict: **a genuine methodological limitation to record**, not a bug that
a new seed would remove. Mechanism is model misspecification against
near-ceiling binary responses (M0 approximates a saturating curve worse
than M1, giving M1 a near-universal held-out NLL edge regardless of true
growth). The independent split-half leg agrees: syn_sat split-half gap
.259-.292 blows through the 0.10 tolerance by 2.5-3x while the other three
twins stay inside it. **Actionable flag for real KDD:** high-performing
(near-ceiling) KCs will trip the gate spuriously; a saturation-aware null
or near-ceiling down-weighting is needed before trusting a fire on
easy/mastered KCs.

## 3. Active posture (ACT) — direction, not magnitude

The active readout recovers the **presence and direction** of growth but
**not per-KC magnitude or ranking**.

- syn_kg: act_p1 fires (pop 0.0574 >= 0.05 bar); act_p0 does not (0.0435).
  growing_rank_corr ~0.27 on both, far below the CG1a 0.6 bar — ACT fires
  in aggregate but cannot rank which KCs rose.
- syn_ns: both variants fire (pop 0.060 / 0.074), overshoot bounded
  (2.2-2.7%, clears CG1b); rank_corr ~0.33-0.38, below the 0.5 bar. Misfit
  fires on only 6-7% of growing KCs (needs 80%) and MORE on the silent
  subset than on growers — wrong-direction laundering.
- syn_ng: act_p1 silent (good); **act_p0 p95 0.044 breaches the 0.01
  silence bar** — the primary variant is marginally not-silent on the null
  twin. Only twin that otherwise matches its designed posture.

Verdict: **direction achieved, magnitude and per-KC rank not.** Consistent
with the gauge-bound magnitude story elsewhere in the program. act_p1 is
the more disciplined variant (silent on null, fires on growth); act_p0
leaks on null.

## 4. Neural readout audit (CG7-CG10) — scope to PAS-N1

The neural tracker fails the faithfulness battery **consistently and by
design**, but the headline must be scoped or it misleads (CHECK 3).

- CG7 (untrained-null) 0/12, CG8 (drill contamination) 0/12, CG9 (order
  stress) 0/12 across all 4 twins x 3 seeds — exact.
- CG10 (direction audit) 11/12 fail; the one exception is syn_sat seed2
  (0.028 < 0.10), a genuine, previously unreported borderline pass. NG,
  KG, NS are each 0/3.
- **This is PAS-N1**, the field-representative shared-state tracker, whose
  failure is a designed finding (the PAS-N1 disease), not a bug. CG8/CG9
  are the intended PAS-N1 failure modes; CG7/CG10 failing too is beyond
  the designed set and worth noting.
- **PAS-N2** (factorized per-KC tracker) is architecturally immune to
  these failure modes, but the 12 cells contain **no CG7-CG10 verdicts for
  PAS-N2** — only a same-ballpark held-out NLL (e.g. kg seed0: 0.4326 vs
  N1 0.4446) as an implementation sanity probe. So "PAS-N2 does better on
  the audits" is a **construction guarantee, not a measured result**, and
  must be stated as such.

## 5. Rate reliability (MIX / split-half) — one leg present, one missing

CHECK 4 corrects the premise that split_half is None: the **per-learner**
split-half (odd/even within slice, Spearman-Brown corrected, the field
CG3 consumes) is present and finite in all 20 cells — tight and passing
for ng/kg/ns (gaps .066-.098, within 0.10), diagnostically blown out for
syn_sat (.259-.292). MIX r_c medians 0.13-0.15; predicted-vs-observed
split-half gaps within tolerance for the three non-saturated twins.

The real gap: the **KC-level** split-half leg that gate RB4 needs to
certify Tier-2 "trustworthy per-KC rate" claims is **dead code** —
`kc_level_split_half` / `rb4_kc_rates` exist in battery.py but are never
called from run.py, so no KC-level reliability was computed for any cell.
This is a **harness wiring gap, not a statistical null and not insufficient
data.** It does not change syn_kg's CG3 verdict (which already fails on
discovery fraction and bank), but **RB4 / Tier-2 per-KC rate reliability
cannot be certified from these artifacts at all.** If a future run cleared
CG3's other legs, the per-KC rate claim would still have no reliability
check behind it. Fix the wiring before any Tier-2 rate claim.

## 6. Scope and caveats

- **KDD profile only.** EdNet corroboration is still running; nothing here
  generalizes across beds until that lands.
- **Seed-clustered**, per the frozen design (4x5 slice, 4x3 neural).
- **Twin-level pass tally:** syn_ng is the only twin matching its designed
  posture (with the act_p0 p95 breach flagged). syn_kg (positive control)
  detects and fires but **fails full per-KC certification** (BH 0%, bank
  0.73, rank 0.27). syn_ns detects but fails CG5 misfit. syn_sat inverts
  CG6 as a real near-ceiling limitation.
- **kg/ns p=0.001 is a permutation floor** — detection is maxed, strength
  is not further resolvable.
- **Bank recovery 0.73 across kg/ns/sat** is well under the 0.90 bar
  everywhere it is barred.
- **PAS-N2 audit superiority is unmeasured** (construction claim only).
- **RB4 per-KC rate reliability is unwired**, cannot be certified.

Net: G2 is **partially certified on the KDD profile.** The passive
existence gate is a validated coarse (twin-level) growth detector with
clean seed behavior and a correct null; per-KC certification, active-rank
recovery, saturation robustness, neural faithfulness, and Tier-2 rate
reliability are each **not** earned, for reasons that are understood and,
for two of them (saturation null, RB4 wiring), fixable.
