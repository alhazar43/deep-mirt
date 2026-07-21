# A4 / G2 synthetic certification verdict — COMPLETE (both profiles)

Goal G2: per-KC growth beyond noise. This folds the frozen KDD-matched
verdict (`verdict_kdd_g2.md`, `verdict_kdd_working.md`) and the completed
EdNet-matched corroboration into one paper-grade certification over the
full synthetic evidence base: 4 twins x (5 slice seeds KDD, ~5 slice seeds
EdNet) + 4 twins x 3 model seeds each. EdNet-matched is the thin-density
profile (C=189 KCs, 6000 learners, median 2 opportunities per learner-KC,
mean tag arity 2.2, many short slices) chosen as the density-opposite of
KDD's few-huge-KC profile. Verdicts are seed-clustered per the frozen
design: a claim must be sign-consistent in every seed AND significant
seed-pooled. Scope and the two open decisions are at the end. EdNet is
Tier-1-capped by design (population corroboration only; ACT runs on
synthetic EdNet but never on the real EdNet bed).

Design + bars: `_planning/design/a4_design.md` v1.1 (CG1-CG10, section 5;
EdNet-relaxed bars in 5.1-5.2). Primitives: `src/kt_mirt/growth/report.py`.

---

## 1. Headline — the coarse existence detector is CERTIFIED and profile-robust

**The passive existence gate certifies as a twin-level (pooled) growth
detector on BOTH density profiles. It does NOT certify as a per-KC
detector on EITHER.** G2 asks for per-KC growth, so the claim must stay
split.

The calibrated decision — silent on the no-growth twin, firing on both
growth twins, in every seed — generalizes from KDD to EdNet exactly:

| Twin | KDD bed_pvalue x5 | EdNet bed_pvalue | decision |
|---|---|---|---|
| syn_ng  | .268 .640 .128 .973 .743 | .326 .551 .692 .222 .876 | NULL, both |
| syn_kg  | .001 x5 | .001 x5 | DETECT, both |
| syn_ns  | .001 x5 | .001 x4 (seed1 absent) | DETECT, both |

Zero p-value overlap on both beds: 5/5 ng calls null, 9/9 kg+ns calls
detect on EdNet. CG2 (null holds) and CG4a (EdNet passive existence
detection, p<0.001) both PASS. On EdNet the null twin is cleaner and
tighter than KDD (split-half gaps .042-.051 vs KDD .066-.098; BH 0/189
x5, BY 0). This is the strongest single G2 result and it survives a full
density inversion.

**One profile difference worth recording (CHECK A, verdict PARTIAL).**
The gate's *calibrated* separation is profile-robust, but the *raw*
`bed_stat` magnitude separation that looked so clean on KDD (ng <= 915,
kg/ns >= 2375, ~2.6x non-overlapping gap) does NOT carry to EdNet. There
the raw bands overlap heavily (ng 5977-6461, kg 5689-6247, ns 6238-7625);
4 of 5 kg stats fall inside the ng range. The gate is unaffected because
it always decides on the per-seed permutation p-value, not the raw stat —
but the paper must not sell "clean non-overlapping statistic magnitude" as
a general property. It was partly a KDD-density artifact. At thin density
only the permutation calibration recovers the separation, which is exactly
why the gate is built on it.

The permutation floor (p=0.001 = 1/(999+1)) caps detection strength on
both beds: the test says "growth present," never "how much."

---

## 2. Saturation limitation — GENERAL, severity is density-modulated (CHECK B, HOLDS)

CG6 (pre-registered null-under-saturation: the gate should FAIL to detect
growth once near-ceiling saturation destroys observability) **inverts on
both profiles**. The syn_sat twin fires the gate hardest exactly where it
must stay silent. This is a general property of the passive existence gate
under model misspecification near ceiling, NOT a KDD-density artifact.

Shared signature across beds:
- Gate fires p=0.001 x5 on both (identical to the true-growth twins).
- Stat inflated vs the true-growth twin in the same direction, driven
  distributionally (median kc_stat, not outliers): KDD median 10.4 vs kg
  3.1; EdNet median ~39-43 vs kg ~28-30.
- Same numerical-degeneracy fingerprint: `r_c_se` pinned to the 1e-6 ridge
  floor on saturated KCs (z ~ 2-3e5), `saturation.*_unsaturated` False for
  effectively 100% of the 189/515 KCs every seed.
- Split-half corroborates almost identically: EdNet gap 0.270-0.290, KDD
  0.259-0.292, both ~2.7-2.9x through the 0.10 tolerance while all three
  non-saturated twins stay inside it on both beds.

**Density modulates the severity, not the existence:**
- Raw `bed_stat` inflation ratio (sat/kg) is ~3.7x on KDD but only 1.32x
  on EdNet — the over-firing is much milder at thin density.
- Per-KC BH under saturation: flat 0/515 on KDD; sporadic [0,20,0,0,19] on
  EdNet — thin data lets a handful of KCs cross by chance/leakage.

Mechanism (unchanged from KDD): the no-growth reference model approximates
a saturating curve worse than the growth model, handing the growth model a
near-universal held-out NLL edge regardless of true dynamics. **Actionable
flag for the real beds:** near-ceiling / mastered KCs will trip the gate
spuriously; report saturation robustness as a general limitation whose
magnitude scales with density, and add a saturation-aware null or
near-ceiling down-weighting before trusting a fire on easy KCs.

---

## 3. Per-KC resolution — FUNDAMENTAL limit, NOT sparsity-fixable (CHECK C, FAILS)

The hypothesis that per-KC failure is a KDD-specific sparsity artifact a
different density would relieve is **rejected**. Per-KC certification fails
at BOTH density extremes, by comparable margins, via the same failure
signature. If sparsity were the cause, EdNet's median-2 density should have
diverged markedly (better or worse). It did neither.

Bank recovery rank_corr (bar 0.90; `passed=False` in every cell, both beds,
n_items=1512 both):
- kg: KDD 0.718-0.739 (mean .727) vs EdNet 0.718-0.770 (mean .750) —
  EdNet marginally HIGHER, the opposite of a sparsity-starvation story.
- ns: KDD .727 vs EdNet .767 (4 seeds).
- ng: KDD 0.766-0.789 vs EdNet 0.799-0.866 (highest cell in either bed).
- sat: both collapse near zero (KDD .07-.10, EdNet .09-.21).

Per-KC BH discoveries on the positive control (kg, needs >=60% for CG3):
- KDD kg 0/515 x5; EdNet kg 0/189 x5 — identical zero power despite far
  fewer opportunities per KC.
- ns: KDD sporadic [65,0,55,0,0] (2/5 seeds); EdNet [0,0,0,0] — flatter
  than KDD, no firing at all.

Both beds land in the same 0.70-0.80 rank-corr band, both fail 0.90 by the
same ~0.17-0.18 margin, both show near-flat BH discoveries. This is a
property of the estimator/test construction — a gauge/identifiability
limit documented elsewhere in the program — not a density artifact. K3 in
the design anticipated exactly this: an "identifiability floor at
EdNet-class density" that kills nothing and is itself reportable. CG3
(positive control) and CG5 fail on per-KC resolution on both profiles.

CG4b rate recovery (EdNet only): rank corr(r_hat, r_true) ~0.08 (-.06 to
.15), FAIL vs 0.6 — the K7 pre-registered floor finding, a reportable
density limit, not a kill.

---

## 4. Active posture (ACT) and neural audit — consistent across profiles

**ACT — direction, not per-KC magnitude, on both beds; rank recovery
markedly BETTER on EdNet.**
- syn_kg: fires on both (EdNet pop 0.127/0.160 >> 0.05). growing_rank_corr
  jumps from KDD ~0.27 to EdNet 0.502 (p0) / 0.599 (p1). act_p1 CLEARS the
  relaxed CG1a EdNet bar (0.5); p0 sits at it. This is the single biggest
  positive cross-bed difference — richer multi-tag co-occurrence at thin
  density gives ACT more rank signal.
- syn_ns: fires on both (EdNet pop 0.123/0.143); rank 0.394/0.493, just
  under the 0.5 CG1b bar (KDD 0.33/0.38 — EdNet better); overshoot 4%.
- syn_ng (null): here EdNet is WORSE. CG1 silence breaches on BOTH variants
  (act_p0 pop 0.009 ~at bar, p95 0.040 > 0.02 EdNet bar; act_p1 pop 0.028,
  p95 0.161). On KDD act_p1 was dead silent (p95 0.0004). ACT leaks on the
  null more at thin density — the discipline that made act_p1 the trusted
  variant on KDD does not hold on EdNet.

Net ACT: presence and direction recovered on both; per-KC magnitude not.
Rank recovery improves at thin density and clears the EdNet positive
control, but null-silence degrades — no ACT variant is clean on both the
null and the growth twins across both profiles simultaneously.

**Neural tracker (CG7-CG10) — PAS-N1, fails consistently and largely by
design, on both beds.**
- CG7 (untrained-null) 0/3, CG8 (drill contamination) 0/3, CG10 (direction)
  0/3 per twin on EdNet, matching KDD's 0/12 sweep. CG8/CG9 are the DESIGNED
  PAS-N1 failure (shared-state contamination and order-sensitivity); CG7's
  near-zero trained recovery and CG10's reconstruction violations (.36-.42
  EdNet, .21-.47 KDD) are beyond the designed set and worth noting.
- CG9 (order stress) is the one place EdNet is slightly less bad: partial
  passes appear (kg 1/3, ns 2/3) where KDD was flat 0/3 — thin, short
  slices give the order-invariance test less to break.
- PAS-N2 (factorized per-KC tracker) remains architecturally immune but has
  NO measured CG7-CG10 verdict in any cell on either bed — its audit
  superiority is a construction guarantee, not a result, and must be stated
  as such.

**Rate reliability (RB4):** the KC-level split-half leg that Tier-2 per-KC
rate reliability needs is dead code (`kc_level_split_half` / `rb4_kc_rates`
defined, never called) on both profiles. A harness wiring gap, not a null —
no Tier-2 per-KC rate claim can be certified from these artifacts. The
per-learner split-half leg is present and passes for the three non-saturated
twins on both beds.

---

## 5. Certification roll-up (both profiles)

| Twin | designed posture | KDD verdict | EdNet verdict |
|---|---|---|---|
| syn_ng  | all detectors silent/null | gate NULL, CG2 pass; act_p0 p95 marginal | gate NULL, CG2/CG4a-null pass; ACT breaches BOTH variants |
| syn_kg  | passive detect + ACT fire | pooled DETECT; CG3 FAIL (BH 0%, bank .73, rank .27); act_p1 fires | pooled DETECT; CG3 FAIL (BH 0%, bank .75, rate .08); ACT rank clears EdNet bar |
| syn_ns  | detect + misfit fire | pooled DETECT; CG5 FAIL (misfit 6%, bank .73) | pooled DETECT; CG5 FAIL (misfit fires everywhere ~60-74%, non-informative) |
| syn_sat | gate FAIL-to-detect (CG6) | FIRES x5, stat ~10955 (3.7x kg); CG6 INVERTED | FIRES x5, stat ~7883 (1.32x kg); CG6 INVERTED, milder |

Note the syn_ns misfit clause fails on both beds by OPPOSITE numeric
routes to the SAME conclusion: KDD under-fires (6-7% of growing KCs, needs
80%); EdNet over-fires (64-74% on growers AND 58-71% on the silent subset
that should carry none). At median-2 density the misfit flag fires almost
everywhere and is non-informative. The clause is not usable on either
profile.

---

## 6. Scope and caveats

- **Both synthetic profiles now closed.** KDD-matched (few huge KCs) and
  EdNet-matched (median-2, multi-tag) agree on every G2 headline. Real-bed
  behavior is a separate question; nothing here is a real-data result.
- **Seed-clustered** per frozen design; no single seed carries any verdict.
- **p=0.001 is a permutation floor** on both beds — detection maxed,
  strength not resolvable.
- **Bank recovery 0.70-0.80** under the 0.90 bar everywhere it is barred,
  both beds; **fundamental, not sparsity-fixable** (CHECK C).
- **PAS-N2 audit superiority is unmeasured** (construction claim only).
- **RB4 per-KC rate reliability is unwired**, cannot be certified.
- **EdNet is Tier-1-capped**; ACT never runs on the real EdNet bed.

---

## 7. What this certifies for the two open decisions (evidence, not a call)

**G2 net across both profiles: PARTIALLY CERTIFIED.** The passive existence
gate is a validated coarse (twin-level) growth detector — correct sign,
correct twins, seed-stable, and now robust across a full density inversion,
with a correct null. Per-KC resolution, active per-KC rank, saturation
robustness, neural faithfulness, and Tier-2 rate reliability are each NOT
earned, for reasons that are now understood on both beds.

### Decision A — pursue the fixable gaps toward per-KC certification
Evidence FOR:
- Two named gaps are wiring/method, not statistical nulls: RB4 KC-level
  split-half is dead code (a fix, not a study); the saturation false-fire
  has a clear mechanism and a named remedy (saturation-aware null /
  near-ceiling down-weighting).
- ACT per-KC rank recovery is not hopeless: it clears the EdNet positive
  control (act_p1 0.599 > 0.5), showing rank signal exists at density.
Evidence AGAINST:
- The core per-KC failure is the one thing that did NOT move under a full
  density inversion (CHECK C): identical zero BH power, same 0.70-0.80 bank
  band, both beds. It reads as a gauge/identifiability floor of the
  estimator, not a tunable deficit. No cheap knob is in evidence.
- Rate recovery (0.08) and misfit (non-informative both beds) are far from
  their bars; closing them is open-ended research, not a patch.

### Decision B — take the coarse detector to the real-data KDD pilot now
Evidence FOR:
- The one thing certified IS the coarse detector, and it is the most
  profile-robust result in the program (clean null, clean detect, both
  densities, seed-clustered). It is ready to be exercised on real data.
- Real KDD is the primary bed; the coarse existence claim is exactly a
  Tier-1 population claim, which the design licenses.
Evidence AGAINST / preconditions:
- The saturation false-fire MUST be handled first (design's
  positive-control-first ordering): real near-ceiling KCs will trip the
  gate spuriously, and on the real bed there is no ground truth to catch it.
- Any real-bed reporting must be scoped to coarse twin-level detection —
  no per-KC discovery claims, no rate claims, no neural-faithfulness
  claims — because none of those certified on either synthetic profile.

Both paths are live and mutually compatible (the saturation fix is a
precondition for B and a deliverable of A). The evidence is laid out; the
call is the user's.
