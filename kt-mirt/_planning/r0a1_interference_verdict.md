# R0-A1 verdict: the interference read -- L1 refuted, dose floor certified, causal framing killed at this grain

Status: run 2026-08-10 overnight, direct-drive (scripts
`r0a1_study.py`, `r0a1_floor_cert.py`, `r0a1_kill_arms.py`; raw JSONs
under `outputs/a1/r0a1/`, numbers transcribed here because outputs/ is
never committed). Follows `ct0_power_result.md` (CT0: negative half
blocked by true-zero leak ~ the dose) and the frozen
`design/a1_design.md` v1.1. Seed discipline held throughout: the L1
study ran on held-out tuning seeds 100-102; because it produced
NO-WINNER, the default config was never tuned, and certification seeds
0-4 remained clean for the floor certification and the arms.

## 1. The L1 hypothesis is REFUTED (phase 1, tuning seeds 100-102)

CT0's hypothesis was that a stronger off-diagonal L1 would hold
true-zero cells at zero and separate the -0.02 edge. It does not. The
grid (KDD-shaped density, D=3, N=500, decoupling 0.90, reference dose
+0.05/-0.02; sep = |Gneg_mean|/band):

```
l1     ceil  band    zLeak   Gpos    Gneg     sep   negF1  FER
1e-3    500  0.0145  0.0140  0.0614  -0.0195  1.35  1.000  0.083
1e-3   1500  0.0172  0.0105  0.0608  -0.0194  1.13  0.333  0.000
3e-3    500  0.0125  0.0158  0.0608  -0.0190  1.52  1.000  0.167
3e-3   1500  0.0122  0.0102  0.0536  -0.0185  1.51  1.000  0.167
1e-2    500  0.0164  0.0181  0.0308  -0.0168  1.02  0.667  0.083
1e-2   1500  0.0341  0.0376  0.0307  -0.0172  0.50  0.000  0.167
3e-2    500  0.0482  0.0350  0.0237  -0.0164  0.34  0.000  0.083
3e-2   1500  0.0258  0.0172  0.0261  -0.0188  0.73  0.000  0.083
```

L1 shrinks the TRUE edges faster than the zeros (Gpos 0.061 -> 0.024
across the ladder while the band grows 0.014 -> 0.048). No config met
the pre-registered qualification (sign-correct all seeds at both
ceilings, |Gneg| within [0.5x, 2x] truth, FER <= 0.05). The leak is a
small-signal identifiability property of the objective at this grain,
not a regularization artifact: there is no L1 operating point that pins
zeros while sparing a 0.02 edge. The epoch-growth of the band at weak
L1 (0.0145 -> 0.0172) reproduces CT0's drift observation on fresh
seeds; the winner rule's ceiling axis (500 vs 1500) was the robustness
test and nothing survived it.

## 2. The certified detectable-dose floor (seeds 0-4, default config)

Because no tuning touched the default (l1=1e-3, ceiling 500), seeds 0-4
were clean for it. Dose ladder on the negative/interference edge,
pooled 5-seed NG band 0.0138, pre-registered per-dose bar (negF1 >=
0.75, negative sign in every seed, FER <= 0.05):

```
dose   Gneg     sep   negF1  posF1  FER    detected
0.01  -0.0112  0.82   0.400  0.833  0.150  no
0.02  -0.0192  1.40   1.000  0.833  0.150  no   (FER clause)
0.04  -0.0425  3.09   1.000  0.933  0.050  YES  <- certified floor
0.08  -0.0874  6.36   1.000  0.900  0.100  no   (FER noise)
```

**Certified floor: |g| = 0.04, 2x the reference dose.** Honest
decomposition: the negative COEFFICIENT is recovered near-truth from
0.02 up and the negative-half F1 is perfect from 0.02 up; the limiting
factor is the true-zero false-edge BACKGROUND (~5-15%; 1-3 crossings
out of 20 zero-cell draws per dose row), which is dose-independent in
expectation and made 0.04 the only row satisfying the strict
simultaneous clause (0.08 formally failed on 2/20 crossings). Two
consequences carried forward: (a) any per-edge existence claim on real
data needs multiplicity control across edges (BH-FDR, as the A4 gate
already uses per-KC) or a stricter band quantile; (b) the reference
dose -0.02 was always a guess -- the floor is now a measured property
of the estimator at this density/N, which is what the design asked
dose-response to deliver.

## 3. The confound arms at the certified dose (seeds 0-4)

**CT6 phantom-gamma sensitivity control: 5/5 seeds fabricate**
(per-learner p95 transfer magnitude 0.13-0.33 on the NULL twin vs
pinned 0.005-0.016 and band 0.0138). The pre-registered expectation
held: the tail metric bites on per-learner fabrication, and the gamma
pin is retained on the design's structural grounds (v1.1 reframing:
non-kill either way, sensitivity confirmed).

**CT3-iii shuffle-order: the KILL FIRED.** Refit on SYN-T-KG (dose
0.04) with each learner's cross-KC interleaving re-drawn (per-KC
internal order preserved; `battery.permute_cross_kc_interleaving`):

```
                 matched   shuffled(mean)  collapse ratio  bar
positive edge    +0.0564*  +0.0418         0.741           <=0.10  FAIL
negative edge    -0.0425   -0.0347         0.815           <=0.10  FAIL
```

(*matched Gpos from the floor-cert JSON at dose 0.04.) Destroying the
temporal lag leaves ~3/4 of both signed magnitudes intact with correct
signs in every seed. Mechanism: the shuffle preserves each learner's
per-KC practice COUNTS, and a uniform re-interleaving preserves the
EXPECTED number of source-practices preceding a typical target attempt;
only the lag structure dies. The estimator therefore reads signed
DOSE-ASSOCIATION (how much source practice a learner accumulated),
not temporal-causal transfer. The ~19-26% drop is at most the
lag-driven component at this grain, and nothing certified tonight
separates it from noise.

## 4. What G1 can now claim (and what it cannot)

- SUPPORTED (certified, D=3, KDD-shaped density, N=500, eff/edge
  ~4.3k): per-edge SIGNED DOSE-ASSOCIATION -- practicing source KC A is
  associated with raised (facilitation) or lowered (interference)
  performance on target KC B, sign recovered seed-consistently,
  magnitude near-truth, with a detectable-dose floor of |g|=0.04 for
  the negative half (positive half robust from +0.05 with sep ~4).
- KILLED at this grain (pre-registered arm, design's accepted
  association-framing fallback now in force): the CAUSAL-TEMPORAL
  reading "B moved BECAUSE A was practiced earlier in time". All A1
  claim language from here on says association, not influence-as-cause.
- UNCHANGED from CT0: the EdNet-shaped (multi-tag) density remains a
  genuine per-density identifiability limit (coefficient collapse under
  co-tagging); Junyi-graph external alignment is the positive half's
  future external check.

## 5. Consequences for the certification matrix (not run tonight)

1. Re-baseline the negative reference dose to the measured floor
   (0.04) for CT1+ cells, reported AS the floor, never as "the" effect
   size; keep the ladder so the floor is re-measured per density.
2. Add per-edge multiplicity control (BH-FDR across off-diagonal
   cells) to the existence read; the ~10% zero-cell background makes
   uncontrolled per-edge claims untenable at full K.
3. CT2+ claim wording drops to signed association (section 3's kill
   fallback). A lag-isolating estimator (e.g. the matched-minus-
   shuffled CONTRAST as its own statistic with a permutation null) is
   a candidate NEW arm if the causal reading is ever to be revived;
   logged as an idea, unvalidated, speculative.
4. R0-A1's stationarity re-derivation half is CLOSED with "the default
   numbers stand": no stopping-time tune can rescue 1x-dose
   discrimination (section 1), and the certified floor makes the
   default trainer adequate at >= 2x dose.
