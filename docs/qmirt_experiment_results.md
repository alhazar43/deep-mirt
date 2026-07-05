# Q-MIRT (paper 2) experiment record

Raw record for the campaign defined in docs/qmirt_paper_plan.md (v1.1).
Entries append chronologically; gate rulings marked GATE. Numbers here are the
source of truth; the plan carries only re-registered kill thresholds. Code:
deep_irt/bench/_qm2_*.py (fresh build; recovered _qmirt_*.py are reference
only). Outputs: deep_irt/bench/outputs/qm2/.

Conventions: C = concepts (KC-dim), K = answer categories, N = learners,
T = steps, J = items. Seed-clustered stats; sign consistency next to means.
Negative results recorded, never re-litigated.

## Session 0 (2026-07-04 overnight, autonomous)

User authorized autonomous execution ("go play with it"); v1.1 checklist
defaults in force: doctrine = item-side invariance + occasion-specific
ability; FB-OFF identification default (R9); C4 demoted to appendix
validation; C5 three-way verdict space; JEDM lean.

Targets tonight, in order: G0 bridge (recovered venue-1 forecast reproduces),
_qm2_ build + adversarial code review, C1 misspecification certification
(matched vs mismatch generators, dynamic vs anchor-first banks), P1b spine
bridge (forecast gap + sparse A->B + obs-vs-pred growth scores, FB-OFF simple
structure, C=3). Stretch: venue-2-style battery preview at C=3.

### G0: bridge run -- GATE PASS (2026-07-04 ~23:30)

Recovered `_qmirt_forecast.py` rerun as-is, repo root, g_true=0.10, seeds
42/43/44, N=400, e1/e2=80/60, l1_G=0.01, FB off, RTX 4060 (~4 min).
Log: deep_irt/bench/outputs/qm2/g0_bridge_venue1.log.

- Active gap B = +0.5863 +- 0.1874 (no-G minus with-G forecast NLL; 3/3
  positive). Venue-1 record was +0.263 at the script's default weaker dose;
  direction, structure, and pass status reproduce.
- U specificity = -0.0002 +- 0.0003 and null gap = +0.0022 +- 0.0185:
  EXACT digit-level match to the venue-1 record (these arms are
  g-independent, so identical RNG streams reproduce them verbatim). Strong
  fidelity evidence for the transcript-replay recovery.
- Passive frozen-LSTM leg fails (-0.056), exactly as the campaign itself
  retracted in venue 4 (free-decoder confound). Script predates the
  retraction; its internal "Overall: FAIL" verdict refers to this dropped
  leg only. Not carried.

RULING: environment + recovered machinery faithful. Readout flip may
proceed; _qm2_ build underway (datagen + model written; per-learner z0
replaces the FB crutch for individuation, transition stays response-free).
### Pilot findings (P1b shakedown, N=100, ds=42, ms=0): three
identification lemmas + a metric ruling

The build was piloted iteratively; each failure was diagnosed to a mechanism
before fixing (exhaust discipline). All four are paper-grade material for the
identification note (P1 deliverable). Model file docstring carries the
lemmas; diagnostics in this record are reproducible from the pilot configs.

LEMMA 1 (free asymptote = always-on growth channel). With OU pull toward a
FITTED mu, stage 2 explained B's whole conditioning rise as mean reversion
(mu_B fitted high): G unidentified, no-G forecast arm still rises, all
ablation gaps ~0 or negative, null twin fabricated G. RULE: mu is a fixed
buffer at 0; every growth route must be practice-gated.

LEMMA 2 (free persistence = decay compensator). With fitted rho < 1 on
monotone data, decay-down-per-step and G-up-per-step cancel on the
conditioning window: the null twin fabricated G (+0.03) as a decay
compensator and the no-G arm spuriously decayed (null gap +0.14 WRONG
direction). Monotone data cannot identify decay separately from gains. RULE:
rho frozen at 1 on monotone beds; the decay/OU variant only enters beds with
non-monotone identification content.

LEMMA 3 (gain-form misfit launders into G). With CONSTANT per-practice
own-gain against decelerating true gains, stage 2 sculpted the residual from
practice-count bases: fabricated G[B,A]=+0.041 WITH G[B,U]=-0.102 (a zigzag
fit; diagnostic dump shows z_hat_B non-monotone against smooth truth). The
positivity variation in the schedule (A-doses 12/4/20) is necessary for
identification but also supplies the sculpting basis if the gain family
cannot express deceleration. RULE: own-gain gap-scaled (mastery ceiling,
venue-4 M2, sanctioned in plan C2); the same fitted practice-gated ceiling
gates transfer. After the fix the null residual dropped to G_BA=+0.02 with
a NULL forecast gap (no predictive content; the venue-2 pattern).

METRIC RULING (forecast certification is read on the SCORE scale).
[MECHANISM SENTENCE RETRACTED 2026-07-05: the "NLL punishes the with-G arm
under true transfer" fragility story was built on sign-inverted NLL
readings (see G0.5 finding 1) and is void; on fixed code NLL and score
gaps agree in direction. Score-scale remains primary on rendering-contract
grounds only.] Primary certification metric =
forecast expected-score error gap (proportion-of-max units, bounded,
sharpness-insensitive), read as the matched-null PAIRED contrast (campaign
correction, carried). NLL reported secondary. Pilot after all fixes:
pos scoregapB=+0.0091, null=-0.0007, neg=+0.0145 (sign of G correct both
signed kinds), U ~0 everywhere. Structure correct; full grid running.

Generator calibration notes: transfer gap-gated (saturation multiplier) to
prevent ceiling blowout; pacing theta0=-1.5, rate=0.035, ceil=2.5 keeps B
mid-range at T_cond (transfer room in forecast); conditioning P-block
lengths 12/4/20 give positivity variation. Schedule co-scheduling with fixed
M:P ratio VIOLATES positivity even in the bridge bed (first pilot's
fabrication) -- the design rule now lives in the generator docstring.

### G0.5: adversarial code review -- GATE PASS WITH FIXES (2026-07-05 ~01:30)

Three fresh-context lenses (correctness/leakage Opus, psychometric fidelity
Opus, numerics Sonnet). Verdicts: all TRUSTWORTHY-WITH-FIXES; the causal
core (responses never in the transition, conditioning-only training masks in
both stages, arm construction, GPCM fidelity, gauge-free score metric) was
confirmed correct. Findings and actions, all applied:

1. CONFIRMED BUG (numerics): forecast_gaps called with swapped args -> every
   NLL gap printed before the review was SIGN-INVERTED. Fixed (signature
   aligned to call). RETRACTION: the pre-review pilot narratives quoting NLL
   gap directions are void, including the earlier "NLL punishes the with-G
   arm under true transfer" fragility mechanism. On the fixed design, NLL
   and score gaps AGREE in direction on all kinds. The score-scale metric
   stays PRIMARY on rendering-contract grounds (bounded, proportion-of-max
   units, the paper's language), not on the retracted fragility argument.
   Lemmas 1-3 are unaffected in substance (their evidence is
   parameter-level: fitted mu high, fabricated null G, the G[B,U] zigzag
   dump), but their gap-direction side-remarks are struck.
2. Measurement was practice (psych): reference-item administration fired
   own-gain in generator and model, so "B's only route is transfer" was
   false as stated and the fitted ceiling could act as a measurement-cadence
   asymptote. FIXED by design: reference items are STATE-INERT in both
   (ref_inert=True default); null-B is now literally flat (generator U rise
   = 0.0 exactly). Measurement-as-practice becomes a named robustness twin
   later.
3. Per-concept gauge (psych): simple structure leaves a location/scale gauge
   per concept; C1 metrics now center within concept (else gauge reads as
   item pathology -- the exact confusion C1 exists to avoid).
4. Position-bias detector was blind on infosched (leak+psych): now pools
   realized per-learner positions from seq_eff.
5. Matched-form transfer (both): generator transfer now uses the same
   unnormalized gap as the model; doses rescale to +-0.025.
6. Isolation check rewritten (leak+numerics): now a REAL leak test (G
   nonzero, zero-row concept must stay flat; positive control that the G
   route moves its target; asserted). Deviation 0.0 exact under rho=1.
   The old check could never fail on a G-leak and its "1e-8" was rho-pinned
   flat-path residue, not an isolation proof.
In-sample growth agreement (leak MINOR) noted: the conditioning M0-vs-M3
obs/pred exhibit is a fit check; the held-out C2 read is the forecast-window
score error (now stored per arm per cell).

### Design finding: bounded interference (2026-07-05)

Symmetric ceiling-gating made the interference twin saturate at the response
floor (neg score gap exactly 0.000: B sank to where both arms predict
identical zeros). Interference is now gated by the mirror quantity, you can
only lose what you have built: positive transfer scales with (ceil-theta)+,
negative with (theta-floor)+, in generator and model alike; interference
beds start B high (theta0_B=+0.5). Post-fix pilot (N=100, ds42/ms0,
l1=3e-3): pos scoregapB=+0.0526 (nll +0.180, G_BA=+0.012 sign+), null
+0.0039 (~0, G_BA=-0.022 no predictive content), neg +0.0138 (nll +0.047,
G_BA=-0.017 sign+), U ~0 everywhere. All three certification legs live.
Known residue: L1 attenuates |G| (magnitude gauge-bound anyway; a
LASSO-then-refit debias arm is the upgrade if magnitudes are ever wanted);
in-sample growth gap inherits the attenuation.

### P1b: spine bridge grid v1 -- SUPERSEDED (pre-review code, inverted NLL
column, measurement-as-practice bed); v2 below.

### P1b spine bridge v2 -- GATE G1b: PASS (2026-07-05 ~02:10)

27 cells: {pos +0.025, null 0, neg -0.025} x data seeds {42,43,44} x model
seeds {0,1,2}, N=400, FB-OFF simple structure, e1/e2=120/80, l1_G=3e-3,
reviewed code. outputs/qm2/p1b/ (v2 = run2 log; superseded by run3 which
adds the debias arm, certification numbers unchanged by construction).

- POS (canonical sparse A->B): score_gap_B = +0.0410 +- 0.0089, 9/9
  positive; matched-null paired contrast +0.0389 +- 0.0072, 9/9; NLL gap
  agrees (+0.149 +- 0.028, 9/9); G sign correct 9/9; U gap +0.0005 (~0).
- NULL: score_gap_B = +0.0021 +- 0.0017 (~0); G_hat_BA = -0.0095 +- 0.0071
  (straddles zero, no predictive content).
- NEG (interference): score_gap_B = +0.0166 +- 0.0022, 9/9; paired contrast
  +0.0145 +- 0.0038, 9/9; G sign correct (negative) 9/9; NLL agrees.
- Within-learner trajectory rank 0.996 (deterministic matched bed; the
  informative version of this number comes from noisy/mismatch beds later).

RULING: the editor's blocking concern is RESOLVED on the canonical case;
the campaign's sparse A->B sign reversal was the response-feedback pathway
(R9), not the simple-structure readout. Kill thresholds to re-register from
these runs (P1b contract): score_gap_B paired contrast > 0 in >= 8/9 with
null |gap| < 0.005; G sign 9/9 at |g|=0.025 dose.
Known rendering debt: L1 attenuates predicted conditioning growth (obs
+0.54 vs pred +0.29 under pos); stage-2b G-only no-L1 debias added (run3)
for rendering; certification always read from the L1 model.

### C1 bank certification v2 -- GATE G1: FAIL AS DESIGNED, INSTRUMENT WORKS
(2026-07-05 ~02:15)

45 cells: 5 twins x {dynamic, static_early(W=30), oracle} x 3 seeds, N=400,
mixed schedule T=90, within-concept-centered metrics. outputs/qm2/p1/.

- DISCRIMINATION DOES NOT SURVIVE JOINT DYNAMIC CALIBRATION even on the
  matched twin: alpha rank rho = 0.046 (dynamic) vs 0.815-0.912 (oracle
  refit on true theta, same data). Location recovers (d_rho 0.86 dynamic).
  The information is in the data; the joint fit discards it -- paper 1's
  disease signature WITHOUT a shared head. Alpha ranks are scale-gauge-free
  within concept, so this is genuine non-recovery, not gauge.
- static_early (anchor-first, W=30) fails BOTH alpha (0.02-0.07) and d
  (0.51-0.58): too little spread/motion in the early window. Anchor-first
  as naive early-window calibration is NOT the fix at this design.
- The refit-discrepancy instrument DETECTS the failure truth-free: delta =
  0.34-1.16 across twins (healthy would be << 0.15 in paper-1 terms). C1's
  self-certification works even though the bank fails.
- infosched: d_rho collapses to ~0 under informative administration and the
  REPAIRED position-bias detector flags it at -0.38 (strongest of all
  twins; matched sits at -0.10). Detector validated.
- Refit-on-own-states (run3): PARTIAL only. a_refit 0.19-0.48 vs oracle
  0.76-0.91; the fitted states are the bottleneck. Not a certifiable
  recipe alone. Boundary on the instrument: delta is LOWEST on infosched
  (0.34) where even the oracle fails (0.29, range restriction under
  adaptive administration) -- delta detects slack, not information absence.

### C1 closing tests (2026-07-05 ~03:30): the real diagnosis is
incidental parameters, and the rendering debt was optimization budget

STATIC-COHORT CALIBRATION FAILS TOO: a rate-0 cohort (no motion at all),
full window, joint static fit -> alpha rho 0.014/0.128/0.194 across seeds
(d_rho 0.53-0.74), while the oracle refit on the same regime recovers high.
So the alpha failure was NEVER motion contamination: every failing arm is a
JOINT fit with free per-learner person parameters, and the only working arm
knows theta.
MECHANISM: HYPOTHESIS ONLY, editor-downgraded 2026-07-05 (was asserted as
the incidental-parameters/Neyman-Scott problem). Three live rivals the
first write-up ignored: (a) INSTRUMENT CONFOUND, the oracle refit ran 300
epochs at lr 2e-2 on 2J params vs the joint fits' 150 at 5e-3 on the full
problem, so theta-knowledge is confounded with budget and conditioning;
(b) MAGNITUDE, classical JML discrimination bias is O(1/T) ~ 1% at T=60-90
and cannot alone produce a 0.90 -> 0.05 rank collapse; (c) RANGE
RESTRICTION / low-Fisher, the location-recovers-scale-fails signature, and
alpha was never re-run at the converged budget that fixed the rendering.
DECIDING EXPERIMENT (morning queue 1): matched twin, budget and LR held
fixed, vary only theta-knowledge: per-concept quadrature MML vs converged
joint JML vs oracle. MML~oracle>>JML = incidental parameters; JML~oracle =
budget; all-fail = range restriction.
CROSS-PAPER NOTE (contingent on the diagnosis): paper 1's separate-head
alpha (~0.94) rode encoder-amortized abilities, which act like
marginalization across learners; if the incidental-parameter reading wins,
that is the continuity story, and D3's recipe is a marginal (or given,
pre-calibrated) bank with dynamics fit on top; an amortized-z0 encoder is
the alternative individuation.
RECONCILIATION DEBTS (editor audit): record ranges must be regenerated from
persisted cells (dynamic matched alpha mean 0.046 HIDES a -0.273 seed, sign
consistency must be reported; oracle range as-persisted 0.68-0.90 not
"0.815-0.912"); the long-epoch cell OVERWROTE the 3-seed oracle-bank
summary (p1b_oracle now n=1), so the oracle-certification default run must
be re-persisted at >= 3 seeds.

ORACLE-BANK P1B (causal-chain test): with TRUE items frozen, certification
is clean (pos score gap +0.0434 +- 0.0106, 3/3; null ~0) but the growth
rendering compression PERSISTS (obs +0.542 vs pred +0.285) -> the bank was
not the (sole) cause. DISAMBIGUATION: one cell at e1/e2=500/300 epochs
collapses the growth gap 0.259 -> 0.086 and lifts G_BA 0.0148 -> 0.0192
(toward true 0.025), certification unchanged. The rendering debt is mostly
UNDER-OPTIMIZATION of the per-learner-z0 geometry at the 120/80 default,
not model class. Morning queue: converged-budget rerun (or plateau-based
stopping) for all rendering numbers; certification numbers are
budget-insensitive (9/9 at both budgets).
Note: oracle-bank NULL cells fabricate G_BA ~ -0.039 at l1=3e-3 with ~0
predictive content; watch under converged budgets and consider the L1 sweep
before reading any magnitude.

GATE G1 RULING (as amended by the editor audit): the bank certification
gate correctly FAILED the joint calibration paths, truth-free detection
works within its mapped boundary, and the failure is NOT motion-caused; the
MECHANISM is unresolved pending the fixed-budget MML/JML/oracle experiment.
C1 passes as an INSTRUMENT, fails as a BANK SOURCE. G2/G3 may proceed on
frozen oracle banks meanwhile.

### Editor audit of Session 0 (2026-07-05 ~04:10) -- rulings adopted

Fresh-context Opus editor audited the night. Verdicts: G0 SOUND; G0.5 SOUND
(retraction was incomplete in-place, now annotated); G1b SOUND with scope
caveat (matched bed certifies best-case identifiability, not robustness;
the mismatched-gain-form arm is the live C3 threat and is queued); G1 FAIL
sound, mechanism OVERCLAIMED (downgraded above). Lemmas 1-3 ruled real
identifiability facts; "bounded interference" relabeled an engineering
patch, not a finding. MORNING QUEUE (editor's ordering, adopted):
1. Fixed-budget MML vs converged-JML vs oracle alpha experiment (settles
   the G1 mechanism; C2/C3 both ride on a certified bank).
2. Re-run C1 with oracle-refit budget/LR equalized (de-confound the
   instrument).
3. Regenerate the full seed-clustered p1 summary from cells, reconcile
   every quoted range, report sign consistency next to means.
4. Converged-budget P1b rendering rerun; re-persist oracle-bank
   certification at >= 3 seeds.
5. Mismatched-gain-form P1b arm (price the matched-form dependence).

### Morning-queue execution (2026-07-05 ~05:45-06:15): items 1, 2, 4 CLOSED

MML RACE (queue 1+2; outputs/qm2/mml_race.json; static cohort, budgets
equalized per the editor): mml a_rho=0.568+-0.062 slope 0.917; jml_eq
(300@2e-2) 0.470+-0.242 slope 0.730; jml_conv (1000@5e-3) 0.446+-0.216;
oracle 0.725+-0.021 slope 0.856. WIDE-COHORT prediction test (theta0 ~
N(0,1) calibration sample): mml 0.775+-0.037 vs oracle 0.802+-0.049,
statistical parity.

G1 MECHANISM RESOLVED, three components, in order of size:
(1) OPTIMIZATION BUDGET dominated the original collapse (0.05 -> 0.47 by
    equalizing budget alone; the editor's instrument-confound attack was
    correct);
(2) MARGINALIZATION genuinely helps (rank +0.10-0.12 over converged JML,
    slope de-attenuation 0.75 -> 0.92, seed variance 0.24 -> 0.06; the
    incidental-parameters effect is real but SECONDARY, and the original
    Neyman-Scott headline would have been an overclaim);
(3) COHORT SPREAD is the ceiling (theta0 sd 0.5 is range-restricted for
    slope information; sd 1.0 lifts MML to oracle parity).
CALIBRATION RECIPE, demonstrated: the bank comes from measurement-regime
data with adequate spread, calibrated marginally; then frozen. No true
abilities needed. C1 re-run at converged budgets + this recipe is the
remaining formality (queue 3 reconciliation still owed).

CONVERGED RERUNS (queue 4; p1b run4 at e1/e2=500/300, oracle-bank
re-persisted at 9 cells): certification unchanged and cleaner -- pos
score_gap +0.0401+-0.0090 (9/9, paired +0.0398), null +0.0003 with
G_hat_BA = -0.0004 (the null fabrication is GONE at convergence), neg
+0.0176 (9/9, G sign 9/9). RENDERING HEALED: obs vs pred growth +0.542 vs
+0.491 (gap 0.051) on pos, -0.371 vs -0.365 (gap 0.007) on neg; the
whole rendering debt was optimization budget. Oracle-bank: pos
+0.0428+-0.0064 (9/9), null clean.

PROVISIONAL KILL REGISTRATION (G1b contract, lead ruling, editor to
countersign): C3 pos paired contrast >= +0.02 with >= 8/9 positive and
null |score_gap| <= 0.005, |null G_BA| <= 0.005, sign 9/9 at |g|=0.025;
C2 growth agreement <= 0.08 proportion-of-max at converged budget
(measured 0.051 pos / 0.007 neg). All fits at converged budget (500/300 or
plateau-stopped) from here on.

STILL OWED (next session): queue 3 (regenerate seed-clustered p1 summary
from cells, reconcile quoted ranges, sign consistency next to every mean);
queue 5 (mismatched-gain-form arm, the live C3 threat); C1 rerun at
converged budget with the MML recipe; L1 sweep before any magnitude read.

REPORTING RULE (user challenge 2026-07-05 morning, "that's almost noise
level" -- a reviewer would say the same): certification gaps and agreement
residuals are NEVER reported as bare decimals. Every such number carries,
in the same breath: (a) the null-twin floor measured in-design (pos gap
0.0401 +- 0.0085 vs floor 0.0003 +- 0.0005, ~50-80x, 9/9; neg 0.0176 +-
0.0020, ~35x), (b) the relative forecast-error change (ablating G worsens
B forecast error 0.174 -> 0.214, +19%; interference +10%), (c) the
fraction of achievable window headroom captured (~60%, irreducible ordinal
noise floor ~0.147), and (d) category units for effects (obs growth +0.542
prop-max = +2.2 categories on 0-4; model tracks 91% of it; interference
-1.5 categories, 98% tracked). The absolute per-window gap is bounded by
the short masked window (18 A-steps), a design constant, not signal
weakness; window length and dose are P3 sweep axes. Binding on E1-E5.

### Scale sweep (day item 2, 2026-07-05 ~11:30) -- pattern HOLDS

Converged budgets, pos+null, 3 data seeds x model seed 0, tags
p1b_n1000/n2000/j40/j120. N-axis (j=12): pos score gap +0.0394 (N=1000),
+0.0387 (N=2000) vs +0.0401 reference; null floor tightens with N
(-0.0000 +- 0.0004 at N=2000, the 1/sqrt(N) behavior); G_hat stable
~0.0185. Bank-axis (N=400, per-learner variable sequences, administered
count fixed): j120 (360-item bank) +0.0398 +- 0.0046, indistinguishable
from reference and tightest; j40 shows one weak seed (+0.0240 +- 0.0201,
n=3) -- recheck at full 3x3 before any claim about that cell. Growth
rendering stable (gaps 0.052-0.066). CAT-driven administration deliberately
deferred to the positivity-audited design (C1 infosched lesson).

### Trajectory exhibit, mechanism vs free tracker (day item 5, ~13:00)

Full budget (500 epochs each), oracle bank frozen for BOTH models, same
frozen-link score scale, N=400 x 3 seeds. outputs/qm2/traj/.
- NULL TWIN: mechanism trace flat as truth (wobble <= 0.00005 every seed;
  truth = 0); the free tracker manufactures visible "learning" from
  response noise (wobble 0.0042-0.0097; step-to-step total variation 0.057
  vs ours 0.0001, truth 0). Report per-seed values, never the mean ratio
  (denominator ~0 explodes it).
- TRANSFER TWIN (the second leg, consult-mandated): the mechanism wins
  in-sample tracking too: within-learner rank 0.996 vs 0.734, RMSE 0.062
  vs 0.137, amplitude nearer truth; the tracker jitters around the trend
  (TV 0.067 vs truth's own 0.008).
HONEST CAVEAT (binding on the exhibit caption): this bed is MATCHED-FORM
for the mechanism; a correctly-specified parametric model beating a free
tracker on its own generator is the expected bias-variance outcome, not a
general victory. The mismatched-gain-form arm (editor queue 5, pending) is
where the honest trade appears: expect the mechanism to pay bias there
while the tracker stays robust. The exhibit's claim is variance-side only:
freedom costs wobble, and wobble reads as learning where none exists.
Figure prototype traj_exhibit.png sent for user sign-off (rendering
contract).

### Item drift + growth-score reliability (day items 3-4, ~14:00)

outputs/qm2/drift/. Full budgets, 3 seeds; consult-Q1/Q2 design.

ITEM DRIFT (v1 verdict: calibrated but BLUNT; integrity PASSES).
- Circularity control 0.96: fitted-state window refits are NO wider than
  oracle-state refits, so person misfit is not being read as item drift
  (the consult's disqualifying failure mode, avoided).
- The empirical null quantile is large even on ORACLE states: window
  refits see different theta ranges (low early, high late), and
  range-restriction bias moves refitted locations systematically. Exactly
  why the null is calibrated empirically, but v1 power is poor.
- The exposure regression (+0.51) is confounded by my sqrt(n) weighting,
  not evidence of retest leakage (there is none by construction). The R10
  detector needs the anchor-linked window-refit refinement before real
  data; queued.

GROWTH-SCORE RELIABILITY (the referee-first statistic): BELOW BAR at the
base design, and that is a finding, not a failure. Split-half
Spearman-Brown = 0.694 +- 0.056 (bar 0.80; the quick smoke's 0.800 was a
1-cell flattery). Cause is structural: homogeneous learning rates leave
little true between-learner growth variance, so observed growth is
noise-heavy at 8 reference items x 1 block per window. Validity
+0.10 +- 0.14 (the smoke's -0.26 was noise), attenuation-consistent.
Constructive framing for the paper: certified per-learner growth has a
measurable measurement price. E1's per-learner panels carry SEM bands and
this reliability number in the caption, per the consult ("a growth
trajectory with no error band is not believed").

CLOSURE (~15:00, variants + decomposition): hetrate and pooled-blocks
variants did NOT move the bar (SB 0.67 / 0.66) -- and the no-fitting
decomposition explains everything with classical theory, no bug and no
model deficiency:
- Reliability follows Spearman-Brown arithmetic on response-sampling noise
  exactly (predicted 0.52 unpooled / 0.68 pooled; observed 0.53 /
  0.66-0.69). The measurement-price FORMULA is therefore in hand:
  reliability 0.80 at this noise level needs ~4x the base measurement
  density (~16 reference observations per window half). A deployment
  design rule, quantified.
- Validity is CEILINGED by nature, not the model: in the sigma=0.15 bed
  the predictable share of between-learner growth variance is 5.6-8.8%
  (the rest is accumulated random walk); ceiling = corr(pred,true) x
  sqrt(reliability) ~ 0.24; measured +0.10 +- 0.14 is near-ceiling
  performance. No tracker can beat this ceiling either -- it can only
  overfit the walk in-sample, which is the wobble exhibit's other face.
  hetrate not helping confirms B's growth variance is gap+walk-driven;
  per-learner TRANSFER heterogeneity (a gamma trait, the PSI-KT bridge) is
  the bed that would raise the predictable share -- queued as a P3 axis.
Report rule: per-learner growth claims ship as growth score + SEM band +
reliability; per-learner growth PREDICTION claims always cite the
predictable-share ceiling of the regime.

### Redesign blueprint (2026-07-05 evening) -- adopted after two reviews

Author redirect: KT encoder + IRT decoder; growth MODELED from event
content (partial-credit outcome, spacing, item features, graph), not
detected; one-to-many mapping reconsidered; generative training
considered. Five-lane research sweep run (papers + repos; mapping /
modeled-growth / generative / repos / locked-phenomenon lanes; results in
the session log, citations pending P0 verification). Blueprint drafted,
attacked by the editor and the hostile psychometrician, folded to v1.1 at
docs/qmirt_blueprint.md. Decisive review outcomes: prediction-primary
inversion (ELBO demoted to appendix ablation + uncertainty
quantification); GATE B encoder honesty (the recognition network is a
response-to-transfer channel until the encoder-ON null/permuted twins
clear it); GATE C bank gate (frozen cross-loading discriminations must be
certified recoverable at C>1 before any one-to-many spend; C1 precedent);
the exact compensatory multidimensional GPCM emission is now specified,
NESTING Chapter 0's published unidimensional GPCM at C=1 (Chapter 0's own
Limitations names the multidimensional version as future work -- the
continuity claim is corrected accordingly); citation-class fix (Xu-Zhang /
Gu-Xu are discrete-DCM laws; the operative condition is the
confirmatory-MIRT anchor + rank condition); the projected-mastery
stopping study survives only as a controlled appendix with harm twin and
lower-credible-bound stopping. Plain-language state doc for the author at
docs/qmirt_plain_state.md.

## Session 1 (2026-07-05 evening, ultracode): blueprint gate execution

Code era: _qm3_ prefix (the blueprint's model class); _qm2_ stays frozen as
the certified instrument and regression target. Emission adopted exactly as
review-specified: discrimination VECTOR a_j with Q-row support, scalar step
thresholds b_{j,i} on the composite scale, category-k logit = k*eta -
sum_{i<k} b_{j,i}, eta = sum_c q_jc a_jc z_c; nests Chapter 0's GPCM at C=1
with d_i = b_i / a.

### GATE C PRE-REGISTRATION (written before any result)

Bed: C=3, J=36, K=5; per concept n_pure pure items, remainder two-concept
cross-loaders cycling AB/BC/AC; loadings lognormal(0,0.3); static
calibration cohort N=400, T=60 mixed administration, theta ~ MVN(0, R(r))
with unit variances (the identification constraint), r in {0.0, 0.3, 0.6};
anchor density n_pure in {1, 3, 6}; seeds {42,43,44}. Fits: (a) marginal ML,
7-node Gauss-Hermite per dimension, Cholesky-correlated prior, 200 epochs;
(b) oracle refit on true abilities (ceiling). Readouts, all within-concept-
column centered: pooled alpha rank rho over nonzero loading entries;
CROSS-LOADER RATIO recovery rho(log(a_jB/a_jC) fitted vs true) -- the
quantity every one-to-many attribution claim rides on; d location rank.

PASS RULE (cell level): alpha rank >= 0.85 AND ratio rank >= 0.80 AND both
within 0.10 of the oracle arm's value in that cell. GATE PASSES if the
realistic cell (r=0.3, n_pure=3) passes on >= 2/3 seeds; the full grid maps
the boundary either way. KILL: no density >= 3 passes at any correlation ->
the between-item readout is the paper's design; one-to-many spend stops
(blueprint Gate C).

### GATE C RESULT -- PASS (2026-07-05 ~21:40)

Grid 3 correlations x 3 anchor densities x 3 seeds, marginal ML (7-node
Gauss-Hermite per dimension, Cholesky-correlated prior, 400 epochs) vs
oracle refit, N=400 T=90. outputs/qm2/gatec/. Verdict per the
pre-registration: PASS at the realistic cell (r=0.3, n_pure=3): 2/3 seeds,
alpha rank 0.889 +- 0.040, cross-loading RATIO rank 0.913 +- 0.025 (the
attribution-bearing quantity), both above the oracle arm on alpha (0.824;
marginal integration's implicit shrinkage helps ranks). Boundary mapped:
ONE pure anchor per concept is insufficient at every correlation (1/3
everywhere); three is the knee; r=0.6 degrades ratio recovery (0.833 at
p3), the collinearity limitation stated in advance by the psychometric
review. CONSEQUENCE: the one-to-many program proceeds to V1; frozen
cross-loading discriminations are certifiable under the marginal recipe
with >= 3 pure anchors per concept and moderate concept correlation.
NESTING CHECK also passed: the vector-loading emission reproduces the
qm2/Chapter-0 GPCM to 1e-10 on converted banks (_qm3_model.py __main__).

LEAK CAUGHT PRE-RUN (self-review while briefing the reviewers): Gate B's
"full window" variant as first coded read the ENTIRE sequence including
masked forecast responses into the person parameters; fixed to resolve to
the conditioning window before any Gate B cell ran. Recorded because the
class of bug is exactly what Gate B exists to police.

### qm3 code review verdicts and amendments (2026-07-05 ~22:20)

Three lenses, no correctness bug in the core: emission bit-identical
across generator/calibrator/model (and matches the blueprint spec),
Gauss-Hermite marginalization exactly correct, gather indexing verified,
the pre-run leak fix independently confirmed as the one hard leak (Gate
B's full-window variant would have read masked forecast responses; fixed
before any cell ran). Amendments and corrections, adopted:

1. GATE C IS CONDITIONAL ON KNOWN R (math lens, HIGH). The calibrator was
   handed the generating ability correlation as its prior; the
   certification and the shrinkage-above-oracle both benefit. Misspecified
   (independence-prior) and plug-in-estimated-R arms are coded and queued;
   the Gate C claim does not travel until they report.
2. PASS-RULE AMENDMENT (math lens, MEDIUM). The coded oracle-relative
   criterion is one-sided (MML not worse than oracle by more than 0.10);
   the pre-registration's "within 0.10" read two-sided. AMENDED to
   one-sided, recorded here, because penalizing the estimator for beating
   a noisy per-item oracle is senseless; for the audit trail, at the gate
   cell the two-sided rule also holds for the ratio metric (MML 0.913 vs
   oracle 0.943) and fails only in the flattering direction for alpha
   (MML 0.889 ABOVE oracle 0.824).
3. INERT MODEL-SEED CORRECTION, RETROACTIVE (numerics lens, HIGH honesty).
   Deterministic zero/constant inits plus full-batch Adam made the model-
   seed loop near-inert in the qm3 free path AND in this week's qm2 grids:
   every "9/9 (3 data x 3 model seeds)" claim in this record is honestly
   "3/3 data seeds with ~3 deterministic replicates". No verdict flips
   (margins vs floors were 35-80x and sign-consistent across the REAL
   replication unit), but every seed count in prose must read n=3 until
   reruns with noise-seeded inits (now in the qm3 code: init noise under
   torch.manual_seed makes the model-seed dimension real going forward).
4. Gate A carries 2N extra person degrees of freedom (lambda, gamma) that
   qm2 never had (leakage lens, MEDIUM): the running Gate A is evaluated
   against its pre-registered BANDS, and a matched control with lambda =
   gamma = 1 frozen is queued to attribute any null-floor shift.
5. Hygiene batch applied: quick-mode outputs namespaced (no clobbering of
   official summaries), quick smoke now targets the actual gate cell,
   dead code and variable shadowing removed, full-cohort invariant
   documented, threshold docstring drift fixed.

### GATE A PRE-REGISTRATION

The _qm3_ model with encoder OFF, one-hot Q (between-item), constant gain,
run on this week's certified forecast beds must reproduce the v2/converged
p1b results within seed noise: pos score gap in [0.02, 0.06] with 9/9
positive and paired-null contrast positive 9/9; null |gap| <= 0.005 and
|G_BA| <= 0.01; neg sign 9/9. Kill thresholds for the new class are then
RE-REGISTERED from these runs (nothing inherited automatically).

### GATE B PRE-REGISTRATION (encoder honesty)

Encoder ON (LSTM recognition network producing per-learner z0, lambda,
gamma posteriors; identification constraints E[lambda]=E[gamma]=1), null
twin and permuted twin, gamma free, both inference variants (conditioning-
window-only; full-sequence). FABRICATION RULE: on the null twin the
population transfer effect must satisfy |score gap| <= 0.005 AND the
PER-LEARNER transfer distribution must not hide it (95th percentile of
per-learner |gamma_n * G_BA| effect on forecast score <= 0.01); on the
permuted twin, routing must follow truth as in Gate A. KILL: the encoder
fabricates transfer where the response-free transition could not -> the
redesign is unsound; the v1.1 explicit model stands (blueprint Gate B).

### GATE A RESULT -- PASS; GATE B RESULT -- FAIL (2026-07-05 ~23:10)

GATE A (encoder off, frozen converted bank, converged budgets, honest
replication note: 3 data seeds x 3 noise-seeded model inits from the
review fix): pos score gap +0.0422 +- 0.0085 (all cells positive; inside
the pre-registered [0.02, 0.06]), null +0.0000 +- 0.0013 with G_BA
-0.0046 (inside bands), neg +0.0173 all-positive with sign-correct G.
The qm3 class reproduces the certified instrument. Kill thresholds for
the class re-registered from these cells (same bands).

GATE B (encoder on, gamma free, both windows): FAIL per pre-registration.
Population nulls are clean (+0.0005 / -0.0001) -- and the per-learner
tail metric the pre-registration insisted on shows why population means
cannot be trusted: p95 |per-learner transfer effect| = 0.0232 (early
window) and 0.0167 (conditioning window), both above the 0.01 ceiling,
riding fabricated null G_BA of -0.0173 / -0.0060. The recognition
network manufactures per-learner phantom transfer from null data. The
editor's blocking finding, instantiated as a gate, fired.

### GATE B2 PRE-REGISTRATION (written before running)

Surgical variant to localize the fabrication wire: encoder proposes z0
and lambda ONLY; gamma pinned at 1 (population transfer only). Same
beds, windows, budgets, and fabrication criteria as Gate B (null
population |gap| <= 0.005; p95 per-learner effect <= 0.01; the
per-learner effect under gamma=1 reduces to headroom heterogeneity times
population G, so the criterion binds through the fabricated-G route).
READINGS: B2 PASS -> amortized inference is honest for initial states
and learning rates, and the per-learner TRANSFER-ABILITY trait is the
uncertifiable wire at this design; PSI-KT's gamma trait is thereby
annotated as exactly the quantity a response-driven encoder cannot be
trusted to report (a paper-grade boundary). B2 FAIL -> the encoder
branch is unsound wholesale; the free-parameter/marginalized path
stands alone. Queued in the same chain: the Gate C misspecified-R and
estimated-R arms; the Gate A matched control with lambda = gamma = 1
frozen (attributes the extra-freedom null-floor shift).

### Gate C misspecification arms -- the known-R caveat DISSOLVES
(2026-07-05 ~23:40; outputs/qm2/gatec_misspec/)

At r=0.3: independence prior costs little (ratio 0.862 vs 0.913 known-R);
the one-iteration EAP plug-in estimates the ability correlation almost
exactly (Rhat 0.30 vs true 0.30; 0.55 vs 0.60) and its refit recovers
ratio 0.872. At r=0.6 the estimated-R arm MATCHES known-R (0.846 vs
0.833, within seed noise). RECIPE, now unconditional: calibrate under an
independence prior, estimate R from EAP correlations, recalibrate once.
Gate C's certification travels without the known-R assumption. (The
review's HIGH caveat, answered with data the same evening.)

STOPPED CHAIN NOTE: the follow-up chain was externally stopped mid-run
(~23:35): misspec arms COMPLETE (above); Gate A matched control ~5 cells
in (numbers tracking Gate A closely; lambda/gamma freedom shows no
null-floor shift so far); Gate B2 not started. Remainder relaunches on
one command.

### GATE B ATTRIBUTION FLIP -- the traits fabricate, not the encoder
(2026-07-06 early; ACTRL complete 27/27 PASS on population bands; B2
null/w20 chunk complete)

Null-twin per-learner p95 by configuration: traits PINNED (lambda=gamma=1,
encoder off) = 0.0073 PASS with fabricated G -0.0011 (~0); free traits
(encoder off, the Gate A config) = 0.0306 FAIL, G -0.0046; encoder+all
traits = 0.0167-0.0232 FAIL; encoder with gamma cut but lambda live (B2
early-window chunk) = 0.0234 FAIL, G -0.0086. READING, replacing Gate B's
original attribution: the recognition network is NOT the fabrication
source -- per-learner TRAIT MULTIPLIERS are, however estimated (free
parameters are worst; amortization partially shrinks the pathology --
the amortization-as-regularization story again). Only trait-pinning
passes, with margin, which also validates the 0.01 ceiling as
well-calibrated. Mechanism: Lemma 3 at the person level -- trait freedom
lets the optimizer sculpt noise into trait spread plus a small fabricated
population G, fanned into per-learner phantom transfer by headroom gating;
population means never see it.
BOUNDARY (paper-grade): per-learner ability LEVELS (z0) are certifiable;
per-learner learning-rate and transfer-ability TRAITS are not, at this
design, under any estimation posture tested -- which annotates PSI-KT's
interpretable traits (gamma included) as exactly the uncertified objects.
CANONICAL MODEL CONFIG going forward: traits pinned (the ACTRL
configuration) = the certified core; trait heterogeneity only via z0.
Gate A note: its population bands PASSED but its free-trait config fails
the per-learner criterion; the re-registered kills bind to the ACTRL
configuration.

### GATE B2 COMPLETE -- the encoder branch SURVIVES in one configuration
(2026-07-06; full table, 36 cells)

null_w20: p95 0.0234 FAIL (G -0.0086) | null_wcond: p95 0.0068 PASS
(G -0.0011, ACTRL-clean) | pos_w20: gap +0.0340, G +0.0307, U-routing
smeared (-0.0065) | pos_wcond: gap +0.0374, G +0.0278, U-routing clean
(-0.0006).
FINAL GATE B RULING (amending the first attribution twice, each time on
evidence): (1) the per-learner transfer-ability trait gamma is
UNCERTIFIABLE here -- every gamma-bearing configuration fails the
per-learner fabrication ceiling; (2) the learning-rate trait lambda IS
certifiable, but only through AMORTIZED inference reading the full
conditioning window -- as a free parameter (p95 0.0306) or
information-starved (early-window, 0.0234) it fabricates; (3) with gamma
cut and full-window amortization, the encoder configuration passes at the
trait-pinned level (0.0068) with clean transfer certification and clean
routing on the pos twin. Amortization here is not merely regularization;
at fixed data it is the difference between pass and fail.
FROZEN CONFIG for V1/V2: encoder-amortized z0 + lambda over the full
conditioning window, gamma pinned at 1 (population G only), traits-pinned
variant retained as the audit baseline. PSI-KT annotation sharpened: its
per-learner transfer-ability trait is exactly the object that fails
certification under every posture tested; its other traits would need the
full-information amortized treatment to survive.

VOCABULARY RULING (user correction 2026-07-05): the training objective is
NEVER described as "the GPCM likelihood" or maximum-likelihood calibration.
It is a next-category prediction objective (cross-entropy through the
GPCM-shaped readout; soft-over-hard-WOL is a recorded campaign ruling).
Per-response it coincides numerically with a conditional NLL given the
trajectory; the estimation posture does not, and the project's claims never
lean on likelihood semantics: measurement calibration is marginal ML on
measurement-regime data (C1 recipe), dynamics are prediction-trained, and
everything read off the model is certified by instruments, not by
estimation theory. "Likelihood" is reserved for the marginal bank
calibration. Binding on all prose and agent briefs.
