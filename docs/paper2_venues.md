# Paper 2 venue assessment

The finding. In neural IRT and amortized knowledge tracing, the
interpretable item parameters are recovered at different rates from
prediction training. Ability and step thresholds recover fast,
discrimination slow, in the order their Fisher leverage on the
prediction predicts. Giving the low-leverage parameter its own
representation (decoupling) or access to the inferred ability (dynamic
conditioning) strengthens its recovery. The framing is representation
learning. The contribution is a property of the learned representation
in the finite regime that real use occupies, not a challenge to IRT
identifiability.

## Update after the overnight runs (read with the comparison below)

The runs the comparison listed as pending or assumed are now in, and they move the
analysis. The consolidated firm-ground story is in docs/paper2_story.md. The corrections
that bear on venue choice:

- The finite-sample sweep is done and shows a clean lawful scaling of the deficit with
  data scarcity, the oracle gap grows from +0.10 at N=800 to +0.37 at N=200. This was
  the ICLR and representation-learning track's main missing piece, so that track is
  stronger than the 3.0 in the table.
- The deficit is largest at K=2 and flat above, not growing with category count. The
  graded many-category angle that the EDM and knowledge-tracing rows leaned on is gone.
- The intervention is weaker than the table assumed. Decoupling via a separate static
  embedding is null on rank at every sample size at 8 seeds. Dynamic conditioning
  survives but is small and fragile, +0.04 to +0.08. The robust contribution is the
  characterization, not a fix.
- The under-encoding is real after all, as an asymmetry rather than an absence. A
  powered cross-validated probe over five hundred items decodes step thresholds at R^2
  0.96 and discrimination at 0.69, with 1.4% of the embedding variance along the
  discrimination direction. The earlier Q=60 artifact is replaced.
- The leverage ordering is now a formal but conditional proposition
  (docs/paper2_leverage_proposition.md), discrimination is least informative for
  typically discriminating items and the ordering inverts for under-discriminating ones.

Bearing on the recommendation. Psychometrika remains the best pure fit, but it requires
the psychometrics-theory framing the standing project rule rules out. With the scaling
law now anchoring the machine-learning track and the standing rule favoring it, the
recommendation shifts to lead with the machine-learning framing at a representation
learning or learning dynamics workshop, building toward ICLR, with Behaviormetrika as
the measurement backup. Full reasoning in docs/paper2_story.md.

## Comparison table

| Venue | Fit | Best framing | Key objection | Evidence still needed | Nearest deadline |
|---|---|---|---|---|---|
| Psychometrika (Behaviormetrika fallback) | 3.5 | Parameter-specific measurement validity of amortized estimators. Lead with attenuation slope and oracle dissociation, anchor with a formal leverage proposition. | Asymptotic consistency, is this a finite-sample artifact unworthy of theory. | Pending N x K x budget sweep, a formal leverage proposition, mechanistic account of the non-monotone decay, real-data check. | Rolling. Behaviormetrika rolling, median 7-day first decision. |
| ICLR / NeurIPS representation-learning (amortized inference) | 3.0 | Shared amortized encoder under-serves its low-Fisher readout, worsens with training, fixed by decoupling. Never say IRT, GPCM, psychometrics in title or abstract. | Effect size modest at convergence, non-monotone decay has no mechanistic why, kappa-flat result weakens the leverage story. | The generality sweep, gradient-conflict analysis of the decay, cross-validated non-psychometric control, a second backbone, a downstream cost. | ICLR 2027, abstract approximately September 2026. |
| EDM / JEDM | 3.0 | When prediction training hurts parameter recovery. Lead with the non-monotone curve as the practitioner warning, leverage as the diagnostic, decoupling as the fix. | All experiments synthetic, EDM currency is real student data. | One real-data graded replication, a binary arm, a downstream task cost, the sweep. | EDM 2027 approximately January-February 2027. JEDM rolling. |
| ML learning-dynamics (HiLD, UniReps, MechInterp, main tracks) | 2.5 | Low-leverage parameters suffer in amortized prediction, a non-monotone recovery phenomenon and its fix. Oracle gap and decay as the lead, IRT as the analytic tool. | Effect modest, mechanism named not derived, education domain a liability. | A quantitative scaling law, a representation-level account of the decay (CKA or probing), a stronger non-IRT replication at scale, architecture-independence with CIs. | NeurIPS 2026 workshops approximately August 2026, ICLR 2027 main track approximately September 2026. |
| AIED | 2.5 | Neural IRT recovers discrimination poorly and it worsens with training, the fix is architectural and free. Avoid amortized, Fisher, representation learning in the abstract. | Synthetic only, modest convergence gain, vocabulary foreign to half the committee. | A real-data experiment, a downstream task validation, a binary or transformer replication. | AIED 2027 approximately February 2027. |
| Knowledge Tracing subfield (umbrella, scored separately above) | 2.5 | Measurement-validity consequence, oracle dissociation as the hook. | No prediction-accuracy comparison table, GPCM is a niche within a niche. | Standard-benchmark AUC table, binary replication, real-data IRT ground truth, the sweep. | Tracks the EDM and AIED cycles above. |

## Ranked recommendation

### 1. Psychometrika, with Behaviormetrika as the named backup

Psychometrika is the venue where every load-bearing finding is read in
its own language and counts in full. Leverage-ordered recovery is a
genuine gap closed in closed form, which is the kind of derivation this
readership rewards. The oracle dissociation separates the
data-information argument from the architectural one in one clean number.
The attenuation slope maps directly onto classical measurement
attenuation, vocabulary the audience already owns. The nearest published
ancestor, Urban and Bauer 2021, and the March 2026 Deep CAT paper show
the journal continues to publish mathematically grounded neural-IRT
work. The non-psychometric encoder control, which is a liability or a
translation burden everywhere else, here reads as the move that lifts the
paper above a narrow knowledge-tracing study.

The reasons to lead here over the ML venues are concrete. The effect size
that reads as modest to an ICLR reviewer reads as a measurement-validity
result to this audience, where a compressed discrimination slope has a
direct interpretation. The education and psychometric framing that is a
liability at NeurIPS is the native register here. Submission is rolling,
so there is no closed window forcing a weaker venue. The path to
acceptance is well-defined rather than open-ended, complete the pending
sweep, state the leverage result as a formal proposition, give the
non-monotone decay a mechanistic account, and pre-empt the asymptotic
objection by framing the finite regime as the operative one and noting
the non-monotone decay makes convergence to the oracle unlikely in
practice.

Behaviormetrika is the backup for one reason above the others. The 2025
special feature on bridging machine learning and psychological
measurement shows an editorial team that actively wants this exact
intersection, the acceptance bar is lower than Psychometrika, the
time-to-decision is fast, and the modest convergence effect is not
penalized. If Psychometrika returns a major revision that the evidence
package cannot yet satisfy, Behaviormetrika absorbs the same paper with
light reframing.

### 2. ICLR / NeurIPS amortized-inference thread (parallel ML track)

The amortized-inference gap literature is the natural ML home, and the
oracle dissociation and non-monotone decay slot into it directly. This is
the higher-ceiling, higher-variance option. It is ranked second because
the evidence the community demands is heavier and partly missing today,
the generality sweep is load-bearing, the kappa-flat result undercuts the
strongest leverage claim, and the non-monotone decay has no mechanistic
why yet. The right relation between the two targets is sequential, not
either-or. The same study supports a Psychometrika paper now and an ICLR
2027 paper once the sweep, the gradient-conflict analysis, and the
cross-validated control are in hand, with the abstract rewritten in
purely architectural terms.

## Venues to drop and why

- BJET. Education-technology applications, orthogonal to a parameter-
  recovery contribution. No path.
- AIED as a primary target. Structurally mismatched. Synthetic-only and
  theory-forward against a committee that wants a deployed system or
  real-classroom data, with vocabulary foreign to half the program
  committee. Revisit only with a real-data experiment in hand, and prefer
  EDM over AIED if going applied.
- ML learning-dynamics main tracks (NeurIPS, ICML, ICLR new-phenomenon
  slot). The effect is too modest and the mechanism named rather than
  derived to clear a bar set by grokking-scale results. The associated
  workshops (HiLD, UniReps, MechInterp) remain a possible secondary
  outlet once a scaling law and a representation-level account of the
  decay exist, but they are not a primary target now.
- EDM and the wider KT subfield as a first submission. The community
  currency is real student data and a benchmark AUC table, both of which
  the current package lacks, and the binary-IRT default makes the
  GPCM K-scaling result, the best numerical leg, apply to a corner case.
  Viable later as an EDM 2027 short paper or a JEDM submission once a
  real-data graded replication and a binary arm exist, but not the lead.
