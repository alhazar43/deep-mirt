# DuoLingo Mini, an exploration plan

Status, PROMOTED FROM EXPLORATION TO ACTIVE TRACK on 2026-06-11.
Scope, rl/-side adapters plus this docs file, with minimum or zero
edits to the public ma-irt repo. This document is self-contained.

## 0. Decisions log

**2026-06-11, priority locked.** The Duolingo / SLAM track is the
active build priority (user decision). D1 (SlamAdapter plus the first
MAGPCM run on real Duolingo en\_es data) is merged, ACC 0.682, QWK
0.374, binary-collapsed AUC 0.773 on 2,593 learners. The next
concrete milestone is D2 (second language track es\_en plus LSTM and
logistic-regression baselines). The OrdRec headline track (E5 on
Eedi) is paused pending the user downloading the Eedi NeurIPS 2020
Task 3+4 csvs, after which it runs largely unattended; it is not
abandoned, only deprioritized behind the time-sensitive Duolingo
play.

Branch note. `feat/duolingo-mini` is now the live working branch for
this track and was brought current with `feat/ordrec` at the D1 merge
(`e960be9`), so it carries the full OrdRec code (E1 through E4.7) plus
the D1 SLAM adapter plus this plan. Future D-milestones land here.

**Still open, not yet decided by the user.** Three forks were
presented alongside the priority question and remain open. The
recommendation on each is recorded so the eventual call has context.
(1) Paper structure, recommend two papers two venues, OrdRec RL
recommender to IJAIED and the ordinal-calibration SLAM result to BEA
or EDM where the Duolingo assessment team publishes; deferred until
D4 and E5 results are in. (2) The mixed-K item-bank feature, the only
proposed change to the public ma-irt repo, recommend HOLD to keep
ma-irt clean since single-K adapters cover the current work. (3)
Duolingo Research outreach, recommend plan it but gate the cold
contact on D4 (synthetic recovery) so the email leads with two pieces
of evidence and names the S2A3 authors; held for now.

**Reversible config defaults taken without a user call.** SLAM
coercion stays at K=3 (the clean all-wrong / partial / all-correct
mapping) for D2; D1 found the all-wrong category is rare at 2.7
percent of exercises, handled with class weights and reported
honestly, not by silently shifting thresholds. en\_es was the D1
track; es\_en is D2.

## 1. The three threads

This plan merges three research streams into one ranked program.

Thread 1, Duolingo collaboration. How can ma-irt connect to Duolingo,
whose Duolingo English Test (DET) is the closest operational cousin to
ma-irt's ambition, a mixed-format computer-adaptive assessment with an
IRT-anchored scale. Duolingo's PUBLISHED calibration line is
dichotomous IRT (2PL, 3PL) and their own June 2026 paper (S2A3,
arXiv 2606.07364) names the graded-response or partial-credit
extension as required future work. AutoIRT (Sharpnack et al. 2024)
and BanditCAT (Sharpnack et al. 2025, PMLR v264 workshop) are
binary-only, verified by adversarial deep research (see the
verification update in Section 5). The operational DET already
ingests continuous and polytomous grades under an undocumented model
class, while multi-grade formats are decomposed into binary
pseudo-items. The SLAM 2018 dataset (Settles et al. 2018, BEA at
NAACL) is public, CC0, and immediately usable.

Thread 2, mixed-format positioning. ma-irt brings a new look to both
knowledge tracing and item response theory, a single theta maintaining
a student profile across mixed-format tests over time. Classical
concurrent calibration already places multiple-choice and
constructed-response items on one theta with the generalized partial
credit model (Muraki 1992) under marginal maximum likelihood (Bock and
Aitkin 1981), and dynamic IRT already evolves theta over time (Wang,
Berger and Burdick 2013). Deep knowledge tracing already conditions
ability on raw response sequences (Piech et al. 2015, Zhang et al.
2017, Ghosh et al. 2020). The defensible novelty is the amortized,
encoder-emitted union of all three, a calibrated, format-agnostic,
sequential ability estimator that needs no per-occasion re-fit, with
recovery demonstrated on synthetic K=4 at r_theta about 0.96 and
r_beta above 0.95.

Thread 3, reality check. After the imagination, ma-irt still has hard
limits. Without a major overhaul, what does the current pipeline
enable or disable. The standing rule is minimum ma-irt edits, so every
direction is classified by whether it runs with no ma-irt change, a
tiny additive change, or not at all short of an overhaul.

## 2. What ma-irt actually is

An encoder times decoder framework. Encoders, DKVMN (the paper
headline), LSTM, Transformer (SAKT-style). Decoders, GPCM (ordinal
K >= 2), Rasch, Binary, Softmax. One global n_categories K per model
instance. The GPCM decoder consumes only an integer ordinal label in
0 to K-1 and never inspects how the response was elicited (forward is
`(questions, responses)`, ma-irt/models/base.py:136). It emits theta,
alpha, and beta, with the scale fixed by the alpha-norm rescaling in
GPCMLogits (ma-irt/models/components/irt.py). The separated ability
pathway (separate_theta) and item_conditioned flag are the
architectural signatures. There is no forgetting model, single-session
assumption holds, no response-time or behavioral inputs, item
parameters are per-item-ID so unseen items are cold-start blind under
the default learned embeddings, the Transformer caps at max_seq_len
256, and n_traits > 1 is unvalidated (rotation indeterminacy
unresolved).

The rl/ extension adds the machinery DuoLingo Mini reuses. Data
adapters subclass OrdinalDatasetBase (rl/src/ordrec/data/base.py),
materialize a four-file artefact (sequences.json, metadata.json,
coercion_artefacts.json, optional q_matrix.npz, schema in
rl/src/ordrec/data/schema.py), and a bridge
(rl/src/ordrec/data/ma_irt_bridge.py) wraps any materialized adapter
into ma-irt's SequenceDataset with no ma-irt edit. The Eedi adapter
(K=4 distractor ordering) and EdNet adapter (K=4 correctness-by-time
quadrant) are the canonical templates. A ma-irt config
(configs/ordrec_eedi_k4.yaml) reads n_questions and n_categories off
the artefact metadata at DataModule.build time and trains via the
stock scripts/train.py.

## 3. The headline story

ma-irt gives knowledge tracing a calibrated, recoverable measurement
scale, and it gives item response theory a longitudinal deep state in
one forward pass. Because the GPCM decoder consumes only an ordinal
label, any grader that compresses a signal to 0..K-1, a
multiple-choice key, a distractor rank, a dictation error count, an
essay rubric score, lands on one maintained theta. This is the
amortized union of concurrent calibration and dynamic IRT that no
deep KT model has assembled. We demonstrate it on real Duolingo
language data (SLAM 2018) for prediction credibility and on synthetic
mixed-format data for falsifiable recovery and cross-format
concordance.

The honest boundary. SLAM has no ground-truth theta, so on real data
we report prediction metrics (AUC, log-loss) and cross-format theta
concordance, never parameter recovery. Recovery, the central IJAIED
result, stays on synthetic data where theta, alpha, and beta are
known. The single-theta claim is currently a single-construct claim,
since n_traits > 1 is unvalidated. The two tracks are complementary,
synthetic carries the recovery and linking claims, SLAM carries the
real-Duolingo credibility, and the related-work positioning carries
the framing.

## 4. The enable and disable matrix

Every direction classified against the standing rule of minimum
ma-irt edits.

| Direction | Status | Mechanism and cost |
|---|---|---|
| Rubric-scored open responses as GPCM items | ENABLED-as-is | Decoder forward is `(questions, responses)` in 0..K-1, never inspects elicitation. Pure adapter. Zero code anywhere. |
| SLAM per-token to per-exercise ordinal coercion | ENABLED-via-rl-adapter | New adapter aggregating token errors to a K=3 label, mirrors EdNetAdapter. About 100-150 lines, zero ma-irt edits. |
| Response-time integration beyond EdNet | ENABLED-via-rl-adapter | New coercion table (e.g. K=6 fast/medium/slow). Zero ma-irt edits. Days. |
| Multi-session gap-token forgetting | ENABLED-via-rl-adapter | Reserve one ID as a session-boundary marker. Honest caveat, a learned perturbation not a decay model. Zero ma-irt edits. |
| Format-effect estimation (per-format alpha) | ENABLED-via-rl-adapter | Post-hoc join of the recovered alpha/beta table with format metadata. Zero ma-irt edits. Days. |
| Encoder-invariance probe (DKVMN vs LSTM vs Transformer) | ENABLED-as-is | Cross-encoder agreement of recovered theta/alpha/beta. Uses existing encoders and the evaluate harness. Zero new code. |
| Cite-and-compare positioning | ENABLED-as-is | Related-work prose only. Zero code. |
| Cross-test anchoring via merged ID space | NEEDS-minor-additive-ma-irt (FLAGGED) | Merged-ID adapter, config n_questions raise, retrain. No ma-irt source edit but a research-design commitment. Weeks. |
| Mixed-K item banks in one model | NEEDS-minor-additive-ma-irt (FLAGGED) | Per-category -inf mask threaded through trainer._flatten_mask and GPCMLogits.forward, plus a per-item K table in the adapter schema. About 20-50 lines touching two ma-irt files. |
| Item cold-start on unseen items | BLOCKED-without-overhaul | No feature-based item pathway, q_embed sized to the training bank. Appendix only. |
| D>1 multi-dimensional traits | BLOCKED-without-overhaul | Rotation indeterminacy unresolved, out of scope. Appendix only. |
| Response time as a continuous model input | BLOCKED-without-overhaul | Encoder forward signature is fixed. Appendix only. |
| Adaptive FORMAT selection in OrdRec (true action axis) | BLOCKED-without-overhaul | Env action space is flat item IDs. Only the inflate-IDs workaround fits, with a mandatory retrain. Appendix only. |

The ranked plan draws only from the first three status classes plus
the two explicitly flagged minor-additive items. The four blocked
directions live in Appendix B.

## 5. Company context, the AI-era pressure and what it means for the pitch

The plan targets Duolingo Research, the Assessment and Psychometrics
group (LaFlair, Liao, von Davier, Attali, Lockwood, Belzak, Cardwell),
not the consumer product org. That distinction matters more now than
when this plan was first drafted, because the company is under acute
AI-era pressure and that pressure reshapes which research is fundable
and welcomed.

The backlash. Von Ahn's April 28 2025 AI-first memo, which announced
phasing out contractors and accepted "small hits on quality" for
generation speed, triggered a severe public reaction, a social-media
purge on May 17 2025, and two CEO walk-backs through April 2026.
Stock fell roughly 78% from its May 2025 peak, DAU growth halved into
2026, and full-year 2026 bookings guidance sits near 10.5%. The
lasting liability is reputational, Duolingo is now read as a company
that traded measurement-grade quality for automation speed.

The reframe. The struggles do not make collaboration less likely,
they sharpen which collaboration lands. Two pressure points raise the
value of measurement-grade research. First, the efficacy measurement
gap, the strongest external RCT (Kim et al. 2026, Studies in Second
Language Acquisition, n=183) shows comparable-to-classroom, not
better, and Duolingo has no published calibrated longitudinal ability
estimate from its own data, only engagement metrics (DAU, streaks,
lesson completions) that measure habit not proficiency. Second, the
AI-content calibration gap, 148 AI-generated courses shipped with
documented quality errors and 100-plus curriculum specialists laid
off, yet there is no published benchmark comparing AI-generated to
human-built course learning outcomes. Both gaps are exactly where a
calibrated ordinal IRT contributes.

The radioactivity rule. Anything that reads as "another AI feature" is
poison post-backlash, the brand wound is precisely over-automation.
DuoLingo Mini must present as measurement and validation science, the
discipline that repairs the trust the AI pivot spent, never as a new
generative capability. This is why the collaboration-surface ranking
leads with an ordinal extension of AutoIRT (a calibration-rigor
contribution, pitchable on public data with no inside access) rather
than with the data-hungry efficacy-infrastructure frame, which is
stronger but needs a partnership to land. The two framings reconcile
in Section 6.

### Adversarial verification update, 2026-06-11

A deep-research pass (101 agents, 3-vote adversarial verification per
claim, full report at docs/cleanup/_det_deep_research_report.md)
stress-tested the pitch's load-bearing claim. The outcome sharpens the
pitch in three ways and corrects two errors.

What survived, narrowed and stronger. AutoIRT and BanditCAT are
verifiably dichotomous-only (grades coded G in {0,1}, binary
log-likelihood, zero occurrences of polytomous, partial credit,
graded response, ordinal, or GPCM across both papers, 3-0 verifier
votes on exhaustive full-text search). Deployment was the DET practice
test only, two vocabulary item types. Most consequentially, Duolingo's
own June 2026 follow-up (S2A3, arXiv 2606.07364, Sharpnack, Tsigler,
Lockwood, Nydick, von Davier) states the implementation "is restricted
to dichotomous items under the 2PL and 3PL models" and names the
graded-response or partial-credit extension as required future work.
The gap is therefore author-acknowledged, current, and still
unpublished. The pitch fills a hole the contacts themselves have named
in print, and the S2A3 co-authorship confirms the assessment group is
active at Duolingo.

What was refuted and must change in our framing. The blanket claim
"Duolingo's IRT layer handles only binary responses" is false at the
operational layer. Dictation's continuous grades already feed EAP
scoring under unnamed "appropriate IRT models," and the August 2025
scoring whitepaper adds an explicit Polytomous grade category (0, 1,
2, ..., n) for Interactive Listening comprehension. No public document
names the operational model class, so the honest claim is that the
operational layer is undocumented, while the PUBLISHED calibration
line is dichotomous-only and multi-grade formats (C-test blanks,
Interactive Reading parts) are decomposed into separate binary
pseudo-items rather than modeled with partial-credit IRT. That
decomposition is precisely the within-item dependency structure a
GPCM treatment targets.

The urgency consequence. S2A3 shows the classical polytomous
extension is on Duolingo's own roadmap, so the window for pitching
static GPCM calibration alone is closing, possibly within months. The
defensible differentiation is what their roadmap does not cover, the
deep sequential encoder that carries a longitudinal calibrated theta
with recoverable item parameters, demonstrated by D1 on their own
public corpus and by the E4.7 dynamic-tracking result. The pitch
leads with ordinal plus longitudinal, not ordinal alone.

Citation corrections adopted throughout this document. BanditCAT
appeared in PMLR v264, the Proceedings of Large Foundation Models for
Educational Assessment, an ICML-affiliated workshop, not the ICML
main conference. Vocabulary-in-Context is a typed constructed-response
format, binary-scored. Read Aloud was removed from the DET in the
July 2025 revision and should be cited as a 2024-era example. The
"discrete and continuous grades" future-goal sentence attributed to
the Interactive Reading paper was never located verbatim and must not
be quoted; the verified anchors are the writing whitepaper's
"currently no IRT-scored items" hedge and the S2A3 future-work
passage.

## 6. Ranked opportunities

Each carries its reality-check status and a one-line
collaboration-leverage note from the struggles lens. The experiment
ordering is by feasibility and is unchanged. The collaboration view,
which research surface to pitch first, runs in the opposite direction,
the rank-1 collaboration surface is the ordinal AutoIRT extension, and
the experiments below are the evidence that makes that pitch credible.
The reconciliation, experiment D1 (SLAM) is the demonstration that
earns the rank-1 ordinal-AutoIRT pitch, a working ordinal calibration
on real Duolingo data is what lets the cold contact name a gap in
their own 2PL/3PL work and show it already solved on public data.
Blocked items appear only in Appendix B, never here.

1. SLAM 2018 adapter plus DKVMN+GPCM training on real Duolingo data.
   ENABLED-via-rl-adapter. The zero-coordination real-data anchor.
   Payoff, first real Duolingo corpus result, comparable to SLAM-era
   baselines. Cost, one adapter and one config, days, zero ma-irt
   edits. Leverage, this is the demonstration that earns the rank-1
   collaboration pitch, a working ordinal calibration on their own
   public data is what makes the ordinal-AutoIRT extension credible
   to the Assessment team without any inside access.
2. Synthetic mixed-format recovery and convergent-validity experiment.
   ENABLED-via-rl-adapter. The load-bearing scientific evidence.
   Payoff, turns the single-theta slogan into a measured result on
   ground truth. Cost, a synthetic generator and an analysis script,
   days to two weeks, zero ma-irt edits. Leverage, ground-truth
   recovery is the methodological backbone of the ordinal-AutoIRT
   pitch, calibration rigor is the trust-repair currency post-backlash.
3. Cite-and-compare positioning. ENABLED-as-is. Payoff, credibility
   with the Duolingo group and a sharper contribution claim. Cost,
   prose only. Leverage, this is the prose that names the gap in their
   own 2PL/3PL work (AutoIRT, BanditCAT) and frames ma-irt as the
   ordinal generalization, the rank-1 collaboration surface in words.
4. Rubric-scored open responses as GPCM items. ENABLED-as-is. Payoff,
   the cleanest statement of format-agnosticism, answers Duolingo's
   stated discrete-plus-continuous-grade gap. Cost, pure adapter,
   hours for a synthetic-rubric demo. Leverage, ordinal rubric scoring
   is the format their AI-generated writing and vocabulary items need,
   it speaks to the uncalibrated AI-content gap, not to a new feature.
5. Encoder-invariance probe. ENABLED-as-is. Payoff, a robustness
   claim no deep KT paper can make. Cost, zero new code, days.
   Leverage, an invariance result reads as measurement discipline,
   the opposite of the over-automation the brand is wounded over.
6. Cross-test anchoring via merged ID space. NEEDS-minor-additive,
   FLAGGED. Payoff, amortized cross-test linking, unexplored in deep
   KT. Cost, merged-ID adapter, config raise, retrain. Weeks.
   Leverage, cross-test linking is the technical core of the
   efficacy-infrastructure frame, the rank-2 collaboration surface
   held for a second conversation once a data partnership is on the
   table.
7. Format-effect estimation as post-hoc analysis. ENABLED-via-rl-
   adapter. Payoff, a measured format effect on alpha from a
   sequential tracer. Cost, metadata plus an analysis script, days.
   Leverage, per-format discrimination is directly the AI-content
   calibration question, do AI-built item formats behave like
   human-built ones, framed as quality control not generation.
8. Mixed-K item banks in one model via a per-category mask. NEEDS-
   minor-additive, FLAGGED. Payoff, true mixed-K banks, the realistic
   data shape. Cost, about 20-50 lines across two ma-irt files, in a
   reviewed worktree. Leverage, mixed-K is the realistic DET item-bank
   shape, supports the calibration-rigor story but is the last
   experiment to undertake and only if D4 motivates it.

## 7. What DuoLingo Mini concretely is

Four pieces, all on the rl/ adapter side plus this doc.

(a) SlamAdapter, rl/src/ordrec/data/slam.py, subclassing
OrdinalDatasetBase, mirroring EdNetAdapter. Loads SLAM 2018 from
Harvard Dataverse (CC0), aggregates per-token binary edits into a
per-exercise ordinal label (K=3 all-wrong/partial/all-correct, or K=4
by error-count thresholds), persists thresholds to
coercion_artefacts.json fit on train only, writes the four-file
artefact.

(b) ordrec_slam_k3.yaml, cloned from ordrec_eedi_k4.yaml, consumes the
artefact with n_questions and n_categories read from metadata at build
time, trains DKVMN+GPCM via scripts/train.py, evaluates AUC and
log-loss via scripts/evaluate.py against SLAM-era LSTM and logistic
baselines. Prediction metrics only, no recovery claim.

(c) Synthetic mixed-format experiment, the recovery and convergent-
validity flagship. Per-student histories interleave two formats
coerced to one ordinal scale with shared anchor items and known
(theta, alpha, beta). Scale fixed by the existing alpha-norm
identification. Three falsifiable targets, cross-format theta
concordance high under one model and collapsing to chance when anchors
are removed, alpha and beta recovery within each format at the
synthetic ceiling, held-out format predicted better than
format-specific baselines.

(d) Positioning write-up citing AutoIRT, BanditCAT, BERT-IRT,
Jump-Starting, ML-Driven Assessment, and Deep-IRT, framing ma-irt as
the ordinal generalization those binary-IRT methods name as needed.

Relation to OrdRec E5, parallel exploration, not a replacement.
OrdRec is the RL recommender over a frozen ma-irt env. DuoLingo Mini
is the measurement-side companion that strengthens the theta OrdRec
recommends on, adds SLAM as a second real dataset next to Eedi and
EdNet, and opens the Duolingo collaboration surface. They share the
adapter framework, the bridge, and the frozen-env machinery.

Open question, the outreach sequencing. The struggles lens settles
the order. Post a preprint first that demonstrates ordinal calibration
on public data (D1 SLAM plus D4 synthetic recovery), so the contact
starts from evidence not from a proposal. Then make cold contact that
names the published gaps in their own work by author, Sharpnack,
LaFlair, Yancey, and von Davier on the 2PL/3PL-only AutoIRT and
BanditCAT line, and pitches the ordinal extension as the rank-1
surface, which needs no inside data. Hold the efficacy-infrastructure
frame (longitudinal theta as proof the app teaches) for a second
conversation, it is the stronger frame but it requires a data
partnership and reads as a bigger ask. Carry the trust-repair
co-publication angle as a sweetener, a peer-reviewed measurement paper
with named academic co-authors directly counters the post-backlash
narrative that Duolingo traded quality for speed, but it strengthens
the ordinal-AutoIRT or efficacy pitch rather than standing alone.

## 8. Milestones

D1, SLAM adapter and first real-data run. Days. Write SlamAdapter and
its unit test (following test_ednet_adapter.py), add
ordrec_slam_k3.yaml, train DKVMN+GPCM, report AUC and log-loss on the
en_es track. Proves ma-irt runs end-to-end on real Duolingo data with
zero ma-irt edits. Runnable on the current pipeline immediately. This
is also the collaboration anchor, a working ordinal calibration on
Duolingo's own public corpus is the demonstration that makes the
rank-1 ordinal-AutoIRT pitch credible, so D1 feeds the cold contact
directly.

D2, SLAM second track and baselines. Days. Add es_en, fit LSTM and
logistic-regression baselines, tabulate AUC and log-loss. Proves the
real-data result is competitive, not just runnable.

D3, synthetic mixed-format generator. About one week. Build the
interleaved two-format generator with shared anchors and known
parameters. Proves the data design exists with ground truth.

D4, recovery and convergent validity. About one week. Run recovery,
measure cross-format theta concordance, run the no-anchor control.
Proves the single-theta claim is a measured result, not a slogan.
This is the IJAIED core.

D5, positioning and related work. Days, in parallel. Write the
cite-and-compare section. Proves credibility with the Duolingo group.

D6, encoder-invariance probe. Days. Cross-encoder agreement of
recovered parameters. Proves the scale is a property of the data and
the GPCM identification, not one architecture.

D7 (flagged), cross-test anchoring. Weeks. Merged-ID adapter,
retrain, Stocking-Lord agreement of a held-out single-bank theta to
the shared metric versus a no-anchor control. Proves amortized linking
inside the encoder.

D8 (flagged, needs the per-category mask edit), mixed-K banks. Worktree
ma-irt edit of about 20-50 lines plus the adapter K table, then a
mixed-K recovery run. Proves genuine mixed-K coexistence. Deferred
behind D1 to D6 and undertaken only if D4 motivates it.

## 9. Risks

SLAM is not a proficiency test. It reflects beginner app learners in a
short window, not high-stakes assessment, so theta on SLAM is not DET
theta. Mitigation, frame SLAM as a real-data prediction demonstration,
keep the DET bridge as motivation not as a claim.

No ground-truth theta in SLAM. Recovery cannot be evaluated on real
data. Mitigation, recovery lives on synthetic data, SLAM reports
prediction metrics, stated plainly.

Binary-IRT mismatch with Duolingo. Their deployed pipeline is 2PL and
3PL binary, a GPCM-is-better claim may meet skepticism without their
partial-credit items. Mitigation, claim a generalization that handles
the cases they name as open, not a replacement.

Cold-start gap. ma-irt is item-ID-bound, AutoIRT and BERT-IRT solve
cold-start by item features. Mitigation, acknowledge it as out of
scope and an explicit complement, not a competitor.

Single-construct caveat. One theta is legitimate only if the formats
load on one construct. Mitigation, test residual format dependence
after one theta, flag a method or testlet factor if it appears
(Rodriguez 2003, Bradlow, Wainer and Wang 1999).

Scale asymmetry and unidimensionality. Duolingo operates at a scale
our synthetic experiments do not, and DET reports four subscores
implying multidimensionality. Mitigation, keep claims to the
single-construct, single-bank regime DuoLingo Mini actually tests.

Confounded format claim. Without anchors in the histories, format and
ability are confounded. Mitigation, the no-anchor control is a
required experiment, a passing claim requires the correlation to
degrade to chance when anchors are removed.

## 10. References

Attali, Runge, LaFlair, Yancey, Goodwin, Park, von Davier (2022). The
Interactive Reading Task, transformer-based automatic item generation.
Frontiers in Artificial Intelligence.

Bock and Aitkin (1981). Marginal maximum likelihood estimation of item
parameters. Psychometrika.

Bradlow, Wainer and Wang (1999). A Bayesian random effects model for
testlets. Psychometrika.

Ghosh, Heffernan and Lan (2020). Context-aware attentive knowledge
tracing (AKT). KDD.

Kolen and Brennan (2014). Test Equating, Scaling, and Linking, 3rd ed.
Springer.

McCarthy, Yancey, LaFlair, Egbert, Liao, Settles (2021).
Jump-starting item parameters for adaptive language tests. EMNLP.

Muraki (1992). A generalized partial credit model. Applied
Psychological Measurement.

Piech et al. (2015). Deep knowledge tracing. NeurIPS.

Rodriguez (2003). Construct equivalence of multiple-choice and
constructed-response items. Journal of Educational Measurement.

Samejima (1969). Estimation of latent ability using a response pattern
of graded scores. Psychometrika Monograph.

Settles and Meeder (2016). A trainable spaced repetition model for
language learning (half-life regression). ACL.

Settles, Brust, Gustafson, Hagiwara, Madnani (2018). Second language
acquisition modeling (SLAM shared task). BEA at NAACL.

Settles, LaFlair, Hagiwara (2020). Machine learning-driven language
assessment. TACL.

Sharpnack, Mulcaire, Bicknell, LaFlair, Yancey (2024). AutoIRT,
calibrating IRT models with automated machine learning. arXiv
2409.08823.

Sharpnack, Hao, Mulcaire, Bicknell, LaFlair, Yancey, von Davier
(2025). BanditCAT and AutoIRT. PMLR v264, Proceedings of Large
Foundation Models for Educational Assessment (ICML-affiliated
workshop).

Sharpnack, Tsigler, Lockwood, Nydick, von Davier (2026). S2A3.
arXiv 2606.07364. Names the graded-response or partial-credit
extension of the 2PL/3PL line as required future work.

Stocking and Lord (1983). Developing a common metric in item response
theory. Applied Psychological Measurement.

Vie and Kashima (2019). Knowledge tracing machines. AAAI.

Wang, Berger and Burdick (2013). Bayesian estimation of dynamic item
response models. Annals of Applied Statistics.

Yancey, Runge, LaFlair, Mulcaire (2024). BERT-IRT, accelerating item
piloting with BERT embeddings and explainable IRT. BEA at ACL.

Yeung (2019). Deep-IRT, making deep learning based knowledge tracing
interpretable. EDM.

Zhang, Shi, King, Yeung (2017). Dynamic key-value memory networks for
knowledge tracing (DKVMN). WWW.

## Appendix A. Datasets

SLAM 2018, Harvard Dataverse doi 10.7910/DVN/8SWHNO, CC0, mirrored at
github.com/NYUCCL/duolingoSLAM. About 7M tokens, 6000+ learners, three
tracks (en_es, es_en, fr_en), three exercise formats. Per-token
binary, coerced to per-exercise ordinal.

Half-Life Regression, github.com/duolingo/halflife-regression, 13M
traces, binary recall. Low fit for ordinal, useful only as a
longitudinal-state comparison point. Not in the D1 to D6 plan.

## Appendix B. Later, requires overhaul

These are BLOCKED-without-overhaul and appear here only.

Item cold-start on unseen items. Needs a feature-based item encoder
branch added to the model.

D>1 multi-dimensional traits. Needs rotation constraints or post-hoc
alignment, a methodology question, not a code change.

Response time as a continuous third input channel. Needs the encoder
forward signature, embedding layer, and data pipeline changed.

Adaptive FORMAT selection in OrdRec as a true action axis. Needs a
format-conditioned world model, or the inflate-IDs workaround with a
mandatory retrain.
