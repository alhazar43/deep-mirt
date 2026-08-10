# External landscape sweep (2026-08-10/11)

Author order: escape the internal-metrics loop by grounding the paper
externally — datasets, metrics, architectures, models, IRT literature.
Four parallel web-research agents; findings recorded here in full, then
read independently by a research-scientist and a psychometric-researcher
(reads appended), then compiled into the author brief.

---

## A. Datasets with external anchors

Central finding: real option-level response data and expert-authored
quality anchors almost never coexist publicly. Eedi is the one candidate
where both halves exist and are plausibly joinable.

### A.1 Access table

| Dataset | Anchor | Provenance | Size | Option-level? | Access |
|---|---|---|---|---|---|
| Eedi NeurIPS 2020 log | none in-file (raw option responses) | Eedi platform 2018-2020 | Tasks 1-2: 27,613 questions, 118,971 students, 15,867,850 answers; Tasks 3-4: 948 q / 4,918 students / 1.38M answers with images | YES: QuestionId, UserId, AnswerValue (1-4), CorrectAnswer, IsCorrect | direct zip dqanonymousdata.blob.core.windows.net/neurips-public/data.zip, CC BY-NC-ND 4.0; papers arXiv:2104.04034, arXiv:2007.12061 |
| Eedi "Mining Misconceptions" (Kaggle 2024) | expert misconception code per distractor | Eedi content team | 1,857 questions, 2,587 misconceptions, 7,428 options (4,338 labeled) | content only, no responses | kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics |
| Junyi15 (PSLC #1198) | expert prerequisite + crowd 1-9 ratings (difficulty/similarity; ~6.6 raters/pair) | Chang et al. EDM 2015 | 837 exercises, 1,131 rated pairs, ~26-40M logs | NO (binary correct) | PSLC DataShop, non-commercial |
| Junyi 2020 (Kaggle) | none confirmed | Junyi Academy | 72,758 students, 1,330 exercises, 16.2M logs | NO | kaggle |
| PISA | official IRT params in technical-report annexes | OECD scaling team | all cycles since 2000, ~5k-40k students/country | partial (2015+ computer-based logs raw responses) | raw files free; params buried in per-cycle PDF annexes; ID crosswalk manual, coverage partial |
| NAEP | official 3PL/GPCM params, independent NCES calibration | NCES | 2,000+ released items, params 2000-2022+ | restricted microdata | param tables free (nces.ed.gov scaling_irt_*); raw responses need restricted-use license (weeks-months, institutional) |
| Duolingo SLAM | self-referential only | Duolingo | 2M+ tokens, 6k students | no MC options | Harvard Dataverse |
| ASSISTments 09/12/15/17 | none (skill tags = classification, not quality) | WPI | up to 27k students / 2.5M interactions | NO | free |
| EdNet KT2-KT4 | none new vs KT1 (behavioral extras only) | Riiid | up to 131M interactions | options exist; no expert labels | github.com/riiid/ednet CC BY-NC |
| MalAlgoQA | expert rationale per answer choice | grade 3-11 assessment-style | 807 math + 290 reading items x 4 options | content only | gated Google form; arXiv:2407.00938 |
| MaE | expert misconception list (peer-reviewed grounding) | Otero et al. 2025 | 55 misconceptions x 4 examples | content only | github/HF; arXiv:2412.03765 |
| MalruleLib | 101 malrules from 67 papers, EXECUTABLE | Chen et al. 2026 | 498 templates, 1M+ synthetic instances | synthetic generator, no real students | CC BY 4.0; arXiv:2601.03217 |
| PIRLS/ICCS/ICILS/TIMSS-Adv | official IEA IRT/GPCM params, same methodology as our TIMSS | IEA | TIMSS-scale per cycle | same as TIMSS (MC per-option coding in raw files) | iea.nl repository, same open terms |
| LSAT classic / medical banks | none / none public | — | tiny / secure | — | not defensible as validation corpora |

### A.2 Rankings

Option-level discrimination vs labeled misconceptions: (1) **Eedi
2020 log x 2024 misconception bank** — the only both-halves candidate;
JOIN RISK: releases built 4 years apart, no public confirmation
QuestionId aligns; fallback = text matching; budget days. Domain =
UK KS1-4 math (domain-shift caveat vs our banks). (2) MalAlgoQA —
cleaner taxonomy, no responses (face-validity check only, gated).
(3) MaE — 1/25th the scale, completeness only.

Binary/ordinal vs official parameters: (1) NAEP — best calibration
pedigree, but raw microdata = restricted license (months). (2)
**IEA siblings (PIRLS etc.)** — lowest friction on the whole list:
reuses our TIMSS 2019 GPCM pipeline nearly verbatim; under a day of
engineering. (3) PISA — both halves exist but parameters scattered in
per-cycle PDF annexes with manual ID crosswalks; days of archaeology,
partial coverage documented by prior researchers.

Effort estimates: Eedi join = few days (download trivial; loader ~ a
day; ID reconciliation the real cost). IEA sibling = <1 day. NAEP
license = administrative project, not code.

---

## B. Psychometric literature

### B.1 Distractor/option modeling and its validation standard

Foundations: Bock (1972) NRM (Psychometrika); Thissen (1976) — NRM on
raw responses carries more information than 2PL on right/wrong,
concentrated at low ability; Lukhele, Thissen & Wainer (1994) same in
achievement data; Thissen, Steinberg & Fitzpatrick (1989) "the
distractors are also part of the item" (JEM 26:161-176) = the standard
citation. Suh & Bolt (2010, Psychometrika 75:454-473) nested logit
models (correct-vs-incorrect nest + distractor nest, distractor info
separable); Suh & Bolt (2012) multidimensional version testing whether
distractor choice reflects a DIFFERENT trait; Suh & Bolt (2011, JEM)
distractor-driven DIF.

**Classical validation standard for options = per-option point-biserial**:
keyed option positive (conventionally >= .3-.4), each distractor
NEGATIVE, distractor above ~ -.05 flagged "non-functioning". Review:
Gierl, Bulut, Guo & Zhang (2017), Review of Educational Research
(10.3102/0034654317726529). Our empirical anchor construction IS this
standard.

Option-level KT: Ghosh, Raspat & Lan, Option Tracing (AIED 2021,
arXiv:2104.09043) — validation = held-out option accuracy (+ qualitative
distractor clustering, unscored); An et al. AAAI 2022 (arXiv:2204.14006);
CRKT (Knowledge-Based Systems 2024, arXiv:2408.12996) uses unchosen
options + concept map. NONE report point-biserial-style or expert
misconception validation. **The option-level KT literature's validity
bar is currently BELOW the classical psychometric bar — meeting the
classical bar is a positive contribution, not a caveat.**

### B.2 Parameter-recovery conventions and NRM sample sizes

Conventions: bias + RMSE (or RMdSE) + true-vs-recovered correlation,
sometimes CI coverage (Frontiers 2023 IRTree study; Frontiers 2019
mixture-polytomous study discouraging N<1,500 for mixtures). Modern
tutorial: Schroeders & Gnambs, Assessment 2025
(10.1177/25152459251314798): fixed rules of thumb are not defensible;
run model-specific Monte Carlo precision studies.

NRM-specific: **De Ayala & Sava-Bolesta (1999, APM,
10.1177/01466219922031130): examinee-to-item-parameter ratio 10:1
adequate under approximately NORMAL ability; ability-distribution shape
dominated accuracy (~42.5% of variance vs ~29.5% for the ratio);
counterintuitively, MORE discriminating items are HARDER to recover.**
DeMars (2003, APM 27:275-288): more items does not help a given item;
more categories per item HURTS; error grows with discrimination and
skew. K=4 NRM = 6 free params/item (one category anchored); at 10:1
that is a few-hundred-respondents-per-item floor under favorable
conditions, rising with categories/discrimination/skew. Dichotomous
Rasch guidance (Linacre ~8+8 responses) explicitly does NOT transplant
to NRM. Outer bounds: N=500 adequate for MGRM except very long tests
(Frontiers 2016); mixtures ~2,000+.

Consequence for us: the canonical EdNet-NRM cell (5.1 responses per
parameter; heavy option skew) sits far below every published floor —
citable grounding for the starvation diagnosis; report the achieved
ratio and the option-frequency skew, not a bare N.

### B.3 Amortized/variational IRT: identifiability

VIBO (Wu et al. EDM 2020, arXiv:2002.00276) — scalability, no
consistency proof. ML2P-VAE (Converse et al. AIED 2021 + Machine
Learning 2021; CRAN ML2Pvae) — decoder IS the M2PL response function;
small accuracy loss for interpretability. **Plain autoencoders produce
large item-parameter bias consistent with identification failure;
variational/prior structure removes most of it**: Urban & Bauer
(Psychometrika 2021), Molenaar, Grasman & Curi (MBR 60(4) 2025,
10.1080/00273171.2025.2456598). Theory: Kivva et al.
(arXiv:2206.10044) identifiability up to affine transform for specific
VAE classes — conditional, not blanket. Quantitative consistency:
IW-VAE for 3/4-param MIRT (PMC9421264): RMSE .05-.24 improving N=500
-> 10,000, 100% convergence vs partial MCEM/MHRM failures.

Directly relevant recent stress tests: **Jiang et al.,
"Can We Trust Item Response Theory for AI Evaluation?"
(arXiv:2607.15190)** — difficulty recovery correlation <.50 and
discrimination <.60 at N=30; ranking recovery collapses .85 -> <.60
with ability-skew; states plainly that applied-IRT studies do not
verify estimator reliability under regime mismatch. **Yan, Tang &
Shimada, "Recovering Stranded Discrimination in Knowledge Tracing"
(ECML PKDD 2026, arXiv:2606.14123)** — per-item discriminative
information goes UNUSED under sparse exposure (+4.1pp AUC recovered on
sparse items vs +0.24pp dense); rank-preserving calibrators (Platt)
cannot fix it. PRIOR-ART CHECK REQUIRED before novelty claims: adjacent
(exposure-driven stranding, prediction-side) but must be read and
cited; our mechanism (readout-path separation, parameter-side) appears
distinct.

### B.4 Item-fit tradition vs our audit — the named ancestor

Classical item/person fit: infit/outfit (Wright & Panchapakesan 1969),
Yen's Q1 (1981; known misses), Orlando & Thissen S-X2 (modern default;
polytomous: Kang & Chen 2008), person-fit Lz (Drasgow et al. 1985),
Lz* (Snijders 2001). These are WITHIN-one-joint-calibration residual
diagnostics.

**The audit's true relative is item-parameter INVARIANCE/DRIFT testing**
(split-sample recalibrate-and-compare; DIF family; Robust-z; 3-sigma;
delta-plot; Mantel-Haenszel drift). Citable precedent that the two
families are NOT interchangeable: **Smith & Suh (2003, J Applied
Measurement 4(2):153-63): infit/outfit flagged 7/80 items; split-half
recalibration flagged 60/80 (differences up to 21 SEs). Their words:
relying on fit statistics alone "would cause one to miss one of the
most important threats to the usefulness of the measurement model."**
No paper found applying EITHER family to neural KT readouts — open gap,
not duplication.

Refit-ban consequence: the audit is the neural-KT analog of an
ESTABLISHED psychometric methodology (invariance testing), measuring
the deployed model without modifying it; distinct in kind from
refit-as-repair.

### B.5 Recommendations (psychometric sweep)

1. Cite Smith & Suh (2003); frame the audit as drift/invariance
   testing's neural analog — pre-empts "isn't this just item fit."
2. Report examinee-to-parameter ratios and option-skew per De Ayala &
   Sava-Bolesta / DeMars; never flat responses-per-item numbers.
3. Name the option-level KT validation gap as a contribution: our
   readouts meet the classical per-option point-biserial standard;
   the existing option-tracing line does not attempt it.

---

## C. Architecture landscape

Question: which models give item parameters a dedicated pathway vs a
shared embedding? (Code-verified where possible via pykt-toolkit and
original repos.)

### C.1 Pathway table

| Model | Pathway | Parameter | Source |
|---|---|---|---|
| AKT (KDD 2020) | SHARED | Rasch difficulty mu_q spliced into attention Q/K input (code-verified) | arXiv:2007.12324 |
| SAINT / SAINT+ | none | — | arXiv:2002.07033 / 2010.12042 |
| SAKT (EDM 2019) | none | — | arXiv:1907.06837 |
| simpleKT (ICLR 2023) | SHARED | same Rasch pattern into attention (code-verified) | arXiv:2302.06881 |
| sparseKT (SIGIR 2023) | SHARED | Rasch scalar CALLED "discrimination" | arXiv:2407.17097 |
| stableKT (IJCAI 2024) | SHARED | difficult_param embedding, AKT pattern reused (code-verified) | ijcai.org/proceedings/2024/654 |
| DIMKT (SIGIR 2022) | PARTIALLY separate | own qd_emb/sd_emb tables, but concatenated into the state update AND reused in gates — no gradient boundary, dynamics reaches it twice (code-verified) | 10.1145/3477495.3531939 |
| QIKT (AAAI 2023) | shared, deliberately parameter-free | sigma(alpha+beta+zeta) of shared-module scores | arXiv:2302.06885 |
| IKT | separate but non-neural | hand-engineered difficulty into TAN Bayes | AAAI 21560 |
| LPKT (KDD 2021) | none | no difficulty table despite framing (code-verified absence) | 10.1145/3447548.3467237 |
| folibiKT (CIKM 2023) | inherits base | forgetting bias plug-in only | arXiv:2309.14796 |
| Deep-IRT (EDM 2019) | HYBRID (clearest pre-existing partial split) | beta head reads static key emb; theta head reads value-memory read; but key emb ALSO drives DKVMN addressing; discrimination = fixed 3.0 constant | arXiv:1904.11738 |
| KQN (LAK 2019) | none | — | arXiv:1908.02146 |
| CAKQN (2022) | SHARED | retrofitted Rasch scalars into shared vectors | frontiers 2022.846621 |
| ATKT (MM 2021) | none | — | arXiv:2108.04430 |
| MIKT (WWW 2024) | shared (abstract-level only; paywalled) | IRT layer with difficulty+discrimination on shared states | 10.1145/3589334.3645373 |
| extraKT (ECAI 2024) | SHARED | identical difficult_param pattern (code-verified) | FAIA240651 |
| DisKT (WWW 2025) | SHARED | Rasch difficulty + discrimination-like familiar/unfamiliar state difference; no parameter validation | arXiv:2503.02539 |
| AT-DKT (WWW 2023) | shared | aux heads off shared LSTM state; no IRT | arXiv:2302.07942 |
| DTransformer (WWW 2023) | SHARED | p_diff_embed AKT-style (code-verified) | github.com/yxonic/DTransformer |
| PAKT (2026) | shared, masked downstream | difficulty vector baked into one shared representation; branches diverge only via downstream mask | arXiv:2607.13103 |
| DenoiseKT (IJCAI 2025) | shared | difficulty fused into graph interaction module feeding dynamics | ijcai 2025/1069 |
| "Ability attribute + attention" (2023) | shared | ability/difficulty as attention input features | arXiv:2302.02146 |
| Vie & Kashima (ICCE 2023) | theory | DKT = IMPLICIT dynamic MIRT; parameters smeared across shared weights, unrecoverable per-item | arXiv:2309.12334 |

Caveat (honest): simpleKT/sparseKT/stableKT/extraKT/DTransformer share
the IDENTICAL pykt-toolkit `difficult_param` code block — inheritance
through tooling, not five independent rediscoveries. Independent lines:
AKT, DIMKT, Deep-IRT, DisKT.

### C.2 The recurring gap

Across all 23 architectures, 2019-2026: (1) NO model routes item
parameters behind a boundary the sequence-dynamics gradient cannot
reach; (2) NO true learned multiplicative discrimination exists
("discrimination" = absent, a fixed constant, or a difficulty-shaped
scalar wearing the name); (3) NO parameter-recovery, invariance, or
DIF-style validation anywhere — AUC plus at most a qualitative plot.

### C.3 Positioning verdict (agent's, endorsed)

Per-item parameters are not novel (AKT 2020 onward); a dedicated
pathway is not novel in storage terms (DIMKT) or head terms (Deep-IRT).
What is absent everywhere: separation with a GRADIENT BOUNDARY —
reachable only by a parameter-specific objective. The separated key is
the missing rung of a ladder the literature has been visibly climbing
since 2020: a generalization/completion of a known trajectory, with
the theory (why every earlier rung corrupts) and the validation
standard (recovery + invariance audit + classical point-biserial) the
lineage skipped.

---

## D. Evaluation practices in interpretable-KT / neural-IRT papers

### D.1 The four real-data classical-comparison precedents (the field's bar)

Only four papers found that directly correlate learned parameters
against a classical estimator on real data:

| paper | data | result vs classical | quality |
|---|---|---|---|
| Deep-IRT (Yeung EDM 2019) | — | r = .56 vs item analysis | weak |
| SAD-IRT | N = 20 students | r = .97-.98 | tiny N |
| **ML2P-VAE** (Converse et al., Machine Learning 110:1463-1480, 2021) | ECPE, 2,922 real students, 28 items | discrimination r ~= .998 (full model; .979-.982 for freer variants); difficulty/ability ~.9996 vs MHRM (mirt) | strong; their own words: "no true values... compare directly with MHRM's estimates" — the same move as ours |
| **Tabak, Molenaar & Curi** (Behaviormetrika 52:293-316, 2025) | ENEM 2022 national exam, 5,000 examinees, 112 items, 3 dims | difficulty r = .90; discrimination r = .96/.93/.88 per dim vs mirt MML; sim bias mirt .09 vs AE .11-.13 | the high-water mark: real named exam, realistic scale |

Everything else stops short: AutoIRT (Duolingo English Test, BEA 2024,
arXiv:2409.08823) reports calibration/reliability only, never
parameter-to-parameter (cold-start mean-grade corr 2PL .718 vs AutoIRT
.785 vs BERT-IRT .814; warm .994/.998); WD-FAB (AISTATS 2023) =
qualitative heatmap, no statistic; amortized LLM-eval calibration
(arXiv:2503.13335) = scatter plot only; Urban & Bauer real-data
application = simulation validation only. Journal sweeps: CAEAI —
confirmed EMPTY of neural-IRT parameter-validation work (three
searches); IEEE TLT — one out-of-scope hit (IRT ensembling of essay
scorers); JEDM — leaderboard-placement only.

### D.2 What this means for us (the uncomfortable part, stated plainly)

The two well-executed precedents land at r = .88-.96 (ENEM) and
.98-.9996 (ECPE) against MML. Our real-data MML concordances are
.68-.74 on the best cells (EdNet-2PL), .44 (KDD), .32-.56 (TIMSS). A
reviewer who knows either precedent will ask. The honest and defensible
answer is the REGIME: both precedents amortize STATIC IRT on
complete-matrix designed-exam data — the amortizer sees exactly what
MML sees. Our object is a sequential, causally-constrained,
prediction-trained KT model on sparse logs — which NO paper in this
survey has ever validated against classical estimates at all. On dense
static synthetic beds our SK readout reaches .90-.95, i.e., the
precedent range, exactly where the regimes match. The paper must state
this regime distinction explicitly and early, or the raw numbers will
read as failure; stated properly, "first classical-standard validation
of sequential KT readouts" is the accurate claim, with the static
amortizers as the adjacent (easier-regime) benchmark.

### D.3 Practice recommendations (evaluation sweep)

1. Cite ML2P-VAE and Behaviormetrika 2025 as the field's standard for
   what counts as good MML agreement; position our numbers with the
   regime distinction, never bare.
2. CAEAI's confirmed emptiness on this topic = venue opportunity (first
   of its kind there), also a burden (reviewers may import the static
   bar without the regime caveat — write the caveat into the abstract
   region of the argument).
3. The stop-at-AUC norm across the entire architecture lineage (C.2)
   plus the four-precedent shortlist here = the paper's validation
   section IS the contribution frontier; say so.

---

## E. Expert reads

[Research-scientist and psychometric-researcher reads appended after
section D lands.]
