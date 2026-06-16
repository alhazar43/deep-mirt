# RQ reframe proposal

Maps the existing RQ1-8 + RQ5c battery and the content-channel cold-start
onto the PI's settled question framework, exposes the cells the new frame
opens, and proposes a reframed, prioritized RQ list. This feeds
`docs/RESULTS_v2.md`. It is a planning document, not paper prose. The result
ledger stays `substrate/RESULTS.md`; the live state stays memory
`thesis-vision`.

## The new frame, in one place

Two axes organize everything.

**AGNOSTICISM = predict AND differentiate.** Predict is the headline
(estimate a response without being told the format, respondent, or item).
Differentiate is the supporting backtrack probe (recover format, respondent,
or item identity from the learned representation). The two together form a
2x2 over what is held fixed.

| | PREDICT (headline) | DIFFERENTIATE (supporting) |
|---|---|---|
| **FIX ITEM** vary respondents | place a respondent's response on a held item without knowing who | tell students/LLMs apart on a fixed item set. The clean respondent-agnosticism test, because fixing items removes the construct-mismatch confound that capped RQ3. **OPEN.** |
| **FIX STUDENT/LLM** vary items | place a response on a held-out item for a fixed respondent | tell items apart for one fixed respondent. This is RQ5c, the counterfactual fixed-respondent design. ORDER-separable ~0.88, LEVEL offset by construct. |

**IRT-as-FLAVOR.** One response-prediction loss flows through a swappable,
format-matched head (GPCM ordinal, NRM nominal and MC, 2PL binary, BT
pairwise, plus the continuous RT/Beta/Poisson heads). The engine is a
lightweight encoder plus a swappable decoder. The real program is the
universal-feature search, not a particular IRT form.

**UNIVERSAL FEATURES.** The item side is content, an LLM text embedding. The
content channel works (cold-start from text 0.96 synthetic, 0.47 real SLAM).
The student side is "true alpha", a universal student feature beyond scalar
theta that holds across formats, items, and student types. The candidate is
response speed or drift rate. Expected universal in ORDER, bounded in LEVEL by
construct match, the human-to-LLM wall that the RQ5c level offset already
exposed.

---

## 1. Mapping table, every existing result onto the frame

P = predict, D = differentiate. Cell = which 2x2 cell, or "engine/feature"
for results that are not a 2x2 probe but support the IRT-as-flavor or
universal-feature legs. Keep/demote/drop is the verdict under the new frame.

| Result | P or D | Cell | Verdict | One-line why |
|---|---|---|---|---|
| **RQ1 format-agnostic, real binary+NRM+RT** (EdNet) | P | IRT-as-flavor (item side, fix respondent, vary format) | **KEEP, promote to headline** | The cleanest real "one loss, swappable format-matched head" evidence: NRM->2PL 0.597, binary->NRM 0.863, RT joins the scale (RT-only 0.572 above its own 0.426 ceiling). This IS the IRT-as-flavor leg on real data. |
| **RQ2 item anchoring** (SLAM anchored extension) | P | FIX STUDENT, vary items, predict (place a new item) | **KEEP, demote to a lemma** | Anchored extension places new-item difficulty 0.89x and discrimination 0.83x of a full-recal ceiling, ~100x cheaper. Real and solid, but it is calibration plumbing under the new frame (the classical fixed-param-vs-concurrent comparison), supporting not headline. |
| **RQ3 respondent transfer 0.34** (SLAM human vs small LLM) | P | FIX ITEM, vary respondents, predict (cross-population difficulty) | **DEMOTE to a corroborated boundary** | 0.34 is robust to pool diversity but capped by SLAM's exact-match grading construct AND by the general fact that cross-population difficulty transfer caps low for everyone (He-Yueya ~0.4-0.6, anesthesiology ~0.28). Keep as the LEVEL-boundary evidence, not a positive result. The DIFFERENTIATE version of this cell is the OPEN test that replaces it. |
| **RQ4 cross-test alignment** (EdNet TOEIC parts) | D | FIX STUDENT-ish, differentiate items/instruments | **KEEP** | Construct-distance gradient (within 0.76 > cross 0.62) plus anchored cross-part consistency 0.72. Item-side differentiation: the scale tells instruments apart sensibly. Survives the classical-control circularity kill. |
| **RQ5/RQ5a/RQ5b separability** (EdNet) | D | both cells (person and item separability) | **KEEP as method, demote multidim** | Scalar-ability leakage control AUC 0.51 = chance (encoding does not carry instrument); anchored readout lifts separability 0.40->0.72. The anchoring-as-identifiability methods lesson is the durable content; multidim theta is now a refinement, not a need. |
| **RQ5c order/level** (fixed LLM, two SLAM banks) | D | FIX STUDENT/LLM, differentiate items | **KEEP, promote to the differentiate headline** | The most original design: true ability constant by construction. ORDER-separable (cross-bank 0.88 vs floor 0.66), LEVEL offset +1.29 logits = construct mismatch. This IS the fix-respondent differentiate cell, and the order-vs-level distinction is the universal-feature claim (universal in order, bounded in level). |
| **RQ7 coercion** (EdNet K2/K3/K4 binning) | P | IRT-as-flavor (format robustness) | **KEEP, fold into RQ1** | Rank-preserving but lossy across binnings (cross-coercion 0.71 raw / 0.79 disatt; precision K2 0.81 < K3 0.92 < K4 0.95). Inoculates the IRT-as-flavor claim against "your formats are relabelings." A robustness rider on RQ1, not standalone. |
| **RQ8 invariance/DIF** (EdNet subpops) | D | FIX ITEM, differentiate respondents (the placebo direction) | **KEEP, but it is the WEAK form of the open cell** | Item params subpopulation-invariant beyond a placebo floor (gaps 0.04-0.10). This is "items do NOT differentiate the respondent groups" (no DIF). The OPEN test wants the opposite: can we differentiate respondents on fixed items? RQ8 is the null-DIF complement, supporting evidence for the open test's validity, not the open test itself. |
| **Content-channel cold-start** (synthetic + SLAM) | feature | universal item feature | **KEEP, the item-side lever, primary** | Item cold-start from text alone: synthetic gap +0.996, real SLAM +0.47 (0.71x of warm placement). The committed lever and the only universal-feature leg currently proven. |
| Engine decision (ma-irt sep_theta -> dual-channel substrate) | engine | n/a | **KEEP as settled infra** | Dual-channel substrate = best-of-both (dynamic net-drift 0.728, discrimination tunable by width to ~0.90). The swappable encoder for IRT-as-flavor. Settled; not an RQ. |

### What the table says

- The **PREDICT side is well-stocked**: RQ1 (format), RQ2 (item), RQ3
  (respondent, capped). RQ1 promoted to headline, RQ2 demoted to a lemma,
  RQ3 demoted to a boundary.
- The **DIFFERENTIATE side has only the FIX-STUDENT cell filled** (RQ5c,
  RQ4, RQ5). The FIX-ITEM differentiate cell (tell respondents apart on fixed
  items) is EMPTY of a direct positive probe. RQ8 sits adjacent to it as the
  null-DIF complement.
- The **universal-feature search has one leg** (content, item side). The
  student-side "true alpha" leg (speed/drift) is unbuilt; the only evidence
  bearing on it is RQ5c's order-vs-level split and RQ1-3format's RT-joins-scale.

---

## 2. Open cells the new frame exposes

### Open cell A, the cleaner respondent test, FIX-ITEM differentiate

**The cell.** Fix a common item set. Vary respondents (humans and LLMs, or
ability strata). Ask whether the learned representation can DIFFERENTIATE
respondents, and whether respondent ordering recovered from a fixed item set
is stable across disjoint item subsets.

**Why it is the cleaner respondent-agnosticism test.** RQ3 capped at 0.34
because it varied items AND populations at once, so a construct mismatch
between the human grading and the LLM generation confounded the transfer.
Fixing the items removes that confound: every respondent meets the same
items, so a respondent difference is a pure ability difference, not an
item-by-population interaction. This is the respondent analog of RQ5c (which
fixed the respondent to clean the item side); RQ5c's success on order
suggests the symmetric fix-item design should also clean the respondent side.

**Concrete test.** Two halves. (i) PREDICT: held-out respondent response on a
fixed item set, encoder never told who. Metric = accuracy/AUC against an
identity-informed ceiling. (ii) DIFFERENTIATE: recover respondent ability
ordering from disjoint item subsets of the fixed set, measure cross-subset
Spearman against a within-subset split-half floor (the RQ5c metric, transposed
from items to respondents). Order-separable if cross >= floor.

**Data.** A fixed common item set answered by a heterogeneous respondent pool.
EdNet TOEIC has dense common-item overlap across students (clean MCQ, no
exact-match artifact). The LLM arm reuses the RQ3/RQ5c local-model pool on the
same fixed EdNet item set IF the TOEIC passages can be shown to the models
(content gate, see critical pass). SLAM reverse_tap is the fallback if EdNet
content is blocked.

### Open cell B, the student-side universal feature, "true alpha"

**The cell.** Beyond scalar theta, is there a student feature that is
universal across formats, items, and respondent types. Candidate = response
speed or drift rate (the rate of getting better, the thesis's own estimand).

**Why now.** The content channel proved the item-side universal feature. The
symmetric student-side claim is unbuilt. RQ1-3format already showed RT joins
the shared scale as an independent observable (RT-only places at 0.572, above
a dedicated RT model's 0.426 ceiling). RQ5c already showed the order-vs-level
shape that a universal student feature should have: universal in ORDER,
bounded in LEVEL by construct match.

**Concrete test.** Fit speed/drift as a per-respondent latent on one
format/item set, test whether it rank-predicts the same respondent's
speed/drift on a DISJOINT format/item set (cross-format student-feature
transfer), against the within-set reliability floor. Pass = order transfers
(Spearman >= floor) even where level shifts (the predicted human-to-LLM wall).

**Status.** Buildable on EdNet (has response time) for the human side. The
LLM side is harder (generation latency is not comparable to human RT), so the
student-side universal feature is most likely a HUMAN-only order-transfer
result plus an honest statement of why it does not cross the human-to-LLM
boundary in level.

---

## 3. Reframed, prioritized RQ list

Each entry: the question, the concrete test, the data and engine, and whether
it reuses an existing result or needs a new run.

### Tier 1, headline (predict)

**N1. Format-agnosticism via swappable format-matched heads (IRT-as-flavor).**
- Test. One shared item scale read by 2PL, NRM, GPCM, BT, and RT heads;
  held-out-format placement beats an independent-fits floor.
- Data/engine. EdNet (binary+NRM+RT), dual-channel substrate. SLAM/synthetic
  for GPCM and BT.
- Reuse. **REUSES RQ1-real (binary+NRM+RT), RQ7 (coercion robustness rider),
  jointfmt synthetic (GPCM+BT mechanism).** No new run; consolidation only.

**N2. Item cold-start from content (universal item feature).**
- Test. Place a new item from text alone, no responses, beating an ID-only
  floor and approaching warm response-based placement.
- Data/engine. SLAM (capped) done; Eedi-text (uncapped MCQ + content + human
  difficulty) is the clean follow-up.
- Reuse. **REUSES content-channel cold-start on SLAM.** NEW run for Eedi-text
  (needs the Eedi public release on disk; adapter exists at
  `rl/src/ordrec/data/eedi.py`). Cross-domain (SLAM language -> Eedi math)
  cold-start is the vision test and needs the new run.

### Tier 2, the open differentiate cells

**N3. Respondent differentiation on fixed items (the cleaner respondent
test), open cell A.**
- Test. Fix a common item set; predict respondent responses without identity;
  differentiate respondent ability ordering across disjoint item subsets vs a
  split-half floor.
- Data/engine. EdNet TOEIC fixed common-item set, dual-channel substrate;
  LLM arm reuses the RQ3/RQ5c local pool IF content is showable.
- Reuse. **NEW run.** This is the highest-value new experiment (see critical
  pass). RQ8's null-DIF result supports its validity.

**N4. Item differentiation for a fixed respondent (order vs level), the
fix-student cell.**
- Test. Fixed respondent across two banks; cross-bank ability-ordering
  consistency vs a within-bank floor; report order separately from level.
- Data/engine. Fixed LLM, two SLAM banks, anchored Rasch theta.
- Reuse. **REUSES RQ5c directly.** Optional new run only to raise N from 7
  respondents (proof-of-concept) to a higher-N estimate, which needs a
  supervised Ollama session.

### Tier 3, supporting and methods

**N5. Anchoring as the identifiability backbone (lemma).**
- Test. Anchored extension recovers item params at a high fraction of a
  weakly-identified full-recal ceiling; anchored readout > free-encoder
  readout for cross-instrument consistency.
- Reuse. **REUSES RQ2 (SLAM) + RQ5/RQ4 (EdNet, with the classical
  circularity control).** No new run. Positioned as supporting, cited to
  classical fixed-param calibration, not claimed as the contribution.

**N6. Cross-instrument item differentiation with a sensible construct
gradient.**
- Reuse. **REUSES RQ4 + RQ5.** Feeds N3's validity (the scale tells
  instruments apart sensibly, so telling respondents apart is meaningful).

**N7. Student-side universal feature, speed/drift order-transfer, open cell
B.**
- Test. Per-respondent speed/drift latent transfers in ORDER across a disjoint
  format/item set vs a reliability floor; level may shift.
- Data/engine. EdNet response time, human-only.
- Reuse. **NEW run**, but small (the RT head exists, decoders_ext.py;
  RQ1-3format already wired RT). Lower priority than N3.

### Cut or banked

- **RQ6 dynamic-real.** Stays cut. No independent external criterion in
  education; the chess substitute is circular. Demote to a labeled
  consistency appendix only if a criterion appears.
- **RQ1 score-vs-pairwise on real connected data.** Banked. The educational
  CJ+grades venue is empty (verified); IMDB-WIKI-SbS is off-domain. N1 carries
  format-agnosticism on the abundant binary/NRM/RT formats instead.

---

## 4. Critical pass

### Old RQs that do NOT survive the reframe

- **RQ3 as a positive result.** It was the respondent-agnosticism keystone;
  under the new frame it is a capped boundary (0.34, construct-limited, and
  cross-population difficulty transfer caps low for everyone). It survives only
  as the LEVEL-boundary evidence that motivates the cleaner fix-item test (N3).
  Do not headline 0.34.
- **RQ6.** Already cut, stays cut. No reframe rescues a missing criterion.
- **RQ1 score-vs-pairwise specifically.** The hard-to-source pair is replaced
  by the abundant-format version (N1). The structural-requirement finding
  (formats must agree, the pairwise graph must be connected, content must have
  a ceiling) survives as a methods note, not a result.
- **Multidim theta.** Demoted from a fork to a refinement. The scalar scale is
  more adequate than the RQ5 encoder-readout artifact suggested. Not an RQ.

### New questions that are cloud-castle vs buildable

- **N3 respondent differentiation, buildable** on EdNet for the human-strata
  arm (fixed common items, ability strata, dual-channel substrate, no new
  data). The LLM arm is gated on whether TOEIC passage content can be shown to
  the local models. Failure mode: if EdNet content is restricted (the TOEIC
  passages are not released for LLM consumption, the same block that stopped a
  cleaner RQ3 on EdNet), the LLM arm falls back to SLAM reverse_tap, which
  reintroduces a milder construct cap. Assumption: a fixed common-item set with
  enough respondent overlap exists in EdNet (P0.4-style feasibility check
  needed first, cheap). Cost: one feasibility check + one training run, days
  not weeks.
- **N2 cross-domain cold-start, buildable but gated on a download.** Needs the
  Eedi public release on disk. Adapter exists. Failure mode: Eedi-text cold-
  start could also cap if MiniLM embeddings do not separate math-item
  difficulty (content ceiling), the same risk that capped SLAM. Assumption:
  LLM text embeddings carry math-item difficulty signal (plausible, He-Yueya
  found human-LLM difficulty transfer ~0.5 on Eedi MCQ). Cost: download +
  one run.
- **N7 student-side universal feature, partly cloud-castle.** The human-only
  speed/drift order-transfer is buildable on EdNet. The cross-respondent-type
  (human-to-LLM) version is cloud-castle, because LLM generation latency is
  not commensurable with human response time, so the level boundary is
  structural and the experiment can only state WHY it does not cross, not
  cross it. Honest scope: a human-only positive plus a principled boundary
  statement.
- **N4 higher-N RQ5c, buildable but infra-risky.** Raising N=7 needs a
  supervised single-flight Ollama session (the runaway-process lesson). Low
  marginal value; the proof-of-concept order-vs-level distinction is already
  the contribution.

### The single highest-value next experiment

**N3, respondent differentiation on a fixed item set (open cell A), human-
strata arm first on EdNet.**

Why. (i) It fills the one empty 2x2 cell, the FIX-ITEM differentiate cell, and
the PI named it the cleaner respondent-agnosticism test. (ii) It is the
symmetric partner to RQ5c, which already succeeded on order for the fix-
respondent side, so a positive here completes the differentiate diagonal and a
negative is itself informative (it would say the asymmetry is real). (iii) The
human-strata arm needs no new data and no Ollama (EdNet on disk, dual-channel
substrate built), so it de-risks the design before spending the gated LLM arm.
(iv) It directly tests "can we tell students apart on common items", which is
the operational core of respondent-agnosticism that RQ3 only approached
through the confounded cross-population route.

First step. A cheap feasibility check (EdNet common-item overlap across an
ability-stratified student sample, the RQ8/P0.4 pattern), then the two-half
test (predict without identity; differentiate ordering across disjoint item
subsets vs a split-half floor). Gate: cross-subset ordering >= within-subset
floor = respondents are order-separable on fixed items = the clean respondent-
agnosticism positive RQ3 could not deliver.
