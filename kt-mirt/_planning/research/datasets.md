# Dataset survey for kt-mirt (G1/G2 avenue map)

Compiled 2026-07-17. Access URLs below were checked via web search only —
direct page fetch (WebFetch/curl) was unavailable in this environment
(network egress blocked), so "page exists" claims rest on search-engine
snippets and secondary citations (survey papers, toolkit docs, dataset
cards), not a live HTTP check. Flagged per-item below where this matters.

---

## 1. Junyi Academy

### 1a. PSLC DataShop `junyi15` release (with expert prerequisite annotations)

- **Access**: `https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198`
  (PSLC DataShop dataset-info page). DataShop requires a free account and
  PI approval/export step for full transaction export; some datasets are
  openly downloadable, others gated — not independently confirmed here.
- **Scale**: 247,606 students, 25,925,992 interactions, 722 distinct
  questions, 41 KCs, collected Nov 2010–Mar 2015. [load_bearing]
- **KC structure / prerequisites**: this is the release with the
  human-annotated prerequisite graph — for each exercise, Junyi Academy's
  own knowledge map gives its prerequisite exercise, topic, area, and a
  2D coordinate position in the map. This is the only Junyi release with
  expert-authored prerequisite edges. [load_bearing]
- **Response format**: binary correct/incorrect per exercise attempt.
- **Item-KC arity**: reported as close to 1 exercise ↔ 1 KC-topic in most
  downstream use (41 KCs over 722 exercises is coarse-grained, topic-level,
  not fine multi-tag).
- **License/terms**: not confirmed directly; DataShop datasets are
  typically usable for research once exported, exact license unclear from
  search alone.
- **Pitfall**: heavy long-tail — most students complete under 50
  questions, few over 100 (comparison made explicitly against EdNet-KT3
  in a knowledge-tracing survey), so per-student trajectories are short.

### 1b. 2020 Kaggle release ("Junyi Academy Online Learning Activity Dataset")

- **Access**: `https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy`
  (Kaggle dataset page; confirmed reachable in search results, standard
  Kaggle dataset-card page — not independently HTTP-verified here).
  License stated as CC BY-NC-SA 4.0.
- **Scale**: reported two ways across sources — the dataset card describes
  16,217,311 problem-attempt rows from 72,630 students over Aug
  2018–Jul 2019 (`Log_Problem.csv`, `Info_Content.csv`,
  `Info_UserData.csv`); a downstream re-filtered version used in KT
  benchmarking papers reports 11,468,379 interactions / 25,649 students /
  1,701 questions. Treat the smaller numbers as a filtered subset, not a
  different release.
- **Does it carry prerequisites?** Yes — `Info_Content.csv` includes a
  `prerequisite` field naming an exercise's parent in the knowledge map,
  plus `h_position`/`v_position` map coordinates, i.e. the same
  prerequisite-graph annotation lineage as the 2015 DataShop release, not
  a stripped-down version. [load_bearing — this directly answers the
  question asked; corroborated across three independent search snippets
  (EduData docs, Kaggle dataset card, a KT survey), but not verified from
  the raw CSV itself]
- **KC structure**: exercise-level topic/knowledge-map only, no separate
  fine-grained KC tagging layer beyond the exercise hierarchy itself.
- **Response format**: binary correct/incorrect, plus metadata (time
  taken, hints used, exercise problem type).
- **Pitfall**: this is a *different* student cohort and time window than
  junyi15 (2018-19 vs. 2010-15) — they are not the same students, do not
  merge them as one continuous log.

---

## 2. ASSISTments

### 2a. 2009 skill-builder

- **Access**: `https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010`
  (official ASSISTments data site, Google Sites page — reachable per
  search results, not independently HTTP-verified). Also mirrored on
  IEEE DataPort (`ieee-dataport.org/documents/assistments-dataset-2009-2010`)
  and via community re-hosts (Kaggle, figshare `skill_builder_data.csv`).
- **The duplicate-row defect**: the original `skill_builder_data.csv`
  duplicates any student-problem row that has multiple associated skills,
  once per skill — these duplicated rows are about 25% of the file, and
  several papers showed that treating them as independent observations
  artificially inflates next-step prediction AUC. [load_bearing]
- **Corrected release**: a second file was published with one row per
  student-problem; when a problem carries multiple skills they are
  concatenated into a single compound label (`skill1_skill2`) instead of
  being duplicated. Both the original and corrected files are still
  hosted from the same page; always state explicitly which one was used.
  [load_bearing]
- **Scale**: reported numbers vary by which file/preprocessing a paper
  used — commonly cited figures range from ~4,150-4,417 students and
  ~279k-328k interactions over ~110-124 skills. This spread itself is a
  symptom of the duplicate-row/filtering ambiguity above; do not treat
  any single number as canonical without naming the exact file.
- **Response format**: binary correct/incorrect (first-attempt correctness
  is the field typically modeled).
- **Item-KC arity**: many-to-many — a problem can carry more than one
  skill tag, which is exactly what produces the duplicate-row artifact.
- **Overall correct rate**: ~65.8% reported in at least one
  cross-dataset comparison table — moderate saturation, less severe than
  EdNet/ASSISTments-2015 reported in the same table (~73%), but this
  number's provenance (which file, which filtering) was not verified
  further.
- **License**: research use has historically been unrestricted/open on
  the ASSISTments site; no formal license text located in search results.

### 2b. 2012-2013 school-year (with affect predictions)

- **Access**: `https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect`
  (per search snippet; an alternate path
  `.../2012-13-school-data-with-affect` vs `.../home/...` was returned
  inconsistently across queries — the official ASSISTments Google Sites
  navigation has moved data around before, so treat the exact path as
  soft and confirm by browsing from the ASSISTments data site root before
  citing in the paper).
- **Scale**: inconsistent across sources — 2,709,436 exercises / 27,485
  students / 265 skills / 53,065 questions in one KT-benchmark citation;
  2,541,201 interactions / 27,066 students / 45,716 questions in another;
  6,123,270 responses / 198 skills in a third (likely the un-filtered raw
  release vs. filtered KT-benchmark subsets). Confirm the exact file
  before use.
- **Response format**: submission-level (one row per problem submission,
  not just first attempt), with additional affect-detector predictions
  (boredom, confusion, frustration, engaged concentration) per Pardos et
  al. 2013 — cite that paper if the affect columns are used.
- **Item-KC arity**: many-to-many, same style as 2009.
- **License**: research use, no formal license text located.

### 2c. 2017 (Longitudinal / STEM data-mining competition)

- **Access**: `https://sites.google.com/view/assistmentsdatamining/dataset`
  (separate competition site from the main ASSISTments data site — per
  search snippet, page exists; not independently HTTP-verified). Free
  registration required; a Terms of Use agreement (primarily a
  no-deanonymization clause) is a condition of access.
- **Scale**: 942,816 action-level rows from 1,709 students, distributed
  as 9 `student_log` files plus a `training_label.csv` carrying the
  competition's target label (`isSTEM`, i.e., whether the student later
  pursued a STEM career) — this is a *longitudinal outcome-prediction*
  dataset, not a standard next-response KT log, and the label is a
  student-level career outcome rather than a per-item response.
  [load_bearing — this changes what the dataset is useful for]
- **Response format**: click-stream / action-level, richer than
  correct/incorrect alone.
- **Fit for kt-mirt**: likely a poor match for G1/G2 as scoped (no dense
  per-KC response stream comparable to the 2009/2012 skill-builder logs);
  flagging for completeness only.

---

## 3. XES3G5M (NeurIPS 2023 Datasets & Benchmarks track)

- **Paper**: NeurIPS 2023 D&B track,
  `https://proceedings.neurips.cc/paper_files/paper/2023/hash/67fc628f17c2ad53621fb961c6bafcaf-Abstract-Datasets_and_Benchmarks.html`
  (OpenReview mirror: `https://openreview.net/forum?id=Mn9oHNdYCE`).
- **Access**: `https://github.com/ai4ed/XES3G5M` — confirmed to exist via
  search (repo title and README snippet both returned); not independently
  HTTP-verified here. [load_bearing]
- **Scale**: 18,066 students, 7,652 questions (6,142 fill-in-the-blank +
  1,510 multiple-choice), 5,549,635 interactions, 865 KCs. [load_bearing]
- **KC structure**: KCs are organized in an expert-annotated hierarchical
  tree; a "KC route" is the root-to-leaf path in that tree. Questions are
  associated with **leaf** KCs only (not internal tree nodes), and a
  single question can map to **multiple** leaf KCs — the released
  interaction format expands question-level sequences into KC-level
  sequences when a question carries more than one KC.
  [load_bearing — confirms many-to-many item-KC arity]
- **Response format**: binary correct (1) / incorrect (0), with
  timestamp; also ships auxiliary text content, question type, and
  worked-solution analysis (this is the "auxiliary information" the
  dataset is named for).
- **License**: one search summary stated MIT for the released data/code;
  this should be re-verified against the repo's actual LICENSE file
  before relying on it for redistribution decisions — not independently
  confirmed here.
- **Pitfall**: because interactions are exploded to KC-level, naive
  train/test splits at the KC-expanded row level leak information across
  KCs of the same question; the benchmark paper's own splitting protocol
  should be followed rather than re-deriving one.

---

## 4. Eedi (NeurIPS 2020 Education Challenge)

- **Task framing / paper**: "Instructions and Guide for Diagnostic
  Questions: The NeurIPS 2020 Education Challenge," arXiv:2007.12061;
  results paper arXiv:2104.04034 ("Results and Insights from Diagnostic
  Questions..."), also `proceedings.mlr.press/v133/wang21a/wang21a.pdf`.
- **Access**: direct zip at
  `https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip`
  (Azure blob storage — reported as still publicly reachable in a recent
  (2024-era) secondary source, not independently HTTP-verified here). A
  related but not identical dataset (with question text/images) is also
  distributed via the Kaggle competition "Eedi – Mining Misconceptions in
  Mathematics." Eedi's own research pages
  (`eedi.com/research`, `eedischool.com/projects/neurips-education-challenge`)
  point to the same materials. Won EDM Society's "Best Publicly Available
  Educational Data Set" prize in 2021, and multiple 2024-era papers still
  build on it, which is reasonable indirect evidence of continued
  availability. [load_bearing, moderate confidence — no direct HTTP check]
- **Scale**: 118,971 students, ~20+ million answer records (one source
  says "over 17 million," another "over 20 million" — treat as
  order-of-magnitude ~2×10^7, exact figure not reconciled), collected
  over two school years (Sept 2018–May 2020).
- **KC / topic structure**: questions are tagged against a **4-level
  topic ontology tree**; each question carries a list of subjects/topics
  at varying granularity (not a flat single-KC tag). Distractor-level
  **misconception labels** are a separate annotation layer from the topic
  tags — each wrong-answer option can carry a specific misconception ID,
  which is a genuinely distinct signal from "which KC was tested."
  [load_bearing]
- **Response format**: multiple-choice (4-option), so responses are
  categorical/selected-option, not raw binary — correctness is derivable
  from the selected option, but the selected-*wrong*-option identity is
  itself informative (misconception diagnosis was one of the three
  competition tasks).
- **Overall correct rate / saturation**: not located; the competition's
  winning **model** accuracy was 74.74% on the answer-prediction task,
  which is a model metric, not the raw student correct-rate — do not
  conflate the two. [uncertain — flagged, not to be used as a saturation
  number]
- **License**: not confirmed in detail; used broadly for research without
  reported access friction, but the exact license text was not located.

---

## 5. EdNet KT1 (Riiid)

- **Paper**: "EdNet: A Large-Scale Hierarchical Dataset in Education,"
  arXiv:1912.03072 (also PMC7334672, Springer AIED 2020).
- **Access**: `https://github.com/riiid/ednet` (confirmed to exist via
  search, including README snippet; not independently HTTP-verified
  here). Also mirrored on Kaggle,
  `https://www.kaggle.com/datasets/gmhost/ednetkt1`. Data is split by
  student, one CSV per user ID, distributed as separate compressed
  per-level archives (KT1/KT2/KT3/KT4).
- **Scale**: KT1 (the base question-solving-log level) is reported as
  131,441,538 interactions from 784,309 students, over 13,169 problems
  and 1,021 lectures. [load_bearing]
- **Tag structure**: items are tagged with **293 distinct skill/tag
  types**; EdNet's defining structural feature is that questions are
  grouped into **bundles** (a shared reading passage / picture / audio
  clip), and a student answers all questions in a bundle together — tags
  are per-question, but exposure and timing are bundle-level, which is a
  known confound for any "practice of KC A" causal read since bundle-mates
  are answered in the same sitting. [load_bearing]
- **Item-KC (tag) arity**: multiple tags per question is standard (exact
  average tags/question not located in search — flagged as a gap; check
  the released `questions.csv` directly rather than relying on a
  secondary summary).
- **Response format**: for KT1, effectively binary correct/incorrect per
  question response (KT2+ layers add richer action logs — lecture views,
  UI events — not needed for a KT-only read).
- **Overall correct rate**: not directly located; downstream model
  accuracy figures cluster around 65-73% (e.g. DKT ≈70.4%, BKT ≈70.1%,
  Code-DKT ≈65.1%), but these are model accuracies, not raw student
  correct-rate, and accuracy on a binary task is not the same statistic
  as %-correct when classes are imbalanced — do not reuse these as a
  saturation number without checking class balance directly.
  [uncertain]
- **License**: Creative Commons Attribution-NonCommercial 4.0
  International (CC BY-NC 4.0), stated explicitly for research use;
  contact `research@riiid.co` for questions per the repo.

---

## 6. KDD Cup 2010 (Algebra / Bridge-to-Algebra, Carnegie Learning Cognitive Tutor)

- **Access**: `https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp`
  (dataset downloads page) and `https://pslcdatashop.web.cmu.edu/KDDCup/rules_data_format.jsp`
  (data-format spec); competition mirror at
  `https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data`.
  Both reported to exist via search (title + content snippets returned
  directly from the pages); not independently HTTP-verified here. PSLC
  DataShop hosting implies a free account for full access in general, but
  the KDD Cup pages themselves appear to be static/public.
- **Five released files total**: three "development" sets — Algebra I
  2005-2006, Algebra I 2006-2007, Bridge to Algebra 2006-2007 — plus two
  held-out "challenge" sets (Algebra 2008-2009 and Bridge to Algebra
  2008-2009) used for the actual competition leaderboard. [load_bearing]
- **Scale** (development sets, as reported in one benchmark
  reconstruction — treat as one paper's numbers, not the official
  DataShop totals, since different papers filter differently):
  Algebra I 2005-2006: 884,098 interactions, 4,712 student-sequences,
  173,113 "questions" (Cognitive-Tutor "steps"), 112 KCs, average 1.35
  KCs/step. Bridge to Algebra 2006-2007: 1,824,310 interactions, 9,680
  sequences, 129,263 steps, 493 KCs, average ~1.01 KCs/step. The overall
  competition dataset was described as the largest educational-technology
  dataset released at the time (>9 GB raw). [load_bearing]
- **KC models / opportunity count**: each "step" (one unit of problem-
  solving interaction within Carnegie Learning's Cognitive Tutor) can be
  tagged by **more than one** knowledge component under a given "KC
  model" (DataShop supports multiple competing KC-model taggings for the
  same step log); a student's **opportunity count** for a KC increments
  by 1 every time that KC is encountered again, and steps with multiple
  KCs list multiple opportunity numbers separated by `~~` in the raw
  file. [load_bearing]
- **Response format**: not simple binary — Cognitive Tutor logs a
  first-attempt correctness plus counts of hints requested and incorrect
  attempts per step, i.e. partial-credit-adjacent (first-attempt
  correct/incorrect is the field usually modeled as the KT target).
- **Overall correct rate**: not located in search results; flagged as a
  gap — would need direct DataShop inspection.
- **License/terms**: full terms of use not located beyond general
  competition rules; DataShop-hosted PSLC data typically carries a
  research-use data-use agreement, exact text not confirmed here.
- **Known pitfall**: this is Cognitive-Tutor step-level data from an
  intelligent tutoring system, not free-response item log data — "steps"
  are sub-problem actions, not discrete test items, so comparing scale
  numbers 1:1 against item-level datasets (ASSISTments, Eedi, XES3G5M)
  overstates KDD Cup's item count; the natural unit here is closer to a
  "skill demonstration opportunity" than a "question."

---

## Cross-dataset notes relevant to the avenue map

- **Saturation risk (G2 concern).** Reported correct rates cluster
  moderate-to-high across everything checked: ASSISTments 2009 ≈65.8%,
  ASSISTments 2015 ≈73.2% (both from the same comparison table, source
  not independently re-verified), Eedi's winning model accuracy 74.74%
  (a model metric, not student correct-rate — do not reuse directly),
  EdNet downstream model accuracies mostly 65-73%. None of these confirm
  or rule out the near-80%-correct saturation problem flagged in program
  context for EdNet/ASSISTments/KDD; the raw correct-rate statistic
  (not a model's prediction accuracy) was not cleanly located for any
  dataset in this pass and should be computed directly from the raw
  files before betting a per-KC decomposition study on any one dataset.
  [uncertain, load_bearing for G2 planning]
- **Item-KC arity is many-to-many almost everywhere except Junyi's
  topic-level tagging.** ASSISTments (both 2009 and 2012), XES3G5M, and
  KDD Cup all natively support multiple KC/skill tags per item/step, with
  documented artifacts from mishandling this (ASSISTments 2009 duplicate-
  row inflation being the clearest, best-documented case). Any G1
  cross-KC transfer analysis needs to fix an explicit item→KC expansion
  policy per dataset and state it, not assume 1-to-1.
- **True prerequisite/expert-annotated structure exists only for Junyi**
  (both the 2015 DataShop and 2020 Kaggle releases carry the same
  prerequisite-graph lineage) **and, in a different form, for XES3G5M**
  (a hierarchical KC tree, which encodes taxonomic containment rather
  than prerequisite/temporal ordering) **and Eedi** (a 4-level topic
  ontology, also taxonomic not prerequisite). None of ASSISTments, EdNet,
  or KDD Cup ship expert-authored prerequisite edges — any G1 claim about
  signed cross-KC influence on those three would be inferring structure
  purely from the interaction log, with no external ground truth to
  validate the found edges against. Junyi is the one dataset where a
  discovered cross-KC effect could be checked against a human-authored
  prerequisite graph as an external consistency test.
