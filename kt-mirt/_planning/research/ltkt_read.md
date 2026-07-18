# LTKT read: Xu, Tang, Lv, Yu, Yu, Chen. "LTKT: Knowledge Tracing Based on
Positive and Negative Learning Transfers." Tsinghua Science and Technology,
2026, 31(3): 1894-1917. DOI 10.26599/TST.2024.9010201. Open access on
SciOpen. Received 2024-07-10, accepted 2024-10-20, published in the June
2026 issue (volume slipped past the nominal "2024" cover year).

PDF retrieved directly: `https://www.sciopen.com/local/article_pdf/10.26599/TST.2024.9010201.pdf`
(2.53 MB, 24 pages). Local copies: `data/ltkt/ltkt.pdf`, extracted text
`data/ltkt/ltkt_text.txt` (both outside kt-mirt/, per download convention).

## Mechanism: is transfer signed?

Yes, explicitly and by design -- this is the paper's whole contribution.
LTKT builds a "Learning Transfer Graph" (LTG) with four edge types:
PTPR (positive transfer, prerequisite), PTSR (positive transfer,
similarity), NTPR (negative transfer, prerequisite), NTSR (negative
transfer, similarity). Positive edges are green, negative are red. Prior
work they cite (GKT, SKT, KTLT, ContextKT, KSGKT, CAKT, SPARSEKT) models
only positive transfer; LTKT's stated first-in-literature move is adding
the negative-transfer arm.

Architecturally, "signed" means two separate additive channels (PTE:
positive-transfer-effect module, NTE: negative-transfer-effect module),
each producing a nonnegative RELU-activated effect vector, which are then
combined with a learned per-student scalar mixing weight beta in (0,1):
effect = beta * (aggregated positive effects) + (1-beta) * (aggregated
negative effects) [two-phase integration, Eqs. 10-14]. So the "sign" lives
in which of the two RELU channels the edge routes through and in the
additive/subtractive role that channel plays when updating a neighbor
concept's knowledge state -- not in a single continuous coefficient that
can flip sign from data. There is no mechanism by which a positive-labeled
edge produces a negative update, or vice versa: the graph construction
step (Section 5.1.2) hard-assigns each edge's polarity before training,
and the network only learns edge *magnitudes* within that fixed polarity.
This is important: LTKT's "signed" transfer is a signed *graph* (a
prior/label baked in by the statistical construction rule below), fed
through unsigned (RELU >= 0) learned magnitude functions. The network
never discovers sign; it only discovers how strongly to apply a sign that
a preprocessing heuristic already assigned.

## Graph construction: statistical, not learned end-to-end, not causal

None of the two datasets provides ground-truth transfer relations. LTGC
(the graph-construction component) mines the LTG from co-occurrence
statistics of the interaction log itself, following the same recipe as
prior work SKT (Ref. [16] in the paper) with an added negative-transfer
extension:
- PTPR (i precedes j, positively, prerequisite): frequency of the
  adjacent event pattern "student answers item on concept i correctly,
  then immediately answers item on concept j correctly" (i-check ->
  j-check), normalized per-row into a probability matrix PT, edge kept
  if PT[i][j] >= threshold = mean of all PT entries.
- PTSR (positive, similarity, undirected): symmetric co-occurrence
  frequency of correct-correct pairs in either order, gap-penalized,
  min-max normalized to [0,1] (matrix PS), edge kept if PS[i][j] >=
  mean(PS).
- NTPR / NTSR: same recipe but built from "wrong-then-correct" and
  "correct-then-wrong" adjacent event counts (matrices L, NT, thresholds
  = row means), reusing the same statistical-adjacency logic, i.e.
  negative transfer is inferred purely from local performance-sequence
  statistics, not from any expert-labeled misconception taxonomy or
  external curriculum ontology.

All four relation types are therefore a *within-dataset frequency-and-
recency heuristic on binary correctness sequences* -- adjacency-in-time of
right/wrong answers across concepts, thresholded at the row mean. This
is expressly acknowledged as a "statistical methodology" limitation in
Section 6, and the recency assumption (only immediately-adjacent
interactions count) is asserted, not tested against alternative windows.
There is no causal identification: co-occurrence and temporal adjacency
of correctness patterns are used as a proxy for causal facilitation or
interference, with no attempt to rule out confounds (e.g., item
difficulty ordering, curriculum sequencing that pre-orders concepts,
within-session fatigue, or shared latent ability driving both correct
answers). The paper's own worked example (Section 5.5, concepts c45 -> c70
/ c72) narrates the inferred sign post hoc as plausible pedagogy, not as
a verified causal claim.

## Validation of transfer claims: predictive ablation only, no external check, no nulls

All empirical support for "negative transfer helps" is internal predictive
ablation on the same two datasets used to fit the model:
- Table 5: LTKT vs. 8 published baselines (DKT, DKVMN, CKT, GKT, SKT,
  DKT+, ContextKT, SPARSEKT) on held-out AUC/F1. LTKT beats the next-best
  (SKT, the positive-transfer-only ancestor) by 2.06% AUC / 11.18% F1 on
  ASSIST15 and 1.01% AUC / 1.79% F1 on ASSIST09.
- Table 6: internal ablation removing whole model pieces -- LTKT-LTE (no
  transfer effect at all, direct-learning only), LTKT-NTE (drop the
  negative-transfer module), LTKT-PTE (drop the positive-transfer
  module), LTKT-beta (replace the learned mixing weight with unweighted
  sum). Removing NTE costs ~1.45% AUC / 1.58% F1 on average; removing PTE
  costs ~0.89% (partial figure visible, likely a similar order); removing
  the whole LTE costs 6.29% AUC / 11.88% F1.
- Section 5.4: sensitivity to a fixed floor hyperparameter gamma
  (0.05 optimal) that pads the graph to hedge against missing/incorrect
  edges -- itself an admission that the mined graph is imperfect and its
  errors need a safety valve, not a design lever validated against ground
  truth.
- Section 5.5-5.6: two illustrative case studies (one student's
  three-concept trajectory heatmap; a T-SNE clustering of concept
  embeddings colored by four manually-labeled knowledge domains,
  compared informally against domain labels the authors assigned by
  hand). These are qualitative narrative/visualization, not statistical
  tests, and are explicitly hedged ("not all clustering results are
  accurate").

There is no external reference standard anywhere in the paper: no expert-
annotated misconception graph, no cross-validation against a separate
curriculum/prerequisite ontology (e.g., no comparison to a domain-expert
concept map), no significance testing (no p-values, no confidence
intervals, no seed-variance reporting -- results in Tables 5-6 read as
point estimates from what appears to be a single run per cell), and no
placebo/null-graph control (e.g., they never compare against a
randomly-signed or randomly-permuted transfer graph to show the specific
NTPR/NTSR edges they mined, rather than "any extra capacity," are doing
the work). The entire causal/validity argument for "negative transfer is
real and this is what it looks like" rests on (a) held-out predictive
lift when you add an NTE module vs. not, and (b) one narrated example
trajectory. Predictive lift from adding a signed module is consistent
with the sign story but does not certify it: the same AUC gain could come
from added model capacity/regularization structure unrelated to the true
sign, since there is no control that isolates "correctly signed edges" from
"extra thresholded co-occurrence features."

## Datasets

Both are the two long-standard, low-concept-count ASSISTments dumps:
- ASSISTments 2015 (ASSIST15), "Skill Builder" format, mastery-gated
  (stops after 3 consecutive correct), N=100 concepts, 19,840 students,
  683,801 interactions, mean sequence length 34.47. Preprocessed split
  reused from Zhang et al. (the DKVMN paper).
- ASSISTments 2009 (ASSIST09), Non-Skill-Builder subset, N=154 concepts,
  8,026 students, 557,030 interactions, mean sequence length 69.40; only
  the first tagged concept per exercise is kept (single-KC assumption).

Mined graph statistics (Table 3): ASSIST15 yields 1112 PTPR, 1384 PTSR,
1203 NTPR, 1394 NTSR edges (NTPR+NTSR = ~26% of all mined relations);
ASSIST09 yields 2587 PTPR, 2152 PTSR, 2692 NTPR, 2796 NTSR (~23%). Both
are classic MOOC/skill-builder benchmarks with no institutional or
demographic diversity claim, no held-out different course, no adult /
LLM data, and no cross-dataset transfer test (a graph mined on one
dataset is never applied to or checked against the other).

## Limitations the authors state themselves (Section 6)

1. Single-concept-per-exercise assumption (shared with most KT
   literature they cite).
2. The statistical graph-mining approach degrades for concepts with
   thin interaction coverage -- an explicit cold-start admission.
3. The transfer-effect model only uses correctness-interaction
   statistics; it ignores response time, concept difficulty, and other
   covariates that likely confound the co-occurrence signal.
4. Future work list (external knowledge sources for graph construction,
   peer assessment, recommendation) implicitly concedes the graph is not
   independently validated now.

## Verdict for the kt-mirt program

LTKT does **not** preempt A1 (a certified signed cross-KC influence
claim) or A2 (a misconception-channel negative-transfer claim), but it
does narrow the novelty window and sets the bar A1/A2 must clear to be
distinguishable.

What LTKT already claims, so A1/A2 cannot claim as new: (a) the idea that
learning transfer between concepts can be negative as well as positive,
(b) a working architecture that carries two signed channels end to end
through a KT model, (c) predictive-ablation evidence that a negative-
transfer channel improves held-out AUC/F1 on ASSISTments-scale data.
Any A1/A2 pitch that stops at "we show negative transfer exists and
predicts better" restates LTKT's Finding 3 almost exactly.

What LTKT leaves wide open, and where A1/A2 have real room:
- **No certification of sign.** LTKT never checks whether the sign it
  assigns to an edge (via the co-occurrence heuristic) is the *correct*
  sign in any sense external to its own predictive loss -- no expert
  labels, no synthetic ground truth, no null/permutation control, no
  significance test. An A1 claim that includes a certification step
  (recovery on synthetic data with known signed ground truth, agreement
  with an external prerequisite/misconception ontology, or a
  permutation-null showing the specific mined signs beat randomly-signed
  graphs of the same density) would be a genuinely new bar LTKT does not
  clear.
  - **Sign is a fixed label, not a continuous learned quantity.** LTKT's
  "sign" is really "which of two RELU pipes an edge is pre-routed
  through by a preprocessing rule," fixed before training. A1/A2 framed
  around a *single* learned signed coefficient (able to flip sign from
  data, or express graded interference strength on a continuous scale)
  is architecturally distinct and untested by LTKT.
- **No misconception semantics.** LTKT's negative edges are
  "prerequisite-interferes" or "similarity-interferes" abstractions with
  no mechanistic story about *why* (no misconception channel, no error-
  type analysis, no distractor/wrong-answer semantics -- ASSISTments
  correctness data here is binary right/wrong, not multiple-choice
  distractor-coded). An A2 claim building an explicit misconception
  channel (e.g., keyed to specific wrong-answer types, not just binary
  incorrectness) is not addressed by LTKT at all.
- **No robustness/growth story.** LTKT is a single-fit predictive
  comparison on two small, old, single-institution datasets, no
  variance/seed reporting, no test of whether the mined graph or its
  transfer estimates are stable across resampling, cohorts, or time --
  exactly the "stable-and-wrong" failure mode this program is auditing
  in KT-IRT readouts generally. That auditing angle (is the sign/graph
  reproducible, or an artifact of one run on one dataset) is untouched.

Net: LTKT is close prior art that must be cited and distinguished from on
methodological grounds (LTKT: signed-graph-as-preprocessing-heuristic +
predictive-ablation-only validation), not ignored as irrelevant. A1/A2
survive as novel only if they add what LTKT explicitly lacks: an external
or synthetic certification of the sign, and/or a stability/reproducibility
audit of the claimed signed effect -- not just another predictive-lift
demonstration of "negative transfer improves AUC."
