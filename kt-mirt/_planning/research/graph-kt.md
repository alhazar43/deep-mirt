# Graph- and structure-based knowledge tracing, 2019-2026

Scope: how KC-KC structure is represented in graph-based KT, whether it is
directional and/or signed, how it is validated, and whether any prior work
operationalizes negative transfer between KCs. Includes a separate pass on
prerequisite/influence-structure discovery from logs, in and outside KT.

Note on sourcing: primary papers were located via web search (titles,
abstracts, and secondary summaries); direct PDF/HTML fetches were blocked by
network policy in this session, so several descriptions below rely on
search-engine-surfaced abstracts and secondary write-ups rather than a full
read of the primary text. Passages drawn only from secondary summaries are
flagged inline.

## Bottom line for the avenue map

**No canonical graph-KT paper (GKT, GIKT, SKT, PEBG) uses signed edges.**
Their KC-KC or question-KC graphs encode nonnegative
association/co-occurrence or "is-a-prerequisite-of" strength, never a
negative-valued or inhibitory relation. Directionality (prerequisite --
successor) is common; sign (facilitation vs. interference) is not, until
very recently.

**One paper claims to be the first to operationalize signed transfer between
KCs**: LTKT, "Knowledge Tracing Based on Positive and Negative Learning
Transfers" (Xu, Tang, Lv, Yu, Yu, Chen; *Tsinghua Science and Technology*,
2024) states explicitly it is "the first attempt to concurrently utilize the
positive and negative learning transfer relations among concepts"
[sciopen.com/article/10.26599/TST.2024.9010201]. That self-declared novelty
claim is itself indirect evidence that, as of the paper's writing, the field
had not done this before. Validation is predictive-ablation only (response
prediction on public benchmarks), not against any expert-labeled negative
edge, and I could not fetch the primary PDF to confirm the exact sign
mechanism (see caveat under LTKT below) -- treat the "signed and validated"
half of this claim as unverified pending direct-text confirmation.

One near-hit to flag and rule out: "A knowledge tracing approach with dual
graph convolutional networks and positive/negative feature enhancement
network" (Wang, Tang, Zheng; *PLOS ONE*, 2025,
journals.plos.org/plosone/article?id=10.1371/journal.pone.0317992) sounds
like signed KC edges but is not -- "positive/negative" there refers to
correct/incorrect student *response* features, not signed KC-KC relations.

## Canonical graph-KT models

### GKT -- Graph-based Knowledge Tracing (Nakagawa, Iwasawa, Matsuo)

- Venue: IEEE/WIC/ACM Web Intelligence 2019 (conference version,
  doi.org/10.1145/3350546.3352513); extended journal version in *Web
  Intelligence* 2021 (journals.sagepub.com/doi/full/10.3233/WEB-210458).
  Not on arXiv under that title; a workshop version appears at ICLR 2019
  RLGM workshop (rlgm.github.io/papers/70.pdf, unconfirmed direct read).
- Structure representation: KCs are nodes of a graph; KT is reformulated as
  a time-series node-classification problem solved with a message-passing
  GNN. Since no graph is given, GKT proposes two families of graph
  construction: (1) **statistics-based** -- build the adjacency from
  transition counts, e.g. how often KC *j* is answered right after KC *i*
  in the data; (2) **learning-based** -- learn the adjacency end-to-end
  with the prediction objective (including a DKT-derived variant, which
  the search summary reports as the best-performing graph-construction
  choice on ASSISTments2009).
- Directional: yes -- edges are typed into "incoming" and "outgoing" in the
  statistics-based construction, and the learned variant uses K distinct
  per-relation-type networks, i.e. an asymmetric, typed-edge structure.
- Signed: no evidence found of negative-valued edges; edge weights derive
  from nonnegative transition counts or softmax/attention-style learned
  weights.
- Datasets: ASSISTments2009, ASSISTments2015, Statics2011 (per search
  summaries; consistent across the Web Intelligence and journal versions).
- Validation of the structure itself: **predictive ablation only** --
  comparing statistics-based vs. learning-based graph construction methods
  by downstream AUC, not against an expert prerequisite map.
- Computational shape: GKT's selling point vs. DKT is that each timestep
  only updates the states of KCs connected to the just-practiced KC in the
  graph, rather than the full concept vector, giving a sparser per-step
  update (search summaries describe this as "updates only related
  concepts" vs. DKT's dense RNN state update); no explicit FLOP/parameter
  count was recoverable from search snippets.

### GIKT -- Graph-based Interaction model for Knowledge Tracing (Yang, Shen, Cui, Yu, Wang, Wang, Peng)

- Venue: ECML-PKDD 2020; arXiv 2009.05991 (arxiv.org/abs/2009.05991); code
  at github.com/ApexEDM/GIKT.
- Structure representation: a **question-skill bipartite graph** (built
  from the Q-matrix / exercise-skill tagging, i.e. given, not discovered)
  processed with a graph convolutional network (GCN) to propagate
  embeddings across question<->skill edges, so a question's embedding
  incorporates its own skills and, transitively, other questions sharing
  those skills (high-order question-skill correlation). The interaction
  model then further relates the student's current state, history states,
  the target question, and related skills.
- Directional: the bipartite graph is between two node types
  (question, skill); propagation is symmetric message-passing (GCN), not
  a directed prerequisite DAG.
- Signed: no; this is a co-occurrence/tagging graph, weights nonnegative.
- Datasets: ASSIST09 (123 skills, 3,841 students, 15,911 questions,
  190,320 records), ASSIST12 (265 skills, 27,405 students, 47,104
  questions, ~1.87M records), EdNet (>130M interactions, subsampled for
  experiments) -- reported 2-6 pp absolute AUC gain over prior SOTA.
- Validation of structure: none beyond the Q-matrix itself being taken as
  ground truth (i.e., the graph is *given*, not *learned or checked*
  against an independent structure); GCN depth L is ablated for predictive
  performance (L=0 vs L=1 propagation), which is a predictive ablation of
  a *given* graph's use, not validation of a *discovered* graph.
- Computational shape: GCN over question-skill bipartite graph plus an
  LSTM/attention interaction module; specific parameter counts not
  recoverable from available secondary sources in this session.

### SKT -- Structure-based Knowledge Tracing: An Influence Propagation View (Tong, Liu, et al.)

- Venue: appears to be IJCAI 2020 (paper mirror at
  home.ustc.edu.cn/~tongsw/files/SKT.pdf; Semantic Scholar record
  semanticscholar.org/paper/349d9d19fa823ee1392592e826a3618e38008392).
- Structure representation: **two separate, expert/derived graphs** over
  KCs are used simultaneously -- an undirected graph for *similarity*
  relations and a directed acyclic graph (DAG) for *prerequisite*
  relations. This is the clearest multi-relation-type design among the
  classic graph-KT papers.
- Directional: mixed by design -- similarity edges are undirected and
  propagate bidirectionally ("synchronization propagation"); prerequisite
  edges are directed and propagate unidirectionally, predecessor to
  successor only ("partial propagation"). Gated update functions combine
  the two channels at each timestep before feeding a recurrent predictor.
- Signed: no evidence of negative or inhibitory edges; both relation types
  are positive-association channels (co-membership/similarity, or
  prerequisite-enablement), differing in propagation topology, not in
  sign.
- Datasets: ASSIST09 among the benchmarks (per search summaries); the
  prerequisite/similarity graphs themselves appear to be externally
  supplied (curriculum-derived) rather than learned from logs -- I could
  not confirm the exact provenance (hand-built vs. platform metadata)
  without a direct read of the PDF.
- Validation: predictive-ablation style (does adding structure improve
  AUC), not verification of the supplied graphs against an independent
  gold structure -- SKT's contribution is architectural (how to propagate
  over a given multi-relation structure), not discovery of the structure.

### PEBG -- Pre-training Embeddings via Bipartite Graph (Liu et al.)

- Venue: "Improving Knowledge Tracing via Pre-training Question
  Embeddings", IJCAI 2020 (per liner.com review and ResearchGate mirror);
  extended as "Pre-training Question Embeddings for Improving Knowledge
  Tracing with Self-supervised Bi-graph Co-contrastive Learning," ACM
  TKDD 2024 (dl.acm.org/doi/10.1145/3638055).
- Structure representation: a **question-skill bipartite graph** (explicit
  Q-matrix relations) plus two auxiliary implicit-similarity graphs
  (question-question and skill-skill), fused via a product-layer network
  and pre-trained with a self-supervised objective to recover side
  information (skills, difficulty) before being plugged into downstream
  KT models (PEBG+DKT, PEBG+DKVMN).
- Directional / signed: bipartite bipartite/co-occurrence edges,
  nonnegative, no directionality claim (it is a pre-training embedding
  method, not a propagation-over-time model like SKT/GKT).
- Datasets: ASSIST09, ASSIST12, EdNet.
- Validation: purely predictive (downstream AUC when plugged into DKT /
  DKVMN); no comparison to an external prerequisite ground truth.

## Other relation-aware / prerequisite-aware KT since 2020 (selective, most relevant to signed/directional structure)

### PKT -- learnable prerequisite adjacency, "Prerequisite Structure Discovery in Intelligent Tutoring Systems" (arXiv 2402.01672, ar5iv-confirmed read)

- Represents the KC structure as a **learnable adjacency matrix** M,
  optimized jointly with the KT model by backpropagation from the
  prediction loss; M_ij = 1 if KC *i* is a prerequisite of KC *j*, so the
  structure is directional by construction.
- Signed: **no** -- confirmed by direct read; the matrix is binary/weighted
  positive only, no negative or inhibitory relation is modeled.
- Datasets: deliberately synthetic -- the authors argue public KT datasets
  already reflect expert-curated orderings and would make discovered
  structure circular to validate, so they build 10 simulators (10 KCs, 30
  exercises each) with a generative student model incorporating known
  prerequisite structure, difficulty, learner profiles, and forgetting.
- Validation: **two-pronged and unusually rigorous for this literature** --
  (1) directly score discovered-vs-ground-truth graph agreement with
  F1 (since the synthetic generator's true graph is known), and (2) an
  indirect downstream check, using the discovered graph to drive a
  simulated tutoring policy and measuring student outcomes. This is the
  strongest example found of validating discovered structure against a
  known-true graph rather than only via response-prediction ablation --
  but it buys that rigor by giving up real data.

### PSI-KT -- "Predictive, scalable and interpretable knowledge tracing on structured domains" (arXiv 2403.13179, ar5iv-confirmed read)

- Represents KC structure as a **weighted directed graph** A with edge
  a_ik = probability that KC *i* is a prerequisite of KC *k*, derived from
  low-dimensional KC embeddings (avoids O(K^2) blowup). Learned jointly
  across all learners (pooled, "prerequisites are time- and
  learner-independent") inside a hierarchical Bayesian state-space model
  with amortized variational inference.
- Directional: yes, by construction (a_ik is a prerequisite-direction
  probability).
- Signed: **this needs a precise flag.** The paper enforces
  a_ik + a_ki = 1 ("no mutual prerequisites") and a_ik is a sigmoid/
  probability output, i.e., confined to [0,1]. On a second, targeted read
  I confirmed the paper does **not** discuss negative transfer or
  inhibitory KC-KC influence anywhere; "signed" in this context means only
  that the direction is resolved (i-precedes-k vs. k-precedes-i), not that
  edges can carry a negative-valence weight. This is an important
  disambiguation: an early summary loosely called PSI-KT's edges "signed
  and directional," but the sign there is a *direction* label, not a
  *valence* (facilitative vs. interfering) label. **PSI-KT does not
  operationalize negative transfer.**
- Datasets: ASSIST12 (46,674 learners, 263 KCs, 3.5M interactions),
  ASSIST17 (1,709 learners, 102 KCs, 0.9M interactions), Junyi15 (247,606
  learners, 722 KCs, 26M interactions, includes crowd-sourced/expert KC
  relation annotations).
- Validation: the strongest real-data validation found in this survey --
  (1) ground-truth alignment against Junyi15's human-annotated KC
  relations using mean reciprocal rank, Jaccard similarity, and NLL;
  (2) cross-check against Bayesian "causal support" estimates computed
  independently from consecutive-interaction statistics; (3) downstream
  predictive-ablation. Junyi15 is flagged in the source material as one of
  the few public KT datasets with any human-annotated prerequisite
  relations at all, which is why it recurs across PSI-KT and other
  structure-validation attempts.
- Computational shape: embedding dimension D << number of KCs K to avoid
  quadratic scaling in the adjacency; hierarchical Gaussian state-space
  transitions allow analytic marginalization; mean-field variational
  inference with amortized (neural) inference network; a small number
  (4) of learner-specific trait dimensions. The paper's continual-learning
  experiments report psi-kt needing the least retraining time among
  compared methods (no absolute wall-clock/FLOP figures recovered).

### LTKT -- "Knowledge Tracing Based on Positive and Negative Learning Transfers" (Xu, Tang, Lv, Yu, Yu, Chen; Tsinghua Science and Technology, 2024/2026; also on SSRN, papers.ssrn.com/sol3/papers.cfm?abstract_id=4630827)

- The one paper in this survey that explicitly claims to be first to use
  **both positive and negative learning-transfer relations** among
  concepts. Architecture: a statistically-constructed "learning transfer
  graph" (LTG), plus a "direct learning effect" component (impact of a
  practice result on the concept it targets) and a "learning transfer
  effect" component (impact of that result on neighboring concepts in the
  LTG).
- Caveat: I could not get a direct-text fetch of this paper in this
  session (sciopen.com and SSRN both blocked by network policy here), so
  the mechanism above is reconstructed from search-engine abstract
  summaries only. **Treat as unverified**: I cannot yet confirm (a) exactly
  how "negative" transfer is operationalized numerically (a signed scalar
  edge weight? a separate negative-relation subgraph, SKT-style?), (b)
  whether the LTG is built from co-occurrence statistics alone (so
  "negative" could mean "answering A right predicts answering B wrong,"
  a purely correlational/predictive signal, not a validated causal or
  interference effect), or (c) whether it is validated against anything
  beyond response-prediction benchmarks. This paper is the single most
  important lead for G1 in this literature and should be fetched and read
  in full before being cited as precedent -- follow-up action, not a
  settled fact.

### Other relation/graph-structure KT papers surfaced but judged lower-priority for this survey (structure is either the Q-matrix, similarity only, or heterogeneous-but-unsigned)

- MAHKT -- multi-association heterogeneous graph embedding based on
  "knowledge transfer" (ScienceDirect, 2025); heterogeneous graph over
  multiple entity/relation types, but relations described (association
  types) are not reported as signed.
- DGEKT -- dual graph ensemble learning (arXiv 2211.12881); ensembles a
  skill-relation graph and a Q-matrix-derived graph, unsigned.
- Domain Generalizable KT via Concept Aggregation and Relation-Based
  Attention (arXiv 2407.02547) -- attention over cross-timestep
  question/concept relations for domain transfer, not signed KC influence.
- DAGKT -- difficulty- and attempts-boosted graph KT (arXiv 2210.15470) --
  augments GKT-style graphs with difficulty/attempt features, unsigned.
- "Extracting Causal Relations in Deep Knowledge Tracing" (Hong, Karbasi,
  Pottie; arXiv 2511.03948, EDM 2025 poster) -- argues DKT's predictive
  strength comes from implicitly approximating a **causal DAG** of
  prerequisite dependencies between KCs; they prune/extract exercise
  relation DAGs from DKT's learned representations and show training on
  causal-pruned subsets tracks DKT's own predictive behavior. This is
  directional and discovery-oriented (structure read out of a trained
  predictor, not given), but the write-up available does not indicate
  negative or inhibitory edges -- flagged as an unread primary source
  (only abstract-level summary available), worth a closer look given its
  causal framing overlaps with the lab's own emphasis on avoiding
  passive/shared-encoder mimicry of transfer.
- "Enhancing Explainability of Knowledge Learning Paths: Causal Knowledge
  Networks" (arXiv 2406.17518, EDM/HEXED 2024) -- builds a Bayesian-network-
  derived causal network over KCs for path recommendation; abstract-level
  summary gives no indication of signed edges.

## Prerequisite/influence-structure discovery from logs, outside pure KT

Per the task's explicit ask, discovery methods evaluated against expert or
independent ground truth (not just predictive ablation):

- **COMMAND** (Chen, Gonzalez-Brenes, Tian, "Joint Discovery of Skill
  Prerequisite Graphs and Student Models," EDM 2016,
  faculty.sites.iastate.edu/jtian/files/inline-files/edm-16.pdf) -- jointly
  infers a prerequisite Bayesian network and a student (performance) model
  from an n-students x p-items response matrix plus a Q-matrix. Validation:
  the discovered topological order of sections was checked for consistency
  with the **textbook's own chapter ordering** (a proxy ground truth, not
  formal expert annotation) and reported as fully consistent; also compared
  to a competing method, PARM, showing PARM recovers only a subset of
  COMMAND's relations. This is directional-only (prerequisite DAG), not
  signed.
- **UPreG** (Sabnis, Abhinav, Subramania, Dubey, Bhat, EDM 2021,
  educationaldatamining.org/EDM2021/virtual/static/pdf/EDM21_paper_252.pdf)
  -- infers prerequisite relations between MOOC concepts from unstructured
  course-description text via semantic-relatedness + statistical inference
  (NLP-based, not log-based interaction data). No ground-truth labels
  existed, so validation was a **human user study** (qualitative +
  quantitative), not comparison to an expert graph or predictive ablation.
  Directional only, unsigned.
- **ACE** (Aytekin et al., "AI-Assisted Construction of Educational
  Knowledge Graphs with Prerequisite Relations," JEDM,
  jedm.educationaldatamining.org/index.php/JEDM/article/view/737) --
  human-in-the-loop: a prerequisite-scoring mechanism (semantic/embedding-
  based) ranks candidate concept pairs, routes high-scoring/ambiguous pairs
  to domain experts for confirmation, and propagates confirmed edges
  transitively to reduce total expert labeling effort. Explicitly a hybrid
  discovery+validation loop against real experts, not a from-logs-only
  method (it is language/embedding-driven, not log-driven) -- included
  here because it is one of the field's clearest examples of a graph
  validated by construction against expert judgment rather than only by
  downstream prediction.
- **Prerequisite Relation Learning: A Survey and Outlook** (ACM Computing
  Surveys, 2026, dl.acm.org/doi/full/10.1145/3733593) -- broad survey
  covering both intrinsic evaluation (agreement with expert/gold
  annotations) and extrinsic evaluation (downstream task improvement) of
  prerequisite-relation-learning methods, taxonomized by whether they use
  KC-side features, learning-object-side semantic features, or
  cross-enhanced combinations. Flags that "the precise connections between
  knowledge concepts... depend significantly on pedagogical approach,"
  i.e. the survey itself treats ground-truth prerequisite structure as
  inherently context-dependent and hard to pin down -- relevant caution for
  any G1 validation plan that assumes a single "true" KC graph exists to
  check against.
- Also noted but not deep-dived: "Unsupervised Cross-Domain Prerequisite
  Chain Learning using Variational Graph Autoencoders" (arXiv 2105.03505)
  and its efficient follow-up (arXiv 2109.08722) -- prerequisite-chain
  discovery via VGAE over concept graphs, cross-domain transfer of learned
  structure; abstract-level only, evaluation approach not confirmed in
  this pass.

## Datasets and computational shape, summarized

| Model | KCs/skills | Students | Interactions | Graph type | Structure source |
|---|---|---|---|---|---|
| GKT | ASSIST09/15, Statics2011 sizes not confirmed here | -- | -- | KC-KC, typed directed | statistics- or learning-based from logs |
| GIKT | ASSIST09: 123 skills; ASSIST12: 265 skills | ASSIST09: 3,841; ASSIST12: 27,405 | ASSIST09: 190,320; ASSIST12: ~1.87M; EdNet: >130M | question-skill bipartite | given (Q-matrix) |
| SKT | ASSIST09 (size not confirmed) | -- | -- | KC-KC, dual (undirected similarity + directed DAG) | given/curriculum-derived (provenance unconfirmed) |
| PEBG | ASSIST09/12, EdNet | -- | -- | question-skill bipartite + 2 similarity graphs | given + implicit similarity |
| PKT (2402.01672) | 10 KCs x 10 simulators, 30 exercises each | synthetic | synthetic | KC-KC learnable adjacency | learned, validated vs. known synthetic truth |
| PSI-KT (2403.13179) | ASSIST12: 263; ASSIST17: 102; Junyi15: 722 | 46,674 / 1,709 / 247,606 | 3.5M / 0.9M / 26M | KC-KC weighted directed | learned, pooled across learners, validated vs. Junyi15 human annotations |

Blank cells indicate the figure was not recoverable from the sources
reachable in this session (see sourcing note at top).

## What this means for G1 (signed cross-KC influence) and the anchoring work

1. Directionality is well established in this literature (GKT, SKT, PKT,
   PSI-KT, COMMAND all use directed prerequisite structure); **sign is
   not**. Building a signed KC-KC influence readout is close to a genuine
   gap, not a reinvention -- LTKT is the only found claimant, its
   mechanism is unverified from primary text in this session, and even
   if confirmed it validates only against response-prediction benchmarks,
   not against any independent ground truth for negative transfer
   specifically (no dataset found in this survey has expert-labeled
   *negative* or *interfering* KC pairs -- only positive prerequisite/
   similarity annotations, e.g. Junyi15).
2. The strongest validation methodology found (PSI-KT's three-pronged
   check: expert-annotation alignment + independent causal-support
   statistic + downstream prediction) is a good template to imitate for
   G1, but note it validates *positive, directional* prerequisite
   structure, not signed influence -- there is no existing benchmark to
   transplant for the negative-transfer half of the claim.
3. PKT's synthetic-simulator validation strategy (generate data from a
   known ground-truth structure, then check F1 recovery) is the cleanest
   way found in this literature to get a hard, non-circular check on
   structure recovery, and could be adapted to inject synthetic negative-
   transfer edges as a pre-registered-null-style stress test before
   trying to certify negative transfer on real logs.
4. Immediate follow-up recommended before leaning on LTKT as precedent:
   obtain and read the LTKT primary text (Tsinghua Science and Technology,
   DOI 10.26599/TST.2024.9010201) directly, since it was unreachable via
   WebFetch this session and its claims here rest on secondary summaries
   only.

## Sources

- GKT: Nakagawa, Iwasawa, Matsuo, "Graph-based Knowledge Tracing: Modeling
  Student Proficiency Using Graph Neural Network," Web Intelligence 2019,
  doi.org/10.1145/3350546.3352513; journal version, Web Intelligence 2021,
  journals.sagepub.com/doi/full/10.3233/WEB-210458
- GIKT: Yang et al., arXiv 2009.05991, arxiv.org/abs/2009.05991; code
  github.com/ApexEDM/GIKT
- SKT: Tong, Liu, et al., "Structure-based Knowledge Tracing: An Influence
  Propagation View," home.ustc.edu.cn/~tongsw/files/SKT.pdf; Semantic
  Scholar semanticscholar.org/paper/349d9d19fa823ee1392592e826a3618e38008392
- PEBG: "Improving Knowledge Tracing via Pre-training Question Embeddings,"
  IJCAI 2020; extended version dl.acm.org/doi/10.1145/3638055 (ACM TKDD
  2024)
- PKT / prerequisite structure discovery: arXiv 2402.01672,
  arxiv.org/abs/2402.01672 (read via ar5iv.labs.arxiv.org/html/2402.01672)
- PSI-KT: arXiv 2403.13179, arxiv.org/abs/2403.13179 (read via
  ar5iv.labs.arxiv.org/html/2403.13179)
- LTKT: Xu, Tang, Lv, Yu, Yu, Chen, "LTKT: Knowledge Tracing Based on
  Positive and Negative Learning Transfers," Tsinghua Science and
  Technology, doi.org/10.26599/TST.2024.9010201 (abstract-level only,
  primary text unreachable this session); SSRN mirror
  papers.ssrn.com/sol3/papers.cfm?abstract_id=4630827
- PLOS ONE positive/negative feature enhancement (ruled out as KC-signed):
  journals.plos.org/plosone/article?id=10.1371/journal.pone.0317992
- COMMAND: Chen, Gonzalez-Brenes, Tian, "Joint Discovery of Skill
  Prerequisite Graphs and Student Models," EDM 2016,
  faculty.sites.iastate.edu/jtian/files/inline-files/edm-16.pdf
- UPreG: Sabnis et al., EDM 2021,
  educationaldatamining.org/EDM2021/virtual/static/pdf/EDM21_paper_252.pdf
- ACE: Aytekin et al., JEDM,
  jedm.educationaldatamining.org/index.php/JEDM/article/view/737
- Prerequisite Relation Learning survey: ACM Computing Surveys,
  dl.acm.org/doi/full/10.1145/3733593
- Extracting Causal Relations in Deep Knowledge Tracing: arXiv 2511.03948,
  arxiv.org/abs/2511.03948
- Causal Knowledge Networks: arXiv 2406.17518, arxiv.org/abs/2406.17518
- VGAE prerequisite-chain learning: arXiv 2105.03505 and arXiv 2109.08722
