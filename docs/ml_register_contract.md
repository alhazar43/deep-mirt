# ML register contract

A binding style contract for the register overhaul of `overleaf-sync/main_caeai.tex`,
distilled from three genre-matched exemplars (empirical audit of a modeling
practice, plus a diagnostic, plus statistical rigor). Every rule carries verbatim
evidence. The application map at the end is specific to our Section 3 and figure plan.

## The three exemplars

- **Precipice** = "Deep RL at the Edge of the Statistical Precipice" (arXiv:2108.13264, NeurIPS 2021). The statistics-audit exemplar.
- **Probes** = "Designing and Interpreting Probes with Control Tasks" (arXiv:1909.03368, EMNLP 2019). The diagnostic-construction exemplar.
- **pyKT** = "pyKT: A Python Library to Benchmark DLKT Models" (arXiv:2206.11460, NeurIPS 2022 D&B). The KT-audit neighbor.

---

## Binding rules

### Section and subsection grammar

**R1. Use a custom problem-driven top-level structure, not strict IMRaD.** All three deviate. Precipice runs Introduction / Formalism / Case Study / Recommendations / Re-evaluating Evaluation / Discussion. Probes runs Introduction / Control Tasks / Experiments on Probe Selectivity / Selectivity Differences Confound Layer Comparisons / Related Work / Conclusion.

**R2. Headings are concrete noun phrases, 2 to 8 words.** Evidence: "The Standardized Evaluation Protocol", "Representative DLKT Methods" (pyKT); "Selectivity Differences Confound Layer Comparisons" (Probes, 6 words); "Case Study: The Atari 100k benchmark" (Precipice, topic + colon + instance).

**R3. Cap the setup section at 3 to 4 subsections; push depth into one third level, not breadth.** pyKT's benchmark section is 3.1 Methods / 3.2 Datasets / 3.3 Protocol, with 3.3 alone splitting into 3.3.1 to 3.3.3. Seven flat subsections is off-genre.

**R4. Name the diagnostic in a heading and place it prominently.** Probes makes its instrument the second section outright ("2 Control Tasks"). The novel instrument earns a top-level or lead-subsection slot, not burial among protocol.

### Formal setup and notation

**R5. Introduce notation in one dedicated early section immediately after the Introduction.** All three place it as Section 2: "Formalism" (Precipice), "Control Tasks" (Probes), "Problem Statement" (pyKT).

**R6. Introduce notation in prose first, then crystallize it.** Evidence (Probes): "Let V be the vocabulary containing all word types in a corpus. A sentence of length T is x_1:T, where each x_i in V, and the word representations of the model being probed are h_1:T."

**R7. Carry a small number of focal display equations in the setup: target 2 to 4, never zero, never many.** Precipice has about 6 numbered equations across setup, Probes about 5, pyKT 0 (all inline). The central quantities get display; everything else stays inline. Prose-to-math ratio in setup is about 70:30 in all three.

**R8. The paper's signature quantity gets a numbered display definition AND a caption restatement.** Probes typesets selectivity as a definition and restates it in the Figure 2 caption: "Selectivity is defined as the difference between linguistic task accuracy and control task accuracy, and can vary widely, as shown, across probes which achieve similar linguistic task accuracies."

**R9. No heavy definition environments. Definitions are prose-embedded with inline notation.** Precipice keeps definitions prose-embedded; pyKT uses "No formal definition boxes." No `\begin{definition}`.

### Verbosity on standard details

**R10. Each dataset gets 1 to 3 sentences in the body; full statistics go to a table.** pyKT gives about one sentence per dataset across 7 datasets, with statistics in Table 1. Verbatim (pyKT): "ASSISTments2009: This dataset is made up of math exercises, collected from the free online tutoring ASSISTments platform in the school year 2009-2010. The dataset is widely used and has been the standard benchmark for KT methods over the last decade."

**R11. A benchmark is named and characterized in a single body sentence.** Precipice: "Our case study concerns the Atari 100k benchmark, an offshoot of the ALE for evaluating data-efficiency in deep RL." (18 words). Probes: "We use the Penn Treebank (PTB) dataset Marcus et al. (1993) with the traditional parsing training/development/testing splits without preprocessing." (one sentence).

**R12. Hyperparameters, compute, seeds, and convergence go to an appendix, not the body.** Precipice keeps setup "sparse in body; extensive in Appendix A.2"; the 15,600-run compute census lives in the appendix. pyKT relegates hyperparameters "entirely to Appendix A.3 (Table 7)." Exception: protocol that is itself the contribution stays in the body (Probes spends about 500 body words on complexity control because it is the method).

### Figures

**R13. Open with a page-1 or page-2 figure that is a concept schematic or a teaser.** Probes Figure 1 is a schematic of the control-task construction; pyKT Figure 1 is "The graphical illustration of the KT problem"; Precipice Figure 1 is a teaser data plot (runs per paper over the years) that motivates the whole audit.

**R14. At least the lead figure, and ideally about half of all figures, are schematic or conceptual diagrams.** Probes is 50% schematic (2 of 4), pyKT about 50% schematic. Precipice is data-heavy yet still leads with a teaser and keeps about 15% schematic.

**R15. The diagnostic or method itself gets a schematic.** Probes Figure 1 diagrams how a control task works; pyKT Figure 2 is "A recommended procedure for training and evaluating the DLKT models."

**R16. Caption grammar is register-split.** Data-plot captions are long and interpretive (about 50 to 100 words, and they state the conclusion). Schematic captions are terse (about 10 to 20 words, a label). Evidence, data (Precipice Fig 2, 96 words): "Left. Distribution of median normalized scores computed using 100,000 different sets of N runs subsampled uniformly with replacement from 100 runs ... The reported point estimates of median in publications, as shown by dashed lines, do not provide any information about the variability in median scores and severely overestimate or underestimate the expected median." Evidence, schematic (pyKT Fig 1, 10 words): "The graphical illustration of the KT problem."

### Voice

**R17. Lead with a declarative claim sentence, then qualify.** Precipice: "Ignoring the statistical uncertainty in deep RL results gives a false impression of fast scientific progress in the field."

**R18. Median sentence 20 to 30 words; allow causal chains to about 38.** Representative sentences run 17 to 38 words across all three.

**R19. Hedge lexically and only on causal or interpretive claims; state empirical findings flatly.** Findings: "We find that linear and bilinear models achieve higher selectivity..." (Probes). Hedged interpretation: "This could steer researchers towards superficially beneficial methods..." (Precipice); "We believe this is because..." (pyKT).

**R20. Criticize prior practice firmly but decorously. Signal the error with "Surprisingly" or "Unfortunately", name the mechanism, and keep "wrong / flawed / fake" out of prose.** pyKT: "Surprisingly, this crucial issue is neglected in some existing works ..." then "Unfortunately, this will cause the leakage of the ground truth." Understatement carries the punch: "The improvement of many DLKT approaches is minimal compared to the very first DLKT model."

**R21. Voice limitations pragmatically in a short dedicated section, granting the constraint.** pyKT Section 5 opens: "While pyKT is able to standardize and accelerate research in DLKT, we are aware of some potential limitations described as follows." Precipice frames the barrier as a tradeoff: "more rigor generally entails more nuanced and tempered claims."

### Statistical rigor presentation

**R22. Put uncertainty primarily in figures (shaded bands, error bars); report intervals in text only for headline contrasts.** Precipice presents CIs "predominantly in figures."

**R23. Criticize bare point estimates explicitly as uninformative.** Precipice, verbatim: "The reported point estimates of median in publications ... do not provide any information about the variability in median scores and severely overestimate or underestimate the expected median."

**R24. When a statistic appears in text, pair the estimate with its magnitude-of-variability context.** Precipice: "the score difference between sample medians with 5 and 100 runs for spr (+0.03 points) is about 36% of its mean improvement over drq (+0.08 points)."

---

## Application map for `main_caeai.tex`

### Current Section 3 (7 subsections, 0 display equations)

3 Models, data, and evaluation protocol -> 3.1 Model family and readout designs;
3.2 Synthetic testbeds and real datasets; 3.3 Recovery metrics and reporting
guards; 3.4 Statistical protocol; 3.5 Classical reference estimator; 3.6
Diagnostic instruments; 3.7 Adaptive testing simulation design.

This violates R3 (7 flat subsections), R4 (the slack test, our signature
instrument, is the sixth of seven), R7 and R8 (zero display equations; the slack
statistic is defined in prose: "Define the slack of a trained model as one minus
the Spearman correlation between its readout discriminations and their per-item
refit").

### Proposed Section 3 (5 subsections, 2 display equations)

- **3.1 Audited model family and readouts** (keep, rename). Promote the response-probability logit, now prose at lines 219 to 224, to one numbered display equation. This is the single anchoring equation of R7.
- **3.2 Testbeds and datasets** (keep, rename). Content and Table 1 already conform to R10 and R11; leave.
- **3.3 Metrics and statistical protocol** (MERGE current 3.3 + 3.4). Both are scoring and reporting; current 3.4 is only about 13 lines. Satisfies R3.
- **3.4 Diagnostic instruments: the slack test and the estimator ladder** (ELEVATE current 3.6, rename). This is the methodological contribution (R4). Place the **slack-statistic display equation here** as the second numbered equation, per the Probes selectivity precedent (R8). Add a schematic figure (see gap below). FOLD current 3.5 Classical reference estimator IN HERE as the ladder's top rung (it already is "classical marginal maximum likelihood on the full matrix, the reference estimator"); move its geometric-fact paragraph to a sentence in 3.2 pointing at Appendix A.
- **3.5 Adaptive testing simulation** (keep current 3.7, rename).

**Deaths and merges:** 3.4 merges up into 3.3; 3.5 dies as a standalone and folds
into 3.4. **Equation block:** two numbered display equations, the IRT logit in 3.1
and the slack statistic in 3.4, lifting the paper off the pyKT zero-equation floor
without exceeding the R7 ceiling. **Target length:** hold Section 3 to about 2
two-column pages while adding the schematic and two equations.

### Subsection renames elsewhere

The Results heads already conform to R2 and are stronger than the exemplars'; keep
them, with light trims: "Validation of the slack test as a truth-free diagnostic"
-> "The slack test as a truth-free diagnostic"; "Comparison with classical
calibration on real data" -> "Classical calibration on real data". Section 2
"Background" is generic; consider foregrounding content (for example "The shared
readout and its assumption"). Discussion heads ("Choosing an estimator in
deployment", "Artifacts caught by the certification protocol") conform; keep.

### Figure plan gap (the largest deficit)

The paper has 5 figures, **all data plots** (`fig_flip`, `fig_cat`, `fig_slack`,
`fig_twolaw`, `fig_surface`), and **zero schematics**. The first figure
(`fig_flip.pdf`) sits at line 439 inside Results 4.1; there is **no page-1 or
page-2 figure**. This violates R13, R14, and R15 head-on.

- **Gap 1, no concept Figure 1.** Add a schematic of the shared-versus-decoupled
  readout construction as Figure 1 on page 1 or 2. Note: `fig_architecture.tex`
  already exists in the repo but is **not wired into `main_caeai.tex`**; adapt and
  `\input` it. Highest-value single fix.
- **Gap 2, no estimator-ladder schematic.** The five-rung ladder is pure prose in
  current 3.6. Diagram it (five rungs, each granting one thing), matching the
  Probes Fig 1 and pyKT Fig 2 precedent of diagramming the method. Place in new 3.4.
- **Gap 3, optional teaser.** If the architecture schematic does not lead, promote
  a compact "stable and wrong" panel (matched accuracy versus divergent flags) to
  page 1 as a Precipice-style teaser.

Priority: (1) architecture schematic as Figure 1, (2) estimator-ladder schematic in
3.4, (3) leave the four data plots as is, upgrading their captions to the
interpretive length R16 requires where they fall short.
