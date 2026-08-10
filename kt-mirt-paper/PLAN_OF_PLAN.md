# Plan-of-the-plan, Paper 2 (the kt-mirt paper)

Written 2026-07-31 early morning for author approval before any agent
drafting launches. Nothing below is prose for the paper. The working
directory for everything paper-related is this folder, `kt-mirt-paper/`,
to keep it apart from the code tree (`kt-mirt/`) and from Paper 1's
overleaf (`overleaf-sync/`).

## 0. Where the evidence engine stands right now

Thread A (influence) closed tonight and is committed. Its results are
paper-ready. Thread B (deep-Junyi growth test) is memory-solved but
slow. The measured cost is roughly 1.5 to 2 hours per permutation
replicate, so the pre-registered 39-replicate run needs about 2 to 3
days. It was resubmitted with the measured settings (single-replicate
chunks, 72 hour limit, job 560630) and is monitored. The draft must
therefore carry the deep-Junyi cell as pending, with text ready for
either outcome. That is fine. The paper does not stand or fall on it.

## 1. Where this paper sits

Paper 1 (`overleaf-sync/main_caeai.tex`, the measurement-audit paper,
campaign complete) says the ability readouts of prediction-trained
knowledge tracing can be stable and wrong, diagnoses the mechanism, and
repairs specific parameters. Closed.

Paper 2 generalizes the lesson from parameters to claims. If readouts
can be confidently wrong, then any DOWNSTREAM claim built on them
(this learner grew, skill A helps skill B) needs its own certification
before it is asserted on real data. The kt-mirt program spent the last
two weeks building and executing exactly that discipline. The paper
reports the discipline and its two demonstrations.

## 2. Proposed thesis (F1, recommended)

A multi-skill learner model earns each of its two headline readouts,
per-cohort ability GROWTH and SIGNED skill-to-skill association
(facilitation and interference), by passing a certification harness
before touching real data. The harness is built from matched synthetic
twins (growth and no-growth versions of the same cohort), permutation
nulls, pre-registered decision bars, and confound arms designed to kill
the claim. The product is not only the two detectors but the BOUNDARY
MAP that certification produces, stated as measured quantities.

The boundary map is the interesting part, and every entry is a result
we actually obtained, several of them refusals.

- Growth is detectable in the pooled cohort but per-skill resolution
  hit a fundamental identifiability floor. The framework refuses
  per-skill growth claims.
- Saturated skills make the gate fire falsely. The framework masks
  near-ceiling skills and refuses range-starved claims (verified on
  twins where the fix removes the false fire and keeps the true one).
- On real data the growth gate fires on deep-practice KDD and stays
  correctly silent on thin-practice Junyi. The per-student depth
  requirement is bracketed by a designed ladder (105, 557, 2688 mean
  rows per student), with the middle rung in flight.
- Signed association is recoverable per edge, with a MEASURED
  sensitivity floor for interference (twice the reference dose). Below
  it, fabricated edges on truly-empty pairs are indistinguishable from
  signal. The floor was certified on clean seeds after the tuning
  hypothesis (a stronger sparsity penalty) was tested and refuted.
- A pre-registered order-shuffle arm KILLED the causal-temporal
  reading. The certified claim is signed dose-association. The paper
  leads with this honesty rather than hiding it.
- A deliberately-freed per-learner multiplier fabricates influence
  from nothing on 5 of 5 seeds, which validates the metric and the
  design's parameter pinning.
- Multi-tag (EdNet-shaped) item densities collapse the coefficients.
  Association reading is refused on that density class.

Framing rule honored (from the program's standing memory): the home is
prediction-trained knowledge tracing, IRT is the explainability lens.
The contribution is the certify-then-claim protocol plus the measured
boundary map, not psychometric theory.

Working title directions (pick later, not now): "Certified readouts
for multi-skill knowledge tracing", "A boundary map for learner-model
claims", "Refuse, then claim".

## 3. Alternative framings (weighed, not recommended)

- F2, influence-first. Lead with signed association (the novel
  detector), growth as substrate. Weakness: the influence leg has no
  real-data edge yet, so the headline sits on synthetic certification
  alone. Risky as the lead but fine as a section.
- F3, growth-only with the density boundary. Cleanest evidence but
  discards half the assets and shades into Paper 1's territory.
- F1 subsumes both and degrades gracefully if a reviewer demands a
  split.

## 4. Honest gaps and threats (the stress-test)

1. The influence read is certified on synthetic beds only. The
   designed external check (Junyi's curated prerequisite graph as an
   answer key for positive edges) has not run. Without it the paper
   must scope the association claim as certified-in-harness, not
   demonstrated-in-the-wild. Optional add-on below.
2. Association, not causation, after the shuffle kill. This must be
   presented as the certification working as intended. A reviewer who
   wanted causal transfer gets a measured reason it is not claimable
   at this grain, plus the logged revival path (a lag-contrast
   statistic, unvalidated, future work).
3. Nearest analogs must be differentiated early. Growing Pains
   (2604.12843) is the closest for growth-vs-noise. PSI-KT, GKT/GIKT,
   LTKT/HawkesKT for structured influence. The originality sweep says
   the certification-and-boundary angle is open. The related-work
   stage re-verifies this against 2026 literature rather than trusting
   the old sweep.
4. The deep-Junyi cell lands in 2 to 3 days. Both outcomes are
   writable. Fires: density mechanism demonstrated. Silent: the
   boundary narrows and the claim stays a boundary statement.
5. Venue collision. Paper 1 is CAEAI-first. Recommendation below goes
   to JEDM for this one (methods-heavy, boundary-map-shaped, and the
   audience that audits KT models). IJAIED holds Chapter 0.
6. Cost. The overnight pipeline below is token-heavy by design. The
   author approved ultracode. Stages gate on my read of intermediate
   output, so a bad direction dies at a checkpoint, not at the end.

## 5. Optional evidence add-on (decision requested)

Start the Junyi prerequisite-graph external-alignment pilot now, in
parallel with drafting. It is the single strongest add for the
influence half (positive edges predicted by a curated human graph).
Honest cost statement: it is a full-K fit (roughly 835 exercise-grain
skills), a scale the association read has never run at, where the
false-edge background makes multiplicity control mandatory, and the
design gates it behind stages not yet executed. It may not certify by
morning or at all. If approved it runs as exploration feeding a
scoped subsection or an appendix, never as a headline dependency.

## 6. Format and register

Structural model is `overleaf-sync/main_caeai.tex` (the program's own
journal-article architecture). Register comes from the archived MA-GPCM
paper (`overleaf-sync/old/main_magpcm_ijaied.tex`) plus the
writing-style memory, both passed verbatim to every prose agent. No
LaTeX tonight. Deliverables are markdown. Conversion to tex happens in
a later, separate session once the author signs off on content.

## 7. The overnight pipeline (launches only on approval)

Stage 0, scaffold (minutes, mechanical). This folder gets PLAN.md
(frozen version of this document plus decisions), LOG.md (running
ledger of the pipeline itself), and subfolders `evidence/`,
`related-work/`, `outline/`, `drafts/`, `review/`.

Stage 1, grounding (parallel readers, no prose). One agent team
extracts every claimable result with its provenance from
`kt-mirt/_planning/` (LEDGER, THINKING, both verdict files, CT0
report, designs, triage) into `evidence/claim_evidence_map.md` with a
strength tag per claim (certified / real-data / synthetic-only /
pending / killed-refused). A second team reads Paper 1's plan and tex
for continuity language and the no-overclaim boundaries between the
papers. A third runs the external sweep (deep-research skill plus web
verification) on the named analogs and any 2026 newcomers, producing
`related-work/dossier.md` with verified citations only.

Stage 2, framing stress-test (adversarial panel). Independent agents
attack F1 from reviewer personas (the KT skeptic, the psychometrician,
the causal-inference reviewer, the "just engineering" dismissal). A
judge consolidates surviving framing into `outline/framing_memo.md`.
I gate here and adjust before drafting.

Stage 3, architecture. Section-by-section outline with every claim
bound to its evidence row and figure plan (each figure named with the
exact source data under `kt-mirt/outputs/`), in
`outline/paper_outline.md`. The ARS outline mode contributes a second
independent outline; divergences get reconciled, not averaged.

Stage 4, section drafting (max effort, register-locked). One agent per
section writes markdown prose into `drafts/`, claims sized exactly to
the evidence map, kills and refusals written as results. Introduction
and discussion drafted last, after body sections exist.

Stage 5, internal review and revision. The ARS reviewer panel plus my
own adversarial verification workflow review the assembled draft.
Findings triage into must-fix (applied tonight) and author-decision
(listed). Morning package: complete markdown draft v0, outline,
evidence map, related-work dossier, review report, and a plain-language
summary of what was claimed and what was refused.

## 8. Decisions requested from the author

1. Framing F1 (certify-then-claim framework with boundary map) yes/no.
2. External-alignment pilot in parallel, yes/no.
3. Venue target JEDM (with CAEAI fallback), or otherwise.
4. Confirm deliverable form, markdown draft tonight, tex later.
