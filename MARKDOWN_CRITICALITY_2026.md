# Markdown Paper-Criticality Audit (Post T0+T1)

Companion to `MARKDOWN_INVENTORY_2026.md` (ml-system-architect) and `CLEANUP_PLAN_2026.md`. Scope of this report is paper-criticality only. The classifications here are verdicts that should override an architect-only KEEP/MOVE/DELETE when the two disagree, because they answer a different question: does the paper, its appendix, or the rebuttal phase depend on this file.

Method: cross-referenced every non-obvious markdown against `overleaf-sync/main.tex` with grep on filename stems, label references, and topical keywords (linking, Kolen, Stocking, Haebara, equating, appendix, Deep-IRT, Synthetic-5). Read each candidate end-to-end. Verdicts below cite line numbers in main.tex where relevant.

---

## 1. Per-markdown verdicts

Files in the obvious-KEEP set (`README.md`, `CLAUDE.md`, `benchmarks.md`, `phd_research_proposal.md`, `CLEANUP_PLAN_2026.md`, `CLEANUP_VERIFICATION_2026.md`, `cleanup_log.md`) are not adjudicated here.

### 1.1 `ma-irt/NOTES_linking_appendix.md` -> KEEP (paper-source-material)

Status. Source material for an IJAIED linking-constants appendix that does not yet exist in `overleaf-sync/main.tex`.

Cross-reference check. Grep on `main.tex` for `linking`, `Kolen`, `Stocking`, `Haebara`, `equating`, `orbit`, `linking_constants` returns no matches. Grep on `appendix`, `\appendix`, `begin{appendix}`, `\section{Appendix` returns no matches either. The paper has no appendix at all in the current main.tex.

Why KEEP anyway.
1. The file is a structured plan, not a scratchpad. It defines the linking transform `est = A * true + B`, points at a real working script (`ma-irt/scripts/compute_linking.py`, exists on disk, last modified 2026-04-24), and seeds an aggregator schema for the 5-seed bulk retrain.
2. `REVIEW_psychometric.md` issue 1 explicitly flags raw-scale RMSE without linking as a severity-1 reviewer hazard. The linking appendix is the planned mitigation for that hazard.
3. `main.tex` line 367 already discusses the identifiability orbit and line 568 already discusses the alpha rotation that the linking transform quantifies. The appendix would directly support those existing passages.
4. The file is recent (2026-04-23, post-sigma=0.5 retrain) and its content is downstream of work that has actually happened.

Verdict. KEEP. Treat as live source material for the next paper revision. Do not move, do not delete.

### 1.2 `ma-irt/REVIEW_converged.md` -> KEEP (rebuttal-asset)

Status. Synthesized verdict of the two reviewer simulations, written 2026-04-24, after the sigma=0.5 retrain. Issue tracker for the next paper pass.

Cross-reference check. No grep match in `main.tex`. The file is not cited from the paper, because it is review material, not paper content.

Why KEEP.
1. The paper is under review at IJAIED. When the reviewer round comes back, the candidate will need a pre-built mental map of weak spots, conceded points, and prepared defenses. This file is exactly that map.
2. The Tier 1, 2, 3, 4 action list is operational and unfinished. Items like "drop bolding on sub-0.005 margins" (Table 3 K=5), "exp(raw_alpha) asymmetry clarifying sentence" (Sec 4.2.2), "best overall balance restated quantitatively" appear to be partially absorbed in the current main.tex but a side-by-side check is needed and that check requires this file.
3. The file flags one mechanical defect that was real at the time of writing (stale appendix `_full` tables disagreeing with main body). Grep on `_full` in current `main.tex` returns no matches, so the stale appendix has already been flushed; that disposes of one finding but not the rest.
4. The file is the only place that captures the cross-agent agreement on the headline contribution (separated-pathway threshold collapse). Useful for rebuttal framing.

Verdict. KEEP. Same disposition as `NOTES_linking_appendix.md`.

### 1.3 `ma-irt/REVIEW_psychometric.md` -> KEEP (rebuttal-asset, primary source)

Status. Mid-project psychometric reviewer simulation. 2026-04-24. The longer of the two sibling reviews (184 lines, 32 KB).

Cross-reference check. No grep match in `main.tex`. Not cited.

Why KEEP.
1. Same rebuttal logic as REVIEW_converged.md, but this is the primary source. The converged doc is a synthesis; the psychometric doc is the original evidence.
2. Contains the only written audit trail of the stale appendix table discrepancy (item 14, lines 156-166). If a reviewer raises the same concern, the candidate needs the original cross-check, not a summary.
3. Has specific operational findings that the converged file compresses (P1 through P10 sections). Some of these, like P2 (Kolen-Brennan linked RMSE) and P7 (bootstrap CI on r), are unresolved and may need to be acted on in a revision.
4. Cites Baker, Bock-Aitkin, Reckase, Embretson, de Ayala citations that may be needed for an appendix or rebuttal addendum.

Verdict. KEEP. Higher priority than REVIEW_converged.md when both contain the same finding because this one has the underlying evidence.

### 1.4 `ma-irt/REVIEW_research_scientist.md` -> KEEP (rebuttal-asset, primary source)

Status. Sibling DL/EduAI reviewer simulation. 2026-04-24.

Cross-reference check. No grep match in `main.tex`. Not cited.

Why KEEP.
1. Same logic as 1.3 but from the DL angle. A mixed reviewer panel at IJAIED can include both psychometric and ML reviewers, and the candidate needs prep notes for both.
2. Captures specific DL-side weak points (R1 modularity, R5 d_v ablation, R7 Deep-IRT head-to-head at K=2, R9 online-arrival experiment) that are unresolved and may be raised by an EDM/AIED reviewer.
3. R6 (training stability documentation) and R10 (Q-matrix analogy) suggest concrete prose tightenings that the candidate may still want to apply.
4. Notes a code-vs-paper detail (line 116, `irt.py` legacy `exp(0.3 * raw_alpha)` path not used by live MA-GPCM forward) that would be embarrassing if a reviewer reproduces the code and finds the mismatch. Useful flag for the next code-paper alignment pass.

Verdict. KEEP. Same disposition as 1.3.

### 1.5 `ma-irt/scripts/_bench_writeup_draft.md` -> ARCHIVE-eligible (writeup landed)

Status. 2026-04-27. Draft replacement paragraph for the Binary KT benchmark section (originally targeting lines ~523-543 of main.tex).

Cross-reference check. The current `main.tex` lines 468-471 contain the production binary-compatibility paragraph and Table `tab:combined_perf` lines 478-492. Comparing the draft against the current paper.

| Element | Draft | Current `main.tex` |
|---|---|---|
| Setup sentence | "Setting K = 2 collapses ... two-parameter logistic IRT model" | Line 468, near-identical phrasing |
| Five-model framing | "DKT through DKVMN, Deep-IRT, DKVMN+GPCM, MA-GPCM" | Line 469, identical model set, identical citations |
| Table caption | "Binary KT benchmark ... five seeds" | Line 478, "Binary prediction on Synthetic-Static, Synthetic-5, ASSIST2009, and ASSIST2017" |
| Narrative outcome | "within run-to-run variability on every dataset" | Line 471, "remain within run-to-run variance of the strongest baseline on every dataset" |

The draft content has been integrated into `main.tex`. Three small differences. The current paper distinguishes synthetic vs ASSISTments performance more sharply (line 471) than the draft. The current paper added the embretson 2000 attempt-invariance citation. The draft's "morning finalization" notes are stale.

Why ARCHIVE-eligible, not KEEP.
1. The writeup goal has shipped. Lines 468-471 of `main.tex` are the production version of this draft.
2. Nothing in the draft is not already in `main.tex` or stronger in `main.tex`.
3. The draft's seed-0 preview numbers (Synthetic-5, ASSIST2009) were superseded by the full 5-seed numbers in Table `tab:combined_perf`.

Caveat. The file has been useful as a paragraph-writing scratchpad and the candidate may have similar drafts in flight for the next revision. If there is a pattern of using underscore-prefixed `_*_draft.md` files for in-flight prose, the convention should be preserved even if this specific file is archived. Confirm with candidate before moving.

Verdict. ARCHIVE-eligible. Move to `docs/archive/2026-06-cleanup/ma-irt/scripts/` if the candidate confirms the draft is no longer load-bearing for the next pass. Default action absent confirmation is MOVE-to-archive, not DELETE.

### 1.6 `ma-irt/scripts/_profile_dkvmn_report.md` -> ARCHIVE-eligible (engineering artifact)

Status. 2026-04-27. Profiling output table from a DKVMN forward+backward profiling run. CPU profile only, no GPU.

Cross-reference check. No mention of FLOPs, wall-clock, profile, or efficiency table in `main.tex`. Grep on `efficiency`, `wall-clock`, `latency`, `FLOPs`, `profile` against `main.tex` returns nothing.

Why ARCHIVE-eligible.
1. The data is engineering scratch, not paper content. The paper does not currently report any efficiency or profiling numbers (REVIEW_research_scientist.md item 9 specifically flags this as a missing experiment, but not a missing prose section).
2. The profile was run with B=64, S=200, Q=123, K=2, which does not match the K=4 cells the paper headlines. If an efficiency table is ever added, the profile would need re-running on canonical configurations anyway.
3. Source script (`_profile_dkvmn.py`) exists and is callable; this report is a frozen output, not the live measurement.

Caveat. If REVIEW_research_scientist.md item 9 (efficiency comparison table) ever becomes a Tier 3 action, this file is the seed for it. Until then it adds nothing.

Verdict. ARCHIVE-eligible. Move with the bench writeup draft.

### 1.7 Legacy repo READMEs (`mirt-dkvmn/README.md`, `deep-gpcm/README.md`, `deep-1pl/README.md`, `dkvmn-ori/README.md`, `dkt-ori/README.md`, `akt/README.md`, `pykt/README.md`, plus `pykt/docs/source/*.md`) -> NO SEPARATE DISPOSITION

Status. Reference reading material from the seven legacy repos that account for 2.3 GB at the root.

Cross-reference check. Grep on `mirt-dkvmn`, `deep-gpcm`, `deep-1pl`, `dkvmn-ori`, `dkt-ori`, `pykt`, `akt` against `main.tex` returns no matches. None of these READMEs are cited by the paper.

Verdict. Ride T2 (legacy repo disposition) wholesale. No paper-criticality reason to break them out separately.

Note. The legacy repos themselves are not all disposable. `CLEANUP_PLAN_2026.md` notes that `dkvmn-ori/data/synthetic/` and `deep-1pl/data/synthetic/` are read by `ma-irt/scripts/_build_pykt_synthetic5.py` and `_convert_yeung_synthetic.py`, and `deep-gpcm/data/assist2009_dkvmn/` is referenced from `convert_dkvmn_format.py`. T2 must preserve those data subtrees regardless of whether the README at the repo root survives.

### 1.8 `ma-irt/.pytest_cache/README.md` -> NOT A REAL FILE FOR THIS PURPOSE

Pytest-generated, auto-regenerated on first test run. Ride T0 (build artifact) if not already gone.

### 1.9 Markdown inside `docs/archive/2026-06-cleanup/` -> ALREADY ARCHIVED

Already moved by T1 (per `cleanup_log.md` entry 2026-06-02). Includes the phd_blueprint series, the per-Mar-29 changelog and plan docs, `RETRAIN_PLAN.md`, `PAPER_NOTES.md`, `BENCH_OPT_PLAN.md`, `BINARY_BENCH_TODO.md`, `proxy-ord-mapping.md`, etc. No further action.

One subtlety. `proxy-ord-mapping.md` is referenced by docstrings in `ma-irt/scripts/convert_assistments.py` and `ma-irt/scripts/convert_assistments_2009.py`. Those docstrings were updated to point at the new archived location per the cleanup log. Verify the docstring path still resolves before any further moves under `docs/archive/`.

---

## 2. Hard-stop list

Files that must NOT be touched even if a later cleanup tier flags them as stale. These are paper-load-bearing or reproduction-load-bearing.

### 2.1 Paper source and figures

- `C:\Users\steph\documents\deep-mirt\overleaf-sync\main.tex`
- `C:\Users\steph\documents\deep-mirt\overleaf-sync\title_page.tex`
- Any `.pdf`, `.pgf`, `.tex` under `C:\Users\steph\documents\deep-mirt\overleaf-sync\figures\` (the paper reads from here, not from top-level `figures/`)
- `C:\Users\steph\documents\deep-mirt\overleaf-sync\ref.bib` (if present)

### 2.2 Paper-source markdowns

- `C:\Users\steph\documents\deep-mirt\ma-irt\NOTES_linking_appendix.md` (linking appendix source material)
- `C:\Users\steph\documents\deep-mirt\ma-irt\REVIEW_converged.md` (rebuttal asset)
- `C:\Users\steph\documents\deep-mirt\ma-irt\REVIEW_psychometric.md` (rebuttal asset, primary source)
- `C:\Users\steph\documents\deep-mirt\ma-irt\REVIEW_research_scientist.md` (rebuttal asset, primary source)

### 2.3 Top-level orientation and reproduction docs

- `C:\Users\steph\documents\deep-mirt\README.md`
- `C:\Users\steph\documents\deep-mirt\CLAUDE.md`
- `C:\Users\steph\documents\deep-mirt\benchmarks.md` (canonical experiment table)
- `C:\Users\steph\documents\deep-mirt\phd_research_proposal.md` (PhD proposal, separate from paper but explicitly KEEP per cleanup plan)
- `C:\Users\steph\documents\deep-mirt\CLEANUP_PLAN_2026.md`, `CLEANUP_VERIFICATION_2026.md`, `cleanup_log.md` (cleanup orchestration)

### 2.4 Project README

- `C:\Users\steph\documents\deep-mirt\ma-irt\README.md` (slightly outdated relative to current code; see Section 5)

### 2.5 Scripts referenced by the linking appendix

- `C:\Users\steph\documents\deep-mirt\ma-irt\scripts\compute_linking.py` (referenced by `NOTES_linking_appendix.md`; not a markdown but required to remain callable for the next aggregation pass)

---

## 3. KEEP-but-relocate recommendations

The four KEEP markdowns at `ma-irt/` root could be moved to a `ma-irt/docs/` subdirectory to reduce root-level noise. Per-file recommendation.

### 3.1 `ma-irt/NOTES_linking_appendix.md` -> relocate to `ma-irt/docs/notes/linking_appendix.md`

Rationale. The file is single-purpose, references one script, and is unambiguously a notes file. Moving it to `ma-irt/docs/notes/` (or `ma-irt/docs/`) signals "appendix source material" rather than "live working document." No cross-reference in `main.tex` to update. The script reference inside the file is to a path (`scripts/compute_linking.py`) that does not change.

Caveat. If the candidate already mentally indexes this file as "at the root of ma-irt," moving it may cost retrieval time during the rebuttal phase. Confirm before moving.

### 3.2 `ma-irt/REVIEW_converged.md` -> relocate to `ma-irt/docs/reviews/converged.md`

Rationale. Same as 3.1. Reviews are not source code, do not need to sit at the root.

### 3.3 `ma-irt/REVIEW_psychometric.md` -> relocate to `ma-irt/docs/reviews/psychometric.md`

Same.

### 3.4 `ma-irt/REVIEW_research_scientist.md` -> relocate to `ma-irt/docs/reviews/research_scientist.md`

Same.

### 3.5 Subdirectory naming

If the candidate creates `ma-irt/docs/reviews/` for these, also consider a `ma-irt/docs/notes/` sibling for the linking appendix file, so that future notes (next reviewer round, future appendices) have a home.

### 3.6 Do not relocate

- `ma-irt/README.md` stays at `ma-irt/` root. GitHub and other tools surface the top-level README of a directory automatically.
- `ma-irt/scripts/_bench_writeup_draft.md` and `_profile_dkvmn_report.md` are ARCHIVE-eligible, not KEEP-and-relocate.

---

## 4. Status of paper-rebuttal materials

The candidate is under review at IJAIED. The three REVIEW_*.md files were created 2026-04-24, after the sigma=0.5 retrain and before the IJAIED submission (May 8 submission archive `IJAIED-sub.zip` at root). Their function shifts depending on review stage.

### 4.1 Pre-first-round (now)

KEEP. The reviews supplied the framing decisions and identifiability discussion that landed in the current submission. They also flagged the stale `_full` appendix tables, which the candidate handled by flushing the appendix (grep confirms no `_full` labels remain in `main.tex`). Several Tier 2 and Tier 3 findings in `REVIEW_converged.md` may be in flight or unaddressed and need cross-reference against the current paper before declaring them obsolete.

Specific unresolved-as-of-2026-06-02 items worth tracking.
- F7 (exp(raw_alpha) asymmetry clarifying sentence). Grep on `exp(raw_alpha)` or "absorb" or "scale factor" in current `main.tex` does not show the clarification. Likely still open.
- P2 (Kolen-Brennan linking). Not addressed in current `main.tex` (no grep match on Kolen, Stocking, Haebara). Linked to the NOTES_linking_appendix.md plan.
- R5 (d_v hyperparameter ablation). Not in current `main.tex`. Likely still open.
- R8 (efficiency comparison table). Not in current `main.tex`. Still open.

### 4.2 First reviewer round comes back

KEEP, escalate. The pre-built findings list in `REVIEW_converged.md` is exactly the rebuttal-prep document the candidate needs. Do not archive until the reviewer round is closed.

### 4.3 After resubmission accepted

ARCHIVE-eligible. At that point the reviews have served their purpose and can move to `docs/archive/2026-06-cleanup/ma-irt/reviews/` (or wherever the archive convention puts them by that time). Do this in a separate cleanup tier, not before.

### 4.4 If the paper is rejected

KEEP indefinitely. The reviews would seed the next submission's framing.

### 4.5 Rule-of-thumb decision tree for next tier

- If the next tier runs before reviewers return. KEEP all four (NOTES + 3 REVIEWs) untouched.
- If the next tier runs after reviewers return and before resubmission. KEEP all four, possibly relocate per Section 3.
- If the next tier runs after acceptance. ARCHIVE the three REVIEWs. KEEP NOTES_linking_appendix.md until the linking appendix is either in the published paper or definitively dropped.

---

## 5. Side observations worth flagging

These are not paper-criticality verdicts but they fell out of the cross-reference pass and may affect the next tier.

### 5.1 `ma-irt/README.md` is mildly stale

The README references `OneHotEmbedding` and a `linear_decay` embedding default, while CLAUDE.md and the actual code now default to `LearnedEmbedding` (per `CLAUDE.md` line "embedding_type: 'onehot', 'learned' (default), or 'static_item'"). The README also lists scripts (`plot_recovery.py`, `plot_recovery_figure.py`, `plot_learner_trajectories.py`) that do not all exist or have been renamed. This is content drift, not deletion-worthy, but a small refresh pass is overdue. Not in scope for this audit.

### 5.2 The linking script exists, the aggregator does not

`ma-irt/scripts/compute_linking.py` exists. The aggregator over 5 seeds described in `NOTES_linking_appendix.md` (TODO section) does not. If the linking appendix is on the rebuttal roadmap, the aggregator needs to be written before the appendix can be drafted. Not blocking for this audit.

### 5.3 The "stale appendix tables" problem is already solved

`REVIEW_psychometric.md` issue 14 and `REVIEW_converged.md` TL;DR flag `tab:irt_recovery_k_full`, `tab:block_recovery_full`, `tab:rw_recovery_full` as stale. Grep on `_full` in current `main.tex` returns no matches. The appendix has been flushed, so this specific Tier 1 mechanical item in `REVIEW_converged.md` is closed. The REVIEW_*.md files do not need an update note for this, but the candidate should know it is resolved.

### 5.4 `submission_2026-05-09/` and `_overleaf_old/`

Both directories sit at the repo root and may contain markdown that this audit did not enumerate (the Glob output did not list any `.md` under them). If a later tier discovers paper-related markdown there, treat with the same KEEP rules as `overleaf-sync/`.

---

## 6. Summary table

| Path | Verdict | Reason |
|---|---|---|
| `ma-irt/NOTES_linking_appendix.md` | KEEP | Source material for planned linking appendix; addresses severity-1 reviewer hazard |
| `ma-irt/REVIEW_converged.md` | KEEP | Rebuttal asset; pre-built findings list for IJAIED reviewer round |
| `ma-irt/REVIEW_psychometric.md` | KEEP | Rebuttal asset; primary source with full evidence trail |
| `ma-irt/REVIEW_research_scientist.md` | KEEP | Rebuttal asset; DL-side weaknesses inventory |
| `ma-irt/scripts/_bench_writeup_draft.md` | ARCHIVE-eligible | Content landed at `main.tex` lines 468-471 |
| `ma-irt/scripts/_profile_dkvmn_report.md` | ARCHIVE-eligible | Engineering scratch; no paper section to feed |
| Legacy repo READMEs (7 files) | Ride T2 | Not cited by paper; legacy-repo disposition tier |
| `ma-irt/README.md` | KEEP, mildly stale | Project README; refresh pass overdue (Section 5.1) |
| `ma-irt/.pytest_cache/README.md` | Ride T0 | Build artifact, auto-regenerated |
| `docs/archive/2026-06-cleanup/**/*.md` | Already archived | T1 already moved these |

End of report.
