# TODO: DEEP-GPCM Paper

Active project: `kt-gpcm/`. Main paper: `paper.tex`. See `PAPER_PLAN.md` for venue strategy.

## Done
- [x] Recovery figures (split: student theta KDE + item alpha/beta scatter)
- [x] Learner trajectory figure (4 archetypes)
- [x] RQ1-RQ4 writing complete
- [x] RQ5 class imbalance experiment and writeup
- [x] Theta recovery bug fix (student ID alignment)
- [x] DKVMN+Ordinal removed (was never a real baseline)
- [x] Focal loss removed from paper (WOL only)
- [x] Dead code archived (9 scripts)
- [x] Notation consistency fixes (N/B/j collisions, threshold count)
- [x] All bib entries cited
- [x] PGF/pdflatex compatibility macro

## Paper
- [ ] Add framework interface paragraph to Methodology (DEEP-MIRT head contract)
- [ ] Rewrite Contributions 1-3 to separate framework from GPCM instantiation
- [ ] Add standard deviations to tables (run >= 3 seeds per condition)
- [ ] Resolve any `\textcolor{red}` placeholders
- [ ] Second pass on abstract and conclusion
- [ ] Add ~10 more bibliography references (target 40-70)
- [ ] Switch to Springer Nature template (currently article class)

## Experiments
- [ ] Real-world data evaluation (ASSISTments proxy-ordinality or similar)
- [ ] Split-half reliability: correlate alpha/beta across student halves
- [ ] Consider dynamic theta data generation for genuine learning trajectories
- [ ] Regenerate Q=2000 dataset (5000 students, seq_len [100,200]) for better recovery
- [ ] Multiple simulation DGPs (vary alpha spread, beta spacing)

## Code
- [ ] Add model_type validation to build_model() — raise on unknown type
- [ ] Remove legacy `use_separable_embed` / `response_dim` from config and model (check checkpoint compat)
- [ ] Clarify `dkvmn_ordinal` model_type handling in baseline configs
