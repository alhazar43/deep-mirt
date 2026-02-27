# TODO: kt-gpcm Paper (AAAI 2026)

Active project: `kt-gpcm/`. Main paper: `paper.tex`.

---

## Done

- [x] Recovery figure — 3×6 compact layout (theta KDE + alpha + β1–β4, one row per model), proper IRT linking (log-linked α, z-scored β), inserted into paper.tex RQ2
- [x] RQ2 writing — competitive α recovery, slightly lower β, dynamic θ framing
- [x] RQ3 writing — Dynamic GPCM vs memory models, DEEP-GPCM vs DKVMN+Softmax similarity insight
- [x] All figures set to `[htb!]`
- [x] `\def\mathdefault#1{#1}` added to preamble (PGF/pdflatex compatibility)
- [x] Trajectory figure height reduced (4.6×3.6)

---

## In Progress / Remaining

### RQ1 — Prediction

- [ ] Confirm QWK, ordinal accuracy, MAE numbers in `tab:comp_results` match final checkpoints
- [ ] Resolve any `\textcolor{red}` placeholders in RQ1 section

### RQ2 — Parameter Recovery

- [ ] Fill `r_θ` column in recovery table (point-to-point theta correlation, noted as construct mismatch in text)
- [ ] Split-half reliability experiment: divide students into 2 halves, correlate α̂_j, β̂_{j,k} across halves

### RQ3 — Learner State Dynamics

- [ ] Current data has *static* θ — trajectories show posterior convergence, not learning
- [ ] Optionally: add `--dynamic_theta` flag to `data_gen.py` and retrain for genuine learning curves

### RQ4 — Scalability

- [ ] Confirm wall-clock numbers in paper match final runs (127–129 s/epoch)

### RQ5 — Ecological Validity

- [ ] ASSISTments binary subset → ordinal proxy pipeline (`scripts/prepare_assistments.py` exists)
- [ ] Train DEEP-GPCM and DKVMN+Softmax on proxy dataset; fill `tab:proxy_ordinality`

### Paper / Writing

- [ ] Resolve all `\textcolor{red}` placeholders before submission
- [ ] Second pass on abstract and conclusion once all results are final
