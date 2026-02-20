# TODO: kt-gpcm Paper (IJAIED)

Active project: `kt-gpcm/`. Checkpoint: `outputs/large_q5000_static/best.pt`.

---

## In Progress

- [ ] **Recovery figure** — combined 3-panel plot (theta KDE + alpha scatter + beta scatter, normalized) from `large_q5000_static` checkpoint; insert into paper.tex RQ2 section.

---

## RQ1 — Prediction

- [ ] Train DKVMN+Softmax baseline on `large_q5000` dataset (same splits, same seed)
- [ ] Train DKVMN+Ordinal baseline
- [ ] Compute QWK, ordinal accuracy, MAE for all 3 models → fill `tab:comp_results`

---

## RQ2 — Parameter Recovery

- [ ] Fill `r_θ` column in `tab:recovery` (run theta correlation for each encoding)
- [ ] Split-half reliability experiment: divide students into 2 random halves, correlate α̂_j, β̂_{j,k} across halves; target r > 0.90

---

## RQ3 — Learner State Dynamics (requires dynamic θ data)

**Problem**: current synthetic data has *static* θ drawn once per student from N(0,1). Trajectories show posterior convergence, not learning. The current trajectory plot is valid as "convergence to θ*" but NOT as "learning dynamics."

- [ ] Modify `kt-gpcm/scripts/data_gen.py` to support `--dynamic_theta` flag:
  - Draw θ_0 ~ N(0, 0.5) per student
  - Apply a power-law growth curve: θ_t = θ_∞ + (θ_0 − θ_∞) · exp(−λ·t), where θ_∞ ~ N(1, 0.5) and λ ~ Uniform(0.005, 0.03)
  - Generate responses at each step using θ_t (not θ_0)
- [ ] Retrain on dynamic-θ data and regenerate trajectory plot showing genuine learning curves
- [ ] Once DKVMN+Softmax baseline is trained, overlay its ŝ_t on each panel for comparison

---

## RQ4 — Scalability (mostly done)

- [ ] Confirm wall-clock numbers in paper match final runs (currently 127–129 s/epoch)
- [ ] Add `r_θ` to `tab:recovery` (blocked by RQ2 above)

---

## RQ5 — Ecological Validity / Proxy-Ordinality

- [ ] Identify suitable ASSISTments binary subset with attempt-count metadata
- [ ] Implement attempt-count → ordinal-category bucketing pipeline
- [ ] Train DEEP-GPCM and DKVMN+Softmax on the ordinal proxy dataset
- [ ] Compare QWK, AUC-ROC (binary collapsed), fill `tab:proxy_ordinality`

---

## Paper / Writing

- [ ] Replace trajectory figure once dynamic-θ retrain is complete
- [ ] Update pendingbox in RQ3 section once DKVMN+Softmax baseline trains
- [ ] Fill recovery metrics table (`tab:recovery`) `r_θ` column
- [ ] Resolve all `\textcolor{red}` placeholders before submission
