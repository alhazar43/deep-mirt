# Lens A5: experiment-inventory (framing review, 2026-07-17)

## Experiment-Inventory Findings: What Was Fit vs. What the Paper Shows

### 0. Framing correction that changes how to read everything below

`docs/paper_plan_v2.md` (the "plan of record" I was pointed to) describes a **different paper** than what is in `overleaf-sync/main_caeai.tex` today. The plan's architecture is the "disease / slack-test / refit / rebuild / invoice" selling frame (S1-S11, T1-T5+F1-F3, CAT decision-cost receipts, "196.8% length"). The actual tex file's own header comment reads:

> `% "On the Prediction-Recovery Trade-off in Interpretable Knowledge Tracing` (line 2)

This is a plainer SH-vs-SK (shared-head vs. separated-key item-parameter path) amortization paper. It contains **no slack test, no "disease" framing, no dollar/percentage invoice, no reviewer checklist**. Two section headers are still tagged `[FULL-REWORK]` (Discussion, Conclusion, lines 1370/1487), confirming this is mid-rewrite. **Whoever produced the "glued together, no aha" criticism read this draft, not the plan.** Any diagnosis of the storyline problem has to be grounded in this tex, not the plan.

### 1. The single most concrete finding: a dangling cross-reference to a deleted section

Line 1382: *"The simulation in Section~\ref{sec:downstream} does not instantiate a fully online calibration system."* **No section or subsection in the document is labeled `sec:downstream`.** The only adaptive-testing paragraph in the file is commented out (lines 872-880, `% \paragraph{Adaptive-testing simulation.}` … all lines `%`-prefixed). Grep for every "adaptive test/CAT/cut-score" mention (18 hits) shows the paper **asserts CAT results in the abstract (line 51), intro (line 93), tab:beds row (922-925), tab:hyper block (1719-1723), Discussion (1381-1399, a full paragraph of specific claims — "SH parameters increase test length and cut-score error… SK reduces these costs but does not remove them"), Limitations (1466, 1478), and Conclusion (1511-1515) — with zero supporting table, figure, or Experiments subsection.** This will render as an undefined-reference LaTeX warning at compile time. This is not a stylistic gap; it is the mechanical signature of "point that was never there" — a third of the paper's own claimed contributions point at content that isn't in the draft.

Underlying CAT simulation data **does exist** on disk (not lost, just disconnected — see §5 below).

### 2. The experiment matrix — what was actually fitted

**Synthetic grid** (ground truth known), stored at `kt-irt/results/p2_toggle` (2PL/GPCM) + `kt-irt/results/p2_nrm_repar/arm1r` (NRM):

| Axis | Values fit |
|---|---|
| Encoder | LSTM, Transformer, DKVMN (3) |
| Decoder | 2PL, GPCM, NRM (3) |
| Design | shared (SH), decoupled/separate (SK) (2) |
| N | 500, 1000, 2000, 5000 (4) |
| Q | **200 only**, in the cells any table/figure reads |

3×3×2×4 = **72 static cells, fully populated**, no gaps — verified by directory listing (`tog_{lstm,transformer,dkvmn}_{2pl,gpcm}_{shared,decoupled}_static_N{500,1000,2000,5000}_Q200`, 48 dirs; `rep1r_{lstm,transformer,dkvmn}_nrm_{shared,decoupled}_static_N{500,1000,2000,5000}_Q200`, 24 dirs). This exactly covers tab:mass (18 cells, N=2000, line 941) + tab:massfull (54 cells, N∈{500,1000,5000}, line 1534) — "no cell repeated" (line 1538) checks out.

**Undisclosed but fully-fit sibling grid**: the paper's own Methods text claims *"The main grid crosses learner count N∈{500,1000,2000,5000} with item-bank size Q∈{200,500,1000,2000}"* (line 821). That full N×Q cross **was fit** — `outputs/p2_exposure` (641M, shared) and `outputs/p2_exposure_sep` (184M, SK) at the repo root, confirmed present, e.g. `grid_dkvmn_2pl_N2000_Q500`, `grid_lstm_2pl_sep_N5000_Q2000`. It was archived at port time ("no CAEAI figure/table" — usage map) and **never made it into `kt-irt/results/`**. No table or figure in the current draft reports anything as a function of Q. The Methods paragraph over-describes the grid relative to what Results shows — compute that's already paid for and sitting unused.

**Real data** — frozen pre-registered panel, identical hardcoded constant duplicated across **four** driver files (`_p2_realstudy.py:DECODER_DATASET`, `_p2_v3_metrics.py:PANEL`, `_p2_v3_metrics_allenc.py:PANEL`, `_p2_real_prediction_baselines.py:PANEL`, all literally `[("ednet","2pl"),("kdd","2pl"),("timss","gpcm"),("ednet","nrm")]`):

| Decoder × Dataset | Fit? | Encoders | Fold coverage |
|---|---|---|---|
| 2PL × EdNet | yes | LSTM/TF/DKVMN | 25/25 (LSTM,TF); DKVMN reduced |
| 2PL × KDD | yes | LSTM/TF/DKVMN | 25/25 (LSTM,TF); DKVMN reduced |
| GPCM × TIMSS | yes | LSTM/TF/DKVMN | 25/25 all three |
| NRM × EdNet (routed/arm1r) | yes | LSTM/TF/DKVMN | 25/25 (LSTM,TF); DKVMN reduced |
| GPCM × EdNet | **no** | — | — |
| GPCM × KDD | **no** | — | — |
| 2PL × TIMSS | **no** | — | — |
| NRM × KDD | **no (impossible)** | — | — |
| NRM × TIMSS | **no (impossible)** | — | — |

Measured DKVMN fold reduction, verified against the paper's own footnote (line 1081-1084): *"DKVMN-direct uses 3 folds; DKVMN-SH uses 5–6 folds; DKVMN-SK uses 5 folds."* I sampled `timing_s.total` from stored fold JSONs and this is not arbitrary — DKVMN is the cost outlier by 10-50×:

| Cell (encoder × dataset-decoder) | measured s/unit |
|---|---|
| LSTM × 2PL-EdNet | 12-23 |
| LSTM × 2PL-KDD | 12.7 |
| LSTM × GPCM-TIMSS | 24.9 |
| LSTM × NRM-EdNet(routed) | 82-100 |
| Transformer × (any of above) | 32-39 |
| DKVMN × GPCM-TIMSS | 62-65 |
| DKVMN × 2PL-EdNet | 632-640 |
| DKVMN × 2PL-KDD | 528-638 |
| DKVMN × NRM-EdNet(routed) | 1196-1248 (~20 min/unit) |

DKVMN cost tracks the **dataset** (EdNet's 250-item bank, 200-step sequences) far more than the decoder — TIMSS (small poly-triplet matrix) stays cheap under DKVMN while EdNet does not. This is the real reason DKVMN's real-data folds were cut, and it directly informs the cost table below (the task's assumed "~1-2 min/unit" holds only for LSTM/Transformer; DKVMN blows past it by an order of magnitude on EdNet/KDD).

MML classical references (`p2_realstudy/mirt/`) mirror the frozen panel exactly: `ednet_2pl, ednet_nrm, kdd_2pl, timss_gpcm` — no more, no less.

### 3. Paper-usage overlay (cross-checked line-by-line against the tex)

| Exhibit | Line | Coverage |
|---|---|---|
| tab:mass | 941 | synthetic, 3enc×3dec×SH/SK, N=2000 |
| tab:massfull | 1534 | synthetic, same grid, N∈{500,1000,5000} |
| fig:dd | 989 | synthetic, 18 pts = 3enc×3dec×2 param families, N=2000 |
| fig:scatter | 1006 | synthetic, **LSTM only**, one fold (seed 1, fold 4) |
| tab:real_prediction | 1052 | EdNet-2PL, KDD-2PL, TIMSS-GPCM, EdNet-NRM × 3 encoders × {direct,SH,SK} + MML — the **only** real-data exhibit with all 3 encoders |
| fig:agreement_both | 1114 | **LSTM only**; EdNet-2PL + TIMSS-GPCM + EdNet-NRM (routed); KDD absent |
| fig:timss_case_shsk | 1142 | **LSTM only**, TIMSS-GPCM only |
| fig:ednet_2pl_shsk | 1194 | **LSTM only**, EdNet-2PL only |
| tab:ednet_two_resolution | 1221 | EdNet-2PL vs EdNet-NRM, seed-mean/25 folds |
| fig:reversal_bridge | 1293 | **LSTM only**, EdNet-NRM, "fold 0 of five seeds" (caption, line 1302) — not even the full 25 units |
| fig:ednet_case_shsk | 1346 | **LSTM only**, EdNet-NRM (routed), same fold-0 restriction |
| tab:app_timss_item_thresholds | 1590 | TIMSS-GPCM, item-level, both designs |
| tab:app_ednet_2pl_shsk | 1642 | EdNet 2PL+NRM within-reading |

**KDD is an orphan column.** Grep for "KDD" across all 1762 lines returns exactly 6 hits: one table column (line 1063) and five generic dataset-listing sentences (885, 886, 929, 1028, 1048). Zero figures, zero external-agreement checks, zero prose analysis paragraph — every other dataset (EdNet, TIMSS) gets a subsection (§4.3 "Ordinal KT with Partial-Credit," §4.4 "Align Correctness and Option Structure"); KDD gets one number per encoder-arm in one table and is never mentioned again. If the "glued together" feeling has one single sharpest instance, this is it: a whole dataset that contributes a column and nothing else.

Every figure and every in-depth real-data claim is **LSTM-only**; the 3-encoder claim rests entirely on one table.

### 4. Costed extension menu

Grounded in loader code, not assumption:

**(a) EdNet-GPCM — feasible, code exists, unused.**
`_p2_real.py::_load_ednet` (lines 152-176) already branches on `decoder=="gpcm"`, calling `er.build_ordinal_responses(items, corr, elapsed, all_train)` — confirmed K=3 ordinal (`0=wrong, 1=correct-slow, 2=correct-fast`, item-median elapsed-time split, train-fold-only to avoid leakage; `_ednet_reliability.py:183-196`). Not wired into the frozen panel (`DECODER_DATASET` has no `("gpcm","ednet")`). No cache exists (`data/cache/` has `ednet_2pl_*` and `ednet_nrm_*`, no `ednet_gpcm_*`).
- Code touch: add one tuple to 4 files' `PANEL`/`DECODER_DATASET` constants (`_p2_realstudy.py`, `_p2_v3_metrics.py`, `_p2_v3_metrics_allenc.py`, `_p2_real_prediction_baselines.py`).
- Raw data: `EdNet-KT1`/`EdNet-Contents` present at the **deep-mirt repo root** (4.1G/605K) but not staged inside `kt-irt/` (PORT_ROOT resolves to wherever `pyproject.toml` sits, i.e. `kt-irt/`) — needs a one-time copy/stage before the cache-build scan can run.
- Compute (measured LSTM/TF proxies from 2PL/NRM-EdNet, extrapolated for DKVMN from EdNet's DKVMN-2PL/NRM cost since DKVMN scales with dataset not decoder): LSTM ≈17 min, Transformer ≈31 min, DKVMN ≈2.75-12.5 GPU-hours (reduced-fold vs full-25) for 6 cells (3 enc × SH/SK) × 25 units. Plus one-time raw-scan cache build ("minutes," per `_p2_datacache.py` docstring).
- Command (after the edit): `python -m deep_irt.bench._p2_realstudy --device cuda --skip-done --only "*_gpcm_ednet_*"` — **not** the unit CLI, because `deep-irt-train-unit --driver realstudy` resolves through `_p2_realstudy_hardnrm.all_cells()`, which is hardcoded to NRM×EdNet only (`_p2_realstudy_hardnrm.py:99-102`) — a different, narrower driver than the one that made the frozen-panel cells.
- MML row needs a separate `_p2_mml_realstudy_reference.py --run` pass [PC-R, needs `DEEP_IRT_RSCRIPT`].

**(b) KDD-GPCM — feasible, code exists, unused.**
`_p2_real.py::_load_kdd` (lines 227-245) already branches on `decoder=="gpcm"` (attempts/hints K=3 proxy-ordinal, `_kdd_reliability.py:8-13`: cat2=correct-first-attempt, cat1=no-hint struggle, cat0=hint-used). Same 4-file PANEL edit. Raw KDD (`data/kdd/algebra_2008_2009_train.txt`, 3.1GB) present at deep-mirt root, not yet staged in `kt-irt/`. Compute: 6 cells × 25 units ≈ 11 min (LSTM) + 28 min (Transformer) + 1.8-8.3 GPU-hours (DKVMN, reduced/full). Same command pattern, same MML caveat.
**KDD-NRM is impossible** — `_load_kdd` explicitly raises `ValueError(f"kdd: unsupported decoder {cell.decoder!r} (no options)")` (line 245): the KDD Cup log has no recorded distractor/option identity, only correctness+attempts+hints.

**(c) TIMSS-2PL — feasible but requires new modeling code, not a flag flip.**
`_p2_realstudy.py::load_dataset` hard-blocks this today: *"if decoder != 'gpcm': raise ValueError('TIMSS is GPCM-only in this study')"* (lines 276-277). No dichotomization function exists anywhere in the codebase. Filling this cell means **writing** a rule collapsing the 0/1/2 partial-credit score into binary (candidates: "any credit correct" vs "full credit only correct") — this is a genuine scientific choice, not infrastructure, and changes what "TIMSS-2PL" even measures. Cheapest in raw compute of the three feasible fills (TIMSS's poly-triplet matrix is small; DKVMN there costs 62-65s/unit, nothing like EdNet/KDD): 6 cells × 25 units ≈ 85 GPU-minutes total, all three encoders, no fold reduction needed. No new raw-data staging (TIMSS CSVs already in `kt-irt/data/timss/`).

**(d) TIMSS-NRM — impossible, no caveat.**
`_load_timss` (lines 187-215) reads `data/timss/timss_g8_usa_poly_triplets.csv`, schema `(student, item, resp)` where `resp` is a partial-credit **score category** (0/1/2), not a selected answer option. TIMSS constructed-response items are free-response, rubric-scored — there is no distractor-option field to model at all in this dataset. This is a data-acquisition problem (a different TIMSS release with multiple-choice items), not a code problem. Say so and stop.

### 5. Highest-leverage item not asked for but found in-lens: reviving CAT is cheap relative to the new-cell fills above

The CAT simulation this draft's abstract/intro/discussion/conclusion assert (§1 above) is not vaporware — its outputs sit on disk: `outputs/p2_cat/` (per-fold JSON, e.g. `dkvmn_2pl_fold_d0_f0.json`), `outputs/p2_cat_retrofit/` (fold-level), `outputs/p2_cluster/cat_clustered.json` (the seed-clustered summary the plan quotes: "shared inflation [180.4,210.8], decoupled [144.1,168.0]…"). These consume the **already-fit, unchanged, frozen synthetic grid** (§2), so the numbers are very likely still valid without any retraining.

The gap is the driver script, not the data: `git log --all --diff-filter=A -- "*p2_cat*.py"` finds `deep_irt/bench/_p2_cat.py` was added historically but **does not exist** in the current parent tree or in `kt-irt/` — it was retired in the 2026-07-14 port (consistent with the port-time usage map already noting "the CAT figure/table is commented out of CAEAI"). Recovering it (`git show <commit>:deep_irt/bench/_p2_cat.py`), re-porting it to the current module/`_portroot` layout, and re-verifying against the present engine API is real but bounded engineering work — cheaper than any of (a)-(c) in GPU-hours, since no new training is implied, only re-plumbing + writing the T4/F3 exhibit and the results paragraph the framing sections already narrate in prose. This is the fork worth putting in front of the author explicitly: either restore the CAT section that the rest of the draft already assumes exists, or strip the dangling CAT claims out of abstract/intro/discussion/conclusion/tab:hyper to match a draft that is honestly SH/SK-recovery-plus-real-data-checks only.

### Key file paths
- Paper: `C:/Users/steph/documents/deep-mirt/overleaf-sync/main_caeai.tex`
- Frozen real-data panel (4 files, identical constant): `kt-irt/src/deep_irt/bench/_p2_realstudy.py:94-100`, `_p2_v3_metrics.py:60`, `_p2_v3_metrics_allenc.py:74`, `_p2_real_prediction_baselines.py:58`
- EdNet/KDD loaders (GPCM branches already coded): `kt-irt/src/deep_irt/bench/_p2_real.py:152-245`
- TIMSS loader + hard GPCM-only guard: `kt-irt/src/deep_irt/bench/_p2_realstudy.py:187-215, 276-277`
- Modern unit CLI is NRM-EdNet-only, not the frozen-panel driver: `kt-irt/src/deep_irt/bench/_p2_realstudy_hardnrm.py:99-102`
- Dataset cache dir (what's actually cached): `kt-irt/data/cache/`
- Archived, un-ported, unused-by-paper Q-exposure grid: `outputs/p2_exposure/`, `outputs/p2_exposure_sep/` (repo root)
- Retired CAT driver (git-recoverable only): `deep_irt/bench/_p2_cat.py` (historical path, not in current tree)