# Binary Benchmarking TODO

## Goal
Replace stale `tab:combined_perf` (currently DKT/DKVMN/MA-GPCM × Static/ASSIST2015/Synthetic-5) with a fresh comparison across **5 models × 5 datasets**, 5 seeds each, mean ± sd.

## Final table layout
Rows top-to-bottom:
1. DKT
2. DKVMN
3. Deep-IRT
4. DKVMN+GPCM
5. **MA-GPCM** (bold)

Columns:
- Static (synthetic Q=200 K=2)
- ASSIST2009 (binary)
- ASSIST2015 (binary)
- ASSIST2017 (binary)
- Synthetic-5

## Discovery summary
- **Submodules** in `.gitmodules`: `dkvmn-torch`, `dkvmn-ori` (MXNet), `deep-1pl` (Deep-IRT), `akt`, `deep-gpcm` (cloned)
- **Raw data** in `assisstment-raw/`: `skill_builder_data_2009.csv`, `non_skill_builder_data_2009.csv`, `anonymized_full_release_competition_dataset_2017.csv`
- **Archived prepared data** at `archive_sigma03_20260422_0534/data/`: `assist2009_ord_k4`, `assist2015`, `synthetic5`, plus `raw_sources/{assist2015,synthetic5}`
- **Archived benchmark runs** at `archive_sigma03_20260422_0534/outputs/bench_*_synthetic5_s42`, `assist2009_dkvmn_gpcm_*` etc.
- **Already-trained MA-GPCM and DKVMN+GPCM at Static Q=200 K=2** with 5 seeds (`outputs/static_{magpcm,dkvmn_gpcm}_q200_k2_learned_s*`)

## Phases

### Phase 1: Discovery (this session)
- [x] Inspect parent dir, submodules, archived data
- [ ] Audit `deep-gpcm` submodule for DKT / DKVMN / Deep-IRT model code (might already have what we need)
- [ ] Verify archived `assist2009_ord_k4`, `assist2015`, `synthetic5` data formats
- [ ] Check raw ASSIST2009, 2017 to plan binary conversion

### Phase 2: Datasets
- [ ] Prepare/verify `data/assist2009_bin` (binary correctness from skill builder)
- [ ] Prepare/verify `data/assist2015_bin`
- [ ] Prepare/verify `data/assist2017_bin` (from already-converted K=4 → squash to K=2 by correctness)
- [ ] Prepare/verify `data/synthetic5_bin`
- [ ] Static synthetic K=2 already at `data/static_q200_k2`

### Phase 3: Models
- [ ] DKT: implement or adapt (probably need a small PyTorch DKT model)
- [ ] DKVMN: clone `dkvmn-torch` or adapt our existing DKVMN backbone with binary head
- [ ] Deep-IRT: clone `deep-1pl` or adapt
- [ ] DKVMN+GPCM: existing MAGPCM with separate_theta=False, set K=2
- [ ] MA-GPCM: existing MAGPCM, set K=2

### Phase 4: Configs
- [ ] DKT × 5 datasets × 5 seeds = 25 configs
- [ ] DKVMN × 5 datasets × 5 seeds = 25 configs
- [ ] Deep-IRT × 5 datasets × 5 seeds = 25 configs
- [ ] DKVMN+GPCM × 4 new datasets × 5 seeds = 20 configs (Static already done)
- [ ] MA-GPCM × 4 new datasets × 5 seeds = 20 configs (Static already done)
- Total: ~115 new configs

### Phase 5: Training (next session)
- Estimated 115 runs × 5-6 min/run on synthetic ≈ 12-15 hours
- ASSISTments datasets may be longer per run

### Phase 6: Aggregate + write table
- Compute mean ± sd per (model, dataset)
- Replace `tab:combined_perf` in `main.tex`
- Best per column in bold (excluding MA-GPCM bold-row convention)
