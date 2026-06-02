# Bench Optimization + Finalization Plan

## Constraints (user)
- Allowed: debugging, minor fixes (e.g., seq loading, batch size, DataLoader workers, JIT compile)
- Forbidden: any change that modifies model behavior (different math = different output)
- Goal: speed up DKVMN-backbone training so ASSIST2015/2017 become feasible

## Current state
- Phase 2a (seeds 1, 7) running in background `b9parzogc`
- ETA Phase 2a ~02:00, Phase 2b ~03:55, all 5 seeds done by ~04:00
- 65 bench configs total: 5 models × 3 datasets × 5 seeds, minus 10 already-done static MA-GPCM/DKVMN+GPCM
- ASSIST2015 (~20K students) and ASSIST2017 (~22K students) deferred at user request earlier; configs at `configs/bulk/_bench_deferred/`

## Bottleneck (pre-investigation hypothesis)
- 4 of 5 models share `models/components/memory.py::DKVMN` with a Python `for t in range(S)` loop in their forward (`models/dkvmn.py:125`, `models/deep_irt.py:124`, `models/magpcm.py:236`)
- Per-timestep: tanh + matmul + softmax (attention), bmm (read), 2 sigmoid + 2 bmm (write)
- For B=64, S=200, that is 200 sequential Python-level ops per batch, roughly 50ms each on GPU
- `nn.LSTM` in DKT bypasses this via fused CUDA → DKT runs ~10× faster

## Phase A: investigate
- Delegate to ml-system-architect agent
- Profile a single training step on Static Q=200 K=2 to find the real bottleneck
- Compare to a reference DKVMN PyTorch implementation if available
- Identify safe optimizations

## Phase B: apply safe optimizations
Candidates (rank by expected impact):
1. **Larger batch size** (64 → 256 or 512). Amortizes Python overhead per timestep. No behavior change beyond random training-step variance; train final probabilities are not batch-size dependent given enough epochs
2. **DataLoader pin_memory + num_workers** for ASSIST datasets. Helps when CPU-GPU transfer is the slow part
3. **`torch.jit.script` the timestep loop body** if numerically equivalent
4. **Move all per-step tensor allocations out of the loop** (e.g., pre-compute attention over the whole sequence which we already do partially in dkvmn.py:120 but not in magpcm/deep_irt)
5. **Explicit CUDA streams** if I/O-bound

## Phase C: verify behavior preserved
- Train for 1 epoch on Static Q=200 K=2 with optimizations applied
- Compare ACC/AUC/loss to a baseline run
- Difference should be within run-to-run seed variance (~0.005 ACC)

## Phase D: deploy to ASSIST2015/2017 if speedup is real
- Move configs back from `_bench_deferred/`
- Launch sweep
- Hold until current Phase 2b finishes to avoid GPU contention

## Phase E: finalize (after 5-seed Phase 2b done)
1. Run aggregator → `outputs/bench_aggregated_5seeds.tex` (chained wrapper handles)
2. Replace `tab:combined_perf` in `main.tex` with the new table
3. Replace `\paragraph{Binary compatibility.}` prose using `_bench_writeup_draft.md`
4. If ASSIST2015/2017 results came in, add 2 more columns to the table
5. Compile, verify clean
6. Commit + push (overleaf-sync)

## Safety net
- Any optimization that changes output > 0.01 ACC at the same seed reverts
- Any change to model code requires checkpoint compatibility check (state_dict same shape and keys)
- Worst case: revert and live with the 3-dataset table
