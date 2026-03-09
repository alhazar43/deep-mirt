# Immediate Actions Required

## Critical Issues Confirmed

I've analyzed the paper and data. You're correct on all counts:

1. ✅ **Section references broken** - Found 5 instances of `\ref{sec:...}` that will fail
2. ✅ **RQ1 table wrong** - K=2 included, no recovery metrics, 15 columns (too wide)
3. ✅ **Missing recovery in ablations** - ALL ablation tables lack r_α and r_β
4. ✅ **RQ4 scalability wrong** - Missing Q=2000, no recovery, not systematic
5. ✅ **Writing purely numerical** - Lists metrics without psychometric insights

## What I've Created

### 1. Recovery Computation Script
**File**: `kt-gpcm/scripts/compute_all_recovery.py`

Computes r_α and r_β correlations for all 144 experiments by:
- Loading each checkpoint
- Running inference on train+test data
- Computing item-level parameter estimates
- Correlating with ground truth (with IRT linking)
- Saving to CSV

### 2. Data Merging Script
**File**: `kt-gpcm/scripts/merge_all_metrics.py`

Merges prediction + recovery data and generates tables:
- RQ1 main comparison (K=3,4,5,6 with recovery)
- RQ4 scalability (Q × encoding × recovery)
- Ablation tables (loss, monotonic, sequence length)
- Full merged dataset

### 3. Comprehensive Plans
**Files**:
- `PAPER_REVISION_PLAN.md` - Detailed revision plan with examples
- `CRITICAL_ISSUES_ANALYSIS.md` - Issue analysis and action plan
- `IMMEDIATE_ACTIONS.md` - This file

## Data Status

### What We Have ✅
- **Prediction metrics**: Complete (144 experiments)
  - summary_all_experiments.csv (64 experiments)
  - summary_baselines.csv (80 baselines)
- **Checkpoints**: All present (best.pt in each output dir)
- **Configs**: All present (configs/*.yaml)
- **Recovery plots**: PNG files exist (but no CSV with correlations)

### What We Need ❌
- **Recovery correlations CSV**: Must compute from checkpoints
- **Merged dataset**: prediction + recovery in one table

## Immediate Next Steps

### Step 1: Compute Recovery Correlations (REQUIRED FIRST)

This is the bottleneck. Without recovery data, we can't fix any tables.

```bash
cd /c/Users/steph/documents/deep-mirt/kt-gpcm

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vrec-env
export PYTHONPATH=src

# Run recovery computation (2-4 hours)
python scripts/compute_all_recovery.py \
  --outputs_dir outputs \
  --configs_dir configs \
  --output_csv outputs/recovery_correlations.csv
```

**Expected output**: `outputs/recovery_correlations.csv` with:
- experiment name
- r_alpha (discrimination recovery)
- r_beta_mean (average threshold recovery)
- r_beta_thresholds (per-threshold correlations)

**Time estimate**: 2-4 hours (144 experiments × 1-2 min each)

### Step 2: Merge Data and Generate Tables

```bash
cd /c/Users/steph/documents/deep-mirt/kt-gpcm
python scripts/merge_all_metrics.py
```

**Outputs**:
- `outputs/merged_all_metrics.csv` - Full dataset
- `outputs/table_rq1_main_comparison.csv` - RQ1 table data
- `outputs/table_rq4_scalability.csv` - RQ4 table data
- `outputs/table_ablation_loss.csv` - Loss ablation data
- `outputs/table_ablation_monotonic.csv` - Monotonic ablation data

### Step 3: Review Generated Tables

Before rewriting narratives, verify:
1. Do the numbers make sense?
2. Are there any missing experiments?
3. Do recovery correlations look reasonable (r > 0.7)?

### Step 4: Fix Section References

Found 5 instances to fix:

**Line 408**: `Section~\ref{sec:gpcm_head}`
→ "the GPCM head formulation below"

**Line 451**: `Section~\ref{sec:ablation_sensitivity}`
→ "the ablation study"

**Line 508**: `Section~\ref{sec:polytomous_irt}`
→ "the polytomous IRT background"

**Line 548**: `Section~\ref{sec:proxy_results}`
→ "the proxy-ordinality results"

**Line 564**: `Section~\ref{sec:ablation_sensitivity}`
→ "the encoding ablation"

### Step 5: Rewrite Results Sections

**Priority order**:
1. RQ1 (main comparison) - Sets the tone
2. RQ3 (parameter recovery) - Core contribution
3. RQ4 (scalability) - Practical impact
4. Ablations - Supporting evidence

**Writing template** (from your memory):
```
1. Lead with insight (what does this mean for measurement theory?)
2. Theoretical significance (why does it matter?)
3. Supporting evidence (numbers as proof, not narrative)
4. Practical implications (what can practitioners do?)
```

## Key Messages to Emphasize

From your agent memory, these are the critical insights:

1. **Implicit Psychometric Inference**
   - Neural networks recover discrimination WITHOUT supervision
   - DKVMN memory learns which items are discriminating by observing patterns
   - Opens path to "data-driven psychometrics"

2. **No Interpretability-Prediction Tradeoff**
   - DEEP-GPCM achieves BOTH simultaneously
   - Contradicts conventional wisdom
   - r_α = 0.800 at K=5 (exceeds Static GPCM's 0.759)

3. **Architectural Tradeoffs**
   - Threshold recovery gap (r_β ≈ 0.91 vs 0.97-0.98) is a design consequence
   - Indirect gradient path through memory/summary/extraction
   - Still psychometrically meaningful (r_β > 0.88)

4. **GPCM Head as Psychometric Lens**
   - Preserves latent dynamics while adding interpretability
   - DEEP-GPCM and DKVMN+Softmax produce identical trajectories
   - Can gain interpretability without altering representations

5. **Scalability with Interpretability**
   - SIE maintains recovery at Q=5000
   - Reduces parameters by 80%
   - Critical for operational testing programs

## Questions for You

1. **Should I start compute_all_recovery.py now?**
   - It will take 2-4 hours
   - Blocks all other work (need the data first)
   - Can run in background if you have other tasks

2. **Do you want to review generated tables before narrative rewriting?**
   - Safer to verify numbers first
   - Or trust the process and start writing?

3. **Any specific IJAIED papers for style reference?**
   - You mentioned reading IJAIED papers
   - Any particular ones you want me to emulate?

4. **Priority order for rewriting?**
   - I suggest: RQ1 → RQ3 → RQ4 → Ablations
   - Or different order based on your needs?

## Estimated Timeline

**If starting now**:
- Compute recovery: 2-4 hours (can run overnight)
- Merge data: 10 minutes
- Review tables: 30 minutes
- Fix section refs: 15 minutes
- Rewrite RQ1: 2 hours
- Rewrite RQ3: 3 hours
- Rewrite RQ4: 2 hours
- Rewrite ablations: 2 hours
- Polish: 2-3 hours

**Total**: ~15-20 hours of focused work

**Bottleneck**: Recovery computation (must run first)

## What to Do Right Now

**Option A: Start computation immediately**
```bash
cd /c/Users/steph/documents/deep-mirt/kt-gpcm
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vrec-env
export PYTHONPATH=src
nohup python scripts/compute_all_recovery.py > recovery.log 2>&1 &
```

**Option B: Review my scripts first**
- Check `compute_all_recovery.py` logic
- Check `merge_all_metrics.py` table generation
- Verify they'll produce what you need

**Option C: Start with section reference fixes**
- Quick win while thinking about bigger issues
- Doesn't require recovery data

Let me know which path you want to take.
