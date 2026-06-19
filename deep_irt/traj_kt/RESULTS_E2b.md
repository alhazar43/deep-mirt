# E2b Results: ASSISTments 2009-2010 Repeated-Practice

## Dataset

- **Source**: C:\Users\steph\documents\VocRecSys\deep2pl\data\assist2009_updated\assist2009_updated.csv
- **Format**: DKVMN/DKT triplet (skill-builder variant, ASSISTments 2009-2010 updated)
- **Students total**: 4151
- **Total interactions**: 325,637
- **Unique skills**: 110
- **Fields**: seq_len, skill_ids (1..110), responses (0/1)
- **Seq len**: mean 78.4, median 23.0, p25/p75 9/67
- **Mean skill repetitions per student**: 7.66 (key repeated-practice property)

## Cohort (after filtering: >=50 interactions + >=5 reps on >=1 skill)

- **Students**: 1262
- **Interactions**: 277,449
- **Skill vocabulary size**: 110
- **Unique skills**: 107

## Training

- **Model**: DeepIRTModel(binary, lstm, decouple=True), device=cuda
- **Split**: 80/20 student split (1009 / 253)
- **Epochs run**: 20 (early stopping patience=4)
- **Final train loss**: 0.4681
- **Best val loss**: 0.4679
- **Peak VRAM**: 1043.6 MB
- **Wall time total**: 85.4s

## Rate Recovery

- **r_hat finite**: 1262/1262 students
- **Mean r_hat**: 0.3444, std=0.3764

## Validation

### (a) Predictive validity
rho=-0.079 [-0.134, -0.023] (n=1262)
- Negative control (shuffled r_hat): rho=-0.018 [-0.077, 0.040]

### (b) AFM concurrent validity (PRIMARY)
rho=0.138 [0.080, 0.194] (n=1260)

### (c) Split-half reliability
rho=0.352 [0.297, 0.407] (n=1262)

### (d) Convergent validity (aligned vs responsive)
rho=0.580 [0.522, 0.639] (n=1262)

## Verdict

**Weak and confounded, not a clean human-front win.** Read the two external
tests against each other. The AFM concurrent correlation is small but
nonzero (0.138, CI excludes zero), and it does clear the EdNet bar where
the same CI spanned zero, consistent with repeated-practice structure
carrying some signal. But two facts pull it back. First, AFM circularity,
the model's only item key here is skill_id, the exact granularity the AFM
slope is fit on, so the encoder's theta already conditions on the
skill-recurrence structure AFM measures and the 0.138 is partly
self-referential (different functionals, a per-skill logistic slope versus
an exponential fit on the theta trajectory, so not an identity, but
inflated). Second, the cleaner external test, predictive validity, is
mildly NEGATIVE here (-0.079, CI excludes zero) and was also negative on
EdNet (-0.036), the recovered first-half rate slightly anti-predicts the
second-half gain, most likely regression to the mean (fast early apparent
improvers plateau). Internal consistency is fine (split-half 0.35,
convergent 0.58) but that only says the fitted rate is a stable feature,
not that it tracks real learning.

Honest bottom line. Real human learning-rate recovery is NOT established.
The one positive (AFM) is small and confounded by skill-id-as-item, and
the cleaner predictive-validity test is null to negative on both real
datasets. The clean positives in the program remain the synthetic E0 and
the machine E1b. The decisive next test is a source that retains
problem_id (KDD Cup 2010 Algebra, via DataShop) to break the skill-id
circularity, plus a treatment of the regression-to-the-mean confound in
the predictive test. The run script is parameterized so swapping the item
key is a one-line change.

**Update (positive control).** A later positive control on synthetic data
with known rates (`deep_irt/traj_synth/RESULTS_poscontrol.md`) showed the
predictive-validity metric is structurally ill-posed, recovery there is
0.46 yet the metric is -0.26 and even the TRUE rate scores -0.38, because
fast learners plateau inside the window. So the -0.079 here is a metric
artifact and is DISCARDED, not evidence of failure. The standing real-data
signal is therefore only the small, skill-id-confounded AFM 0.138, the
human front is unestablished, not negative.

## Contrast with EdNet-KT1 null

EdNet-KT1 was single-pass (each student sees each item at most once), so no
opportunity-count curves exist and AFM concurrent validity cannot be computed.
ASSISTments 2009-2010 is the canonical AFM/Koedinger repeated-practice setting:
mean 7.7 skill repetitions per student enables the AFM concurrent test.
