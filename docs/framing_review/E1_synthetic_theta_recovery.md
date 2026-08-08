# Synthetic ability (theta) recovery -- the person tier, all 72 cells

Seed-mean Spearman(theta_hat_lastvalid, theta_true) with sd over folds.
Produced 2026-07-17 from the tracked fold records (no new fits);
the column tab:mass never surfaced (framing review, lens A3 gap).

| encoder | decoder | design | N | theta rho (sd) | n folds |
|---|---|---|---|---|---|
| lstm | 2pl | SH | 500 | 0.789 (0.045) | 25 |
| lstm | 2pl | SH | 1000 | 0.807 (0.033) | 25 |
| lstm | 2pl | SH | 2000 | 0.808 (0.055) | 25 |
| lstm | 2pl | SH | 5000 | 0.845 (0.015) | 25 |
| lstm | 2pl | SK | 500 | 0.887 (0.035) | 25 |
| lstm | 2pl | SK | 1000 | 0.910 (0.016) | 25 |
| lstm | 2pl | SK | 2000 | 0.923 (0.009) | 25 |
| lstm | 2pl | SK | 5000 | 0.924 (0.013) | 25 |
| lstm | gpcm | SH | 500 | 0.904 (0.032) | 25 |
| lstm | gpcm | SH | 1000 | 0.911 (0.025) | 25 |
| lstm | gpcm | SH | 2000 | 0.928 (0.013) | 25 |
| lstm | gpcm | SH | 5000 | 0.954 (0.009) | 25 |
| lstm | gpcm | SK | 500 | 0.953 (0.008) | 25 |
| lstm | gpcm | SK | 1000 | 0.966 (0.007) | 25 |
| lstm | gpcm | SK | 2000 | 0.970 (0.004) | 25 |
| lstm | gpcm | SK | 5000 | 0.976 (0.004) | 25 |
| lstm | nrm | SH | 500 | 0.746 (0.071) | 25 |
| lstm | nrm | SH | 1000 | 0.772 (0.048) | 25 |
| lstm | nrm | SH | 2000 | 0.792 (0.046) | 25 |
| lstm | nrm | SH | 5000 | 0.851 (0.038) | 25 |
| lstm | nrm | SK | 500 | 0.890 (0.017) | 25 |
| lstm | nrm | SK | 1000 | 0.901 (0.024) | 25 |
| lstm | nrm | SK | 2000 | 0.919 (0.011) | 25 |
| lstm | nrm | SK | 5000 | 0.930 (0.013) | 25 |
| transformer | 2pl | SH | 500 | 0.738 (0.077) | 25 |
| transformer | 2pl | SH | 1000 | 0.772 (0.069) | 25 |
| transformer | 2pl | SH | 2000 | 0.786 (0.072) | 25 |
| transformer | 2pl | SH | 5000 | 0.819 (0.069) | 25 |
| transformer | 2pl | SK | 500 | 0.899 (0.021) | 25 |
| transformer | 2pl | SK | 1000 | 0.898 (0.026) | 25 |
| transformer | 2pl | SK | 2000 | 0.920 (0.015) | 25 |
| transformer | 2pl | SK | 5000 | 0.932 (0.010) | 25 |
| transformer | gpcm | SH | 500 | 0.885 (0.049) | 25 |
| transformer | gpcm | SH | 1000 | 0.895 (0.037) | 25 |
| transformer | gpcm | SH | 2000 | 0.877 (0.043) | 25 |
| transformer | gpcm | SH | 5000 | 0.919 (0.029) | 25 |
| transformer | gpcm | SK | 500 | 0.954 (0.013) | 25 |
| transformer | gpcm | SK | 1000 | 0.963 (0.013) | 25 |
| transformer | gpcm | SK | 2000 | 0.971 (0.005) | 25 |
| transformer | gpcm | SK | 5000 | 0.978 (0.004) | 25 |
| transformer | nrm | SH | 500 | 0.635 (0.124) | 25 |
| transformer | nrm | SH | 1000 | 0.620 (0.094) | 25 |
| transformer | nrm | SH | 2000 | 0.608 (0.085) | 25 |
| transformer | nrm | SH | 5000 | 0.696 (0.075) | 25 |
| transformer | nrm | SK | 500 | 0.786 (0.101) | 25 |
| transformer | nrm | SK | 1000 | 0.816 (0.046) | 25 |
| transformer | nrm | SK | 2000 | 0.864 (0.031) | 25 |
| transformer | nrm | SK | 5000 | 0.895 (0.016) | 25 |
| dkvmn | 2pl | SH | 500 | 0.785 (0.093) | 25 |
| dkvmn | 2pl | SH | 1000 | 0.802 (0.082) | 25 |
| dkvmn | 2pl | SH | 2000 | 0.841 (0.045) | 25 |
| dkvmn | 2pl | SH | 5000 | 0.889 (0.033) | 25 |
| dkvmn | 2pl | SK | 500 | 0.904 (0.018) | 25 |
| dkvmn | 2pl | SK | 1000 | 0.907 (0.016) | 25 |
| dkvmn | 2pl | SK | 2000 | 0.932 (0.015) | 25 |
| dkvmn | 2pl | SK | 5000 | 0.939 (0.007) | 25 |
| dkvmn | gpcm | SH | 500 | 0.902 (0.043) | 25 |
| dkvmn | gpcm | SH | 1000 | 0.944 (0.026) | 25 |
| dkvmn | gpcm | SH | 2000 | 0.955 (0.009) | 25 |
| dkvmn | gpcm | SH | 5000 | 0.964 (0.011) | 25 |
| dkvmn | gpcm | SK | 500 | 0.960 (0.009) | 25 |
| dkvmn | gpcm | SK | 1000 | 0.971 (0.008) | 25 |
| dkvmn | gpcm | SK | 2000 | 0.973 (0.007) | 25 |
| dkvmn | gpcm | SK | 5000 | 0.981 (0.004) | 25 |
| dkvmn | nrm | SH | 500 | 0.754 (0.096) | 25 |
| dkvmn | nrm | SH | 1000 | 0.770 (0.049) | 25 |
| dkvmn | nrm | SH | 2000 | 0.813 (0.049) | 25 |
| dkvmn | nrm | SH | 5000 | 0.869 (0.034) | 25 |
| dkvmn | nrm | SK | 500 | 0.879 (0.031) | 25 |
| dkvmn | nrm | SK | 1000 | 0.877 (0.030) | 25 |
| dkvmn | nrm | SK | 2000 | 0.906 (0.026) | 25 |
| dkvmn | nrm | SK | 5000 | 0.934 (0.017) | 25 |

## N=2000 cohort (tab:mass companion)

| encoder | decoder | SH theta | SK theta | delta |
|---|---|---|---|---|
| lstm | 2pl | 0.808 | 0.923 | +0.116 |
| lstm | gpcm | 0.928 | 0.970 | +0.042 |
| lstm | nrm | 0.792 | 0.919 | +0.127 |
| transformer | 2pl | 0.786 | 0.920 | +0.134 |
| transformer | gpcm | 0.877 | 0.971 | +0.095 |
| transformer | nrm | 0.608 | 0.864 | +0.255 |
| dkvmn | 2pl | 0.841 | 0.932 | +0.091 |
| dkvmn | gpcm | 0.955 | 0.973 | +0.017 |
| dkvmn | nrm | 0.813 | 0.906 | +0.094 |
