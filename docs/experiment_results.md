# Benchmark results (record only, no interpretation)

Bed: realistic (Q=200, N=2000, admin Uniform(40,80), uniform exposure), static theta unless _rw (drift); dense control = N800/Q60 all-items. 150 epochs, 5 data seeds x 5 folds = 25 fits/cell. Recovery: discrimination/slope = Spearman rho; difficulty/intercept + ability = Pearson r (rho also shown).

| cell | enc | dec | K | discrim a (rho) | difficulty/intercept (r) | ability th (r) | ability th (rho) | acc | QWK | AUC | NLL | macroAUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bench_dense_lstm_gpcm | lstm | gpcm | 4 | 0.786 | 0.940 | 0.970 | 0.972 | 0.487 | 0.608 | 0.845 | - | - |
| bench_dkvmn_2pl | dkvmn | binary | 2 | 0.821 | 0.967 | 0.621 | 0.632 | 0.715 | 0.430 | 0.796 | - | - |
| bench_dkvmn_gpcm | dkvmn | gpcm | 4 | 0.797 | 0.928 | 0.517 | 0.499 | 0.491 | 0.619 | 0.847 | - | - |
| bench_dkvmn_nrm | dkvmn | nrm | 4 | 0.814 | 0.905 | 0.115 | 0.108 | 0.485 | - | - | 1.173 | 0.738 |
| bench_lstm_2pl | lstm | binary | 2 | 0.783 | 0.965 | 0.568 | 0.562 | 0.712 | 0.424 | 0.792 | - | - |
| bench_lstm_gpcm | lstm | gpcm | 4 | 0.823 | 0.916 | 0.494 | 0.467 | 0.484 | 0.613 | 0.845 | - | - |
| bench_lstm_gpcm_rw | lstm | gpcm | 4 | 0.840 | 0.908 | - | - | 0.528 | 0.672 | 0.878 | - | - |
| bench_lstm_nrm | lstm | nrm | 4 | 0.802 | 0.893 | -0.108 | -0.105 | 0.476 | - | - | 1.193 | 0.728 |
| bench_transformer_2pl | transformer | binary | 2 | 0.421 | 0.890 | 0.781 | 0.787 | 0.693 | 0.385 | 0.765 | - | - |
| bench_transformer_gpcm | transformer | gpcm | 4 | 0.584 | 0.858 | 0.895 | 0.904 | 0.462 | 0.579 | 0.838 | - | - |
| bench_transformer_nrm | transformer | nrm | 4 | 0.666 | 0.789 | 0.339 | 0.372 | 0.435 | - | - | 1.252 | 0.688 |

## Per-cell full config + metrics

### bench_dense_lstm_gpcm
config: encoder=lstm decoder=gpcm K=4 regime=realistic N=800 Q=60 admin=(None,None) dense=True drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=ordinal_ce seeds=5x5 n_runs=25
  a_spearman: 0.7856 [0.7537, 0.8192] n=25
  a_pearson: 0.7118 [0.6649, 0.7614] n=25
  b_spearman: 0.9492 [0.9467, 0.9517] n=25
  b_pearson: 0.9399 [0.9377, 0.9419] n=25
  theta_spearman: 0.9717 [0.9693, 0.9739] n=25
  theta_pearson: 0.9696 [0.9674, 0.9716] n=25
  acc: 0.4872 [0.4816, 0.4926] n=25
  qwk: 0.6079 [0.5994, 0.6162] n=25
  auc: 0.8452 [0.8413, 0.8485] n=25

### bench_dkvmn_2pl
config: encoder=dkvmn decoder=binary K=2 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=- seeds=5x5 n_runs=25
  a_spearman: 0.8210 [0.7987, 0.8424] n=25
  a_pearson: 0.7543 [0.7209, 0.7864] n=25
  b_spearman: 0.9806 [0.9794, 0.9818] n=25
  b_pearson: 0.9674 [0.9652, 0.9694] n=25
  theta_spearman: 0.6317 [0.6077, 0.6575] n=25
  theta_pearson: 0.6213 [0.5959, 0.6484] n=25
  acc: 0.7153 [0.7119, 0.7190] n=25
  qwk: 0.4299 [0.4232, 0.4375] n=25
  auc: 0.7959 [0.7920, 0.8000] n=25

### bench_dkvmn_gpcm
config: encoder=dkvmn decoder=gpcm K=4 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=ordinal_ce seeds=5x5 n_runs=25
  a_spearman: 0.7968 [0.7640, 0.8265] n=25
  a_pearson: 0.7388 [0.7105, 0.7640] n=25
  b_spearman: 0.9439 [0.9384, 0.9493] n=25
  b_pearson: 0.9280 [0.9216, 0.9344] n=25
  theta_spearman: 0.4989 [0.4675, 0.5308] n=25
  theta_pearson: 0.5171 [0.4904, 0.5442] n=25
  acc: 0.4907 [0.4850, 0.4962] n=25
  qwk: 0.6191 [0.6131, 0.6248] n=25
  auc: 0.8472 [0.8439, 0.8503] n=25

### bench_dkvmn_nrm
config: encoder=dkvmn decoder=nrm K=4 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=- seeds=5x5 n_runs=25
  a_spearman: 0.8143 [0.7839, 0.8418] n=25
  a_pearson: 0.8275 [0.8081, 0.8463] n=25
  c_spearman: 0.9069 [0.8990, 0.9149] n=25
  c_pearson: 0.9052 [0.8978, 0.9133] n=25
  theta_spearman: 0.1076 [-0.1227, 0.3428] n=25
  theta_pearson: 0.1146 [-0.1115, 0.3463] n=25
  acc: 0.4852 [0.4813, 0.4890] n=25
  nll: 1.1729 [1.1677, 1.1780] n=25
  macro_auc: 0.7380 [0.7348, 0.7409] n=25

### bench_lstm_2pl
config: encoder=lstm decoder=binary K=2 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=- seeds=5x5 n_runs=25
  a_spearman: 0.7835 [0.7693, 0.7960] n=25
  a_pearson: 0.7098 [0.6898, 0.7304] n=25
  b_spearman: 0.9783 [0.9772, 0.9794] n=25
  b_pearson: 0.9653 [0.9639, 0.9666] n=25
  theta_spearman: 0.5621 [0.5284, 0.5955] n=25
  theta_pearson: 0.5684 [0.5384, 0.5978] n=25
  acc: 0.7122 [0.7091, 0.7158] n=25
  qwk: 0.4237 [0.4176, 0.4309] n=25
  auc: 0.7920 [0.7883, 0.7960] n=25

### bench_lstm_gpcm
config: encoder=lstm decoder=gpcm K=4 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=ordinal_ce seeds=5x5 n_runs=25
  a_spearman: 0.8229 [0.7922, 0.8521] n=25
  a_pearson: 0.7498 [0.7257, 0.7746] n=25
  b_spearman: 0.9332 [0.9274, 0.9389] n=25
  b_pearson: 0.9158 [0.9083, 0.9234] n=25
  theta_spearman: 0.4674 [0.4117, 0.5183] n=25
  theta_pearson: 0.4936 [0.4476, 0.5374] n=25
  acc: 0.4843 [0.4801, 0.4882] n=25
  qwk: 0.6126 [0.6058, 0.6193] n=25
  auc: 0.8451 [0.8417, 0.8483] n=25

### bench_lstm_gpcm_rw
config: encoder=lstm decoder=gpcm K=4 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=0.15 state_alpha=False item_key_dim=- epochs=150 gpcm_loss=ordinal_ce seeds=5x5 n_runs=25
  a_spearman: 0.8404 [0.8185, 0.8617] n=25
  a_pearson: 0.7710 [0.7517, 0.7892] n=25
  b_spearman: 0.9220 [0.9134, 0.9301] n=25
  b_pearson: 0.9083 [0.8984, 0.9175] n=25
  theta_netdrift_spearman: 0.2996 [0.2741, 0.3233] n=25
  theta_netdrift_pearson: 0.3347 [0.3105, 0.3580] n=25
  acc: 0.5279 [0.5244, 0.5310] n=25
  qwk: 0.6721 [0.6668, 0.6771] n=25
  auc: 0.8781 [0.8751, 0.8811] n=25

### bench_lstm_nrm
config: encoder=lstm decoder=nrm K=4 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=- seeds=5x5 n_runs=25
  a_spearman: 0.8022 [0.7792, 0.8231] n=25
  a_pearson: 0.8142 [0.7989, 0.8292] n=25
  c_spearman: 0.8947 [0.8896, 0.8998] n=25
  c_pearson: 0.8932 [0.8903, 0.8965] n=25
  theta_spearman: -0.1051 [-0.2996, 0.1063] n=25
  theta_pearson: -0.1085 [-0.3010, 0.1031] n=25
  acc: 0.4758 [0.4710, 0.4804] n=25
  nll: 1.1926 [1.1872, 1.1979] n=25
  macro_auc: 0.7282 [0.7251, 0.7314] n=25

### bench_transformer_2pl
config: encoder=transformer decoder=binary K=2 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=- seeds=5x5 n_runs=25
  a_spearman: 0.4215 [0.3906, 0.4525] n=25
  a_pearson: 0.3839 [0.3472, 0.4216] n=25
  b_spearman: 0.9233 [0.9079, 0.9372] n=25
  b_pearson: 0.8899 [0.8740, 0.9044] n=25
  theta_spearman: 0.7866 [0.7662, 0.8046] n=25
  theta_pearson: 0.7807 [0.7629, 0.7959] n=25
  acc: 0.6928 [0.6889, 0.6966] n=25
  qwk: 0.3847 [0.3770, 0.3925] n=25
  auc: 0.7651 [0.7603, 0.7698] n=25

### bench_transformer_gpcm
config: encoder=transformer decoder=gpcm K=4 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=ordinal_ce seeds=5x5 n_runs=25
  a_spearman: 0.5835 [0.5525, 0.6137] n=25
  a_pearson: 0.5229 [0.4921, 0.5553] n=25
  b_spearman: 0.8823 [0.8766, 0.8882] n=25
  b_pearson: 0.8578 [0.8510, 0.8648] n=25
  theta_spearman: 0.9038 [0.8959, 0.9106] n=25
  theta_pearson: 0.8949 [0.8872, 0.9020] n=25
  acc: 0.4616 [0.4568, 0.4667] n=25
  qwk: 0.5791 [0.5719, 0.5862] n=25
  auc: 0.8375 [0.8343, 0.8406] n=25

### bench_transformer_nrm
config: encoder=transformer decoder=nrm K=4 regime=realistic N=2000 Q=200 admin=(40,80) dense=False drift_sigma=- state_alpha=False item_key_dim=- epochs=150 gpcm_loss=- seeds=5x5 n_runs=25
  a_spearman: 0.6657 [0.6595, 0.6729] n=25
  a_pearson: 0.7270 [0.7228, 0.7313] n=25
  c_spearman: 0.7771 [0.7619, 0.7924] n=25
  c_pearson: 0.7891 [0.7761, 0.8026] n=25
  theta_spearman: 0.3720 [0.1494, 0.5562] n=25
  theta_pearson: 0.3392 [0.1291, 0.5160] n=25
  acc: 0.4351 [0.4302, 0.4396] n=25
  nll: 1.2516 [1.2469, 1.2566] n=25
  macro_auc: 0.6876 [0.6844, 0.6906] n=25
