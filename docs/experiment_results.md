# Experiment record (no interpretation). NRM ability sign-corrected (mean of |per-fold r|).

## Benchmark (fixed setup, realistic bed)
| cell | enc | dec | discrim/slope rho | diff/intcpt r | ability r | ability rho | acc | QWK | AUC | NLL | mAUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bench_dense_lstm_gpcm | lstm | gpcm | 0.786 | 0.940 | 0.970 | 0.972 | 0.487 | 0.608 | 0.845 | - | - |
| bench_dkvmn_2pl | dkvmn | binary | 0.821 | 0.967 | 0.621 | 0.632 | 0.715 | 0.430 | 0.796 | - | - |
| bench_dkvmn_gpcm | dkvmn | gpcm | 0.797 | 0.928 | 0.517 | 0.499 | 0.491 | 0.619 | 0.847 | - | - |
| bench_dkvmn_nrm | dkvmn | nrm | 0.814 | 0.905 | 0.585 | 0.593 | 0.485 | - | - | 1.173 | 0.738 |
| bench_lstm_2pl | lstm | binary | 0.783 | 0.965 | 0.568 | 0.562 | 0.712 | 0.424 | 0.792 | - | - |
| bench_lstm_gpcm | lstm | gpcm | 0.823 | 0.916 | 0.494 | 0.467 | 0.484 | 0.613 | 0.845 | - | - |
| bench_lstm_gpcm_rw | lstm | gpcm | 0.840 | 0.908 | - | - | 0.528 | 0.672 | 0.878 | - | - |
| bench_lstm_nrm | lstm | nrm | 0.802 | 0.893 | 0.534 | 0.535 | 0.476 | - | - | 1.193 | 0.728 |
| bench_transformer_2pl | transformer | binary | 0.421 | 0.890 | 0.781 | 0.787 | 0.693 | 0.385 | 0.765 | - | - |
| bench_transformer_gpcm | transformer | gpcm | 0.584 | 0.858 | 0.895 | 0.904 | 0.462 | 0.579 | 0.838 | - | - |
| bench_transformer_nrm | transformer | nrm | 0.666 | 0.789 | 0.614 | 0.655 | 0.435 | - | - | 1.252 | 0.688 |

## Toggles 2PL/GPCM (lstm): shared/decoupled x static/dynamic
| cell | dec | decouple | dyn | discrim rho | diff r | ability r | acc | QWK | AUC |
|---|---|---|---|---|---|---|---|---|---|
| toggle_2pl_decoupled_dynamic | binary | decoupled | dyn | 0.898 | 0.973 | 0.563 | 0.713 | 0.425 | 0.794 |
| toggle_2pl_decoupled_static | binary | decoupled | stat | 0.903 | 0.974 | 0.536 | 0.713 | 0.425 | 0.794 |
| toggle_2pl_shared_dynamic | binary | shared | dyn | 0.713 | 0.961 | 0.523 | 0.712 | 0.424 | 0.791 |
| toggle_2pl_shared_static | binary | shared | stat | 0.783 | 0.965 | 0.568 | 0.712 | 0.424 | 0.792 |
| toggle_gpcm_decoupled_dynamic | gpcm | decoupled | dyn | 0.965 | 0.972 | 0.515 | 0.502 | 0.611 | 0.848 |
| toggle_gpcm_decoupled_static | gpcm | decoupled | stat | 0.963 | 0.971 | 0.573 | 0.503 | 0.611 | 0.848 |
| toggle_gpcm_shared_dynamic | gpcm | shared | dyn | 0.842 | 0.929 | 0.459 | 0.488 | 0.618 | 0.846 |
| toggle_gpcm_shared_static | gpcm | shared | stat | 0.823 | 0.916 | 0.494 | 0.484 | 0.613 | 0.845 |

## Toggles NRM (lstm): 10 couplings x static/dynamic
| cell | coupling | dyn | slope rho | intcpt r | ability r | acc | mAUC |
|---|---|---|---|---|---|---|---|
| toggle_nrm_a_only_dec_static | a_only_dec | stat | 0.499 | 0.308 | 0.302 | 0.459 | 0.717 |
| toggle_nrm_a_only_dec_dynamic | a_only_dec | dyn | 0.257 | 0.408 | 0.301 | 0.476 | 0.731 |
| toggle_nrm_all_decoupled_static | all_decoupled | stat | 0.816 | 0.847 | 0.408 | 0.483 | 0.740 |
| toggle_nrm_all_decoupled_dynamic | all_decoupled | dyn | 0.687 | 0.832 | 0.489 | 0.494 | 0.751 |
| toggle_nrm_c_only_dec_static | c_only_dec | stat | 0.947 | 0.965 | 0.609 | 0.496 | 0.752 |
| toggle_nrm_c_only_dec_dynamic | c_only_dec | dyn | 0.596 | 0.936 | 0.503 | 0.492 | 0.747 |
| toggle_nrm_decoupled_static | decoupled | stat | 0.960 | 0.969 | 0.478 | 0.495 | 0.751 |
| toggle_nrm_decoupled_dynamic | decoupled | dyn | 0.716 | 0.941 | 0.460 | 0.494 | 0.750 |
| toggle_nrm_shared_static | shared | stat | 0.758 | 0.881 | 0.470 | 0.469 | 0.721 |
| toggle_nrm_shared_dynamic | shared | dyn | 0.500 | 0.913 | 0.515 | 0.484 | 0.736 |
