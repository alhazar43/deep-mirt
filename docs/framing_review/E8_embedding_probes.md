# E8: linear decodability of true item parameters from trained embeddings

Clean bed (drift dose 0), ridge with 5-fold item-CV, score = Spearman(held-out prediction, truth). Channels: thin value embedding (Q x 8, drives dynamics + SH readout); wide key (Q x 64, SK readout only).

| encoder | dec | arm | channel | decode log a | decode b | n units |
|---|---|---|---|---|---|---|
| dkvmn | 2pl | SH | value | 0.799 | 0.986 | 4 |
| lstm | 2pl | SH | value | 0.751 | 0.984 | 25 |
| lstm | 2pl | SK | key | 0.715 | 0.975 | 25 |
| lstm | 2pl | SK | value | 0.222 | 0.449 | 25 |
| lstm | gpcm | SH | value | 0.816 | 0.984 | 25 |
| lstm | gpcm | SK | key | 0.871 | 0.983 | 25 |
| lstm | gpcm | SK | value | 0.309 | 0.605 | 25 |
| transformer | 2pl | SH | value | 0.364 | 0.978 | 25 |
| transformer | 2pl | SK | key | 0.641 | 0.972 | 25 |
| transformer | 2pl | SK | value | 0.064 | 0.329 | 25 |
| transformer | gpcm | SH | value | 0.619 | 0.977 | 25 |
| transformer | gpcm | SK | key | 0.826 | 0.983 | 25 |
| transformer | gpcm | SK | value | 0.068 | 0.526 | 25 |
