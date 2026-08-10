# E8: linear decodability of true item parameters from trained embeddings

Clean bed (drift dose 0), ridge with 5-fold item-CV, score = Spearman(held-out prediction, truth). Channels: thin value embedding (Q x 8, drives dynamics + SH readout); wide key (Q x 64, SK readout only).

| encoder | dec | arm | channel | decode log a | decode b | n units |
|---|---|---|---|---|---|---|
| dkvmn | 2pl | SH | value | 0.790 | 0.985 | 25 |
| dkvmn | 2pl | SK | key | 0.741 | 0.973 | 25 |
| dkvmn | 2pl | SK | value | 0.111 | 0.704 | 25 |
| dkvmn | gpcm | SH | value | 0.868 | 0.983 | 25 |
| dkvmn | gpcm | SK | key | 0.880 | 0.984 | 25 |
| dkvmn | gpcm | SK | value | 0.439 | 0.811 | 25 |
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

## Reading (final, all cells n=25)

1. **Difficulty information is never lost.** b decodable at .97-.99 from
   every SH value embedding, including the two cells whose trained heads
   read it worst (transformer .604, dkvmn .652 recovery). Those failures
   are head misalignment (theory: mechanism A, displacement), not
   representation loss.
2. **Slope shows both mechanisms.** lstm/dkvmn SH: a decodable .75-.87
   vs recovery .55-.88 (present, under-extracted -> mechanism A);
   transformer 2pl: decode .364 = recovery .373 (absent -> mechanism B,
   crowding). GPCM raises the transformer to .619 decodable (regime
   transition the theory attributes to the higher slope Fisher
   information at K=4).
3. **Channel specialization under SK.** The key decodes both families
   everywhere; the value embedding is purged of slope (a-decode
   .06-.44) while retaining a difficulty residue whose size ORDERS
   dkvmn (.70-.81) > lstm (.45-.61) > transformer (.33-.53) -- the
   dynamics' own demand for difficulty, measured in isolation, largest
   exactly where the architecture's per-item pathway is strongest.

Comparability caveat: the probe is OUT-OF-SAMPLE (5-fold item-CV
ridge), a conservative lower bound on information content; recovery
evaluates the trained head in-sample over the full bank. Direction and
orderings, not level differences of a few points, carry the claims.

