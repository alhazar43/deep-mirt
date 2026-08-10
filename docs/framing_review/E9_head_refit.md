# E9 exhibit: frozen-table head refit (theory discriminator + free repair)

Fresh linear heads refit by likelihood on the FROZEN trained item table with ability clamped at the model's own causal estimates. No encoder retraining, no ground truth.

| cell | channel | n | a refit | b refit |
|---|---|---|---|---|
| lstm_2pl_drift0_shared | value | 25 | 0.779 | 0.988 |
| lstm_gpcm_drift0_shared | value | 25 | 0.839 | 0.969 |
| transformer_2pl_drift0_shared | value | 25 | 0.435 | 0.982 |
| transformer_gpcm_drift0_shared | value | 25 | 0.665 | 0.952 |
| dkvmn_2pl_drift0_shared | value | 25 | 0.813 | 0.989 |
| dkvmn_gpcm_drift0_shared | value | 25 | 0.879 | 0.974 |
| lstm_2pl_drift0_separate | key | 25 | 0.834 | 0.988 |

## Verdict: theory prediction 1 confirmed in every cell

Refit recovery matches the probe ceiling (within ~.02-.05) wherever the
table retains the information, and stays at the crowded level where it
does not:

| cell | SH recovery a/b | refit a/b | probe a/b | reading |
|---|---|---|---|---|
| lstm 2pl | .553/.723 | .779/.988 | .751/.984 | displacement repaired |
| lstm gpcm | .719/.826 | .839/.969 | .816/.984 | repaired |
| transformer 2pl | .373/.604 | **.435**/.982 | .364/.978 | b repaired; a ABSENT, not repairable post hoc (mechanism B) |
| transformer gpcm | .438/.768 | .665/.952 | .619/.977 | a to its decodable level; b repaired |
| dkvmn 2pl | .752/**.652** | .813/**.989** | .790/.985 | the DKVMN difficulty failure fully repaired WITHOUT retraining |
| dkvmn gpcm | .879/.849 | .879/.974 | .868/.983 | repaired |
| SK control (key) | .898/.957 | .834/.988 | .715/.975 | control: already near-optimal, refit does not move it up |

Consequences: the two-mechanism decomposition (displacement vs
crowding) is experimentally verified by its own pre-registered
discriminator, and the case for SK sharpens: only the trained-in
separated key repairs both mechanisms.

SCOPE (author ruling, strict): this refit exists ONLY as an offline
scientific instrument for testing the theory. Refit-style corrections
are BANNED as methods or deliverables in this research -- the model
operates under a real-time assumption, and any post-hoc refit violates
the premise it lives under. Nothing in this exhibit is a proposed
repair; the cure story is SK plus the audit, full stop.

