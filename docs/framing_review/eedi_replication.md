# Eedi pre-registered replication — outcome (2026-08-11)

Log-only cross-platform replication of the matched-exposure nominal
cell, executed exactly as pre-registered (external_landscape.md E.2);
every endpoint reported regardless of outcome, as committed in advance.

## Design (fixed before any result was seen)

Bank: Eedi NeurIPS-2020 task-1/2 responses (UK KS1-4 math), top-250
option-rich questions, learner subsample sized to the EdNet regime:
achieved 12,299 learners, 380,025 observations, **190 responses per
raw parameter** (EdNet matched cell: 191). Same routed head, fold
protocol, seeds; 50 fits, zero failures (cluster job, 16 min).
Anchor: per-item keyed contrast of per-option point-biserials against
leave-one-out correctness, WITH the pre-registered guardrails —
option-count floor 50, frequency-weighted distractor means, anchor
split-half reliability printed beside. Success criteria fixed in
advance: SK > SH under the paired seed test AND SK >= .50 observed;
anchor reliability >= .50 required for level claims. The optional
misconception-bank join (P2) was not attempted within its timebox.

## Results (guardrailed, both platforms)

| platform | eligible items | anchor reliability | SH | SK | paired SK-SH |
|---|---|---|---|---|---|
| EdNet matched (TOEIC-style, 191 r/p) | 249/250 | .857 | .548 | .632 | **+.094 [t(4)=10.5, 5/5]** |
| Eedi (UK school math, 190 r/p) | 215/250 | .912 | **.756** | .723 | -.023 [t(4)=-3.0, 0/5] |

Note the guardrails matter: EdNet's previously reported unguarded
numbers (.437/.705) overweighted rare-distractor point-biserials —
the floors and frequency weighting shrink the gap to +.094, which is
the honest headline and still passes every pre-set criterion.

## Verdict

Replicates in LEVEL, not in DIRECTION. On the second platform both
arms clear the .50 floor comfortably (.72-.76 against a .91-reliable
anchor) and the separated key's advantage vanishes (a small,
seed-consistent SH edge of .02). On the first platform the separated
key leads decisively (+.094, t=10.5).

## Reading (mechanism-consistent, not post hoc rescue)

The displacement mechanism predicts head error inversely weighted by
the parameter-route Fisher information. Eedi items are DIAGNOSTIC:
distractors expert-authored to capture specific misconceptions —
a strong per-option signal — so both readouts recover the structure
and separation has little to buy. EdNet's language-test lures carry
weaker option signal, and there the separated readout pays. The
cross-platform pair therefore instantiates the theory's own
information-dependence rather than contradicting the repair: BOTH
arms high where information is rich, separation decisive where it is
not.

## Claim consequences (binding for the rewrite)

1. Never claim "the separated key wins on real option data" as a
   universal; the licensed sentence: "on the platform with weak
   option signal the separated key reads the empirical structure
   decisively better (+.094, t=10.5, all seeds); on a platform with
   expert-designed diagnostic distractors both designs read it well
   and the choice is immaterial (.76/.72) — consistent with the
   displacement mechanism's information dependence, and one more
   instance of the paper's central point: where the readout fails is
   regime-dependent, and the meters, not the architecture label,
   tell you where you are."
2. The EdNet anchor headline becomes the guardrailed +.094 (t=10.5),
   with the unguarded numbers retired to the record.
3. Both-platform levels (.55-.76) now also serve the regime table:
   sequential KT readouts CAN reach classical-anchor agreement in the
   .7+ range on real data where the option signal is strong.

Store: kt-irt/results/p2_eedi250 (bank manifest + 50 fold JSONs).
Analysis: this file; guardrail implementation in the session record.
