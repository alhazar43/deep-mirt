# Supervisor brief: what this paper found, and how to sell it (2026-08-10)

One page of claims, each with its number. Everything cited is committed
in the repo; table sources at the end.

## The claim

Prediction-trained knowledge-tracing models learn item parameters only
as a byproduct of what their sequence dynamics need. Where the readout
fails depends on the architecture and is unpredictable in practice; a
one-embedding separated item key repairs it uniformly, at zero
prediction cost; and a truth-free audit detects the failure without
ground truth. Unpredictable disease, uniform cure, and a meter that
says when you need it.

## Three statements, each carried by one table

**1. The untreated failure is architecture-dependent (so per-case
claims are fragile; ours is not).** Same data, same decoder, same
training; only the encoder changes. Shared-readout (SH) recovery of
discrimination a and difficulty b, vs the separated key (SK):

| encoder | acc SH/SK | a: SH -> SK | b: SH -> SK | laggard |
|---|---|---|---|---|
| lstm (2PL) | .712/.714 | .553 -> .898 | .723 -> .957 | a |
| transformer (2PL) | .708/.713 | .373 -> .806 | .604 -> .955 | both |
| dkvmn (2PL) | .716/.715 | .752 -> .914 | .652 -> .950 | **b** |

The laggard FLIPS with architecture (DKVMN fails on difficulty, the
others on discrimination), while accuracy is identical everywhere: the
failure is invisible to prediction metrics. We do not claim which
family fails where (that would be fragile to new datasets and
architectures, and we have not exhausted either); we claim the failure
location is unpredictable, and that is precisely why the fix and the
audit matter.

**2. The cure is uniform, cheap, and robust.** SK = one extra
embedding table read only by the parameter heads; no inference cost,
no tuning, drop-in for any encoder. It lifts every family on every
encoder to the same plateau (a ~.90-.95, b ~.95-.97 above). Under a
seven-violation misspecification battery (2500 fits: ability drift,
local dependence, threshold disorder, exposure imbalance, DIF,
response style, noisy thresholds; doses; paired seeds): the SK
advantage is positive in 49/50 cells (paired t 2.3-18.4), never
reverses, and GROWS under local dependence and threshold disorder;
accuracy differences stay within .011 everywhere. Boundary honestly
stated: under extreme exposure starvation both arms fail together.

**3. The audit works without truth.** The refit-discrepancy check
(refit item slopes with ability clamped at the model's own estimates;
measure disagreement with the readout) tracks true corruption at
Spearman .93 across the battery and rises with dose within every
violation family. On real data it fires exactly where it should: the
TIMSS calibration the paper previously presented as reassuring. Bonus
real-data result: read WITHOUT the eval-time sort, the neural
thresholds reproduce the classical calibration's category-order
structure at Spearman .98 (including all 12 classically non-modal
items) -- the model had learned true structure the export path was
erasing.

## The mechanism (why this is a finding, not a trick)

The thin item embedding is a contested channel: it serves the sequence
dynamics AND the parameter readout, and training is prediction-only.
Linear probes of the trained embeddings (fresh ridge, item-CV) against
ground truth separate two failure components:

| finding | evidence |
|---|---|
| Difficulty information is never lost -- the trained HEAD is misaligned | b decodable at >= .97 from every SH embedding, even where the head recovers .60-.65 |
| Slope information is partially crowded out, worst under global attention | transformer SH: decode .364 ~= recovery .373 (truly absent); lstm: decode .75 vs recovery .55 (present, under-extracted) |
| Separation makes the channels specialize | under SK the key decodes both families (.64-.87 / .97+) while the value embedding is PURGED of them (a .06-.31); the residual b trace (.33-.61) shows difficulty is what dynamics still demand |

DKVMN is the mechanism's fingerprint, not an exception: it already
separates addressing from state INTERNALLY (static key memory, dynamic
value memory), which changes where the contention lands -- and the
failure moves exactly there (difficulty), while SK (which separates
the READOUT, an orthogonal cut) still helps it at full size. The two
separations resolve the apparent contradiction: DKVMN has one of them
natively, benefits from the other.

[Theory section: gradient-routing derivation in
docs/framing_review/theory_contention.md -- being finalized; the
stationary-point account of contested-channel training, the two-route
gradient decomposition, and the falsifiable predictions it makes,
two of which the probe table above already confirms.]

## The storyline that answers the criticism

The reviewer's "glued math and datasets" dissolves under two moves,
both already documented: (a) the three decoders are one
divide-by-total family at three doses of slope structure, and one law
-- the shared path preserves location-family information and corrupts
slope-family information -- is tested at each dose
(docs/framing_review/format_unification.md); (b) the paper stops
presenting agreement as reassurance and RUNS ITS OWN AUDIT on its own
case studies: EdNet is the rung where the repair is verifiable against
MML; TIMSS is the rung where truth is absent and the audit is what a
practitioner has.

## What exists vs what remains

In hand, committed: the full synthetic grid with paired statistics
(strengthens the headline: t 8-47), the battery, the raw-threshold
TIMSS rerun, completed DKVMN real cells, capacity controls (the 2.6x
parameter objection: effect survives width-matching), the probe
tables, the CAT decision-cost results (shared readout stops testing at
8 items certifying SE .29 when truth is .69). Remaining: the theory
writeup lands today; the manuscript rewrite is the author's, from
docs/results_critique.md (Parts I-VII) + exhibits E1-E8.

## Discussion points for the meeting

1. Lead methodological (the contention mechanism + repair) or
   practical (the audit + invoice for deployed KT)? The evidence
   carries either; the mechanism lead is more novel, the audit lead
   more useful.
2. How much theory in the main text vs appendix (derivation vs the
   probe table, which is self-explanatory).
3. Whether to present the DKVMN flip as the headline demonstration
   (my recommendation: yes -- it converts the biggest apparent
   weakness into the mechanism's cleanest evidence).

Sources: recovery matrix kt-irt/results/p2_toggle (aggregate
p2_mass_table); battery kt-irt/results/p2_misspec/battery_report.md;
probes docs/framing_review/E8_embedding_probes.md; TIMSS
kt-irt/results/p2_realstudy_rawbeta + E6; audit-on-real
kt-irt/results/p2_realstudy tables; defect ledger and menu
docs/results_critique.md.
