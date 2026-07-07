# Real-data section — PARKED as placeholder (supervisor discussion pending)

**DECISION (2026-07-07).** The real-data study is left blank / placeholder in
the manuscript for now. The signal is mixed enough that it does not validate the
borrowed IRT interpretability reliably on real data, and it goes to the
supervisor before any write-up. Runs continue (data preserved, `--skip-done`
resumable) but nothing enters the `.tex`. This file is the brief for that
discussion.

## The honest result in one place

Separation wins cleanly on **one of eight** LSTM+transformer cells run so far
(LSTM EdNet-2PL). Cell by cell:

| encoder | cell | concordance (sep−shared) | δ (better head) | verdict |
|---|---|---|---|---|
| LSTM | 2PL EdNet | +0.051 (excl 0) | separate | separate wins |
| LSTM | 2PL KDD | +0.012 (incl 0) | ~tie | null |
| LSTM | GPCM TIMSS | +0.019 (incl 0) | separate | δ sep, concordance null (31 items) |
| LSTM | NRM EdNet | +0.025 (floor) | **shared** | reverses (option-level, documented) |
| transformer | 2PL EdNet | −0.040 | **shared** | **flips vs LSTM** |
| transformer | KDD / TIMSS / NRM | — | — | pending |

The head that calibrates better is **both dataset- and encoder-dependent**, and
the flip appears on δ (the credentialed diagnostic), not just concordance — so
it is robust, not noise.

## Psychometrician's verdict (why we cannot just write it up)

- "δ agrees with concordance" is **two correlated proxies agreeing** (both hinge
  on the model's own θ̂), not validation. A confidently-wrong head scores low δ.
- δ IS credentialed against true recovery on synthetic (§4.4, r=0.986) — a real
  strength the specialist underrated. Concordance is not.
- **Non-negotiable before publishing:** (1) re-fit the classical reference — a
  non-converged MML with ~5% negative-α items (EdNet-2PL) is not a silver
  standard; (2) compare both estimators on the SAME response definition
  (first-attempt) — neural reads a filtered sequential posterior, classical a
  static marginal fit; not apples-to-apples.
- **Missing tool to add (primary):** parameter INVARIANCE across disjoint person
  subsamples — cleanest truth-free discrimination-stability check. Then
  held-out-person cross-validated item params, and absolute item fit (S-X²,
  infit/outfit).
- Honest framing if pursued: real data **cannot certify calibration** without
  truth; it can rank stability / predictive consistency and validate the
  diagnostic's internal coherence. Encoder disagreement is a **finding**
  (architecture-dependent calibration), not noise.
- NRM near-floor concordance may be a sign/permutation-gauge artifact of option
  slopes (Spearman won't fix it) — check before calling it non-recovery.

---

# (Superseded) earlier draft — kept for prose only

**Status.** LSTM leg complete and reported (8 cells, 25/25 folds each, pre-reg
frozen). Transformer running, DKVMN queued. This draft is anchored on LSTM;
the multi-encoder rows fill in as they land. **The framing is yours to settle**
(see "Framing decision" at the end) — the result is heterogeneous, which is the
dilemma you named, so I have written it grant-then-qualify and honest, not as a
clean sweep.

---

## ⚠ UPDATE (transformer, partial) — READ THIS FIRST

The transformer's first cell **reverses LSTM on the flagship EdNet-2PL**, and
this changes the story materially:

| encoder | concord shared | concord separate | sep−shared | δ shared | δ separate | winner |
|---|---|---|---|---|---|---|
| LSTM | 0.685 | 0.736 | **+0.051** | 0.390 | 0.257 | **separate** |
| transformer | 0.722 | 0.682 | **−0.040** | 0.272 | 0.443 | **shared** |

So on real data the two encoders disagree about *which head is better on the
same dataset*. "Separation wins on EdNet-2PL" was an LSTM statement, not a
universal one. This pushes hard toward the safe framing (option 1) — maybe past
it.

**But the salvageable, honest contribution is the diagnostic, not the fix.**
Within each encoder, the truth-free δ and the classical concordance **agree on
the winner**: LSTM's δ and concordance both pick separate; the transformer's
both pick shared. So δ correctly identifies the better-calibrated head *whichever
head that is*. The deployable real-data claim is therefore not "use separate
heads" but "**you can tell which head to trust without ground truth, cheaply,
and it is not always the same head.**" That is honest, survives the reversal,
and is arguably a stronger fit for the paper's δ-diagnostic thread than a
separation win would have been.

**To verify before committing:** does δ agree with concordance about the winner
on *every* cell and both remaining encoders? I check this once transformer and
DKVMN finish. If yes, the section leads with the diagnostic. If δ and
concordance disagree anywhere the concordance is informative, even that retreats.

The LSTM-only draft below is now superseded on framing by this update; I keep it
for the prose and the table.

---

## The LSTM result (primary metric = concordance with classical MML)

| decoder / dataset | concord separate | concord shared | sep − shared (95% CI) | δ shared | δ separate | acc sep − shared | reading |
|---|---|---|---|---|---|---|---|
| **2PL / EdNet** | 0.736 | 0.685 | **+0.051 [+.037,+.065]** | 0.390 | 0.257 | +0.026 | separation wins on every axis |
| 2PL / KDD | 0.449 | 0.437 | +0.012 [−.012,+.033] | 0.577 | 0.590 | +0.003 | null |
| GPCM / TIMSS | 0.339 | 0.321 | +0.019 [−.035,+.072] | 0.809 | 0.676 | −0.002 | concordance null, δ favors separate |
| **NRM / EdNet** | 0.262 | 0.237 | +0.025 [+.002,+.055] | 0.380 | **0.782** | −0.062 | reverses (separate worse on δ + accuracy) |

Reliability (cautionary only): 2PL EdNet .755/.847, KDD .748/.802, GPCM TIMSS
.688/.644, NRM EdNet **.929 shared / .423 separate**. It does not track
concordance — see the demotion paragraph.

---

## Proposed prose (MA-GPCM register, honest)

On real data no ground truth exists, so the study measures agreement and
detection rather than recovery. Every model's item parameters are compared
against a classical marginal-maximum-likelihood calibration of the same
responses, the field's reference instrument, and the panel, the metrics, and
the seeds were fixed before any model ran. Four decoder-dataset pairs carry a
coercion-free source apiece: binary correctness on EdNet and on the KDD Cup
algebra log, genuine partial credit on the TIMSS 2019 constructed-response
items, and native option selection on EdNet for the nominal decoder.

The separation benefit that synthetic recovery shows without exception is, on
real data, real but not universal. It is clearest on the binary EdNet bank,
where the separate head concords with the classical calibration more closely
than the shared head, its truth-free discrepancy is the lower of the two, and
prediction is if anything slightly better, so the same head wins on agreement,
on detection, and on accuracy at once. On the KDD algebra log the two heads are
indistinguishable, both concording with the classical fit to the same modest
degree, which is the honest reading of a bank where the shared head was never
far from the calibration to begin with. On the small TIMSS ordinal bank the
concordance does not separate the heads, yet the truth-free discrepancy still
favors the separate one, a split we read as the concordance being the noisier
instrument on thirty-one items while the discrepancy retains its signal.

The nominal decoder reverses, and we report it plainly. On EdNet option
selection the separate head is the worse of the two on every axis that carries
weight, a higher truth-free discrepancy and lower prediction accuracy, with the
concordance itself near its floor for both heads. This is the same
synthetic-to-real reversal the option-level nominal head shows in
Section~[limitations]: the mitigation that recovers the slope on synthetic
banks does not transfer to real option data, where the classical nominal model
is itself estimable only on a well-covered core.

Split-half reliability, reported here only as the cautionary screen it is, does
not track any of this. On the nominal decoder the shared head is the *more*
reliable of the two by a wide margin while concording no better with the
calibration, exactly the reward-for-smoothing that a consistency measure blind
to systematic bias produces. Reliability is not the instrument that separates a
head that agrees with the calibration from one that does not.

---

## Framing decision (yours)

The honest one-line summary is: **separation helps clearly on the flagship
binary bank, is absent or ambiguous on the other ordered banks, and reverses on
the option-level nominal decoder.** Three ways to frame that, in ascending order
of how much they lean on the result:

1. **Minimal / safe.** Real data neither confirms nor refutes the synthetic
   benefit universally; it confirms it where the amortization gap is largest
   (EdNet-2PL) and shows the benefit is dataset-dependent. Pre-registration is
   the contribution here, not a clean win. (Recommended if transformer/DKVMN
   also come back heterogeneous.)
2. **Moderate.** The benefit holds on the binary bank and in the truth-free
   diagnostic on the ordinal bank; the option-level reversal is the known
   §4.5 caveat, now confirmed on real data. Frame as "holds where predicted."
3. **Strong.** Lean on EdNet-2PL as the deployment-relevant case and treat the
   rest as scope. (I would not; the KDD null undercuts it.)

I lean **1, hardening toward 2 if transformer and DKVMN corroborate EdNet-2PL.**
The KDD null and the TIMSS concordance-vs-δ split are real and should not be
smoothed over — they are what makes the pre-registration credible.

**Open caveat to resolve:** the classical EdNet-2PL reference flagged
non-convergence (800 EM cycles, one negative discrimination). The concordance
is rank-based and the direction is clear, but before this goes in the paper I
should re-fit it (or confirm the negative-α items are few and drop them). Noted,
not yet done.
