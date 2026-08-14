# Resubmission campaign: results digest (2026-08-14, overnight)

All five tasks of docs/resubmission_experiment_plan.md executed to
completion in one night. 925 training fits + 6 mirt reference fits, zero
failures anywhere. Every number regenerates from a committed module; the
generating store is named per block. Statistics: fold-mean within seed,
paired condition differences within seed, t over 5 independent seeds
(df=4), positive-seed counts. Historical stores untouched.

## T1. Gradient-isolated SH (the causal mechanism test) -- CONFIRMED

Store results/p2_gradiso (675 fits; report.md by _p2_gradiso_report).
Intervention verified bit-exact before any training (30 autograd tests:
forward torch.equal, identical parameters, gradient change confined to
the item table, leak-free on all three encoders).

Recovery under SH -> isolated -> SK, with the isolated-SH contrast = the
pure gradient effect (all 27 cells positive; representative cells):

- lstm-2pl a: .553 -> .698 -> .898 (iso-SH +.145 [t=6.0, 5/5])
- transformer-gpcm a: .438 -> .538 -> .900 (+.100 [4.0]; SK-iso +.363
  = the capacity/parameterization share, largest under attention,
  matching crowding)
- dkvmn-2pl b: .652 -> .929 -> .950 (iso-SH +.277 [6.3]; SK-iso +.021
  [1.9] NOT significant: DKVMN's failure is essentially pure gradient
  contention)
- dkvmn-nrm a: .881 -> .918 -> .960; intercepts .737 -> .910 -> .914
  (the nominal cells were RERUN with the manuscript's canonical routed
  head after GPT's config audit; the first pass had used the retired
  toggle sweep's plain channel decoder, a different model. The rerun's
  SH and SK columns reproduce tab:mass EXACTLY on all three encoders
  and both parameter families, so all three conditions now sit on
  identical footing. Store results/p2_gradiso_arm1r, 225 fits.)
- Isolation is positive in all 5 seeds of every cell for BOTH item
  families; the single exception in the whole grid is ability recovery
  in the nominal transformer cell (-.004, 1/5). Accuracy is flat.

NLL, archived per fit for every cell (the P1 gap): the isolated and
separated paths have HELD-OUT NLL <= SH essentially everywhere (e.g.
dkvmn-2pl iso-SH -.001 [t=-8.8, 0/5 positive]; dkvmn-nrm -.012
[-10.3]). This CONTRADICTS the direction of the retired P1 exhibit
(which had no committed generator, store, or weights and is hereby
superseded): prediction pays NOTHING for the shared path's entanglement
-- the corruption is not even purchased. The purchased-corruption
framing is retired; the honest sentence is stronger: separation costs
nothing on any prediction metric and repairs measurement.

## T2. Key width = capacity axis, not definition -- COMPLETE

Store results/p2_narrowkey_fill (100 fits; section B of the report).
key16 sits strictly between shared and key64 on every new cell:
transformer-2pl .373/.452/.806, transformer-gpcm .438/.637/.900,
dkvmn-2pl .752/.874/.914, dkvmn-gpcm .879/.928/.952 (shared/k16/k64 a),
completing lstm's existing key-16 evidence across all three encoders.
Combined with T1: routing explains the isolated-SH step, width explains
the remainder; 64 is an implementation choice.

## T3. Matched EdNet-250 2PL -- COMPLETE, with a ceiling result

Frozen bank results/p2_nrm250/bank.npz (key hard-asserted against the
shipped manifest; 6 correspondence tests incl. byte-identical fold
splits vs the shipped NRM traj). Store results/p2_ednet250_2pl (150
fits). MML 2PL reference on the identical first-attempt histories:
250/250 items, split-half ceilings a ~.883 / b ~.964.

Agreement with MML (across-fit mean readout): lstm a .878 SH -> .901 SK
(AT the reference's own ceiling), dkvmn .889 -> .893, transformer .904
-> .903; b ~.918-.920 everywhere. On a well-observed bank the
sequential readouts reach the classical estimator's self-agreement
limit -- the strongest real-data level the program has produced.

## T4. Cross-format coherence on identical histories -- NEW HEADLINE

Module _p2_crossformat; store results/p2_crossformat. Spearman between
the 2PL discrimination and the NRM keyed contrast, same bank, same
histories, per path:

- lstm: SH .414 vs SK .782 (paired +.368 [t=62.9, 5/5])
- dkvmn: SH .480 vs SK .801 (+.320 [35.8])
- transformer: SH .238 vs SK .387 (+.148 [10.5])
- Control: the two 2PL readouts agree .95-.98 across paths, localizing
  the incoherence to the shared path's option slopes.

Under the separated key, two response formulations of the same behavior
tell one story about the items; under the shared path they tell two.
(4B hardness statistic ~.98 everywhere = the pre-stated confirmatory
ceiling; carried with its caveat.)

## T5. Sorted-vs-raw step recovery -- CONTRAST SURVIVES RAW

Report results/p2_beta_sortraw + the raw columns of the gradiso report.
Verdict on the historical headline: sorted-vs-sorted (order-blind); raw
not recoverable for old stores (documented). On the new grid: raw
levels sit below sorted where thresholds disorder (transformer-gpcm SH
sorted .768 vs raw .659) and the SK-SH contrast survives and GROWS in
raw form (+.269 raw vs +.179 sorted). 2PL raw == sorted (K=2).

## T6. Training-duration control (added 2026-08-14) -- SH IS NOT SLOW

Store results/p2_duration (225 fits, zero failures; report by
_p2_duration_report). Every SH cell trained to 4x the standard budget
with early stopping DISABLED, recovery scored at nested 1x/2x/4x marks
of the same run (identical seed, init, batch order).

NO cell reaches SK by training longer. Eight of nine degrade (lstm-2pl
-.050 [t=-8.9, 5/5], dkvmn-2pl -.074 [-10.7], transformer-nrm -.088
[-6.9]); the ninth (dkvmn-nrm) improves +.031 and still lands .922
against SK's .960. QUALIFICATION, stated in print: held-out accuracy
falls and NLL rises over the same windows in every cell, so past the
standard budget these models overfit -- the licensed claim is the narrow
negative one (more optimization does not rescue the shared readout), NOT
that optimization trades prediction for parameter error.

## T7. TIMSS raw steps, all three encoders (added 2026-08-14)

100 new fits closed the transformer/dkvmn gap (the frozen panel archived
only the sorted readout; sorting is not invertible). Learned steps are
ordered in .41-.43 of items raw, against the sorted export's artifactual
1.00 and the classical calibration's OWN .613 -- disordered steps are a
property of these data, not a model defect. Scored consistently (raw
against the reference's raw steps) agreement is .92-.94 under both paths,
SK-SH negligible (+.009/-.002/+.010). The .25-.30 figure is what sorting
one side only produces and is not evidence about the model.

## Provenance corrections surfaced

1. P1 NLL exhibit: unreproducible from the tree; superseded by the
   archived per-fit NLL with the OPPOSITE direction (above).
2. The stale CLAUDE.md test path fixed (kt-irt/tests/).
3. Cluster queue turned hostile mid-campaign (7 jobs pinned on
   Priority); per the standing fail-safe everything ran on the local
   4060: 675 + 100 fits in under two hours. The cancelled jobs burned
   no cycles.

## Paper-ready sentences these results license

- "Removing only the dynamics gradient from the shared item table --
  leaving forward computation bit-identical -- recovers a large share
  of the lost parameter fidelity in every cell, at unchanged accuracy
  and NLL; for the memory-addressing encoder it recovers essentially
  all of it."
- "The residual gap to the separated key is a capacity effect: a
  16-wide key already closes most of it, and width is monotone."
- "On a well-observed real bank, the separated readout agrees with
  classical MML at the reference's own split-half ceiling."
- "Fit to identical histories, binary and nominal formulations agree
  with each other under the separated key (.78-.80) and disagree under
  the shared path (.24-.48)."
