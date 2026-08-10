# P1: held-out NLL / ECE gap (theory falsification test)

Positive d = SK better (lower NLL/ECE). Paired per seed, df=4 t.
Theory prediction: SK <= SH beyond seed noise, largest for
transformer and dkvmn; exact NLL tie falsifies mechanism A.

| enc | dec | n | NLL SH | NLL SK | d_NLL [t, pos] | d_ECE [t] | d_acc |
|---|---|---|---|---|---|---|---|
| lstm | 2pl | 5 | 0.5824 | 0.6000 | -0.0176 [t=-6.9, 0/5] | -0.0129 [t=-5.4] | -0.0074 |
| lstm | gpcm | 5 | 1.0962 | 1.1073 | -0.0111 [t=-6.8, 0/5] | -0.0074 [t=-5.9] | +0.0019 |
| transformer | 2pl | 5 | 0.8552 | 0.8984 | -0.0432 [t=-3.0, 1/5] | +0.0100 [t=3.3] | +0.0023 |
| transformer | gpcm | 5 | 1.2318 | 1.2682 | -0.0364 [t=-3.0, 0/5] | -0.0103 [t=-2.0] | +0.0074 |
| dkvmn | 2pl | 5 | 0.5476 | 0.5522 | -0.0046 [t=-19.8, 0/5] | -0.0025 [t=-2.3] | -0.0037 |
| dkvmn | gpcm | 5 | 1.0731 | 1.0725 | +0.0006 [t=0.8, 3/5] | -0.0014 [t=-0.5] | +0.0036 |

## Verdict: Prediction 1 REFUTED AS STATED; corrected account

The theory predicted SK <= SH in held-out NLL. Observed: SH wins in
5/6 cells, small but systematic (.005-.043 nats, paired t 3.0-19.8),
ECE mostly agreeing, accuracy tied throughout; the one tie is
dkvmn-gpcm (the encoder with internal separation pays least).

Corrected account (consistent with the routing theory's own core): the
shared arm optimizes over a LARGER effective function class -- the item
embedding co-adapts with the trajectory -- so its held-out likelihood
should weakly exceed SK's. The displaced parameters are not merely
likelihood-cheap; they are the likelihood's optimum under sharing.
Mechanism A (head displacement from the conditional-MLE readout, with
the information retained in the table) stands on the probe evidence;
what dies is the specific sign prediction.

Paper consequences: (1) the claim scopes to ZERO ACCURACY COST, with
the ~.01-.04-nat shared-arm NLL edge reported honestly; (2) the
sharper story: the corruption is PURCHASED -- prediction training pays
a measurable soft-probability premium for exactly the entanglement
that destroys measurement. The trade-off now has a price on both
sides, which strengthens, not weakens, the dissociation argument.

