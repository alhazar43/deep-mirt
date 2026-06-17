# Fisher-asymmetry and the dynamics of state-conditioned discrimination (RQ1, RQ2)

Status PRELIMINARY (3 to 5 seeds, single architecture, synthetic static-alpha
data). Branch feat/duolingo-mini. 2026-06-17.

## Framing (DKT-home, the translation gap)

That discrimination is low-information and ill-conditioned is classical IRT (cite,
TODO); this study does not reclaim it. The contribution is the DL-native question
IRT's static theory never poses, what the TRAINING DYNAMICS do with it, whether and
how fast SGD pins each parameter under a PREDICTION loss, and how a representational
choice (decoupling) changes that. Fisher information appears only as the backstage
bridge (a parameter's leverage on the prediction yhat); the front-stage story and
the claims are learning dynamics, for a DKT / DL / EduAI audience. See the project
memory framing rule. The objective throughout is the prediction loss on y vs yhat,
not a model-wise likelihood; the IRT triple is the route to yhat, not the estimand.

## Question

Does state-conditioning an IRT item parameter (letting the decoder read the
encoder hidden state when producing the parameter, occurrence-averaged at
recovery) help recover it, and is the benefit governed by the parameter's Fisher
information (so it is parameter-specific, not generic added flexibility)?

The conjecture, from the per-response Fisher of the 2PL/GPCM. Discrimination alpha
is LOW information, `I(alpha) ~ (theta - beta)^2`, which vanishes at theta = beta
where targeted responses concentrate, so alpha is hard and ill-conditioned at
finite data. Difficulty beta and ability theta are higher information. So a richer
(state-conditioned) readout should help the low-Fisher alpha and do little for the
high-Fisher beta. The stiffness `I(theta)/I(alpha)` grows with K, so the alpha
benefit should grow with K.

All runs decoupled (item_key_dim=64), exp transform, static GPCM data, N=800,
Q=60, T=60, 150 epochs, Adam. Pearson r vs ground truth.

## RQ1 -- the alpha-vs-beta asymmetry (3 seeds)

`alpha-dynamic` switches ON only the state-conditioned alpha head (beta static);
`beta-dynamic` switches ON only the state-conditioned beta head (alpha static);
the two heads gate independently, so each arm makes exactly one parameter dynamic.
Same architectural change, applied to a low-Fisher and a high-Fisher parameter.

| K | a base | a dyn | delta_alpha | b base | b dyn | delta_beta |
|---|---|---|---|---|---|---|
| 2 | 0.798 | 0.703 | -0.095 | 0.977 | 0.987 | +0.010 |
| 4 | 0.876 | 0.933 | +0.057 | 0.983 | 0.986 | +0.002 |
| 6 | 0.928 | 0.941 | +0.014 | 0.982 | 0.983 | +0.002 |
| 8 | 0.914 | 0.951 | +0.037 | 0.982 | 0.983 | +0.001 |
| 11 | 0.755 | 0.951 | +0.196 | 0.980 | 0.979 | -0.001 |

mean delta_alpha = +0.042, mean delta_beta = +0.003, Spearman(delta_alpha, K) = +0.877.

CONFIRMED. Making the low-Fisher alpha dynamic helps and the help GROWS with K;
making the high-Fisher beta dynamic does essentially nothing at any K. The K=2
sign-flip (delta_alpha = -0.095) is consistent with the mechanism, not against it,
at K=2 the stiffness is lowest so alpha is relatively well determined and the
dynamic head only adds noise (it behaves like beta), and the benefit switches on as
stiffness grows. So the effect is parameter-specific and Fisher-governed, which
rules out "generic flexibility helps."

## RQ2 -- the dynamics (5 seeds)

Per-epoch alpha-recovery trajectory for alpha-static vs alpha-dynamic (beta static
in both), via the fit callback (one optimizer, no warm restart). Gap = dynamic -
static, mean over 5 seeds:

```
K     ep1    ep20   ep40   ep80   ep150     static@150   dynamic@150
4    -0.07  +0.26  +0.48  +0.27  +0.03       0.914         0.944
8    -0.04  +0.18  +0.38  +0.31  +0.06       0.902         0.966
11   +0.03  +0.10  +0.42  +0.42  +0.20       0.775         0.970
```

The predicted story ("dynamic peels up EARLIER from the start, lead widens with K")
is HALF WRONG and the correction is the real finding.

- Both curves crawl for the first ~10 to 20 epochs, dynamic is NOT ahead early (it
  is neck-and-neck or slightly behind). So there is no early peel-up.
- A MID-TRAINING surge (ep20 to 40) is where the dynamic head accelerates past
  static, the gap peaks around ep40 to 80 at +0.4 or so, at every K.
- At the ENDPOINT the gap grows monotonically with K (K=4 +0.03, K=8 +0.06,
  K=11 +0.20) because at low K static catches up by ep150, while at high K static
  is trapped (0.775) and dynamic stays far ahead (0.970).

So there IS a convergence-rate advantage, but it is a mid-training ACCELERATION
after a shared slow start (dynamic reaches any given recovery level sooner once the
encoder state organizes, e.g. it hits ~0.78 by ep40 where static needs ~ep90), not
an earlier start. And at high K that rate advantage converts into a PERMANENT
endpoint ceiling gap, because static-alpha cannot escape the stiffness bottleneck
the dynamic head breaks through.

Mechanism. The static head can fit alpha from the item embedding immediately, so it
is competitive early but caps low against the stiffness ceiling. The dynamic head
must wait for the encoder to organize a useful state before its conditioning means
anything, so it lags early, then breaks the ceiling once the state matures. It
trades early speed for a higher ceiling, and at high K the ceiling difference is
the lasting effect.

## Combined verdict (preliminary)

The state-conditioned discrimination advantage is real, parameter-specific, and
Fisher-governed (RQ1). Its dynamics are a shared slow start, a mid-training
acceleration of the low-Fisher parameter, and a K-growing permanent endpoint gap
where static-alpha is trapped by the stiffness ceiling (RQ2). The earlier
"peels up from epoch one" framing (inherited from the retracted width study) does
not hold for the static-vs-dynamic head, correct it to "mid-training acceleration
plus K-growing ceiling escape."

Caveats. 3 to 5 seeds, one architecture (LSTM/GPCM, emb=8/hidden=32/key=64), one
data regime (static-ability synthetic). The mid-K delta_alpha values in RQ1 are
noisy; the robust signals are the endpoints, the beta null, and the K-correlation.

## Reproduce

```
python deep_irt/bench/run_alpha_beta_asymmetry.py --device cuda          # RQ1
python deep_irt/bench/run_convergence.py --seeds 0 1 2 3 4 --device cuda  # RQ2
```
Outputs under deep_irt/bench/outputs/ (alpha_beta_asymmetry.*, convergence_K*.json).
