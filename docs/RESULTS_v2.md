# Research Ledger v2 -- post engine-reset (2026-06-15)

Papers are paused. The engine is being re-decided: a new ORGANIC ma-irt x substrate
integration (embedding the ma-irt decoder into the substrate architecture as a true
unified design) is being benchmarked against using ma-irt as the initial state s_0, before
any thesis chapter is drafted. In parallel, the RQ battery is being reframed around the
predict-vs-differentiate axis and the universal-feature search. See
`docs/rq_reframe_proposal.md` for the full RQ reframing.

---

## Invalidated: the dual-channel engine (FAIL)

The dual-channel "best-of-both" engine, proposed as Strand 1b in `substrate/RESULTS.md`,
is invalidated. Two independent defects combine to make its positive result an artifact.

**(a) The design was ad-hoc, not a genuine two-theta architecture.** The DIRECT stream
computed theta and fed it into the GPCM training loss. The ITEM-BLIND stream fed alpha
alone, via an occurrence-averaged state-conditioned readout borrowed from ma-irt. Theta
in the loss was only the direct stream. The item-blind stream never contributed a second
theta estimate, so the engine did not actually have two theta channels and did not
resolve any theta-vs-discrimination trade-off. The "best-of-both" claim rested on
reading discrimination from the item-blind state while using only direct-stream theta
everywhere else -- a two-stream hack, not a resolved architectural tension.

**(b) The benchmark that justified it was contaminated by a double-shift bug.** The
item-blind stream is read-before-write internally (the DKVMN/separate-theta pattern:
the current item is excluded from the summary the stream reads). The benchmark then
shifted the target sequence a second time before comparison, so the item-blind stream
was effectively predicting step t from history through step t-2, dropping the most
recent interaction. This is the likely cause of the item-blind stream's worse dynamic
tracking relative to the direct stream -- not any intrinsic smoothness of the
item-blind summary. The discrimination advantage the dual reported was therefore
measured under a contaminated baseline.

**Net verdict.** The "trade-off resolved" result was an artifact, not a finding. The
dual-channel engine is invalidated and is not being carried forward.

This supersedes the "Strand 1b -- DUAL-CHANNEL" bullet in `substrate/RESULTS.md`
(lines 169-170, the struck-through block). That bullet is the historical record and is
not edited here.

---

## Engine decision: DECOUPLED SUBSTRATE is the candidate s_0 (gate flipped, positivity confound controlled)

This supersedes the earlier "ma-irt is s_0" call from the organic bench. Two refinements overturned it.

**1. Decoupled alpha.** The organic bench left ma-irt winning on one axis only, discrimination. Giving alpha its OWN wide item key, a separate emb=64 item table that feeds only the state-conditioned alpha head while the theta-encoder stays cheap (e8h32), lifts substrate alpha to ma-irt's level without the theta tax that widening the shared encoder pays.

**2. Unified positivity transform.** The first flip used softplus for the substrate alpha but ma-irt uses exp(log_scale*raw). That was a confound, since softplus and exp are different positivity maps (nothing is "linked" here, this is just the activation that makes the raw discrimination output positive). ma-irt sets log_scale=1.0, and the MLP-driven alpha head absorbs any scale constant, so the apples-to-apples map is plain exp(raw). Re-running all three substrate variants on exp(raw), the exact transform ma-irt uses, removes the confound. The gate STILL flips.

4-way, GPCM, 3 seeds, 150 epochs, Spearman recovery (exp(raw) on all substrate variants):

| axis | cheap (e8h32) | 64x64 (trap) | DECOUPLED | ma-irt |
|---|---|---|---|---|
| static theta | 0.971 | 0.826 | 0.967 | 0.976 |
| dynamic drift | 0.732 | 0.643 | 0.729 | 0.654 |
| discrimination a | 0.654 | 0.766 | 0.929 | 0.935 |
| difficulty b | 0.977 | 0.978 | 0.972 | 0.988 |
| params | 7022 | 54214 | 10918 | 14917 |
| passes | 1 | 1 | 1 | 2 |

DECOUPLED vs ma-irt (tol 0.02): WINS=[drift +0.075] TIES=[static -0.009, disc -0.006, diff -0.016] LOSSES=[none].

**DECISION.** The decoupled substrate is the candidate s_0. It keeps the cheap theta (static 0.967, drift 0.729), ties ma-irt on discrimination (0.929 vs 0.935) and difficulty, WINS dynamic drift (+0.075), at 10.9k params vs 14.9k and one encoder pass vs two. It matches or beats ma-irt on every recovered axis and owns the dynamic-tracking axis the learning program leans on most.

**Caveats, honest.**
- The cheap baseline alpha is high-variance (0.654 +- 0.171, dragged by a seed where it collapses). The exp transform does not rescue it. The narrow 8-dim shared key is the cause, which is exactly what the separate wide key fixes. This confirms the decoupling story, it is not a positivity-transform effect.
- The decoupled theta holds at the 150-epoch budget but the bare substrate's responsive theta softens under long training (overfit probe, seed 0: 0.962 at 150ep, 0.903 at 300ep, 0.824 at 500ep; ma-irt 0.975, 0.942, 0.871). Decoupling buys alpha capacity without widening the theta path, but it does not add ma-irt's LayerNorm and q-residual regularisation, so the static-theta stability gap reopens with heavy training. Open item for the learning program, port a light regulariser onto the cheap theta path or pin the training budget.
- The wide alpha table is per-item, so it needs occurrences to calibrate and may starve on sparse real banks. Authorised fallback, a learned state-projection feeding only the alpha head. Untested on real data.

Source: substrate/bench/outputs/alpha_fix_table.md, exp(raw) (log_scale=1.0, ma-irt's setting), 3 seeds.

---

## Encoder swappability (the decoupling is backbone-agnostic)

The decoupled-alpha feature is a DECODER-side feature that reads the encoder's per-step state but is otherwise backbone-agnostic. Tested across three substrate backbones at the cheap theta-encoder (e8h32), exp(raw) alpha transform, decoupled emb=64 alpha key, 3 seeds, 150 epochs, GPCM static+dynamic. Per backbone, decoupling KEEPS theta (static + drift) and LIFTS discrimination to ma-irt's level:

| backbone | static theta (cheap -> dec) | drift (cheap -> dec) | alpha (cheap -> dec) |
|---|---|---|---|
| LSTM | 0.971 -> 0.967 | 0.732 -> 0.729 | 0.654 -> 0.929 |
| Transformer | 0.941 -> 0.938 | 0.686 -> 0.678 | 0.650 -> 0.925 |
| DKVMN | 0.983 -> 0.984 | 0.725 -> 0.724 | 0.708 -> 0.916 |

ma-irt reference: static 0.976, drift 0.654, alpha 0.935.

VERDICT: SWAPPABLE. The decoupling is a backbone-agnostic decoder feature, not an LSTM-specific trick. Extra signal: cheap-alpha is not just low but high-variance on every backbone (+-0.11 to +-0.19 across seeds), and decoupling COLLAPSES that variance (+-0.02 to +-0.035) as well as lifting the mean - the signature of capacity-starvation (a thin shared key gives the alpha head too few degrees of freedom, so recovery is biased low and unstable; its own wide key fixes both).

IMPLICATION (the emerging research angle): the alpha-vs-theta capacity conflict appears identically whether the backbone is recurrent, attention, or memory-addressed, so it is intrinsic to AMORTIZED NEURAL PSYCHOMETRIC RECOVERY, not an architecture quirk. Classical IRT cannot have it (independent per-item parameters, no shared representation); DKT never examines it (prediction-only, recovers nothing). Neural IRT is the only place a shared representation meets a demand for faithful recovered parameters, so it inherits this coupling. The decoupling is therefore a MEASUREMENT-VALIDITY condition, not just a performance tweak - it keeps the recovered discrimination faithful, which the thesis's "stable scale" claim depends on. Open: prior-art clearance and real-data confirmation before claiming.

Infra: new substrate-native encoders substrate/core/transformer_encoder.py and dkvmn_encoder.py, both subclassing the new BaseSeqEncoder extracted in substrate/core/encoder.py (LSTM kept bit-identical); SubstrateModel gained an encoder= selector. 129 substrate tests green. DKVMN is ~10x slower per epoch (python timestep loop); the substrate Transformer is param-heavy (~43-47k) from attention. Source: substrate/bench/outputs/swap_table.md.

---

## New question framework

| Question | Type | 2x2 cell | Status |
|---|---|---|---|
| predict across formats | predict | -- | headline |
| fix item -> tell students/LLMs apart | differentiate | fix-item | OPEN |
| fix student/LLM -> tell items apart | differentiate | fix-student | = RQ5c order-yes level-no |
| item universal feature = content | -- | item-side | WORKS, cold-start |
| student universal feature = "true alpha" (speed/drift) | -- | student-side | OPEN |

Old RQ -> new-framework mapping lives in `docs/rq_reframe_proposal.md`.

---

## Results (new engine)

Engine gate resolved above. Item/theta recovery numbers are the 4-way table in the engine-decision section. Downstream RQ results land here as the learning program runs.
