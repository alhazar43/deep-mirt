# The contested item channel

A routing account of why prediction-trained knowledge-tracing models corrupt their item-parameter readouts, why a separated key repairs them at zero accuracy cost, and why the linear-probe matrix looks the way it does.

Status. Theory support for the measurement-audit paper. The empirical matrix and the probe table are the authority; every claim below is graded in the honesty ledger (Section 11). Notation is consistent with the gradient identity already in the paper (Section 3.1 block, `main_caeai.tex`), extended here to displacement magnitudes, the probe-head split, the purge, and crowding.

---

## 0. What must be explained

Recovery matrix (Spearman of recovered vs true, N=2000, Q=200, ~600 responses/item, 25 fits/cell; accuracy tied SH vs SK in every row):

| encoder | dec | arm | acc | $a$ | $b$ |
|---|---|---|---|---|---|
| dkvmn | 2pl | SH | .716 | .752 | .652 |
| dkvmn | 2pl | SK | .715 | .914 | .950 |
| dkvmn | gpcm | SH | .496 | .879 | .849 |
| dkvmn | gpcm | SK | .502 | .952 | .966 |
| lstm | 2pl | SH | .712 | .553 | .723 |
| lstm | 2pl | SK | .714 | .898 | .957 |
| lstm | gpcm | SH | .487 | .719 | .826 |
| lstm | gpcm | SK | .502 | .941 | .965 |
| transformer | 2pl | SH | .708 | .373 | .604 |
| transformer | 2pl | SK | .713 | .806 | .955 |
| transformer | gpcm | SH | .459 | .438 | .768 |
| transformer | gpcm | SK | .492 | .900 | .947 |

Ridge decodability of true parameters from the trained embedding tables (5-fold item-CV Spearman; recovery repeated for comparison):

| enc | dec | arm | channel | $\log a$ probe | $b$ probe | recovery $a$/$b$ |
|---|---|---|---|---|---|---|
| dkvmn | 2pl | SH | value | .799 | .986 | .752 / .652 (probe n=4, preliminary) |
| lstm | 2pl | SH | value | .751 | .984 | .553 / .723 |
| lstm | gpcm | SH | value | .816 | .984 | .719 / .826 |
| transformer | 2pl | SH | value | .364 | .978 | .373 / .604 |
| transformer | gpcm | SH | value | .619 | .977 | .438 / .768 |
| lstm | 2pl | SK | key | .715 | .975 | .898 / .957 |
| lstm | 2pl | SK | value | .222 | .449 | |
| lstm | gpcm | SK | key | .871 | .983 | |
| lstm | gpcm | SK | value | .309 | .605 | |
| transformer | 2pl | SK | key | .641 | .972 | |
| transformer | 2pl | SK | value | .064 | .329 | |
| transformer | gpcm | SK | key | .826 | .983 | |
| transformer | gpcm | SK | value | .068 | .526 | |

Patterns to produce. (P1) SK lifts both parameter families on every encoder to a near-uniform plateau. (P2) Under SH the lagging family flips with architecture (DKVMN lags $b$, LSTM and transformer lag $a$). (P3) The transformer degrades both worst. (P4) Under the misspecification battery the SK-SH gap never reverses and grows under local dependence and threshold disorder. (P5) Accuracy is tied in every row. (F1) $b$ information is retained near-perfectly in the SH value table everywhere (probe $\geq .977$) even where trained-head recovery is far lower; the $b$ failure is readout misalignment. (F2) $a$ information shows two regimes, present-but-under-extracted (LSTM, DKVMN) and genuinely absent (transformer 2PL, probe $.364 \approx$ recovery $.373$). (F3) Under SK the value table is purged of parameter information, with an asymmetric residue ($b$ trace $.33$-$.61$, $a$ trace $\approx 0$).

The account in one paragraph. The shared embedding $e_i$ receives gradients through two routes, the parameter route (through the heads into the current-step likelihood) and the dynamics route (through the encoder into future ability estimates). At any near-stationary point of shared training the two routes balance rather than separately vanish, so the head-read parameters sit displaced from their per-item maximum-likelihood values by an amount inversely weighted by the parameter-route Fisher information and structured by what the dynamics want from the item, not by the item's true parameters. That displacement corrupts ranks in the head output while the table itself keeps clean, linearly decodable parameter information, deposited over training and additionally demanded by the dynamics in the case of difficulty. This is mechanism A (displacement), and it is universal under SH. On top of it, architectures whose dynamics make broadband demands on $e_i$ overwrite the weakly re-deposited discrimination signal entirely; this is mechanism B (crowding), strongest under global attention. Separating the readout onto a key $k_i$ that the dynamics never touch zeroes the balance condition, making the trained heads exact conditional M-estimators (mechanism A gone) with private capacity (mechanism B gone), while $e_i$ relaxes to a purely dynamics-serving code (the purge, F3). Nothing in this trades prediction quality, because the displacement is taken in likelihood-cheap directions and argmax accuracy near the Bayes ceiling is insensitive to it.

---

## 1. Setup, pinned to the code

Learners $j$, items $i \in \{1,\dots,Q\}$, ordinal responses $y_{jt} \in \{0,\dots,K-1\}$ ($K=2$ is 2PL). Per item, a thin value embedding $e_i \in \mathbb{R}^{8}$ and, in the SK arm only, a wide key $k_i \in \mathbb{R}^{64}$ (`item_val_emb`, `item_key_emb`).

Likelihood (from `decoders.py`). GPCM category logits

$$\psi_{0}=0,\qquad \psi_{k} = \sum_{c=1}^{k} a_i(\theta_{jt} - b_{i,c}), \quad k = 1,\dots,K-1,$$

with $a_i = \mathrm{softplus}(w_a^\top x_i)$ and $b_i = W_b x_i \in \mathbb{R}^{K-1}$, where $x_i = e_i$ (SH) or $x_i = k_i$ (SK, Option A: both static heads on the key). The loss is next-response cross-entropy $L = -\sum_{j,t} \log p(y_{jt})$. Both arms in the matrix use the static item-only heads; the state-conditioned variants are not part of this matrix.

Ability (from `encoder.py`). $\theta_{jt} = w_\theta^\top h_{j,t-1}$, where $h$ is the backbone hidden and the single causal shift is universal. This gives the load-bearing structural fact:

> **The alignment contract.** For every backbone, the state that predicts step $t$ is a function of the history strictly before $t$. $\theta_{jt}$ never sees the current item $q_t$. The current item enters the step-$t$ likelihood only through the heads.

Consequences. (i) There is no per-item location gauge here. In the archived MA-GPCM chapter, $\theta$ read the current item's key jointly, which opened a continuum of per-item $(\theta,b)$ splits and collapsed $b$ at the identifiability level. That door is closed in this design; whatever corrupts $b$ here is not an identifiability failure of the likelihood. (ii) Given the trajectory $\hat\theta$, the per-item parameters are identified (up to the global affine gauge below), so any corruption must come from where training leaves the heads, not from what the likelihood can distinguish.

Roles of $e_i$ per backbone (the dynamics route's demand structure):

- **LSTM.** $e_i$ is half of the input $[e_{q_t}, \mathrm{emb}(r_t)]$ to all gates at each occurrence; its influence persists through the cell state within a sequence.
- **DKVMN.** Two uses per occurrence. Addressing, $\kappa_t = W_k e_{q_t}$, softmaxed against 20 static anchors $M_k$; and value writing, $v_t = W_v[e_{q_t}, \mathrm{emb}(r_t)]$, erase/add into $M_v$ at the addressed slots. The summary is $h_t = \tanh(W_s[\mathrm{read}_t, \kappa_t])$, so after the shift the next-step ability $\theta_{j,t+1}$ contains a short, high-gain path from the current item's embedding, $e_{q_t} \to \kappa_t \to h_t \to \theta_{j,t+1}$. This is the post-response correction channel; it is difficulty-conjugate by function (what the filter must subtract after seeing a response is the item's difficulty).
- **Transformer.** $e_i$ enters the token $x_t = W_{in}[e_{q_t}, \mathrm{emb}(r_t)] + \mathrm{pos}_t$, which then serves as attention key and value for every later position in every layer and head, and as query for every earlier one (2 layers, 4 heads). The number of distinct linear functionals of $e_i$ the loss is sensitive to scales with heads, layers, and contexts. This is the broadest demand profile of the three.

---

## 2. Gauge, expressivity, finite sample, in that order

Before any dynamics claim, the standard ladder.

**Gauge.** The 2PL/GPCM likelihood is invariant under the global affine gauge $\theta \to c\theta + d$, $b \to cb + d$, $a \to a/c$ (one $(c,d)$ for the whole fit). These maps are monotone per parameter, and Spearman is invariant under monotone maps. The gauge is therefore rank-inert. No entry of the matrix can be a gauge artifact, and no gauge fixing is needed for the analysis. (The per-item location gauge that would not be rank-inert is excluded by the alignment contract.)

**Expressivity.** The heads are linear (plus softplus) in free per-item tables. Any assignment $i \mapsto (a_i, b_i)$ is realizable in either arm, in the SH arm because $e_i$ is free per item and $d=8 \geq K$. There is no expressivity wall; SH and SK realize the same class of item-parameter tables. This is why the arms can tie in accuracy while differing in readout, and why the corruption must be about which point training selects, not about what the architecture can represent.

**Finite sample.** With $n_i \approx 600$ responses per item, the per-item conditional MLE given a good trajectory has standard error $\mathrm{SE} \approx (n_i \bar I)^{-1/2}$ per parameter. Per-response Fisher information for the 2PL at a response with $z = \theta - b$, $w = p(1-p)$:

$$I_{aa} = \mathbb{E}[w z^2], \qquad I_{bb} = a^2\, \mathbb{E}[w], \qquad I_{ab} = -a\,\mathbb{E}[w z] \approx 0 \ \text{by symmetry}.$$

At $a = 1$, $\theta \sim N(0,1)$, $b=0$: $\mathbb{E}[w] \approx 0.21$ and $\mathbb{E}[w z^2] \approx 0.15$. Two consequences. First, with $\sigma_b \approx 1$ the reliability of $\hat b^{\mathrm{MLE}}$ is near $.99$; adding trajectory error of the size the oracle ladder measures (refit at own $\hat\theta$ reaches $.934$, oracle clamp $.979$ on the LSTM-GPCM bed) puts the achievable plateau at roughly $.93$-$.97$. The SK $b$ values $.947$-$.966$ sit exactly there. Second, $a$ has the smaller information and additionally suffers errors-in-variables attenuation noise from $\hat\theta$ (a multiplicative parameter regressed on a noisy regressor), so its plateau sits lower; SK $a$ $.806$-$.952$, lowest for the transformer, whose trajectory is noisiest. The SK plateau is the finite-sample floor, not a residual mystery.

**Residual.** Everything below the plateau in the SH rows is the candidate dynamics effect. The rest of this document derives it.

Two banked facts about the Fisher entries that the derivation uses and that must be quoted with their conditions. (i) Unconditional suppression, $\mathbb{E}_w[z^2] < \mathbb{E}[z^2]$, because $w$ and $z^2$ are oppositely monotone in $|z|$ (correlation inequality). Information about $a$ is suppressed exactly where responses concentrate. (ii) Conditional ordering, $I_{aa} < I_{bb}$ iff $\mathbb{E}_w[z^2] < a^2$, which holds for $a \gtrsim a_\star \approx 1$ and inverts below it. With a generating $a$ distribution centered near 1 the ordering is typical across items, not a law (`docs/paper2_leverage_proposition.md`).

---

## 3. The routing identity and the two stationarity regimes

Write the item-$i$ head parameters $\phi_i = (a_i, b_i) \in \mathbb{R}^{K}$, $\phi_i = g(x_i)$ with Jacobian $J_i = \partial g / \partial x \in \mathbb{R}^{K \times d}$. Let $s_i = \sum_{(j,t):\, q_t = i} \nabla_{\phi} \log p(y_{jt})$ be the accumulated per-item score, and let

$$D_i \;=\; \sum_{j,t} \frac{\partial L}{\partial \theta_{jt}} \frac{\partial \theta_{jt}}{\partial e_i}$$

be the dynamics-route gradient (nonzero only for occurrences of $i$ strictly before $t$, by the alignment contract; head path and dynamics path never touch the same position twice). Then, exactly and everywhere,

$$\nabla_{e_i} L = -J_i^\top s_i + D_i \qquad (\text{SH}), \qquad \nabla_{k_i} L = -J_i^\top s_i \qquad (\text{SK}),$$

the second because $\partial \theta_{jt} / \partial k_i \equiv 0$ identically (the key feeds only the heads; verified in `encoder.py`, `decoders.py`). This pair is the paper's Section 3.1 identity; everything new starts from it.

**Proposition 1 (SK endpoint = conditional M-estimation).** At any stationary point of SK training, $J_i^\top s_i = 0$; if $J_i$ has full row rank (generic for a linear-plus-softplus head with $d = 64 \geq K$), then $s_i = 0$ for every item. The trained head parameters solve the per-item likelihood equations at the model's own trajectory $\hat\theta$, i.e., they are the per-item conditional MLE given $\hat\theta$, exactly the refit-rung estimator. Their recovery is then the finite-sample plateau of Section 2, uniform across architectures because the estimating equation no longer references the encoder. This is P1, including its level.

*Assumptions.* Stationarity in the $k_i$ block (early stopping leaves a small residual gradient; the statement degrades continuously in $\|\nabla_{k_i}L\|$); no explicit weight decay on the key table (else $s_i = O(\lambda)$).

**Proposition 2 (SH endpoint = displaced M-estimation).** At any point where the $e_i$ block is stationary, $J_i^\top s_i = D_i$. Two consequences.

(a) *The conflict is squeezed into the head plane.* $D_i \in \mathrm{range}(J_i^\top)$, a $K$-dimensional subspace of $\mathbb{R}^{8}$. In the remaining $8-K$ directions the table is free, and training uses that freedom to satisfy the dynamics fully; the residual fight lives entirely in the directions the heads read. Widening $e_i$ does not change the form of this balance (the head plane is still $K$-dimensional and the balance condition is unchanged), which is why extra width alone cannot remove this mechanism. It can only relieve mechanism B below.

(b) *Displacement formula.* Solving the balance, $s_i = (J_i J_i^\top)^{-1} J_i D_i$, and expanding the score around the conditional MLE $\phi_i^*$ (where $s_i(\phi^*) = 0$), $s_i \approx -n_i \bar I_i (\hat\phi_i - \phi_i^*)$, gives

$$\hat\phi_i - \phi_i^* \;\approx\; -\,(n_i \bar I_i)^{-1} (J_i J_i^\top)^{-1} J_i\, D_i \;=:\; \delta_i .$$

The displacement is (i) proportional to the dynamics-route gradient at the endpoint, (ii) amplified by the inverse per-item Fisher information, so for the same conflict the low-information parameter moves further ($a$, conditionally on $a \gtrsim a_\star$), and (iii) structured across items by whatever structures $D_i$, which is the item's role in the dynamics (exposure, cluster membership, co-occurrence, position statistics), not its true parameters.

**Corollary (rank corruption).** Spearman is destroyed only by item-heterogeneous, non-monotone contamination. A displacement field constant across items, or monotone in the true parameter, would be rank-inert. $\delta_i$ inherits the geometry of $D_i$, which is a function of dynamics covariates; to the extent those are not monotone functions of $(a_i, b_i)$, the head output loses rank precisely by $\mathrm{Var}(\delta)/\mathrm{Var}(\phi^*)$ in the usual attenuation form $\rho \approx \rho_0 / \sqrt{1 + \mathrm{Var}(\delta)/\mathrm{Var}(\phi^*)}$ (orthogonal-contamination approximation). SK has $\delta_i = 0$.

**Why the two routes genuinely conflict.** If the dynamics-optimal table were also parameter-route optimal, $D_i$ and $s_i$ could vanish together and SH would match SK. Both must vanish simultaneously in the same $d$ coordinates, an over-determined condition that fails generically whenever the dynamics demand anything from $e_i$ beyond head-readable $(a_i, b_i)$, which Section 6 shows they do (cluster geometry, update corrections). The degenerate case is instructive, though. If the dynamics demanded nothing (a decoder-only model), SH would be a pure M-estimator too. The corruption is exactly the price of the second job.

---

## 4. Mechanism A. Displacement explains F1

F1 says the SH table linearly contains $b$ at $.98$ everywhere while the trained head recovers $.60$-$.83$. Both are linear maps of the same table. The resolution has two halves, what the head reports and what the table retains.

**What the head reports.** By Proposition 2 the head output is $\hat b_i = b_i^* + \delta_i^{(b)}$, the conditional MLE plus the displacement field. The refit rung measures $\mathrm{rank}(b^*) \approx .93$ at the model's own trajectory; the head's $.65$-$.72$ against the probe's $.98$ therefore measures a large, structured $\delta^{(b)}$, exactly what the balance condition manufactures. The head is not a broken decoder. It faithfully reports the values that the two-route equilibrium selects, and those values are displaced.

**What the table retains.** Three deposits keep clean $b$ information linearly present.

1. *The dynamics' own demand.* The update after a response must weigh correct-on-hard differently from correct-on-easy, so the encoder itself wants difficulty readable from $e_i$. The SK value-channel trace measures this demand in isolation, $b$ decodable at $.45$-$.61$ with no parameter route at all (F3). This deposit exists in every architecture.
2. *The parameter route's accumulated deposits.* The table is a time integral, $e_i(T) = e_i(0) - \int_0^T [\,-J^\top s_i + D_i\,]\,dt$. During training the head direction rotates while the score term writes $b^*$-correlated increments along it; those increments persist in directions that nothing later repurposes. The endpoint head reads one direction with the equilibrium displacement; the probe integrates over the whole deposited fan and over deposit 1, and can subtract the contamination because the contamination is itself table-decodable. Formally, if $\delta_i = f(\xi_i)$ for nuisance coordinates $\xi_i$ readable from the table (which holds whenever $D_i$ is driven by features the dynamics encode in $e_i$, and holds exactly in the linear model of Section 8), then the probe $w^* = w_b - f \circ P$ achieves at least $\mathrm{rank}(b^*)$, strictly above the head whenever $\delta$ has a non-monotone item-structured component. The observed $.98$ says the displacement field is almost entirely table-predictable, consistent with its drivers (cluster, exposure geometry) being exactly what the dynamics store.
3. *Combination.* Ridge combines deposits 1 and 2 with partially independent errors, which is how the probe can exceed even the refit rung's $.934$.

**Why accuracy still ties (P5).** The balance condition takes the parameter error where it is likelihood-cheap. The per-response loss cost of a displacement is $\tfrac12 \delta^\top \bar I \delta$, weighted by the same Fisher matrix whose inverse sized the displacement, so the equilibrium trades a second-order likelihood cost for a first-order dynamics gain, and the reported functional, argmax accuracy near the Bayes ceiling of the generating process, is additionally insensitive to calibration-level changes. Two honest corollaries. First, the tie should be exact only in accuracy; in NLL and calibration error the SH arm should sit measurably above SK (the refit rung strictly lowers NLL at fixed everything else, so the displaced endpoint is not likelihood-optimal). This is a falsifiable exposure of mechanism A (Prediction 2). Second, if NLL also tied exactly beyond seed noise, the displacement account as stated would need revision. We state this openly rather than immunize it.

---

## 5. Mechanism B. Crowding explains F2

F2 says $a$ information is present but under-extracted for LSTM and DKVMN (probe $.75$-$.82$ vs recovery $.55$-$.75$) and genuinely absent for the transformer at 2PL (probe $.364 \approx$ recovery $.373$). Mechanism A covers under-extraction (the displacement $\delta^{(a)}$ is the inverse-Fisher-amplified component, so present-but-displaced is the default fate of $a$). Absence needs a second mechanism, about the table's content rather than the head's alignment.

**Deposit strength follows route information.** The only writer of $a$ information into $e_i$ is the parameter route ($a$ is essentially not dynamics-demanded; the SK value trace is $.06$-$.31$, F3). Its writing pressure per response is the $a$-score, whose size is governed by $I_{aa} = \mathbb{E}[w z^2]$, the suppressed entry. The deposit is weak by the same suppression lemma that makes the displacement large.

**Overwriting under broadband demand.** Deposits persist only in table directions that no other consumer repurposes. The transformer's dynamics make demands on every direction of $e_i$ (each occurrence's token feeds keys, values, and queries of every later attention computation across layers and heads), so the co-occurrence geometry that attention optimizes continually rewrites the whole $8$-dimensional layout. A weak $a$ deposit does not survive; the ridge probe finds mostly attention-serving structure. The LSTM's demand is broad but gated and sequence-local; the DKVMN's demand on $e_i$ is the narrowest in effective dimension (an addressing direction against 20 static anchors plus a write projection), leaving the most slack. The predicted crowding order, dkvmn $<$ lstm $\ll$ transformer, matches the probe row for $a$, $.799 > .751 \gg .364$. We do not claim to derive this ordering from first principles; the mechanism-bearing statistic is the effective rank (participation ratio) of the per-item gradient covariance $\mathrm{Cov}_t(\partial L / \partial e_i)$ over training, and the ordering is a measurable prediction (Prediction 5), currently an architecture-grounded reading of the probe data.

**Two within-data checks.** (i) $K$ raises $I_{aa}$ (more categories, more informative scores; the banked GPCM information ratios grow with $K$), so GPCM should strengthen the deposit and shrink the displacement simultaneously. Every GPCM SH cell has higher $a$ recovery than its 2PL sibling ($.752\to.879$, $.553\to.719$, $.373\to.438$), and the transformer's probe moves from absent to mixed ($.364 \to .619$ present, recovery $.438$ still displaced), exactly the regime transition the two mechanisms predict. (ii) Width relieves crowding but not displacement (Proposition 2a). The banked width sweep (shared $w8 \to w64$ closes roughly half the SH-SK gap, the rest closing only under separation) is the arithmetic of a two-mechanism decomposition, one width-sensitive, one routing-sensitive. The probe data now locates them, crowding on the $a$ side of some architectures, displacement everywhere.

---

## 6. The purge under SK explains F3

**Proposition 4 (purge; theorem in the linear model, Section 8).** Under SK, $\nabla_{e_i} L = D_i$ only. Under gradient flow from small initialization, $e_i(T) - e_i(0) = -\int_0^T D_i\,dt$ lies in the span of the dynamics sensitivities $\{\partial \theta_{jt}/\partial e_i\}$ pulled back through the loss. The limiting table contains only dynamics-demanded structure plus initialization noise, and the ridge decodability of any item covariate equals its incidental correlation with that structure. Parameter information appears in $e_i$ exactly to the extent the dynamics themselves want it.

The measured residue is then a direct readout of the dynamics' demand profile, and its asymmetry is the differential-demand premise verified. Difficulty is demanded (trace $.449/.605$ LSTM, $.329/.526$ transformer; higher under GPCM, where richer responses make the difficulty correction more valuable to the update). The slope is not (trace $.064$-$.309$, the nonzero remainder consistent with incidental correlation in the generating bank and with nonlinear encoding the ridge lower-bounds).

**Why the dynamics demand $b$ but not $a$.** In the filtering picture the post-response update needs the innovation $y - p_i(\theta)$, and computing $p_i$ requires the item's location at first order; ignoring $b_i$ biases the ability update directionally (correct-on-easy is treated as correct-on-hard). Knowing $a_i$ improves only the weighting of innovations, a second-order gain of order $\mathrm{Var}(a) \cdot \mathbb{E}[wz^2]$, suppressed by the same factor as $I_{aa}$. In the Gaussian model of Section 8 this is exact, first-order demand for $b$, zero demand for $a$ when $\mathrm{Var}(a) = 0$ and second-order otherwise. One suppression principle thus appears three times, weak $a$-score (deposit), large $a$-displacement (inverse Fisher), and negligible $a$-demand (purge trace).

---

## 7. The matrix, cell by cell

**P2, the flip.** Which family lags under SH is set by which head's input direction is shared with a vigorous dynamics consumer and by the inverse-Fisher amplification.

- *DKVMN.* The difficulty-conjugate directions of $e_i$ are consumed by the post-response correction path ($e_{q_t} \to \kappa_t \to h_t \to \theta_{t+1}$, one $\tanh$ from $\theta$) and by slot addressing against the static anchors, both first-order, both restless (addressing competes across items; the correction co-adapts with slot contents and the learned `value_init`). The $b$-head, aligned with the direction where difficulty information lives, gets dragged; $\delta^{(b)}$ is large and cluster-structured. Meanwhile the demands on $e_i$ are low-dimensional, so the $a$ deposit survives in a comparatively quiet private direction with a small conflict component; $a$ recovery $.752$ is the best SH $a$ in the matrix, with the smallest probe-recovery gap ($.799$ vs $.752$). Hence the flip. The strong form of the old internalization hypothesis (slots absorb difficulty so $e_i$ stops carrying it) is refuted by the probe ($b$ decodable at $.986$ while the head reads $.652$); what the memory changes is not the table's content but the vigor with which the difficulty directions move and hence the head's displacement.
- *LSTM.* Difficulty is demanded and its direction moves gently (gated, no slot competition, no persistent cross-learner store beyond the weights), so $\delta^{(b)}$ is the smallest ($.984$ probe vs $.723$ head). The $a$ deposit is weak, the demand profile broad enough to produce a sizable $a$-conjugate conflict, and the inverse-Fisher amplification does the rest, $.751$ present vs $.553$ recovered. $a$ lags.
- *Transformer (P3).* Broadband demand crowds the $a$ deposit out entirely at 2PL (mechanism B) and produces the largest displacement fields for both families (largest probe-recovery gap for $b$, $.978$ vs $.604$). Both lag; $a$ worst.

The $b$ displacement ordering lstm $<$ dkvmn $<$ transformer (probe-recovery gaps $.26$, $.33$, $.37$) and the crowding ordering dkvmn $<$ lstm $\ll$ transformer are the two one-dimensional summaries the mechanisms need; both are measured, neither is assumed.

**GPCM vs 2PL rows.** GPCM raises $I_{aa}$ (deposit up, displacement down; every SH $a$ improves) and hands the $b$-head $K-1$ thresholds whose recovery benefits from more informative per-response scores (every SH $b$ improves). Accuracy falls from 2PL to GPCM only because $K$-way classification is harder; the arms stay tied within rows, as P5 requires.

**P1, the SK plateau.** Proposition 1 plus the Section 2 arithmetic. Uniformity across encoders because the estimating equation is encoder-free; residual non-uniformity (transformer SK $a = .806$) tracks trajectory noise entering as errors-in-variables, the one encoder-dependent input left.

**P4, the misspecification battery (heuristic).** Local dependence adds sequence-predictable signal reachable only through the dynamics route, raising $\|D_i\|$ and its item-heterogeneity, so SH displacement grows while SK's estimating equation is untouched (its target degrades only through $\hat\theta$ quality, second order). Threshold disorder raises the parameter-route residual, and under SH that residual pressure cycles through $e_i$ and enlarges the balance scores, while under SK it is absorbed by $k_i$ alone. Exposure imbalance heterogenizes $n_i \bar I_i$, amplifying displacement dispersion for SH and only widening the MLE noise for SK. All three grow the gap; none reverses its sign, since $\delta_i = 0$ is architectural under SK. This is the weakest section, argued not derived, and marked accordingly.

---

## 8. A minimal model in which the propositions are theorems

**Model M.** Items carry $(\alpha_i, \beta_i)$, $\beta_i \sim N(0, \sigma_\beta^2)$, $\alpha_i = 1 + \epsilon_i$ with $\mathrm{Var}(\epsilon)$ small. Learner ability $\theta^*_j$ fixed (or slow AR(1)). Responses $y_{jt} = \alpha_i(\theta^*_j - \beta_i) + \varepsilon_{jt}$, $\varepsilon \sim N(0, \sigma^2)$, squared-error loss (the Gaussian analog of the GLM; local expansion of Section 3 maps it back). The model predicts $\hat y_{jt} = \hat\alpha_i(\hat\theta_{jt} - \hat\beta_i)$ with linear heads $\hat\alpha_i = 1 + w_a^\top x_i$, $\hat\beta_i = w_b^\top x_i$, and a linear filter $\hat\theta_{j,t} = \hat\theta_{j,t-1} + g^\top [e_{q_{t-1}}, y_{j,t-1}]$ (the one-shift contract). All maps linear, loss quadratic, gradient flow exact.

Results, with proof sketches.

- **M1 (routing).** The identities of Section 3 hold with $J = [w_a, w_b]^\top$ constant. SK stationarity gives $s_i = 0$, hence $(\hat\alpha_i, \hat\beta_i)$ equal the per-item least-squares estimates given $\hat\theta$; consistency and the $1/\sqrt{n_i}$ plateau follow from standard M-estimation. *(Theorem.)*
- **M2 (displacement).** SH stationarity gives $\hat\phi_i - \phi_i^* = -(n_i \bar I_i)^{-1}(JJ^\top)^{-1} J D_i$ exactly (quadratic loss, no expansion error). $D_i$ is computable, a linear functional of occurrence statistics (exposure, adjacency, learner overlap) times table and filter weights, hence item-structured and $\beta$-independent in distribution; the Pearson attenuation $\rho = \rho_0 (1 + \mathrm{Var}\,\delta/\mathrm{Var}\,\beta)^{-1/2}$ follows, and ranks inherit it monotonically for jointly Gaussian fields. *(Theorem.)*
- **M3 (probe-head split).** In Model M, $D_i$ is linear in the table, so $\delta_i$ is a linear functional of $e_i$; the ridge probe attains $\mathrm{rank}(\beta^*)$ by subtracting it, while the head reports $\beta^* + \delta$. Probe strictly exceeds head whenever $\mathrm{Var}(\delta) > 0$ with a non-monotone component. *(Theorem, and the formal content of F1.)*
- **M4 (purge and demand asymmetry).** Under SK gradient flow from $e_i(0) = 0$, $e_i(T) \in \mathrm{span}\{\text{filter sensitivities}\}$. The optimal filter needs $\beta_{q_{t-1}}$ at first order (the innovation $y/\alpha + \beta = \theta + \varepsilon/\alpha$ requires the difficulty correction) and needs $\alpha$ only through the precision weighting, a term of order $\mathrm{Var}(\epsilon)$; with $\mathrm{Var}(\epsilon) = 0$ the demand for $\alpha$ is exactly zero. Hence $\beta$-decodability of the SK value table is positive and $\alpha$-decodability is $O(\mathrm{Var}\,\epsilon)$ plus incidental correlation. *(Theorem.)*
- **M5 (crowding, semi-quantitative).** Give the dynamics a broadband demand (the filter reads $R e_i$ for a full-rank, training-rotating $R$, the linear caricature of attention). The stationary table is dominated by the strong-pressure structure; the finite-$Q$ ridge probe of the weak $\alpha$ deposit has SNR $\propto$ (deposit amplitude)$^2$ against interference filling all $d$ dimensions, and fails below a threshold amplitude. This reproduces regime 2 of F2 but as an SNR statement under stated interference, not a clean theorem. *(Proposition with assumptions.)*

Transfer to the real system. M1-M3 transfer under local quadratic expansion of the GLM around the conditional MLE with Fisher weights (Section 3's derivation is exactly this), with the caveats that stationarity is approximate under early stopping and that $J$ varies with $x_i$ through the softplus. M4 transfers as stated up to the span of a training-time-varying sensitivity set. M5 is qualitative everywhere.

---

## 9. What the hypotheses got right and wrong

- **H1** (global state cannot internalize per-item difficulty, so $b$ survives in $e_i$; $a$'s weak Fisher pressure loses the contested capacity). The conclusion survives, the mechanism is corrected twice. First, no architecture can carry current-item difficulty in $\theta_t$, not because of state capacity but because of the one-shift alignment contract; the constraint is universal, which is why $b$ is present at $.98$ in every SH table, transformer included. Second, $a$'s weakness manifests as displacement (LSTM, present-but-under-extracted), not as loss of information, except where crowding removes it (transformer).
- **H2** (DKVMN internalizes difficulty into slot content, so $b$ starves in $e_i$). Refuted in its strong form by F1: the DKVMN table carries $b$ at $.986$ while the head reads $.652$. The correct statement is that the memory architecture makes the difficulty-conjugate directions of $e_i$ the most contested and mobile (correction path and addressing), displacing the $b$-head while leaving the table's content intact. Architecture-dependence lives in the heads' alignment, not the tables' content.
- **H3** (the key's stationary point is an almost-clean M-estimator; no dynamics pathway, so prediction unchanged). Correct, and now Proposition 1 plus the purge (Proposition 4), with the plateau level derived rather than assumed.
- **H4** (nuisance structure raises contention, so the gap grows under misspecification). Kept, as an argued mechanism (Section 7), not a theorem.

---

## 10. Predictions (pre-registered, falsifiable)

1. **NLL and calibration gap despite the accuracy tie.** SK $\leq$ SH in test NLL and expected calibration error beyond seed noise, largest for transformer and DKVMN. If NLL ties exactly, mechanism A as stated is wrong. (Cluster the test by data seed; effective replicates are ~5 per cell, not 25.)
2. **DKVMN memory-size sweep, two-part.** As `memory_size` grows toward $Q$, addressing competition sharpens and SH $b$ recovery declines further while the $b$ probe stays $\geq .97$ and SK is flat. As `memory_size` shrinks to 1, SH $b$ does not recover to the LSTM level, because the direct key-to-summary correction path is slot-independent. The second half is the surgical signature of the correction-path mechanism.
3. **Summary-key ablation.** Remove or stop-gradient the $\kappa_t$ concatenation in the DKVMN summary (read-only summary). Predicted: SH $b$ recovery rises substantially toward the LSTM level; $a$ recovery moves little or falls slightly. Directly tests the claim that the correction path is the dominant $b$-displacer.
4. **Gradient-geometry ordering.** The participation ratio (effective rank) of $\mathrm{Cov}_t(\partial L/\partial e_i)$ over training orders dkvmn $<$ lstm $\ll$ transformer, and within DKVMN the $b$-head direction's overlap with the top gradient subspace exceeds the $a$-head direction's. Measurable from checkpoints; carries the crowding and flip claims.
5. **Residual structure.** Regress per-item rank residuals of the SH heads on exposure $1/\sqrt{n_i}$ plus dynamics covariates (cluster/attention profile, co-occurrence centrality). SH shows significant dynamics-covariate terms (largest for DKVMN $b$ against slot-attention profiles); SK residuals depend on exposure only. This is the displacement field's fingerprint.
6. **Interventions on the Fisher weights.** Widening the ability distribution (population SD $0.5 \to 2$) raises $I_{aa}$ relative to $I_{bb}$, so SH $a$ recovery and $a$ probe rise while $b$ barely moves, and the transformer's $a$ absence is partially repaired. Extending $K$ ($3,5,7$) continues the observed 2PL$\to$GPCM trend in both deposit and recovery with diminishing returns, and raises the SK value-table $b$ trace (already $.449 \to .605$, $.329 \to .526$) while the $a$ trace stays near zero.
7. **Nonlinear-probe check on "absence."** An MLP probe on the transformer 2PL SH table stays low ($\lesssim .5$) for $\log a$. If it recovers $.7+$, regime 2 is nonlinear encoding rather than absence, and mechanism B needs restating as format mismatch rather than crowding.

(Note: an earlier draft carried a frozen-table head-refit prediction; it
is withdrawn on the author's standing ruling that refit constructs have
no place in this research — the model operates under a real-time
assumption. The mechanism separation above is carried by the probe
matrix and Predictions 1, 4, and 5.)

---

## 11. Honesty ledger

**Proved under stated assumptions.** The routing identities (exact, everywhere; architectural fact $\partial\hat\theta/\partial k \equiv 0$ verified in code). Proposition 1 (needs block stationarity, full-row-rank head Jacobian, no key weight decay). Proposition 2 and its corollary (needs $e$-block stationarity and a local quadratic expansion; exact in Model M). Propositions M1-M4 in Model M. The suppression lemma (unconditional) and the conditional $I_{aa} < I_{bb}$ ordering with its explicit inversion below $a_\star \approx 1$ (banked, verified numerically).

**Argued, with the mechanism-bearing statistic named.** The crowding ordering across architectures (Prediction 4 carries it). The identification of DKVMN's correction path as the dominant $b$-displacer (Predictions 2b and 3 carry it). Deposit strength proportional to route information (Prediction 6 carries it). The time-integration account of why deposits persist (M3 proves the subtraction step; persistence in the deep model is argued). The accuracy-tie argument (Prediction 1 exposes it). The misspecification battery account (Section 7, weakest).

**Conjecture.** That the displacement field's table-predictability is near-complete in general (measured $\approx$ complete here). That the SK residual $a$ trace is purely incidental correlation.

**Corrected or refuted by the data.** H2's strong internalization form (probe kills it). H1's capacity mechanism (replaced by the alignment contract). Any per-item location-gauge account of this study (excluded by the same contract; it remains the correct account of the archived MA-GPCM configuration, a deliberate design contrast worth one line in the paper).

**Known confounds and their status.** (i) SK's wider key is a capacity confound for raw recovery comparisons; the banked width sweep shows width closes roughly half the gap, and the probe data now assigns the width-sensitive share to mechanism B ($a$-side content) and the width-insensitive share to mechanism A ($b$-side alignment, Proposition 2a). Any defense should lead with the matched-capacity pair. (ii) All endpoint statements are quasi-stationary (early stopping); the 500-epoch trajectories show SH $a$ peaking mid-training then declining, consistent with continued drift along the balance manifold, but residual gradient norms were not measured. (iii) Ridge probes lower-bound information; "present" claims are safe at $.98$, "absent" claims are soft until Prediction 7 runs. (iv) The DKVMN probe rows are final at n=25 per cell (E8), and the purge residue ordering dkvmn $>$ lstm $>$ transformer stands with full samples. (v) Significance language must respect the fold clustering (5 seeds x 5 CV folds, effective n $\approx$ 5 per cell).

**Scope.** Everything here is local (near the trained point), conditional (the Fisher ordering), and within-model (Gauss-Newton equals Fisher only near a well-specified optimum; the real beds are misspecified by construction in the battery). The theory sets signs, orderings, and mechanisms, and it is corroborated by rank patterns and the probe matrix; it does not predict magnitudes, and the one magnitude-flavored quantity it does fix, the SK plateau level, follows from finite-sample arithmetic, not from dynamics.
