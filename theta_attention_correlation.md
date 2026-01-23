# Theta–Attention Correlation: What to Compute (Practical + Psychometrics-Meaningful)

This note operationalizes **“\(\theta\)–attention correlation”** for a multidimensional DKVMN-style model with a **polytomous IRT head** (e.g., GPCM/GRM). The goal is to avoid “pretty but ambiguous” heatmaps and instead compute quantities with clear measurement interpretation.

---

## 0) Notation (consistent with DKVMN–MIRT framing)

At interaction time \(t\):

- Item/exercise: \(q_t\)  
- Observed category: \(x_t\in\{0,\dots,K-1\}\)  
- Attention over memory slots (soft multi-KC): \(w_t\in\Delta^{N-1}\), where \(w_t(n)\ge 0\), \(\sum_n w_t(n)=1\)  
- Slot/value memory row (slot state): \(s_{n,t}\in\mathbb{R}^D\)  
- Trait used by the measurement head:
  \[
  \theta_t \;=\; \sum_{n=1}^N w_t(n)\, s_{n,t} \in \mathbb{R}^D
  \]
- Polytomous predictive distribution from the IRT head:
  \[
  \hat p_t(k) \;=\; P(X_t=k\mid \theta_t, q_t)
  \]
- Expected score (ordinal mean):
  \[
  \mu_t = \mathbb{E}[X_t]=\sum_{k=0}^{K-1} k\,\hat p_t(k)
  \]

If the head uses an MIRT-like linear predictor, define an item direction/discrimination vector \(a_{q_t}\in\mathbb{R}^D\) and score
\[
u_t = a_{q_t}^\top \theta_t \quad (\text{plus thresholds/difficulty depending on head})
\]

---

## 1) Primary diagnostic (recommended): Attention vs *actual contribution to the score*

This tests whether attention corresponds to what *actually drives the psychometric score*.

### 1.1 Slot contribution to score
Define the per-slot contribution:
\[
c_t(n) \;=\; w_t(n)\,\big(a_{q_t}^\top s_{n,t}\big).
\]

Interpretation:
- \(w_t(n)\): “relevance” assigned by the model.
- \(a_{q_t}^\top s_{n,t}\): how much that slot’s state supports the item direction.
- \(c_t(n)\): **effective influence** of slot \(n\) on measurement for \(q_t\).

### 1.2 Alignment statistic
Compute per-time-step **rank correlation**:
\[
\rho_t = \mathrm{Spearman}\big(w_t,\; |c_t|\big).
\]

Then summarize across time:
- median/mean \(\rho_t\)
- IQR or 5–95% range
- stratify by **single-KC vs multi-KC items**, item groups, and \(D\).

### 1.3 Distributional match (optional but strong)
Normalize contribution magnitudes:
\[
\tilde c_t(n)=\frac{|c_t(n)|}{\sum_m |c_t(m)|}
\]
and compute Jensen–Shannon divergence:
\[
\mathrm{JS}_t = \mathrm{JS}(w_t,\tilde c_t).
\]
Lower \(\mathrm{JS}_t\) indicates better attention–influence agreement.

**Why this is psychometrically meaningful:** it evaluates whether “where the model attends” matches **where the measurement model says the trait signal comes from**.

---

## 2) Gradient-based diagnostic: Attention vs likelihood-relevant dimensions (model-native)

This is especially defensible if you motivate your write/update as likelihood-consistent.

### 2.1 Likelihood gradient
Compute the gradient of the observed log-likelihood w.r.t. \(\theta_t\):
\[
g_t \;=\; \nabla_{\theta_t}\log P(x_t\mid \theta_t, q_t) \in \mathbb{R}^D.
\]

For GPCM/GRM this is readily available via autograd.

### 2.2 Gradient-aligned slot importance
Define slot importance:
\[
I_t(n) \;=\; w_t(n)\,\big| g_t^\top s_{n,t} \big|.
\]

Then compute:
- \(\mathrm{Spearman}(w_t, I_t)\)
- or JS divergence between \(w_t\) and \(I_t/\sum I_t\).

Interpretation:
- \(g_t\) is the “direction in trait space that increases likelihood of the observed category.”
- If attention is meaningful, high-attended slots should align with high \(I_t(n)\).

---

## 3) Sensitivity diagnostic (finite differences): Attention vs prediction leverage

Use this if extracting \(a_{q_t}\) is inconvenient or you want a head-agnostic check.

### 3.1 Predicted mean score sensitivity per slot
Define perturbed traits:
\[
\theta_t^{(n,+)} = \theta_t + \epsilon\, w_t(n)\, s_{n,t},\quad
\theta_t^{(n,-)} = \theta_t - \epsilon\, w_t(n)\, s_{n,t}.
\]
Then compute sensitivity:
\[
S_t(n) = \frac{\mu(\theta_t^{(n,+)})-\mu(\theta_t^{(n,-)})}{2\epsilon}.
\]

Now compute alignment:
- \(\mathrm{Spearman}(w_t, |S_t|)\)

This measures whether attended slots are truly “levers” on the model’s predicted score.

**Practical note:** expensive (2 forward passes per slot). Use a subset of \(t\) and top-\(M\) slots.

---

## 4) Special case (useful if \(D=N\)): Attention vs attended mastery

If you run the interpretable regime \(D=N\) and \(\theta_t\) is per-slot mastery:

\[
m_t = w_t^\top \theta_t.
\]

Then correlate \(m_t\) with:
- \(\mu_t\) (expected score)
- \(u_t\) (if defined)

This is intuitive but weaker than Section 1 because it collapses slot detail into one scalar.

---

## 5) Practical implementation details (to avoid noisy results)

### 5.1 Use rank-based correlation
Use **Spearman** rather than Pearson: \(w_t\) is sparse/heavy-tailed.

### 5.2 Top-\(M\) slots
Compute correlation over the top-\(M\) slots by attention (e.g., \(M=10\)) to avoid near-zero mass dominating.

### 5.3 Item-level aggregation
Compute an item-level alignment score:
\[
\rho_j = \mathrm{median}_{t: q_t=j} \rho_t.
\]
Then you can plot \(\rho_j\) vs item discrimination norm or item entropy.

### 5.4 Null baseline (highly recommended)
Shuffle \(w_t\) across slots (within \(t\)) and recompute \(\rho_t\). Report that your observed \(\rho_t\) distribution is significantly above the null.

---

## 6) What to plot (one figure that convinces reviewers)

Instead of a raw attention heatmap:

**Panel A (alignment distribution):** violin/histogram of \(\rho_t\) for:
- single-KC items
- multi-KC items

**Panel B (item-level structure):** scatter of \(\rho_j\) vs:
- \(\lVert a_j\rVert\) (discrimination magnitude) **or**
- attention entropy

This connects interpretability to **measurement-relevant structure**.

---

## 7) Recommended “first computation”

If you do only one thing, do **Section 1**:

1. Compute \(c_t(n)=w_t(n)(a_{q_t}^\top s_{n,t})\).  
2. Compute \(\rho_t=\mathrm{Spearman}(w_t,|c_t|)\).  
3. Report distribution of \(\rho_t\) and item-level medians \(\rho_j\), with a shuffled null comparison.

This is the cleanest operational definition of “theta–attention correlation” that remains psychometrically grounded.
