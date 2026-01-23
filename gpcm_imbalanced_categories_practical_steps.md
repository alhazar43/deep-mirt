# Practical steps to avoid “50% in category 0” when simulating GPCM (e.g., in `mirt`)

## 1) Diagnose the cause (10–15 minutes)
- **Compute per-item category proportions** (not just overall). Often a subset of items causes the floor effect.
- **Check targeting**: compare your simulated \(\theta\) distribution to the item step locations.  
  - If most step locations sit above most \(\theta\), category 0 will dominate.
- **Check slope scale**: very high discriminations make the dominant category even more dominant.

**Deliverable:** a small table per item: `p(cat=0..K)`, mean score, and whether any category is < ~5% used.

---

## 2) Decide what “realistic” means for your use case
Pick one of these targets up front:
- **Psychometrically realistic**: categories can be imbalanced; the goal is *functioning* categories.
- **ML-friendly balanced**: you want roughly balanced labels for downstream classifiers (this is an engineering constraint, not a psychometric one).

Your next steps differ depending on which target you choose.

---

## 3) Generate item parameters in a *targeted* way (instead of i.i.d. priors)
A robust recipe is to decompose each item into:
- an **item location** \(b_i\) (overall difficulty)
- **step offsets** \(\delta_{ik}\) (relative category transitions)
- a **slope** \(a_i\)

### Recommended priors / ranges (good defaults)
- \(\theta \sim \mathcal{N}(0, 1)\) (or widen to \(\mathcal{N}(0, 1.3^2)\) if you want more use of upper categories)
- \(a_i\): moderate (avoid very large values)  
  - e.g., draw from a lognormal or truncated normal so most values are roughly **0.6–1.8**
- \(b_i\): spread across your \(\theta\) range  
  - e.g., **Uniform(-2, 2)** or Normal(0, 1.2) with truncation
- Step offsets \(\delta_{ik}\): ordered and moderately spaced  
  - e.g., base offsets like `[-1.5, -0.5, 0.5, 1.5]` + small noise, then **sort** to enforce ordering

**Key idea:** ensure the *step locations* \(b_i + \delta_{ik}\) overlap the bulk of \(\theta\).

---

## 4) Pre-screen items by **expected category usage** (accept–reject items)
Before simulating the full dataset:
1. Sample many \(\theta\) values from your population prior (e.g., 10k).
2. For each candidate item, compute the implied category probabilities and the **marginal** usage.
3. Reject/resample items that violate your minimum-usage rule.

### Simple screening rules
- Reject if `max_k P(X=k) > 0.50` (too peaked)
- Reject if any category has `P(X=k) < 0.03–0.05` (too sparse)

This step usually fixes the “half are zeros” issue without distorting the population.

---

## 5) Tune the levers if category 0 still dominates
Use these in order (most principled first):

1. **Shift item locations downward**  
   - Decrease \(b_i\) (or shift the whole \(b\) distribution left).
2. **Widen the population variance**  
   - \(\theta \sim \mathcal{N}(0, 1.3^2)\) or use a mixture (e.g., low/medium/high groups).
3. **Reduce discrimination**  
   - Cap large \(a_i\) values; high slopes amplify dominance.
4. **Adjust step spacing**  
   - If steps are too high or too clustered, you get underused middle/upper categories.

---

## 6) If you want ML-style balance, enforce it explicitly (and label it as such)
If you truly need balanced labels:
- **Post-stratify respondents**: oversample high-\(\theta\) individuals (changes the effective population).
- **Post-stratify item responses**: acceptance–rejection at the response level (distorts the IRT generative story).
- **Condition steps on target marginals**: choose step locations so that, under your \(\theta\), each category hits desired proportions.

**Recommendation:** prefer **item pre-screening** + **targeted \(b_i\)** over response-level balancing.

---

## 7) For real questionnaire handling: collapse categories when sparse
If (in real or simulated-but-realistic settings) some categories are rarely used:
- **Collapse adjacent categories** (e.g., merge 3&4, or 0&1) and re-fit.
- Treat it as a **scale design** issue: revise anchors, reduce categories, or retarget items.

---

## 8) Estimation stability in `mirt` when categories are sparse
When you fit and see unstable thresholds/SEs:
- **Constrain or simplify**: fix extreme step parameters, share slope structures, or reduce category count.
- Consider estimation options that are more robust to complex models (e.g., stochastic methods like `MHRM`), and use good starting values.
- If you need strong shrinkage, consider a Bayesian IRT workflow (outside `mirt`) or impose stronger constraints in `mirt`.

---

## Minimal workflow checklist
- [ ] Plot / tabulate category proportions per item  
- [ ] Ensure step locations overlap \(\theta\) mass (targeting)  
- [ ] Use moderate slopes  
- [ ] Pre-screen items by marginal category usage  
- [ ] Only if necessary: widen \(\theta\) or enforce balance explicitly  
- [ ] If sparse in fitting: collapse categories or constrain parameters  
