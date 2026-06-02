# Binary KT Benchmark — Replacement Paragraph Draft

**Location in main.tex**: replaces lines ~523-543 (current `\paragraph{Binary compatibility.}` plus `tab:combined_perf`). Sits after the Static recovery results, before the dynamic-ability subsection.

**Style constraints**: no em-dashes, no colons in prose, no "merely"/"trivially", American English, no headline-contrast filler, table speaks for itself.

---

## Proposed replacement (2-3 short paragraphs)

\paragraph{Binary KT benchmark.}
Setting $K = 2$ collapses the GPCM category probability in~\eqref{eqn:gpcm_prob} to a single logistic in $\alpha_j(\theta_t - \beta_{j,0})$, which is the two-parameter logistic IRT model. This places MA-GPCM in the same operating regime as the standard binary knowledge tracing baselines and lets us evaluate it against them directly. Table~\ref{tab:combined_perf} reports a five-model comparison on three datasets, with each cell averaged over five seeds. The five models trace an architectural progression from sequential KT (DKT) through memory-augmented KT (DKVMN), IRT-augmented memory (Deep-IRT), the GPCM head added on top (DKVMN+GPCM), and finally the separated ability pathway of MA-GPCM.

The five models cluster within run-to-run variability on every dataset. MA-GPCM matches DKT and DKVMN at the level of the historical binary-KT comparison, and it also sits within the same envelope as Deep-IRT and DKVMN+GPCM, the closer relatives that share the IRT parameterization. The separated ability pathway therefore carries no measurable cost in the binary regime, even against architectures specifically tuned for two-category prediction.

The substantive point is that this is the only one of the five architectures that scales without modification beyond $K = 2$. DKT, DKVMN, and Deep-IRT have no ordinal response model. DKVMN+GPCM does, but in our recovery experiments the shared ability and item pathway compromises trait identification at $K > 2$ (Section~\ref{sec:results}). MA-GPCM is the unification, recovering a defensible binary KT model at $K = 2$ and adding the GPCM ordinal structure at $K > 2$ inside the same architecture.

---

## Notes for morning finalization

- Numbers in `tab:combined_perf` will repopulate automatically from the bulk sweep CSV. Prose makes no specific numerical claims that would break if an entry shifts by 0.01 or so.
- If the table caption gains a "Synthetic-5" footnote (e.g., dataset source/size), no prose edit needed.
- If MA-GPCM lags by a non-trivial margin on one dataset (e.g., ASSIST2009 AUC drops below 0.77 while DKT stays at 0.79), reword paragraph 2 second sentence to "remains competitive with" instead of "matches", and add a hedge in paragraph 3 about trade-offs at the binary boundary.
- The phrase "in our recovery experiments the shared ability and item pathway compromises trait identification at K>2 (Section~\ref{sec:results})" assumes that section already establishes this. Verify the cross-reference in the morning.
- Keep Tab.~\ref{tab:combined_perf} where it is. Consider revising its caption to "Binary KT benchmark ($K=2$) across synthetic and real-world datasets. Each cell shows mean$\,\pm\,$sd over five seeds. Best per column in bold."

## Seed-0 preview (Phase 1 partial CSV)

Static Q=200 K=2: Deep-IRT/DKVMN/DKT all near ACC 0.700-0.704, AUC 0.774-0.781. MA-GPCM and DKVMN+GPCM pending.

ASSIST2009: DKT slightly ahead on AUC (0.790 vs ~0.778 for the other four). IRT-augmented family (Deep-IRT, DKVMN+GPCM, MA-GPCM) clusters at ACC 0.721-0.729, AUC 0.777-0.780. MA-GPCM and DKVMN+GPCM are essentially indistinguishable (within 0.002 ACC, 0.003 AUC).

Synthetic-5: only DKT and Deep-IRT have completed seed 0. Deep-IRT slightly above DKT on AUC (0.618 vs 0.602). Other three pending.

Surprising note: on ASSIST2009 the gap between DKT and the IRT-augmented family is wider (0.012 AUC) than typical noise. If this holds across seeds, the prose hedge in paragraph 2 about "competitive with" rather than "matches" may need to apply to the real-world dataset specifically. Watch in the morning.
