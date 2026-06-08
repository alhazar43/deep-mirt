# Archived, Job-Recommendation Direction

These documents capture the design and v1/v2 build of an attempt to
pair MA-IRT's deep IRT belief tracker with a DRL job recommender over
an O\*NET occupation pool. The direction was archived on 2026-06-04
because the synthetic preference field could not produce a within-
session climbing recommendation trajectory under 1D theta matching,
the simulator structure made the DRL value proposition untestable, and
no real outcome data was available to ground the recommendation reward.

The active direction is item recommendation on real KT datasets,
reframed as ExRec (Ozyurt et al. 2025, arXiv 2507.11060) but for
ordinal KT using MA-IRT. See the new plan under `docs/` once landed.

## Contents

| File | What it was |
|---|---|
| `drl_mairt_background.md` | Codex's general feasibility dossier on coupling deep IRT with a DRL recommender. Has broad context on POMDP framing, simulator risks, and CDM mapping that may inform the new direction. Worth referring back to. |
| `drl_mairt_recommender_plan.md` | Codex's full job-rec proposal. Deprecated. |
| `drl_mairt_synthesis.md` | Plan-level synthesis comparing Codex vs the 7-agent workflow on the job-rec direction. Deprecated. |
| `drl_mairt_evidence.md` | Literature-anchored evidence synthesis with four publishability hooks for the job-rec direction. Deprecated. |
| `drl_mairt_track_assessment.md` | Path A from the earlier triage: adaptive assessment with deep IRT belief. **Closest to the new direction.** Worth re-reading as a starting point. |
| `drl_mairt_track_recommendation.md` | Path B: theory-driven mapping for job/career recommendation. Deprecated. |
| `drl_mairt_plan_v1.md` | The canonical v1 implementation plan for the job-rec build. Deprecated. |
| `drl_mairt_brief.md` | The academic brief with v1 and v2 simulator results, the PPO + hybrid reward formulation. Deprecated. |

## What is still alive

- The progress log at `docs/drl_mairt_progress.md` is being repurposed
  as the running build log for the new direction.
- The rl/ subdirectory under deep-mirt/ has scaffolding (BeliefTracker,
  pool registration, RetrievalIndex, JobTower text encoder) and the
  v1/v2 synthetic data + plots. Some of it is reusable for item
  recommendation. The new direction's plan will identify what to keep
  and what to replace.
- The ma-irt online step API on `feat/online-step-api` is reusable as
  is. Item recommendation needs the same per-step belief surface.

## Diagnosis of why this direction stalled

Two structural failures in the v2 simulator made the DRL value
proposition unmeasurable.

1. The 1D theta with K=5 GPCM cumulative-tail preference function
   produces a job ranking that is theta-invariant. Specifically,
   `argsort_j P(y >= 3 | theta_u, lambda_u, delta_j)` reduces to
   `argsort_j (-delta_j)` for any (theta, lambda). The optimal top-K
   does not change as theta_hat sharpens. The CaRReL-style
   recommendation-over-time curve is flat by construction.
2. The preference field has no externally grounded validation source.
   Slate-lift and predictive log-likelihood rewards both depend on the
   same simulator that generated the data, so the policy is being
   trained against and evaluated by the same artifact. Any "win" is
   internal to the simulator and does not transfer.

Item recommendation on real KT datasets sidesteps both. Knowledge gain
or mastery improvement is computable from the same KT model that
trains the policy, but the data-generating process is real student
responses, not a simulator the policy can game.

## Reactivation

If the job-rec direction is reactivated later, start from
`drl_mairt_synthesis.md` and `drl_mairt_evidence.md` for the literature
context. The diagnosed structural failures are the gating items to
solve before any new build effort.
