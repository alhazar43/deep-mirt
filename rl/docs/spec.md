# DRL-MAIRT v1 Spec Lock

Date: 2026-06-04
Status: locked

This file is the canonical contract of the eight decisions locked for DRL-MAIRT v1, the realtime interactive job recommender built on top of ma-irt's deep IRT belief tracker. Any code, test, or document that disagrees with the values below is wrong by definition and must be reconciled. For the full plan, rationale, and v2 roadmap see `docs/drl_mairt_plan_v1.md`.

```yaml
project:
  name: irtrec
  version: 0.1.0

decisions:
  D1_repo_placement: "subdir deep-mirt/rl/"
  D2_theta_dim: 1
  D3_job_pool_source: "O*NET 2024"
  D4_items_text: false
  D5_rating_fidelity: "binary"
  D6_decision_controller: "heuristic"
  D7_eval_simulator: "replay over synthetic users"
  D8_preference_model: "option_A_sigmoid_on_theta_minus_delta_j"
  D8_delta_j_source: "work_zone (z-scored)"

deferred_to_v2:
  - "bandit DecisionController (Section 10.1)"
  - "SNIPS off-policy evaluation (Section 10.2)"
  - "iEvaLM LLM simulator (Section 10.3)"
  - "Option B RIASEC preference model (Section 10.4)"
  - "2D MIRT (Section 10.5)"
  - "Cross-simulator robustness (Section 10.6)"
  - "Real data integration (Section 10.7)"
  - "LLM-as-orchestrator (Section 10.8)"
```
