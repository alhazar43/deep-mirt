# kt-mirt Ledger (changelog; expectation versus reality)

Rule: one entry per event or run. Record the expectation before the
run, the reality after, and a verdict from {supports, weak support,
mixed, null, refutes, blocked}. Negatives are recorded once and never
re-litigated. Hypotheses live in THINKING.md; this file tracks how
results move them.

## 2026-07-17 Program opened

- Brief received: multi-KC knowledge tracing with IRT as the
  explainability layer; goals G1 (signed cross-KC influence) and G2
  (growth beyond noise); kt-irt core migrates to a sideline package.
- Grill answered by user: validity-gated evidence bar; all seven real
  beds in scope; in-tree vendored core; full autonomy including HPC.
- P0 started: planning documents seeded; vendor agent dispatched to
  copy the kt-irt core into kt-mirt/ (target: tests green, no runtime
  import from deep_irt, kt-irt untouched).
- P1 started: research workflow launched (internal archaeology on the
  Q-MIRT and trajectory/learning-dynamics threads; external sweeps on
  PSI-KT, graph KT, interpretable KT, transfer/forgetting, growth
  methodology, dataset facts; adversarial verification of load-bearing
  claims; synthesis into an avenue map).
- Hypotheses H1-H4 registered in THINKING.md.

## 2026-07-17 Infrastructure de-risked (same day)

- HPC validated: connected to hpc-head1.ewi.utwente.nl as yuanw,
  installed a dedicated ed25519 key (passwordless auth confirmed,
  alias `ssh uthpc`); cluster home already holds a kt-irt clone.
  Password never written to disk; user advised to rotate it.
- Raw-data inventory (beyond kt-irt caches): EdNet-KT1/ per-user logs
  plus EdNet-Contents/contents/questions.csv (correct_answer, multi-KC
  `tags`, bundles -- supports NRM option tracing and multi-KC mapping);
  data/kdd/algebra_2008_2009_train.txt (KC(SubSkills)/KTracedSkills/
  Rules columns with per-KC opportunity counts); data/timss/ (R build
  chain + GPCM matrices); data/slam_raw/. Junyi, ASSISTments, XES3G5M,
  Eedi are the ones to acquire.

## 2026-07-17 P0 vendoring DONE

- Expectation: kt-irt core vendors cleanly, tests green, no runtime
  dependency on deep_irt.
- Reality: all 10 core files vendored from kt-irt @ df3aee1; zero
  surgery needed (core was self-contained); deps collapse to numpy +
  torch; 139 passed / 0 skipped / 0 failed including three NEW test
  files covering losses, anchoring, and Bradley-Terry (gaps in the
  source suite). Independent smoke by the orchestrator: LSTM+GPCM
  track/fit/recover and DKVMN+NRM fit all work. Interface summary with
  five per-KC attachment points in _planning/vendor_report.md.
- Verdict: supports (P0 complete).
- Note for the user: the untracked repo-root file
  kt_structured_response_heads_revision_plan.md disappeared during this
  session; neither the orchestrator nor the vendor agent touched it
  (agent command history audited). Its full text survives in the
  2026-07-17 conversation transcript if it needs recovering.

## 2026-07-17 P1 research sweep DONE

- Expectation: the sweep locates prior art for signed cross-KC
  influence and per-KC growth detection, and settles bed choices.
- Reality: 33 agents, 0 errors; avenue map delivered
  (_planning/research/avenue_map.md, six avenues + ranked order).
  Headlines: signed influence is near white space (one unread
  claimant, LTKT; PSI-KT is direction-only and its per-learner
  transfer trait is exactly the object our Gate B killed); readout
  auditing is confirmed white space; population growth anchor is 0.1
  log-odds per opportunity (PNAS 2023, confirmed at primary) but
  per-KC and individual-level validity tests exist nowhere; NO raw
  correct rate was located for any bed, so the saturation bet is
  unmeasured everywhere, which promotes bed triage to stage 0.
  Verification errata absorbed (Junyi 2020 Kaggle has NO prerequisite
  field, only junyi15 does; Deep-IRT external correlations are
  0.56/0.58/0.69; several venue corrections).
- Verdict: supports (the program's two bets remain open and novel;
  both now have concrete certified paths and named risks).
- Deviation from the map's ordering, with reason: the primary-text
  reads (LTKT, HawkesKT, two interpretability critiques, PSI-KT
  reviews) are pulled INTO stage 0 as blocking novelty checks rather
  than deferred, since A1/A2 framing depends on them and they are
  cheap.

## 2026-07-17 Stage 0, local bed triage DONE

- Expectation (from the trajectory-program verdict): local beds near
  saturation, per-KC growth headroom doubtful; decoupling unknown.
- Reality (computed from raw files, kt-mirt/_planning/triage/):
  - KDD Algebra 08-09 (full 8.9M steps): 82-86% correct, 37-52% of
    KCs saturated, BUT a clean model-free per-KC growth signal
    (first-attempt rate rises 12-19 points over opportunities 1-10;
    73-77% of KCs positive-sloped) and the deepest per-learner-KC
    density (median 8-11). Decoupling is KC-model-dependent:
    KTracedSkills passes (80% of top pairs clear 0.75), SubSkills
    fails (27%, top pair 0.01), Rules' 100% is inflated by a
    catch-all bug tag. Anchors comfortable. KC model of choice:
    KTracedSkills.
  - EdNet KT1 (4000-user sample): raw correct rate 65.2% (the feared
    ~80% number was model accuracy, not raw rate -- now resolved);
    only 2.1% of KCs saturated; growth curve 57%->70% over opp 1-10
    (82% of KCs positive); decoupling 87%; anchors moderate (40% of
    tags with >=3 pure items); but per-learner-KC density is thin
    (median 2 opportunities, 7.2% reach 20), so EdNet supports
    population-level growth reads, not individual rates.
  - TIMSS G8 USA: clean unsaturated single-session bed (mean score
    proportion 0.418, zero saturated items); growth structurally NA;
    role = static calibration / A5 anchor demos.
  - SLAM: raw data ABSENT locally (the .bin files are guestbook-gated
    download error payloads, verified by reading them); no KC tags
    anyway; deprioritized.
- Verdict: supports H1 at the population-passive level on BOTH growth
  beds (the per-KC escape from saturation is real in the raw data);
  the individual-rate question stays open and is exactly what A4's
  posture matrix must decide. KDD-KTracedSkills becomes the first
  real bed for BOTH G2 (density) and G1 (decoupling passes); EdNet
  becomes the population-corroboration bed (bundle confound still
  bars it from G1 causal reads).

## 2026-07-17 Stage 0, reads and acquisition probes DONE

Reads (notes in _planning/research/):
- LTKT read at primary (ltkt_read.md). It IS signed-transfer prior
  art: four edge types including negative, BUT sign is a fixed
  co-occurrence-heuristic label assigned before training, never
  learned, and validated by predictive ablation only (no ground
  truth, no nulls, no seed variance). CONSEQUENCE: A1/A2 novelty
  narrows to CERTIFIED sign (synthetic sign recovery, nulls,
  external reference, stability audit); a bare "negative transfer
  helps prediction" framing is LTKT's claim, not ours. Must-cite.
- HawkesKT read at primary (hawkeskt_read.md). Its excitation matrix
  is architecturally unconstrained (negative reachable) but sign is
  never validated (the CMI analysis is non-negative by construction;
  the expert check scores a softmax-collapsed positive quantity). A3
  stands as audit-is-the-contribution, with HawkesKT's kernel as the
  natural audit target.
- Interpretability critiques read at primary
  (interpretability_critiques_read.md). Ding and Larson 2021 forces
  TWO NEW MANDATORY battery arms: (i) an untrained/frozen-encoder
  null (a random core nearly matches trained DKT, so any readout
  must beat it), (ii) a single-KC-drill contamination probe (drilling
  one KC moves ALL of vanilla DKT's readouts; phantom cross-KC
  effects live here), plus an order-invariance stress test. Khalid
  et al. 2025 sets a scope boundary: we certify measurement
  validity, never pedagogical decision utility.
- PSI-KT referee record BLOCKED (Cloudflare Turnstile; documented in
  psikt_reviews_read.md with self-reported-limitations proxy).
Acquisitions:
- Eedi NeurIPS 2020: OPEN Azure blobs (no gate); CRITICAL FINDING:
  it carries NO misconception labels (the guide says so explicitly);
  the per-distractor MisconceptionId dataset is the SEPARATE 2024
  Kaggle competition (1,857 questions, 2,587 misconceptions, no
  interaction logs). A2's design must be revised: either derived
  misconception clusters on the 2020 logs, or the 2024 mapping as an
  auxiliary resource, or both; the original "labels on the logs"
  premise is dead.
- XES3G5M: fully de-risked (Drive file id, 356 MiB, MIT license,
  pyKT-protocol splits); download now running serialized.
- Blocked on user credentials: Junyi 2020 (Kaggle token), junyi15 +
  KDD Bridge-to-Algebra (PSLC DataShop account), Eedi-2024
  misconceptions (Kaggle token + competition rules).
- ASSISTments probe died on a network error; retry queued.
- Orchestration lesson recorded: five parallel large downloads
  saturated the uplink (~450 B/s each); downloads are SERIALIZED
  from now on (probe when quiet: 2.2 MB/s).

## 2026-07-18 Acquisitions LANDED (serialized rerun)

- XES3G5M: complete and tar-verified at 373,235,093 bytes (the
  373,528,576 figure read off Drive's interstitial was wrong; five
  byte-identical downloads settled it). 3,886 archive entries;
  kc_level/, question_level/, metadata/ with KC routes + embeddings.
- Eedi NeurIPS 2020: data.zip (656,787,242 B) and starter_kit.zip
  (79,651,393 B), both size-exact against Azure Content-Length.
- Junyi 2020 (Kaggle, user token): Log_Problem.csv 3.02 GB +
  Info_Content.csv + Info_UserData.csv.
- Eedi 2024 misconceptions: kaggle download 403s until the user
  accepts the competition rules in-browser (pending user).
- Next: triage extension over XES3G5M, Junyi 2020, Eedi 2020 with
  the same script family and metrics as the local-bed triage.

## 2026-07-18 User-gated items LANDED (user delivered)

- junyi15 (DataShop): junyi_Exercise_table.csv with a NON-EMPTY
  prerequisites column on 742 of 837 exercises (secondary sources
  feared 370/722 -- the expert graph is denser than advertised),
  plus both problem-log zips. The external reference for
  signed-transfer validation is REAL and on disk (data/junyi15/).
- KDD Cup bundle kddcup_challenge.tar.gz (740 MB) delivered;
  SHA1 verification + bridge_to_algebra_2006_2007 extraction
  running. Unlocks the near-1-to-1 KC challenge bed.
- PSI-KT referee record delivered as browser-saved review.pdf;
  extraction agent rewriting psikt_reviews_read.md from it.
- Remaining user item: ONLY the Eedi 2024 rules click on Kaggle.
- Post-extraction facts: KDD bundle SHA1 MATCHES the published
  checksum; it contains bridge_to_algebra_2008_2009 (5.7 GB train),
  not the 2006-2007 development set the avenue map cited -- same
  curriculum, larger scale; triage will measure its KC arity and
  decoupling directly (06-07 remains one extra download if ever
  needed). junyi15 fully extracted (2.67 GB problem log) INCLUDING
  relationship_annotation_{training,testing}.csv, the crowdsourced
  exercise-relationship judgments -- a second external reference
  beside the prerequisite graph.
