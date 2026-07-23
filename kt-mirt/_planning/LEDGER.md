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
- PSI-KT referee record recovered (OCR; the saved PDF had no text
  layer): scores 8/8/6/5, Accept spotlight. Sharpest objection:
  self-consistency metrics are not proof of interpretability
  ("correct by construction"); only the reactively-added behavioral
  held-out validation satisfied reviewers. NOTHING contradicts the
  kt-mirt design; the lesson is to FRONT-LOAD the certification
  battery in the eventual paper. Full record + pre-emption map in
  research/psikt_reviews_read.md.
- User rulings (2026-07-18): git policy = COMMIT + PUSH explicit
  kt-mirt paths at milestones (never data/, never weight blobs);
  autonomous run approved, ultracode standing. First checkpoint
  pushed: e95113e. Disk freed to 23 GB (redundant verified archives
  deleted, all redownloadable).
- Post-extraction facts: KDD bundle SHA1 MATCHES the published
  checksum; it contains bridge_to_algebra_2008_2009 (5.7 GB train),
  not the 2006-2007 development set the avenue map cited -- same
  curriculum, larger scale; triage will measure its KC arity and
  decoupling directly (06-07 remains one extra download if ever
  needed). junyi15 fully extracted (2.67 GB problem log) INCLUDING
  relationship_annotation_{training,testing}.csv, the crowdsourced
  exercise-relationship judgments -- a second external reference
  beside the prerequisite graph.

## 2026-07-18 A4 design v1.1 (pre-run review revision)

- Two independent reviews of design/a4_design.md v1 returned 8
  blocking, 18 important, 6 minor findings. All blocking items fixed
  in place BEFORE any run (permitted: thresholds may change now, never
  after runs begin; none loosened, several tightened). Headlines: ACT
  now certified on all four twins (CG1a positive control; CG1b
  mismatched-generator/laundering arm on SYN-NS, which gains a 20%
  silent-KC subset; CG1c saturation no-overshoot); a pre-registered
  ACT real-bed firing definition (RB-A); the shared existence gate's
  structural consequence written into the disagreement matrix;
  blockwise-growth bank calibration with hierarchical shrinkage for
  KDD's 1.31M-step bank plus an RB0 tri-spec bank-robustness audit
  (battery arm 10); the real-bed null rebuilt as an M0 parametric
  bootstrap (item structure preserved) replacing the i.i.d. slice-mean
  resample; penalized bounded Newton against separation; an Adam bank
  calibration compute line. Important items folded (validation-regime
  spec, CG4a/b split, cumulative truncation band, split-half tolerance
  0.15 -> 0.10, CG7-to-RB2 bridge, ragged kc_ids, BY-sensitivity
  reporting, XES3G5M own-density certification, build/compute/calendar
  re-estimates). Full dispositions in a4_design.md section 10; four
  rulings escalated to the program lead in section 11 (KDD item
  granularity, ACT decline asymmetry, EdNet Tier-2 ambition, budget).
- Verdict: supports (design hardened pre-registration; no run started).

## 2026-07-18 Rulings closed; API outage; stage 0 data COMPLETE

- The four escalated rulings decided by the orchestrator (reasoning
  in THINKING.md): step-plus-shrinkage; ACT stays growth-only with a
  recorded scope limit; EdNet Tier-2 cap kept; budget approved in
  full. Design v1.1 FROZEN; build workflow launched.
- Transient API outage killed three streams mid-flight; recovery:
  the triage extension had FINISHED its work first (all outputs on
  disk), the ASSISTments agent had landed the main file, the build
  workflow died before writing anything and was resumed clean.
- Triage extension results (report + JSONs in _planning/triage/):
  Eedi 2020 is the new decoupling leader (96.7% of top pairs clear
  0.75) and the A2 bed by a distance (median 49% of wrong answers
  pile on ONE distractor vs 33% flat baseline -- the misconception
  clustering signal is real); XES3G5M is middling (0.795 correct,
  median 2 opportunities, weak pooled growth 0.778->0.816);
  Junyi 2020 offers growth depth; Junyi topic-KCs never co-occur
  (strict tree) so decoupling is structurally undefined there.
  Verdict: no universal bed; choose per avenue, as the design does.
- ASSISTments 2009 corrected-collapsed VERIFIED on disk (346,861
  rows, canonical schema); stale 522 MB partial of the optional 2012
  set deleted; acquisition debris cleaned.
- Remaining stage-0 pieces: junyi15 + KDD Bridge 08-09 triage
  (running); Eedi 2024 rules click (user, optional).

## 2026-07-18 Stage 0 CLOSED: all nine beds triaged

- junyi15 (25.9M attempts, 247k users): the STRONGEST growth
  magnitude of all nine beds (pooled 0.626->0.815 over opp 1-10,
  97.4% of topics positive-sloped, median slope 0.0174), deep
  density (topic median 8, 33% reach 20+), only 12.8% of topics
  saturated. Caveat logged: within-topic difficulty sequencing could
  inflate the curve; the design's frozen-difficulty measurement
  layer is the machinery that separates this. Prerequisite graph:
  835 nodes, 981 edges, NOT a DAG (212+ nodes in cycles, 2
  self-loops) -- A1 validation must handle cycles. KEY
  METHODOLOGICAL CAUTION: volume-ranked prerequisite pairs are the
  worst-decoupled (6.7% clear 0.75) while the full 898-edge
  population clears 72.2% (median 0.851); A1 pair selection must
  sample across the decoupling distribution, never rank by volume.
- KDD Bridge 08-09 (20.0M steps): KTracedSkills holds arity 1.081
  and decoupling 80%, matching Algebra 08-09 -- KTracedSkills
  confirmed as the family-wide default; SubSkills corrupted by
  non-content tags (arity 1.55, decoupling 50%).
- Campaign consequence: junyi15 enters the A4 real-bed queue right
  after KDD (subject to the same RB licensing checks as any bed)
  and is independently the A1 external-reference bed. Bed choice
  stays per-avenue.
- Stage 0 is COMPLETE. Open user item: only the optional Eedi 2024
  rules click. Build workflow (P3) still running.

## 2026-07-18 Compute policy change (user directive)

- The user's other work occupies the local GPU and part of the CPU.
  Standing policy from now on: NOTHING in this program touches the
  local 4060. The build's end-to-end dry run is patched to CPU-only
  (CUDA_VISIBLE_DEVICES=""); the entire synthetic certification
  campaign (bank calibration 3-9 GPU-h + tripled ACT matrix 18-36
  GPU-h + neural trackers) moves to the UT HPC cluster under the
  2-GPU QOS cap, using the kt-irt autopilot pattern adapted for the
  kt-mirt harness. Local machine keeps only light CPU work (tests,
  small verification fits), throttled.
- Cluster prep queued for after the build gate: rsync kt-mirt/ to
  the cluster (avoids repo-credential questions; synthetic campaign
  needs no datasets), venv + pip -e + pytest on the login node,
  adapt slurm array + autopilot scripts for the A4 unit grid.

## 2026-07-18 Cluster probe: better than the runbook believed

- Association is now account bms-code / QOS research: MaxJobsPU 8
  (not 2), MaxSubmitPU 100, no listed TRES or wall cap. Whether a
  separate GPU-TRES cap still binds gets probed empirically at first
  submission (autopilot degrades gracefully either way).
- CPU partitions main-cpu and main are open (AllowAccounts=ALL): the
  campaign splits into GPU jobs (trackers, ACT amortizer, bank
  calibration) and CPU jobs (existence gates, MIX ladders,
  permutation/bootstrap farms, battery stats) that never touch GPU
  slots. Submitter design: lean ~30-min unit chains, autopilot
  top-up, coverage-first cell ordering, spare-core piggybacking
  inside GPU jobs.
- Cluster prep DONE: kt-mirt synced, venv with cu118 torch built,
  193/193 tests green on the login node (py3.10) -- the mid-build
  snapshot including stages 1-2 passes on the cluster.

## 2026-07-18 P3 build: code complete, execution pass running

- The five-stage build finished with a twist: the first stage-3
  agent (killed by the API outage) had already implemented ALL
  remaining modules (14 under src/kt_mirt/growth/, 11 test files);
  stages 3 and 4 became independent design-fidelity audits of that
  code (every cross-module call traced against frozen v1.1; no
  stubs, no TODO markers, suite green on the cluster). Stage 5
  returned prematurely mid-sync, so the EXECUTION deliverables
  (generator acceptance checks, tiny end-to-end dry run, final
  counts) are running now via a dedicated finisher agent. Local
  compute re-permitted by the user (render done); local 4060 will
  join the campaign as an auxiliary worker.

## 2026-07-18 P3 build EXECUTED and verified

- Suites: 397 passed / 0 failed locally AND on the cluster (identical
  counts). Generator acceptance: PASS on both full-scale profiles
  (non-KG twin "fails" are the twins' dynamics overrides working as
  designed; gate is scoped to SYN-KG per the frozen interpretation).
  Tiny end-to-end dry run: 16 cells, no crashes, quiet-side gates
  behave (ACT-P1 silent on no-growth, saturation refusal fires);
  power-hungry gates fail exactly as an underpowered wiring check
  should. Zero fixes needed.
- Smells logged from the dry run (not tuned away): (1) ACT-P0 shows
  a ~0.3 implied rise on EVERY twin including no-growth at tiny
  scale -- wrong-side failure, inverted from design expectation;
  medium-scale probe scheduled BEFORE the campaign (1 GPU-h to
  protect 40). (2) CG7 sign-flip and CG10 at chance -- plausibly
  tiny-scale noise, probe covers them. (3) r_c_se floor clamp makes
  r_c_z meaningless -- report-only field, never to be quoted. (4)
  BH discoveries structurally impossible at dry-run replicate
  counts -- fine, irrelevant at B=199/999. (5) RunConfig still
  hard-pins CPU from the render ban -- to be lifted deliberately
  with a test, not as a drive-by.
- Next: campaign-prep agent (device-guard lift, ACT-P0 probe on the
  4060, slurm layer with GPU/CPU job split under bms-code/research,
  empirical concurrency probe, local worker). Full campaign waits on
  the probe verdict and my go.

## 2026-07-18 Probe verdict, campaign GO (rulings by orchestrator)

- ACT-P0 probe (C=64, N=2000, full epochs, 2x2 seeds, 4060): the
  fabrication is SCALE-PERSISTENT -- SYN-NG rise 0.246 (25x the
  silence bar) and SYN-KG reads 0.20, the same amplitude as
  no-growth, so the P0 read is fabrication-dominated. ACT-P1 is
  silent on NG (0.00067) and jumps ~20x on KG (0.013). RULING:
  ACT-P0 EXCLUDED from the campaign pending a code-level diagnosis
  (parallel agent); exclusion is a pre-campaign revision (no
  certification runs begun), reversible via the idempotent store.
  ACT-P1 carries the active posture.
- Smoke: 4 GPU + 2 CPU chains fully concurrent; the empirical limit
  is MaxJobsPU=8 TOTAL (no separate GPU cap) -- the guest-era 2-GPU
  cap is gone. Smoke also caught 4 unthreaded device call sites in
  battery.py (fixed, 402/402 green both machines) -- the smoke
  earning its keep.
- Campaign launch plan: phase 0 = two production-scale single-unit
  timing jobs (GPU + CPU, generous walltime) to size chains; then
  autopilot with <=8 in flight, GPU-heavy mix, ACT-P0 cells masked;
  local 4060 worker on a partitioned id space; results pulled home
  periodically; coverage-first ordering throughout.

## 2026-07-18 ACT-P0 mechanism found; repair ruled

- Diagnosis (research/act_p0_diagnosis.md): the fabrication is a
  TRAINING-CONVERGENCE pathology, not structure. softplus(0)~0.69
  gain init saturates the ceiling gap by itself; train_active runs a
  bare 20-epoch loop with no convergence check, so the gain never
  descends from its fabricating start. Ablations removing ALL
  structural suspects at once still fabricate (0.65 rise); epochs
  200/1000/2000 collapse it monotonically 0.046/0.014/0.009; at 1000
  epochs the known-growth gain recovers 0.154 vs true 0.15. The
  battery's no-growth twin caught a real defect the prediction
  metrics never would -- the program's own thesis, demonstrated on
  its own code.
- RULING: repair, not retire. Convergence-gated stopping mirroring
  bank.calibrate_bank's dual criterion; no design constraint
  touched. CONSEQUENCE: ACT-P1 shares the same loop, so ALL ACT
  cells must run on the repaired trainer; any ACT cells completed
  under the old trainer are invalidated and rerun (idempotent
  store). Non-ACT cells (slice/gate/tracker/battery) are unaffected.
  Cluster swap coordinated by the orchestrator after the launcher
  hands off.

## 2026-07-18 evening: user AFK, autonomous overnight mode

- User directives in force: keep the campaign running, log
  everything; local 4060 granted for overnight campaign use where it
  clears bottlenecks (a neural-unit worker joins alongside the
  CPU-unit worker; every neural chain absorbed locally frees a
  cluster slot for CPU chains, the long pole).
- Watchdog armed: a persistent 25-minute-poll monitor alerts on
  store stagnation with an empty cluster queue, repeated ssh
  failures, or failed-unit accumulation.
- In flight at hand-off: campaign launcher (timing calibration done
  for CPU ~40-60 min/unit; driving autopilot + both local workers to
  a running hand-off), ACT trainer repair (stationarity-based
  convergence calibration; the tolerance choice is bound to
  stationarity diagnostics, never to the silence bar). ACT cells run
  last, on the repaired trainer only; stale ACT results invalidated
  before rerun.
- Standing overnight loop: agent reports -> ledger entries ->
  checkpoint commits pushed at milestones -> parked agents nudged
  (the end-turn-while-waiting failure pattern has hit four times;
  every notification gets checked for it).

## 2026-07-18 night: ACT repair verified; converged truth splits the gates

- Repair landed (stationarity-based, never bar-tuned): windowed-mean
  NLL criterion + growth-param drift guard, floor 3000-epoch ceiling;
  406/406 tests incl. a matched-family regression (g_c recovers
  0.146-0.164 vs true 0.15). Epochs-to-convergence 652-1942 (~59x the
  old 20-epoch cost; CPU 8-32 min/fit, GPU proportionally cheap).
- The honest converged picture at half-scale probe (C=32, N=1000):
  P0 NG mean 0.0033 PASSES the mean-silence bar but p95 ~0.022 FAILS
  the p95 clause (real estimator property); P0 KG detection collapses
  (0.038 vs true 0.667, RB-A fail; rank corr 0.66 survives). P1 NG
  silence DEGRADES at convergence (0.012/0.054, both clauses fail-ish);
  P1 KG fires at 0.055-0.058 but magnitude ratio 0.086x vs CG1a's
  [0.5,1.5] corridor. Family-matched toy recovers gains exactly, so
  the magnitude loss is gain-form mismatch expressing as
  UNDER-detection (Lemma 3's other face).
- EPISTEMIC HEADLINE: pre-fix P0 fabrication AND pre-fix P1 silence
  were BOTH optimization artifacts -- 20-epoch reads were noise in
  opposite directions. Paper-grade material for the certification
  story regardless of ACT's final verdict.
- RULINGS: (1) repair ACCEPTED and committed. (2) The P0 exclusion is
  SUPERSEDED: BOTH ACT variants enter the campaign at production
  scale on the repaired trainer, symmetric treatment, and the
  pre-registered gates rule -- half-scale CPU probe evidence cannot
  adjudicate production-scale thresholds (power differs), and the
  design's posture matrix wants the active posture answered formally,
  not by probe fiat. (3) G2 does not hinge on ACT: passive and mixed
  carry detection; ACT failing its own gates at production scale is a
  legitimate, reportable posture-matrix outcome. (4) Cluster package
  swap happens BETWEEN chain generations (launcher-coordinated, no
  mid-run tree replacement), then ACT units unmask.

## 2026-07-19 ~01:00 Campaign LAUNCHED (hand-off report logged)

- Running: 6 GPU + 2 CPU chains under autopilot (detached on
  hpc-head1, PID + stop/relaunch recipes in the hand-off), local CPU
  slice worker + local 4060 neural worker (KDD-sized cells only;
  EdNet does not fit 8 GB). 3 KDD neural cells complete and pulled
  home within the first hour; ACT swap done in a clean window
  (cluster suite 406/406 post-swap), both ACT variants unmasked,
  3 stale pre-swap cells deleted for recompute.
- Timing truth: post-repair neural unit 14.2 min (rtx-6000; syn_kg
  ~28 min on a40); slice unit EXCEEDS 2 h at B=999/199 (CPU chains
  resized to 1 unit / 240 min); local 4060 neural unit 47 min.
- Launcher fixed two launch-blockers (RunConfig.run_act_p0 mask
  mechanism; autopilot kind-position semantics that would have
  silently skipped half the neural pool and units 48-63) plus UPT
  drift, a walltime printf bug, a CPU-widening deadlock, and a
  quoted-tilde bug in pull_results.sh. Suite 406/406 throughout.
- OPEN DEFECT -> RULING: EdNet neural cells OOM on 44 GiB a40s
  (whole-cohort backward >50 GiB). Ruled: (1) probe whether the
  a100 pool has 80 GB cards and fits one EdNet unit as-is; (2) only
  if that fails, thread the vendored core's existing batch_size
  through the tracker path -- an implementation-level change (the
  design fixes estimators, not batch sizes), tests + one verified
  EdNet unit before unmasking the rest; profile concessions
  rejected. No invalidation needed (no EdNet cell ever completed).
- Projections at hand-off: KDD neural complete ~1-2 h in; slice
  coverage 5-7 h; full reachable scope (52/64 units) by midday;
  EdNet's 12 units follow the OOM ruling.

## 2026-07-19 ~09:30 Windows-update reboot: damage and recovery

- Local machine rebooted 02:41 (Windows update): both local workers,
  the watchdog, and the EdNet-resolution agent's process died. The
  cluster never blinked: autopilot ran ~6 h unattended, 8 CPU chains
  alive, KDD neural grid COMPLETE (12/12 cells), one EdNet neural
  cell complete.
- The dead agent's work survived and validates: it went to step 2
  (mini-batching through the tracker path; a100 probe job got
  cancelled unrun), synced the change to the cluster (trees
  byte-identical), and the completed EdNet syn_ng cell is production
  proof. Local validation post-reboot: 54/54 on the modified test
  files; full suite re-running in background. Batching committed as
  the sanctioned implementation-level change (KDD path unchanged by
  default-None batch_size).
- Recovery executed: autopilot restarted (GPU track now feeds the
  remaining 11 EdNet units), both local workers relaunched on their
  original partitions, results pulled home, watchdog v2 armed (adds
  autopilot-process check; failure alerts only on count increases).
- Damage total: a few hours of local throughput and a delayed EdNet
  unmask. The campaign spine never stopped. Diagnostic scratch
  (_diag_*.py) removed after serving; the regression test carries
  the evidence.

## 2026-07-19 Reboot root cause corrected; updates paused (user order)

- Event-log verdict: the 02:41 reboot WAS the Windows servicing
  stack -- MoUsoCoreWorker initiated at 02:35, TrustedInstaller
  drove two further clean kernel-API reboots by 02:41 (a deferred
  update cashing in its restart; history shows the install date,
  not the reboot date). NEGATIVES that matter: no BSOD, no GPU
  driver TDR, no WHEA, no thermal/power event -- the overnight
  4060 worker at 7947/8188 MiB did NOT destabilize the machine.
- Windows updates paused through 2026-07-25 at the user's order
  (registry-verified). The campaign's reboot-recovery machinery
  remains armed regardless.

## 2026-07-19 ~06:40 Local workers restored; 4060 unblocks EdNet

- Full suite post-batching: 413 passed / 0 failed.
- Worker-restart lessons (operational, for future recoveries): MSYS
  nohup does not survive the launching shell's teardown -- detach
  via PowerShell Start-Process on a launcher SCRIPT (inline
  -ArgumentList quoting mangles payloads); launcher scripts must
  activate the research conda env themselves (a bare bash.exe
  resolves the wrong python and the enumeration pipeline dies with
  an empty-JSON error).
- Bottleneck fix: cluster GPU chains for the ~10 remaining EdNet
  units are queue-blocked behind 2.5-4 h CPU chains at the 8-job
  cap. Pulled results home (local store synced at 13), widened the
  local GPU worker to the full neural id space (mod 1) -- skip-done
  makes cluster/local overlap wasteful-not-wrong, and the 4060 now
  eats EdNet units (batched path) while cluster GPU chains wait.
  Verified: unit 12 skipped as already_done, unit 13 training.
- State: cluster 8 chains + autopilot + pending top-up; local CPU
  worker on slice units, local GPU worker on EdNet neural; watchdog
  v2 armed; updates paused through 07-25.

## 2026-07-19 ~11:00 Watchdog alert triaged: stale markers, cleared

- The failed-count rise (8->12) was STALE: units 20-23 OOM'd in the
  window between the ACT swap and the later batching sync
  (pre-batching whole-cohort attempts), pulled home afterward. Post
  -sync the cluster has not re-attempted EdNet (GPU chain pending
  behind CPU chains at the QOS cap) while the local 4060 grinds
  EdNet units on the batched path (96% util inside 8 GB).
- Cleared ALL stale EdNet failure markers on both stores (12 files)
  so counters stay meaningful and nothing can read them as
  terminal. Real-bed bridge workflow running in parallel (build ->
  hostile review -> fix; three agents, token-lean).

## 2026-07-19 Real-bed bridge BUILT (workflow: build -> hostile review -> fix)

- New modules: qmatrix.py (expansion policy, ragged Q-matrix,
  pure-anchor stats, circularity guard) and kc_data.py (chunked KDD
  loader producing LearnerLogs + 3-level hierarchy), plus additive
  run.py wiring (--profile kdd_real through the SAME measurement
  layer; synthetic path proven byte-equivalent by result hash).
- The hostile review earned its keep: the loader-vs-triage agreement
  table's EXACT matches on two fields were a COINCIDENCE (different
  row-universe scoping that happened not to overlap on this file) --
  fixed to triage's exact scope with a fixture exercising the
  overlap case. Also fixed: silent NaN row-number casting, missing
  student-id/timestamp validation (a missing id could have minted a
  bogus learner). Honest caveat retained in the notes: most of the
  agreement table is parser self-consistency, not ground truth; the
  independent checks are the learner count and the anchor stats.
- pandas added to runtime deps (the packaging gap the review
  flagged). Bridge tests 36/36; suite collects 454.
- Consequence: the KDD real-bed pilot can start the moment synthetic
  certification licenses it -- no build gap between verdict and
  real-data work.

## 2026-07-19 ~15:40 Slice-chain timeout crisis; neural pool COMPLETE

- NEURAL POOL DONE: 24/24 cells (12 KDD + 12 EdNet), zero failures.
  The local 4060 cleared the entire EdNet backlog on the batched
  path and exited cleanly; local GPU idle again.
- CRISIS: slice units exceed the 240-min chain walltime -- the
  previous CPU generation died 4x TIMEOUT at 04:00 with zero banked
  work, the current 8 will follow, and the autopilot would loop
  forever. Root cause: the timing calibration EXTRAPOLATED (its own
  probe hit a 2-h cap; "2h + margin" was a guess) -- the local
  ground-truth unit is 9+ h at 6 threads and still running. B=999
  is frozen; the fix is scheduling only. Repair agent dispatched:
  measure live core/RSS usage, pack concurrent units per chain if
  thread-starved, raise walltime to measured-safe, scancel the
  doomed generation, restart autopilot, project the real ETA.
- Consequence for the verdict timeline: slice-dependent gates slip
  (ETA from the repair measurements); the NEURAL-based gates (ACT
  CG1 family, tracker CG7-CG9) are computable NOW from the complete
  pool -- partial verdict read next.

## 2026-07-19 evening: user override -- cluster never waits on local

- User correction (now standing policy, joins the runner-check rule):
  the cluster must never idle while non-blocked work exists;
  generous walltimes replace measured ones whenever a measurement
  would serialize the pipeline (a finished job releases early; only
  an undersized wall wastes). My error: pausing the cluster to wait
  for the local GPU trial's sizing number.
- Relaunch ordered: 6 GPU slice chains (--device cuda, -t 360,
  ampere/lovelace pools; carries the gate.py device fix) + 2 CPU
  insurance chains (-t 1440) = the full 8-slot budget; autopilot
  restarted in this configuration; the local trial demoted to a
  bonus data point. Monitoring is passive-only (trial completion
  task, hang detector, cluster watchdog v5); ad-hoc probing ended.
- Also this hour: unit-7 CPU worker killed by ruling (13 h sunk vs
  ~2 h GPU redo); my kill sweep over-matched and took the first GPU
  trial + watchdog with it (restarted both; precision rule saved to
  memory: enumerate targets, never pattern-sweep).

## 2026-07-19 night: login relaunch fallout; relaunch staged, ssh blocked

- The user's login relaunch killed the repair agent (two 403 deaths
  mid-checklist), the local GPU trial, and both monitors. Ruling:
  the orchestrator executes the cluster relaunch DIRECTLY -- the
  chain interface is now fully known (chain_runner.sbatch: UNIT_KIND
  positions, UNITS_PER_TASK chains, CLI -t overrides the header).
  Planned submission: 6 GPU slice chains, UNIT_KIND=slice,
  UNITS_PER_TASK=7, -t 720, spread over l40/a40/a100 by free count
  -- covers all 40 positions in one unattended generation; local
  machine stays quiet (trial not restarted; generous walltimes made
  its sizing number moot).
- BLOCKED at the last step: ssh to hpc-head1 times out while ping
  succeeds -- the university VPN did not survive the relogin. A
  reachability tripwire (4-min checks, single event) fires the
  moment port 22 opens; the relaunch then executes immediately.
- User action needed: restore the VPN connection.

## 2026-07-19 ~22:20 Campaign RELAUNCHED direct; heartbeat monitoring

- VPN restored; head2 host key accepted as standing fallback (alias
  uthpc2 -- head1 gets crowded). Source synced with the gate device
  fix (verified present cluster-side); head-node usage kept light
  (no suite runs, import-check only) per user note.
- Submitted DIRECTLY (no agent): jobs 540534 (ampere, tasks 0-2) +
  540535 (lovelace, tasks 3-5), UNIT_KIND=slice, UNITS_PER_TASK=7,
  -t 720 -- 6 GPU chains covering all 40 slice positions in one
  generation; 5 RUNNING within 30 s (2 ampere + 3 lovelace), 1
  pending on Resources; cuda "available True" confirmed in 4 chain
  logs; unit 0 computing. Minimal pool separation per user
  directive; 2 QOS slots left free.
- Local 4060 re-enlisted BY USER for slice units: worker on mod-8
  remainder-3 partition (--device cuda), unit 3 started.
- Monitoring per user spec -- active periodic, not passive: 30-min
  heartbeat emitting a status line EVERY cycle (slices done, chain
  states, per-chain LOG GROWTH as the hung-vs-slow discriminator,
  local worker + GPU liveness, failure count) with explicit warns on
  static logs and GPU-idle-while-alive. This replaces both the
  silent watchdog and ad-hoc probing.

## 2026-07-20 ~01:00 HB-WARN investigated: slow, not hung

- The damped warn fired (60 min, all logs stale, zero completions).
  Runner-state diagnosis per contract: the local unit advances
  (CPU-time delta 0.73 min per 45 s, GPU duty-cycling 10-98%) -- the
  profile of the B=999 permutation battery: GPU-accelerated fits
  inside a SERIAL replicate loop. Cluster chains share the shape.
  Slow, not hung; no intervention tonight.
- Revised arithmetic: GPU slice units land near 2.5-3 h, so 7-unit
  chains hit their 12-h walls around unit 4-5. Expected by morning:
  ~25-30 of 40 slices banked (store banks per-unit; walls lose only
  the in-flight unit); one top-up generation finishes the pool
  tomorrow; verdict tomorrow evening. Optimization noted for the
  ledger, NOT tonight: parallelizing the permutation replicate
  orchestration would collapse unit times, worth one lean pass
  before any future campaign (junyi15 real-bed will reuse this
  battery).

## 2026-07-20 ~03:00 Root cause of slice slowness FOUND (py-spy)

- Stack dump of the live unit: penalized_bounded_newton computes
  Hessians via nested vmap/jvp/vjp in EAGER functorch -- every op
  routes through torch._refs Python dispatch, and the permutation
  loop re-pays that overhead per replicate (B=999). Dispatch-bound,
  not compute-bound: explains GPU~=CPU unit times, low GPU duty
  cycle, and the 13-h CPU unit. ~190 GPU-h at stake on the
  remaining pool.
- Fix ruled (design-neutral, statistics identical): batch permutation
  replicates INTO the vmap width (one dispatch for all 999), with an
  EQUIVALENCE GATE -- same seeds must reproduce current statistics
  within float tolerance on small configs for every consumer of the
  solver (PAS-G, MIX), full suite green, honest before/after timing
  on one real-scale cell. Chains keep running meanwhile (they bank
  per unit; zero waste from letting them grind). Swap only after
  equivalence passes, at a chain-generation boundary.

## 2026-07-20 ~03:45 Batched engine SWAPPED IN; generation 3 running

- Equivalence verdict ACCEPTED: all gate-consumed quantities
  (p-values, BH/BY flags) reproduce EXACTLY; raw-stat drift ~1e-3
  attributed to pre-existing float32 solve kernels (proven by a
  float64 self-recompute control), not to batching. Old loop path
  preserved behind use_batched=False; determinism of permutation
  draws bit-identical. Suite 471/471. Commit 6b579ac.
- Swap executed at a clean boundary: generation-2 chains cancelled
  (zero banked units lost -- nothing had completed), local unit
  killed by explicit PID (18564). Synced; batched path verified
  importable cluster-side; generation 3 submitted IDENTICAL
  configuration (6 GPU chains, UPT=7, -t 720, now heavily oversized
  walls) -- all 6 RUNNING within 25 s; local worker relaunched on
  unit 3 (90% GPU).
- The number that matters next: first post-fix unit completion time.
  Dispatch arithmetic projects minutes-to-tens-of-minutes per unit;
  the heartbeat's completion counter now tells the truth directly.

## 2026-07-20 ~04:45 Generation 3 FAILED at scale: two memory bugs

- 11 units OOM'd on 44 GiB a40s with two signatures: one absurd
  313 GiB single allocation (chunk estimator ignores the dominant
  data-tensor/autodiff term) and death-by-accumulation (42+ GiB
  retained across chunks -- missing per-chunk cleanup). Small-config
  equivalence was necessary but NOT sufficient; lesson absorbed into
  the bar: no engine change ships again without ONE real
  production-scale unit completing on cluster hardware.
- Generation cancelled (nothing banked, nothing lost but queue
  time); local 4060 unit left running as a data point (smaller
  auto-chunks may survive there). Fix agent resumed with both
  diagnoses, empirical chunk calibration + per-chunk cleanup
  required, and the raised verification bar; full-generation
  resubmit is gated on its production-unit proof.

## 2026-07-20 ~09:15 Morning state: analytic-derivative surgery commissioned

- Overnight verdict on the batched engine: NECESSARY BUT INSUFFICIENT.
  py-spy on the live local unit pins the remaining cost inside the
  Newton ITERATION loop -- eager functorch jvp/vjp Hessians pay
  Python dispatch per iteration; batching only amortized per-call
  overhead. Both cluster verification attempts died at their 2 h
  walls (sacct: TIMEOUT/CANCELLED); the local batched unit ran 6.5 h
  unfinished. The optimization agent went silent inside a multi-hour
  tool call and was formally stopped after its queued messages could
  never deliver.
- Cleanup: verification job cancelled; local worker killed properly
  this time (wrapper + child, enumerated PIDs -- several stale
  wrappers from successive relaunches were found and removed);
  local machine and cluster both idle by intent.
- THIRD surgery commissioned (fresh agent): ANALYTIC gradient +
  Hessian for the penalized-logistic objectives (closed-form
  einsums, no functorch in the hot path; generic path kept as
  reference), stage-timing instrumentation in unit logs, and the
  full verification ladder ending in the raised bar -- one
  production unit COMPLETING on an a40 before any resubmission.
  Every failure mode from tonight is baked into the brief.
- Slice pool remains 0/40. Neural pool (24/24) and all
  non-slice-dependent verdict inputs stay banked and safe.

## 2026-07-20 ~13:30 Analytic surgery verified; TRUE bottleneck: oversized KCs

- The analytic path is correct and fast (10.7x on its scope; full
  suite 504/504; trajectory equivalence exact at the production
  iteration cap). My dispatch-fallback hypothesis was REFUTED by the
  agent with direct instrumentation (3/3 identity checks engage) --
  recorded as the night's second lesson in hypothesis humility.
- Real cause, profiled: KC-joint pooled fits give each slice a free
  intercept, so 1700-3000-slice KCs yield P~2000 parameter vectors;
  the dense Newton solve pays O(P^3) per iteration, hits the 25-iter
  cap, forces replicate chunks down to ~4, and costs 50-90 min PER
  oversized KC in the battery. Orthogonal to derivative computation;
  explains every timeout of the last 36 hours.
- RULING: the KC-joint Hessian is an ARROW matrix (block-diagonal
  per-slice intercepts + small shared border). Commissioning an
  exact Schur-complement block solve: identical iterates and
  statistics BY CONSTRUCTION (block elimination is exact algebra),
  ~1000x on the giants, memory drops from S^2 to S*k so chunks
  recover. Execution strategy, not design change; the genuinely
  design-adjacent alternatives (walltime acceptance, cap change,
  reparameterization) are NOT taken. Fourth surgery, same
  verification ladder, same production-proof bar.

## 2026-07-20 ~17:30 The TRUE dominant cost: single-core permutation assembly

- Node-level forensics on the Schur proof unit (srun overlap into the
  allocation; py-spy blocked by ptrace policy): unit python at 101%
  CPU (ONE core), 23 GB RSS, its GPU at 0% through the battery
  stage. The battery was never solve-bound: per-replicate permutation
  DATA ASSEMBLY runs single-threaded in Python/numpy, serially
  across replicates x 119k slices, off-device. All three solver
  surgeries (replicate batching, analytic derivatives, Schur arrow
  solve) were real improvements hiding this layer beneath them.
- Agent redirected with a PROFILE-FIRST mandate: instrument the
  battery's internal phases, measure locally at production scale in
  one run, then fix all hot phases in one pass (precomputed
  permutation index tensors, batched on-device gathers for whole
  chunks, no per-replicate Python loops), same equivalence and
  production-proof bar. Lesson logged: sequential single-layer
  optimization against an unprofiled pipeline finds layers one
  timeout at a time; instrument-then-measure should have led.

## 2026-07-20 evening: USER RESET -- re-sort + first partial verdict DELIVERED

- User stopped the firefighting to ask for the bigger picture. Correct
  call. Re-read the three docs + inspected the store: found 24 banked
  production neural cells never aggregated. Mis-measurement exposed
  (progress != slices/40; the deliverable is verdicts). Strategic
  lesson written to THINKING; task order re-sorted in PLAN.
- FIRST PARTIAL VERDICT (partial_verdict_neural.md), from banked data,
  no new compute: ACTIVE posture certifies as a direction-detector,
  not a magnitude-estimator (NG silent ~0.0001-0.009; KG fires but
  undershoots true rise ~5-10x on both profiles; SAT correctly
  near-silent). Neural tracker PAS-N1 FAILS all four audit gates on
  every twin (CG7 untrained-null 0/3, CG8 drill-contam 0/3, CG9
  order-inv 0/3 kdd, CG10 direction ~0.4 near-chance). The
  faithfulness-audit half of the thesis is now evidenced at
  production scale.
- STILL PENDING (the G2 headline, needs slices): PAS-G model-free
  existence gate + MIX ladder -- "does per-KC growth clear noise".
- Re-sort: EdNet-matched slices FIRST (C=189, small KCs, sidesteps the
  oversized-KC O(P^3) cost); KDD-matched behind the assembly fix as a
  parallel track. One-unit EdNet-slice validation LAUNCHED locally to
  test the sidestep empirically before committing the re-sort.

## 2026-07-20 ~10:30 Assembly fix WORKS; cluster fanned out; tail deferred

- Info-first (per the lesson) before acting: the assembly agent had
  ALREADY synced a working fix to the cluster. Proof: KDD slice unit
  540782 ran bank(30s)+slice_fits(56s)+permutation_battery(27min,
  analytic path 12/12 no fallback) -- the battery that was UNBOUNDED
  now completes. Fix committed locally cd725e0 (99 targeted tests
  green), preserving the agent's work; agent stopped cleanly.
- NEW tail found, NOT chased: a post-battery CPU stage (~356% CPU,
  14GB, GPU idle) grinds ~15-40min before banking. DECISION per the
  strategic lesson: do NOT rabbit-hole it. KDD units still complete
  (~1h total); EdNet faster. Get the VERDICT, not the last 2x.
- ACTED on user's "cluster = outsourced workers, spare the laptop":
  local canary killed (laptop freed); fanned out the SLICE COVERAGE
  PASS (positions 0-7 = seed0 of all 8 profile x twin cells) to
  cluster GPU (jobs 540807/540808, 4 chains, -t 240) alongside the
  proof unit. 5 cluster jobs busy, zero laptop load.
- Watchers: proof-unit bank (end-to-end confirm) + coverage-complete
  (-> triggers first FULL A4 verdict: neural + slice sub-matrices
  combined across all postures). Tail-optimization DEFERRED as a
  refinement (idempotent store; optimized units fill gaps). Ultracode
  reserved for the verdict synthesis, where fan-out + adversarial
  verification actually add value -- not burned on non-blocking perf.

## 2026-07-20 ~13:00 FIRST G2 SIGNAL (KDD existence gate, seed0)

- The model-free existence gate (PAS-G, passive posture) at KDD seed0:
  NG p=0.268 (correctly SILENT), KG p=0.001 (DETECTS, true rise
  0.673), NS p=0.001 + 65/515 KCs individually reject (DETECTS
  non-standard, true 0.560), SAT p=0.001 bed_stat 11039 (FIRES -
  anomaly, design expects null under saturation).
- READ: the core G2-positive capability WORKS -- silent on no-growth,
  detects real growth of both shapes. This is the per-KC escape from
  the saturation wall the trajectory program failed on aggregate,
  now demonstrated at the KC level on a certified detector. The
  constructive half of G2, first evidence.
- FINDING to confirm/diagnose: SAT fires hard where it should be null
  (stat 5x the others). Either a genuine saturation false-positive (a
  real method limit, matters because real KDD is partly saturated) or
  a numerical pathology in near-ceiling fits. Needs the seed cluster
  + synthesis; NOT over-read from one seed. split_half came back None
  on these cells -- also for the synthesis to check.
- SCOPE: seed0/5, KDD only, EdNet coverage still landing. Verdict is
  seed-clustered. Shape is indicative, not final. Pool banking
  healthy (~3-4 cells/30min, 0 fails).

## 2026-07-20 ~14:30 CORRECTION: EdNet is the EXPENSIVE profile, not the cheap one

- My 2026-07-20 re-sort premise ("EdNet-matched is the cheap profile,
  sidesteps the oversized-KC cost") was WRONG. Measured: EdNet battery
  = 80-105 min/unit vs KDD's 27 min. Mechanism: the deferred
  single-core assembly cost scales with SLICE COUNT; EdNet's thin
  density (median 2 opp/learner-KC) means MORE, shorter slices, so
  EdNet hits the assembly bottleneck HARDER than KDD's oversized KCs.
  The two profiles have DIFFERENT dominant costs (KDD: few huge KCs
  -> Schur fixed it; EdNet: many small slices -> assembly, unfixed).
- Symptom: interleaved UPT=8 chains mixed fast KDD + slow EdNet units;
  EdNet units timed out (540808_2) and blocked KDD units behind them.
- FIX (targeted reconfig, not panic restart): cancelled interleaved
  chains, resubmitted array 0-39%8 UPT=1 -t 600 (skip-done preserves
  20 banked). No unit blocks another; 10h walls >> 2.5h EdNet units.
- REVISED plan: KDD (19/20, ~1h to complete) is the HEADLINE G2
  verdict -- deliver it first. EdNet (~6h to grind) is corroboration;
  its cost makes the assembly-tail optimization now GENUINELY
  load-bearing for EdNet (was correctly deferred for KDD). Decision
  on optimizing-vs-grinding EdNet deferred until the KDD verdict is
  in hand. Honest correction logged; the cheap-profile bet failed,
  which is itself information.

## 2026-07-20 ~15:30 FIRST CERTIFICATION VERDICT: G2 PARTIALLY certified (KDD, seed-clustered, adversarially verified)

- The 6-agent verdict workflow (aggregate -> 4 adversarial verifiers ->
  finalize) delivered the first real A4 verdict. Full text
  _planning/verdict_kdd_g2.md; working table verdict_kdd_working.md.
- POSITIVE (HIGH confidence, verified HOLDS): the passive existence
  gate robustly discriminates no-growth (NG null, p .128-.973 all 5
  seeds) from growth (KG + NS detect, p=0.001 all seeds), clean
  non-overlapping bed_stat (ng<=915, kg/ns>=2375). The per-KC escape
  from the saturation wall, demonstrated at the POOLED/twin level --
  the thing the trajectory program failed on aggregate.
- LIMIT (verified): it is a COARSE (twin-level) detector, NOT a per-KC
  instrument. CG3 fails on the positive control (0/515 KC BH
  discoveries; bank recovery 0.73 vs 0.90 bar at KDD per-KC sparsity).
  "Growth exists in this population" certifies; "which KCs grew" does
  not.
- SATURATION (verifier FAILS): CG6 inverts (SAT fires, all 5 seeds,
  stat ~4x KG despite identical true growth) -- a REAL model-
  misspecification limit under near-ceiling data (independent
  SE-floor degeneracy corroborates), not a fluke/bug. Needs a
  saturation-aware null. Matters for real KDD (mastered KCs).
- ACT: direction not magnitude (rank corr 0.27-0.38 < 0.5 bar); act_p0
  leaks on null, act_p1 disciplined. Neural audits fail (scope to
  PAS-N1, field-representative; PAS-N2 immunity unmeasured).
- CORRECTION to my earlier reads: split_half is NOT None (present,
  finite, all 20 cells -- the None I saw was a dead KC-level leg RB4
  never wired into run.py, a fixable harness gap). The single-seed
  optimism was tempered by the seed cluster + adversarial checks --
  exactly their job.
- NET: G2 partially certified on KDD. A validated coarse growth
  detector with a correct null; per-KC resolution, active ranking,
  saturation robustness, and neural faithfulness each not earned. Two
  failures (saturation null, RB4 wiring) are fixable. EdNet
  corroboration still grinding.

## 2026-07-21 EdNet RAM-OOM diagnosed; monitoring gap fixed

- KDD headline verdict delivered. EdNet corroboration: COVERAGE
  (seed0, all 4 twins) COMPLETE; seed-deepening grinding.
- 2 EdNet deepening units (ns seed1, kg seed2) FAILED with host-RAM
  OOM ("Killed" signature, not CUDA; one ran 194min into the battery
  before OOM). Cause: EdNet's many-slice single-core assembly builds
  large HOST tensors exceeding the 16G chain default -- the assembly
  bottleneck's RAM face. Resubmitted at --mem=64G.
- MONITORING GAP (caught, fixed): OOM-killed jobs do NOT write
  _failed markers, so the marker-counting heartbeat read fails:0
  while sacct showed FAILED. Only the banked-vs-running ACCOUNTING
  discrepancy (33 banked, 5 running, 0 pending -> 2 missing) exposed
  it. Heartbeat v3 now counts sacct FAILED/TIMEOUT/OUT_OF_ME states
  directly (runner state = truth, marker files = proxy). The
  check-the-runner lesson, again.
- 5 EdNet deepening units still on 16G (at OOM risk); left running to
  preserve 3-5h progress; heartbeat v3 will flag any OOM for 64G
  resubmit. EdNet corroboration verdict deferred until pool completes.

## 2026-07-21 SYNTHETIC CERTIFICATION COMPLETE -- both profiles, cross-profile verified

- EdNet corroboration (19/20, seed-clustered) + cross-profile checks
  vs KDD delivered the COMPLETE synthetic G2 verdict
  (_planning/verdict_synthetic_complete.md). The density inversion
  (KDD few-huge-KCs vs EdNet many-tiny-slices) made the findings
  decisive:
  1. COARSE DETECTOR: CERTIFIED + PROFILE-ROBUST. Silent on no-growth,
     detects both growth shapes, every seed, BOTH beds, zero p-value
     overlap. Survives full density inversion. The strongest G2
     result. Caveat: KDD's clean raw-STAT separation is a density
     artifact; the gate works on the calibrated p-value (fine), but
     don't claim stat-magnitude separation as general.
  2. SATURATION FALSE-FIRE: GENERAL limit (replicates on both, same
     numerical-degeneracy fingerprint), severity density-modulated
     (3.7x KDD vs 1.3x EdNet). Needs a saturation-aware null before
     any real-bed near-ceiling claim.
  3. PER-KC RESOLUTION: FUNDAMENTAL limit, NOT sparsity-fixable. The
     sparsity hypothesis is REJECTED -- bank recovery 0.70-0.80 on
     BOTH profiles (EdNet marginally higher despite thinner density),
     0 BH discoveries on both. A gauge/identifiability floor, not a
     data problem. Per-KC certification will not come from more data.
  4. ACT direction-not-magnitude (rank improves on EdNet, silence
     degrades); neural PAS-N1 fails audits largely by design.
- G2 NET: PARTIALLY CERTIFIED. Certified = the profile-robust coarse
  twin-level growth detector (the per-KC escape from the saturation
  wall, at population level). Not earned = per-KC resolution (now
  shown fundamental), active per-KC rank, saturation robustness,
  neural faithfulness, Tier-2 reliability.
- Synthetic phase CLOSED. Straggler cell (ns seed1) cancelled --
  verdict complete at 19/20; pool monitor retired. The two strategic
  decisions (pursue-gaps vs real-data-pilot) now have full evidence,
  laid out in the verdict doc, awaiting the user.

## 2026-07-21 B preconditions DONE -- saturation fix PROVEN, Junyi loader ready

- Saturation-aware bed null (verify_sat_fix.py, SYN_DEV): syn_sat
  false-fire ELIMINATED (p 0.01->1.00, all KCs saturated->excluded->
  bed_stat 0), growth STILL detects (kg/ns p=0.01 both), null control
  holds (ng non-sig), per-KC stats byte-identical (mask changes only
  bed inclusion). Default-None = frozen behavior; real-bed cells opt
  in (design RB3). The B precondition is met.
- Junyi15 loader: EXACT agreement on all 14 triage rows (247,606
  learners, 40 topic-KCs, 0.828 correct, opp median 8, over the real
  26M-row log), 22+5 tests, +prerequisite-graph reader for later G1.
- NEXT: sync + full-suite on cluster (spare laptop), then the REAL
  KDD + Junyi coarse-detector pilots on the cluster -- the first time
  the method touches real learners, closing the validity cycle.

## 2026-07-23 Real-data pilots: verdict fix + a bumpy null run

- First real KDD run (35 min) computed the OBSERVED existence stat
  (bed_stat 6114 on 257/515 unsaturated real KTracedSkills KCs, 3310
  real students) but the real-bed cell DEFERRED the permutation null
  -> no p-value verdict. Fixed: both real-bed cells now run the same
  learner-permutation null that certified the detector synthetically
  (commit 65b8bc9, 40+2 tests). --n-perm-bed threaded to the CLI.
- Re-run (B=199) hit two operational snags, both now handled:
  (1) node contention -- both pilots backfilled onto ONE node
  (caserta, load 185), halving throughput; the null ran 4.5h+ and
  didn't finish. (2) walltime mismatch -- my resubmit sed edited only
  the python line, so KDD kept its original 6h wall (Junyi got 8h);
  KDD would TIMEOUT. Users cannot extend TimeLimit (admin-only).
- FIX: cancelled KDD (freeing caserta -> Junyi, 8h wall, now speeds
  up and should finish); resubmitted KDD --exclusive (whole node, no
  contention, ~3-5x faster) with B=99 (p-floor 0.01, ~2x faster) and
  8h wall. LESSON (durable): real-data permutation nulls are
  expensive; give each pilot its OWN node (--exclusive) and set the
  wall explicitly per script; never co-locate two heavy nulls.

## 2026-07-23 Real pilot blocked on a null MEMORY bug (root cause found)

- Cascade of compute walls on the real-data null, now root-caused:
  the batched permutation null's chunk-size estimator targets ~25% of
  the NODE's total RAM, ignoring the SLURM --mem cgroup cap. On big
  exclusive nodes (250G) it allocates chunks toward the cap and OOMs.
  Killed Junyi (200G, 64min, "Killed") and is about to kill KDD
  (RSS 87.7/96G climbing).
- The real KDD OBSERVED statistic DID compute (bed_stat 6114 on
  257/515 unsaturated real KTracedSkills KCs, 3310 learners) -- a
  partial real result -- but the p-value verdict is blocked by the
  null OOM/slowness.
- ACTION: stopped resubmitting into the wall. Dispatched a root fix
  (chunk budget respects KT_MIRT_MEM_BUDGET_GB / cgroup limit, not
  node RAM). After it lands: clean memory-safe re-run at SMALL B
  (39) for a fast first verdict; Junyi needs learner subsampling
  (26M rows too big for the bank-calibration footprint even at 200G).
- HONEST STATE: synthetic certification is the solid deliverable; the
  real-data VERDICT is proving expensive due to the null's poor
  real-scale memory/time behavior -- itself a methodological finding
  (the permutation battery needs a memory-bounded, subsample-capable
  real-bed mode).

## 2026-07-23 FIRST REAL-DATA VERDICT: coarse detector FIRES on real KDD

- Real KDD (3310 students, 515 KTracedSkills KCs, 1655 analysis
  learners), memory-safe re-run (KT_MIRT_MEM_BUDGET_GB=12, B=39,
  exclusive), exit 0:
  - Saturation fix worked on real near-ceiling data: 258/515 KCs
    flagged near-mastered and EXCLUDED from the bed decision, gate
    applied to the 257 unsaturated KCs (real KDD is ~50% saturated,
    exactly the case the fix protects).
  - EXISTENCE GATE: observed bed_stat 6113.8, bed_pvalue = 0.025.
    B=39 -> p-floor = 1/40 = 0.025, so the observed statistic
    exceeded ALL 39 null permutations -> the gate FIRES at the
    maximum confidence B=39 allows. The per-KC growth signal on real
    KDD learners beats chance.
  - per-KC BH discoveries 0/515 -- coarse detector fires, per-KC
    resolution fails, IDENTICAL character to the synthetic verdict.
- SIGNIFICANCE: the synthetic-certified coarse growth detector
  TRANSFERS to real learners. The validity-gate cycle
  (synthetic-certify -> real-confirm) CLOSES for the coarse detector
  on KDD. First real evidence that the per-KC decomposition detects
  real learning at the population level, on a trustworthy (saturation
  -aware) read.
- HONEST CAVEATS: (1) p=0.025 is the B=39 FLOOR; a stronger p<0.01
  claim needs B>=99 (now feasible with the memory fix). (2) One seed
  (one cohort split); seed-robustness untested on real data. (3)
  Junyi corroboration DEFERRED (26M rows overflow the bank-calibration
  footprint; needs learner subsampling). (4) per-KC resolution fails
  on real data too -- consistent, not new.
- Compute lessons banked: real-data null needs a memory budget
  (fixed) + a subsample-capable mode + its own exclusive node +
  small B for a first read. The grind is the finding: the battery
  needs a real-bed deployment mode.

## 2026-07-23 KDD real verdict STRENGTHENED (B=99); Junyi-40k running

- KDD heavier null (B=99, memory-safe via KT_MIRT_MEM_BUDGET_GB=12,
  exclusive, exit 0): observed stat 6113.8, bed_pvalue = 0.01 (the
  observed beat ALL 99 permutations -> at the B=99 floor). Firmer
  than the B=39 read (0.025). Coarse detector fires at p<=0.01 on
  real KDD; per-KC still 0/515. A solid, non-borderline real-data
  positive for the coarse growth detector.
- Junyi corroboration: the full 247k-learner bed OOMs even with the
  null memory budget (per-replicate footprint over all learners too
  big) -> added --junyi-max-learners (memory-bounded streaming
  pre-pass, commit 959a25e). Junyi-40k (40000 learners) running
  memory-safe (RSS fluctuating 38-94G under a 200G cap). Awaiting its
  verdict for the second-bed corroboration.

## 2026-07-23 Junyi corroboration: INCONCLUSIVE (negative stat, confounded)

- Junyi-40k (40000 of 247k learners, B=39, exit clean): observed
  bed_stat = -11038.9 (NEGATIVE -> the no-growth model M0 fits better
  than the growth model M1 = NO growth detected), 35/40 topic-skills
  unsaturated, 0/40 per-skill. p=0.025 is the B=39 floor but with a
  strongly-negative observed it is a degenerate-fit artifact, not a
  clean detection.
- READ: Junyi does NOT corroborate KDD's positive. But it is CONFOUNDED,
  not a clean "no learning" finding:
  (1) the subsample is NON-RANDOM (first 40k learner-ids in sort order,
      the streaming pre-pass rule) -> likely an unrepresentative cohort;
  (2) Junyi per-learner data is very thin (median ~8 opps, many far
      fewer) -> weak/degenerate growth fit, exactly where the negative
      statistic and floor-p artifact appear.
- HONEST STATE: ONE solid real-data positive (KDD, p<=0.01, growth
  detected on deep-practice real students); the second bed is
  UNCONFIRMED (confounded inconclusive), NOT a contradiction. A proper
  Junyi test needs a RANDOM subsample (not first-N-by-id) and likely a
  larger/denser cohort. The subsample rule (deterministic first-N) was
  chosen for memory-streaming simplicity, not representativeness -- a
  known tradeoff now biting.
- Decision open for the user: pursue Junyi properly (random subsample
  re-run) vs bank the KDD real-data positive and move to the influence
  goal.
