"""The A4 campaign harness: a config-driven CLI that runs generator ->
measurement -> postures -> battery -> gate-verdict JSON + a markdown gate
report (`_planning/design/a4_design.md` v1.1, section 8's run order). This
is stage 5/5's own module: it does not implement any new estimator or
battery arm (those are `synth.py`/`bank.py`/`gate.py`/`rate.py`/
`tracker.py`/`active.py`/`battery.py`/`report.py`, prior stages), it wires
the existing modules into unit-of-work cells with deterministic seeds,
idempotent skip-if-done semantics, and CPU-only execution.

Scope discipline (per the harness instructions): this module implements
the SYNTHETIC certification campaign (design section 8's R2 "synthetic
certification matrix" step) end to end. It does NOT implement real-bed
loading (`prep_kdd.py`/`prep_ednet.py`, R3-R8): those scripts live under
`kt-mirt/scripts/a4/`, outside this build stage's permitted file set
(only `src/kt_mirt/growth/`, `tests/`, and `pyproject.toml`), and no real
dataset lives under a path this stage may touch. `RealBedLoader` below is
a Protocol a future `scripts/a4/prep_*.py` can satisfy so the same
campaign machinery (`run_campaign`) generalizes to a real bed without
modification, but no concrete loader is implemented here -- this is a
documented interpretation, not an oversight (the harness instructions
call for "generator OR real-bed loader", and only the generator leg is in
scope for the files this stage may write).

Interpretation notes (recorded per the harness instructions):

1. **Unit of work = one cell.** A "slice cell" is (profile, twin,
   generator seed): it runs the slice-based machinery (bank calibration,
   PAS-G/MIX-L, the permutation battery) once. A "neural cell" is
   (profile, twin, model seed), run only at the FIRST configured
   generator seed (design section 3.2: "neural arms ... run on generator
   seed 0 with 3 model seeds"): it trains the trackers and ACT and reads
   every neural-side CG/battery-arm statistic. Each cell is independently
   re-derivable from its key alone (nothing but the key is needed to
   reproduce it byte-for-byte modulo floating point), and each writes one
   JSON file; a rerun with an existing, non-empty cell file skips
   recomputation unless ``--force`` is given (unit-of-work idempotency).
2. **Deterministic seeding.** Every RNG draw inside a cell is seeded from
   a stable hash of the cell's own key (`derive_seed`), not from wall
   clock or global state, so idempotent skip and fresh recomputation
   agree bit-for-bit (modulo platform floating-point nondeterminism in
   torch's own kernels, which this harness does not attempt to pin
   further than `torch.manual_seed`).
3. **Seed-pooled aggregation is EXACT only within one process
   invocation.** Arm 9's seed-pooled p-value (`battery.seed_pooled_
   pvalue`) needs each generator seed's raw permutation-null array, which
   is NOT persisted in a slice cell's JSON (it would make cell files
   scale with the replicate count for no downstream benefit once the
   p-value is computed). When every slice cell contributing to a twin's
   aggregate was freshly computed in the current `run_campaign` call, the
   exact pooled statistic is used; when one or more contributing cells
   were loaded from a prior run's persisted file (idempotent skip across
   process invocations), the aggregate falls back to the MAX of the
   per-seed empirical p-values (the conservative direction: harder to
   pass, never easier), and ``"seed_pooled_exact": false`` is recorded so
   this is never silently mistaken for the exact statistic. This is a
   documented efficiency/fidelity trade-off of the unit-of-work design,
   not a threshold change.
4. **Cross-model-seed CG aggregation** (arm 9's "sign-consistent in every
   seed" applied to neural gates, which the design does not itself spell
   out a combination rule for): a neural CG (CG7-CG10, CG9-ACT, the
   ACT CG1-family via RB-A) is read as passing at the twin level only if
   it passes in EVERY configured model seed (a conjunction, the stricter
   of the two obvious readings -- consistent with "thresholds may only
   tighten, never loosen"). CG1/CG1a/CG1c's continuous statistics
   (population mean/p95 rise, per-KC rise) are aggregated across model
   seeds by plain averaging before the gate function is applied (these
   are amplitude bars, not per-seed significance tests), and
   ``is_sign_consistent`` is additionally reported alongside.
5. **RB-A's ``null_bed_rise`` reference on synthetic twins.** RB-A
   (design 5.2) defines its null reference as "the same model's RB1-twin
   read" (the real-bed M0-parametric-bootstrap twin), which this harness
   does not run for CG1a (a real-bed-only construction). For CG1a's
   positive-control check on SYN-KG, the harness instead reads the null
   reference from the SAME PROFILE'S SYN-NG ACT silence read (CG1's own
   population-mean-rise statistic, same generator/model seeds): SYN-NG
   and SYN-KG share one substrate (matched-twin discipline, `synth.py`
   note 9), so SYN-NG's ACT read is the closest synthetic analogue to an
   in-situ null available without running a bootstrap-twin ACT fit. This
   only affects CG1a's internal ``rb_a_firing`` call within this harness;
   it does not change `active.rb_a_firing`'s own logic or thresholds.
6. **CG1c's saturation-refusal report-text condition** is read as CG6's
   own pass verdict computed on the same twin/seeds (`battery.py`'s own
   module docstring note 8 defers this exact judgment call to the
   caller): CG6 and CG1c's "insufficient dynamic range" wording certify
   the identical underlying fact (the gate stays flat and the saturation
   flag fires nearly everywhere), so re-deriving it independently would
   only risk disagreement between two readings of one fact.
7. **MIX-L's rate stage runs on every KC**, not only BH-discovered ones,
   at this harness's scale (tiny run, few KCs): the per-KC "gate_passed"
   flag (this twin's BH rejection) is carried alongside every KC's rate
   fit so downstream reporting can filter, but the fit itself is cheap
   enough here to just always compute (avoids an empty rate table on
   twins/tiny configs where BH discovers nothing, which would otherwise
   make CG5's misfit-fraction clauses vacuously undefined).
8. **Device is a config/CLI choice, defaulting to CPU.** ``RunConfig.
   device`` (``--device`` on the CLI) is ``"cpu"`` (default) or ``"cuda"``;
   ``"cuda"`` is validated against ``torch.cuda.is_available()`` at
   ``RunConfig`` construction time so a bad request fails immediately
   rather than silently falling back to CPU. This module no longer forces
   ``CUDA_VISIBLE_DEVICES`` at import time (that was a standing compute-
   policy guard from when the local GPU was off-limits program-wide,
   LEDGER 2026-07-18 "Compute policy change"; the policy was later
   reversed -- LEDGER 2026-07-18 "P3 build EXECUTED", "local 4060 will
   join the campaign as an auxiliary worker" -- and this module's
   `__post_init__` still hard-pinned CPU as a stale leftover of the old
   guard, LEDGER 2026-07-18 "P3 build EXECUTED" smell (5)). Every internal
   call site that used to hardcode the CPU device string now threads
   ``cfg.device`` through instead, so a ``"cuda"`` run actually executes
   on GPU rather than only changing a validated-but-unused field.
9. **Population "true rise" is an unweighted mean over KCs**
   (`true_pop_rise` in `certify_twin`), used by CG1a/CG1c. The design's
   own population curves are Q-matrix-expanded and popularity-weighted
   (`synth.py`'s ``kc_pop_weight``), which this harness does not
   propagate through the slice-cell JSON (keeping cell files small); an
   unweighted per-KC mean is a documented approximation, adequate for a
   diagnostic-scale dry run but not a substitute for the pre-registered
   popularity-weighted population statistic a full campaign would use.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch

from kt_mirt.growth import active as active_mod
from kt_mirt.growth import bank as bank_mod
from kt_mirt.growth import battery as battery_mod
from kt_mirt.growth import gate as gate_mod
from kt_mirt.growth import newton as newton_mod
from kt_mirt.growth import rate as rate_mod
from kt_mirt.growth import report as report_mod
from kt_mirt.growth import synth as synth_mod
from kt_mirt.growth import tracker as tracker_mod
from kt_mirt.growth.slices import build_slices, saturation_stats, slices_by_kc
from kt_mirt.growth.synth import LearnerLog

DEVICE = "cpu"


# =============================================================================
# Deterministic seeding
# =============================================================================


def derive_seed(*parts) -> int:
    """A stable, process-independent 31-bit seed derived from ``parts``
    (interpretation note 2). Uses a cryptographic hash rather than
    Python's salted ``hash()`` so the same key always yields the same
    seed across processes and platforms."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


# =============================================================================
# Density profiles
# =============================================================================


def make_profile(
    name: str,
    n_kcs: int,
    n_learners: int,
    kcs_per_learner: float,
    base: synth_mod.DensityProfile = synth_mod.KDD_MATCHED,
) -> synth_mod.DensityProfile:
    """A smaller density profile sharing ``base``'s density SHAPE (opp/
    rate quantiles, item arity) but overriding scale (``n_kcs``,
    ``n_learners``, ``kcs_per_learner``) -- the "build agent implementation
    freedom" section 3.1 grants for generator mechanics, used here to
    build tiny smoke-test configs the way `synth.SYN_DEV` already does."""
    return dataclasses.replace(
        base, name=name, n_kcs=n_kcs, n_learners=n_learners, kcs_per_learner=kcs_per_learner
    )


TINY_PROFILE = make_profile("tiny_kdd", n_kcs=8, n_learners=200, kcs_per_learner=5.0)


# =============================================================================
# JSON-safe conversion
# =============================================================================


def _jsonable(obj):
    """Recursively converts dataclasses, numpy scalars/arrays, tuples, and
    NaN/inf floats into strict-JSON-safe native Python."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


# =============================================================================
# Real-bed extension point (not implemented this stage -- module docstring)
# =============================================================================


class RealBedLoader(Protocol):
    """Structural interface a future `scripts/a4/prep_*.py` real-bed
    loader can satisfy so `run_campaign` generalizes beyond synthetic
    twins without modification. Not implemented in this build stage (see
    module docstring)."""

    def load(self, seed: int) -> tuple[list[LearnerLog], int, int]:
        """Returns ``(learners, n_learners, n_kcs)`` for one seeded draw
        (e.g. a fixed user sample) of a real bed."""
        ...


# =============================================================================
# Run configuration
# =============================================================================


@dataclass
class RunConfig:
    output_dir: Path
    profile: synth_mod.DensityProfile = TINY_PROFILE
    twins: Sequence[str] = field(default_factory=lambda: list(synth_mod.TWIN_NAMES))
    generator_seeds: Sequence[int] = field(default_factory=lambda: [0, 1])
    model_seeds: Sequence[int] = field(default_factory=lambda: [0, 1])
    n_perm_bed: int = 39
    n_perm_kc: int = 19
    bank_epochs: int = 40
    tracker_hidden: int = 16
    tracker_emb: int = 8
    tracker_lr: float = 0.05
    tracker_epochs: int = 15
    # Mini-batch size for BOTH tracker trainers (tracker.py module docstring
    # note 6). None (default) = whole-cohort training, the exact pre-existing
    # loop, so every profile that does not set it (KDD) is byte-unchanged.
    # Set per-profile by slurm/enumerate_units.py for EdNet-matched neural
    # cells, whose whole-cohort PAS-N2 backward (>50 GiB) OOMs every GPU in
    # the cluster menu (a40/l40 48 GB, a100 40 GB). Implementation-level
    # memory concession: the frozen design fixes estimators and thresholds,
    # not batch sizes.
    tracker_batch_size: Optional[int] = None
    act_hidden: int = 16
    act_emb: int = 8
    act_lr: float = 0.05
    act_epochs: int = 20
    run_act_p1: bool = True
    # Campaign-exclusion switch for the ACT-P0 variant (default True keeps
    # every pre-existing behavior and test unchanged). Added at A4 campaign
    # launch under the orchestrator's binding ruling that ACT-P0 is EXCLUDED
    # from the synthetic certification campaign (scale-persistent fabrication,
    # ledger 2026-07-18): ACT-P0 is computed INSIDE each neural cell rather
    # than being its own enumerable unit, so the only place it can be masked
    # is here. slurm/enumerate_units.py's PRODUCTION_HYPERPARAMS sets this
    # False for campaign units; ACT-P1 carries the active posture.
    run_act_p0: bool = True
    n_reshuffles: int = 3
    drill_repeats: int = 30
    device: str = DEVICE
    force: bool = False

    def __post_init__(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"RunConfig.device must be 'cpu' or 'cuda', got {self.device!r}")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "RunConfig.device='cuda' requested but torch.cuda.is_available() is False "
                "(no CUDA device visible to this process)"
            )


def _act_variants(cfg: "RunConfig") -> list[str]:
    """Which ACT variants this run computes/aggregates (see RunConfig.run_act_p0)."""
    return (["act_p0"] if cfg.run_act_p0 else []) + (["act_p1"] if cfg.run_act_p1 else [])


# =============================================================================
# Cell paths + idempotency
# =============================================================================


def _cell_dir(cfg: RunConfig, twin: str) -> Path:
    d = Path(cfg.output_dir) / cfg.profile.name / twin
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slice_cell_path(cfg: RunConfig, twin: str, seed: int) -> Path:
    return _cell_dir(cfg, twin) / f"slice_seed{seed}.json"


def _neural_cell_path(cfg: RunConfig, twin: str, model_seed: int) -> Path:
    return _cell_dir(cfg, twin) / f"neural_modelseed{model_seed}.json"


def _load_if_done(path: Path, force: bool) -> Optional[dict]:
    if force or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_cell(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


# =============================================================================
# Slice cell: bank calibration + PAS-G/MIX-L + permutation battery
# =============================================================================


def _log_stage(twin: str, seed: int, stage: str, wall_s: float) -> None:
    """Emits one stage-timing line into the unit's own stdout (captured by
    `slurm/chain_runner.sbatch` into `logs/chain_%A_%a.out`, the "unit
    log"): cheap, sanctioned instrumentation so future diagnosis of a slow
    slice unit does not require a fresh py-spy dive (the 2026-07-20
    incident that found the `torch.func` per-iteration dispatch cost)."""
    print(f"[slice_cell stage] twin={twin} seed={seed} stage={stage} wall_s={wall_s:.2f}", flush=True)


def run_slice_cell(cfg: RunConfig, twin: str, seed: int) -> tuple[dict, Optional[dict]]:
    """Returns ``(json_result, raw_nulls)``; ``raw_nulls`` is ``None`` iff
    loaded from an existing cell file (interpretation note 3)."""
    path = _slice_cell_path(cfg, twin, seed)
    cached = _load_if_done(path, cfg.force)
    if cached is not None:
        return cached, None

    unit_t0 = time.perf_counter()
    newton_mod.reset_dispatch_counts()  # scoped to this cell; read after permutation_battery below

    profile = cfg.profile
    twin_data = synth_mod.generate_twin(twin, profile, seed)
    acceptance = synth_mod.run_generator_acceptance_checks(twin_data)

    rows_all = bank_mod.build_calibration_rows(twin_data.learners)
    calib, analysis = bank_mod.split_learners(profile.n_learners, seed=derive_seed("cohort", twin, seed))
    hierarchy = bank_mod.flat_hierarchy(twin_data.item_bank.n_items)
    bank_cfg = bank_mod.BankModelConfig(
        n_epochs_max=cfg.bank_epochs, lr=0.05, batch_size=4096, patience_epochs=2,
        seed=derive_seed("bank", twin, seed), device=cfg.device,
    )
    t_stage = time.perf_counter()
    fit = bank_mod.calibrate_bank(
        rows_all, profile.n_learners, twin_data.n_kcs, calib, hierarchy,
        bank_mod.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=bank_cfg,
    )
    _log_stage(twin, seed, "bank_calibration", time.perf_counter() - t_stage)
    frozen = bank_mod.freeze_bank(fit)
    recovery = bank_mod.synthetic_bank_recovery_check(fit, twin_data.item_bank.b_true)

    calib_set = set(int(c) for c in calib.tolist())
    calib_learner_logs = [l for l in twin_data.learners if l.learner in calib_set]
    rows_calib = bank_mod.build_calibration_rows(calib_learner_logs)
    sat_rate_calib, sat_unsat_calib = saturation_stats(rows_calib, twin_data.n_kcs)

    analysis_set = set(int(a) for a in analysis.tolist())
    analysis_learners = [l for l in twin_data.learners if l.learner in analysis_set]
    rows_analysis = bank_mod.build_calibration_rows(analysis_learners)
    sat_rate_analysis, sat_unsat_analysis = saturation_stats(rows_analysis, twin_data.n_kcs)
    slices = build_slices(rows_analysis)

    # Saturation-aware bed statistic (gate.py module docstring note 6, the
    # certified fix for the syn_sat CG6 inversion): the bed-level existence
    # decision sums only over UNSATURATED KCs (the raw, model-free
    # calibration-cohort flag), since a near-ceiling KC carries no
    # observable growth information and otherwise spuriously fires the gate.
    # The SAME mask goes to the observed gate and the permutation null so
    # both bed statistics are summed over an identical KC set (the mask is a
    # calibration-cohort property, permutation-invariant). Per-KC statistics
    # (kc_stat/BH/BY) are unaffected.
    bed_kc_mask = np.asarray(sat_unsat_calib, dtype=bool)

    t_stage = time.perf_counter()
    gate_result = gate_mod.compute_gate_result(
        slices, frozen, twin_data.n_kcs, device=cfg.device, bed_kc_mask=bed_kc_mask
    )
    _log_stage(twin, seed, "slice_fits", time.perf_counter() - t_stage)

    n_rep = max(cfg.n_perm_bed, cfg.n_perm_kc)
    t_stage = time.perf_counter()
    null = gate_mod.permutation_null(
        analysis_learners, len(analysis_learners), twin_data.n_kcs, frozen,
        n_replicates=n_rep, seed=derive_seed("perm", twin, seed), device=cfg.device,
        bed_kc_mask=bed_kc_mask,
    )
    _log_stage(twin, seed, "permutation_battery", time.perf_counter() - t_stage)
    # Dispatch-path visibility (per the review note that a silent fallback
    # to the generic torch.func path in this hot loop must never again be
    # invisible): confirms the analytic fast path actually engaged for
    # every binary_logit_nll call made so far in this cell (bank
    # calibration's own model is a separate neural fit, not this solver).
    _counts = newton_mod.dispatch_counts()
    print(
        f"[slice_cell dispatch] twin={twin} seed={seed} "
        f"analytic_calls={_counts['analytic']} generic_calls={_counts['generic']}",
        flush=True,
    )
    bed_null = null["bed"][: cfg.n_perm_bed]
    kc_null = null["kc"][: cfg.n_perm_kc]
    bed_pvalue = gate_mod.empirical_pvalue(gate_result.bed_stat, bed_null)
    kc_pvalue = np.array(
        [gate_mod.empirical_pvalue(gate_result.kc_stat[c], kc_null[:, c]) for c in range(twin_data.n_kcs)]
    )
    bh_reject = gate_mod.bh_fdr(kc_pvalue)
    by_reject = gate_mod.by_correction(kc_pvalue)

    by_kc = slices_by_kc(slices, twin_data.n_kcs)
    kc_rate_fits = {}
    misfit_fits = {}
    for c, kc_slices in enumerate(by_kc):
        fit_c = rate_mod.fit_kc_rate(kc_slices, frozen) if kc_slices else None
        kc_rate_fits[c] = fit_c
        if fit_c is not None:
            misfit_fits[c] = rate_mod.misfit_flag(kc_slices, frozen, fit_c, seed=derive_seed("misfit", twin, seed, c))

    # Arm 7 (split-half): observed vs the density-predicted value, via a
    # per-slice theta(n) built from each KC's own bounded-exponential rate
    # fit where available (else a flat constant-rate fallback).
    theta0_lookup = {}
    for c, kc_slices in enumerate(by_kc):
        fit_c = kc_rate_fits.get(c)
        if fit_c is None:
            continue
        for local_idx, sl in enumerate(kc_slices):
            theta0_lookup[(sl.learner, sl.kc)] = (float(fit_c.theta0[local_idx]), fit_c.m_c, fit_c.r_c)

    def _theta_fn(sl):
        key = (sl.learner, sl.kc)
        if key not in theta0_lookup:
            p = np.clip(float(sl.response.mean()) if sl.T else 0.5, 1e-3, 1 - 1e-3)
            return np.full(sl.T, np.log(p / (1 - p)))
        theta0, m_c, r_c = theta0_lookup[key]
        n = sl.opportunity.astype(float)
        return m_c - (m_c - theta0) * np.exp(-r_c * (n - 1))

    all_slices = list(slices.values())
    observed_split_half = battery_mod.per_learner_split_half(all_slices)
    predicted_split_half = battery_mod.predicted_split_half_via_bootstrap(
        all_slices, frozen, _theta_fn, n_bootstrap=20, seed=derive_seed("splithalf", twin, seed),
    )
    split_half_gap = (
        observed_split_half - predicted_split_half
        if np.isfinite(observed_split_half) and np.isfinite(predicted_split_half)
        else float("nan")
    )

    result = {
        "twin": twin,
        "seed": seed,
        "n_learners": profile.n_learners,
        "n_kcs": twin_data.n_kcs,
        "n_analysis_learners": len(analysis_learners),
        "acceptance": {"all_passed": acceptance.all_passed, "checks": acceptance.checks},
        "bank_recovery": recovery,
        "saturation": {
            "calib_rate": sat_rate_calib, "calib_unsaturated": sat_unsat_calib,
            "analysis_rate": sat_rate_analysis, "analysis_unsaturated": sat_unsat_analysis,
        },
        "gate": {
            "bed_stat": gate_result.bed_stat,
            "kc_stat": gate_result.kc_stat,
            "bed_pvalue": bed_pvalue,
            "kc_pvalue": kc_pvalue,
            "bh_reject": bh_reject,
            "by_reject": by_reject,
        },
        "rate": {
            str(c): (
                {"m_c": fit_c.m_c, "r_c": fit_c.r_c, "r_c_se": fit_c.r_c_se, "r_c_z": fit_c.r_c_z,
                 "misfit_fired": misfit_fits[c].fired if c in misfit_fits else None,
                 "misfit_pvalue": misfit_fits[c].p_value if c in misfit_fits else None}
                if fit_c is not None else None
            )
            for c, fit_c in kc_rate_fits.items()
        },
        "split_half": {
            "observed": observed_split_half, "predicted": predicted_split_half, "gap": split_half_gap,
        },
        "true_rise_per_kc": twin_data.truth.true_rise_per_kc,
        "r_base_per_kc": twin_data.truth.r_base,
        "silent_kc_mask": twin_data.truth.silent_kc_mask,
        "b_hat": fit.b_hat,
    }
    clean_result = json.loads(json.dumps(_jsonable(result)))
    _write_cell(path, clean_result)
    _log_stage(twin, seed, "total", time.perf_counter() - unit_t0)
    raw_nulls = {"bed_null": bed_null, "observed_bed_stat": gate_result.bed_stat}
    return clean_result, raw_nulls


# =============================================================================
# Neural cell: trackers + ACT, and every neural-side CG/battery-arm read
# =============================================================================


def _train_held_split(learners: Sequence[LearnerLog], seed: int, train_frac: float = 0.8):
    """The neural 80/20-learner-split forecast regime (design section
    2.2 / R1-I11), applied within the analysis cohort."""
    n = len(learners)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    train = [learners[i] for i in perm[:n_train]]
    held = [learners[i] for i in perm[n_train:]]
    return train, held


def _direction_audit_pooled(model, learners: Sequence[LearnerLog], device: str = "cpu") -> float:
    """CG10, pooled over every (learner, KC) primary-tagged subsequence
    (the same "primary tag" convention `battery.py` already uses, module
    docstring note 4 there): each slice's own consecutive-transition
    violation booleans are pooled (transition-count-weighted) into one
    fraction, rather than averaging per-slice fractions unweighted, so a
    long slice does not count the same as a 2-point one."""
    total_violations = 0.0
    total_transitions = 0
    for log in learners:
        if len(log.item_ids) == 0:
            continue
        item_ids = torch.as_tensor(log.item_ids, dtype=torch.long, device=device).unsqueeze(0)
        responses = torch.as_tensor(log.responses, dtype=torch.long, device=device).unsqueeze(0)
        ability = battery_mod.pas_n1_full_ability_trajectory(model, item_ids, responses)  # (T, C)
        primary = np.asarray(log.tag_ids)[:, 0]
        for c in np.unique(primary):
            rows = np.where(primary == c)[0]
            if len(rows) < 2:
                continue
            theta_c = ability[rows, c]
            resp_c = np.asarray(log.responses)[rows]
            frac = battery_mod.direction_violation_fraction(theta_c, resp_c)
            if np.isfinite(frac):
                n_trans = len(rows) - 1
                total_violations += frac * n_trans
                total_transitions += n_trans
    return float(total_violations / total_transitions) if total_transitions else float("nan")


def _act_implied_rises(
    model: active_mod.ActiveModel,
    analysis_learners: Sequence[LearnerLog],
    slices_analysis: dict,
    frozen: bank_mod.FrozenBank,
    n_kcs: int,
    b_ref: float,
    kc_filter: Optional[set] = None,
    device: str = "cpu",
):
    """E-A1's population/per-learner/per-KC reads (CG1/CG1a/CG1b/CG1c,
    RB-A), computed from the model's amortized recognition outputs over
    the FULL analysis cohort's conditioning window plus the closed-form
    implied-rise formula (never a data replay, per `active.py`'s own
    module docstring). ``kc_filter``, if given, restricts which KCs'
    slices contribute to the per-learner/population aggregate (CG1b's
    silent-subset read); the per-KC array is always computed over every
    KC regardless.

    ``slices_analysis``'s learner key is `build_calibration_rows`'s
    POSITIONAL index into ``analysis_learners`` (not `LearnerLog.
    learner`; `bank.py`'s own docstring: "into the learner list passed
    in"), so as long as ``batch_full`` is built from that SAME ordered
    list (it is, below), the slice key indexes ``u_all``/``lam_all``
    directly with no id-remapping step."""
    batch_full = tracker_mod.build_learner_batch(analysis_learners, bank=frozen, device=device)
    seq_lens = active_mod.seq_lens_from_mask(batch_full.seq_mask)
    with torch.no_grad():
        u_all, lam_all = model.recognition(batch_full.item_ids, batch_full.responses, seq_lens)
    u_all = u_all.cpu().numpy()
    lam_all = lam_all.cpu().numpy() if lam_all is not None else np.ones_like(u_all)
    g_c = model.g_c.detach().cpu().numpy()
    v_c = model.v_c.detach().cpu().numpy()
    M = float(model.M.detach().cpu().item())

    per_learner: dict[int, list[float]] = {}
    per_kc: dict[int, list[float]] = {c: [] for c in range(n_kcs)}
    for (i, kc), sl in slices_analysis.items():
        z0 = np.array([u_all[i] + v_c[kc]])
        lam_val = np.array([lam_all[i]])
        g_val = np.array([g_c[kc]])
        rise = float(active_mod.implied_score_rise(z0, lam_val, g_val, M, np.array([b_ref]))[0])
        per_kc[kc].append(rise)
        if kc_filter is None or kc in kc_filter:
            per_learner.setdefault(i, []).append(rise)

    per_learner_mean = np.array([float(np.mean(v)) for v in per_learner.values()]) if per_learner else np.zeros(0)
    pop_mean = float(per_learner_mean.mean()) if per_learner_mean.size else float("nan")
    p95_abs = float(np.percentile(np.abs(per_learner_mean), 95)) if per_learner_mean.size else float("nan")
    kc_rise = np.array([float(np.mean(v)) if v else float("nan") for v in per_kc.values()])
    return pop_mean, p95_abs, kc_rise, dict(u=u_all, lam=lam_all, g_c=g_c, v_c=v_c, M=M)


def run_neural_cell(cfg: RunConfig, twin: str, model_seed: int) -> dict:
    """Trains PAS-N1 (trained + frozen-encoder null), PAS-N2 (trained),
    and ACT (ACT-P0, and ACT-P1 if configured), all on generator seed
    ``cfg.generator_seeds[0]`` (design section 3.2's compute concession),
    and computes every neural-side CG/battery-arm read this harness's
    scope covers: CG7 (frozen-encoder margin), CG8 (contamination), CG9
    (order-invariance), CG10 (direction audit), CG9-ACT (recognition
    stability), and the raw ACT implied-rise statistics CG1/CG1a/CG1b/
    CG1c are built from at the aggregation stage (`certify_twin`, which
    needs the OTHER twins' cells for cross-twin references -- see module
    docstring note 5)."""
    path = _neural_cell_path(cfg, twin, model_seed)
    cached = _load_if_done(path, cfg.force)
    if cached is not None:
        return cached

    profile = cfg.profile
    seed0 = cfg.generator_seeds[0]
    twin_data = synth_mod.generate_twin(twin, profile, seed0)

    rows_all = bank_mod.build_calibration_rows(twin_data.learners)
    calib, analysis = bank_mod.split_learners(profile.n_learners, seed=derive_seed("cohort", twin, seed0))
    hierarchy = bank_mod.flat_hierarchy(twin_data.item_bank.n_items)
    bank_cfg = bank_mod.BankModelConfig(
        n_epochs_max=cfg.bank_epochs, lr=0.05, batch_size=4096, patience_epochs=2,
        seed=derive_seed("bank", twin, seed0), device=cfg.device,
    )
    fit = bank_mod.calibrate_bank(
        rows_all, profile.n_learners, twin_data.n_kcs, calib, hierarchy,
        bank_mod.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=bank_cfg,
    )
    frozen = bank_mod.freeze_bank(fit)
    b_ref = float(np.median(fit.b_hat))

    analysis_set = set(int(a) for a in analysis.tolist())
    analysis_learners = [l for l in twin_data.learners if l.learner in analysis_set]
    train_learners, held_learners = _train_held_split(
        analysis_learners, seed=derive_seed("traineval", twin, seed0)
    )
    n_items = twin_data.item_bank.n_items
    n_kcs = twin_data.n_kcs

    def _tseed(tag: str) -> int:
        return derive_seed("torch", twin, seed0, model_seed, tag)

    # --- PAS-N1 trained + frozen-encoder null (arm 3, CG7) ---
    tcfg = tracker_mod.TrackerConfig(
        hidden_dim=cfg.tracker_hidden, emb_dim=cfg.tracker_emb, lr=cfg.tracker_lr,
        n_epochs=cfg.tracker_epochs, seed=model_seed, device=cfg.device, train_encoder=True,
        batch_size=cfg.tracker_batch_size,
    )
    train_batch1 = tracker_mod.build_learner_batch(train_learners, bank=frozen, device=cfg.device)
    held_batch1 = tracker_mod.build_learner_batch(held_learners, bank=frozen, device=cfg.device)

    torch.manual_seed(_tseed("pasn1_trained"))
    pasn1_trained = tracker_mod.build_tracker("pas_n1", n_items, n_kcs, tcfg)
    tracker_mod.train_pas_n1(pasn1_trained, train_batch1, tcfg)
    trained_nll1 = tracker_mod.forecast_nll(pasn1_trained, held_batch1, "pas_n1")

    fcfg = dataclasses.replace(tcfg, train_encoder=False)
    torch.manual_seed(_tseed("pasn1_frozen"))
    pasn1_frozen = tracker_mod.build_tracker("pas_n1", n_items, n_kcs, fcfg)
    tracker_mod.train_pas_n1(pasn1_frozen, train_batch1, fcfg)
    frozen_nll1 = tracker_mod.forecast_nll(pasn1_frozen, held_batch1, "pas_n1")

    abilities_trained = battery_mod.full_ability_batch(pasn1_trained, analysis_learners, device=cfg.device)
    abilities_frozen = battery_mod.full_ability_batch(pasn1_frozen, analysis_learners, device=cfg.device)
    trained_profile = np.array(
        [tracker_mod.displacement(battery_mod.collect_kc_trajectory(abilities_trained, analysis_learners, c)) for c in range(n_kcs)]
    )
    frozen_profile = np.array(
        [tracker_mod.displacement(battery_mod.collect_kc_trajectory(abilities_frozen, analysis_learners, c)) for c in range(n_kcs)]
    )
    true_rise = np.asarray(twin_data.truth.true_rise_per_kc)
    cg7 = battery_mod.cg7_verdict(trained_profile, frozen_profile, true_rise)

    # --- PAS-N2 trained (arm 4's contamination-proof-by-construction control) ---
    rows_train = bank_mod.build_calibration_rows(train_learners)
    rows_held = bank_mod.build_calibration_rows(held_learners)
    slices_train = build_slices(rows_train)
    slices_held = build_slices(rows_held)
    slice_batch_train = tracker_mod.build_slice_batch(list(slices_train.values()), bank=frozen, device=cfg.device)
    slice_batch_held = tracker_mod.build_slice_batch(list(slices_held.values()), bank=frozen, device=cfg.device)
    torch.manual_seed(_tseed("pasn2_trained"))
    pasn2_trained = tracker_mod.build_tracker("pas_n2", n_items, n_kcs, tcfg)
    tracker_mod.train_pas_n2(pasn2_trained, slice_batch_train, tcfg)
    trained_nll2 = tracker_mod.forecast_nll(pasn2_trained, slice_batch_held, "pas_n2") if slice_batch_held.item_ids.numel() else float("nan")

    # --- CG8 contamination probe (trained PAS-N1) ---
    drilled = list(range(min(5, n_kcs)))
    item_for_kc = {c: int(twin_data.item_bank.items_by_primary[c][0]) for c in drilled}
    contamination = battery_mod.contamination_probe(
        pasn1_trained, n_kcs, drilled, item_for_kc, n_repeats=cfg.drill_repeats, device=cfg.device,
    )

    # --- CG9 order-invariance stress (trained PAS-N1) ---
    order_inv = battery_mod.order_invariance_stress(
        pasn1_trained, analysis_learners, n_kcs, n_reshuffles=cfg.n_reshuffles,
        seed=derive_seed("order", twin, seed0, model_seed), device=cfg.device,
    )

    # --- CG10 direction audit (trained PAS-N1; certified on SYN-KG, reported elsewhere) ---
    violation_fraction = _direction_audit_pooled(pasn1_trained, analysis_learners, device=cfg.device)

    # --- Saturation flags (calibration cohort), for CG1c/CG6 report-text ---
    calib_set = set(int(c) for c in calib.tolist())
    calib_learner_logs = [l for l in twin_data.learners if l.learner in calib_set]
    rows_calib = bank_mod.build_calibration_rows(calib_learner_logs)
    sat_rate, sat_unsat = saturation_stats(rows_calib, n_kcs)

    # --- ACT: ACT-P0 (and ACT-P1) ---
    m_init = active_mod.ceiling_init(fit.b_hat)
    rows_analysis = bank_mod.build_calibration_rows(analysis_learners)
    slices_analysis = build_slices(rows_analysis)
    silent_mask = np.asarray(twin_data.truth.silent_kc_mask, dtype=bool)
    silent_kcs = set(int(c) for c in np.where(silent_mask)[0])

    act_results = {}
    variants = _act_variants(cfg)
    for variant in variants:
        acfg = active_mod.ActiveConfig(
            variant=variant, hidden_dim=cfg.act_hidden, emb_dim=cfg.act_emb, lr=cfg.act_lr,
            n_epochs=cfg.act_epochs, seed=model_seed, device=cfg.device, m_fixed=False,
        )
        torch.manual_seed(_tseed(f"act_{variant}"))
        model = active_mod.ActiveModel(num_items=n_items, n_kcs=n_kcs, cfg=acfg, m_init=m_init)
        model = model.to(cfg.device)  # ActiveModel does not self-place (unlike tracker.build_tracker)
        train_batch_act = tracker_mod.build_learner_batch(train_learners, bank=frozen, device=cfg.device)
        active_mod.train_active(model, train_batch_act, acfg)

        pop_mean, p95_abs, kc_rise, extra = _act_implied_rises(
            model, analysis_learners, slices_analysis, frozen, n_kcs, b_ref, device=cfg.device,
        )
        silent_pop_mean, silent_p95, _, _ = _act_implied_rises(
            model, analysis_learners, slices_analysis, frozen, n_kcs, b_ref, kc_filter=silent_kcs, device=cfg.device,
        ) if silent_kcs else (float("nan"), float("nan"), None, None)

        growing_kcs = set(range(n_kcs)) - silent_kcs
        growing_finite = sorted(
            c for c in growing_kcs if np.isfinite(kc_rise[c]) and np.isfinite(true_rise[c])
        )
        growing_rank_corr = (
            bank_mod.spearman_rank_correlation(kc_rise[growing_finite], true_rise[growing_finite])
            if len(growing_finite) >= 2
            else float("nan")
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(kc_rise) / np.abs(true_rise)
        overshoot_mask = np.zeros(n_kcs, dtype=bool)
        for c in growing_kcs:
            overshoot_mask[c] = np.isfinite(ratio[c]) and ratio[c] > 1.5
        overshoot_fraction = float(overshoot_mask[sorted(growing_kcs)].mean()) if growing_kcs else float("nan")

        act_stability = battery_mod.act_recognition_stability(
            model, analysis_learners, n_kcs, b_ref=b_ref, n_reshuffles=cfg.n_reshuffles,
            seed=derive_seed("act_order", twin, seed0, model_seed, variant), device=cfg.device,
        )

        act_results[variant] = {
            "pop_mean_rise": pop_mean,
            "p95_abs_rise": p95_abs,
            "kc_rise": kc_rise,
            "silent_pop_mean_rise": silent_pop_mean,
            "silent_p95_abs_rise": silent_p95,
            "growing_rank_corr": growing_rank_corr,
            "overshoot_fraction": overshoot_fraction,
            "cg9_act": dataclasses.asdict(act_stability),
        }

    result = {
        "twin": twin,
        "model_seed": model_seed,
        "generator_seed": seed0,
        "n_kcs": n_kcs,
        "tracker": {
            "trained_nll_pas_n1": trained_nll1,
            "frozen_nll_pas_n1": frozen_nll1,
            "trained_nll_pas_n2": trained_nll2,
            "trained_profile": trained_profile,
            "frozen_profile": frozen_profile,
            "cg7": dataclasses.asdict(cg7),
            "cg8": {"ratio": contamination.ratio, "passed": contamination.passed},
            "cg9": dataclasses.asdict(order_inv),
            "cg10_violation_fraction": violation_fraction,
        },
        "saturation": {"rate": sat_rate, "unsaturated": sat_unsat},
        "true_rise_per_kc": true_rise,
        "act": act_results,
    }
    clean_result = json.loads(json.dumps(_jsonable(result)))
    _write_cell(path, clean_result)
    return clean_result


# =============================================================================
# Twin-level certification: aggregate cells across seeds into CG verdicts
# =============================================================================


def certify_twin(
    cfg: RunConfig,
    twin: str,
    slice_pairs: Sequence[tuple[dict, Optional[dict]]],
    neural_cells: Sequence[dict],
    ng_ref: Optional[dict] = None,
) -> dict:
    """Aggregates every generator-seed slice cell and every model-seed
    neural cell for one twin into the CG gates this harness's scope
    covers (interpretation notes 3-6, 9). ``ng_ref`` is SYN-NG's own
    aggregate for this twin's profile (needed only when ``twin ==
    "syn_kg"``, RB-A's null reference, note 5)."""
    slice_results = [p[0] for p in slice_pairs]
    slice_raws = [p[1] for p in slice_pairs]
    r0 = slice_results[0]
    n_kcs = r0["n_kcs"]
    profile_name = cfg.profile.name
    variants = _act_variants(cfg)

    bed_pvalues = np.array([r["gate"]["bed_pvalue"] for r in slice_results])
    if all(raw is not None for raw in slice_raws):
        observed_by_seed = np.array([raw["observed_bed_stat"] for raw in slice_raws])
        null_by_seed = [np.asarray(raw["bed_null"]) for raw in slice_raws]
        pooled_bed_pvalue = battery_mod.seed_pooled_pvalue(observed_by_seed, null_by_seed)
        seed_pooled_exact = True
    else:
        pooled_bed_pvalue = float(np.max(bed_pvalues))
        seed_pooled_exact = False

    bh_matrix = np.array([r["gate"]["bh_reject"] for r in slice_results], dtype=bool)
    discovery_fraction = float(bh_matrix.mean()) if bh_matrix.size else float("nan")
    bh_majority = bh_matrix.mean(axis=0) >= 0.5 if bh_matrix.size else np.zeros(n_kcs, dtype=bool)

    bank_recovery_corr = float(np.mean([r["bank_recovery"]["rank_corr"] for r in slice_results]))

    r_base = np.array(r0["r_base_per_kc"], dtype=float)
    silent_mask = np.array(r0["silent_kc_mask"], dtype=bool)
    r_hat = np.full(n_kcs, np.nan)
    misfit_fired = np.zeros(n_kcs, dtype=bool)
    for c in range(n_kcs):
        rc = r0["rate"].get(str(c))
        if rc is not None and rc.get("r_c") is not None:
            r_hat[c] = rc["r_c"]
            misfit_fired[c] = bool(rc.get("misfit_fired")) if rc.get("misfit_fired") is not None else False

    finite_mask = bh_majority & np.isfinite(r_hat) & np.isfinite(r_base)
    r_rank_corr = (
        bank_mod.spearman_rank_correlation(r_hat[finite_mask], r_base[finite_mask])
        if finite_mask.sum() >= 2
        else float("nan")
    )

    split_half_gaps = [r["split_half"]["gap"] for r in slice_results if r["split_half"]["gap"] is not None]
    split_half_gap = float(np.nanmean(split_half_gaps)) if split_half_gaps else float("nan")

    sat_unsat = np.array(r0["saturation"]["calib_unsaturated"], dtype=bool)
    saturation_flag_fraction = float((~sat_unsat).mean()) if sat_unsat.size else float("nan")

    true_rise_local = np.array(r0["true_rise_per_kc"], dtype=float)
    true_pop_rise = float(np.mean(true_rise_local))  # unweighted approximation, see module docstring note 9

    gates: dict = {}
    raw_stats: dict = {
        "pooled_bed_pvalue": pooled_bed_pvalue,
        "seed_pooled_exact": seed_pooled_exact,
        "discovery_fraction": discovery_fraction,
        "bank_recovery_corr": bank_recovery_corr,
        "r_rank_corr": r_rank_corr,
        "split_half_gap": split_half_gap,
        "saturation_flag_fraction": saturation_flag_fraction,
    }

    if twin == "syn_ng":
        gates["CG2"] = battery_mod.cg2_gate_false_positives(pooled_bed_pvalue, bh_majority)
    elif twin == "syn_kg":
        gates["CG3"] = battery_mod.cg3_passive_detection_kdd(
            pooled_bed_pvalue, discovery_fraction, r_rank_corr, split_half_gap, bank_recovery_corr
        )
    elif twin == "syn_ns":
        non_silent = ~silent_mask
        misfit_fire_fraction = float(misfit_fired[non_silent].mean()) if non_silent.any() else float("nan")
        unflagged_fraction = 1.0 - misfit_fire_fraction if np.isfinite(misfit_fire_fraction) else float("nan")
        gates["CG5"] = battery_mod.cg5_non_standard_shape(
            pooled_bed_pvalue, misfit_fire_fraction, unflagged_fraction, bank_recovery_corr
        )
        raw_stats["misfit_fire_fraction"] = misfit_fire_fraction

    if neural_cells:
        cg7_list = [n["tracker"]["cg7"]["passed"] for n in neural_cells]
        cg8_list = [n["tracker"]["cg8"]["passed"] for n in neural_cells]
        cg9_list = [n["tracker"]["cg9"]["passed"] for n in neural_cells]
        cg10_vals = [n["tracker"]["cg10_violation_fraction"] for n in neural_cells]
        gates["CG7"] = battery_mod.GateVerdict("CG7", bool(all(cg7_list)), {"per_seed": [n["tracker"]["cg7"] for n in neural_cells]})
        gates["CG8"] = battery_mod.GateVerdict("CG8", bool(all(cg8_list)), {"per_seed": [n["tracker"]["cg8"] for n in neural_cells]})
        gates["CG9"] = battery_mod.GateVerdict("CG9", bool(all(cg9_list)), {"per_seed": [n["tracker"]["cg9"] for n in neural_cells]})
        gates["CG10"] = battery_mod.cg10_direction_audit(float(np.nanmean(np.array(cg10_vals, dtype=float))))

        for variant in variants:
            pop_means = np.array([n["act"][variant]["pop_mean_rise"] for n in neural_cells], dtype=float)
            p95s = np.array([n["act"][variant]["p95_abs_rise"] for n in neural_cells], dtype=float)
            kc_rises = np.array([n["act"][variant]["kc_rise"] for n in neural_cells], dtype=float)
            kc_rise_mean = np.nanmean(kc_rises, axis=0)
            cg9_act_pass = bool(all(n["act"][variant]["cg9_act"]["passed"] for n in neural_cells))
            gates[f"CG9-ACT[{variant}]"] = battery_mod.GateVerdict(
                f"CG9-ACT[{variant}]", cg9_act_pass, {"per_seed": [n["act"][variant]["cg9_act"] for n in neural_cells]}
            )
            pop_mean_agg = float(np.nanmean(pop_means))
            p95_agg = float(np.nanmean(p95s))
            raw_stats.setdefault("act", {})[variant] = {
                "pop_mean_rise": pop_mean_agg, "p95_abs_rise": p95_agg,
                "kc_rise_mean": kc_rise_mean, "sign_consistent": battery_mod.is_sign_consistent(pop_means),
            }

            if twin == "syn_ng":
                gates[f"CG1[{variant}]"] = battery_mod.cg1_active_silence(pop_mean_agg, p95_agg, profile_name)
            elif twin == "syn_kg":
                null_ref = ng_ref["act"][variant]["pop_mean_rise"] if ng_ref else 0.0
                firing = active_mod.rb_a_firing(pop_means, kc_rises, null_bed_rise=null_ref)
                kc_rank_corr = bank_mod.spearman_rank_correlation(kc_rise_mean, true_rise_local)
                gates[f"CG1a[{variant}]"] = battery_mod.cg1a_active_positive_control(
                    firing.bed_fires, pop_mean_agg, true_pop_rise, kc_rank_corr, profile_name
                )
            elif twin == "syn_ns":
                silent_pop = float(np.nanmean(np.array([n["act"][variant]["silent_pop_mean_rise"] for n in neural_cells], dtype=float)))
                silent_p95 = float(np.nanmean(np.array([n["act"][variant]["silent_p95_abs_rise"] for n in neural_cells], dtype=float)))
                growing_corr = float(np.nanmean(np.array([n["act"][variant]["growing_rank_corr"] for n in neural_cells], dtype=float)))
                overshoot = float(np.nanmean(np.array([n["act"][variant]["overshoot_fraction"] for n in neural_cells], dtype=float)))
                gates[f"CG1b[{variant}]"] = battery_mod.cg1b_active_misfit_robustness(
                    silent_pop, silent_p95, growing_corr, overshoot
                )
            elif twin == "syn_sat":
                cg6 = battery_mod.cg6_saturation_refusal(
                    pooled_bed_pvalue, saturation_flag_fraction,
                    verdict_is_insufficient_range=bool(pooled_bed_pvalue > 0.05 and saturation_flag_fraction >= 0.95),
                )
                gates["CG6"] = cg6
                gates[f"CG1c[{variant}]"] = battery_mod.cg1c_active_saturation_refusal(
                    pop_mean_agg, true_pop_rise, p95_agg, true_pop_rise, verdict_is_insufficient_range=cg6.passed
                )

    return {"twin": twin, "gates": gates, "raw": raw_stats}


# =============================================================================
# Campaign orchestration
# =============================================================================

_TWIN_ORDER = ("syn_ng", "syn_kg", "syn_ns", "syn_sat")  # SYN-NG first: RB-A's null reference (note 5)


def run_campaign(cfg: RunConfig) -> dict:
    """Runs every requested twin's slice cells (all generator seeds) and
    neural cells (all model seeds, generator seed 0 only), certifies each
    twin, and assembles the final machine-readable verdict plus a compact
    markdown gate report. Twins are processed in ``_TWIN_ORDER`` so
    SYN-KG's CG1a can reference SYN-NG's already-computed aggregate."""
    twins = [t for t in _TWIN_ORDER if t in cfg.twins]
    t0 = time.time()
    per_twin: dict[str, dict] = {}
    ng_ref: Optional[dict] = None

    for twin in twins:
        slice_pairs = [run_slice_cell(cfg, twin, s) for s in cfg.generator_seeds]
        neural_cells = [run_neural_cell(cfg, twin, ms) for ms in cfg.model_seeds]
        cert = certify_twin(cfg, twin, slice_pairs, neural_cells, ng_ref=ng_ref if twin == "syn_kg" else None)
        per_twin[twin] = cert
        if twin == "syn_ng":
            ng_ref = cert["raw"]

    # K1 (posture machinery) and K5 (tracker leg): cross-twin kill conditions.
    def _gate(twin: str, name: str):
        g = per_twin.get(twin, {}).get("gates", {}).get(name)
        return g.passed if g is not None else False

    cg1_family_passed = all(
        _gate("syn_ng", f"CG1[{v}]") and _gate("syn_kg", f"CG1a[{v}]") and _gate("syn_ns", f"CG1b[{v}]") and _gate("syn_sat", f"CG1c[{v}]")
        for v in _act_variants(cfg)
    ) if all(t in per_twin for t in _TWIN_ORDER) else False
    cg2_passed = _gate("syn_ng", "CG2")
    cg3_passed = _gate("syn_kg", "CG3")
    cg5_passed = _gate("syn_ns", "CG5")
    cg6_passed = _gate("syn_sat", "CG6")
    k1 = battery_mod.k1_posture_machinery(
        cg1_family_passed, cg2_passed, cg5_passed, cg6_passed, cg3_passed, revision_budget_exhausted=False
    )
    cg7_any = any(per_twin[t]["gates"].get("CG7") and per_twin[t]["gates"]["CG7"].passed for t in per_twin)
    cg8_any = any(per_twin[t]["gates"].get("CG8") and per_twin[t]["gates"]["CG8"].passed for t in per_twin)
    k5 = battery_mod.k5_tracker_leg(cg7_any, cg8_any)

    all_gates = {}
    for twin, cert in per_twin.items():
        for name, g in cert["gates"].items():
            all_gates[f"{twin}.{name}"] = g
    kills = {"K1": k1, "K5": k5}

    verdict = report_mod.assemble_verdict_json(
        gates=all_gates, kills=kills,
        extra={
            "profile": cfg.profile.name, "twins": twins,
            "generator_seeds": list(cfg.generator_seeds), "model_seeds": list(cfg.model_seeds),
            "wall_clock_s": time.time() - t0,
            "raw_by_twin": {t: c["raw"] for t, c in per_twin.items()},
        },
    )
    out_dir = Path(cfg.output_dir) / cfg.profile.name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gate_verdict.json"
    md_path = out_dir / "gate_report.md"
    json_path.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    md_path.write_text(render_markdown(verdict))
    return {"verdict": verdict, "json_path": str(json_path), "md_path": str(md_path)}


# =============================================================================
# Markdown gate report
# =============================================================================


def render_markdown(verdict: dict) -> str:
    lines = ["# A4 synthetic certification -- gate report", ""]
    extra = verdict.get("extra", {})
    lines.append(f"Profile: `{extra.get('profile')}`  ")
    lines.append(f"Twins: {', '.join(extra.get('twins', []))}  ")
    lines.append(f"Generator seeds: {extra.get('generator_seeds')}  ")
    lines.append(f"Model seeds: {extra.get('model_seeds')}  ")
    lines.append(f"Wall clock: {extra.get('wall_clock_s', 0):.1f} s")
    lines.append("")
    lines.append("| Gate | Passed | Key detail |")
    lines.append("|---|---|---|")
    for name in sorted(verdict.get("gates", {})):
        g = verdict["gates"][name]
        passed = g.get("passed")
        detail = g.get("detail", {})
        detail_str = ", ".join(f"{k}={v}" for k, v in list(detail.items())[:3])
        mark = "PASS" if passed else "FAIL"
        lines.append(f"| {name} | {mark} | {detail_str} |")
    lines.append("")
    lines.append("| Kill condition | Triggered |")
    lines.append("|---|---|")
    for code, k in verdict.get("kills", {}).items():
        lines.append(f"| {code} | {'YES' if k.get('triggered') else 'no'} |")
    lines.append("")
    lines.append("Note: this is a TINY smoke-scale certification run (reduced KC/learner")
    lines.append("counts, replicate counts, and epoch budgets), not the pre-registered")
    lines.append("full-scale replicate campaign of design section 8 -- gate FAILs here are")
    lines.append("expected at this scale and certify the harness's plumbing, not G2 itself.")
    return "\n".join(lines) + "\n"


# =============================================================================
# Real-bed KDD wiring (additive; feature-flagged, harness task "real-bed
# bridge"). Nothing above this section is touched: `run_slice_cell`,
# `run_neural_cell`, `certify_twin`, `run_campaign`, and `RunConfig` are
# byte-for-byte unmodified, so every synthetic cell this harness already
# produces is untouched (verified by a before/after cell-hash diff,
# `_planning/design/realbed_bridge_notes.md`). This section wires a slice
# cell through the SAME measurement layer `run_slice_cell` uses (bank
# calibration, saturation, slices, PAS-G), sourced from
# `kt_mirt.growth.kc_data`'s real-bed loader instead of a synthetic twin
# (design section 8's R3 "KDD slice-based" step, the slice-only subset of
# it -- the heavier RB0 tri-spec refit and the full permutation battery
# are later stages, `scripts/a4/prep_kdd.py`, out of this build stage's
# permitted file set). `pandas` (`kc_data.py`'s own dependency, module
# docstring note 6 there) is imported lazily inside these functions only,
# so importing `run.py` for synthetic-only work never pays that cost.
# =============================================================================


class KddRealBedLoader:
    """A concrete `RealBedLoader` (the Protocol declared above) backed by
    `kt_mirt.growth.kc_data.load_kdd_kc_traced`. Feature-flagged and
    purely additive: constructing or calling this class has no effect on
    any synthetic code path above."""

    def __init__(
        self, path, kc_model: str = "KTracedSkills", chunksize: int = 500_000, nrows: Optional[int] = None,
    ) -> None:
        self.path = path
        self.kc_model = kc_model
        self.chunksize = chunksize
        self.nrows = nrows
        self.last_result = None  # set by `.load()`; carries hierarchy + qmatrix

    def load(self, seed: int):
        """Satisfies `RealBedLoader.load`. ``seed`` is accepted for
        Protocol conformance and reserved for a future seeded user-
        subsample (as EdNet's bed-table row anticipates); the current
        full-file KDD loader is deterministic regardless of seed."""
        from kt_mirt.growth import kc_data

        result = kc_data.load_kdd_kc_traced(
            self.path, kc_model=self.kc_model, chunksize=self.chunksize, nrows=self.nrows,
        )
        self.last_result = result
        return result.learners, result.n_learners, result.n_kcs


def run_kdd_slice_cell(cfg: RunConfig, loader: KddRealBedLoader, seed: int) -> dict:
    """Runs the bank calibration + saturation + slices + PAS-G measurement
    layer on a real-bed KDD draw (the same functions `run_slice_cell` above
    calls on synthetic twins: `bank_mod.calibrate_bank`, `saturation_stats`,
    `build_slices`, `gate_mod.compute_gate_result`). Real beds carry no
    generator ground truth, so the synthetic-only reads `run_slice_cell`'s
    result dict carries (`bank_recovery`, `true_rise_per_kc`,
    `silent_kc_mask`) have no analogue here and are simply absent; this
    function's result dict instead reports the Q-matrix pure-anchor
    statistics (`qmatrix.pure_anchor_stats`) the design's per-KC
    attribution claims are gated on (section 6).
    """
    from kt_mirt.growth import qmatrix as qmatrix_mod

    path = _cell_dir(cfg, "kdd_real") / f"slice_seed{seed}.json"
    cached = _load_if_done(path, cfg.force)
    if cached is not None:
        return cached

    learners, n_learners, n_kcs = loader.load(seed)
    hierarchy = loader.last_result.item_hierarchy

    rows_all = bank_mod.build_calibration_rows(learners)
    calib, analysis = bank_mod.split_learners(n_learners, seed=derive_seed("realbed_cohort", "kdd", seed))
    bank_cfg = bank_mod.BankModelConfig(
        n_epochs_max=cfg.bank_epochs, lr=0.05, batch_size=4096, patience_epochs=2,
        seed=derive_seed("realbed_bank", "kdd", seed), device=cfg.device,
    )
    fit = bank_mod.calibrate_bank(
        rows_all, n_learners, n_kcs, calib, hierarchy,
        bank_mod.KDD_HIERARCHY_SPEC, growth_mode="blockwise", config=bank_cfg,
    )
    frozen = bank_mod.freeze_bank(fit)

    calib_set = set(int(c) for c in calib.tolist())
    calib_learner_logs = [l for l in learners if l.learner in calib_set]
    rows_calib = bank_mod.build_calibration_rows(calib_learner_logs)
    sat_rate_calib, sat_unsat_calib = saturation_stats(rows_calib, n_kcs)

    analysis_set = set(int(a) for a in analysis.tolist())
    analysis_learners = [l for l in learners if l.learner in analysis_set]
    rows_analysis = bank_mod.build_calibration_rows(analysis_learners)
    sat_rate_analysis, sat_unsat_analysis = saturation_stats(rows_analysis, n_kcs)
    slices_analysis = build_slices(rows_analysis)

    # Saturation-aware bed statistic (gate.py module docstring note 6): on
    # the real bed the bed-level existence read is restricted to the
    # calibration-cohort unsaturated subset (design RB3: "bed-level pooled
    # gate p < 0.01 on the unsaturated subset"), so near-ceiling / mastered
    # KCs cannot spuriously fire the gate where there is no ground truth to
    # catch it. Whichever later stage computes this bed's permutation null
    # (e.g. via battery.run_permutation_battery) MUST pass this same
    # bed_kc_mask so observed and null bed statistics share one KC set.
    bed_kc_mask = np.asarray(sat_unsat_calib, dtype=bool)
    gate_result = gate_mod.compute_gate_result(
        slices_analysis, frozen, n_kcs, device=cfg.device, bed_kc_mask=bed_kc_mask
    )

    # Permutation null (mirrors run_slice_cell's synthetic-path block, gate.py
    # module docstring note 6): the observed existence statistic above is not
    # itself a verdict without a null to compare it to. Same bed_kc_mask goes
    # to both so the observed and null bed statistics sum over an identical
    # (calibration-cohort, permutation-invariant) KC subset.
    n_rep = max(cfg.n_perm_bed, cfg.n_perm_kc)
    null = gate_mod.permutation_null(
        analysis_learners, len(analysis_learners), n_kcs, frozen,
        n_replicates=n_rep, seed=derive_seed("realbed_perm", "kdd", seed), device=cfg.device,
        bed_kc_mask=bed_kc_mask,
    )
    bed_null = null["bed"][: cfg.n_perm_bed]
    kc_null = null["kc"][: cfg.n_perm_kc]
    bed_pvalue = gate_mod.empirical_pvalue(gate_result.bed_stat, bed_null)
    kc_pvalue = np.array(
        [gate_mod.empirical_pvalue(gate_result.kc_stat[c], kc_null[:, c]) for c in range(n_kcs)]
    )
    bh_reject = gate_mod.bh_fdr(kc_pvalue)

    result = {
        "bed": "kdd_real",
        "seed": seed,
        "n_learners": n_learners,
        "n_kcs": n_kcs,
        "n_analysis_learners": len(analysis_learners),
        "saturation": {
            "calib_rate": sat_rate_calib, "calib_unsaturated": sat_unsat_calib,
            "analysis_rate": sat_rate_analysis, "analysis_unsaturated": sat_unsat_analysis,
        },
        "gate": {
            "bed_stat": gate_result.bed_stat, "kc_stat": gate_result.kc_stat,
            "bed_pvalue": bed_pvalue, "bh_reject": bh_reject,
        },
        "pure_anchor_stats": qmatrix_mod.pure_anchor_stats(loader.last_result.qmatrix),
        "b_hat": fit.b_hat,
    }
    clean_result = json.loads(json.dumps(_jsonable(result)))
    _write_cell(path, clean_result)
    return clean_result


# =============================================================================
# Real-bed Junyi15 wiring (additive; feature-flagged, mirrors the KDD
# section immediately above). Nothing above this section (including the
# KDD real-bed wiring) is touched. This section wires a slice cell through
# the SAME measurement layer, sourced from `kt_mirt.growth.junyi_data`'s
# real-bed loader instead of a synthetic twin or the KDD bridge. Two
# differences from the KDD wiring, both `junyi_data.py` module docstring
# notes: (a) the item/KC vocabularies are the FULL exercise-table catalog,
# fixed before the log is streamed, not interned from the log itself; (b)
# the item hierarchy is FLAT (`bank.flat_hierarchy`/`FLAT_HIERARCHY_SPEC`,
# the EdNet/synthetic code path), since an exercise has no natural
# multi-level ancestry the way a KDD step does.
# =============================================================================


class JunyiRealBedLoader:
    """A concrete `RealBedLoader` (the Protocol declared above) backed by
    `kt_mirt.growth.junyi_data.load_junyi_kc_traced`. Feature-flagged and
    purely additive: constructing or calling this class has no effect on
    any other code path above."""

    def __init__(
        self, problem_log_path, exercise_table_path, chunksize: int = 2_000_000, nrows: Optional[int] = None,
        max_learners: Optional[int] = None,
    ) -> None:
        self.problem_log_path = problem_log_path
        self.exercise_table_path = exercise_table_path
        self.chunksize = chunksize
        self.nrows = nrows
        # Deterministic learner cap (junyi_data.py module docstring note
        # 10): the pilot-scale OOM fix for Junyi15's 247,606-learner cohort,
        # where a permutation-null replicate rebuilding every learner's
        # slices exceeds memory even with the null-chunk budget already in
        # place. ``None`` (default) loads every learner, byte-identical to
        # this parameter not existing.
        self.max_learners = max_learners
        self.last_result = None  # set by `.load()`; carries hierarchy + qmatrix

    def load(self, seed: int):
        """Satisfies `RealBedLoader.load`. ``seed`` is accepted for
        Protocol conformance and reserved for a future seeded user-
        subsample; the current full-file Junyi15 loader is deterministic
        regardless of seed (the `max_learners` cap, when set, is itself a
        fixed sorted-order rule, not seed-dependent)."""
        from kt_mirt.growth import junyi_data

        result = junyi_data.load_junyi_kc_traced(
            self.problem_log_path, self.exercise_table_path,
            chunksize=self.chunksize, nrows=self.nrows, max_learners=self.max_learners,
        )
        self.last_result = result
        return result.learners, result.n_learners, result.n_kcs


def run_junyi_slice_cell(cfg: RunConfig, loader: JunyiRealBedLoader, seed: int) -> dict:
    """Runs the bank calibration + saturation + slices + PAS-G measurement
    layer on a real-bed Junyi15 draw -- the Junyi analogue of
    `run_kdd_slice_cell` above (same functions, same result-dict shape),
    using a flat item hierarchy (`bank_mod.flat_hierarchy`/
    `FLAT_HIERARCHY_SPEC`) rather than KDD's 3-level one."""
    from kt_mirt.growth import qmatrix as qmatrix_mod

    path = _cell_dir(cfg, "junyi_real") / f"slice_seed{seed}.json"
    cached = _load_if_done(path, cfg.force)
    if cached is not None:
        return cached

    learners, n_learners, n_kcs = loader.load(seed)
    hierarchy = loader.last_result.item_hierarchy

    rows_all = bank_mod.build_calibration_rows(learners)
    calib, analysis = bank_mod.split_learners(n_learners, seed=derive_seed("realbed_cohort", "junyi", seed))
    bank_cfg = bank_mod.BankModelConfig(
        n_epochs_max=cfg.bank_epochs, lr=0.05, batch_size=4096, patience_epochs=2,
        seed=derive_seed("realbed_bank", "junyi", seed), device=cfg.device,
    )
    fit = bank_mod.calibrate_bank(
        rows_all, n_learners, n_kcs, calib, hierarchy,
        bank_mod.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=bank_cfg,
    )
    frozen = bank_mod.freeze_bank(fit)

    calib_set = set(int(c) for c in calib.tolist())
    calib_learner_logs = [l for l in learners if l.learner in calib_set]
    rows_calib = bank_mod.build_calibration_rows(calib_learner_logs)
    sat_rate_calib, sat_unsat_calib = saturation_stats(rows_calib, n_kcs)

    analysis_set = set(int(a) for a in analysis.tolist())
    analysis_learners = [l for l in learners if l.learner in analysis_set]
    rows_analysis = bank_mod.build_calibration_rows(analysis_learners)
    sat_rate_analysis, sat_unsat_analysis = saturation_stats(rows_analysis, n_kcs)
    slices_analysis = build_slices(rows_analysis)

    # Saturation-aware bed statistic (gate.py module docstring note 6): on
    # the real bed the bed-level existence read is restricted to the
    # calibration-cohort unsaturated subset (design RB3: "bed-level pooled
    # gate p < 0.01 on the unsaturated subset"), so near-ceiling / mastered
    # KCs cannot spuriously fire the gate where there is no ground truth to
    # catch it. Whichever later stage computes this bed's permutation null
    # (e.g. via battery.run_permutation_battery) MUST pass this same
    # bed_kc_mask so observed and null bed statistics share one KC set.
    bed_kc_mask = np.asarray(sat_unsat_calib, dtype=bool)
    gate_result = gate_mod.compute_gate_result(
        slices_analysis, frozen, n_kcs, device=cfg.device, bed_kc_mask=bed_kc_mask
    )

    # Permutation null (mirrors run_slice_cell's synthetic-path block, gate.py
    # module docstring note 6): the observed existence statistic above is not
    # itself a verdict without a null to compare it to. Same bed_kc_mask goes
    # to both so the observed and null bed statistics sum over an identical
    # (calibration-cohort, permutation-invariant) KC subset.
    n_rep = max(cfg.n_perm_bed, cfg.n_perm_kc)
    null = gate_mod.permutation_null(
        analysis_learners, len(analysis_learners), n_kcs, frozen,
        n_replicates=n_rep, seed=derive_seed("realbed_perm", "junyi", seed), device=cfg.device,
        bed_kc_mask=bed_kc_mask,
    )
    bed_null = null["bed"][: cfg.n_perm_bed]
    kc_null = null["kc"][: cfg.n_perm_kc]
    bed_pvalue = gate_mod.empirical_pvalue(gate_result.bed_stat, bed_null)
    kc_pvalue = np.array(
        [gate_mod.empirical_pvalue(gate_result.kc_stat[c], kc_null[:, c]) for c in range(n_kcs)]
    )
    bh_reject = gate_mod.bh_fdr(kc_pvalue)

    result = {
        "bed": "junyi_real",
        "seed": seed,
        "n_learners": n_learners,
        "n_kcs": n_kcs,
        "n_kcs_attempted": loader.last_result.n_kcs_attempted,
        "n_analysis_learners": len(analysis_learners),
        "saturation": {
            "calib_rate": sat_rate_calib, "calib_unsaturated": sat_unsat_calib,
            "analysis_rate": sat_rate_analysis, "analysis_unsaturated": sat_unsat_analysis,
        },
        "gate": {
            "bed_stat": gate_result.bed_stat, "kc_stat": gate_result.kc_stat,
            "bed_pvalue": bed_pvalue, "bh_reject": bh_reject,
        },
        "pure_anchor_stats": qmatrix_mod.pure_anchor_stats(loader.last_result.qmatrix),
        "b_hat": fit.b_hat,
    }
    clean_result = json.loads(json.dumps(_jsonable(result)))
    _write_cell(path, clean_result)
    return clean_result


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A4 synthetic certification campaign harness")
    p.add_argument("--output-dir", default="kt-mirt/outputs/a4")
    p.add_argument("--profile", choices=["tiny", "kdd", "ednet", "kdd_real", "junyi_real"], default="tiny")
    p.add_argument(
        "--kdd-path", default=None,
        help="path to the raw KDD Algebra 2008-2009 file (required with --profile kdd_real; "
             "feature-flagged real-bed slice cell via kt_mirt.growth.kc_data, see run_kdd_slice_cell)",
    )
    p.add_argument(
        "--junyi-log-path", default=None,
        help="path to junyi_ProblemLog_original.csv (required with --profile junyi_real; "
             "feature-flagged real-bed slice cell via kt_mirt.growth.junyi_data, see run_junyi_slice_cell)",
    )
    p.add_argument(
        "--junyi-exercise-path", default=None,
        help="path to junyi_Exercise_table.csv (required with --profile junyi_real)",
    )
    p.add_argument(
        "--junyi-max-learners", type=int, default=None,
        help="deterministically cap the number of distinct Junyi15 learners loaded "
             "(--profile junyi_real only; JunyiRealBedLoader's max_learners, see "
             "kt_mirt.growth.junyi_data.load_junyi_kc_traced): keeps the first N "
             "distinct user ids in sorted order via a memory-bounded pre-pass, so a "
             "pilot run's per-replicate memory scales with N rather than the full "
             "247,606-learner file; None (default) loads every learner",
    )
    p.add_argument("--n-kcs", type=int, default=None)
    p.add_argument("--n-learners", type=int, default=None)
    p.add_argument("--twins", nargs="+", default=list(synth_mod.TWIN_NAMES))
    p.add_argument("--generator-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-perm-bed", type=int, default=39)
    p.add_argument("--n-perm-kc", type=int, default=19)
    p.add_argument("--bank-epochs", type=int, default=40)
    p.add_argument("--tracker-epochs", type=int, default=15)
    p.add_argument("--act-epochs", type=int, default=20)
    p.add_argument("--n-reshuffles", type=int, default=3)
    p.add_argument("--drill-repeats", type=int, default=30)
    p.add_argument("--no-act-p1", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--device", choices=["cpu", "cuda"], default=DEVICE,
        help="compute device for this run (default cpu); 'cuda' is validated against "
             "torch.cuda.is_available() at RunConfig construction time",
    )
    return p


def main(argv=None) -> dict:
    args = build_arg_parser().parse_args(argv)
    if args.profile == "kdd_real":
        # Feature-flagged real-bed branch (additive; see run_kdd_slice_cell's
        # module-section docstring). Every other --profile value takes the
        # untouched synthetic path below, unchanged.
        if not args.kdd_path:
            raise ValueError("--kdd-path is required when --profile kdd_real")
        profile = make_profile("kdd_real", n_kcs=0, n_learners=0, kcs_per_learner=0.0)
        cfg = RunConfig(
            output_dir=Path(args.output_dir), profile=profile, force=args.force, device=args.device,
            n_perm_bed=args.n_perm_bed, n_perm_kc=args.n_perm_kc,
        )
        loader = KddRealBedLoader(args.kdd_path)
        result = run_kdd_slice_cell(cfg, loader, seed=0)
        cell_path = _cell_dir(cfg, "kdd_real") / "slice_seed0.json"
        print(json.dumps({"cell_path": str(cell_path)}, indent=2))
        return {"result": result, "cell_path": str(cell_path)}

    if args.profile == "junyi_real":
        # Feature-flagged real-bed branch (additive; see run_junyi_slice_cell's
        # module-section docstring). Every other --profile value takes the
        # untouched path above/below, unchanged.
        if not args.junyi_log_path or not args.junyi_exercise_path:
            raise ValueError(
                "--junyi-log-path and --junyi-exercise-path are both required with --profile junyi_real"
            )
        profile = make_profile("junyi_real", n_kcs=0, n_learners=0, kcs_per_learner=0.0)
        cfg = RunConfig(
            output_dir=Path(args.output_dir), profile=profile, force=args.force, device=args.device,
            n_perm_bed=args.n_perm_bed, n_perm_kc=args.n_perm_kc,
        )
        loader = JunyiRealBedLoader(
            args.junyi_log_path, args.junyi_exercise_path, max_learners=args.junyi_max_learners,
        )
        result = run_junyi_slice_cell(cfg, loader, seed=0)
        cell_path = _cell_dir(cfg, "junyi_real") / "slice_seed0.json"
        print(json.dumps({"cell_path": str(cell_path)}, indent=2))
        return {"result": result, "cell_path": str(cell_path)}

    base = {"tiny": TINY_PROFILE, "kdd": synth_mod.KDD_MATCHED, "ednet": synth_mod.EDNET_MATCHED}[args.profile]
    if args.n_kcs is not None or args.n_learners is not None:
        profile = make_profile(
            base.name, n_kcs=args.n_kcs or base.n_kcs, n_learners=args.n_learners or base.n_learners,
            kcs_per_learner=base.kcs_per_learner, base=base,
        )
    else:
        profile = base

    cfg = RunConfig(
        output_dir=Path(args.output_dir),
        profile=profile,
        twins=args.twins,
        generator_seeds=args.generator_seeds,
        model_seeds=args.model_seeds,
        n_perm_bed=args.n_perm_bed,
        n_perm_kc=args.n_perm_kc,
        bank_epochs=args.bank_epochs,
        tracker_epochs=args.tracker_epochs,
        act_epochs=args.act_epochs,
        n_reshuffles=args.n_reshuffles,
        drill_repeats=args.drill_repeats,
        run_act_p1=not args.no_act_p1,
        force=args.force,
        device=args.device,
    )
    result = run_campaign(cfg)
    print(json.dumps({"json_path": result["json_path"], "md_path": result["md_path"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
