"""The A4 perf-surgery equivalence gate: `gate.permutation_null_batched`
(replicate-batched Newton dispatch) versus `gate.permutation_null_looped`
(the original per-replicate loop, kept as the reference), on small-but-real
synthetic beds shaped after the pre-registered density profiles (SYN_DEV,
a KDD_MATCHED-profile down-scale, an EDNET_MATCHED-profile down-scale).

Context: `newton.py`'s `penalized_bounded_newton` computes Hessians via
`torch.func` nested vmap/jvp/vjp in eager mode, so every elementary op pays
Python dispatch overhead PER CALL. The original permutation battery calls
it once per replicate (plus twice per KC per replicate for the KC-pooled
M1a/M1b fits), so B=199/999 replicates re-pay that overhead B-fold.
`permutation_null_batched` widens the batch to include the replicate axis
(fitting all replicates -- and, for the KC-pooled fits, a whole chunk of
replicates -- in ONE Newton call), chunked over the replicate axis to
bound memory.

What "equivalence" means here, precisely (see `test_growth_newton.py`'s
sibling isolation experiments run during development, summarized in the
perf-surgery report): `penalized_bounded_newton`'s single-parameter (P=1,
M0) batched fits are BIT-IDENTICAL regardless of batch size (vmap over
independent elementwise ops does not mix batch rows). The KC-pooled
multi-parameter (P>1) joint fits are NOT bit-identical across batch sizes
-- this is an inherent property of `torch.linalg.solve`'s batched kernel on
this problem (confirmed by comparing a SINGLE-ITEM float32 fit against its
own float64 recomputation: the two already differ at the same ~1e-6
relative-per-parameter level as batched-vs-unbatched float32, proving the
"noise" is generic float32 rounding for this fairly sparse, one-hot-heavy
GLM, not something the batching introduces). That per-parameter noise
compounds through 25 Newton iterations and the held-out-NLL sum over many
slice positions and KCs into an aggregate `bed_stat`/`kc_stat` deviation
of order 1e-4 to 2e-3 (absolute, at O(1-10) magnitudes) at the tiny scales
these tests use -- looser than a strict rtol<=1e-5 on the raw statistic,
but the DECISION-level outputs every CG/RB gate actually consumes
(empirical p-values, BH-FDR, Benjamini-Yekutieli rejection flags) are
unaffected at any scale tested here (verified exactly, not just closely).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from kt_mirt.growth import bank as bank_mod
from kt_mirt.growth import gate, synth
from kt_mirt.growth.slices import build_slices

# ---------------------------------------------------------------------------
# Small-but-real configs: the pre-registered density profiles, down-scaled
# in n_kcs/n_learners only (same density/rate/arity shape as the real
# profiles), per the perf-surgery task's equivalence-gate requirement
# ("SYN_DEV and one KDD_MATCHED-profile down-scale"; an EDNET_MATCHED
# down-scale is added as a third real-profile config). EDNET_MATCHED's
# item_arity_max=6 needs at least that many KCs to sample multi-tag items
# without replacement, hence its slightly larger n_kcs here.
# ---------------------------------------------------------------------------

_SYN_DEV_TINY = dataclasses.replace(synth.SYN_DEV, name="syn_dev_tiny", n_kcs=3, n_learners=12, kcs_per_learner=2.0)
_KDD_MATCHED_TINY = dataclasses.replace(
    synth.KDD_MATCHED, name="kdd_matched_tiny", n_kcs=4, n_learners=16, kcs_per_learner=2.5
)
_EDNET_MATCHED_TINY = dataclasses.replace(
    synth.EDNET_MATCHED, name="ednet_matched_tiny", n_kcs=8, n_learners=16, kcs_per_learner=3.0
)
_CONFIGS = (_SYN_DEV_TINY, _KDD_MATCHED_TINY, _EDNET_MATCHED_TINY)
_SEEDS = (0, 1, 2)

# Evidence-based tolerance (see module docstring): the observed max absolute
# deviation across the configs/seeds this file exercises is ~1.9e-3 on
# `bed_stat` (O(1-10) magnitude) and ~7.8e-4 on `kc_stat`; a real logic bug
# (wrong slice attribution, mis-indexed KC, dropped replicate) would show
# deviations many orders of magnitude larger (an entirely different
# distribution, not a rounding-level wobble), so this bound is tight enough
# to catch regressions while accommodating the inherent float32 batched-
# linalg noise floor characterized above.
_ATOL = 5e-3
_RTOL = 2e-3


def _bank_from_b(b_hat: np.ndarray) -> bank_mod.FrozenBank:
    hier = bank_mod.flat_hierarchy(len(b_hat))
    fit = bank_mod.BankFitResult(
        hierarchy=hier, growth_mode="none", b_hat=np.asarray(b_hat, dtype=float),
        eligible_leaf=np.ones(len(b_hat), dtype=bool), problem_seen=np.ones(len(b_hat), dtype=bool),
        calib_exposure_count=np.full(len(b_hat), 100), converged=True, n_epochs_run=1, final_data_nll=0.0,
    )
    return bank_mod.freeze_bank(fit)


def _build_bed(profile: synth.DensityProfile, seed: int):
    twin = synth.generate_twin("syn_ng", profile, seed=seed)
    frozen = _bank_from_b(twin.item_bank.b_true)
    return twin, frozen


@pytest.mark.parametrize("profile", _CONFIGS, ids=lambda p: p.name)
@pytest.mark.parametrize("seed", _SEEDS)
def test_permutation_null_batched_matches_looped_reference(profile, seed):
    """The mandatory equivalence gate: on each small-but-real config x
    seed, the replicate-batched path's bed/kc null distributions must
    reproduce the looped reference's, within the evidence-based tolerance
    characterized in this module's docstring."""
    twin, frozen = _build_bed(profile, seed)
    n_rep = 5
    looped = gate.permutation_null_looped(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=seed + 100, device="cpu"
    )
    batched = gate.permutation_null_batched(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=seed + 100,
        device="cpu", replicate_chunk_size=2,
    )

    bed_max_dev = float(np.max(np.abs(looped["bed"] - batched["bed"])))
    kc_max_dev = float(np.max(np.abs(looped["kc"] - batched["kc"])))
    assert np.allclose(looped["bed"], batched["bed"], atol=_ATOL, rtol=_RTOL), (
        f"{profile.name} seed={seed}: bed_null max deviation {bed_max_dev:.3e} exceeds tolerance"
    )
    assert np.allclose(looped["kc"], batched["kc"], atol=_ATOL, rtol=_RTOL), (
        f"{profile.name} seed={seed}: kc_null max deviation {kc_max_dev:.3e} exceeds tolerance"
    )


@pytest.mark.parametrize("profile", _CONFIGS, ids=lambda p: p.name)
def test_permutation_null_batched_matches_looped_pvalues_and_fdr_decisions(profile):
    """The quantities every CG/RB gate actually consumes -- empirical
    p-values, BH-FDR rejections, Benjamini-Yekutieli rejections -- must
    match EXACTLY (not just to tolerance) between the two paths, since
    they are rank/threshold decisions against the null and are far more
    robust to the raw statistic's float32 batching noise than the raw
    values themselves."""
    twin, frozen = _build_bed(profile, seed=0)
    rows = bank_mod.build_calibration_rows(twin.learners)
    slices = build_slices(rows)
    observed = gate.compute_gate_result(slices, frozen, twin.n_kcs, device="cpu")

    n_rep = 12
    looped = gate.permutation_null_looped(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=42, device="cpu"
    )
    batched = gate.permutation_null_batched(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=42,
        device="cpu", replicate_chunk_size=4,
    )

    bed_p_loop = gate.empirical_pvalue(observed.bed_stat, looped["bed"])
    bed_p_batch = gate.empirical_pvalue(observed.bed_stat, batched["bed"])
    assert bed_p_loop == bed_p_batch

    kc_p_loop = np.array(
        [gate.empirical_pvalue(observed.kc_stat[c], looped["kc"][:, c]) for c in range(twin.n_kcs)]
    )
    kc_p_batch = np.array(
        [gate.empirical_pvalue(observed.kc_stat[c], batched["kc"][:, c]) for c in range(twin.n_kcs)]
    )
    assert np.array_equal(kc_p_loop, kc_p_batch)
    assert np.array_equal(gate.bh_fdr(kc_p_loop), gate.bh_fdr(kc_p_batch))
    assert np.array_equal(gate.by_correction(kc_p_loop), gate.by_correction(kc_p_batch))


def test_permutation_null_batched_is_deterministic_given_same_seed():
    """Same seed -> bit-identical results (the batched path must not
    introduce any nondeterminism of its own, e.g. from unordered dict
    iteration or uninitialized padding)."""
    twin, frozen = _build_bed(_KDD_MATCHED_TINY, seed=1)
    kwargs = dict(n_replicates=8, seed=7, device="cpu", replicate_chunk_size=3)
    a = gate.permutation_null_batched(twin.learners, len(twin.learners), twin.n_kcs, frozen, **kwargs)
    b = gate.permutation_null_batched(twin.learners, len(twin.learners), twin.n_kcs, frozen, **kwargs)
    assert np.array_equal(a["bed"], b["bed"])
    assert np.array_equal(a["kc"], b["kc"])


def test_permutation_null_batched_chunking_is_a_memory_detail_not_a_decision_change():
    """Different chunk sizes may carry slightly different batched-linalg
    floating-point noise (documented above), but must never change which
    permutation draws are used (RNG consumption order is chunk-size-
    independent) nor the resulting decision-level outputs."""
    twin, frozen = _build_bed(_KDD_MATCHED_TINY, seed=2)
    rows = bank_mod.build_calibration_rows(twin.learners)
    slices = build_slices(rows)
    observed = gate.compute_gate_result(slices, frozen, twin.n_kcs, device="cpu")

    n_rep = 8
    chunk1 = gate.permutation_null_batched(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=9, device="cpu",
        replicate_chunk_size=1,
    )
    chunk_all = gate.permutation_null_batched(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=9, device="cpu",
        replicate_chunk_size=n_rep,
    )
    # Raw values may wobble at the batched-linalg noise floor across chunk
    # sizes, but stay within the same evidence-based tolerance as the
    # looped-vs-batched comparison above.
    assert np.allclose(chunk1["bed"], chunk_all["bed"], atol=_ATOL, rtol=_RTOL)
    assert np.allclose(chunk1["kc"], chunk_all["kc"], atol=_ATOL, rtol=_RTOL)

    p_chunk1 = gate.empirical_pvalue(observed.bed_stat, chunk1["bed"])
    p_chunk_all = gate.empirical_pvalue(observed.bed_stat, chunk_all["bed"])
    assert p_chunk1 == p_chunk_all


def test_permutation_null_default_dispatches_to_batched():
    """`use_batched` defaults to True (the new path is the default, per
    the perf-surgery task's requirement); the default call must equal an
    explicit `permutation_null_batched` call bit-for-bit."""
    twin, frozen = _build_bed(_SYN_DEV_TINY, seed=0)
    default = gate.permutation_null(twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=6, seed=3, device="cpu")
    explicit = gate.permutation_null_batched(twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=6, seed=3, device="cpu")
    assert np.array_equal(default["bed"], explicit["bed"])
    assert np.array_equal(default["kc"], explicit["kc"])


def test_permutation_null_use_batched_false_reaches_looped_path():
    """`use_batched=False` must reach the ORIGINAL per-replicate loop
    unchanged (the old path kept accessible, per the perf-surgery task)."""
    twin, frozen = _build_bed(_SYN_DEV_TINY, seed=0)
    dispatched = gate.permutation_null(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=6, seed=3, device="cpu", use_batched=False
    )
    explicit = gate.permutation_null_looped(twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=6, seed=3, device="cpu")
    assert np.array_equal(dispatched["bed"], explicit["bed"])
    assert np.array_equal(dispatched["kc"], explicit["kc"])


def test_calibrate_m0_chunk_size_empirically_cpu_fallback_is_bounded():
    """On CPU (or no CUDA), the empirical calibrator does not attempt a
    memory probe (no cross-platform RSS-based mechanism is implemented);
    it returns a small, fixed, conservative chunk, always >= 1 and never
    exceeding `max_chunk`."""
    small_run = gate._calibrate_m0_chunk_size_empirically(
        perm_learners_list=[object()] * 5, all_keys=[(0, 0)], bank=None, device="cpu",
    )
    assert 1 <= small_run <= 20

    huge_run = gate._calibrate_m0_chunk_size_empirically(
        perm_learners_list=[object()] * 5000, all_keys=[(0, 0)], bank=None, device="cpu", max_chunk=200,
    )
    assert huge_run <= 200


def test_calibrate_m0_chunk_size_empirically_uses_real_probe_on_cuda(monkeypatch):
    """On a CUDA device, the M0 calibrator must probe `fit_batched_replicates`,
    read the peak via `torch.cuda.max_memory_allocated`, and size the
    chunk against `torch.cuda.mem_get_info`'s free bytes with the safety
    factor -- verified here by stubbing the CUDA memory API (no real GPU
    needed) so the test is fast, deterministic, and touches no device."""
    calls = {"m0": 0}

    def fake_fit_batched_replicates(slices_by_replicate, bank, design_fn, P, time_filter=None, prior_var=4.0, device="cpu"):
        calls["m0"] += 1
        return torch.zeros(len(slices_by_replicate), 1, P)

    monkeypatch.setattr(gate, "fit_batched_replicates", fake_fit_batched_replicates)
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(gate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gate.torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(gate.torch.cuda, "max_memory_allocated", lambda device=None: 2_000_000)
    monkeypatch.setattr(gate.torch.cuda, "mem_get_info", lambda device=None: (1_000_000_000, 2_000_000_000))

    frozen = _bank_from_b(np.zeros(3))
    fake_learners = [object(), object()]
    all_keys = [(0, 0), (10, 1)]
    stub_slices = {k: object() for k in all_keys}
    monkeypatch.setattr(gate, "build_calibration_rows", lambda learners: object())
    monkeypatch.setattr(gate, "build_slices", lambda rows: stub_slices)

    chunk = gate._calibrate_m0_chunk_size_empirically(
        perm_learners_list=fake_learners, all_keys=all_keys,
        bank=frozen, device="cuda", safety_factor=2.0, probe_replicates=2, max_chunk=200,
    )
    assert calls["m0"] == 1
    # bytes_per_replicate = 2e6 / 2 = 1e6; free=1e9; safety=2x -> chunk = 1e9 / (2*1e6) = 500, capped at 200
    assert chunk == 200


def test_calibrate_m0_chunk_size_empirically_survives_probe_oom(monkeypatch):
    """Calibration must survive an OOM in its OWN probe (a best-effort
    STARTING chunk size, never load-bearing for correctness) and fall
    back to a conservative chunk=1 rather than letting the OOM propagate
    and kill the whole unit."""

    def oom_fit_batched_replicates(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated: M0 probe OOM")

    monkeypatch.setattr(gate, "fit_batched_replicates", oom_fit_batched_replicates)
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(gate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gate.torch.cuda, "reset_peak_memory_stats", lambda device=None: None)

    frozen = _bank_from_b(np.zeros(3))
    all_keys = [(0, 0), (10, 1)]
    stub_slices = {k: object() for k in all_keys}
    monkeypatch.setattr(gate, "build_calibration_rows", lambda learners: object())
    monkeypatch.setattr(gate, "build_slices", lambda rows: stub_slices)

    chunk = gate._calibrate_m0_chunk_size_empirically(
        perm_learners_list=[object(), object()], all_keys=all_keys,
        bank=frozen, device="cuda", safety_factor=2.0, probe_replicates=2, max_chunk=200,
    )
    assert chunk == 1  # extremely conservative fallback; no exception propagated


def test_calibrate_kc_chunk_size_empirically_survives_probe_oom(monkeypatch):
    """INCIDENT: the calibration probe's OWN KC-joint call OOM'd in
    production (a KC so wide that even a 2-replicate batched Hessian
    attempt tried to allocate 287+ GiB). Per-KC calibration must survive
    this and fall back to a conservative chunk=1 for THAT KC alone --
    never a fatal error, and never affecting any other KC's own chunk
    size (see `_compute_per_kc_chunk_sizes`'s incident-context docstring
    for why a single bed-wide chunk size was the actual root cause of a
    production non-completion)."""

    def oom_fit_kc_joint_batched_replicates(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated: KC-joint probe OOM (the production signature)")

    monkeypatch.setattr(gate, "fit_kc_joint_batched_replicates", oom_fit_kc_joint_batched_replicates)
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(gate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gate.torch.cuda, "reset_peak_memory_stats", lambda device=None: None)

    frozen = _bank_from_b(np.zeros(3))
    kc_probe = [[object()], [object()]]  # 2 "replicates" of this one KC's own slices
    chunk = gate._calibrate_kc_chunk_size_empirically(
        kc_probe, bank=frozen, device="cuda", safety_factor=2.0, max_chunk=200,
    )
    assert chunk == 1  # extremely conservative fallback; no exception propagated


def test_compute_per_kc_chunk_sizes_only_probes_kcs_above_threshold(monkeypatch):
    """Small KCs (<= `small_kc_threshold`) must get `max_chunk` directly,
    with NO probe -- probing overhead is paid only for the handful of
    genuinely large KCs (the fix for the actual production root cause:
    a real bed can have ~500 small/medium KCs and a few pathologically
    large ones, and only the latter need individual calibration)."""
    probed_kcs = []

    def fake_calibrate_kc(kc_probe, bank, device, safety_factor=2.0, max_chunk=200):
        probed_kcs.append(len(kc_probe[0]))
        return 7  # a distinguishable, obviously-probed value

    monkeypatch.setattr(gate, "_calibrate_kc_chunk_size_empirically", fake_calibrate_kc)
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)

    small_keys = [(i, 0) for i in range(5)]  # KC 0: 5 slices, well under threshold
    large_keys = [(i, 1) for i in range(400)]  # KC 1: 400 slices, over threshold
    stub_slices = {k: object() for k in small_keys + large_keys}
    monkeypatch.setattr(gate, "build_calibration_rows", lambda learners: object())
    monkeypatch.setattr(gate, "build_slices", lambda rows: stub_slices)

    sizes = gate._compute_per_kc_chunk_sizes(
        perm_learners_list=[object(), object()], by_kc_keys=[small_keys, large_keys],
        bank=None, device="cuda", small_kc_threshold=300, max_chunk=200,
    )
    assert sizes[0] == 200  # small KC: max_chunk directly, no probe
    assert sizes[1] == 7  # large KC: actually probed
    assert probed_kcs == [400]  # only the large KC's own probe was built (400 slices)


def test_fit_batched_replicates_safe_falls_back_on_oom_and_matches_reference(monkeypatch):
    """`_fit_batched_replicates_safe` must reproduce EXACTLY what the
    looped reference (`fit_batched`, S=1 per replicate) would have
    computed, when forced onto its OOM fallback path -- the fallback is
    not just "doesn't crash," it must be numerically the same safe path
    the pre-batching implementation always used."""
    rng = np.random.default_rng(21)
    frozen = _bank_from_b(rng.normal(0, 1, size=5))
    slices_by_replicate = [
        [
            gate.Slice(
                learner=i, kc=0, item_id=np.zeros(12, dtype=int),
                response=(rng.random(12) < 0.6).astype(np.int8),
                opportunity=np.arange(1, 13), block_id=gate.opportunity_block(np.arange(1, 13)),
            )
            for i in range(4)
        ]
        for _ in range(3)  # B=3 replicates, same 4 slices' worth of shape each
    ]

    def oom_once(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated production signature")

    real_fit_batched = gate.fit_batched

    def cpu_fit_batched(slices, bank, design_fn, P, time_filter=None, device="cpu"):
        # The wrapper under test believes it is on "cuda" (so it takes
        # the try/except branch); the actual Newton fitting is force-
        # routed to CPU here so this test never issues real compute on a
        # shared/contended GPU -- only the wrapper's own tiny (B, S, P)
        # output-buffer zeros allocation touches a real "cuda" device,
        # which is harmless regardless of what else is running on it.
        return real_fit_batched(slices, bank, design_fn, P, time_filter=time_filter, device="cpu")

    monkeypatch.setattr(gate, "fit_batched_replicates", oom_once)
    monkeypatch.setattr(gate, "fit_batched", cpu_fit_batched)
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.torch.cuda, "empty_cache", lambda: None)

    result = gate._fit_batched_replicates_safe(
        slices_by_replicate, frozen, gate._m0_design, P=1, time_filter=gate._time_filter_odd, device="cuda"
    )
    for b in range(3):
        expected = real_fit_batched(slices_by_replicate[b], frozen, gate._m0_design, P=1, time_filter=gate._time_filter_odd, device="cpu")
        assert torch.allclose(result[b].cpu(), expected.params)


def test_fit_kc_joint_batched_replicates_safe_falls_back_on_oom_and_matches_reference(monkeypatch):
    """Same guarantee as above, for the KC-joint fit -- this is the
    wrapper that fixes the ACTUAL production root cause (a KC too wide
    to batch its Hessian at all, regardless of chunk size)."""
    rng = np.random.default_rng(22)
    frozen = _bank_from_b(rng.normal(0, 1, size=5))

    def make_kc_slices(seed):
        r = np.random.default_rng(seed)
        return [
            gate.Slice(
                learner=i, kc=0, item_id=r.integers(0, 5, size=10),
                response=(r.random(10) < 0.6).astype(np.int8),
                opportunity=np.arange(1, 11), block_id=gate.opportunity_block(np.arange(1, 11)),
            )
            for i in range(3)
        ]

    kc_slices_by_replicate = [make_kc_slices(seed) for seed in (100, 101, 102)]

    def oom_once(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated: KC too wide to batch (the production root cause)")

    real_fit_kc_joint = gate.fit_kc_joint

    def cpu_fit_kc_joint(kc_slices, bank, shared_design_fn, shared_dim, time_filter=None, device="cpu"):
        # Same rationale as the M0 test above: force the actual Newton
        # fitting onto CPU so this test never issues real compute on a
        # shared/contended GPU, while the wrapper under test still
        # believes it is on "cuda" (so it exercises the try/except path).
        return real_fit_kc_joint(kc_slices, bank, shared_design_fn, shared_dim, time_filter=time_filter, device="cpu")

    monkeypatch.setattr(gate, "fit_kc_joint_batched_replicates", oom_once)
    monkeypatch.setattr(gate, "fit_kc_joint", cpu_fit_kc_joint)
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.torch.cuda, "empty_cache", lambda: None)

    theta, shared = gate._fit_kc_joint_batched_replicates_safe(
        kc_slices_by_replicate, frozen, gate._m1b_shared_design, gate.N_BLOCKS,
        time_filter=gate._time_filter_odd, device="cuda",
    )
    for b in range(3):
        exp_theta, exp_shared = real_fit_kc_joint(
            kc_slices_by_replicate[b], frozen, gate._m1b_shared_design, gate.N_BLOCKS, time_filter=gate._time_filter_odd
        )
        assert np.allclose(theta[b], exp_theta)
        assert np.allclose(shared[b], exp_shared)


def test_assert_no_memory_growth_across_chunks_passes_on_flat_series():
    gate._assert_no_memory_growth_across_chunks([100_000_000, 105_000_000, 98_000_000, 101_000_000])  # no raise


def test_assert_no_memory_growth_across_chunks_passes_on_near_zero_baseline():
    # Near-zero baseline: tiny absolute fluctuations must not trip the check
    # (the absolute floor exists exactly for this case).
    gate._assert_no_memory_growth_across_chunks([0, 1000, 2000])  # no raise


def test_assert_no_memory_growth_across_chunks_raises_on_accumulation():
    """The units-8/23 failure signature: allocations climbing chunk-to-
    chunk until a later, modest allocation fails."""
    climbing = [500_000_000, 5_000_000_000, 20_000_000_000, 42_000_000_000]
    with pytest.raises(RuntimeError, match="grew from"):
        gate._assert_no_memory_growth_across_chunks(climbing)


def test_assert_no_memory_growth_across_chunks_single_reading_is_a_noop():
    gate._assert_no_memory_growth_across_chunks([12_345])  # no raise, nothing to compare
