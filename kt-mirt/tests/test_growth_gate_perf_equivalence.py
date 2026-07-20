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
from kt_mirt.growth.slices import build_slices, permute_learner_order

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


# =============================================================================
# KC-joint arrow-structured fast path (2026-07-20, 4th A4 perf-surgery,
# `_planning/LEDGER.md`): `fit_kc_joint`/`fit_kc_joint_batched_replicates`'s
# default `use_arrow=True` (exact Schur-complement block elimination) versus
# `use_arrow=False` (the original dense one-hot-design path, kept as the
# equivalence reference). Ladder rung 2's "dense path vs Schur path produce
# identical iterates ... on small and MEDIUM (S~500) shapes" requirement --
# distinct from (and much tighter than) the batched-vs-looped noise floor
# documented at the top of this file: arrow-vs-dense is two representations
# of the IDENTICAL Hessian (exact block elimination), not two different
# batch-size linalg kernel calls, so agreement is expected near the float32
# noise floor (~1e-6 relative), not the 1e-3-ish aggregate noise above.
# =============================================================================


def _make_kc_slices(n_slices: int, seed: int, T: int = 12, item_pool: int = 20) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_slices):
        opp = np.arange(1, T + 1)
        out.append(
            gate.Slice(
                learner=i,
                kc=0,
                item_id=rng.integers(0, item_pool, size=T),
                response=(rng.random(T) < 0.55).astype(np.int8),
                opportunity=opp,
                block_id=gate.opportunity_block(opp),
            )
        )
    return out


@pytest.mark.parametrize(
    "shared_design_fn,shared_dim",
    [(gate._m1a_shared_design, 1), (gate._m1b_shared_design, gate.N_BLOCKS)],
    ids=["m1a_pooled_shared_dim1", "m1b_pooled_shared_dim4"],
)
@pytest.mark.parametrize("n_slices", [5, 30, 500], ids=["small5", "small30", "medium500"])
def test_fit_kc_joint_arrow_matches_dense_reference(n_slices, shared_design_fn, shared_dim):
    """Ladder rung 2: `fit_kc_joint`'s arrow path (default `use_arrow=True`)
    must reproduce the dense one-hot-design reference (`use_arrow=False`)
    at small (5, 30) and MEDIUM (500) slice counts, for both M1a-pooled
    (shared_dim=1) and M1b-pooled (shared_dim=N_BLOCKS=4) shapes.

    Tolerance note (2026-07-20 4th A4 perf-surgery phase 3): the arrow
    path's internal `torch.matmul` (replacing `torch.einsum`, which paid
    `opt_einsum` contraction-path-search overhead on every Newton
    iteration at production scale -- confirmed by direct cProfile) sums
    the SAME quantities in a different floating-point order than the
    dense path's own einsum/autodiff route. At n_slices=5 with
    shared_dim=4 (5 slices' worth of data identifying a 4-wide shared
    block -- a comparatively weakly-identified regime), this measured up
    to ~5e-4 absolute on theta_ic (scale ~1-2); every other (n_slices,
    shared_dim) combination in this test stays at or below ~1e-7. This is
    the same float32-noise-floor phenomenon this file's module docstring
    already documents for batched-vs-looped comparisons, one level down
    (arrow-vs-dense, not batched-vs-looped) -- exactness itself is
    verified separately and tightly in `tests/test_growth_newton.py`
    (rtol=1e-6 against a dense `torch.linalg.solve` on the identical
    matrix, which the matmul rewrite does not change)."""
    kc_slices = _make_kc_slices(n_slices, seed=100 + n_slices)
    theta_arrow, shared_arrow = gate.fit_kc_joint(
        kc_slices, None, shared_design_fn, shared_dim, time_filter=gate._time_filter_odd, use_arrow=True
    )
    theta_dense, shared_dense = gate.fit_kc_joint(
        kc_slices, None, shared_design_fn, shared_dim, time_filter=gate._time_filter_odd, use_arrow=False
    )
    assert np.allclose(theta_arrow, theta_dense, rtol=1e-3, atol=1e-3), (
        f"n_slices={n_slices}: theta_ic max abs diff {np.abs(theta_arrow - theta_dense).max():.3e}"
    )
    assert np.allclose(shared_arrow, shared_dense, rtol=1e-3, atol=1e-3), (
        f"n_slices={n_slices}: shared max abs diff {np.abs(shared_arrow - shared_dense).max():.3e}"
    )


@pytest.mark.parametrize("n_slices", [8, 200], ids=["small8", "medium200"])
def test_fit_kc_joint_batched_replicates_arrow_matches_dense_reference(n_slices):
    """Same equivalence, batched over replicates (the permutation
    battery's actual call shape): B=4 replicates of the SAME ``n_slices``
    KC, arrow vs dense. Unlike the single-replicate (S=1) test above, BOTH
    sides here go through a batched (B>1) `torch.linalg.solve`/Schur solve,
    which carries its own float32 batched-kernel noise regardless of
    arrow-vs-dense (this file's module docstring already characterizes
    that noise at up to ~2e-3 for the bed/kc aggregates it was written
    for); the tolerance here reuses that same evidence-based bound rather
    than the tighter single-replicate one, since the noise source is
    identical."""
    B = 4
    kc_slices_by_replicate = [_make_kc_slices(n_slices, seed=200 + n_slices + b) for b in range(B)]
    theta_arrow, shared_arrow = gate.fit_kc_joint_batched_replicates(
        kc_slices_by_replicate, None, gate._m1b_shared_design, gate.N_BLOCKS,
        time_filter=gate._time_filter_odd, use_arrow=True,
    )
    theta_dense, shared_dense = gate.fit_kc_joint_batched_replicates(
        kc_slices_by_replicate, None, gate._m1b_shared_design, gate.N_BLOCKS,
        time_filter=gate._time_filter_odd, use_arrow=False,
    )
    assert np.allclose(theta_arrow, theta_dense, rtol=_RTOL, atol=_ATOL), (
        f"n_slices={n_slices}: theta_ic max abs diff {np.abs(theta_arrow - theta_dense).max():.3e}"
    )
    assert np.allclose(shared_arrow, shared_dense, rtol=_RTOL, atol=_ATOL), (
        f"n_slices={n_slices}: shared max abs diff {np.abs(shared_arrow - shared_dense).max():.3e}"
    )


def test_build_kc_joint_arrow_arrays_matches_dense_design_construction():
    """CRITICAL CORRECTNESS CHECK at the gate.py level: the compact arrow
    representation (`slice_idx`, `shared_cols`) must encode EXACTLY the
    same design as `_build_kc_joint_arrays`'s dense one-hot-bordered
    matrix -- `design[:, :k] == one_hot(slice_idx, k)` and
    `design[:, k:] == shared_cols`, row for row."""
    kc_slices = _make_kc_slices(25, seed=55)
    y_d, mask_d, logit_d, design = gate._build_kc_joint_arrays(
        kc_slices, None, gate._m1b_shared_design, gate.N_BLOCKS, gate._time_filter_odd
    )
    y_a, mask_a, logit_a, slice_idx, shared_cols = gate._build_kc_joint_arrow_arrays(
        kc_slices, None, gate._m1b_shared_design, gate.N_BLOCKS, gate._time_filter_odd
    )
    k = len(kc_slices)
    assert np.array_equal(y_d, y_a)
    assert np.array_equal(mask_d, mask_a)
    assert np.array_equal(logit_d, logit_a)
    onehot = np.zeros((len(y_d), k), dtype=np.float32)
    onehot[np.arange(len(y_d)), slice_idx] = 1.0
    assert np.array_equal(design[:, :k], onehot)
    assert np.allclose(design[:, k:], shared_cols)


# =============================================================================
# Vectorized replicate assembly (2026-07-20, 4th A4 perf-surgery, phases 2-3,
# `_planning/LEDGER.md`): node-level profiling of a running production proof
# unit found the permutation battery GPU-idle and single-CPU-core-bound the
# entire time -- the arrow fix above made the SOLVE fast, but every replicate
# still paid several O(n_slices) PYTHON-LEVEL loops
# (`build_slices`/`_pad_design`/`_build_kc_joint_arrow_arrays`) to rebuild its
# data from scratch. `permutation_null_batched`'s new default
# (`use_vectorized_assembly=True`) replaces all of that with vectorized numpy
# scatter-assigns straight from `bank.build_calibration_rows`'s own output,
# plus (phase 3) batching MULTIPLE KCs' pooled fits into one Newton call
# (`_fit_kc_bucket_pooled_and_held_out_vectorized`) to remove per-KC GPU-
# dispatch overhead. These tests check the new primitives directly against
# the ORIGINAL Slice-based reference at the array level (not just via the
# end-to-end permutation_null_batched-vs-looped tests above, which already
# exercise this path by default but wouldn't localize a bug to a specific
# function).
# =============================================================================


def _make_bed(n_learners, n_kcs, seed, T_range=(4, 15), A=2, n_items=25):
    """A small-but-real multi-tag bed via `LearnerLog`-shaped objects
    (`bank.build_calibration_rows`'s own input contract), for testing the
    vectorized assembly primitives directly against `build_slices`."""

    class _LG:
        def __init__(self, learner, item_ids, responses, tag_ids, tag_mask):
            self.learner = learner
            self.item_ids = item_ids
            self.responses = responses
            self.tag_ids = tag_ids
            self.tag_mask = tag_mask

    r = np.random.default_rng(seed)
    learners = []
    for i in range(n_learners):
        T = r.integers(*T_range)
        item_ids = r.integers(0, n_items, size=T)
        responses = (r.random(T) < 0.5).astype(np.int8)
        tag_ids = r.integers(0, n_kcs, size=(T, A))
        tag_mask = r.random((T, A)) < 0.7
        tag_mask[:, 0] = True  # at least one real tag per row
        learners.append(_LG(i, item_ids, responses, tag_ids, tag_mask))
    return learners, n_items


def _make_frozen_bank(n_items, seed):
    rng = np.random.default_rng(seed)
    b_hat = rng.normal(0, 1, size=n_items)
    hier = bank_mod.flat_hierarchy(n_items)
    fit = bank_mod.BankFitResult(
        hierarchy=hier, growth_mode="none", b_hat=b_hat,
        eligible_leaf=np.ones(n_items, dtype=bool), problem_seen=np.ones(n_items, dtype=bool),
        calib_exposure_count=np.full(n_items, 100), converged=True, n_epochs_run=1, final_data_nll=0.0,
    )
    return bank_mod.freeze_bank(fit)


def test_vectorized_replicate_padded_arrays_matches_build_slices():
    """CRITICAL CORRECTNESS CHECK: the vectorized scatter must reproduce
    `build_slices`'s per-slice (response, opportunity, block_id, item_id)
    sequences and validity exactly -- no per-slice/per-group Python loop,
    same data."""
    learners, n_items = _make_bed(n_learners=8, n_kcs=5, seed=3)
    rows = bank_mod.build_calibration_rows(learners)
    sd = build_slices(rows)
    all_keys = list(sd.keys())
    T_max = max(sl.T for sl in sd.values())
    S = len(all_keys)
    key_to_col = gate._bed_key_scaffold(all_keys, n_kcs=5, n_learners=8)

    y, valid_pos, opp, block, item = gate._vectorized_replicate_padded_arrays(rows, key_to_col, 5, S, T_max)

    for col, key in enumerate(all_keys):
        sl = sd[key]
        t = sl.T
        assert np.array_equal(y[col, :t], sl.response.astype(np.float32))
        assert np.array_equal(opp[col, :t], sl.opportunity)
        assert np.array_equal(block[col, :t], sl.block_id)
        assert np.array_equal(item[col, :t], sl.item_id)
        assert valid_pos[col, :t].all()
        assert not valid_pos[col, t:].any()


def test_fit_m0_and_held_out_vectorized_matches_reference():
    """`_fit_m0_and_held_out_vectorized` must reproduce
    `_fit_batched_replicates_safe` + `held_out_nll`'s held-out NLL exactly
    (both use `binary_cross_entropy_with_logits`, same formula)."""
    learners, n_items = _make_bed(n_learners=10, n_kcs=5, seed=42)
    frozen = _make_frozen_bank(n_items, seed=7)
    perm_rng = np.random.default_rng(0)
    Bc = 6
    chunk_learners = [permute_learner_order(learners, perm_rng) for _ in range(Bc)]
    chunk_rows = [bank_mod.build_calibration_rows(lg) for lg in chunk_learners]
    chunk_slices = [gate.build_slices(r) for r in chunk_rows]
    all_keys = list(chunk_slices[0].keys())
    for sd in chunk_slices:
        assert set(sd.keys()) == set(all_keys)
    S = len(all_keys)
    T_max = max(sl.T for sd in chunk_slices for sl in sd.values())
    key_to_col = gate._bed_key_scaffold(all_keys, n_kcs=5, n_learners=10)

    y, valid_pos, opp, block, item = gate._stack_chunk_padded_arrays(chunk_rows, key_to_col, 5, S, T_max)

    slices_by_replicate_all = [[sd[k] for k in all_keys] for sd in chunk_slices]
    m0_params = gate._fit_batched_replicates_safe(
        slices_by_replicate_all, frozen, gate._m0_design, P=1, time_filter=gate._time_filter_odd, device="cpu"
    )
    m0_nll_even_ref = np.stack(
        [
            gate.held_out_nll(
                slices_by_replicate_all[b], frozen, gate._m0_design, m0_params[b], gate._time_filter_even,
                device="cpu",
            )
            .cpu()
            .numpy()
            for b in range(Bc)
        ]
    )
    m0_nll_even_new = gate._fit_m0_and_held_out_vectorized(y, opp, valid_pos, item, frozen, "cpu")
    assert np.allclose(m0_nll_even_ref, m0_nll_even_new, atol=1e-5, rtol=1e-5), (
        f"max abs dev {np.abs(m0_nll_even_ref - m0_nll_even_new).max():.3e}"
    )


@pytest.mark.parametrize("shared_dim", [1, gate.N_BLOCKS], ids=["m1a_shared_dim1", "m1b_shared_dim4"])
def test_fit_kc_pooled_and_held_out_vectorized_matches_reference(shared_dim):
    """`_fit_kc_pooled_and_held_out_vectorized` must reproduce
    `_fit_kc_joint_batched_replicates_safe` (arrow path) +
    `held_out_total_nll_kc_joint_batched_replicates`'s held-out NLL within
    the established batched-linalg noise floor."""
    shared_design_fn = gate._m1a_shared_design if shared_dim == 1 else gate._m1b_shared_design
    learners, n_items = _make_bed(n_learners=10, n_kcs=5, seed=42)
    frozen = _make_frozen_bank(n_items, seed=7)
    perm_rng = np.random.default_rng(0)
    Bc = 6
    chunk_learners = [permute_learner_order(learners, perm_rng) for _ in range(Bc)]
    chunk_rows = [bank_mod.build_calibration_rows(lg) for lg in chunk_learners]
    chunk_slices = [gate.build_slices(r) for r in chunk_rows]
    all_keys = list(chunk_slices[0].keys())
    S = len(all_keys)
    T_max = max(sl.T for sd in chunk_slices for sl in sd.values())
    key_to_col = gate._bed_key_scaffold(all_keys, n_kcs=5, n_learners=10)
    y, valid_pos, opp, block, item = gate._stack_chunk_padded_arrays(chunk_rows, key_to_col, 5, S, T_max)

    by_kc_keys = [[] for _ in range(5)]
    for key in all_keys:
        by_kc_keys[key[1]].append(key)

    for c in range(5):
        keys_c = by_kc_keys[c]
        if not keys_c:
            continue
        kc_slices_by_replicate = [[sd[k] for k in keys_c] for sd in chunk_slices]
        theta_ref, shared_ref = gate._fit_kc_joint_batched_replicates_safe(
            kc_slices_by_replicate, frozen, shared_design_fn, shared_dim, time_filter=gate._time_filter_odd,
            device="cpu",
        )
        nll_ref = gate.held_out_total_nll_kc_joint_batched_replicates(
            kc_slices_by_replicate, frozen, shared_design_fn, theta_ref, shared_ref, gate._time_filter_even
        )
        cols = np.array([key_to_col[k[0] * 5 + k[1]] for k in keys_c])
        nll_new = gate._fit_kc_pooled_and_held_out_vectorized(
            y, opp, block, valid_pos, item, cols, shared_design_fn, shared_dim, frozen, "cpu"
        )
        assert np.allclose(nll_ref, nll_new, atol=2e-3, rtol=2e-3), (
            f"KC {c} (k={len(keys_c)}): max abs dev {np.abs(nll_ref - nll_new).max():.3e}"
        )


@pytest.mark.parametrize("shared_dim", [1, gate.N_BLOCKS], ids=["m1a_shared_dim1", "m1b_shared_dim4"])
def test_fit_kc_bucket_pooled_matches_individual_per_kc_path(shared_dim):
    """CRITICAL CORRECTNESS CHECK: batching MULTIPLE, DIFFERENTLY-SIZED KCs
    into ONE Newton call (padding every KC to the bucket's own k_max) must
    reproduce `_fit_kc_pooled_and_held_out_vectorized`'s per-KC results
    exactly -- this is the padding/masking logic that must never let one
    KC's fit see another KC's data, nor let padding rows perturb a real
    KC's own fit."""
    shared_design_fn = gate._m1a_shared_design if shared_dim == 1 else gate._m1b_shared_design
    learners, n_items = _make_bed(n_learners=10, n_kcs=6, seed=42)
    frozen = _make_frozen_bank(n_items, seed=7)
    perm_rng = np.random.default_rng(0)
    Bc = 6
    chunk_learners = [permute_learner_order(learners, perm_rng) for _ in range(Bc)]
    chunk_rows = [bank_mod.build_calibration_rows(lg) for lg in chunk_learners]
    chunk_slices = [gate.build_slices(r) for r in chunk_rows]
    all_keys = list(chunk_slices[0].keys())
    S = len(all_keys)
    T_max = max(sl.T for sd in chunk_slices for sl in sd.values())
    key_to_col = gate._bed_key_scaffold(all_keys, n_kcs=6, n_learners=10)
    y, valid_pos, opp, block, item = gate._stack_chunk_padded_arrays(chunk_rows, key_to_col, 6, S, T_max)

    by_kc_keys = [[] for _ in range(6)]
    for key in all_keys:
        by_kc_keys[key[1]].append(key)
    nonempty_kcs = [c for c in range(6) if by_kc_keys[c]]
    cols_by_kc = {c: np.array([key_to_col[k[0] * 6 + k[1]] for k in by_kc_keys[c]]) for c in nonempty_kcs}

    ref = {
        c: gate._fit_kc_pooled_and_held_out_vectorized(
            y, opp, block, valid_pos, item, cols_by_kc[c], shared_design_fn, shared_dim, frozen, "cpu"
        )
        for c in nonempty_kcs
    }
    bucket_results = gate._fit_kc_bucket_pooled_and_held_out_vectorized(
        y, opp, block, valid_pos, item, [cols_by_kc[c] for c in nonempty_kcs], shared_design_fn, shared_dim,
        frozen, "cpu",
    )
    for j, c in enumerate(nonempty_kcs):
        assert np.allclose(ref[c], bucket_results[j], atol=1e-5, rtol=1e-5), (
            f"KC {c} (k={len(by_kc_keys[c])}): max abs dev {np.abs(ref[c] - bucket_results[j]).max():.3e}"
        )


def test_bucket_kcs_by_size_splits_small_and_large():
    """KCs at/below `small_kc_threshold` are grouped into `bucket_size`-
    sized buckets (sorted ascending by k, so members are similarly
    sized); KCs above it each get their own singleton bucket (routed by
    the caller to the ORIGINAL per-KC path, never batched with others)."""
    by_kc_keys = [
        [(i, 0) for i in range(5)],     # KC 0: k=5 (small)
        [(i, 1) for i in range(400)],   # KC 1: k=400 (large)
        [(i, 2) for i in range(10)],    # KC 2: k=10 (small)
        [],                              # KC 3: empty, dropped
        [(i, 4) for i in range(500)],   # KC 4: k=500 (large)
    ]
    buckets = gate._bucket_kcs_by_size(by_kc_keys, small_kc_threshold=300, bucket_size=25)
    singleton_large = {b[0] for b in buckets if len(b) == 1 and b[0] in (1, 4)}
    assert singleton_large == {1, 4}
    multi_member_kcs = {c for b in buckets if len(b) > 1 for c in b}
    assert multi_member_kcs == {0, 2}
    assert 3 not in [c for b in buckets for c in b]


def test_permutation_null_batched_vectorized_assembly_matches_slice_based_path():
    """End-to-end: `permutation_null_batched`'s new default
    (`use_vectorized_assembly=True`, which now also bucket-batches
    multiple KCs per Newton call) must reproduce
    `use_vectorized_assembly=False` (the ORIGINAL per-replicate
    Slice-based path) within the established batched-linalg noise floor,
    on a real (down-scaled) synthetic bed."""
    twin, frozen = _build_bed(_KDD_MATCHED_TINY, seed=3)
    n_rep = 10
    vectorized = gate.permutation_null_batched(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=11, device="cpu",
        use_vectorized_assembly=True,
    )
    original = gate.permutation_null_batched(
        twin.learners, len(twin.learners), twin.n_kcs, frozen, n_replicates=n_rep, seed=11, device="cpu",
        use_vectorized_assembly=False,
    )
    assert np.allclose(vectorized["bed"], original["bed"], atol=_ATOL, rtol=_RTOL), (
        f"bed_null max dev {np.abs(vectorized['bed'] - original['bed']).max():.3e}"
    )
    assert np.allclose(vectorized["kc"], original["kc"], atol=_ATOL, rtol=_RTOL), (
        f"kc_null max dev {np.abs(vectorized['kc'] - original['kc']).max():.3e}"
    )
