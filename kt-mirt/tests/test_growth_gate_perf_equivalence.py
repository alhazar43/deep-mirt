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


def test_estimate_replicate_chunk_size_bounds():
    """Memory-aware chunk-size heuristic: always >= 1, never exceeds
    `max_chunk`, and shrinks as the per-replicate footprint (n_slices *
    T_max) grows."""
    small = gate._estimate_replicate_chunk_size(n_slices_bed=10, t_max_bed=20, device="cpu", target_bytes=1_000_000)
    large = gate._estimate_replicate_chunk_size(n_slices_bed=100_000, t_max_bed=60, device="cpu", target_bytes=1_000_000)
    assert small >= 1
    assert large >= 1
    assert small >= large  # a bigger bed footprint must not get a bigger (or equal-but-not-smaller) chunk
    capped = gate._estimate_replicate_chunk_size(
        n_slices_bed=1, t_max_bed=1, device="cpu", target_bytes=10**15, max_chunk=200
    )
    assert capped == 200
