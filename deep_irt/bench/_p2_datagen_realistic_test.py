"""_p2_datagen_realistic_test.py -- invariants for the step-6a realistic bed
and the real-data split-half reliability fill-in.

Run:  python -m pytest deep_irt/bench/_p2_datagen_realistic_test.py -q
 or:  python deep_irt/bench/_p2_datagen_realistic_test.py
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np

from deep_irt.bench import metrics_bench, nrm_metrics
from deep_irt.bench.datagen import BenchDataset
from deep_irt.bench._p2_datagen_realistic import generate_realistic
from deep_irt.bench import _p2_reliability as REL


# ---------------------------------------------------------------------------
# Minimal P2Config stand-in (avoids importing the full config for a shape test)
# ---------------------------------------------------------------------------

def _cfg(decoder="gpcm", kind="static", N=300, Q=40, T=30, K=4,
         admin_min=8, admin_max=25, dense=False, popularity="uniform",
         holdout_frac=0.2, n_holdout=5):
    data = types.SimpleNamespace(
        kind=kind, n_learners=N, n_items=Q, seq_len=T, n_cats=K,
        drift_sigma=0.15, alpha_theta_slope=0.0, n_holdout=n_holdout,
        train_frac=0.8, alpha_sigma=0.3, beta_sigma=1.0,
        regime="realistic", admin_min=admin_min, admin_max=admin_max,
        dense=dense, popularity=popularity, holdout_frac=holdout_frac,
    )
    model = types.SimpleNamespace(decoder=decoder)
    report = types.SimpleNamespace(cell_name="test_cell")
    return types.SimpleNamespace(data=data, model=model, report=report)


# ---------------------------------------------------------------------------
# Contract + shapes
# ---------------------------------------------------------------------------

def test_returns_benchdataset_with_expected_shapes():
    ds = generate_realistic(_cfg(), data_seed=0)
    assert isinstance(ds, BenchDataset)
    N, Q, T, K = 300, 40, 30, 4
    assert ds.items0.shape == (N, T)
    assert ds.responses.shape == (N, T)
    assert ds.aux["mask"].shape == (N, T)
    assert ds.theta_at_step.shape == (N, T)
    assert ds.items0.dtype == np.int64
    assert ds.responses.min() >= 0 and ds.responses.max() < K
    assert 0 <= ds.items0.min() and ds.items0.max() < Q


def test_aux_survives_dataclasses_replace():
    """The CV fold loop calls dataclasses.replace; aux (mask, seen) must survive."""
    ds = generate_realistic(_cfg(), data_seed=1)
    fold = dataclasses.replace(ds, train_idx=np.arange(10), val_idx=np.arange(10, 20))
    assert "mask" in fold.aux and fold.aux["mask"].shape == ds.aux["mask"].shape
    assert "seen_union" in fold.aux


# ---------------------------------------------------------------------------
# Variable length + mask
# ---------------------------------------------------------------------------

def test_mask_matches_lengths_and_is_variable():
    ds = generate_realistic(_cfg(T=30, admin_min=8, admin_max=25), data_seed=2)
    mask = ds.aux["mask"]
    lengths = ds.aux["lengths"]
    assert np.array_equal(mask.sum(axis=1), lengths)
    # count ~ Uniform[admin_min, admin_max]; each item answered once.
    assert lengths.min() >= 8 and lengths.max() <= 25
    assert (lengths < 30).any(), "variable length should produce short learners"
    # variability: the admin range spans several distinct lengths.
    assert np.unique(lengths).size > 1


def test_padding_responses_are_masked_zero():
    ds = generate_realistic(_cfg(), data_seed=3)
    mask = ds.aux["mask"]
    assert np.all(ds.responses[~mask] == 0)


def test_dense_control_is_rectangular():
    """dense=True administers ALL Q items to every learner (rectangular)."""
    ds = generate_realistic(_cfg(dense=True, Q=25, T=25), data_seed=4)
    assert np.all(ds.aux["mask"])
    assert np.all(ds.aux["lengths"] == 25)
    # every learner sees the whole bank
    assert np.all(ds.aux["exposure_counts"] == 25)


# ---------------------------------------------------------------------------
# The load-bearing seen-union invariant (blueprint seen-mask fix)
# ---------------------------------------------------------------------------

def test_padding_never_pollutes_seen_union():
    """unique(items0[rows]) (padding included, as _p2_run_cell computes it) must
    equal the administered union over those rows (valid positions only)."""
    ds = generate_realistic(_cfg(admin_min=8, admin_max=20, Q=40), data_seed=5)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(ds.items0.shape[0], size=50, replace=False))

    # As the runner derives it: over ALL columns including tail padding.
    seen_runner = np.zeros(ds.cfg.n_items, dtype=bool)
    seen_runner[np.unique(ds.items0[rows])] = True

    # Truth: union of administered (valid-only) items over the same rows.
    seen_truth = np.zeros(ds.cfg.n_items, dtype=bool)
    for i in rows:
        seen_truth[ds.aux["administered"][i]] = True

    assert np.array_equal(seen_runner, seen_truth), (
        "tail padding introduced an item not genuinely administered"
    )


def test_administered_is_subset_of_valid_items_per_learner():
    ds = generate_realistic(_cfg(admin_min=6, admin_max=18, Q=40), data_seed=6)
    mask = ds.aux["mask"]
    for i in range(ds.items0.shape[0]):
        valid_items = np.unique(ds.items0[i][mask[i]])
        assert np.array_equal(np.unique(ds.aux["administered"][i]), valid_items)


# ---------------------------------------------------------------------------
# Variable exposure
# ---------------------------------------------------------------------------

def test_sparse_exposure_limits_distinct_items():
    Q = 60
    ds = generate_realistic(_cfg(Q=Q, admin_min=10, admin_max=15, T=15),
                            data_seed=7)
    # each learner sees exactly its administered count (distinct, no replacement),
    # capped at admin_max and strictly below the whole bank.
    assert ds.aux["exposure_counts"].max() <= 15
    assert ds.aux["exposure_counts"].max() < Q
    assert np.array_equal(ds.aux["exposure_counts"], ds.aux["lengths"])


def test_coverage_counts_match_valid_positions():
    ds = generate_realistic(_cfg(), data_seed=8)
    mask = ds.aux["mask"]
    cov = np.bincount(ds.items0[mask], minlength=ds.cfg.n_items)
    assert np.array_equal(cov, ds.aux["coverage"])
    assert ds.aux["coverage"].sum() == int(mask.sum())


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    a = generate_realistic(_cfg(), data_seed=11)
    b = generate_realistic(_cfg(), data_seed=11)
    assert np.array_equal(a.items0, b.items0)
    assert np.array_equal(a.responses, b.responses)
    assert np.array_equal(a.aux["mask"], b.aux["mask"])


def test_different_seed_differs():
    a = generate_realistic(_cfg(), data_seed=11)
    b = generate_realistic(_cfg(), data_seed=12)
    assert not np.array_equal(a.responses, b.responses)


# ---------------------------------------------------------------------------
# GPCM ground truth
# ---------------------------------------------------------------------------

def test_gpcm_ground_truth_shapes_and_sorted_thresholds():
    ds = generate_realistic(_cfg(decoder="gpcm", K=4, Q=40), data_seed=9)
    assert ds.gt.a.shape == (40,)
    assert ds.gt.b.shape == (40, 3)
    assert np.all(np.diff(ds.gt.b, axis=1) >= 0), "step thresholds must be ascending"
    assert np.all(ds.gt.a > 0)


def test_binary_is_gpcm_k2():
    ds = generate_realistic(_cfg(decoder="binary", K=2, Q=30), data_seed=10)
    assert ds.gt.b.shape == (30, 1)
    assert ds.responses.max() <= 1


# ---------------------------------------------------------------------------
# NRM ground truth
# ---------------------------------------------------------------------------

def test_nrm_ground_truth_bock_centered_in_aux():
    ds = generate_realistic(_cfg(decoder="nrm", K=4, Q=40), data_seed=13)
    assert ds.aux["nrm_a"].shape == (40, 4)
    assert ds.aux["nrm_c"].shape == (40, 4)
    # Bock centering: options sum to ~0 per item.
    assert np.allclose(ds.aux["nrm_a"].sum(axis=1), 0.0, atol=1e-9)
    assert np.allclose(ds.aux["nrm_c"].sum(axis=1), 0.0, atol=1e-9)
    # gt.a / gt.b are inert placeholders for NRM.
    assert np.all(ds.gt.a == 0.0)


# ---------------------------------------------------------------------------
# Dynamic ability
# ---------------------------------------------------------------------------

def test_dynamic_theta_traj():
    ds = generate_realistic(_cfg(kind="dynamic", T=30), data_seed=14)
    assert ds.gt.theta_traj is not None
    assert ds.gt.theta_traj.shape == (300, 30)
    assert np.allclose(ds.gt.theta_traj[:, 0], ds.gt.theta0)
    # padded tail holds the last valid ability (GT net drift stays correct)
    mask = ds.aux["mask"]
    lengths = ds.aux["lengths"]
    short = np.where(lengths < 30)[0][0]
    Tn = int(lengths[short])
    assert np.allclose(ds.gt.theta_traj[short, Tn:], ds.gt.theta_traj[short, Tn - 1])


# ---------------------------------------------------------------------------
# Recovery scoreability on the seen set (the whole point of the bed)
# ---------------------------------------------------------------------------

def test_gpcm_recovery_scoreable_on_seen():
    ds = generate_realistic(_cfg(decoder="gpcm", admin_min=8, admin_max=20),
                            data_seed=15)
    train_rows = ds.train_idx
    seen = np.zeros(ds.cfg.n_items, dtype=bool)
    seen[np.unique(ds.items0[train_rows])] = True
    # ground truth vs itself -> perfect recovery on the seen items, and the call
    # must not choke on unseen items.
    out = metrics_bench.item_recovery(ds.gt.a, ds.gt.b, ds.gt.a, ds.gt.b, seen=seen)
    assert abs(out["a_spearman"]) > 0.999
    assert abs(out["b_spearman"]) > 0.999


def test_nrm_recovery_scoreable_on_seen():
    ds = generate_realistic(_cfg(decoder="nrm", admin_min=8, admin_max=20), data_seed=16)
    seen = ds.aux["seen_union"]
    a_k, c_k = ds.aux["nrm_a"], ds.aux["nrm_c"]
    out = nrm_metrics.item_recovery(a_k, c_k, a_k, c_k, seen=seen)
    assert abs(out["a_spearman"]) > 0.999
    assert abs(out["c_spearman"]) > 0.999


# ---------------------------------------------------------------------------
# Reliability fill-in: pure functions
# ---------------------------------------------------------------------------

def test_spearman_brown_known_values():
    assert abs(REL.spearman_brown(0.5) - 2.0 / 3.0) < 1e-12
    assert REL.spearman_brown(1.0) == 1.0
    assert np.isnan(REL.spearman_brown(float("nan")))
    assert np.isnan(REL.spearman_brown(-1.0))


def test_sign_aligned_corr():
    x = np.arange(20.0)
    assert abs(REL.sign_aligned_corr(x, x, "pearson") - 1.0) < 1e-9
    assert abs(REL.sign_aligned_corr(x, -x, "pearson") - 1.0) < 1e-9  # sign-free
    assert abs(REL.sign_aligned_corr(x, x, "spearman") - 1.0) < 1e-9
    assert np.isnan(REL.sign_aligned_corr(x, np.ones(20), "pearson"))


def test_item_bootstrap_ci_1d_and_2d():
    rng = np.random.default_rng(0)
    v = rng.normal(0.7, 0.1, size=200)
    out = REL.item_bootstrap_ci(v, n_boot=500, seed=0)
    assert out["ci_lo"] <= out["mean"] <= out["ci_hi"]
    assert abs(float(out["mean"]) - v.mean()) < 1e-9

    m = rng.normal(size=(50, 3))
    out2 = REL.item_bootstrap_ci(m, n_boot=300, seed=1)
    assert out2["mean"].shape == (3,)
    assert np.all(out2["ci_lo"] <= out2["mean"]) and np.all(out2["mean"] <= out2["ci_hi"])


def test_item_bootstrap_ci_nan_propagates():
    v = np.array([1.0, 2.0, np.nan, 4.0])
    out = REL.item_bootstrap_ci(v, n_boot=100)
    assert np.isnan(out["mean"])


def test_reliability_by_coverage_binning():
    Q = 100
    cov = np.arange(Q).astype(float)          # strictly increasing coverage
    rel = cov / Q                              # reliability rises with coverage
    out = REL.reliability_by_coverage(rel, cov, n_bins=5)
    assert sum(out["n_items"]) == Q
    finite = [m for m in out["mean_reliability"] if np.isfinite(m)]
    # low-coverage bin < high-coverage bin (the EdNet reversal direction)
    assert finite[0] < finite[-1]


def test_split_half_reliability_with_mock_engine():
    """Item-id-deterministic recovery -> both halves agree -> reliability ~ 1."""
    N, T, Q, K = 80, 20, 15, 4
    rng = np.random.default_rng(0)
    items = rng.integers(0, Q, size=(N, T))
    responses = rng.integers(0, K, size=(N, T))

    class _MockModel:
        _uses_state = False

        def __init__(self, Q, K):
            self.Q, self.K = Q, K

        def recover_item_params(self):
            # Pure function of item id -> identical across halves -> corr 1.
            a = np.linspace(0.5, 2.0, self.Q)
            b = np.tile(np.linspace(-1, 1, self.Q)[:, None], (1, self.K - 1))
            return {"alpha": a, "beta": b}

    class _MockEngine:
        def __init__(self, ds, **kw):
            self.ds = ds
            self.model = _MockModel(ds.cfg.n_items, ds.cfg.n_cats)

        def fit(self, **kw):
            return self

    out = REL.split_half_reliability(
        responses, items, _MockEngine, engine_kwargs={}, train_kwargs={},
        n_splits=3, seed=0, n_cats=K,
    )
    assert out["a_reliability"] > 0.999
    assert out["b_reliability"] > 0.999
    assert np.isnan(out["theta_reliability"])


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} PASS")
    sys.exit(0)
