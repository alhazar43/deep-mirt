"""Tests for the E9 misspecification probe utilities."""

from __future__ import annotations

import numpy as np

from deep_irt.bench.datagen import BenchDataConfig
from deep_irt.bench.run_misspecification_probe import (
    aggregate,
    extreme_response_rate,
    generate_differential_item_functioning_dataset,
    generate_drifting_theta_dataset,
    generate_item_exposure_imbalance_dataset,
    generate_learner_response_style_dataset,
    generate_local_dependence_dataset,
    generate_noisy_threshold_dataset,
    generate_threshold_disorder_dataset,
    gpcm_logits,
    render_markdown,
    response_repeat_rate,
    sample_probs,
    softmax,
    item_exposure_summary,
    threshold_disorder_summary,
)


def test_gpcm_logits_softmax_and_sampling_are_well_formed():
    logits = gpcm_logits(theta=0.5, alpha=1.2, beta=np.array([-0.5, 0.2, 0.8]))
    probs = softmax(logits)

    assert logits.shape == (4,)
    assert probs.shape == (4,)
    assert np.all(probs > 0.0)
    assert np.isclose(probs.sum(), 1.0)
    assert sample_probs(probs, 0.0) == 0
    assert sample_probs(probs, 0.999999) == 3


def test_local_dependence_keeps_ground_truth_and_increases_repeats():
    cfg = BenchDataConfig(
        name="tiny_localdep",
        kind="static",
        n_learners=250,
        n_items=12,
        seq_len=24,
        n_cats=4,
        seed=11,
    )

    null = generate_local_dependence_dataset(cfg, strength=0.0)
    sticky = generate_local_dependence_dataset(cfg, strength=3.0)

    assert np.array_equal(null.items0, sticky.items0)
    assert np.array_equal(null.train_idx, sticky.train_idx)
    assert np.array_equal(null.val_idx, sticky.val_idx)
    assert np.allclose(null.gt.a, sticky.gt.a)
    assert np.allclose(null.gt.b, sticky.gt.b)
    assert null.responses.min() >= 0
    assert sticky.responses.max() < cfg.n_cats
    assert response_repeat_rate(sticky.responses) > response_repeat_rate(null.responses)


def test_noisy_thresholds_keep_ground_truth_but_change_responses():
    cfg = BenchDataConfig(
        name="tiny_noisy_thresholds",
        kind="static",
        n_learners=180,
        n_items=10,
        seq_len=20,
        n_cats=4,
        seed=7,
    )

    null = generate_noisy_threshold_dataset(cfg, strength=0.0)
    noisy = generate_noisy_threshold_dataset(cfg, strength=1.0)

    assert np.array_equal(null.items0, noisy.items0)
    assert np.array_equal(null.train_idx, noisy.train_idx)
    assert np.array_equal(null.val_idx, noisy.val_idx)
    assert np.allclose(null.gt.a, noisy.gt.a)
    assert np.allclose(null.gt.b, noisy.gt.b)
    assert noisy.responses.min() >= 0
    assert noisy.responses.max() < cfg.n_cats
    assert np.mean(null.responses != noisy.responses) > 0.05


def test_learner_response_style_tracks_extreme_response_rate():
    cfg = BenchDataConfig(
        name="tiny_response_style",
        kind="static",
        n_learners=300,
        n_items=12,
        seq_len=24,
        n_cats=4,
        seed=13,
    )

    null = generate_learner_response_style_dataset(cfg, strength=0.0)
    styled = generate_learner_response_style_dataset(cfg, strength=1.5)
    style = styled.aux["context_variables"]["response_style"]
    per_learner_extreme = np.mean(
        (styled.responses == 0) | (styled.responses == cfg.n_cats - 1),
        axis=1,
    )

    assert np.array_equal(null.items0, styled.items0)
    assert np.array_equal(null.train_idx, styled.train_idx)
    assert np.array_equal(null.val_idx, styled.val_idx)
    assert np.allclose(null.gt.a, styled.gt.a)
    assert np.allclose(null.gt.b, styled.gt.b)
    assert styled.responses.min() >= 0
    assert styled.responses.max() < cfg.n_cats
    assert np.mean(null.responses != styled.responses) > 0.05
    assert extreme_response_rate(styled.responses, cfg.n_cats) > 0.0
    assert np.corrcoef(style, per_learner_extreme)[0, 1] > 0.5


def test_differential_item_functioning_keeps_reference_and_plants_interaction():
    cfg = BenchDataConfig(
        name="tiny_dif",
        kind="static",
        n_learners=200,
        n_items=14,
        seq_len=20,
        n_cats=4,
        seed=19,
    )

    null = generate_differential_item_functioning_dataset(cfg, strength=0.0)
    dif = generate_differential_item_functioning_dataset(cfg, strength=1.25)
    group = dif.aux["context_variables"]["dif_group"]
    interaction = dif.aux["context_variables"]["dif_interaction"]
    item_loading = dif.aux["item_variables"]["dif_loading"]

    assert np.array_equal(null.items0, dif.items0)
    assert np.array_equal(null.train_idx, dif.train_idx)
    assert np.array_equal(null.val_idx, dif.val_idx)
    assert np.allclose(null.gt.a, dif.gt.a)
    assert np.allclose(null.gt.b, dif.gt.b)
    assert np.allclose(null.gt.theta0, dif.gt.theta0)
    assert np.isclose(group.mean(), 0.0)
    assert set(np.unique(group)) == {-1.0, 1.0}
    assert interaction.shape == dif.responses.shape
    assert np.allclose(interaction, group[:, None] * item_loading[dif.items0])
    assert np.all(np.diff(dif.gt.b, axis=1) >= 0.0)
    assert dif.responses.min() >= 0
    assert dif.responses.max() < cfg.n_cats
    assert np.mean(null.responses != dif.responses) > 0.05


def test_item_exposure_imbalance_keeps_truth_and_skews_counts():
    cfg = BenchDataConfig(
        name="tiny_exposure_imbalance",
        kind="static",
        n_learners=220,
        n_items=18,
        seq_len=24,
        n_cats=4,
        seed=23,
    )

    null = generate_item_exposure_imbalance_dataset(cfg, strength=0.0)
    skewed = generate_item_exposure_imbalance_dataset(cfg, strength=1.8)
    null_stats = item_exposure_summary(null.items0, cfg.n_items)
    skewed_stats = item_exposure_summary(skewed.items0, cfg.n_items)

    assert np.array_equal(null.train_idx, skewed.train_idx)
    assert np.array_equal(null.val_idx, skewed.val_idx)
    assert np.allclose(null.gt.a, skewed.gt.a)
    assert np.allclose(null.gt.b, skewed.gt.b)
    assert np.allclose(null.gt.theta0, skewed.gt.theta0)
    assert skewed_stats["item_exposure_cv"] > null_stats["item_exposure_cv"] + 0.5
    assert skewed_stats["item_exposure_max"] > 2.0 * skewed_stats["item_exposure_min"]
    assert np.mean(null.items0 != skewed.items0) > 0.05
    assert skewed.responses.min() >= 0
    assert skewed.responses.max() < cfg.n_cats
    assert "item_exposure_rate" in skewed.aux["item_variables"]


def test_drifting_theta_keeps_item_design_and_changes_theta_paths():
    cfg = BenchDataConfig(
        name="tiny_drifting_theta",
        kind="static",
        n_learners=160,
        n_items=12,
        seq_len=22,
        n_cats=4,
        seed=29,
    )

    null = generate_drifting_theta_dataset(cfg, strength=0.0)
    drifted = generate_drifting_theta_dataset(cfg, strength=0.35)
    drift = drifted.theta_at_step - drifted.gt.theta0[:, None]

    assert null.cfg.kind == "dynamic"
    assert drifted.cfg.kind == "dynamic"
    assert np.array_equal(null.items0, drifted.items0)
    assert np.array_equal(null.train_idx, drifted.train_idx)
    assert np.array_equal(null.val_idx, drifted.val_idx)
    assert np.allclose(null.gt.a, drifted.gt.a)
    assert np.allclose(null.gt.b, drifted.gt.b)
    assert np.allclose(null.gt.theta0, drifted.gt.theta0)
    assert np.allclose(null.theta_at_step, null.gt.theta0[:, None])
    assert drifted.aux["theta_drift_step_sigma"] == 0.35
    assert drifted.aux["theta_drift_path_std"] > 0.0
    assert np.std(drift[:, -1]) > np.std(drift[:, 1])
    assert drifted.responses.min() >= 0
    assert drifted.responses.max() < cfg.n_cats
    assert np.mean(null.responses != drifted.responses) > 0.05


def test_threshold_disorder_keeps_design_and_creates_unsorted_thresholds():
    cfg = BenchDataConfig(
        name="tiny_threshold_disorder",
        kind="static",
        n_learners=160,
        n_items=16,
        seq_len=18,
        n_cats=4,
        seed=17,
    )

    null = generate_threshold_disorder_dataset(cfg, strength=0.0)
    disordered = generate_threshold_disorder_dataset(cfg, strength=2.0)
    rate, magnitude = threshold_disorder_summary(disordered.gt.b)

    assert np.array_equal(null.items0, disordered.items0)
    assert np.array_equal(null.train_idx, disordered.train_idx)
    assert np.array_equal(null.val_idx, disordered.val_idx)
    assert np.allclose(null.gt.a, disordered.gt.a)
    assert np.allclose(null.gt.theta0, disordered.gt.theta0)
    assert threshold_disorder_summary(null.gt.b) == (0.0, 0.0)
    assert rate > 0.25
    assert magnitude > 0.0
    assert disordered.responses.min() >= 0
    assert disordered.responses.max() < cfg.n_cats
    assert np.mean(null.responses != disordered.responses) > 0.05


def test_aggregate_and_markdown_include_absorption_fields():
    row = {
        "misspecification": "local_dependence",
        "strength": 1.0,
        "variant": "state_alpha",
        "variant_description": "state-conditioned alpha, static beta",
        "K": 4,
        "N": 20,
        "Q": 5,
        "T": 6,
        "strongest_corr_variable": "prev_response",
        "variables": [
            {
                "variable": "prev_response",
                "pearson": 0.25,
                "spearman": 0.2,
                "cubic_r2": float("nan"),
            }
        ],
    }
    for key in [
        "acc",
        "qwk",
        "auc",
        "theta_spearman",
        "theta_pearson",
        "theta_netdrift_spearman",
        "theta_netdrift_pearson",
        "theta_pooled_pearson",
        "a_spearman",
        "a_pearson",
        "a_high_spearman",
        "a_high_pearson",
        "b_spearman",
        "b_pearson",
        "alpha_p50",
        "alpha_p95",
        "alpha_max",
        "alpha_grad_norm_mean",
        "delta_std",
        "delta_abs_mean",
        "strongest_abs_corr",
        "max_cubic_r2",
        "corr_delta_prev_response",
        "corr_delta_info_model",
        "corr_delta_theta_model",
        "corr_delta_history_pos",
        "corr_delta_exposure_count",
        "corr_delta_response_style",
        "corr_delta_abs_response_style",
        "corr_delta_dif_group",
        "corr_delta_dif_interaction",
        "corr_delta_abs_dif_interaction",
        "corr_delta_dif_loading",
        "corr_delta_abs_dif_loading",
        "corr_delta_item_exposure_count_total",
        "corr_delta_item_exposure_rate",
        "corr_delta_log_item_exposure_rate",
        "corr_delta_item_popularity_score",
        "corr_delta_theta_drift",
        "corr_delta_abs_theta_drift",
        "corr_delta_theta_step_change",
        "corr_delta_threshold_disorder",
        "corr_delta_threshold_disorder_magnitude",
        "beta_delta_std",
        "beta_delta_prev_response_corr",
        "beta_delta_history_pos_corr",
        "beta_delta_response_style_corr",
        "beta_delta_abs_response_style_corr",
        "beta_delta_dif_group_corr",
        "beta_delta_dif_interaction_corr",
        "beta_delta_abs_dif_interaction_corr",
        "beta_delta_dif_loading_corr",
        "beta_delta_abs_dif_loading_corr",
        "beta_delta_item_exposure_count_total_corr",
        "beta_delta_item_exposure_rate_corr",
        "beta_delta_log_item_exposure_rate_corr",
        "beta_delta_item_popularity_score_corr",
        "beta_delta_theta_drift_corr",
        "beta_delta_abs_theta_drift_corr",
        "beta_delta_theta_step_change_corr",
        "beta_delta_threshold_disorder_corr",
        "beta_delta_threshold_disorder_magnitude_corr",
        "theta_prev_response_corr",
        "theta_history_pos_corr",
        "theta_response_style_corr",
        "theta_abs_response_style_corr",
        "theta_dif_group_corr",
        "theta_dif_interaction_corr",
        "theta_abs_dif_interaction_corr",
        "theta_dif_loading_corr",
        "theta_abs_dif_loading_corr",
        "theta_item_exposure_count_total_corr",
        "theta_item_exposure_rate_corr",
        "theta_log_item_exposure_rate_corr",
        "theta_item_popularity_score_corr",
        "theta_theta_drift_corr",
        "theta_abs_theta_drift_corr",
        "theta_theta_step_change_corr",
        "theta_threshold_disorder_corr",
        "theta_threshold_disorder_magnitude_corr",
        "response_repeat_rate",
        "extreme_response_rate",
        "response_abs_step_change",
        "threshold_disorder_rate",
        "threshold_disorder_magnitude_mean",
        "dif_group_mean",
        "dif_loading_abs_mean",
        "dif_threshold_shift_abs_mean",
        "theta_drift_step_sigma",
        "theta_drift_path_std",
        "theta_drift_abs_mean",
        "theta_drift_final_std",
        "item_exposure_cv",
        "item_exposure_gini",
        "item_exposure_min",
        "item_exposure_max",
        "item_exposure_min_rate",
        "item_exposure_max_rate",
        "low_exposure_items",
        "high_exposure_items",
        "a_low_exposure_spearman",
        "a_low_exposure_pearson",
        "a_high_exposure_spearman",
        "a_high_exposure_pearson",
        "b_low_exposure_spearman",
        "b_low_exposure_pearson",
        "b_high_exposure_spearman",
        "b_high_exposure_pearson",
        "n_params",
        "train_time",
        "final_loss",
    ]:
        row[key] = 0.1

    agg, var_agg = aggregate([row])
    md = render_markdown(
        {
            "mode": "TEST",
            "device": "cpu",
            "misspecification": "local_dependence",
            "K": 4,
            "N": 20,
            "Q": 5,
            "T": 6,
            "strengths": [1.0],
            "variants": ["state_alpha"],
            "seeds": [0],
            "epochs": 1,
            "lr": 0.01,
        },
        agg,
        var_agg,
    )

    assert agg[0]["strongest_corr_variable_mode"] == "prev_response"
    assert "corr(delta, prev)" in md
    assert "corr(delta, exposure)" in md
    assert "corr(delta, drift)" in md
    assert "theta_pool" in md
    assert "corr(delta, style)" in md
    assert "corr(delta, DIF)" in md
    assert "corr(delta, disorder)" in md
    assert "prev_response" in md
