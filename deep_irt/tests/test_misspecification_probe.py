"""Tests for the E9 misspecification probe utilities."""

from __future__ import annotations

import numpy as np

from deep_irt.bench.datagen import BenchDataConfig
from deep_irt.bench.run_misspecification_probe import (
    aggregate,
    extreme_response_rate,
    generate_learner_response_style_dataset,
    generate_local_dependence_dataset,
    generate_noisy_threshold_dataset,
    generate_threshold_disorder_dataset,
    gpcm_logits,
    render_markdown,
    response_repeat_rate,
    sample_probs,
    softmax,
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
        "beta_delta_std",
        "beta_delta_prev_response_corr",
        "beta_delta_history_pos_corr",
        "beta_delta_response_style_corr",
        "beta_delta_abs_response_style_corr",
        "theta_prev_response_corr",
        "theta_history_pos_corr",
        "theta_response_style_corr",
        "theta_abs_response_style_corr",
        "response_repeat_rate",
        "extreme_response_rate",
        "response_abs_step_change",
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
    assert "corr(delta, style)" in md
    assert "prev_response" in md
