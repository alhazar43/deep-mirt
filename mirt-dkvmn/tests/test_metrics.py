"""Metric computation tests."""

import torch

from mirt_dkvmn.utils.metrics import compute_metrics, confusion_matrix


def test_compute_metrics_basic():
    probs = torch.tensor([[[0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]])
    targets = torch.tensor([[1, 2]])
    metrics = compute_metrics(probs, targets)

    assert "cat_acc" in metrics
    assert "qwk" in metrics
    assert "mae" in metrics
    assert metrics["cat_acc"] == 1.0


def test_confusion_matrix():
    preds = torch.tensor([[0, 1, 1]])
    targets = torch.tensor([[0, 1, 2]])
    conf = confusion_matrix(preds, targets, n_cats=3)
    assert conf.shape == (3, 3)
    assert conf[0, 0] == 1
    assert conf[1, 1] == 1
    assert conf[2, 1] == 1
