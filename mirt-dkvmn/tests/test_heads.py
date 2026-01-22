"""Head shape tests."""

import torch

from mirt_dkvmn.models.heads.grm import GRMHead
from mirt_dkvmn.models.heads.nrm import NRMHead


def test_grm_shapes():
    head = GRMHead()
    theta = torch.randn(2, 3, 4)
    alpha = torch.randn(2, 3, 4)
    thresholds = torch.randn(2, 3, 3)
    probs = head(theta, alpha, thresholds)
    assert probs.shape == (2, 3, 4)


def test_nrm_shapes():
    head = NRMHead()
    theta = torch.randn(2, 3, 4)
    alpha_cat = torch.randn(2, 3, 5, 4)
    beta_cat = torch.randn(2, 3, 5)
    probs = head(theta, alpha_cat, beta_cat)
    assert probs.shape == (2, 3, 5)
