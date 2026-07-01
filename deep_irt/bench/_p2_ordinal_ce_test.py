"""_p2_ordinal_ce_test.py -- Unit tests for the cumulative-link ordinal CE.

Headline assertion (blueprint section A): at K=2 the cumulative-link CE equals
the 2PL binary cross-entropy to machine precision, unifying the family.

Run:  python -m pytest deep_irt/bench/_p2_ordinal_ce_test.py -q
 or:  python deep_irt/bench/_p2_ordinal_ce_test.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from deep_irt.bench._p2_ordinal_ce import (
    cumulative_link_ce,
    gpcm_logits,
    ordinal_ce_from_params,
)


def test_k2_equals_2pl_binary_ce_mean():
    """K=2 cumulative-link CE == BCE-with-logits(z=psi_1-psi_0) exactly."""
    torch.manual_seed(0)
    N = 256
    psi = torch.randn(N, 2, dtype=torch.float64)     # arbitrary K=2 logits
    y = torch.randint(0, 2, (N,))

    ord_ce = cumulative_link_ce(psi, y, reduction="mean")

    # The codebase's 2PL binary loss: z = logit(class 1) - logit(class 0).
    z = psi[:, 1] - psi[:, 0]
    bce = F.binary_cross_entropy_with_logits(z, y.to(psi.dtype), reduction="mean")

    assert torch.allclose(ord_ce, bce, atol=1e-12, rtol=0), (
        f"K=2 ordinal CE {ord_ce.item():.15g} != 2PL BCE {bce.item():.15g}"
    )


def test_k2_equals_2pl_binary_ce_elementwise():
    """Per-example equality (reduction='none'), the strongest form."""
    torch.manual_seed(1)
    N = 128
    psi = torch.randn(N, 2, dtype=torch.float64)
    y = torch.randint(0, 2, (N,))

    ord_ce = cumulative_link_ce(psi, y, reduction="none")           # (N,)
    z = psi[:, 1] - psi[:, 0]
    bce = F.binary_cross_entropy_with_logits(
        z, y.to(psi.dtype), reduction="none"
    )
    assert torch.allclose(ord_ce, bce, atol=1e-12, rtol=0)


def test_k2_from_irt_params_matches_2pl():
    """Built from (theta, alpha, beta): K=2 psi -> ordinal CE == 2PL BCE."""
    torch.manual_seed(2)
    N = 200
    theta = torch.randn(N, dtype=torch.float64)
    alpha = F.softplus(torch.randn(N, 1, dtype=torch.float64))       # > 0
    beta = torch.randn(N, 1, dtype=torch.float64)                    # single threshold
    y = torch.randint(0, 2, (N,))

    ord_ce = ordinal_ce_from_params(theta, alpha, beta, y, reduction="mean")
    # 2PL logit z = alpha*(theta - beta).
    z = (alpha.squeeze(1) * (theta - beta.squeeze(1)))
    bce = F.binary_cross_entropy_with_logits(z, y.to(theta.dtype), reduction="mean")
    assert torch.allclose(ord_ce, bce, atol=1e-12, rtol=0)


def test_proper_scoring_minimum_at_truth():
    """Sanity: the loss is minimised when logits equal the true log-probs."""
    torch.manual_seed(3)
    N, K = 500, 4
    true_logits = torch.randn(N, K, dtype=torch.float64)
    probs = torch.softmax(true_logits, dim=1)
    y = torch.multinomial(probs, 1).squeeze(1)

    loss_true = cumulative_link_ce(true_logits, y)
    # Perturb -> loss should not decrease on average (proper scoring rule).
    worse = 0
    for s in range(20):
        g = torch.Generator().manual_seed(100 + s)
        pert = true_logits + 0.5 * torch.randn(N, K, generator=g, dtype=torch.float64)
        if cumulative_link_ce(pert, y) >= loss_true:
            worse += 1
    assert worse >= 18, f"only {worse}/20 perturbations were worse"


def test_shift_invariance():
    """Loss is invariant to adding a per-row constant to all K logits."""
    torch.manual_seed(4)
    N, K = 64, 4
    psi = torch.randn(N, K, dtype=torch.float64)
    y = torch.randint(0, K, (N,))
    shift = torch.randn(N, 1, dtype=torch.float64)
    a = cumulative_link_ce(psi, y, reduction="none")
    b = cumulative_link_ce(psi + shift, y, reduction="none")
    assert torch.allclose(a, b, atol=1e-12, rtol=0)


if __name__ == "__main__":
    test_k2_equals_2pl_binary_ce_mean()
    test_k2_equals_2pl_binary_ce_elementwise()
    test_k2_from_irt_params_matches_2pl()
    test_proper_scoring_minimum_at_truth()
    test_shift_invariance()
    print("ALL PASS: K=2 cumulative-link CE == 2PL binary CE to machine precision")
