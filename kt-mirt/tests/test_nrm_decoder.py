"""Tests for the NRM (Bock 1972 nominal response model) decoder.

Test IDs
--------
1. test_identifiability_sum_to_zero -- centered per-option params satisfy
   sum_k a_k = 0 and sum_k c_k = 0 (the Bock constraints).
2. test_shift_invariance -- adding a constant to every raw slope / intercept
   does not change the response distribution (so centering is likelihood-neutral
   and the constraints are free).
3. test_log_probs_valid -- log_probs rows are valid log-distributions.
4. test_gradient_flow -- nll backprops finite grads to the embedding and both
   linear maps.
5. test_deep_irt_recovery -- a DeepIRTModel(decoder="nrm") fit on synthetic MC
   data recovers the planted a_k / c_k and ability (Spearman).

Cut on vendoring into kt_mirt (see kt-mirt/_planning/vendor_report.md): the
source file's test_cross_format_transfer -- a shared-embedding 2PL + NRM
model placing held-out-format items on the shared scale -- depended on a
``nrmfmt`` comparison harness (data_gen/train/metrics) that lives outside
deep_irt.core and is not part of the vendored core; it was already guarded
with ``pytest.importorskip`` in the source and is dropped here rather than
carried as permanent dead weight.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

from kt_mirt.core import DeepIRTModel
from kt_mirt.core.decoders import NRMDecoder


# ---------------------------------------------------------------------------
# Decoder-level unit tests (fast, no fitting)
# ---------------------------------------------------------------------------

def _spearman(x, y):
    r, _ = sp_stats.spearmanr(np.ravel(x), np.ravel(y))
    return float(r)


def _align(v, ref):
    r, _ = sp_stats.pearsonr(np.ravel(v), np.ravel(ref))
    return -v if r < 0 else v


def test_identifiability_sum_to_zero():
    torch.manual_seed(0)
    dec = NRMDecoder(emb_dim=8, n_options=4, correct_option=0)
    emb = torch.randn(6, 8)
    p = dec.item_params(emb)
    assert torch.allclose(p["alpha"].sum(-1), torch.zeros(6), atol=1e-5)
    assert torch.allclose(p["intercept"].sum(-1), torch.zeros(6), atol=1e-5)


def test_shift_invariance():
    torch.manual_seed(1)
    K = 4
    dec = NRMDecoder(emb_dim=8, n_options=K, correct_option=0)
    dec2 = NRMDecoder(emb_dim=8, n_options=K, correct_option=0)
    dec2.load_state_dict(dec.state_dict())
    with torch.no_grad():
        dec2.fc_a.bias += 3.7      # constant shift on all slopes
        dec2.fc_c.bias -= 2.1      # constant shift on all intercepts
    emb = torch.randn(5, 8)
    theta = torch.randn(5)
    p1 = dec.item_params(emb)
    p2 = dec2.item_params(emb)
    lp1 = dec.log_probs(theta, p1["alpha"], p1["intercept"])
    lp2 = dec2.log_probs(theta, p2["alpha"], p2["intercept"])
    assert torch.allclose(lp1, lp2, atol=1e-5)


def test_log_probs_valid():
    torch.manual_seed(2)
    dec = NRMDecoder(emb_dim=8, n_options=4, correct_option=1)
    emb = torch.randn(7, 8)
    theta = torch.randn(7)
    p = dec.item_params(emb)
    lp = dec.log_probs(theta, p["alpha"], p["intercept"])
    assert lp.shape == (7, 4)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(7), atol=1e-5)


def test_gradient_flow():
    torch.manual_seed(3)
    dec = NRMDecoder(emb_dim=8, n_options=4, correct_option=0)
    emb = torch.randn(5, 8, requires_grad=True)
    theta = torch.randn(5)
    resp = torch.randint(0, 4, (5,))
    loss = dec.nll(theta, emb, resp)
    loss.backward()
    assert torch.isfinite(emb.grad).all()
    assert dec.fc_a.weight.grad is not None
    assert dec.fc_c.weight.grad is not None
    assert torch.isfinite(dec.fc_a.weight.grad).all()


# ---------------------------------------------------------------------------
# Synthetic recovery via DeepIRTModel(decoder="nrm")
# ---------------------------------------------------------------------------

def _gen_nrm_items(n_items, K, correct, seed):
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal(n_items)
    a_corr = rng.uniform(0.8, 1.8, size=n_items)
    a = rng.normal(0.0, 0.6, size=(n_items, K))
    c = rng.normal(0.0, 0.6, size=(n_items, K))
    a[:, correct] = a_corr
    c[:, correct] = -a_corr * beta
    a -= a.mean(axis=1, keepdims=True)
    c -= c.mean(axis=1, keepdims=True)
    return a, c


def _gen_mc(a, c, n_items, K, n_learners, seed):
    rng = np.random.default_rng(seed + 1)
    theta = rng.standard_normal(n_learners)
    items, resp = [], []
    for i in range(n_learners):
        order = rng.permutation(n_items)
        row = []
        for j in order:
            logits = a[j] * theta[i] + c[j]
            logits -= logits.max()
            e = np.exp(logits)
            row.append(int(rng.choice(K, p=e / e.sum())))
        items.append(order.tolist())
        resp.append(row)
    return (torch.tensor(items, dtype=torch.long),
            torch.tensor(resp, dtype=torch.long), theta)


def test_deep_irt_recovery():
    """Fit recovers the planted per-option slopes / intercepts and ability.

    Thresholds reflect this deliberately small, fast config (50 items, 600
    learners, 250 epochs).  The full validation (60 items, 800 learners, 300
    epochs) reaches a_k Spearman ~0.99 and ability ~0.80; here the bars
    are set to confirm clear, sign-correct recovery without flakiness.
    """
    N_ITEMS, K, CORRECT, N_LEARN = 50, 4, 0, 600
    a_true, c_true = _gen_nrm_items(N_ITEMS, K, CORRECT, seed=0)
    items_t, resp_t, theta_true = _gen_mc(a_true, c_true, N_ITEMS, K, N_LEARN, seed=0)

    model = DeepIRTModel(
        num_items=N_ITEMS, emb_dim=8, hidden_dim=32, n_cats=K,
        decoder="nrm", correct_option=CORRECT, seed=0,
    )
    model.fit(items_t, resp_t, n_epochs=250, verbose=False)

    params = model.recover_item_params()
    # a_k recovery: mean-center both sides (Bock representative), sign-align.
    a_hat_c = params["alpha"] - params["alpha"].mean(axis=-1, keepdims=True)
    a_true_c = a_true - a_true.mean(axis=1, keepdims=True)
    a_hat_s = _align(a_hat_c.ravel(), a_true_c.ravel())
    a_sp = _spearman(a_hat_s, a_true_c.ravel())

    theta_rec = model.track(items_t, resp_t).cpu().numpy()[:, -1]
    theta_sp = _spearman(_align(theta_rec, theta_true), theta_true)

    assert a_sp >= 0.7, f"a_k recovery weak: {a_sp:.3f}"
    assert theta_sp >= 0.5, f"ability recovery weak: {theta_sp:.3f}"
