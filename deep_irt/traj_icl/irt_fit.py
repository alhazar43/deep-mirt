"""irt_fit.py -- assemble the examinee-by-item matrix and fit a 2PL.

Each (model, shot count k, condition) is one examinee. The 2PL places all
examinees on one shared item scale,

    P(correct | examinee i, item j) = sigmoid(a_j (theta_i - b_j)),

fit jointly by maximum likelihood (Adam on the BCE). The scale has the
usual location and scale indeterminacy, resolved post hoc by standardizing
theta across examinees. The rate read from theta(k) is affine-invariant,
so the standardization does not bias it.
"""

import glob
import json
import os

import numpy as np
import torch


def load_responses(out_dir):
    """Load per-model response files into an examinee-by-item matrix.

    Reads every responses_*.json in out_dir (the contract written by
    generate.py). Returns R (I, J) int8 correctness, an examinee index
    (list of dicts with model, k, condition), the shared item_ids (J,),
    and a parallel accuracy list.
    """
    files = sorted(glob.glob(os.path.join(out_dir, "responses_*.json")))
    if not files:
        raise FileNotFoundError(f"no responses_*.json in {out_dir}")

    rows, index, acc_rows = [], [], []
    ref_items = None
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        item_ids = d["item_ids"]
        if ref_items is None:
            ref_items = item_ids
        elif item_ids != ref_items:
            raise ValueError(f"item bank mismatch in {fp}; banks must be identical")
        for cond in d["conditions"]:
            for k in d["k_values"]:
                vec = d["correct"][cond][str(k)]
                rows.append(np.asarray(vec, dtype=np.int8))
                index.append({"model": d["model"], "k": int(k), "condition": cond})
                acc_rows.append(float(np.mean(vec)))

    R = np.vstack(rows)                       # (I, J)
    return R, index, np.asarray(ref_items), np.asarray(acc_rows)


def fit_2pl(R, n_epochs=3000, lr=0.05, device="cpu", seed=0, l2_item=1e-3):
    """Joint 2PL MLE. R is (I, J) binary (NaN entries are skipped).

    Returns theta (I,), a (J,), b (J,), standardized so theta has mean 0
    and unit sd across examinees.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)
    obs = ~torch.isnan(Rt)
    Rt = torch.nan_to_num(Rt, nan=0.0)
    I, J = Rt.shape

    theta = torch.zeros(I, device=dev, requires_grad=True)
    a_raw = torch.zeros(J, device=dev, requires_grad=True)   # softplus(0) ~ 0.69
    b = torch.zeros(J, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta, a_raw, b], lr=lr)

    for _ in range(n_epochs):
        opt.zero_grad()
        a = torch.nn.functional.softplus(a_raw)
        logits = a.unsqueeze(0) * (theta.unsqueeze(1) - b.unsqueeze(0))   # (I, J)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[obs], Rt[obs])
        loss = loss + l2_item * (b.pow(2).mean() + a_raw.pow(2).mean())
        loss.backward()
        opt.step()

    with torch.no_grad():
        th = theta.detach().cpu().numpy()
        a = torch.nn.functional.softplus(a_raw).detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        mu, sd = th.mean(), th.std() + 1e-8
        th = (th - mu) / sd
        b = (b - mu) / sd        # keep item difficulty on the same standardized scale
        a = a * sd               # discrimination scales inversely with theta scale
    return {"theta": th, "a": a, "b": b, "final_bce": float(loss.item())}
