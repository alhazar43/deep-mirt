"""
train.py -- Training routines for the binary-vs-NRM joint model and baselines.
"""

import torch

from models import JointModel, IndepBinary, IndepNRM


def _minibatch_nll(fn, ids_tuple, N, batch_size, optim):
    """One epoch over a single-format dataset; returns last loss value."""
    if batch_size >= N:
        loss = fn(*ids_tuple)
        optim.zero_grad(); loss.backward(); optim.step()
        return loss.item()
    perm = torch.randperm(N)
    last = 0.0
    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        loss = fn(*(t[idx] for t in ids_tuple))
        optim.zero_grad(); loss.backward(); optim.step()
        last = loss.item()
    return last


def train_joint(gt, mode_a, mode_b, emb_dim=16, n_epochs=600, lr=0.05,
                batch_size=8192, reg=0.01, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = JointModel(
        n_items=gt["n_items"],
        n_resp_a=mode_a["n_respondents"],
        n_resp_b=mode_b["n_respondents"],
        emb_dim=emb_dim, K=gt["K"], correct_option=gt["correct_option"], seed=seed,
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    ra, ia, ca = mode_a["respondent_ids"], mode_a["item_ids"], mode_a["responses"]
    rb, ib, cb = mode_b["respondent_ids"], mode_b["item_ids"], mode_b["responses"]
    Na, Nb = len(ca), len(cb)

    for epoch in range(n_epochs):
        model.train()
        # joint step: sum both format losses, one backward
        if batch_size >= Na and batch_size >= Nb:
            loss = (model.binary_nll(ra, ia, ca, reg=reg)
                    + model.nrm_nll(rb, ib, cb, reg=reg))
            optim.zero_grad(); loss.backward(); optim.step()
            bl = nl = loss.item()
        else:
            # independent passes per format (separate backward each)
            bl = _minibatch_nll(
                lambda r, i, c: model.binary_nll(r, i, c, reg=reg),
                (ra, ia, ca), Na, batch_size, optim)
            nl = _minibatch_nll(
                lambda r, i, c: model.nrm_nll(r, i, c, reg=reg),
                (rb, ib, cb), Nb, batch_size, optim)
        sched.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  [Joint] ep {epoch+1:>4d}/{n_epochs}  bin={bl:.4f}  nrm={nl:.4f}")
    model.eval()
    return model


def train_indep_binary(gt, mode_a, emb_dim=16, n_epochs=600, lr=0.05,
                       batch_size=8192, reg=0.01, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = IndepBinary(gt["n_items"], mode_a["n_respondents"], emb_dim, seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
    r, i, c = mode_a["respondent_ids"], mode_a["item_ids"], mode_a["responses"]
    N = len(c)
    for epoch in range(n_epochs):
        model.train()
        l = _minibatch_nll(lambda rr, ii, cc: model.nll(rr, ii, cc, reg=reg),
                           (r, i, c), N, batch_size, optim)
        sched.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  [IndepBin] ep {epoch+1:>4d}/{n_epochs}  BCE={l:.4f}")
    model.eval()
    return model


def train_indep_nrm(gt, mode_b, emb_dim=16, n_epochs=600, lr=0.05,
                    batch_size=8192, reg=0.01, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = IndepNRM(gt["n_items"], mode_b["n_respondents"], emb_dim,
                     gt["K"], gt["correct_option"], seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
    r, i, c = mode_b["respondent_ids"], mode_b["item_ids"], mode_b["responses"]
    N = len(c)
    for epoch in range(n_epochs):
        model.train()
        l = _minibatch_nll(lambda rr, ii, cc: model.nll(rr, ii, cc, reg=reg),
                           (r, i, c), N, batch_size, optim)
        sched.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  [IndepNRM] ep {epoch+1:>4d}/{n_epochs}  NLL={l:.4f}")
    model.eval()
    return model
