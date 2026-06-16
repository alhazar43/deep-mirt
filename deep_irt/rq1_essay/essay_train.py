"""essay_train.py -- Training routines for the joint model and baselines.

All models share the FROZEN content features; only the extractor (+ heads) is
learnable. Full-batch Adam with cosine schedule (pilot N is small).
"""

from __future__ import annotations

import numpy as np
import torch

from essay_model import (
    JointEssayModel, IndepGradedModel, IndepBTModel,
)


def _to_long(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.long)


def train_joint(split, comparisons, hidden_dim=64, dropout=0.1,
                n_epochs=800, lr=0.01, wd=1e-4, bt_weight=1.0,
                seed=0, verbose=True) -> JointEssayModel:
    """Joint training: graded NLL (on graded essays) + BT NLL (on comparisons).

    comparisons: list of (i, j, outcome) with outcome==1 meaning i beats j.
    """
    torch.manual_seed(seed)
    model = JointEssayModel(split.features, split.K, hidden_dim, dropout, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    g_idx = _to_long(split.graded_idx)
    g_scores = torch.as_tensor(split.scores[split.graded_idx], dtype=torch.long)

    ci = _to_long([c[0] for c in comparisons])
    cj = _to_long([c[1] for c in comparisons])
    co = torch.as_tensor([c[2] for c in comparisons], dtype=torch.float32)

    for epoch in range(n_epochs):
        model.train()
        graded_loss = model.graded_nll(g_idx, g_scores)
        bt_loss = model.bt_nll(ci, cj, co)
        loss = graded_loss + bt_weight * bt_loss
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch + 1) % 200 == 0:
            print(f"  [Joint] {epoch+1:>4d}/{n_epochs} "
                  f"graded={graded_loss.item():.4f} bt={bt_loss.item():.4f}",
                  flush=True)
    model.eval()
    return model


def train_indep_graded(split, hidden_dim=64, dropout=0.1, n_epochs=800,
                       lr=0.01, wd=1e-4, seed=0, verbose=True) -> IndepGradedModel:
    """Graded-only baseline: extractor fit on graded essays via content."""
    torch.manual_seed(seed)
    model = IndepGradedModel(split.features, split.K, hidden_dim, dropout, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    g_idx = _to_long(split.graded_idx)
    g_scores = torch.as_tensor(split.scores[split.graded_idx], dtype=torch.long)
    for epoch in range(n_epochs):
        model.train()
        loss = model.nll(g_idx, g_scores)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch + 1) % 200 == 0:
            print(f"  [IndepGraded] {epoch+1:>4d}/{n_epochs} nll={loss.item():.4f}",
                  flush=True)
    model.eval()
    return model


def train_indep_bt(split, comparisons, hidden_dim=64, dropout=0.1, n_epochs=800,
                   lr=0.01, wd=1e-4, seed=0, verbose=True) -> IndepBTModel:
    """Pairwise-only baseline: extractor fit on comparisons via content."""
    torch.manual_seed(seed)
    model = IndepBTModel(split.features, hidden_dim, dropout, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    ci = _to_long([c[0] for c in comparisons])
    cj = _to_long([c[1] for c in comparisons])
    co = torch.as_tensor([c[2] for c in comparisons], dtype=torch.float32)
    for epoch in range(n_epochs):
        model.train()
        loss = model.nll(ci, cj, co)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch + 1) % 200 == 0:
            print(f"  [IndepBT] {epoch+1:>4d}/{n_epochs} bce={loss.item():.4f}",
                  flush=True)
    model.eval()
    return model
