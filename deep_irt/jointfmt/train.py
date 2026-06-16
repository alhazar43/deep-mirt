"""
train.py -- Training routines for the joint model and independent baselines.
"""

import torch
from joint_model import JointModel
from independent_model import IndepGPCM, IndepBT


# ---------------------------------------------------------------------------
# Joint training
# ---------------------------------------------------------------------------

def train_joint(
    ground_truth: dict,
    mode_a: dict,
    mode_b: dict,
    emb_dim: int = 16,
    n_epochs: int = 600,
    lr: float = 0.05,
    batch_size: int = 8192,
    reg: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> JointModel:
    """
    Train the joint shared-embedding model.

    Each epoch: one pass over GPCM data + one pass over BT data.
    Loss = GPCM_NLL + BT_NLL (both on mini-batches).
    """
    torch.manual_seed(seed)

    model = JointModel(
        n_items=ground_truth["n_items"],
        n_respondents=mode_a["n_respondents"],
        emb_dim=emb_dim,
        K=ground_truth["K"],
        seed=seed,
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    resp_ids = mode_a["respondent_ids"]
    item_ids_a = mode_a["item_ids"]
    responses = mode_a["responses"]
    N_a = len(responses)

    item_i = mode_b["item_i"]
    item_j = mode_b["item_j"]
    outcome = mode_b["outcome"]
    N_b = len(outcome)

    for epoch in range(n_epochs):
        model.train()

        # -- GPCM mini-batch --
        if batch_size >= N_a:
            gpcm_loss = model.gpcm_nll(resp_ids, item_ids_a, responses, reg=reg)
        else:
            perm_a = torch.randperm(N_a)
            gpcm_loss = torch.tensor(0.0)
            n_batches_a = 0
            for start in range(0, N_a, batch_size):
                idx = perm_a[start : start + batch_size]
                gpcm_loss = gpcm_loss + model.gpcm_nll(
                    resp_ids[idx], item_ids_a[idx], responses[idx], reg=reg
                )
                n_batches_a += 1
            gpcm_loss = gpcm_loss / n_batches_a

        # -- BT mini-batch --
        if batch_size >= N_b:
            bt_loss = model.bt_nll(item_i, item_j, outcome, reg=reg)
        else:
            perm_b = torch.randperm(N_b)
            bt_loss = torch.tensor(0.0)
            n_batches_b = 0
            for start in range(0, N_b, batch_size):
                idx = perm_b[start : start + batch_size]
                bt_loss = bt_loss + model.bt_nll(
                    item_i[idx], item_j[idx], outcome[idx], reg=reg
                )
                n_batches_b += 1
            bt_loss = bt_loss / n_batches_b

        total_loss = gpcm_loss + bt_loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(
                f"  [Joint] Epoch {epoch+1:>4d}/{n_epochs}  "
                f"GPCM={gpcm_loss.item():.4f}  BT={bt_loss.item():.4f}"
            )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Independent GPCM training
# ---------------------------------------------------------------------------

def train_indep_gpcm(
    ground_truth: dict,
    mode_a: dict,
    emb_dim: int = 16,
    n_epochs: int = 600,
    lr: float = 0.05,
    batch_size: int = 8192,
    reg: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> IndepGPCM:
    torch.manual_seed(seed)

    model = IndepGPCM(
        n_items=ground_truth["n_items"],
        n_respondents=mode_a["n_respondents"],
        emb_dim=emb_dim,
        K=ground_truth["K"],
        seed=seed,
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    resp_ids = mode_a["respondent_ids"]
    item_ids = mode_a["item_ids"]
    responses = mode_a["responses"]
    N = len(responses)

    for epoch in range(n_epochs):
        model.train()

        if batch_size >= N:
            loss = model.nll(resp_ids, item_ids, responses, reg=reg)
            optim.zero_grad()
            loss.backward()
            optim.step()
        else:
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                loss = model.nll(resp_ids[idx], item_ids[idx], responses[idx], reg=reg)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1

        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                l = model.nll(resp_ids, item_ids, responses, reg=reg)
            print(f"  [IndepGPCM] Epoch {epoch+1:>4d}/{n_epochs}  NLL={l.item():.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Independent BT training
# ---------------------------------------------------------------------------

def train_indep_bt(
    ground_truth: dict,
    mode_b: dict,
    emb_dim: int = 16,
    n_epochs: int = 600,
    lr: float = 0.05,
    batch_size: int = 8192,
    reg: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> IndepBT:
    torch.manual_seed(seed)

    model = IndepBT(
        n_items=ground_truth["n_items"],
        emb_dim=emb_dim,
        seed=seed,
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    item_i = mode_b["item_i"]
    item_j = mode_b["item_j"]
    outcome = mode_b["outcome"]
    N = len(outcome)

    for epoch in range(n_epochs):
        model.train()

        if batch_size >= N:
            loss = model.nll(item_i, item_j, outcome, reg=reg)
            optim.zero_grad()
            loss.backward()
            optim.step()
        else:
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                loss = model.nll(item_i[idx], item_j[idx], outcome[idx], reg=reg)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1

        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                l = model.nll(item_i, item_j, outcome, reg=reg)
            print(f"  [IndepBT]   Epoch {epoch+1:>4d}/{n_epochs}  BCE={l.item():.4f}")

    model.eval()
    return model
