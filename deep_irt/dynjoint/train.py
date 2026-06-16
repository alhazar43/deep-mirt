"""
train.py -- Training routines for the dynamic-encoder joint model and baselines.

Joint training strategy
-----------------------
Each epoch processes:
  1. GPCM mini-batch: encode full learner sequences -> shifted theta_in -> GPCM NLL.
  2. BT mini-batch: person-free pairwise comparisons -> BT NLL.
Both losses operate on the shared item embedding via the shared DifficultyExtractor.

Independent baselines
---------------------
indep_gpcm: same architecture as the joint model's GPCM path but with a SEPARATE
            item embedding table; no BT training.
indep_bt  : same DifficultyExtractor + BTHead but with a SEPARATE embedding table.

The baselines use STATIC per-respondent theta (nn.Parameter) to match the
independent-fit setup from jointfmt.  This is the correct counterfactual: the
baseline does not get a dynamic encoder; it gets the old static theta.  The gap
in cross-format transfer is then attributable to the shared embedding, not to any
difference in the theta model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_irt.dynjoint.dyn_model import DynJointModel
from deep_irt.jointfmt.joint_model import DifficultyExtractor, GPCMHead, BTHead


# ---------------------------------------------------------------------------
# Independent GPCM baseline (static theta, separate embedding)
# ---------------------------------------------------------------------------

class IndepGPCM(nn.Module):
    """
    GPCM with a separate embedding table and static per-respondent theta.
    Pairwise-only items: embedding never trained -> d_i is noise.
    """

    def __init__(
        self,
        n_items: int,
        n_respondents: int,
        emb_dim: int = 16,
        K: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed + 10)

        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        self.difficulty_extractor = DifficultyExtractor(emb_dim=emb_dim, hidden_dim=8)
        self.theta = nn.Parameter(torch.randn(n_respondents, generator=g) * 0.1)

        self.fc_a = nn.Linear(emb_dim, 1)
        self.fc_offsets = nn.Linear(emb_dim, K - 2) if K > 2 else None

        self.K = K
        self.n_items = n_items

    def nll(
        self,
        respondent_ids: torch.Tensor,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        theta = self.theta[respondent_ids]
        emb = self.item_emb(item_ids)
        d = self.difficulty_extractor(emb)
        a = F.softplus(self.fc_a(emb)).squeeze(-1)
        if self.fc_offsets is not None:
            raw_off = self.fc_offsets(emb)
            b = torch.cat([d.unsqueeze(-1), d.unsqueeze(-1) + raw_off], dim=-1)
        else:
            b = d.unsqueeze(-1)
        step = a.unsqueeze(-1) * (theta.unsqueeze(-1) - b)
        cumsum = torch.cumsum(step, dim=-1)
        log_num = torch.cat(
            [torch.zeros(theta.size(0), 1, device=theta.device), cumsum], dim=-1
        )
        log_p = F.log_softmax(log_num, dim=-1)
        nll = F.nll_loss(log_p, responses)
        theta_reg = reg * self.theta.pow(2).mean()
        off_reg = reg * self.fc_offsets(emb).pow(2).mean() if self.fc_offsets else 0.0
        return nll + theta_reg + off_reg

    @torch.no_grad()
    def get_difficulty(self) -> torch.Tensor:
        all_idx = torch.arange(self.n_items)
        emb = self.item_emb(all_idx)
        d = self.difficulty_extractor(emb)
        return d - d.mean()


# ---------------------------------------------------------------------------
# Independent BT baseline (separate embedding)
# ---------------------------------------------------------------------------

class IndepBT(nn.Module):
    """
    BT model with separate embedding table.
    Direct-only items: embedding never trained -> d_i is noise.
    """

    def __init__(self, n_items: int, emb_dim: int = 16, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed + 20)

        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        self.difficulty_extractor = DifficultyExtractor(emb_dim=emb_dim, hidden_dim=8)
        self.n_items = n_items

    def nll(
        self,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        outcome: torch.Tensor,
        reg: float = 0.01,
    ) -> torch.Tensor:
        emb_i = self.item_emb(item_i)
        emb_j = self.item_emb(item_j)
        d_i = self.difficulty_extractor(emb_i)
        d_j = self.difficulty_extractor(emb_j)
        delta = d_i - d_j
        bce = F.binary_cross_entropy_with_logits(delta, outcome.float(), reduction="mean")
        d_reg = reg * (d_i.pow(2).mean() + d_j.pow(2).mean()) * 0.5
        return bce + d_reg

    @torch.no_grad()
    def get_strength(self) -> torch.Tensor:
        all_idx = torch.arange(self.n_items)
        emb = self.item_emb(all_idx)
        d = self.difficulty_extractor(emb)
        return d - d.mean()


# ---------------------------------------------------------------------------
# Joint training (dynamic encoder)
# ---------------------------------------------------------------------------

def train_joint(
    ground_truth: dict,
    mode_a: dict,
    mode_b: dict,
    emb_dim: int = 16,
    hidden_dim: int = 64,
    n_epochs: int = 600,
    lr: float = 0.01,
    batch_size: int = 64,
    reg: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> DynJointModel:
    """
    Train the dynamic-encoder joint model.

    GPCM path: full learner sequences -> shifted theta -> GPCM NLL on flat obs.
    BT path  : mini-batch pairwise pairs -> BT NLL on shared d_i.
    """
    torch.manual_seed(seed)

    model = DynJointModel(
        n_items=ground_truth["n_items"],
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        K=ground_truth["K"],
        seed=seed,
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    seq_item_ids = mode_a["seq_item_ids"]    # (N, T)
    seq_responses = mode_a["seq_responses"]  # (N, T)
    N_respondents = seq_item_ids.size(0)
    T = seq_item_ids.size(1)

    item_i = mode_b["item_i"]
    item_j = mode_b["item_j"]
    outcome = mode_b["outcome"]
    N_b = len(outcome)

    for epoch in range(n_epochs):
        model.train()

        # -- GPCM path: mini-batch over learner sequences --
        perm = torch.randperm(N_respondents)
        total_gpcm = 0.0
        n_batches_gpcm = 0

        for start in range(0, N_respondents, batch_size):
            idx = perm[start : start + batch_size]
            b_items = seq_item_ids[idx]     # (B, T)
            b_resp = seq_responses[idx]     # (B, T)

            # Shifted theta: (B, T) -- ability BEFORE each step
            theta_in = model.encoder.shifted_theta(b_items, b_resp)  # (B, T)

            # Flatten for GPCM head: use ALL (B*T) positions
            B, T_ = b_items.shape
            theta_flat = theta_in.reshape(B * T_)
            items_flat = b_items.reshape(B * T_)
            resp_flat = b_resp.reshape(B * T_)

            gpcm_loss = model.gpcm_nll_flat(theta_flat, items_flat, resp_flat, reg=reg)
            total_gpcm += gpcm_loss.item()
            n_batches_gpcm += 1

            # BT mini-batch interleaved within GPCM loop for shared gradient step
            bt_idx = torch.randint(0, N_b, (min(batch_size * T_, N_b),))
            bt_loss = model.bt_nll(item_i[bt_idx], item_j[bt_idx], outcome[bt_idx], reg=reg)

            loss = gpcm_loss + bt_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            avg_gpcm = total_gpcm / max(n_batches_gpcm, 1)
            print(
                f"  [DynJoint] Epoch {epoch+1:>4d}/{n_epochs}  "
                f"GPCM~={avg_gpcm:.4f}  BT~={bt_loss.item():.4f}"
            )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Joint training (dynamic encoder) -- REBALANCED PATH
# ---------------------------------------------------------------------------

def _shared_trunk_params(model: DynJointModel):
    """The parameters BOTH heads write to: item embedding + difficulty extractor.

    These are the parameters whose gradient geometry decides the shared d_i
    scale.  Loss-balancing must equalise the two heads' pull HERE, not on the
    full parameter set (the LSTM and the GPCM offset/discrimination heads are
    GPCM-private and should not enter the balance).
    """
    return (
        list(model.encoder.item_emb.parameters())
        + list(model.difficulty_extractor.parameters())
    )


def _grad_norm_on(params, loss: torch.Tensor) -> float:
    """L2 norm of d(loss)/d(params), without disturbing .grad buffers.

    Uses torch.autograd.grad with retain_graph so the caller can still call
    the real backward() afterwards.
    """
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True, create_graph=False
    )
    tot = 0.0
    for g in grads:
        if g is not None:
            tot += g.detach().pow(2).sum().item()
    return tot ** 0.5


def train_joint_balanced(
    ground_truth: dict,
    mode_a: dict,
    mode_b: dict,
    emb_dim: int = 16,
    hidden_dim: int = 64,
    n_epochs: int = 400,
    lr: float = 5e-3,
    batch_size: int = 64,
    reg: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
    # --- rebalancing knobs (all default to the P1.6 behaviour) ---
    balance: str = "none",          # "none" | "fixed" | "uncertainty" | "gradnorm"
    bt_weight: float = 1.0,         # used by balance == "fixed"
    bt_pairs_mult: float = 1.0,     # BT pairs per step, as a multiple of B*T
    head_lr_mult: float = 1.0,      # LR multiplier for the difficulty extractor + heads
    warmup_frac: float = 0.0,       # linear LR warmup over this fraction of epochs
    clip_norm: float = 5.0,
    balance_every: int = 5,         # gradnorm: recompute the weight every k steps
    balance_ema: float = 0.9,       # gradnorm: EMA on the balancing weight
    balance_cap: float = 25.0,      # gradnorm: clamp w_bt to [1/cap, cap]
) -> DynJointModel:
    """Rebalanced dynamic-encoder joint training.

    Adds explicit per-head loss balancing so the GPCM and BT heads pull
    COMPARABLY on the shared trunk (item embedding + difficulty extractor).
    With all knobs at their defaults this reduces EXACTLY to ``train_joint``
    (balance="none", bt_pairs_mult=1, head_lr_mult=1, warmup_frac=0,
    clip_norm=5.0), so the P1.6 baseline stays reproducible through this
    function too.

    Balancing modes
    ---------------
    none        : loss = gpcm + bt   (P1.6 baseline).
    fixed       : loss = gpcm + bt_weight * bt.
    uncertainty : Kendall homoscedastic weighting; two learned log-variances
                  s_g, s_b give loss = exp(-s_g)*gpcm + exp(-s_b)*bt + s_g + s_b.
                  Balances by LOSS scale.
    gradnorm    : measure ||grad_gpcm||, ||grad_bt|| on the shared trunk and set
                  w_bt so that w_bt*||grad_bt|| == ||grad_gpcm|| (EMA-smoothed,
                  clamped).  Balances by GRADIENT magnitude -- the quantity that
                  actually governs the shared d_i scale.

    Other levers
    ------------
    bt_pairs_mult : raises the BT observation volume per step (data-throughput
                    rebalance; the data-generating process is unchanged).
    head_lr_mult  : separate, larger LR for the difficulty extractor + heads
                    relative to the LSTM trunk.
    warmup_frac   : linear LR warmup to stabilise the early, high-LR phase.
    """
    torch.manual_seed(seed)

    model = DynJointModel(
        n_items=ground_truth["n_items"],
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        K=ground_truth["K"],
        seed=seed,
    )

    # Kendall uncertainty parameters (only used when balance == "uncertainty").
    log_var_g = torch.zeros(1, requires_grad=True)
    log_var_b = torch.zeros(1, requires_grad=True)

    # Param groups: LSTM trunk at base lr, difficulty extractor + heads at
    # head_lr_mult * lr.  Item embedding stays in the trunk group (it is the
    # shared currency; scaling its LR separately would bias the balance).
    head_modules = [model.difficulty_extractor, model.gpcm_head, model.bt_head]
    head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
    head_params = [p for m in head_modules for p in m.parameters()]
    trunk_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    param_groups = [
        {"params": trunk_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ]
    if balance == "uncertainty":
        param_groups.append({"params": [log_var_g, log_var_b], "lr": lr})

    optim = torch.optim.Adam(param_groups, lr=lr)

    # Cosine schedule with optional linear warmup, applied as an LR scale.
    warmup_epochs = int(round(warmup_frac * n_epochs))

    def lr_scale(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # cosine from 1 -> 0 over the post-warmup span
        import math
        span = max(n_epochs - warmup_epochs, 1)
        prog = (epoch - warmup_epochs) / span
        return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda e: lr_scale(e))

    seq_item_ids = mode_a["seq_item_ids"]
    seq_responses = mode_a["seq_responses"]
    N_respondents = seq_item_ids.size(0)

    item_i = mode_b["item_i"]
    item_j = mode_b["item_j"]
    outcome = mode_b["outcome"]
    N_b = len(outcome)

    shared_params = _shared_trunk_params(model)
    w_bt_ema = float(bt_weight)  # running balancing weight (gradnorm)
    step_count = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(N_respondents)
        total_gpcm = 0.0
        n_batches_gpcm = 0
        last_bt = 0.0
        last_w = w_bt_ema

        for start in range(0, N_respondents, batch_size):
            idx = perm[start : start + batch_size]
            b_items = seq_item_ids[idx]
            b_resp = seq_responses[idx]

            theta_in = model.encoder.shifted_theta(b_items, b_resp)
            B, T_ = b_items.shape
            theta_flat = theta_in.reshape(B * T_)
            items_flat = b_items.reshape(B * T_)
            resp_flat = b_resp.reshape(B * T_)

            gpcm_loss = model.gpcm_nll_flat(theta_flat, items_flat, resp_flat, reg=reg)
            total_gpcm += gpcm_loss.item()
            n_batches_gpcm += 1

            # BT mini-batch: volume scaled by bt_pairs_mult.
            n_bt = int(min(round(batch_size * T_ * bt_pairs_mult), N_b))
            bt_idx = torch.randint(0, N_b, (n_bt,))
            bt_loss = model.bt_nll(item_i[bt_idx], item_j[bt_idx], outcome[bt_idx], reg=reg)
            last_bt = bt_loss.item()

            if balance == "none":
                loss = gpcm_loss + bt_loss

            elif balance == "fixed":
                loss = gpcm_loss + bt_weight * bt_loss

            elif balance == "uncertainty":
                # Kendall multi-task homoscedastic uncertainty weighting.
                loss = (
                    torch.exp(-log_var_g) * gpcm_loss
                    + torch.exp(-log_var_b) * bt_loss
                    + log_var_g
                    + log_var_b
                ).squeeze()
                last_w = float(torch.exp(log_var_g - log_var_b).item())

            elif balance == "gradnorm":
                # Periodically rebalance so w_bt * ||g_bt|| == ||g_gpcm|| on the
                # shared trunk.  EMA-smooth + clamp for stability; reuse between
                # measurements to keep the cost low.
                if step_count % balance_every == 0:
                    g_gpcm = _grad_norm_on(shared_params, gpcm_loss)
                    g_bt = _grad_norm_on(shared_params, bt_loss)
                    target = g_gpcm / max(g_bt, 1e-8)
                    target = float(min(max(target, 1.0 / balance_cap), balance_cap))
                    w_bt_ema = balance_ema * w_bt_ema + (1.0 - balance_ema) * target
                last_w = w_bt_ema
                loss = gpcm_loss + w_bt_ema * bt_loss

            else:
                raise ValueError(f"unknown balance mode: {balance!r}")

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optim.step()
            step_count += 1

        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            avg_gpcm = total_gpcm / max(n_batches_gpcm, 1)
            print(
                f"  [DynJoint:{balance}] Epoch {epoch+1:>4d}/{n_epochs}  "
                f"GPCM~={avg_gpcm:.4f}  BT~={last_bt:.4f}  w_bt~={last_w:.3f}"
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
    """
    Train the independent GPCM baseline.

    Uses the FLAT (respondent_id, item_id, response) tuples derived from the
    sequential data.  Static theta per respondent.
    """
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

    # Flatten sequential data into flat (obs) format
    N, T = mode_a["seq_item_ids"].shape
    resp_ids = torch.arange(N).unsqueeze(1).expand(N, T).reshape(-1)
    item_ids = mode_a["seq_item_ids"].reshape(-1)
    responses = mode_a["seq_responses"].reshape(-1)
    N_obs = len(responses)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(N_obs)

        for start in range(0, N_obs, batch_size):
            idx = perm[start : start + batch_size]
            loss = model.nll(resp_ids[idx], item_ids[idx], responses[idx], reg=reg)
            optim.zero_grad()
            loss.backward()
            optim.step()

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

    model = IndepBT(n_items=ground_truth["n_items"], emb_dim=emb_dim, seed=seed)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    item_i = mode_b["item_i"]
    item_j = mode_b["item_j"]
    outcome = mode_b["outcome"]
    N = len(outcome)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(N)

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            loss = model.nll(item_i[idx], item_j[idx], outcome[idx], reg=reg)
            optim.zero_grad()
            loss.backward()
            optim.step()

        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                l = model.nll(item_i, item_j, outcome, reg=reg)
            print(f"  [IndepBT]   Epoch {epoch+1:>4d}/{n_epochs}  BCE={l.item():.4f}")

    model.eval()
    return model
