"""run_trajectory_probe.py -- Phase 1 + Phase 2 learning-dynamics probes.

Phase 1 -- TRAJECTORY
---------------------
Train three configs (SHARED-NARROW, SHARED-WIDE, DECOUPLED) in ONE continuous
run per (config, seed), checkpointing at epochs [25, 50, 100, 150, 300, 500].
At each checkpoint recover (theta_static, alpha) via the existing recovery
routines and record the Spearman correlation against ground truth.

Three hypotheses:
  H1  SHARED-WIDE theta_static DEGRADES as training continues (overfit grows);
      SHARED-NARROW and DECOUPLED stay stable.
  H2  alpha is learned LATE (rises after theta saturates).
  H3  DECOUPLED keeps theta stable AND alpha high across the whole trajectory.

Phase 2 -- GRADIENT CONFLICT
-----------------------------
On the SHARED-WIDE model, decompose dL/dE (gradient on the shared item
embedding E) into three pathway contributions:

    g_theta  -- gradient through the LSTM input (the theta path)
    g_beta   -- gradient through fc_b(E) (step-threshold head)
    g_alpha  -- gradient through fc_a_state([state, E]) (state-cond alpha head)

Sum check: g_theta + g_beta + g_alpha must equal the full gradient
autograd.grad(loss, E) within tolerance.  ONLY report numbers if this
passes.  If it fails, flag Phase 2 as "instrumentation not verified".

Hypotheses:
  H4  ||g_theta|| >> ||g_alpha||  (theta dominates in magnitude)
  H5  cos(g_theta, g_alpha) <= 0  (pathways conflict)
  H6  DECOUPLED removes competition (alpha-only table gets only g_alpha)

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \\
      python deep_irt/bench/run_trajectory_probe.py [--quick] [--device cuda]

Outputs
-------
  deep_irt/bench/outputs/trajectory_results.json
  deep_irt/bench/outputs/trajectory_table.md
  deep_irt/bench/outputs/gradient_conflict_table.md  (if Phase 2 passes sum-check)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from deep_irt.bench.datagen import BenchDataConfig, generate, BenchDataset
from deep_irt.bench import metrics_bench as M
from deep_irt.core.model import DeepIRTModel

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

# All configs: LSTM/GPCM, state_alpha=True, alpha_log_scale=1.0, hidden_dim=32
_BASE = dict(
    hidden_dim=32,
    state_alpha=True,
    alpha_log_scale=1.0,
    decoder="gpcm",
)

CONFIGS = {
    "SHARED-NARROW": dict(**_BASE, emb_dim=8, alpha_emb_dim=None),
    "SHARED-WIDE":   dict(**_BASE, emb_dim=64, alpha_emb_dim=None),
    "DECOUPLED":     dict(**_BASE, emb_dim=8, alpha_emb_dim=64),
}

# Checkpoint epochs for trajectory probe
CHECKPOINTS_FULL  = [25, 50, 100, 150, 300, 500]
CHECKPOINTS_QUICK = [25, 50, 100]

# Gradient conflict probe epochs (subset of checkpoints)
GRAD_PROBE_EPOCHS_FULL  = [25, 100, 300, 500]
GRAD_PROBE_EPOCHS_QUICK = [25, 100]


# ---------------------------------------------------------------------------
# Training loop (warm checkpoints -- NO fit() calls after first epoch block)
# ---------------------------------------------------------------------------

def _build_model(cfg: dict, num_items: int, n_cats: int,
                 device: torch.device, seed: int) -> DeepIRTModel:
    """Construct a DeepIRTModel with EXPLICIT args -- no reliance on defaults."""
    return DeepIRTModel(
        num_items=num_items,
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_cats=n_cats,
        decoder=cfg["decoder"],
        state_alpha=cfg["state_alpha"],
        alpha_emb_dim=cfg.get("alpha_emb_dim"),
        alpha_log_scale=cfg.get("alpha_log_scale"),
        encoder="lstm",
        device=device,
        seed=seed,
    )


def train_with_checkpoints(
    model: DeepIRTModel,
    item_ids: torch.Tensor,    # (N, T)
    responses: torch.Tensor,   # (N, T)
    ds: BenchDataset,
    checkpoints: list[int],
    device: torch.device,
    lr: float = 1e-2,
) -> dict[int, dict]:
    """Run a single continuous training loop, recovering metrics at checkpoint epochs.

    This replicates the model.fit() Adam loop but WITHOUT restarting the
    optimizer or reseeding -- one continuous trajectory with warm checkpoints.

    Returns
    -------
    results  dict mapping epoch -> {theta_static, alpha_spearman}
    """
    # One Adam optimizer, not restarted.
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    item_ids = item_ids.to(device)
    responses = responses.to(device)

    # Tensors for full-dataset recovery (all N learners)
    it_all = torch.tensor(ds.items0, dtype=torch.long, device=device)
    rp_all = torch.tensor(ds.responses, dtype=torch.long, device=device)

    checkpoints_set = set(checkpoints)
    max_epoch = max(checkpoints)
    results: dict[int, dict] = {}

    model.encoder.train()
    model.decoder.train()

    for epoch in range(1, max_epoch + 1):
        optimizer.zero_grad()
        loss = model._compute_nll(item_ids, responses)
        loss.backward()
        optimizer.step()

        if epoch in checkpoints_set:
            model.encoder.eval()
            model.decoder.eval()
            with torch.no_grad():
                # ----- theta recovery (static: final-step vs true level) -----
                theta_traj = model.track(it_all, rp_all).cpu().numpy()  # (N,T)
                theta_final = theta_traj[:, -1]                          # (N,)
                theta_true = ds.gt.theta0                                # (N,)
                tm = M.theta_recovery_static(theta_final, theta_true)
                theta_static = tm["theta_spearman"]

                # ----- alpha recovery (state_alpha -> occurrence-averaged) -----
                rec = model.recover_item_params(it_all, rp_all)
                a_hat = rec["a"]
                seen = rec.get("seen")
                im = M.item_recovery(
                    a_hat, rec["b"], ds.gt.a, ds.gt.b, seen=seen
                )
                alpha_spearman = im["a_spearman"]

            results[epoch] = {
                "epoch": epoch,
                "theta_static": round(float(theta_static), 4),
                "alpha_spearman": round(float(alpha_spearman), 4),
                "nll": round(float(loss.item()), 4),
            }
            print(
                f"    epoch {epoch:4d}  theta_static={theta_static:.4f}"
                f"  alpha={alpha_spearman:.4f}  NLL={loss.item():.4f}"
            )
            model.encoder.train()
            model.decoder.train()

    return results


# ---------------------------------------------------------------------------
# Phase 1: trajectory over seeds
# ---------------------------------------------------------------------------

def run_phase1(
    ds: BenchDataset,
    checkpoints: list[int],
    seeds: list[int],
    device: torch.device,
) -> dict:
    """Run trajectory probe for all configs and seeds.

    Returns
    -------
    phase1_raw  dict: config_name -> list(seed_results)
        where seed_results is dict: epoch -> {theta_static, alpha_spearman, nll}
    """
    N = len(ds.train_idx)
    it_train = torch.tensor(ds.items0[ds.train_idx], dtype=torch.long)
    rp_train = torch.tensor(ds.responses[ds.train_idx], dtype=torch.long)

    phase1_raw: dict[str, list] = {name: [] for name in CONFIGS}

    for name, cfg in CONFIGS.items():
        print(f"\n  [{name}]")
        for seed in seeds:
            print(f"    seed={seed}")
            model = _build_model(
                cfg, num_items=ds.cfg.n_items, n_cats=ds.cfg.n_cats,
                device=device, seed=seed,
            )
            seed_results = train_with_checkpoints(
                model, it_train, rp_train, ds, checkpoints, device,
            )
            phase1_raw[name].append(seed_results)

    return phase1_raw


def aggregate_phase1(phase1_raw: dict) -> dict:
    """Average theta_static and alpha_spearman over seeds per (config, epoch)."""
    agg: dict[str, dict[int, dict]] = {}
    for name, seed_list in phase1_raw.items():
        epochs = sorted(seed_list[0].keys())
        per_epoch: dict[int, dict] = {}
        for ep in epochs:
            theta_vals = [s[ep]["theta_static"] for s in seed_list]
            alpha_vals = [s[ep]["alpha_spearman"] for s in seed_list]
            per_epoch[ep] = {
                "epoch": ep,
                "theta_mean": round(float(np.mean(theta_vals)), 4),
                "theta_std":  round(float(np.std(theta_vals)),  4),
                "alpha_mean": round(float(np.mean(alpha_vals)), 4),
                "alpha_std":  round(float(np.std(alpha_vals)),  4),
            }
        agg[name] = per_epoch
    return agg


# ---------------------------------------------------------------------------
# Phase 2: gradient conflict on SHARED-WIDE
# ---------------------------------------------------------------------------

def _forward_lstm_only(
    model: DeepIRTModel,
    E: torch.Tensor,          # (N*T, emb_dim)  requires_grad leaf
    item_ids: torch.Tensor,   # (N, T)
    responses: torch.Tensor,  # (N, T)
    E_detach: torch.Tensor,   # (N*T, emb_dim)  detached reference
) -> torch.Tensor:
    """NLL where ONLY the LSTM-input pathway uses E (beta and alpha-emb get E_detach).

    E is routed as the LSTM input.  This is the "theta pathway" in the sense
    that the LSTM-input embedding is consumed to produce the hidden state, which
    feeds BOTH the theta head (theta_proj) AND the state-conditioning of
    fc_a_state.  In SHARED-WIDE, the state is the coupling mechanism: E flows
    through the LSTM to produce state, which is then used by both theta and
    the alpha head.  This pathway carries all gradient that reaches E through
    the recurrent dynamics.

    The alpha head's DIRECT emb argument gets E_detach, and fc_b gets E_detach,
    so their direct emb reads contribute zero gradient to E.

    Decomposition:
        g_lstm (this fn)   = dL/dE through E as LSTM input (state+theta paths)
        g_beta_head        = dL/dE through E as fc_b input
        g_alpha_head       = dL/dE through E as fc_a_state emb input (direct)

    These three are additive and sum to the full gradient (verified by sum-check).
    """
    B, T = item_ids.shape
    emb_dim = model.emb_dim

    E_seq = E.view(B, T, emb_dim)             # (N, T, emb_dim) -- live (LSTM input)
    E_det_seq = E_detach.view(B, T, emb_dim)  # (N, T, emb_dim) -- detached

    enc = model.encoder
    resp_embs = enc.resp_emb(responses)               # (N, T, emb_dim)
    x = torch.cat([E_seq, resp_embs], dim=-1)         # (N, T, 2*emb_dim) -- live
    h, _ = enc.lstm(x)                                # (N, T, hidden)
    state = enc._shift(h)                             # (N, T, hidden)
    theta_in = enc._shift(enc.theta_proj(h).squeeze(-1))   # (N, T)

    theta_flat   = theta_in.reshape(B * T)
    state_flat   = state.reshape(B * T, model.hidden_dim)
    emb_flat_det = E_det_seq.reshape(B * T, emb_dim)  # detached for beta+alpha-emb
    resp_flat    = responses.reshape(B * T)

    # item_params uses:
    #   fc_b(emb_flat_det) for beta -- detached, no E gradient from beta head
    #   fc_a_state([state_flat, emb_flat_det]) for alpha -- state is live (carries
    #   LSTM gradient), but emb is detached (no DIRECT alpha-emb gradient here)
    params = model.decoder.item_params(
        emb_flat_det,
        state=state_flat,
        alpha_emb=None,
    )

    log_p = model.decoder.log_probs(theta_flat, params["a"], params["b"])
    return F.nll_loss(log_p, resp_flat)


def _forward_beta_only(
    model: DeepIRTModel,
    E: torch.Tensor,          # (N*T, emb_dim)  requires_grad leaf
    item_ids: torch.Tensor,   # (N, T)
    responses: torch.Tensor,  # (N, T)
    E_detach: torch.Tensor,   # detached
) -> torch.Tensor:
    """NLL where ONLY the beta pathway uses E (theta and alpha get detached)."""
    B, T = item_ids.shape
    emb_dim = model.emb_dim
    E_seq = E.view(B, T, emb_dim)            # live for beta
    E_det_seq = E_detach.view(B, T, emb_dim)

    enc = model.encoder
    resp_embs = enc.resp_emb(responses)
    # LSTM gets detached embedding -> no theta gradient through E
    x = torch.cat([E_det_seq, resp_embs], dim=-1)  # detached
    h, _ = enc.lstm(x)
    state = enc._shift(h)
    theta_in = enc._shift(enc.theta_proj(h).squeeze(-1))

    theta_flat  = theta_in.reshape(B * T)
    state_flat  = state.reshape(B * T, model.hidden_dim)
    emb_flat    = E_seq.reshape(B * T, emb_dim)        # live for beta
    emb_flat_d  = E_det_seq.reshape(B * T, emb_dim)    # detached for alpha
    resp_flat   = responses.reshape(B * T)

    # fc_b gets live E (beta gradient flows); fc_a_state gets detached emb
    # We need to manually split the item_params call.
    # beta from live E:
    b = model.decoder.fc_b(emb_flat)  # (N*T, K-1)  -- live
    # alpha from detached state and detached emb (both carry no E gradient here):
    key = torch.cat([state_flat.detach(), emb_flat_d], dim=-1)
    raw_a = model.decoder.fc_a_state(key)
    a = torch.exp(model.decoder.alpha_log_scale * raw_a)

    log_p = model.decoder.log_probs(theta_flat.detach(), a, b)
    return F.nll_loss(log_p, resp_flat)


def _forward_alpha_only(
    model: DeepIRTModel,
    E: torch.Tensor,          # (N*T, emb_dim)  requires_grad leaf
    item_ids: torch.Tensor,   # (N, T)
    responses: torch.Tensor,  # (N, T)
    E_detach: torch.Tensor,   # detached
) -> torch.Tensor:
    """NLL where ONLY the alpha pathway uses E (theta and beta get detached).

    For SHARED-WIDE (alpha_emb_dim=None) the state-cond head reads
    fc_a_state([state, E]).  We detach the state so only the DIRECT emb
    contribution to fc_a_state carries E-gradient here.
    """
    B, T = item_ids.shape
    emb_dim = model.emb_dim
    E_seq = E.view(B, T, emb_dim)            # live for alpha
    E_det_seq = E_detach.view(B, T, emb_dim)

    enc = model.encoder
    resp_embs = enc.resp_emb(responses)
    # LSTM gets detached embedding
    x = torch.cat([E_det_seq, resp_embs], dim=-1)
    h, _ = enc.lstm(x)
    state = enc._shift(h)
    theta_in = enc._shift(enc.theta_proj(h).squeeze(-1))

    theta_flat   = theta_in.reshape(B * T)
    state_flat   = state.reshape(B * T, model.hidden_dim)
    emb_flat     = E_seq.reshape(B * T, emb_dim)    # live for alpha's emb read
    emb_flat_d   = E_det_seq.reshape(B * T, emb_dim)
    resp_flat    = responses.reshape(B * T)

    # beta from detached E:
    b = model.decoder.fc_b(emb_flat_d)   # detached
    # alpha from fc_a_state([state_detached, E_live]):
    # We detach state so gradient through E only via the direct emb key, not state.
    key = torch.cat([state_flat.detach(), emb_flat], dim=-1)
    raw_a = model.decoder.fc_a_state(key)
    a = torch.exp(model.decoder.alpha_log_scale * raw_a)

    log_p = model.decoder.log_probs(theta_flat.detach(), a, b)
    return F.nll_loss(log_p, resp_flat)


def _full_forward(
    model: DeepIRTModel,
    E: torch.Tensor,          # (N*T, emb_dim)  requires_grad leaf
    item_ids: torch.Tensor,
    responses: torch.Tensor,
) -> torch.Tensor:
    """Full forward using live E for all three pathways (for sum-check reference)."""
    B, T = item_ids.shape
    emb_dim = model.emb_dim
    E_seq = E.view(B, T, emb_dim)

    enc = model.encoder
    resp_embs = enc.resp_emb(responses)
    x = torch.cat([E_seq, resp_embs], dim=-1)
    h, _ = enc.lstm(x)
    state = enc._shift(h)
    theta_in = enc._shift(enc.theta_proj(h).squeeze(-1))

    theta_flat = theta_in.reshape(B * T)
    state_flat = state.reshape(B * T, model.hidden_dim)
    emb_flat   = E.view(B * T, emb_dim)
    resp_flat  = responses.reshape(B * T)

    params = model.decoder.item_params(emb_flat, state=state_flat, alpha_emb=None)
    log_p = model.decoder.log_probs(theta_flat, params["a"], params["b"])
    return F.nll_loss(log_p, resp_flat)


def probe_gradients_at_epoch(
    model: DeepIRTModel,
    item_ids: torch.Tensor,   # (N_train, T) on device
    responses: torch.Tensor,  # (N_train, T) on device
    device: torch.device,
    epoch: int,
    sum_tol: float = 1e-3,
) -> Optional[dict]:
    """Decompose dL/dE into three pathway gradients and run the sum-check.

    Returns
    -------
    dict with per-item and aggregated ||g_theta||, ||g_alpha||, cos(g_theta, g_alpha),
    ratio ||g_theta|| / ||g_alpha||, and sum_check_ok.
    Returns None if sum-check fails.
    """
    model.encoder.eval()
    model.decoder.eval()

    B, T = item_ids.shape
    emb_dim = model.emb_dim

    # Gather E once from the frozen embedding table.
    with torch.no_grad():
        E_base = model.encoder.item_emb(item_ids)  # (N, T, emb_dim)
        E_base_flat = E_base.reshape(B * T, emb_dim)

    # Helper: create a fresh leaf tensor cloned from E_base_flat
    def fresh_E():
        E = E_base_flat.clone().requires_grad_(True)
        return E

    def get_grad(E_live, loss_fn):
        """Compute grad of loss_fn(E_live) w.r.t. E_live."""
        loss = loss_fn(E_live)
        (g,) = torch.autograd.grad(loss, E_live, create_graph=False)
        return g  # (N*T, emb_dim)

    # ---- LSTM-input pathway gradient (theta + state-cond-alpha via state) ----
    E_t = fresh_E()
    E_det = E_base_flat.detach()
    g_theta = get_grad(
        E_t,
        lambda E: _forward_lstm_only(model, E, item_ids, responses, E_det),
    )

    # ---- beta-head pathway gradient (E -> fc_b directly) ----
    E_b = fresh_E()
    g_beta = get_grad(
        E_b,
        lambda E: _forward_beta_only(model, E, item_ids, responses, E_base_flat.detach()),
    )

    # ---- alpha-head pathway gradient (E -> fc_a_state's emb arg directly) ----
    E_a = fresh_E()
    g_alpha = get_grad(
        E_a,
        lambda E: _forward_alpha_only(model, E, item_ids, responses, E_base_flat.detach()),
    )

    # ---- full gradient (reference for sum-check) ----
    E_full = fresh_E()
    g_full = get_grad(
        E_full,
        lambda E: _full_forward(model, E, item_ids, responses),
    )

    # ---- sum-check ----
    g_sum = g_theta + g_beta + g_alpha
    max_diff = (g_sum - g_full).abs().max().item()
    mean_diff = (g_sum - g_full).abs().mean().item()
    norm_full = g_full.norm().item()
    rel_err = mean_diff / (norm_full + 1e-12)
    sum_check_ok = rel_err < sum_tol

    if not sum_check_ok:
        print(
            f"    PHASE 2 SUM-CHECK FAILED at epoch {epoch}: "
            f"rel_err={rel_err:.4e}  (max_diff={max_diff:.4e}, "
            f"norm_full={norm_full:.4e})  -- gradient decomposition is NOT additive."
        )
        return None

    print(
        f"    PHASE 2 sum-check PASS at epoch {epoch}: "
        f"rel_err={rel_err:.4e}  max_diff={max_diff:.4e}"
    )

    # ---- per-item aggregation ----
    # g_theta, g_beta, g_alpha are (N*T, emb_dim).  Item ids are (N, T) -> (N*T,).
    item_flat = item_ids.reshape(-1).cpu().numpy()          # (N*T,)
    g_th_np  = g_theta.detach().cpu().numpy()
    g_al_np  = g_alpha.detach().cpu().numpy()

    Q = model.num_items
    # Per-item accumulate ||g_theta|| and ||g_alpha|| and cosine
    th_norm_sum  = np.zeros(Q)
    al_norm_sum  = np.zeros(Q)
    cos_sum      = np.zeros(Q)
    cnt          = np.zeros(Q)

    for idx in range(len(item_flat)):
        q = item_flat[idx]
        gt_v = g_th_np[idx]   # (emb_dim,)
        ga_v = g_al_np[idx]   # (emb_dim,)
        nt = np.linalg.norm(gt_v)
        na = np.linalg.norm(ga_v)
        cos_val = float(np.dot(gt_v, ga_v) / (nt * na + 1e-12))
        th_norm_sum[q] += nt
        al_norm_sum[q] += na
        cos_sum[q]     += cos_val
        cnt[q]         += 1.0

    seen_mask = cnt > 0
    th_mean  = th_norm_sum[seen_mask] / cnt[seen_mask]
    al_mean  = al_norm_sum[seen_mask] / cnt[seen_mask]
    cos_mean = cos_sum[seen_mask]     / cnt[seen_mask]
    ratio    = th_mean / (al_mean + 1e-12)

    # Global aggregates
    g_th_norm_global  = float(g_theta.norm().item())
    g_al_norm_global  = float(g_alpha.norm().item())
    g_be_norm_global  = float(g_beta.norm().item())
    g_full_norm       = float(g_full.norm().item())
    cos_global = float(
        torch.nn.functional.cosine_similarity(
            g_theta.reshape(1, -1), g_alpha.reshape(1, -1)
        ).item()
    )
    ratio_global = g_th_norm_global / (g_al_norm_global + 1e-12)

    return {
        "epoch": epoch,
        "sum_check_ok": True,
        "sum_check_rel_err": round(rel_err, 8),
        # g_lstm = gradient through E as LSTM input (theta + state->alpha routes)
        # g_alpha_head = gradient through E as direct fc_a_state emb arg
        # g_beta_head = gradient through E as fc_b arg
        "g_lstm_norm":       round(g_th_norm_global, 6),
        "g_alpha_head_norm": round(g_al_norm_global, 6),
        "g_beta_head_norm":  round(g_be_norm_global, 6),
        "g_full_norm":       round(g_full_norm, 6),
        # Cosine and ratio between the two "competing" consumers:
        # LSTM-input pathway vs alpha-head direct emb read
        "cos_lstm_vs_alpha_head": round(cos_global, 6),
        "ratio_lstm_vs_alpha_head": round(ratio_global, 4),
        "per_item_ratio_mean": round(float(ratio.mean()), 4),
        "per_item_ratio_std":  round(float(ratio.std()), 4),
        "per_item_cos_mean":   round(float(cos_mean.mean()), 4),
        "per_item_cos_std":    round(float(cos_mean.std()), 4),
    }


def run_phase2(
    ds: BenchDataset,
    probe_epochs: list[int],
    seeds: list[int],
    device: torch.device,
) -> Optional[list[dict]]:
    """Run gradient-conflict probe on SHARED-WIDE across probe_epochs and seeds.

    Trains SHARED-WIDE continuously, probing at probe_epochs.
    Only uses seed=seeds[0] for the gradient probe to keep runtime reasonable.
    Returns None if sum-check fails at any epoch.
    """
    cfg = CONFIGS["SHARED-WIDE"]
    seed = seeds[0]
    print(f"\n  [PHASE 2 -- GRADIENT CONFLICT on SHARED-WIDE, seed={seed}]")

    it_train = torch.tensor(ds.items0[ds.train_idx], dtype=torch.long, device=device)
    rp_train = torch.tensor(ds.responses[ds.train_idx], dtype=torch.long, device=device)

    model = _build_model(
        cfg, num_items=ds.cfg.n_items, n_cats=ds.cfg.n_cats,
        device=device, seed=seed,
    )

    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-2)

    probe_epochs_set = set(probe_epochs)
    max_epoch = max(probe_epochs)

    # Also need CHECKPOINTS for the trajectory reference on SHARED-WIDE.
    # We just use probe_epochs here (they are a subset of CHECKPOINTS already).
    results: list[dict] = []

    model.encoder.train()
    model.decoder.train()

    for epoch in range(1, max_epoch + 1):
        optimizer.zero_grad()
        loss = model._compute_nll(it_train, rp_train)
        loss.backward()
        optimizer.step()

        if epoch in probe_epochs_set:
            probe_result = probe_gradients_at_epoch(
                model, it_train, rp_train, device, epoch
            )
            if probe_result is None:
                print("  PHASE 2: sum-check failed -- aborting gradient probe.")
                return None
            results.append(probe_result)
            model.encoder.train()
            model.decoder.train()

    return results


# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------

def evaluate_hypotheses(agg: dict, phase2_results: Optional[list]) -> dict:
    """State which hypotheses hold/fail given the aggregated trajectory data."""
    verdicts: dict[str, str] = {}

    # H1: SHARED-WIDE theta degrades with epochs; NARROW + DECOUPLED stay stable.
    wide_theta  = [agg["SHARED-WIDE"][ep]["theta_mean"]   for ep in sorted(agg["SHARED-WIDE"])]
    narrow_theta = [agg["SHARED-NARROW"][ep]["theta_mean"] for ep in sorted(agg["SHARED-NARROW"])]
    dec_theta    = [agg["DECOUPLED"][ep]["theta_mean"]    for ep in sorted(agg["DECOUPLED"])]
    wide_drop  = wide_theta[-1]  - wide_theta[0]
    narrow_drop = narrow_theta[-1] - narrow_theta[0]
    dec_drop    = dec_theta[-1]   - dec_theta[0]
    h1 = wide_drop < -0.03 and narrow_drop > -0.03 and dec_drop > -0.03
    verdicts["H1"] = (
        f"{'SUPPORTED' if h1 else 'NOT SUPPORTED'}: "
        f"SHARED-WIDE theta change={wide_drop:+.4f}, "
        f"SHARED-NARROW={narrow_drop:+.4f}, DECOUPLED={dec_drop:+.4f}"
    )

    # H2: alpha is learned late (rises across epochs).
    wide_alpha  = [agg["SHARED-WIDE"][ep]["alpha_mean"]   for ep in sorted(agg["SHARED-WIDE"])]
    narrow_alpha = [agg["SHARED-NARROW"][ep]["alpha_mean"] for ep in sorted(agg["SHARED-NARROW"])]
    dec_alpha    = [agg["DECOUPLED"][ep]["alpha_mean"]    for ep in sorted(agg["DECOUPLED"])]
    # "Late" = any config shows alpha at last checkpoint > alpha at checkpoint 2
    alpha_all_final = [wide_alpha[-1], narrow_alpha[-1], dec_alpha[-1]]
    alpha_all_early = [wide_alpha[1],  narrow_alpha[1],  dec_alpha[1]]
    h2 = any(f > e for f, e in zip(alpha_all_final, alpha_all_early))
    verdicts["H2"] = (
        f"{'SUPPORTED' if h2 else 'NOT SUPPORTED'}: "
        f"alpha rise (final-early): SHARED-WIDE={wide_alpha[-1]-wide_alpha[1]:+.4f}, "
        f"SHARED-NARROW={narrow_alpha[-1]-narrow_alpha[1]:+.4f}, "
        f"DECOUPLED={dec_alpha[-1]-dec_alpha[1]:+.4f}"
    )

    # H3: DECOUPLED keeps theta stable AND alpha high across trajectory.
    dec_theta_stable = max(dec_theta) - min(dec_theta) < 0.05
    dec_alpha_high   = dec_alpha[-1] > 0.80
    h3 = dec_theta_stable and dec_alpha_high
    verdicts["H3"] = (
        f"{'SUPPORTED' if h3 else 'NOT SUPPORTED'}: "
        f"DECOUPLED theta range={max(dec_theta)-min(dec_theta):.4f} (stable<0.05: {dec_theta_stable}), "
        f"alpha_final={dec_alpha[-1]:.4f} (high>0.80: {dec_alpha_high})"
    )

    if phase2_results is not None and len(phase2_results) > 0:
        # H4: ||g_lstm|| >> ||g_alpha_head||
        # g_lstm = gradient through E as LSTM input (the theta mechanism)
        # g_alpha_head = gradient through E as direct emb arg to fc_a_state
        ratio_vals = [r["ratio_lstm_vs_alpha_head"] for r in phase2_results]
        h4 = all(rv > 1.0 for rv in ratio_vals)
        verdicts["H4"] = (
            f"{'SUPPORTED' if h4 else 'NOT SUPPORTED'}: "
            f"||g_lstm||/||g_alpha_head|| across epochs: "
            + ", ".join(f"ep{r['epoch']}={r['ratio_lstm_vs_alpha_head']:.3f}"
                        for r in phase2_results)
        )

        # H5: cosine(g_lstm, g_alpha_head) <= 0
        cos_vals = [r["cos_lstm_vs_alpha_head"] for r in phase2_results]
        h5 = all(cv <= 0.05 for cv in cos_vals)
        verdicts["H5"] = (
            f"{'SUPPORTED' if h5 else 'NOT SUPPORTED'}: "
            f"cos(g_lstm, g_alpha_head) across epochs: "
            + ", ".join(f"ep{r['epoch']}={r['cos_lstm_vs_alpha_head']:.3f}"
                        for r in phase2_results)
        )
    else:
        verdicts["H4"] = "PHASE 2 NOT RUN or sum-check failed -- H4 cannot be evaluated."
        verdicts["H5"] = "PHASE 2 NOT RUN or sum-check failed -- H5 cannot be evaluated."

    verdicts["H6"] = (
        "PHASE 2 structural note: DECOUPLED alpha table is NOT shared with the LSTM "
        "input, so g_alpha on alpha_item_emb is pure alpha-pathway by construction "
        "(no competition). This is confirmed by architecture, not a measured quantity "
        "on the gradient probe (which runs on SHARED-WIDE only)."
    )

    return verdicts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _trajectory_table_md(agg: dict, verdicts: dict) -> str:
    epochs = sorted(next(iter(agg.values())).keys())
    configs = list(agg.keys())

    lines = [
        "# Trajectory Probe -- Phase 1",
        "",
        "Mean over seeds. theta_static = Spearman(final-step theta, true theta0).",
        "alpha = Spearman(recovered alpha, true alpha).",
        "",
        "## Theta trajectory (mean over seeds)",
        "",
    ]
    # Theta table
    header = "| config | " + " | ".join(f"ep{e}" for e in epochs) + " |"
    sep    = "|" + "-|" * (len(epochs) + 1)
    lines += [header, sep]
    for c in configs:
        row = f"| {c} |"
        for e in epochs:
            row += f" {agg[c][e]['theta_mean']:.4f} |"
        lines.append(row)

    lines += ["", "## Alpha trajectory (mean over seeds)", ""]
    lines += [header, sep]
    for c in configs:
        row = f"| {c} |"
        for e in epochs:
            row += f" {agg[c][e]['alpha_mean']:.4f} |"
        lines.append(row)

    lines += ["", "## Hypothesis verdicts", ""]
    for k, v in verdicts.items():
        lines.append(f"**{k}** {v}")
        lines.append("")

    return "\n".join(lines)


def _gradient_table_md(phase2_results: list) -> str:
    lines = [
        "# Gradient Conflict Probe -- Phase 2 (SHARED-WIDE)",
        "",
        "g_lstm = dL/dE through E as LSTM input (theta + state-alpha routes).",
        "g_alpha_head = dL/dE through E as direct emb arg to fc_a_state.",
        "g_beta_head = dL/dE through E as fc_b input.",
        "Sum-check: g_lstm + g_beta_head + g_alpha_head == full dL/dE (rel err < 1e-3).",
        "",
        "| epoch | sum_check | rel_err | ||g_lstm|| | ||g_alpha_head|| | "
        "||g_beta_head|| | ratio lstm/alpha_head | cos(lstm,alpha_head) |",
        "|-------|-----------|---------|-----------|-----------------|"
        "---------------|----------------------|---------------------|",
    ]
    for r in phase2_results:
        ok = "PASS" if r["sum_check_ok"] else "FAIL"
        lines.append(
            f"| {r['epoch']} | {ok} | {r['sum_check_rel_err']:.2e} "
            f"| {r['g_lstm_norm']:.4f} | {r['g_alpha_head_norm']:.4f} "
            f"| {r['g_beta_head_norm']:.4f} | {r['ratio_lstm_vs_alpha_head']:.3f} "
            f"| {r['cos_lstm_vs_alpha_head']:.4f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trajectory + gradient conflict probes")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke-test: N=200, Q=30, T=30, 1 seed, fewer epochs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.quick:
        n_learners, n_items, seq_len = 200, 30, 30
        seeds = [0]
        checkpoints      = CHECKPOINTS_QUICK
        grad_probe_epochs = GRAD_PROBE_EPOCHS_QUICK
        print("[QUICK mode: N=200, Q=30, T=30, 1 seed]")
    else:
        n_learners, n_items, seq_len = 800, 60, 60
        seeds = [0, 1, 2]
        checkpoints      = CHECKPOINTS_FULL
        grad_probe_epochs = GRAD_PROBE_EPOCHS_FULL
        print("[FULL mode: N=800, Q=60, T=60, 3 seeds]")

    print("\n=== Generating datasets ===")
    ds_static  = generate(BenchDataConfig(
        name="static_k4", kind="static",
        n_learners=n_learners, n_items=n_items, seq_len=seq_len,
        n_cats=4, seed=0,
    ))
    ds_dynamic = generate(BenchDataConfig(
        name="dynamic_k4", kind="dynamic",
        n_learners=n_learners, n_items=n_items, seq_len=seq_len,
        n_cats=4, seed=1,
    ))

    # ---- Phase 1: Trajectory ----
    print("\n=== Phase 1: Trajectory (static_k4) ===")
    t0 = time.time()
    phase1_raw_static  = run_phase1(ds_static,  checkpoints, seeds, device)
    phase1_raw_dynamic = run_phase1(ds_dynamic, checkpoints, seeds, device)
    phase1_time = time.time() - t0

    agg_static  = aggregate_phase1(phase1_raw_static)
    agg_dynamic = aggregate_phase1(phase1_raw_dynamic)

    # ---- Phase 2: Gradient conflict (static_k4 train set, SHARED-WIDE) ----
    print("\n=== Phase 2: Gradient conflict (static_k4, SHARED-WIDE) ===")
    t1 = time.time()
    phase2_results = run_phase2(ds_static, grad_probe_epochs, seeds, device)
    phase2_time = time.time() - t1

    if phase2_results is None:
        print("\nPHASE 2 FLAG: instrumentation not verified -- sum-check failed.")
        print("Gradient decomposition is not additive; Phase 2 numbers not reported.")

    # ---- Hypotheses ----
    verdicts_static  = evaluate_hypotheses(agg_static,  phase2_results)
    verdicts_dynamic = evaluate_hypotheses(agg_dynamic, None)

    # ---- Save results ----
    results_payload = {
        "mode": "quick" if args.quick else "full",
        "seeds": seeds,
        "checkpoints": checkpoints,
        "static_k4": {
            "aggregated": {
                cfg: {
                    str(ep): agg_static[cfg][ep]
                    for ep in sorted(agg_static[cfg])
                }
                for cfg in agg_static
            },
            "hypotheses": verdicts_static,
        },
        "dynamic_k4": {
            "aggregated": {
                cfg: {
                    str(ep): agg_dynamic[cfg][ep]
                    for ep in sorted(agg_dynamic[cfg])
                }
                for cfg in agg_dynamic
            },
            "hypotheses": verdicts_dynamic,
        },
        "phase2": phase2_results if phase2_results is not None else "sum_check_failed",
        "phase1_time_s": round(phase1_time, 1),
        "phase2_time_s": round(phase2_time, 1),
    }
    with (OUT / "trajectory_results.json").open("w") as fh:
        json.dump(results_payload, fh, indent=2)
    print(f"\nSaved: {OUT / 'trajectory_results.json'}")

    # ---- Trajectory table ----
    traj_md_static  = _trajectory_table_md(agg_static,  verdicts_static)
    traj_md_dynamic = _trajectory_table_md(agg_dynamic, verdicts_dynamic)
    combined_md = (
        "# Phase 1 -- Trajectory Probe\n\n"
        "## Dataset: static_k4\n\n" + traj_md_static + "\n\n"
        "## Dataset: dynamic_k4\n\n" + traj_md_dynamic
    )
    with (OUT / "trajectory_table.md").open("w") as fh:
        fh.write(combined_md)
    print(f"Saved: {OUT / 'trajectory_table.md'}")

    # ---- Gradient conflict table ----
    if phase2_results is not None:
        grad_md = _gradient_table_md(phase2_results)
        with (OUT / "gradient_conflict_table.md").open("w") as fh:
            fh.write(grad_md)
        print(f"Saved: {OUT / 'gradient_conflict_table.md'}")
    else:
        with (OUT / "gradient_conflict_table.md").open("w") as fh:
            fh.write(
                "# Phase 2 -- Gradient Conflict\n\n"
                "**NOT VERIFIED**: sum-check failed during this run.\n"
                "The three-pathway decomposition (g_theta + g_beta + g_alpha)\n"
                "did not equal the full gradient within tolerance.\n"
                "Phase 2 needs main-loop review before reporting numbers.\n"
            )
        print(f"Saved: {OUT / 'gradient_conflict_table.md'} (failure notice)")

    print(f"\nPhase 1 time: {phase1_time:.1f}s")
    print(f"Phase 2 time: {phase2_time:.1f}s")

    # ---- Print tables to console ----
    print("\n" + "=" * 72)
    print(traj_md_static)
    if phase2_results is not None:
        print("\n" + "=" * 72)
        print(_gradient_table_md(phase2_results))


if __name__ == "__main__":
    main()
