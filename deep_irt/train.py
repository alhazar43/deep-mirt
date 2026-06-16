"""
train.py  --  Phase 1: base calibration.

Trains encoder + decoder + B-item embeddings on the B-item responses (full batch,
Adam, a few hundred epochs).  Returns the trained model and recovered parameters.
"""

import time
import numpy as np
import torch
import torch.optim as optim

from data import Dataset
from model import AbilityModel


def _build_tensors(seqs: list, device: torch.device):
    """Convert list-of-dicts to padded tensors (all seqs same length here)."""
    items = torch.tensor([s["items"] for s in seqs], dtype=torch.long, device=device)
    resps = torch.tensor([s["responses"] for s in seqs], dtype=torch.long, device=device)
    return items, resps


def train_base(dataset: Dataset,
               emb_dim: int = 8,
               hidden_dim: int = 32,
               n_epochs: int = 300,
               lr: float = 1e-2,
               device: torch.device = torch.device("cpu"),
               verbose: bool = True) -> dict:
    """
    Phase 1: train on base (B-item) sequences.

    Returns a dict with:
      model:      trained AbilityModel
      theta_base: (N,) recovered ability from B-only sequence
      a_base:     (B,) recovered discrimination
      b_base:     (B, K-1) recovered step thresholds (sorted)
      train_time: wall-clock seconds
    """
    torch.manual_seed(0)
    cfg = dataset.cfg
    N, B, K = cfg.n_learners, cfg.n_base, cfg.n_cats

    model = AbilityModel(num_items=B, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, n_cats=K).to(device)

    items_t, resps_t = _build_tensors(dataset.base_seqs, device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = model(items_t, resps_t)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  [Phase 1] epoch {epoch:4d}  NLL={loss.item():.4f}")
    train_time = time.time() - t0

    # --- recover parameters ---
    model.eval()
    with torch.no_grad():
        theta_base = model.get_final_theta(items_t, resps_t).cpu().numpy()  # (N,)

        item_ids = torch.arange(B, device=device)
        embs = model.item_emb(item_ids)                     # (B, emb_dim)
        a_raw, b_raw = model.decoder.item_params(embs)
        a_base = a_raw.squeeze(-1).cpu().numpy()            # (B,)
        b_base = torch.sort(b_raw, dim=1).values.cpu().numpy()  # (B, K-1)

    return {
        "model": model,
        "theta_base": theta_base,
        "a_base": a_base,
        "b_base": b_base,
        "train_time": train_time,
    }
