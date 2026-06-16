"""
baseline.py  --  full recalibration baseline.

Retrains the WHOLE model from scratch (fresh init) on B+E responses together.
Every learner contributes their B-item sequence.  Anchor learners also contribute
their E-item sequence.  Both are presented as a single concatenated sequence.

Returns recovered parameters for all items and all learners.
"""

import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from data import Dataset
from model import AbilityModel


def _build_combined_seqs(dataset: Dataset):
    """
    For each learner build a combined sequence (base items + ext items if anchor).
    Items: base items keep ids 0..B-1; ext items become B..B+E-1.
    """
    cfg = dataset.cfg
    B = cfg.n_base
    anchor_indices = set(int(i) for i in np.where(dataset.anchor_mask)[0])

    # map anchor learner index -> ext_seq index
    ext_seq_map = {s["learner_idx"]: s for s in dataset.ext_seqs}

    combined = []
    for i, base_seq in enumerate(dataset.base_seqs):
        items = list(base_seq["items"])
        resps = list(base_seq["responses"])
        if i in anchor_indices:
            ext_seq = ext_seq_map[i]
            # shift ext item ids by B
            items += [j + B for j in ext_seq["items"]]
            resps += list(ext_seq["responses"])
        combined.append({"items": items, "responses": resps})
    return combined


def train_baseline(dataset: Dataset,
                   emb_dim: int = 8,
                   hidden_dim: int = 32,
                   n_epochs: int = 300,
                   lr: float = 1e-2,
                   device: torch.device = torch.device("cpu"),
                   verbose: bool = True) -> dict:
    """
    Baseline: train from scratch on B+E responses.

    Non-anchor learners have sequences of length B.
    Anchor learners have sequences of length B+E (base then ext, randomly shuffled
    within each part as generated).

    Because sequences are different lengths, we pad to max length and use
    a mask for the loss.

    Returns:
      model:       trained AbilityModel (B+E items)
      theta_all:   (N,) recovered theta (from each learner's full sequence)
      theta_base_only: (N,) recovered theta using ONLY base items (rerun encoder
                   on the base-item prefix of each sequence, for comparison with
                   Phase 1 theta)
      a_base:      (B,) recovered base discrimination
      b_base:      (B, K-1) recovered base thresholds (sorted)
      a_ext:       (E,) recovered ext discrimination
      b_ext:       (E, K-1) recovered ext thresholds (sorted)
      train_time:  wall-clock seconds
      n_params:    total trainable parameters
    """
    torch.manual_seed(0)
    cfg = dataset.cfg
    N, B, E, K = cfg.n_learners, cfg.n_base, cfg.n_ext, cfg.n_cats

    combined = _build_combined_seqs(dataset)

    # pad sequences to max length
    max_len = max(len(s["items"]) for s in combined)
    items_np = np.zeros((N, max_len), dtype=np.int64)
    resps_np = np.zeros((N, max_len), dtype=np.int64)
    lengths = np.array([len(s["items"]) for s in combined])

    for i, s in enumerate(combined):
        T = len(s["items"])
        items_np[i, :T] = s["items"]
        resps_np[i, :T] = s["responses"]

    items_t = torch.tensor(items_np, device=device)
    resps_t = torch.tensor(resps_np, device=device)
    lengths_t = torch.tensor(lengths, device=device)

    model = AbilityModel(num_items=B + E, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, n_cats=K).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # custom training loop with masking
    def masked_nll(item_ids, responses, lengths_vec):
        theta_seq = model.encode(item_ids, responses)           # (N, T)
        batch, T = theta_seq.shape
        zeros = torch.zeros(batch, 1, device=device)
        theta_in = torch.cat([zeros, theta_seq[:, :-1]], dim=1) # (N, T)

        embs = model.item_emb(item_ids)                         # (N, T, emb_dim)
        embs_flat = embs.view(batch * T, model.emb_dim)
        a_flat, b_flat = model.decoder.item_params(embs_flat)
        theta_flat = theta_in.reshape(batch * T)

        log_p = model.decoder.log_probs(theta_flat, a_flat, b_flat)  # (NT, K)
        resp_flat = responses.reshape(batch * T)

        # mask: position t is valid if t < length[i]
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths_vec.unsqueeze(1)
        mask_flat = mask.reshape(batch * T)

        nll_all = F.nll_loss(log_p, resp_flat, reduction="none")  # (NT,)
        return nll_all[mask_flat].mean()

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = masked_nll(items_t, resps_t, lengths_t)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  [Baseline] epoch {epoch:4d}  NLL={loss.item():.4f}")
    train_time = time.time() - t0

    # --- recover parameters ---
    model.eval()
    with torch.no_grad():
        # theta from full sequence
        theta_all = model.get_final_theta(items_t, resps_t).cpu().numpy()  # (N,)

        # theta from base-item prefix only (length B for all learners)
        base_items_t = items_t[:, :B]
        base_resps_t = resps_t[:, :B]
        theta_base_only = model.get_final_theta(
            base_items_t, base_resps_t).cpu().numpy()

        item_ids = torch.arange(B + E, device=device)
        embs = model.item_emb(item_ids)
        a_raw, b_raw = model.decoder.item_params(embs)
        a_all = a_raw.squeeze(-1).cpu().numpy()
        b_all = torch.sort(b_raw, dim=1).values.cpu().numpy()

    return {
        "model": model,
        "theta_all": theta_all,
        "theta_base_only": theta_base_only,
        "a_base": a_all[:B],
        "b_base": b_all[:B],
        "a_ext": a_all[B:],
        "b_ext": b_all[B:],
        "train_time": train_time,
        "n_params": n_params,
    }
