"""
anchor.py  --  Phase 2: structural anchoring of extension items.

Mechanism:
  1. Freeze the entire trained model (encoder, decoder, B-item embeddings).
  2. For each shared (anchor) learner compute theta_L = encoder(B-only sequence)
     final theta, held fixed as a constant.
  3. Add E new trainable item embeddings (rows B..B+E-1 are new; rows 0..B-1 frozen).
  4. Optimize ONLY the E new embeddings to minimize GPCM NLL of the E responses,
     using frozen decoder and the fixed theta_L values.
  5. Recover E-item params from the fitted embeddings.

The encoder and decoder weights do NOT change.  Existing B-item embeddings do NOT
change.  Only E new embedding vectors are trained.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import Dataset
from model import AbilityModel, GPCMDecoder


class AnchoredExtension(nn.Module):
    """
    Holds only the E new item embeddings (trainable).
    Uses the frozen decoder to compute GPCM log-probs against fixed thetas.
    """

    def __init__(self, n_ext: int, emb_dim: int, n_cats: int,
                 frozen_decoder: GPCMDecoder):
        super().__init__()
        self.n_ext = n_ext
        self.emb_dim = emb_dim
        self.n_cats = n_cats
        # only these are trained
        self.ext_emb = nn.Embedding(n_ext, emb_dim)
        # decoder is frozen -- store reference but do NOT register as submodule
        # to keep its parameters out of self.parameters()
        self._decoder = frozen_decoder

    def nll(self, theta_fixed: torch.Tensor,
            item_local_ids: torch.Tensor,
            responses: torch.Tensor) -> torch.Tensor:
        """
        theta_fixed:    (n_anchor,) fixed ability estimates (no grad)
        item_local_ids: (n_anchor, T_ext) local E-item indices (0..E-1)
        responses:      (n_anchor, T_ext) responses (0..K-1)
        returns mean NLL
        """
        n_anchor, T = item_local_ids.shape
        embs = self.ext_emb(item_local_ids)           # (n_anchor, T, emb_dim)
        embs_flat = embs.view(n_anchor * T, self.emb_dim)

        # repeat theta for each timestep
        theta_rep = theta_fixed.unsqueeze(1).expand(n_anchor, T)
        theta_flat = theta_rep.reshape(n_anchor * T)

        with torch.no_grad():
            # decoder is frozen; we call its methods but block grad on weights
            pass
        a_flat, b_flat = self._decoder.item_params(embs_flat)
        log_p = self._decoder.log_probs(theta_flat, a_flat, b_flat)  # (nT, K)
        resp_flat = responses.reshape(n_anchor * T)
        return F.nll_loss(log_p, resp_flat)


def anchor_extend(dataset: Dataset,
                  phase1: dict,
                  n_epochs: int = 300,
                  lr: float = 1e-2,
                  device: torch.device = torch.device("cpu"),
                  verbose: bool = True) -> dict:
    """
    Phase 2: anchored extension.

    Returns dict with:
      a_ext:     (E,) recovered discrimination
      b_ext:     (E, K-1) recovered step thresholds (sorted)
      train_time: wall-clock seconds
      n_params:  number of trainable parameters updated
    """
    torch.manual_seed(0)
    cfg = dataset.cfg
    B, E, K = cfg.n_base, cfg.n_ext, cfg.n_cats

    base_model: AbilityModel = phase1["model"]
    base_model.eval()
    # freeze everything in the base model
    for p in base_model.parameters():
        p.requires_grad_(False)

    # --- compute fixed thetas for anchor learners from B-only sequences ---
    anchor_indices = np.where(dataset.anchor_mask)[0]
    base_seqs = dataset.base_seqs
    base_items = torch.tensor(
        [base_seqs[i]["items"] for i in anchor_indices],
        dtype=torch.long, device=device
    )
    base_resps = torch.tensor(
        [base_seqs[i]["responses"] for i in anchor_indices],
        dtype=torch.long, device=device
    )

    with torch.no_grad():
        theta_fixed = base_model.get_final_theta(base_items, base_resps)  # (n_anchor,)

    # --- build E-response tensors (anchor learners, all E items, shuffled order) ---
    # ext_seqs is ordered by anchor_indices (same order as anchor_mask)
    ext_item_ids = torch.tensor(
        [s["items"] for s in dataset.ext_seqs],
        dtype=torch.long, device=device
    )  # (n_anchor, E)
    ext_resps = torch.tensor(
        [s["responses"] for s in dataset.ext_seqs],
        dtype=torch.long, device=device
    )  # (n_anchor, E)

    # --- anchored extension module ---
    ext_module = AnchoredExtension(
        n_ext=E, emb_dim=base_model.emb_dim,
        n_cats=K, frozen_decoder=base_model.decoder
    ).to(device)

    # only ext_emb parameters are trainable
    n_params = sum(p.numel() for p in ext_module.parameters() if p.requires_grad)
    optimizer = optim.Adam(ext_module.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ext_module.train()
        optimizer.zero_grad()
        loss = ext_module.nll(theta_fixed.detach(), ext_item_ids, ext_resps)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  [Phase 2] epoch {epoch:4d}  NLL={loss.item():.4f}")
    train_time = time.time() - t0

    # --- recover E-item params ---
    ext_module.eval()
    with torch.no_grad():
        item_ids = torch.arange(E, device=device)
        embs = ext_module.ext_emb(item_ids)                   # (E, emb_dim)
        a_raw, b_raw = base_model.decoder.item_params(embs)
        a_ext = a_raw.squeeze(-1).cpu().numpy()               # (E,)
        b_ext = torch.sort(b_raw, dim=1).values.cpu().numpy() # (E, K-1)

    # --- build extended model: frozen Phase-1 encoder with E new embedding rows ---
    # Copy base model and append E-item embeddings so the frozen encoder can
    # process B+E interleaved sequences in the tracking test.
    extended_model = _build_extended_model(base_model, ext_module, B, E, device)

    return {
        "a_ext": a_ext,
        "b_ext": b_ext,
        "train_time": train_time,
        "n_params": n_params,
        "extended_model": extended_model,   # AbilityModel with B+E items, encoder frozen
    }


def _build_extended_model(base_model: AbilityModel,
                          ext_module: AnchoredExtension,
                          B: int, E: int,
                          device: torch.device) -> AbilityModel:
    """
    Return a new AbilityModel(B+E items) whose encoder weights, B-item embeddings,
    and decoder are identical to the frozen Phase-1 model, and whose rows B..B+E-1
    are the calibrated E-item embeddings from ext_module.

    The returned model has all parameters frozen (requires_grad=False).
    """
    emb_dim = base_model.emb_dim
    hidden_dim = base_model.hidden_dim
    n_cats = base_model.n_cats

    ext_model = AbilityModel(num_items=B + E, emb_dim=emb_dim,
                             hidden_dim=hidden_dim, n_cats=n_cats).to(device)

    with torch.no_grad():
        # copy B-item embeddings
        ext_model.item_emb.weight[:B].copy_(base_model.item_emb.weight)
        # append E-item embeddings
        ext_model.item_emb.weight[B:].copy_(ext_module.ext_emb.weight)

        # copy response embeddings
        ext_model.resp_emb.weight.copy_(base_model.resp_emb.weight)

        # copy LSTM weights
        ext_model.lstm.load_state_dict(base_model.lstm.state_dict())

        # copy theta projection
        ext_model.theta_proj.load_state_dict(base_model.theta_proj.state_dict())

        # copy decoder
        ext_model.decoder.load_state_dict(base_model.decoder.state_dict())

    # freeze everything
    for p in ext_model.parameters():
        p.requires_grad_(False)

    ext_model.eval()
    return ext_model
