"""
anchor.py -- Structural anchoring: extend the item bank without disturbing the encoder.

Mechanism
---------
1. Freeze the trained encoder (LSTM weights, theta projection, existing item
   embeddings, and whatever decoder was used in base calibration).
2. For each shared (anchor) learner, compute theta_L = encoder(B-only sequence).
   This theta is held fixed during extension -- it is a constant, not a variable.
3. Append E new trainable item embedding rows.
4. Optimise ONLY those E embedding vectors to minimise the chosen decoder's NLL
   against the anchor learners' E-item responses (using their pinned theta values).
5. Build an extended encoder that contains both B original and E new embeddings so
   the full B+E interleaved sequence can be processed by the frozen encoder.

The operation is decoder-agnostic: any decoder that exposes .nll(theta, emb,
responses) can be used for extension.  The BradleyTerry decoder is an exception
(it is item-only) and uses .nll_pairs() instead.

Public API
----------
anchored_extend(encoder, decoder, anchor_theta, ext_item_ids, ext_responses,
                n_epochs, lr, device) -> (ext_emb_weight, a_ext, b_ext)

build_extended_encoder(encoder, ext_emb_weight) -> new LSTMEncoder with B+E rows
"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deep_irt.core.encoder import LSTMEncoder
from deep_irt.core.decoders import GPCMDecoder, Binary2PLDecoder


# ---------------------------------------------------------------------------
# Trainable extension module (E new embedding rows only)
# ---------------------------------------------------------------------------

class _ExtEmbeddings(nn.Module):
    """Holds only the E new item embeddings; decoder is referenced but not owned."""

    def __init__(self, n_ext: int, emb_dim: int) -> None:
        super().__init__()
        self.ext_emb = nn.Embedding(n_ext, emb_dim)

    def forward(self, local_ids: torch.Tensor) -> torch.Tensor:
        """local_ids: (...) in {0,..,E-1}; returns (..., emb_dim)."""
        return self.ext_emb(local_ids)


# ---------------------------------------------------------------------------
# Core anchoring function
# ---------------------------------------------------------------------------

def anchored_extend(
    encoder: LSTMEncoder,
    decoder: nn.Module,
    anchor_theta: torch.Tensor,
    ext_item_ids: torch.Tensor,
    ext_responses: torch.Tensor,
    n_epochs: int = 300,
    lr: float = 1e-2,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> dict:
    """
    Fit E new item embeddings against fixed anchor-learner thetas.

    Works with any decoder that exposes:
        decoder.nll(theta_flat, emb_flat, responses_flat) -> scalar

    The encoder and decoder are frozen; only the E new embedding rows are trained.

    Parameters
    ----------
    encoder       : trained LSTMEncoder (will be frozen; not modified in place)
    decoder       : any GPCMDecoder / Binary2PLDecoder (frozen; not modified)
    anchor_theta  : (n_anchor,) fixed per-learner ability, no grad
    ext_item_ids  : (n_anchor, E) local E-item indices {0,..,E-1}
    ext_responses : (n_anchor, E) responses
    n_epochs      : optimisation steps
    lr            : Adam learning rate
    device        : torch device
    verbose       : print every 50 epochs

    Returns
    -------
    dict with:
        ext_emb_weight : (E, emb_dim) fitted embedding tensor (detached, cpu)
        n_params       : number of trainable parameters (E * emb_dim)
        train_time     : wall-clock seconds
        final_nll      : NLL at the last epoch
    """
    # Freeze everything
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    decoder.eval()

    n_anchor, E = ext_item_ids.shape
    emb_dim = encoder.emb_dim

    ext_mod = _ExtEmbeddings(n_ext=E, emb_dim=emb_dim).to(device)
    n_params = sum(p.numel() for p in ext_mod.parameters())
    optimizer = optim.Adam(ext_mod.parameters(), lr=lr)

    # Move data to device
    anchor_theta = anchor_theta.to(device).detach()
    ext_item_ids = ext_item_ids.to(device)
    ext_responses = ext_responses.to(device)

    t0 = time.time()
    final_nll = float("nan")

    for epoch in range(1, n_epochs + 1):
        ext_mod.train()
        optimizer.zero_grad()

        # Flatten for decoder: (n_anchor * E, emb_dim)
        embs = ext_mod(ext_item_ids)                            # (n_anchor, E, emb_dim)
        embs_flat = embs.view(n_anchor * E, emb_dim)

        # Repeat theta for each item position: (n_anchor * E,)
        theta_rep = anchor_theta.unsqueeze(1).expand(n_anchor, E)
        theta_flat = theta_rep.reshape(n_anchor * E)

        resp_flat = ext_responses.reshape(n_anchor * E)

        # Decoder NLL (decoder weights are frozen via requires_grad=False)
        loss = decoder.nll(theta_flat, embs_flat, resp_flat)
        loss.backward()
        optimizer.step()

        final_nll = loss.item()
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  [Anchor] epoch {epoch:4d}  NLL={final_nll:.4f}")

    train_time = time.time() - t0

    ext_mod.eval()
    with torch.no_grad():
        ext_emb_weight = ext_mod.ext_emb.weight.cpu().clone()

    return {
        "ext_emb_weight": ext_emb_weight,
        "n_params": n_params,
        "train_time": train_time,
        "final_nll": final_nll,
    }


# ---------------------------------------------------------------------------
# Build extended encoder (B + E embedding rows, encoder weights frozen)
# ---------------------------------------------------------------------------

def build_extended_encoder(
    base_encoder: LSTMEncoder,
    ext_emb_weight: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> LSTMEncoder:
    """
    Return a new LSTMEncoder with B+E item embedding rows.

    The new encoder is a full copy of base_encoder (LSTM, theta_proj, resp_emb,
    B-item embeddings) with E new embedding rows appended.  All parameters are
    frozen (requires_grad=False) and the model is set to eval mode.

    Parameters
    ----------
    base_encoder    : the trained B-item encoder (will not be modified)
    ext_emb_weight  : (E, emb_dim) the fitted extension embeddings (cpu tensor)
    device          : target device for the new encoder

    Returns
    -------
    extended_encoder : LSTMEncoder(B+E items), all weights frozen, eval mode
    """
    B = base_encoder.num_items
    E = ext_emb_weight.shape[0]
    emb_dim = base_encoder.emb_dim
    hidden_dim = base_encoder.hidden_dim
    n_cats = base_encoder.n_cats

    ext_enc = LSTMEncoder(
        num_items=B + E,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_cats=n_cats,
    ).to(device)

    with torch.no_grad():
        ext_enc.item_emb.weight[:B].copy_(base_encoder.item_emb.weight)
        ext_enc.item_emb.weight[B:].copy_(ext_emb_weight.to(device))
        ext_enc.resp_emb.weight.copy_(base_encoder.resp_emb.weight)
        ext_enc.lstm.load_state_dict(base_encoder.lstm.state_dict())
        ext_enc.theta_proj.load_state_dict(base_encoder.theta_proj.state_dict())

    for p in ext_enc.parameters():
        p.requires_grad_(False)
    ext_enc.eval()

    return ext_enc
