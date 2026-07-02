"""_p2_nrm_channels.py -- Canonical NRM channel-routing for ALL 10 Phase-2 configs.

Consolidates the selective-decoupling NRM routing (the shared / one-key /
separate-keys and the a-only / c-only variants) into ONE importable,
self-contained module.
Core ``NRMDecoder`` offers only shared-vs-both-on-one-key; this module owns the
wide-key tables so it can give a_k and c_k SEPARATE keys, and gate each head's
state-conditioning independently.

The 10 configs (LOCKED decision 3: 5 decoupling x {static, dynamic})
--------------------------------------------------------------------
  1  shared         : {theta, a_k, c_k} all on the thin value embedding.
  2  a_only_dec     : a_k on its own wide key; c_k on the thin value  (informative
                      negative -- a_k gets a channel but c_k, the expected
                      minimal-sufficient one, does not).
  3  c_only_dec     : c_k on its own wide key; a_k on the thin value  (expected
                      minimal-sufficient: the intercept carries the marginal
                      option-frequency signal).
  4  decoupled      : a_k and c_k SHARE one wide key; theta on the thin value.
  5  all_decoupled  : a_k and c_k each on their OWN wide key; theta on the value.
each x {static, dynamic}.

X-mode (state-conditioning)
  static  : per-item readout ``fc(item_feature)`` -- occurrence-invariant.
  dynamic : ``fc([state, item_feature])`` for BOTH a_k and c_k, occurrence-
            averaged at recovery (the ma-irt state_alpha recipe).

The unifying-thesis caveat (blueprint) is preserved here in prose: the NRM
minimal-sufficient config gives the INTERCEPT (c_k) its own channel, the
OPPOSITE parameter to GPCM's slope -- frame NRM as the boundary case.

Scratch file: ``_``-prefixed, gitignored.  Reuses the read-only core encoder
only; owns all decoder-side tables.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# The canonical 10-config enumeration
# ---------------------------------------------------------------------------

# coupling labels (the 5 decoupling regimes).
COUPLINGS = ("shared", "a_only_dec", "c_only_dec", "decoupled", "all_decoupled")
X_MODES = ("static", "dynamic")

# Blueprint numbering + one-line rationale, for report labelling.
COUPLING_INFO = {
    "shared":        (1, "all params on the thin value embedding"),
    "a_only_dec":    (2, "a_k on its own wide key; c_k on the thin value"),
    "c_only_dec":    (3, "c_k on its own wide key; a_k on the thin value"),
    "decoupled":     (4, "a_k and c_k share one wide key"),
    "all_decoupled": (5, "a_k and c_k each on their own wide key"),
}

# The full 10-cell grid in canonical display order.
NRM_CONFIGS = [
    {"name": f"{c}/{x}", "coupling": c, "x_mode": x,
     "config_num": COUPLING_INFO[c][0], "desc": COUPLING_INFO[c][1]}
    for c in COUPLINGS for x in X_MODES
]


def list_configs() -> list[dict]:
    """Return the canonical list of all 10 NRM channel configs."""
    return [dict(cfg) for cfg in NRM_CONFIGS]


# ---------------------------------------------------------------------------
# NRMChannelDecoder -- the promoted routing
# ---------------------------------------------------------------------------

class NRMChannelDecoder(nn.Module):
    """Bock NRM decoder with per-head channel routing for all 5 couplings.

    Owns the wide-key embedding tables so ``all_decoupled`` can give a_k and c_k
    SEPARATE keys.  For each head, the readout input is:

        static  head : item_feature
        dynamic head : [encoder_state, item_feature]

    where ``item_feature`` is the thin encoder value ``item_val`` for a shared
    head, or the head's wide key lookup for a decoupled head.  Logits are
    ``a_k * theta + c_k`` with Bock centering (``sum_k a_k = sum_k c_k = 0``).

    Parameters
    ----------
    num_items    : int  -- Q, item-bank size.
    emb_dim      : int  -- thin value width (matches the encoder's item_val_emb).
    n_options    : int  -- K.
    hidden_dim   : int  -- encoder state width (for the dynamic x-mode).
    coupling     : str  -- one of :data:`COUPLINGS`.
    x_mode       : str  -- "static" | "dynamic".
    item_key_dim : int  -- wide-key width for decoupled heads.
    correct_option : int -- designated-correct option (bookkeeping only).
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int,
        n_options: int,
        hidden_dim: int,
        coupling: str,
        x_mode: str,
        item_key_dim: int = 64,
        correct_option: int = 0,
    ) -> None:
        super().__init__()
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
        if x_mode not in X_MODES:
            raise ValueError(f"x_mode must be one of {X_MODES}, got {x_mode!r}")
        if n_options < 2:
            raise ValueError("n_options must be >= 2")

        self.num_items = num_items
        self.emb_dim = emb_dim
        self.n_options = n_options
        self.hidden_dim = hidden_dim
        self.coupling = coupling
        self.x_mode = x_mode
        self.item_key_dim = item_key_dim
        self.correct_option = correct_option

        # Both heads share the x-mode: static or dynamic together (the 10-config
        # grid; the a_k-only dynamic diagnostic is out of scope here).
        self._a_dynamic = x_mode == "dynamic"
        self._c_dynamic = x_mode == "dynamic"

        # Per-head item-feature width (before concatenating the state).
        if coupling == "shared":
            rdim_a = rdim_c = emb_dim
        elif coupling in ("decoupled", "all_decoupled"):
            rdim_a = rdim_c = item_key_dim
        elif coupling == "a_only_dec":
            rdim_a, rdim_c = item_key_dim, emb_dim
        else:  # c_only_dec
            rdim_a, rdim_c = emb_dim, item_key_dim

        # Wide-key tables (decoder-owned).  Only what each coupling needs.
        if coupling == "decoupled":
            self.shared_key_emb = nn.Embedding(num_items, item_key_dim)
        elif coupling == "all_decoupled":
            self.a_key_emb = nn.Embedding(num_items, item_key_dim)
            self.c_key_emb = nn.Embedding(num_items, item_key_dim)
        elif coupling == "a_only_dec":
            self.a_key_emb = nn.Embedding(num_items, item_key_dim)
        elif coupling == "c_only_dec":
            self.c_key_emb = nn.Embedding(num_items, item_key_dim)

        a_in = (hidden_dim + rdim_a) if self._a_dynamic else rdim_a
        c_in = (hidden_dim + rdim_c) if self._c_dynamic else rdim_c
        self.fc_a = nn.Linear(a_in, n_options)
        self.fc_c = nn.Linear(c_in, n_options)

        # Small-scale init for dynamic heads: a naive state-conditioned NRM slope
        # is unstable; damp early gradients (proven fix in the source probes).
        if self._a_dynamic:
            nn.init.normal_(self.fc_a.weight, std=0.01)
            nn.init.zeros_(self.fc_a.bias)
        if self._c_dynamic:
            nn.init.normal_(self.fc_c.weight, std=0.01)
            nn.init.zeros_(self.fc_c.bias)

    # --- item features per head ---

    def _item_features(
        self,
        item_val: torch.Tensor,   # (..., emb_dim) thin value
        item_ids: torch.Tensor,   # (...) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (feat_a, feat_c): the base item feature for each head."""
        if self.coupling == "shared":
            return item_val, item_val
        if self.coupling == "decoupled":
            key = self.shared_key_emb(item_ids)
            return key, key
        if self.coupling == "all_decoupled":
            return self.a_key_emb(item_ids), self.c_key_emb(item_ids)
        if self.coupling == "a_only_dec":
            return self.a_key_emb(item_ids), item_val
        return item_val, self.c_key_emb(item_ids)     # c_only_dec

    # --- forward ---

    def forward(
        self,
        theta: torch.Tensor,       # (B, T) or (B, T, 1)
        item_val: torch.Tensor,    # (B, T, emb_dim)
        item_ids: torch.Tensor,    # (B, T) long
        state: torch.Tensor | None = None,   # (B, T, hidden_dim)
    ) -> torch.Tensor:
        """Per-step NRM logits ``a_k * theta + c_k`` -> (B, T, K)."""
        feat_a, feat_c = self._item_features(item_val, item_ids)

        if self._a_dynamic:
            if state is None:
                raise RuntimeError("dynamic a_k head requires the aligned state.")
            a_in = torch.cat([state, feat_a], dim=-1)
        else:
            a_in = feat_a
        if self._c_dynamic:
            if state is None:
                raise RuntimeError("dynamic c_k head requires the aligned state.")
            c_in = torch.cat([state, feat_c], dim=-1)
        else:
            c_in = feat_c

        a_raw = self.fc_a(a_in)
        c_raw = self.fc_c(c_in)
        a = a_raw - a_raw.mean(dim=-1, keepdim=True)     # sum_k a_k = 0
        c = c_raw - c_raw.mean(dim=-1, keepdim=True)     # sum_k c_k = 0

        if theta.dim() == 2:
            theta = theta.unsqueeze(-1)                  # (B, T, 1)
        return a * theta + c                             # (B, T, K)

    @property
    def needs_state(self) -> bool:
        """True when either head is state-conditioned (i.e. x_mode == dynamic)."""
        return self._a_dynamic or self._c_dynamic

    # --- recovery: per-item a_k (Q, K) and c_k (Q, K) ---

    @torch.no_grad()
    def recover_item_params(
        self,
        encoder,                       # core LSTMEncoder (read-only)
        item_ids_all: torch.Tensor,    # (N, T) all training sequences
        responses_all: torch.Tensor,   # (N, T)
        device: torch.device,
    ) -> dict:
        """Recover per-item a_k and c_k, plus a ``seen`` mask.

        STATIC heads: direct item-id lookup (any subset gives the same answer);
        every item is marked seen.  DYNAMIC heads: occurrence-average the readout
        over all (learner, step) positions, mirroring the state_alpha recipe;
        ``seen`` marks items that actually appeared.
        """
        self.eval()
        encoder.eval()
        item_ids_all = item_ids_all.to(device)
        responses_all = responses_all.to(device)

        if not self.needs_state:
            all_ids = torch.arange(self.num_items, device=device)
            val = encoder.item_val_emb(all_ids)                    # (Q, emb_dim)
            feat_a, feat_c = self._item_features(val, all_ids)
            a_raw = self.fc_a(feat_a)
            c_raw = self.fc_c(feat_c)
            a = (a_raw - a_raw.mean(-1, keepdim=True)).cpu().numpy()
            c = (c_raw - c_raw.mean(-1, keepdim=True)).cpu().numpy()
            seen = np.ones(self.num_items, dtype=bool)
            return {"a": a, "c": c, "seen": seen}

        # Dynamic: occurrence-average over all positions.
        _, state_in = encoder.aligned_theta_and_state(item_ids_all, responses_all)
        item_val = encoder.item_val_emb(item_ids_all)              # (N, T, emb_dim)
        feat_a, feat_c = self._item_features(item_val, item_ids_all)

        a_inp = torch.cat([state_in, feat_a], dim=-1) if self._a_dynamic else feat_a
        c_inp = torch.cat([state_in, feat_c], dim=-1) if self._c_dynamic else feat_c
        a_raw = self.fc_a(a_inp)
        c_raw = self.fc_c(c_inp)
        a_np = (a_raw - a_raw.mean(-1, keepdim=True)).cpu().numpy()
        c_np = (c_raw - c_raw.mean(-1, keepdim=True)).cpu().numpy()

        items_np = item_ids_all.cpu().numpy()
        a_sum = np.zeros((self.num_items, self.n_options))
        c_sum = np.zeros((self.num_items, self.n_options))
        cnt = np.zeros(self.num_items)
        np.add.at(a_sum, items_np.ravel(), a_np.reshape(-1, self.n_options))
        np.add.at(c_sum, items_np.ravel(), c_np.reshape(-1, self.n_options))
        np.add.at(cnt, items_np.ravel(), 1.0)
        seen = cnt > 0
        denom = np.maximum(cnt, 1.0)[:, None]
        a_hat = a_sum / denom
        c_hat = c_sum / denom
        # Re-center defensively (averaging preserves sum-to-zero only approx).
        a_hat = a_hat - a_hat.mean(axis=1, keepdims=True)
        c_hat = c_hat - c_hat.mean(axis=1, keepdims=True)
        return {"a": a_hat, "c": c_hat, "seen": seen}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_nrm_channel_decoder(
    config: str | dict | tuple,
    num_items: int,
    emb_dim: int,
    n_options: int,
    hidden_dim: int,
    item_key_dim: int = 64,
    correct_option: int = 0,
) -> NRMChannelDecoder:
    """Build an :class:`NRMChannelDecoder` from a config name/dict/tuple.

    ``config`` may be ``"c_only_dec/dynamic"``, a ``NRM_CONFIGS`` dict, or a
    ``(coupling, x_mode)`` tuple.
    """
    if isinstance(config, str):
        coupling, x_mode = config.split("/")
    elif isinstance(config, dict):
        coupling, x_mode = config["coupling"], config["x_mode"]
    else:
        coupling, x_mode = config
    return NRMChannelDecoder(
        num_items=num_items, emb_dim=emb_dim, n_options=n_options,
        hidden_dim=hidden_dim, coupling=coupling, x_mode=x_mode,
        item_key_dim=item_key_dim, correct_option=correct_option,
    )
