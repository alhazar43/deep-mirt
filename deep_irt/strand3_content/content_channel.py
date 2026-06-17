"""content_channel.py -- item representation from item TEXT (frozen embedding).

The deep_irt reads every item parameter (alpha, beta) from the encoder's item
representation table.  In the stock engine that table is a free per-item ID
``nn.Embedding``: each item gets its own learnable row, so a brand-new item has
NO representation until a row is fit against its responses.  That is the
ID-only model's trap -- it is welded to one item bank.

The CONTENT CHANNEL replaces (or augments) the ID row with a representation
computed from the item's TEXT:

    item_repr(i) = proj( x_i )                       (mode="content")
    item_repr(i) = proj( x_i ) + id_emb(i)           (mode="concat")

where ``x_i`` is a FROZEN sentence-transformer embedding of item i's text and
``proj`` is a small learned linear map into the ``emb_dim`` item-embedding
space.  The decoder's readouts (``fc_a``, ``fc_b``) are unchanged; they still
consume an ``emb_dim`` vector.

The load-bearing property -- COLD-START
---------------------------------------
Because ``item_repr`` is a function of frozen text, a NEW item's representation
is ``proj(text_embed(new_text))``: computable from text alone, with no responses
and no trained row.  Its difficulty is then ``fc_b(item_repr(new))`` straight
through the frozen decoder.  ``cold_start_repr`` performs exactly this.

Drop-in for ``nn.Embedding``
----------------------------
``ContentItemEmbedding`` is callable as ``module(item_ids) -> (..., emb_dim)``
and exposes a ``.weight`` property holding the full materialised table, so it
can be assigned to ``encoder.item_val_emb`` and every existing readout in the
core (LSTM input, ``fc_a``, ``fc_b``, the recovery paths) works unchanged.  The
core is never modified; the swap lives entirely in this package.

Modes
-----
mode="content"  -- representation is the projected text embedding only.  This is
                   the pure content channel: items are placed by MEANING.  The
                   ID-only trap is gone (cold-start works).
mode="concat"   -- projected text PLUS a free ID row (residual).  The ID row can
                   absorb item idiosyncrasy the text misses, but a new item's
                   ID row is zero at cold-start, so cold-start falls back to the
                   text term alone.  This is the "content + lightweight ID"
                   variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContentItemEmbedding(nn.Module):
    """Item representation from a frozen text embedding + learned projection.

    Drop-in replacement for the encoder's ``item_val_emb`` (an ``nn.Embedding``):
    callable as ``module(item_ids) -> (..., emb_dim)`` and exposing ``.weight``
    (the full ``(num_items, emb_dim)`` table) and ``.num_embeddings`` so the
    rest of the deep_irt is untouched.

    Parameters
    ----------
    text_features : (num_items, text_dim) float
        FROZEN sentence-transformer embeddings, one row per base item.  Stored
        as a buffer (not a parameter); never updated.
    emb_dim : int
        Target item-embedding dimension consumed by the decoder readouts.
    mode : {"content", "concat"}
        "content" -> ``proj(x_i)`` only.  "concat" -> ``proj(x_i) + id_emb(i)``.
    bias : bool
        Whether the projection has a bias term (default True).
    """

    def __init__(
        self,
        text_features: torch.Tensor,
        emb_dim: int,
        mode: str = "content",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("content", "concat"):
            raise ValueError(f"mode must be 'content' or 'concat', got {mode!r}")
        if text_features.dim() != 2:
            raise ValueError(
                f"text_features must be (num_items, text_dim); got shape "
                f"{tuple(text_features.shape)}"
            )
        self.mode = mode
        self.emb_dim = emb_dim
        self.text_dim = text_features.size(1)
        self.num_embeddings = text_features.size(0)

        # Frozen text features: a buffer, moves with .to(device) but never trains.
        self.register_buffer("text_features", text_features.float())

        # Learned projection text_dim -> emb_dim (the ONLY content-side params
        # besides the optional ID residual).
        self.proj = nn.Linear(self.text_dim, emb_dim, bias=bias)

        # Optional free ID residual (concat mode).  Initialised to zero so a
        # fresh model starts as pure content and a never-seen item's residual is
        # zero -> cold-start falls back to the text term.
        if mode == "concat":
            self.id_emb = nn.Embedding(self.num_embeddings, emb_dim)
            nn.init.zeros_(self.id_emb.weight)
        else:
            self.id_emb = None

    # ------------------------------------------------------------------
    # nn.Embedding-compatible call
    # ------------------------------------------------------------------

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """item_ids: (...) long in [0, num_items) -> (..., emb_dim) float."""
        x = self.text_features[item_ids]          # (..., text_dim)
        rep = self.proj(x)                        # (..., emb_dim)
        if self.id_emb is not None:
            rep = rep + self.id_emb(item_ids)
        return rep

    # ------------------------------------------------------------------
    # Full materialised table (so code that reads `.weight` still works)
    # ------------------------------------------------------------------

    @property
    def weight(self) -> torch.Tensor:
        """The full ``(num_items, emb_dim)`` representation table.

        Recomputed on access from the frozen features and current projection so
        it always reflects the live parameters.  Read-only; assigning to it is
        not supported (the content channel has no free per-item rows to set in
        ``content`` mode).
        """
        ids = torch.arange(self.num_embeddings, device=self.text_features.device)
        return self.forward(ids)

    # ------------------------------------------------------------------
    # Cold-start: representation of NEW items from text alone
    # ------------------------------------------------------------------

    def cold_start_repr(self, new_text_features: torch.Tensor) -> torch.Tensor:
        """Item representation for NEW items, from their text features alone.

        This is the content channel's whole point: no responses, no trained ID
        row.  In ``concat`` mode the ID residual of an unseen item is undefined
        and treated as zero (the residual is the part content cannot supply), so
        cold-start uses the projected text term only -- exactly the information
        available before the item is ever answered.

        Parameters
        ----------
        new_text_features : (n_new, text_dim) float -- frozen ST embeddings of
            the new items' text.

        Returns
        -------
        rep : (n_new, emb_dim) -- representations usable directly by the frozen
            decoder readouts (``fc_a`` / ``fc_b``).
        """
        x = new_text_features.to(self.proj.weight.device).float()
        return self.proj(x)
