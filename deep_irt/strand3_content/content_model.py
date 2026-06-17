"""content_model.py -- compose the frozen deep_irt with a content channel.

A thin wrapper that builds a stock ``DeepIRTModel`` (organic single-pass
engine with the state-conditioned alpha feature, GPCM decoder) and swaps the
encoder's ID ``item_val_emb`` for a ``ContentItemEmbedding`` (the content channel).
Because the content channel is a drop-in for ``nn.Embedding``, ``fit`` /
``track`` / the decoder readouts work unchanged: training optimises the LSTM,
the decoder, and the content projection jointly.

What this wrapper adds on top of ``DeepIRTModel``:

  recover_difficulty(item_ids, responses) -- per-BASE-item difficulty
      (mean step threshold) read through the trained decoder, the warm/in-bank
      number.

  cold_start_difficulty(new_text_features) -- per-NEW-item difficulty from
      TEXT ALONE: project the frozen text embedding, push it through the frozen
      decoder's ``fc_b``, collapse the sorted step thresholds to a scalar.  No
      responses, no trained row.  This is the cold-start lever under test.

Difficulty convention
---------------------
Following the deep_irt's GPCM, item difficulty is the mean of the (sorted)
K-1 step thresholds ``b``.  Higher mean-b -> harder item.  This is the same
scalar location ``slam_extend`` uses for its anchored-extension difficulty.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from deep_irt.core.model import DeepIRTModel
from deep_irt.core.decoders import GPCMDecoder, Binary2PLDecoder

from deep_irt.strand3_content.content_channel import ContentItemEmbedding


class ContentModel:
    """DeepIRTModel with the encoder's item table driven by item text.

    Parameters
    ----------
    text_features : (num_items, text_dim) float -- frozen ST embeddings per item.
    emb_dim, hidden_dim, n_cats : deep_irt sizes.
    mode : "content" | "concat" (see ContentItemEmbedding).
    state_alpha : route discrimination through the organic state-conditioned
        alpha feature (default True; the unified single-pass engine).  Difficulty
        readouts (the cold-start lever) use beta only, so this flag does not
        change the cold-start path; it selects the engine the channel rides on.
    device, seed : passthrough.
    """

    def __init__(
        self,
        text_features: torch.Tensor,
        emb_dim: int = 16,
        hidden_dim: int = 64,
        n_cats: int = 3,
        mode: str = "content",
        state_alpha: bool = True,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
    ) -> None:
        num_items = text_features.size(0)
        self.device = device
        self.seed = seed
        self.n_cats = n_cats
        self.emb_dim = emb_dim
        self.state_alpha = state_alpha

        # Build the stock model FIRST (so its seeded init is unchanged), then
        # swap the ID embedding for the content channel.  The content
        # projection is seeded separately so different modes are comparable.
        self.model = DeepIRTModel(
            num_items=num_items,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_cats=n_cats,
            decoder="gpcm",
            state_alpha=state_alpha,
            device=device,
            seed=seed,
        )

        torch.manual_seed(seed + 7)
        self.content_emb = ContentItemEmbedding(
            text_features=text_features, emb_dim=emb_dim, mode=mode
        ).to(device)
        # Drop in: every readout in encoder/decoder that calls item_val_emb(...)
        # now routes through the content channel.  The freed ID nn.Embedding is
        # no longer referenced.
        self.model.encoder.item_val_emb = self.content_emb

    # ------------------------------------------------------------------
    # Fit (delegates to the deep_irt; content projection is in the graph)
    # ------------------------------------------------------------------

    def fit(self, item_ids, responses, **kwargs) -> dict:
        return self.model.fit(item_ids, responses, **kwargs)

    def track(self, item_ids, responses) -> torch.Tensor:
        return self.model.track(item_ids, responses)

    # ------------------------------------------------------------------
    # Difficulty readouts
    # ------------------------------------------------------------------

    def _decoder(self):
        dec = self.model.decoder
        assert isinstance(dec, (GPCMDecoder, Binary2PLDecoder))
        return dec

    @staticmethod
    def _scalar_difficulty(b_sorted: torch.Tensor) -> np.ndarray:
        """(M, K-1) sorted step thresholds -> (M,) mean-b scalar difficulty."""
        return b_sorted.mean(dim=-1).detach().cpu().numpy()

    def recover_difficulty(self, item_ids: Optional[torch.Tensor] = None) -> np.ndarray:
        """Per-BASE-item difficulty (mean sorted step threshold) -- the in-bank
        number.  Beta is an item-only read (no state needed), so this is valid
        in state_alpha mode too.
        """
        enc = self.model.encoder
        dec = self._decoder()
        if item_ids is None:
            item_ids = torch.arange(self.content_emb.num_embeddings, device=self.device)
        item_ids = item_ids.to(self.device)
        enc.eval()
        dec.eval()
        with torch.no_grad():
            embs = enc.item_val_emb(item_ids)                # content reprs
            b = dec.item_params_sorted(embs)["b"]            # (M, K-1)
        return self._scalar_difficulty(b)

    def cold_start_difficulty(self, new_text_features: torch.Tensor) -> np.ndarray:
        """Per-NEW-item difficulty from TEXT ALONE.

        Project the frozen text embedding of each new item, push it through the
        frozen decoder's step-threshold head, sort, collapse to mean-b.  No
        responses, no trained ID row -- the cold-start lever.
        """
        dec = self._decoder()
        dec.eval()
        with torch.no_grad():
            rep = self.content_emb.cold_start_repr(new_text_features)  # (n_new, emb_dim)
            b = dec.item_params_sorted(rep)["b"]                       # (n_new, K-1)
        return self._scalar_difficulty(b)
