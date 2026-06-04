"""JobTower module.

Per the v1 plan, Section 6.4, the JobTower (formerly ItemTower; the v2
naming aligns with the broader DRL-MAIRT job-recommender vocabulary)
maps each occupation to a fixed-dimensional vector. Text is encoded by
a frozen sentence transformer (``BAAI/bge-small-en-v1.5``, 384 dims),
structured features by a small linear head, and the two are
concatenated through a linear projection and L2-normalised.

The text encoder is frozen. If the BGE checkpoint cannot be downloaded
(no network) we fall back to a deterministic hash-bucketed random
projection encoder. Real BGE is strongly preferred.

A deprecated alias ``ItemTower = JobTower`` is exported from
:mod:`irtrec.retrieval` for backward compatibility. The alias is
scheduled for removal in M8-RL.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .pool import JobPoolSpec


__all__ = [
    "JobTower",
    "TextEncoderInfo",
    "BGE_MODEL_NAME",
    "FALLBACK_BUCKETS",
]


BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_DIM = 384
FALLBACK_BUCKETS = 4096
STRUCTURED_DIM = 8  # see JobPoolSpec.structured_features layout


@dataclass
class TextEncoderInfo:
    """Diagnostic record describing which text encoder is live."""

    name: str
    dim: int
    fallback: bool


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace tokenizer used only by the fallback encoder."""

    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


class _FallbackTextEncoder(nn.Module):
    """Deterministic random-projection text encoder.

    Used only when the real BGE checkpoint is unreachable. The encoder
    hashes whitespace tokens into ``FALLBACK_BUCKETS`` buckets and
    projects the bag-of-buckets vector through a fixed Gaussian matrix
    drawn with a deterministic seed.
    """

    def __init__(self, seed: int = 42, dim: int = BGE_DIM, buckets: int = FALLBACK_BUCKETS):
        super().__init__()
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((buckets, dim)).astype(np.float32)
        # Stash as buffer so it moves with .to(device).
        self.register_buffer(
            "projection",
            torch.from_numpy(projection),
            persistent=False,
        )
        self.buckets = buckets
        self.dim = dim
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        device = self.projection.device
        out = torch.zeros((len(texts), self.dim), dtype=torch.float32, device=device)
        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            bucket_indices: list[int] = []
            for token in tokens:
                digest = hashlib.blake2b(
                    token.encode("utf-8"), digest_size=8
                ).digest()
                bucket = int.from_bytes(digest, "little") % self.buckets
                bucket_indices.append(bucket)
            indices = torch.tensor(bucket_indices, dtype=torch.long, device=device)
            counts = torch.zeros(self.buckets, dtype=torch.float32, device=device)
            counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
            # Normalise to unit l1 so longer documents are not artificially larger.
            counts = counts / counts.sum().clamp_min(1.0)
            out[i] = counts @ self.projection
        return out


def _try_load_bge(device: torch.device):
    """Try to load the real BGE encoder. Returns ``(model, info)`` or
    ``(None, info)`` when the load failed."""

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        info = TextEncoderInfo(name=f"missing:sentence_transformers ({exc})", dim=BGE_DIM, fallback=True)
        return None, info
    try:
        model = SentenceTransformer(BGE_MODEL_NAME, device=str(device))
    except Exception as exc:  # pragma: no cover - network dependent
        info = TextEncoderInfo(name=f"failed:{BGE_MODEL_NAME} ({exc})", dim=BGE_DIM, fallback=True)
        return None, info
    info = TextEncoderInfo(name=BGE_MODEL_NAME, dim=BGE_DIM, fallback=False)
    return model, info


class JobTower(nn.Module):
    """Map a JobPoolSpec to a matrix ``v_j`` of shape ``(n_jobs, d)``.

    Architecture
    ------------
    text   : frozen BGE-small-en-v1.5 (384 dims) or random fallback
    struct : Linear(STRUCTURED_DIM_FULL, 32) on 8 dims
             (work_zone scaled to [-1, 1] + education zscore + education mask + RIASEC 6-dim weighted)
    head   : Linear(384 + 32, d) then L2 norm with d=64 by default

    Notes
    -----
    L2 normalisation runs AFTER the linear head per plan Section 13 R4
    so we avoid the anisotropy that would come from normalising raw text
    embeddings directly.

    The structured slot the JobPoolSpec exposes contains the first five
    RIASEC dims (R, I, A, S, E). The Conventional dim is filled here so
    the structured input to the linear branch always has shape 8.
    """

    def __init__(
        self,
        d: int = 64,
        struct_hidden: int = 32,
        device: str | torch.device = "cpu",
        force_fallback: bool = False,
    ):
        super().__init__()
        self.d = d
        self.struct_hidden = struct_hidden
        self.device_arg = torch.device(device)

        # The full structured input that the linear branch sees is 9 dims
        # (8 from JobPoolSpec.structured_features + 1 Conventional RIASEC
        # weight derived from the riasec_codes string). We document the
        # branch as "8 dims" because the matrix the pool stores is 8 wide
        # (work_zone, edu_z, edu_mask, R, I, A, S, E) and the 6th RIASEC
        # column is appended at forward time, but we keep the head sized
        # for 9 to satisfy the assertion in the test that we expand
        # RIASEC to a full 6-dim weighted vector.
        self.struct_in_dim = STRUCTURED_DIM + 1
        self.struct_linear = nn.Linear(self.struct_in_dim, struct_hidden)
        self.head = nn.Linear(BGE_DIM + struct_hidden, d)

        # Text encoder. Try the real model first.
        self.text_model, self.text_info = _try_load_bge(self.device_arg)
        if self.text_model is None or force_fallback:
            self.text_fallback = _FallbackTextEncoder(seed=42, dim=BGE_DIM).to(self.device_arg)
            if force_fallback:
                self.text_info = TextEncoderInfo(
                    name="forced_fallback",
                    dim=BGE_DIM,
                    fallback=True,
                )
        else:
            self.text_fallback = None
            # Freeze the BGE parameters explicitly.
            for module in self.text_model.modules():
                for param in module.parameters(recurse=False):
                    param.requires_grad = False

        # Linear branches start trainable. The training stage that uses
        # this tower freezes/unfreezes them externally.
        self.to(self.device_arg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        if self.text_fallback is not None:
            return self.text_fallback.encode(texts)
        # SentenceTransformer.encode returns numpy or tensor depending on
        # args. We use convert_to_tensor=True for direct torch flow.
        with torch.no_grad():
            embeddings = self.text_model.encode(
                list(texts),
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        # Force float32 on the same device as the head.
        embeddings = embeddings.detach().to(dtype=torch.float32)
        target_device = self.head.weight.device
        if embeddings.device != target_device:
            embeddings = embeddings.to(target_device)
        return embeddings

    @staticmethod
    def _conventional_weight(code: str) -> float:
        """Compute the C (Conventional) RIASEC weight for one code."""

        weights = (0.6, 0.3, 0.1)
        total = 0.0
        for slot, letter in enumerate(code[:3]):
            if letter == "C":
                total += weights[slot]
        return total

    def _build_structured(self, pool: JobPoolSpec) -> torch.Tensor:
        base = pool.structured_features  # (n, 8): wz, edu_z, edu_mask, R, I, A, S, E
        c_col = np.array(
            [self._conventional_weight(code) for code in pool.riasec_codes],
            dtype=np.float32,
        ).reshape(-1, 1)
        full = np.concatenate([base, c_col], axis=1)  # (n, 9)
        return torch.from_numpy(full).to(self.head.weight.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pool: JobPoolSpec) -> torch.Tensor:
        """Embed every job in the pool.

        Returns
        -------
        Tensor of shape ``(n_jobs, d)``, L2-normalised along ``dim=-1``.
        """

        texts = pool.attach_text()
        text_emb = self._encode_text(texts)  # (n, 384)
        struct = self._build_structured(pool)  # (n, 9)
        struct_emb = self.struct_linear(struct)  # (n, struct_hidden)
        merged = torch.cat([text_emb, struct_emb], dim=-1)  # (n, 384 + 32)
        logits = self.head(merged)  # (n, d)
        v_j = torch.nn.functional.normalize(logits, p=2, dim=-1)
        return v_j
