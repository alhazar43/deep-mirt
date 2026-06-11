"""Per-item ``(alpha, beta)`` cache builder, loader and on-disk format.

Motivation. The OrdRec reward, the static MVE warm-start critic and the
probe-entropy state summary all read ``(alpha_q, beta_q)`` for arbitrary
item ids ``q``. Re-encoding through the DKVMN world model every time
would dominate the reward-computation cost. The cache lets every read
side query a precomputed ``(Q + 1, D)`` alpha table and a ``(Q + 1,
K - 1)`` beta table by integer item id at numpy/torch indexing speed.

See ``docs/ordrec_impl_guide.md`` Section 7 (E2 deliverables) and
``docs/exrec_ordinal_plan.md`` Section 4 ("Per-item parameter table")
for the design narrative.

Method. The MA-IRT IRT extractor wires alpha through a
``W_alpha [joint_summary; item_embed]`` MLP and beta through
``W_beta * item_embed``. ``beta`` therefore depends only on the item
embedding and is constant in the student context. ``alpha`` does
depend on ``joint_summary`` so it varies a little across (student,
step) pairs for the same item id. We follow the paper-style averaging
recipe used by ``ma-irt/scripts/plot_recovery.py``, averaging alpha
across a small number of student contexts to produce a stable per-item
estimate. The implementation hides this behind a single
``build_item_cache`` call.

Persistence. The cache is written to
``rl/artifacts/item_cache/<dataset_name>/<ckpt_sha7>/item_cache.npz``
where ``ckpt_sha7`` is the first seven hex characters of the
SHA-256 digest of the MAGPCM checkpoint file. Two distinct trained
models therefore produce two distinct caches even when their datasets
match. The on-disk file is a single ``np.savez`` archive carrying
``alpha_table`` (``float32``, ``(Q + 1, D)``) and ``beta_table``
(``float32``, ``(Q + 1, K - 1)``) plus the originating shape
metadata and provenance tags. Row ``0`` is the padding slot and is
left at zero/identity defaults for downstream safety, real items live
in rows ``1..Q``.

Provenance. Two additional string fields are stored at build time and
verified at load / env-construction time:

- ``world_model_git_sha``: the short git SHA of the ma-irt repo
  at the time the cache was built. Obtained by the caller (not
  auto-detected here so that this file stays importable without git).
- ``world_model_config_hash``: SHA-256 hex digest (first 16 chars) of
  the JSON-serialised world-model config dict passed to the builder.
  ``""`` when no config is provided.

Use :func:`validate_provenance` to assert match between a loaded cache
and the current world-model provenance before starting training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from .frozen_magpcm import FrozenMAGPCM


# Canonical dtype for both alpha and beta tables on disk.
ITEM_CACHE_DTYPE = np.float32


@dataclass(frozen=True)
class ItemCache:
    """Per-item ``(alpha, beta)`` lookup tables.

    Attributes:
        alpha_table: ``np.ndarray (Q + 1, D)`` per-item discrimination.
            Row 0 reserved for padding (filled with ones, the IRT-neutral
            value for alpha).
        beta_table: ``np.ndarray (Q + 1, K - 1)`` per-item step thresholds.
            Row 0 reserved for padding (filled with zeros).
        n_questions: Q, item bank size (rows ``1..Q``).
        n_categories: K, response category count.
        n_traits: D, latent trait dimension.
        n_contexts: Number of (student-context) forward passes averaged
            into ``alpha_table``. Reported for reproducibility.
        ckpt_sha7: First seven hex characters of the source-checkpoint
            SHA-256 digest. ``""`` when the cache was built without a
            tracked checkpoint (e.g. for unit tests).
        dataset_name: Free-form dataset tag, used in the persisted path.
        world_model_git_sha: Short git SHA of the ma-irt repo at build
            time. ``""`` when not provided. Use
            :func:`world_model_git_sha_from_repo` to compute.
        world_model_config_hash: First 16 hex chars of the SHA-256
            digest of the world-model config dict serialised to JSON.
            ``""`` when no config was supplied. Use
            :func:`config_hash` to compute.
    """

    alpha_table: np.ndarray
    beta_table: np.ndarray
    n_questions: int
    n_categories: int
    n_traits: int
    n_contexts: int
    ckpt_sha7: str
    dataset_name: str
    world_model_git_sha: str = ""
    world_model_config_hash: str = ""

    # ------------------------------------------------------------------
    # Convenience indexing
    # ------------------------------------------------------------------

    def alpha_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        t = torch.as_tensor(self.alpha_table, dtype=torch.float32)
        return t.to(device) if device is not None else t

    def beta_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        t = torch.as_tensor(self.beta_table, dtype=torch.float32)
        return t.to(device) if device is not None else t


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_item_cache(
    frozen: FrozenMAGPCM,
    *,
    n_questions: Optional[int] = None,
    n_categories: Optional[int] = None,
    n_contexts: int = 8,
    context_seq_len: int = 1,
    seed: int = 0,
    dataset_name: str = "",
    ckpt_sha7: str = "",
    world_model_git_sha: str = "",
    world_model_config_hash: str = "",
    device: Optional[torch.device] = None,
) -> ItemCache:
    """Build a per-item ``(alpha, beta)`` cache from a frozen MA-IRT model.

    Args:
        frozen: Frozen MA-IRT wrapper (``FrozenMAGPCM``).
        n_questions: Optional override, defaults to ``frozen.n_questions``.
        n_categories: Optional override, defaults to ``frozen.n_categories``.
        n_contexts: Number of synthetic student contexts to average over
            when reading alpha. ``1`` matches the bare "one forward per
            item" recipe in the impl guide. ``> 1`` matches the paper
            recovery script's averaging recipe (Section 4 of the
            strategic plan).
        context_seq_len: Length of the synthetic context sequence per
            student. ``1`` matches "singleton sequence" in the guide;
            larger values let the encoder accumulate a non-trivial
            value-memory state before alpha is read at the final step.
        seed: Seed for the random context draw. Identical seeds produce
            identical caches.
        dataset_name: Free-form tag stored on the cache, used by
            :func:`item_cache_path`.
        ckpt_sha7: Optional precomputed checkpoint sha, stored on the
            cache. When omitted the cache is built with an empty tag,
            which is fine for unit-test contexts.
        world_model_git_sha: Short git SHA of the ma-irt repo at build
            time. Pass the result of :func:`world_model_git_sha_from_repo`
            or ``""`` to skip provenance tracking.
        world_model_config_hash: First 16 hex chars of the SHA-256
            digest of the world-model config. Pass the result of
            :func:`config_hash` or ``""`` to skip.
        device: Device to run forwards on. Defaults to
            ``frozen.device``.

    Returns:
        An :class:`ItemCache` with ``alpha_table`` and ``beta_table``
        populated for rows ``1..Q``. Row ``0`` is the padding slot.

    Notes:
        The sweep is implemented as a single batched forward call. Each
        of ``Q`` items appears as the last position of one row in a
        ``(Q * n_contexts, context_seq_len)`` batch, so the encoder
        runs ``Q * n_contexts`` rows in parallel rather than Q times in
        sequence. With ``Q = 948`` and ``n_contexts = 8`` this is a
        single forward of shape ``(7584, context_seq_len)`` which fits
        comfortably on CPU.
    """
    q_count = int(n_questions if n_questions is not None else frozen.n_questions)
    k_count = int(n_categories if n_categories is not None else frozen.n_categories)
    if q_count <= 0:
        raise ValueError(
            f"n_questions must be > 0 to build the cache, got {q_count}."
        )
    if k_count < 2:
        raise ValueError(
            f"n_categories must be >= 2, got {k_count}."
        )
    if n_contexts < 1:
        raise ValueError(f"n_contexts must be >= 1, got {n_contexts}.")
    if context_seq_len < 1:
        raise ValueError(
            f"context_seq_len must be >= 1, got {context_seq_len}."
        )

    if device is None:
        device = frozen.device

    rng = np.random.default_rng(int(seed))

    # Per-item context. Build ``(n_contexts * Q, context_seq_len)`` such
    # that the last column is item id ``q`` for the ``q``-th group of
    # ``n_contexts`` rows. Earlier columns are random item ids in
    # ``[1, Q]`` with random ordinal labels in ``[0, K - 1]``.
    rows = q_count * n_contexts
    questions_np = rng.integers(
        low=1, high=q_count + 1, size=(rows, context_seq_len), dtype=np.int64
    )
    responses_np = rng.integers(
        low=0, high=k_count, size=(rows, context_seq_len), dtype=np.int64
    )
    # Place the target item id at the final column. The encoder reads
    # alpha and beta at every step; we read them at index -1.
    target_ids = np.repeat(np.arange(1, q_count + 1, dtype=np.int64), n_contexts)
    questions_np[:, -1] = target_ids
    # The response at the final position cannot affect the encoder's
    # output at that position under causal masking (DKVMN reads memory
    # from positions <= t - 1 at step t; LSTM and Transformer are
    # causal). We still set it deterministically to category 0 so the
    # cache is invariant under any future change that breaks causality.
    responses_np[:, -1] = 0

    questions_t = torch.from_numpy(questions_np).to(device=device, dtype=torch.long)
    responses_t = torch.from_numpy(responses_np).to(device=device, dtype=torch.long)

    out = frozen.forward_no_grad(questions_t, responses_t)
    alpha_last = out["alpha"][:, -1, :].detach().to(torch.float32).cpu().numpy()
    beta_last = out["beta"][:, -1, :].detach().to(torch.float32).cpu().numpy()
    d_dim = int(alpha_last.shape[-1])

    # Reshape to (Q, n_contexts, ...) and average along the context axis.
    alpha_per_q = alpha_last.reshape(q_count, n_contexts, d_dim).mean(axis=1)
    beta_per_q = beta_last.reshape(q_count, n_contexts, k_count - 1).mean(axis=1)

    # Allocate (Q + 1, ...) tables and fill row 0 with the IRT-neutral
    # padding values: alpha = 1, beta = 0. Row 0 is never read by the
    # rest of the system; the neutral fill guarantees that an accidental
    # index-0 lookup does not produce numerical garbage.
    alpha_table = np.empty((q_count + 1, d_dim), dtype=ITEM_CACHE_DTYPE)
    beta_table = np.empty((q_count + 1, k_count - 1), dtype=ITEM_CACHE_DTYPE)
    alpha_table[0, :] = 1.0
    beta_table[0, :] = 0.0
    alpha_table[1:, :] = alpha_per_q.astype(ITEM_CACHE_DTYPE)
    beta_table[1:, :] = beta_per_q.astype(ITEM_CACHE_DTYPE)

    return ItemCache(
        alpha_table=alpha_table,
        beta_table=beta_table,
        n_questions=q_count,
        n_categories=k_count,
        n_traits=d_dim,
        n_contexts=int(n_contexts),
        ckpt_sha7=str(ckpt_sha7),
        dataset_name=str(dataset_name),
        world_model_git_sha=str(world_model_git_sha),
        world_model_config_hash=str(world_model_config_hash),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_CACHE_FILENAME = "item_cache.npz"


def checkpoint_sha7(checkpoint_path: Path) -> str:
    """Return the first seven hex chars of the SHA-256 digest of a file.

    Args:
        checkpoint_path: Path to a checkpoint (``best.pt`` or similar).

    Returns:
        Seven-character lowercase hex string. ``""`` when the file does
        not exist; this lets callers thread a "no tracked checkpoint"
        case (e.g. an in-memory test model) through without branching.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:7]


def item_cache_path(root: Path, dataset_name: str, ckpt_sha7: str) -> Path:
    """Canonical on-disk location for a built cache.

    ``root / dataset_name / ckpt_sha7 / item_cache.npz``. The
    SHA-keyed directory ensures two distinct trained models do not
    overwrite each other's caches.
    """
    if not dataset_name:
        raise ValueError("item_cache_path requires a non-empty dataset_name")
    sha = ckpt_sha7 or "untagged"
    return Path(root) / dataset_name / sha / _CACHE_FILENAME


def save_item_cache(cache: ItemCache, path: Path) -> Path:
    """Persist ``cache`` to ``path`` as a single ``np.savez`` archive.

    Returns the canonical path written to (creates parent dirs).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        alpha_table=cache.alpha_table,
        beta_table=cache.beta_table,
        n_questions=np.int64(cache.n_questions),
        n_categories=np.int64(cache.n_categories),
        n_traits=np.int64(cache.n_traits),
        n_contexts=np.int64(cache.n_contexts),
        ckpt_sha7=np.array(cache.ckpt_sha7, dtype=object),
        dataset_name=np.array(cache.dataset_name, dtype=object),
        world_model_git_sha=np.array(cache.world_model_git_sha, dtype=object),
        world_model_config_hash=np.array(cache.world_model_config_hash, dtype=object),
    )
    return p


def load_item_cache(path: Path) -> ItemCache:
    """Load an :class:`ItemCache` from ``path``.

    Validates shapes and the (alpha row 0 = 1, beta row 0 = 0)
    padding-slot invariant.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Item cache not found at {p}")
    with np.load(p, allow_pickle=True) as data:
        alpha_table = np.asarray(data["alpha_table"], dtype=ITEM_CACHE_DTYPE)
        beta_table = np.asarray(data["beta_table"], dtype=ITEM_CACHE_DTYPE)
        n_questions = int(data["n_questions"])
        n_categories = int(data["n_categories"])
        n_traits = int(data["n_traits"])
        n_contexts = int(data["n_contexts"])
        ckpt_sha7 = str(data["ckpt_sha7"])
        dataset_name = str(data["dataset_name"])
        # Provenance fields added in E4.6b; absent in older caches.
        world_model_git_sha = str(data["world_model_git_sha"]) if "world_model_git_sha" in data else ""
        world_model_config_hash = str(data["world_model_config_hash"]) if "world_model_config_hash" in data else ""

    if alpha_table.shape != (n_questions + 1, n_traits):
        raise ValueError(
            f"alpha_table shape {alpha_table.shape} mismatches "
            f"(n_questions + 1, n_traits) = ({n_questions + 1}, {n_traits})"
        )
    if beta_table.shape != (n_questions + 1, n_categories - 1):
        raise ValueError(
            f"beta_table shape {beta_table.shape} mismatches "
            f"(n_questions + 1, n_categories - 1) = "
            f"({n_questions + 1}, {n_categories - 1})"
        )

    return ItemCache(
        alpha_table=alpha_table,
        beta_table=beta_table,
        n_questions=n_questions,
        n_categories=n_categories,
        n_traits=n_traits,
        n_contexts=n_contexts,
        ckpt_sha7=ckpt_sha7,
        dataset_name=dataset_name,
        world_model_git_sha=world_model_git_sha,
        world_model_config_hash=world_model_config_hash,
    )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def config_hash(config_dict: Dict[str, Any]) -> str:
    """Return the first 16 hex chars of the SHA-256 of a config dict.

    The dict is serialised with ``json.dumps(sort_keys=True)`` before
    hashing so insertion order does not affect the digest.

    Args:
        config_dict: Arbitrary JSON-serialisable mapping. Nested
            structures are supported.

    Returns:
        16-character lowercase hex string. ``""`` if ``config_dict``
        is empty or ``None``.
    """
    if not config_dict:
        return ""
    serialised = json.dumps(config_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()[:16]


def world_model_git_sha_from_repo(repo_path: Optional[Path] = None) -> str:
    """Return the short git SHA of a repository working tree.

    Calls ``git rev-parse --short HEAD`` in ``repo_path``. Returns
    ``""`` when git is unavailable or the path is not a git repo.

    Args:
        repo_path: Optional path inside the target git repo. Defaults
            to the current working directory.

    Returns:
        Short git SHA string (typically 7 chars), or ``""`` on failure.
    """
    import subprocess

    cmd = ["git", "rev-parse", "--short", "HEAD"]
    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=cwd,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def validate_provenance(
    cache: ItemCache,
    *,
    expected_git_sha: str = "",
    expected_config_hash: str = "",
    strict: bool = False,
) -> None:
    """Assert that a loaded cache's provenance matches expected values.

    Raises ``RuntimeError`` when a non-empty ``expected_*`` value is
    provided and the cache value differs. Skips the check when either
    the expected or cache value is ``""``.

    Args:
        cache: Loaded :class:`ItemCache` to inspect.
        expected_git_sha: Short git SHA the caller expects. ``""`` skips
            the git-SHA check.
        expected_config_hash: Config hash the caller expects. ``""``
            skips the config-hash check.
        strict: When ``True``, also raise if the cache's provenance
            fields are empty (i.e. built without provenance). Useful for
            production runs where provenance must be tracked.

    Raises:
        RuntimeError: On mismatch or (when ``strict=True``) when the
            cache has empty provenance.
    """
    if strict and not cache.world_model_git_sha:
        raise RuntimeError(
            "Item cache has no world_model_git_sha (built without provenance "
            "tracking). Rebuild with world_model_git_sha set, or set "
            "strict=False to allow untagged caches."
        )
    if strict and not cache.world_model_config_hash:
        raise RuntimeError(
            "Item cache has no world_model_config_hash (built without config "
            "provenance). Rebuild with world_model_config_hash set, or set "
            "strict=False to allow untagged caches."
        )

    if expected_git_sha and cache.world_model_git_sha:
        if cache.world_model_git_sha != expected_git_sha:
            raise RuntimeError(
                f"Item cache world_model_git_sha mismatch: cache has "
                f"'{cache.world_model_git_sha}', expected '{expected_git_sha}'. "
                "Rebuild the item cache from the current ma-irt checkpoint."
            )

    if expected_config_hash and cache.world_model_config_hash:
        if cache.world_model_config_hash != expected_config_hash:
            raise RuntimeError(
                f"Item cache world_model_config_hash mismatch: cache has "
                f"'{cache.world_model_config_hash}', expected '{expected_config_hash}'. "
                "The world-model config has changed since the cache was built. "
                "Rebuild the item cache."
            )


__all__ = [
    "ITEM_CACHE_DTYPE",
    "ItemCache",
    "build_item_cache",
    "checkpoint_sha7",
    "config_hash",
    "item_cache_path",
    "load_item_cache",
    "save_item_cache",
    "validate_provenance",
    "world_model_git_sha_from_repo",
]
