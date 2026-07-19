"""A4 Q-matrix module: the item-to-KC expansion policy, pure-anchor
identification, and the circularity guard (`_planning/design/a4_design.md`
v1.1, sections 2.5 and 6; `_planning/research/avenue_map.md` line 258's
"ASSISTments lesson").

This module owns the STATIC item-bank side of the real-bed bridge: given,
per item (KDD: step; EdNet: question), the set of KC tags it carries, it
builds the ragged ``kc_ids``/``kc_mask`` buffer section 2.5 specifies
(shape ``(n_items, A_max)``, ``-1``-padded, per-slot mask) and derives the
pure-anchor statistics every per-KC attribution claim in the design is
gated on ("Per-KC attribution claims only on KCs with >= 3 pure steps",
section 6's KDD row). It does not read any raw file itself -- the loader
(`kc_data.py`) parses the bed and hands this module already-interned item
keys and per-item KC-id lists.

Interpretation notes:

1. **One expansion policy in v1: "all tags."** Section 6 states, for both
   real beds, "the interaction enters every tagged KC's slice and
   increments each tagged KC's opportunity counter" -- there is no
   pre-registered alternative (e.g. "primary tag only") in this design, so
   ``ExpansionPolicy`` is a named marker (for reporting and for a future
   policy this design does not specify) rather than a dispatch table with
   more than one live branch.
2. **Pure-anchor gate.** "Pure" means arity exactly 1 (design section 6:
   "Per-KC attribution claims only on KCs with >= 3 pure steps"); the
   ``min_pure`` floor (3) matches both the design's own KDD clause and
   `triage_common.py`'s ``MIN_PURE_ANCHORS`` -- the same gate, computed
   here from the ragged buffer instead of a ``dict[item -> frozenset]``,
   so `qmatrix.pure_anchor_stats` reproduces
   `_planning/triage/kdd_algebra_2008_2009_stats.json`'s
   ``f_g_pure_anchors_and_arity`` block field-for-field on the same input
   (verified in `_planning/design/realbed_bridge_notes.md`'s acceptance
   table).
3. **The circularity guard is a heuristic, not a proof.** The ASSISTments
   failure this guards against is concrete and specific: a skill/KC id
   reused AS the model's item key, so the "item axis" and the "KC axis"
   collapse into literally the same namespace and every per-KC readout
   becomes tautological (`avenue_map.md`: "Circularity if the skill ID
   doubles as the model's item key ... use problem-level item keys
   wherever the bed allows"). The check below fires when a large fraction
   of the item-key vocabulary is ALSO present in the KC-name vocabulary
   (>= 50%, `_CIRCULARITY_OVERLAP_THRESHOLD`) -- the signature of "these
   two axes are actually the same column" -- rather than on any overlap at
   all, since a handful of incidental string collisions between an item
   key and a KC name (unlikely for KDD's composite step keys, but not
   provably impossible for every bed) is not evidence of the ASSISTments
   bug. The threshold is a judgment call, recorded in
   `realbed_bridge_notes.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_CIRCULARITY_OVERLAP_THRESHOLD = 0.5


class CircularQMatrixError(ValueError):
    """Raised when the item-key vocabulary and the KC-name vocabulary look
    like the same namespace (the ASSISTments circularity failure mode)."""


def check_no_circularity(item_keys: Sequence[str], kc_names: Sequence[str]) -> None:
    """Guard against a KC/skill id doubling as the model's item key
    (module docstring note 3). Raises `CircularQMatrixError` if at least
    `_CIRCULARITY_OVERLAP_THRESHOLD` of the item-key vocabulary is also
    present, string-for-string, in the KC-name vocabulary. No-ops on an
    empty vocabulary (nothing to check yet)."""
    item_set = set(item_keys)
    kc_set = set(kc_names)
    if not item_set or not kc_set:
        return
    overlap = item_set & kc_set
    frac = len(overlap) / len(item_set)
    if frac >= _CIRCULARITY_OVERLAP_THRESHOLD:
        raise CircularQMatrixError(
            f"{frac:.0%} of item keys are also KC names -- this is the "
            "ASSISTments circularity failure (a skill/KC id doubling as the "
            "model's item key; avenue_map.md: \"use problem-level item keys "
            "wherever the bed allows\"). Item identity must be the "
            "problem/step key, never the KC tag itself."
        )


@dataclass(frozen=True)
class ExpansionPolicy:
    """Item-to-KC expansion policy (module docstring note 1). ``name``
    labels the policy for reporting; only ``"all_tags"`` is implemented in
    v1 (no alternative policy is pre-registered in the design)."""

    name: str = "all_tags"


ALL_TAGS = ExpansionPolicy("all_tags")


@dataclass
class QMatrix:
    """The static item -> KC-tag map, ragged and padded (design section
    2.5's ``kc_ids``/``kc_mask`` shape, at item-row grain: one row per
    distinct item, ``arity_max`` columns, ``-1`` pad)."""

    item_keys: list  # length n_items, index = item id
    kc_names: list  # length n_kcs, index = kc id
    kc_ids: np.ndarray  # (n_items, arity_max) int64, -1 pad
    kc_mask: np.ndarray  # (n_items, arity_max) bool
    policy: ExpansionPolicy = ALL_TAGS

    @property
    def n_items(self) -> int:
        return len(self.item_keys)

    @property
    def n_kcs(self) -> int:
        return len(self.kc_names)

    @property
    def arity_max(self) -> int:
        return int(self.kc_ids.shape[1]) if self.kc_ids.ndim == 2 else 0

    def arity(self, item_id) -> np.ndarray:
        """Per-item tag count (scalar in, scalar out; array in, array out)."""
        return self.kc_mask[np.asarray(item_id)].sum(axis=-1)


def build_qmatrix(
    item_keys: Sequence[str],
    item_kc_id_lists: Sequence[Sequence[int]],
    kc_names: Sequence[str],
    policy: ExpansionPolicy = ALL_TAGS,
    arity_max: Optional[int] = None,
    skip_circularity_check: bool = False,
) -> QMatrix:
    """Build a `QMatrix` from already-interned per-item KC-id lists
    (``item_kc_id_lists[i]`` is item ``i``'s tag set, ids into
    ``kc_names``). ``item_kc_id_lists`` and ``item_keys`` must be aligned
    and the same length (one entry per item, ordered by item id) -- this
    function does no vocabulary building of its own (`kc_data.py`'s job);
    it only pads the ragged lists into the fixed-width buffer and runs the
    circularity guard (module docstring note 3), skippable only for
    synthetic-fixture tests that intentionally probe the guard's
    complement.
    """
    if len(item_keys) != len(item_kc_id_lists):
        raise ValueError(
            f"item_keys ({len(item_keys)}) and item_kc_id_lists "
            f"({len(item_kc_id_lists)}) must have the same length"
        )
    if not skip_circularity_check:
        check_no_circularity(item_keys, kc_names)

    n_items = len(item_keys)
    observed_max = max((len(tags) for tags in item_kc_id_lists), default=0)
    a_max = arity_max if arity_max is not None else observed_max
    if observed_max > a_max:
        raise ValueError(
            f"arity_max={a_max} given but an item carries {observed_max} tags"
        )
    a_max = max(a_max, 1)

    kc_ids = np.full((n_items, a_max), -1, dtype=np.int64)
    for i, tags in enumerate(item_kc_id_lists):
        if not tags:
            continue
        kc_ids[i, : len(tags)] = np.asarray(tags, dtype=np.int64)
    kc_mask = kc_ids >= 0

    return QMatrix(
        item_keys=list(item_keys),
        kc_names=list(kc_names),
        kc_ids=kc_ids,
        kc_mask=kc_mask,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# Pure-anchor identification and arity statistics
# ---------------------------------------------------------------------------


def pure_anchor_item_mask(qmatrix: QMatrix) -> np.ndarray:
    """Per-item bool: True iff the item is a "pure anchor" (arity exactly
    1, i.e. tagged with exactly one KC)."""
    if qmatrix.n_items == 0:
        return np.zeros(0, dtype=bool)
    return qmatrix.kc_mask.sum(axis=1) == 1


def pure_anchor_kc_ids(qmatrix: QMatrix) -> np.ndarray:
    """The KC id each pure-anchor item is tagged with (one entry per pure
    item; not deduplicated -- feed through ``np.unique`` for the KC set)."""
    is_pure = pure_anchor_item_mask(qmatrix)
    if not is_pure.any():
        return np.zeros(0, dtype=np.int64)
    rows = qmatrix.kc_ids[is_pure]
    row_mask = qmatrix.kc_mask[is_pure]
    return rows[row_mask]


def pure_anchor_stats(qmatrix: QMatrix, min_pure: int = 3) -> dict:
    """Reproduces `triage_common.pure_anchor_and_arity_stats`'s exact
    field set (module docstring note 2), computed vectorized from the
    ragged buffer instead of a `dict[item -> frozenset]`:
    ``n_items``, ``n_kc_in_item_bank`` (KCs that appear on >=1 item),
    ``arity_mean``, ``arity_max``, ``n_pure_items_total``,
    ``min_pure_items_gate``, ``frac_kc_with_ge_min_pure_items``.
    """
    n_items = qmatrix.n_items
    if n_items == 0:
        return {
            "n_items": 0, "n_kc_in_item_bank": 0, "arity_mean": None, "arity_max": None,
            "n_pure_items_total": 0, "min_pure_items_gate": min_pure,
            "frac_kc_with_ge_min_pure_items": None,
        }
    arities = qmatrix.kc_mask.sum(axis=1)
    nonempty = arities > 0
    n_items_nonempty = int(nonempty.sum())
    kc_universe = np.unique(qmatrix.kc_ids[qmatrix.kc_mask]) if qmatrix.kc_mask.any() else np.zeros(0, dtype=np.int64)
    n_kc = int(kc_universe.size)

    pure_kc = pure_anchor_kc_ids(qmatrix)
    if pure_kc.size:
        pure_ids, pure_counts = np.unique(pure_kc, return_counts=True)
        pure_count_by_kc = dict(zip(pure_ids.tolist(), pure_counts.tolist()))
    else:
        pure_count_by_kc = {}
    n_kc_with_ge_min = sum(1 for kc in kc_universe.tolist() if pure_count_by_kc.get(kc, 0) >= min_pure)

    return {
        "n_items": n_items_nonempty,
        "n_kc_in_item_bank": n_kc,
        "arity_mean": float(arities[nonempty].mean()) if n_items_nonempty else None,
        "arity_max": int(arities[nonempty].max()) if n_items_nonempty else None,
        "n_pure_items_total": int((arities == 1).sum()),
        "min_pure_items_gate": min_pure,
        "frac_kc_with_ge_min_pure_items": (n_kc_with_ge_min / n_kc) if n_kc else None,
    }


__all__ = [
    "CircularQMatrixError",
    "check_no_circularity",
    "ExpansionPolicy",
    "ALL_TAGS",
    "QMatrix",
    "build_qmatrix",
    "pure_anchor_item_mask",
    "pure_anchor_kc_ids",
    "pure_anchor_stats",
]
