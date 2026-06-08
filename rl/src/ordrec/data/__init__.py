"""Data adapters for OrdRec.

Public exports cover the adapter ABC, the dataset-specific concrete
adapters that exist in E1, and the ma-irt integration bridge. The
``build_adapter`` factory routes a config dict to the appropriate
concrete class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import AdapterConfig, OrdinalDatasetBase, Split
from .ednet import EdNetAdapter
from .eedi import EediAdapter
from .ma_irt_bridge import adapter_to_dataloader, adapter_to_sequence_dataset
from .placeholder_2pl import Placeholder2PLFit, fit_placeholder_2pl
from .schema import (
    COMMON_RECORD_SCHEMA,
    SchemaError,
    validate_metadata,
    validate_q_matrix,
    validate_sequences,
)
from .split import _chunk_sequences, make_split, stratified_split
from .synthetic import SyntheticAdapter


# Adapter registry, keyed by the value users set in YAML configs.
_ADAPTER_REGISTRY = {
    "synthetic": SyntheticAdapter,
    "eedi": EediAdapter,
    "ednet": EdNetAdapter,
}


def build_adapter(cfg: Dict[str, Any]) -> OrdinalDatasetBase:
    """Factory, ``{type, name, raw_dir, out_dir, ...}`` -> adapter instance.

    Args:
        cfg: Plain-dict config with at least ``type`` (one of
            ``"synthetic"``, ``"eedi"``) plus the fields of
            ``AdapterConfig`` (``name``, ``raw_dir``, ``out_dir``).
            Optional keys forward to ``AdapterConfig``.

    Returns:
        A constructed adapter instance. The caller is responsible for
        calling ``materialise()`` and ``load()`` in order.
    """
    if "type" not in cfg:
        raise KeyError("build_adapter cfg requires a 'type' key.")
    kind = cfg["type"]
    if kind not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter type {kind!r}, must be one of "
            f"{sorted(_ADAPTER_REGISTRY)}"
        )
    adapter_cls = _ADAPTER_REGISTRY[kind]
    payload = {k: v for k, v in cfg.items() if k != "type"}
    payload["raw_dir"] = Path(payload["raw_dir"])
    payload["out_dir"] = Path(payload["out_dir"])
    return adapter_cls(AdapterConfig(**payload))


__all__ = [
    "AdapterConfig",
    "COMMON_RECORD_SCHEMA",
    "EdNetAdapter",
    "EediAdapter",
    "OrdinalDatasetBase",
    "Placeholder2PLFit",
    "SchemaError",
    "Split",
    "SyntheticAdapter",
    "_chunk_sequences",
    "adapter_to_dataloader",
    "adapter_to_sequence_dataset",
    "build_adapter",
    "fit_placeholder_2pl",
    "make_split",
    "stratified_split",
    "validate_metadata",
    "validate_q_matrix",
    "validate_sequences",
]
