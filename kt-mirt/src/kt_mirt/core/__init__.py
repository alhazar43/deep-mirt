"""
deep_irt.core -- Canonical dynamic assessment deep_irt package.

Public exports:
    LSTMEncoder             -- LSTM encoder: response history -> per-step theta
    GPCMDecoder             -- Ordinal K-category GPCM decoder
    Binary2PLDecoder        -- Binary 2PL (GPCM at K=2)
    BradleyTerryDecoder     -- Pairwise Bradley-Terry decoder
    NRMDecoder              -- Bock nominal response model (unordered K options)
    DeepIRTModel          -- High-level API (fit, track, recover_item_params, extend)
    anchored_extend         -- Low-level anchoring function
    build_extended_encoder  -- Build B+E encoder from base + extension weights
    collate_adapter_items   -- Bridge: adapter dicts -> padded deep_irt tensors
    CollatedBatch           -- NamedTuple returned by collate_adapter_items
"""

from kt_mirt.core.encoder import LSTMEncoder
from kt_mirt.core.decoders import (
    GPCMDecoder,
    Binary2PLDecoder,
    BradleyTerryDecoder,
    NRMDecoder,
)
from kt_mirt.core.anchor import anchored_extend, build_extended_encoder
from kt_mirt.core.model import DeepIRTModel
from kt_mirt.core.realdata import collate_adapter_items, CollatedBatch

__all__ = [
    "LSTMEncoder",
    "GPCMDecoder",
    "Binary2PLDecoder",
    "BradleyTerryDecoder",
    "NRMDecoder",
    "DeepIRTModel",
    "anchored_extend",
    "build_extended_encoder",
    "collate_adapter_items",
    "CollatedBatch",
]
