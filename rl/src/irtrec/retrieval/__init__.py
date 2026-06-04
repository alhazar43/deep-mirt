"""Retrieval pillar.

Exposes the JobPoolSpec, the JobTower that embeds the pool, and the
RetrievalIndex that serves top-k queries.

The legacy alias ``ItemTower`` is kept as a one-line shim for
backward compatibility with M0..M3 callers; new code should import
``JobTower`` directly. The shim is scheduled for removal in M8-RL.
"""

from .pool import JobPoolSpec, load_onet_pool
from .job_tower import JobTower, TextEncoderInfo, BGE_MODEL_NAME
from .index import RetrievalIndex

# Deprecated alias, remove in M8-RL.
ItemTower = JobTower

__all__ = [
    "JobPoolSpec",
    "load_onet_pool",
    "JobTower",
    "ItemTower",
    "TextEncoderInfo",
    "BGE_MODEL_NAME",
    "RetrievalIndex",
]
