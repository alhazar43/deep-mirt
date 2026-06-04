"""Embed a job pool parquet and dump v_j + job_ids.

Per the v1 plan, Section 6.4, the ItemTower is trained at M5; for
M2 we just need a frozen forward pass over the O*NET pool so the
RetrievalIndex has something to serve. The script loads the parquet,
builds a JobPoolSpec, runs the ItemTower in eval mode, and saves the
matrix as float32 numpy plus an ordered JSON of occupation codes.

Example
-------
.. code-block:: bash

    PYTHONPATH=rl/src python rl/scripts/register_pool.py \
        --parquet rl/artifacts/onet_v1.parquet \
        --output-embed rl/artifacts/onet_v1_embed.npy \
        --output-jobids rl/artifacts/onet_v1_jobids.json \
        --d 64 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure rl/src is importable when running this as a standalone script.
_HERE = Path(__file__).resolve().parent
_RL_SRC = _HERE.parent / "src"
if str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from irtrec.retrieval import ItemTower, load_onet_pool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("rl/artifacts/onet_v1.parquet"),
        help="Path to the pool parquet.",
    )
    parser.add_argument(
        "--output-embed",
        type=Path,
        default=Path("rl/artifacts/onet_v1_embed.npy"),
        help="Path for the v_j matrix (float32 numpy).",
    )
    parser.add_argument(
        "--output-jobids",
        type=Path,
        default=Path("rl/artifacts/onet_v1_jobids.json"),
        help="Path for the ordered occupation_code list (JSON).",
    )
    parser.add_argument("--d", type=int, default=64, help="Output embedding dim.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the ItemTower forward pass.",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip the BGE checkpoint and use the deterministic fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = load_onet_pool(args.parquet)
    print(f"[register_pool] loaded {pool.n_jobs} jobs from {args.parquet}", flush=True)

    torch.manual_seed(42)
    tower = ItemTower(d=args.d, device=args.device, force_fallback=args.force_fallback)
    tower.eval()
    print(
        f"[register_pool] text encoder: {tower.text_info.name} "
        f"(fallback={tower.text_info.fallback}, dim={tower.text_info.dim})",
        flush=True,
    )

    with torch.no_grad():
        v_j = tower(pool)
    v_j_np = v_j.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(v_j_np, axis=1)
    print(
        f"[register_pool] v_j shape={v_j_np.shape}, "
        f"norm range=[{norms.min():.4f}, {norms.max():.4f}]",
        flush=True,
    )

    args.output_embed.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_embed, v_j_np)
    args.output_jobids.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jobids, "w", encoding="utf-8") as f:
        json.dump(pool.occupation_codes, f)

    print(f"[register_pool] wrote {args.output_embed}", flush=True)
    print(f"[register_pool] wrote {args.output_jobids}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
