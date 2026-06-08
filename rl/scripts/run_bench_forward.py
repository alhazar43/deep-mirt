"""Run the E2 forward-pass bench harness and persist the artifacts.

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
        python rl/scripts/run_bench_forward.py

Writes ``rl/results/E2_bench_forward.json`` and a markdown summary
``rl/results/E2_bench_forward.md`` with median forward-pass latency
across (B, T) = (1, 50) and (B, T) = (128, 50) for DKVMN, LSTM and
Transformer backbones. CUDA columns are included when
``torch.cuda.is_available()``; otherwise only the CPU sweep runs.

This script is the deliverable execution for the E2 bench harness. The
model sizes match the synthetic-smoke YAML at
``ma-irt/configs/ordrec_synth_smoke.yaml`` so the numbers are
reproducible without a trained Eedi checkpoint.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Callable, Dict

import torch

from ordrec.envs.bench_forward import (
    BenchConfig,
    bench_forward_pass,
    write_bench_artifacts,
)
from ordrec.envs.frozen_magpcm import FrozenMAGPCM


def _build_dkvmn(device: torch.device) -> FrozenMAGPCM:
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    m = MAGPCM(
        n_questions=200,
        n_categories=4,
        n_traits=1,
        memory_size=32,
        key_dim=32,
        value_dim=32,
        summary_dim=32,
        embedding_type="learned",
        dropout_rate=0.0,
        ability_scale=1.0,
        separate_theta=True,
        init_value_memory=False,
    ).to(device)
    return FrozenMAGPCM(m)


def _build_lstm(device: torch.device) -> FrozenMAGPCM:
    from models.lstm_gpcm import LSTMGPCM  # type: ignore[import-not-found]

    m = LSTMGPCM(
        n_questions=200,
        n_categories=4,
        n_traits=1,
        d_model=64,
        n_layers=1,
        dropout_rate=0.0,
        key_dim=32,
        value_dim=32,
        embedding_type="learned",
        ability_scale=1.0,
        separate_theta=True,
    ).to(device)
    return FrozenMAGPCM(m, n_questions=200, n_categories=4)


def _build_transformer(device: torch.device) -> FrozenMAGPCM:
    from models.transformer_gpcm import TransformerGPCM  # type: ignore[import-not-found]

    m = TransformerGPCM(
        n_questions=200,
        n_categories=4,
        n_traits=1,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout_rate=0.0,
        max_seq_len=128,
        key_dim=32,
        value_dim=32,
        embedding_type="learned",
        ability_scale=1.0,
        separate_theta=True,
    ).to(device)
    return FrozenMAGPCM(m, n_questions=200, n_categories=4)


def _run_for_device(device_str: str) -> list:
    device = torch.device(device_str)
    encoders: Dict[str, Callable[[], FrozenMAGPCM]] = {
        "dkvmn": lambda d=device: _build_dkvmn(d),
        "lstm": lambda d=device: _build_lstm(d),
        "transformer": lambda d=device: _build_transformer(d),
    }
    cfg = BenchConfig(
        seq_len=50,
        batch_sizes=(1, 128),
        warmup_iters=3,
        repeat=10,
        encoders=encoders,
        device=device_str,
        n_questions=200,
        n_categories=4,
        seed=0,
    )
    return bench_forward_pass(cfg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json-out", type=Path, default=Path("rl/results/E2_bench_forward.json"),
    )
    p.add_argument(
        "--md-out", type=Path, default=Path("rl/results/E2_bench_forward.md"),
    )
    p.add_argument("--skip-cuda", action="store_true")
    args = p.parse_args(argv)

    all_results = []
    print("Running CPU bench sweep...")
    all_results.extend(_run_for_device("cpu"))
    if torch.cuda.is_available() and not args.skip_cuda:
        print("Running CUDA bench sweep...")
        all_results.extend(_run_for_device("cuda"))
    else:
        print("CUDA unavailable, skipping the GPU sweep.")

    extras = {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        ),
        "n_questions": 200,
        "n_categories": 4,
        "warmup_iters": 3,
        "repeat": 10,
    }
    write_bench_artifacts(
        all_results,
        json_path=args.json_out,
        markdown_path=args.md_out,
        extras=extras,
    )
    print(f"wrote bench results -> {args.json_out}")
    print(f"wrote bench summary -> {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
