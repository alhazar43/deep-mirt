"""Forward-pass latency harness and the no_grad invariance regression.

The strategic plan promises a one-off ``bench_forward.py`` measurement
across ``(B in {1, 128}, T = 50)`` for the three MA-IRT encoders
(DKVMN, LSTM, Transformer). The numbers feed into the E2 deliverable
at ``rl/results/E2_bench_forward.{json,md}`` and let downstream
milestones swap encoders without re-architecting if DKVMN's
single-user CPU latency turns out to be a bottleneck.

The harness also exposes :func:`no_grad_invariance_check`, a small
regression confirming the freeze contract holds across repeated
forward calls. Identical ``(questions, responses)`` must yield
identical ``logits`` across calls under ``no_grad`` plus ``eval``.

The module is deliberately encoder-agnostic. ``BenchConfig`` carries
the constructors for each encoder family; the caller builds them once
and passes them in. This keeps the harness portable, the same code
serves the smoke test (synthetic small MAGPCM) and any future
benchmark (a fully trained Eedi K=4 model).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .frozen_magpcm import FrozenMAGPCM


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    """Configuration for one bench sweep.

    Attributes:
        seq_len: Sequence length T per row.
        batch_sizes: Iterable of B values to time.
        warmup_iters: Number of unrecorded forward passes before timing.
        repeat: Number of timed iterations per ``(encoder, B)`` cell.
        encoders: Mapping ``{encoder_name: callable -> FrozenMAGPCM}``,
            each callable returning a fresh frozen model on the desired
            device. ``callable()`` must be deterministic so repeated
            runs of the harness produce comparable numbers.
        device: Torch device the sweep runs on.
        n_questions: Item bank size used when synthesising the input
            tensors. Defaults to a small fixture-friendly value; bigger
            models are fine because the bench inputs only need ids in
            ``[1, n_questions]`` of the underlying model.
        n_categories: Number of ordinal categories used for the
            response tensor; same applicability rule as
            ``n_questions``.
        seed: Seed for the synthetic input tensors.
    """

    seq_len: int = 50
    batch_sizes: Sequence[int] = (1, 128)
    warmup_iters: int = 2
    repeat: int = 5
    encoders: Dict[str, Callable[[], FrozenMAGPCM]] = field(default_factory=dict)
    device: str = "cpu"
    n_questions: int = 64
    n_categories: int = 4
    seed: int = 0


@dataclass(frozen=True)
class BenchResult:
    """Result for one ``(encoder, batch_size)`` cell.

    Attributes:
        encoder: Name passed through from ``BenchConfig.encoders``.
        batch_size: B value.
        seq_len: T value.
        device: Device string ``cpu``/``cuda``.
        median_ms: Median of the per-iteration wall times in ms.
        mean_ms: Arithmetic mean of the same.
        min_ms: Min of the same.
        max_ms: Max of the same.
        n_repeats: Number of timed iterations actually run.
    """

    encoder: str
    batch_size: int
    seq_len: int
    device: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    n_repeats: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    batch_size: int,
    seq_len: int,
    n_questions: int,
    n_categories: int,
    seed: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Synthesise random ``(questions, responses)`` for the harness."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    questions = torch.randint(
        low=1, high=n_questions + 1, size=(batch_size, seq_len),
        generator=g, dtype=torch.long,
    ).to(device)
    responses = torch.randint(
        low=0, high=n_categories, size=(batch_size, seq_len),
        generator=g, dtype=torch.long,
    ).to(device)
    return questions, responses


def _time_one_call(
    model: FrozenMAGPCM, questions: Tensor, responses: Tensor, device: torch.device
) -> float:
    """Return wall-clock time in ms for a single ``forward_no_grad`` call."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model.forward_no_grad(questions, responses)
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))
    t0 = time.perf_counter()
    _ = model.forward_no_grad(questions, responses)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bench_forward_pass(cfg: BenchConfig) -> List[BenchResult]:
    """Run the bench sweep described by ``cfg`` and return per-cell results.

    The function does not write anything to disk. Use
    :func:`write_bench_artifacts` to persist a JSON + markdown summary.
    """
    if not cfg.encoders:
        raise ValueError("BenchConfig.encoders is empty, nothing to time.")
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "BenchConfig requested CUDA but torch.cuda.is_available() is False."
        )
    results: List[BenchResult] = []

    for enc_name, build_model in cfg.encoders.items():
        # Build a fresh frozen model for this encoder.
        model = build_model().to(device)
        if not isinstance(model, FrozenMAGPCM):
            raise TypeError(
                f"Encoder builder for {enc_name!r} must return a "
                f"FrozenMAGPCM, got {type(model).__name__}."
            )
        # Defensive eval/no-grad even if the builder forgot.
        model.eval()

        for b in cfg.batch_sizes:
            questions, responses = _make_inputs(
                b, cfg.seq_len, cfg.n_questions, cfg.n_categories,
                cfg.seed, device,
            )
            # Warmup. Sets cuDNN, allocates workspace, JIT-compiles any
            # graph capture, all of which would bias the first timed call.
            for _ in range(int(cfg.warmup_iters)):
                _ = _time_one_call(model, questions, responses, device)
            samples: List[float] = []
            for _ in range(int(cfg.repeat)):
                samples.append(_time_one_call(model, questions, responses, device))
            samples_sorted = sorted(samples)
            results.append(BenchResult(
                encoder=str(enc_name),
                batch_size=int(b),
                seq_len=int(cfg.seq_len),
                device=device.type,
                median_ms=float(statistics.median(samples_sorted)),
                mean_ms=float(statistics.mean(samples_sorted)),
                min_ms=float(min(samples_sorted)),
                max_ms=float(max(samples_sorted)),
                n_repeats=int(cfg.repeat),
            ))

    return results


def no_grad_invariance_check(
    frozen: FrozenMAGPCM,
    questions: Tensor,
    responses: Tensor,
    n_calls: int = 5,
    atol: float = 0.0,
) -> Dict[str, Any]:
    """Confirm identical inputs yield identical outputs across calls.

    Under ``model.eval()`` plus ``torch.no_grad()``, repeated forward
    passes on the same ``(questions, responses)`` must produce
    bit-identical (``atol = 0``) logits. Any drift indicates the freeze
    contract leaked, e.g. an active dropout layer or a hidden state
    that persisted across calls.

    Args:
        frozen: A ``FrozenMAGPCM`` instance.
        questions: ``LongTensor (B, S)``.
        responses: ``LongTensor (B, S)``.
        n_calls: Number of repeated calls to compare. The first call is
            the reference; calls ``1..n_calls - 1`` are compared
            against it.
        atol: Absolute tolerance for the equality check. Default ``0``
            for strict bit-identity; raise to a small ``1e-7`` if the
            harness is run on a non-deterministic backend.

    Returns:
        A small report dict with ``passed``, ``max_abs_diff``,
        ``n_calls`` and ``n_pairs`` keys.

    Raises:
        AssertionError: When ``max_abs_diff > atol``.
    """
    if n_calls < 2:
        raise ValueError(f"n_calls must be >= 2, got {n_calls}.")
    ref = frozen.forward_no_grad(questions, responses)["logits"]
    max_diff = 0.0
    for _ in range(n_calls - 1):
        out = frozen.forward_no_grad(questions, responses)["logits"]
        diff = float((out - ref).abs().max().item())
        if diff > max_diff:
            max_diff = diff
    passed = max_diff <= atol
    if not passed:
        raise AssertionError(
            f"no_grad invariance failed, max_abs_diff={max_diff} > atol={atol}"
        )
    return {
        "passed": bool(passed),
        "max_abs_diff": float(max_diff),
        "n_calls": int(n_calls),
        "n_pairs": int(n_calls - 1),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_bench_artifacts(
    results: Sequence[BenchResult],
    *,
    json_path: Path,
    markdown_path: Optional[Path] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the bench results to JSON and (optionally) markdown.

    Args:
        results: Output of :func:`bench_forward_pass`.
        json_path: Where to write the canonical JSON record.
        markdown_path: Optional markdown summary path. When provided a
            small table is rendered for human reading.
        extras: Optional dict of metadata (e.g. ``commit`` SHA,
            ``torch_version``) merged into the JSON payload under an
            ``extras`` key.
    """
    payload: Dict[str, Any] = {
        "results": [asdict(r) for r in results],
        "extras": dict(extras or {}),
    }
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    if markdown_path is not None:
        md = ["# E2 bench forward results", ""]
        if extras:
            for k, v in sorted(extras.items()):
                md.append(f"- {k}, {v}")
            md.append("")
        md.append(
            "| Encoder | Device | B | T | Median (ms) | Mean (ms) | "
            "Min (ms) | Max (ms) | Repeats |"
        )
        md.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for r in results:
            md.append(
                f"| {r.encoder} | {r.device} | {r.batch_size} | {r.seq_len} | "
                f"{r.median_ms:.3f} | {r.mean_ms:.3f} | "
                f"{r.min_ms:.3f} | {r.max_ms:.3f} | {r.n_repeats} |"
            )
        md.append("")
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("\n".join(md), encoding="utf-8")


__all__ = [
    "BenchConfig",
    "BenchResult",
    "bench_forward_pass",
    "no_grad_invariance_check",
    "write_bench_artifacts",
]
