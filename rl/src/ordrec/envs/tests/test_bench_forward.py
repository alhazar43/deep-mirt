"""Bench-forward harness smoke tests.

The full benchmark is run as a one-off script in
``rl/scripts`` or via the ``__main__`` workflow during E2. The
unit tests confirm that the harness runs end-to-end and that the
no_grad invariance regression catches a deliberately broken model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ordrec.envs.bench_forward import (
    BenchConfig,
    BenchResult,
    bench_forward_pass,
    no_grad_invariance_check,
    write_bench_artifacts,
)
from ordrec.envs.frozen_magpcm import FrozenMAGPCM


pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


def _build_tiny_magpcm():
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    return MAGPCM(
        n_questions=16,
        n_categories=4,
        n_traits=1,
        memory_size=8,
        key_dim=8,
        value_dim=8,
        summary_dim=8,
        embedding_type="learned",
        dropout_rate=0.0,
        ability_scale=1.0,
        separate_theta=True,
        init_value_memory=False,
    )


def _frozen_factory():
    return FrozenMAGPCM(_build_tiny_magpcm())


def test_bench_forward_runs_end_to_end() -> None:
    cfg = BenchConfig(
        seq_len=12,
        batch_sizes=(1, 4),
        warmup_iters=1,
        repeat=3,
        encoders={"magpcm_tiny": _frozen_factory},
        device="cpu",
        n_questions=16,
        n_categories=4,
        seed=0,
    )
    results = bench_forward_pass(cfg)
    assert len(results) == 2  # one row per (encoder, batch_size)
    for r in results:
        assert isinstance(r, BenchResult)
        assert r.median_ms >= 0.0
        assert r.min_ms <= r.median_ms <= r.max_ms
        assert r.n_repeats == 3
        assert r.device == "cpu"
        assert r.seq_len == 12
        assert r.encoder == "magpcm_tiny"


def test_bench_forward_rejects_empty_encoders() -> None:
    cfg = BenchConfig(encoders={})
    with pytest.raises(ValueError):
        bench_forward_pass(cfg)


def test_no_grad_invariance_passes_on_frozen_model() -> None:
    frozen = _frozen_factory()
    q = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    r = torch.tensor([[0, 1, 2, 3, 1]], dtype=torch.long)
    report = no_grad_invariance_check(frozen, q, r, n_calls=3)
    assert report["passed"] is True
    assert report["max_abs_diff"] == 0.0


def test_no_grad_invariance_catches_a_broken_model() -> None:
    """Construct a model whose forward is intentionally non-deterministic
    and verify the invariance check raises."""

    class JitteryFrozen(FrozenMAGPCM):
        def forward_no_grad(self, questions, responses):  # type: ignore[override]
            out = super().forward_no_grad(questions, responses)
            # Add a non-deterministic perturbation each call to simulate
            # a forgotten dropout layer.
            noise = torch.randn_like(out["logits"]) * 1e-2
            out["logits"] = out["logits"] + noise
            return out

    jittery = JitteryFrozen(_build_tiny_magpcm())
    q = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    r = torch.tensor([[0, 1, 2, 3, 1]], dtype=torch.long)
    with pytest.raises(AssertionError):
        no_grad_invariance_check(jittery, q, r, n_calls=3, atol=0.0)


def test_write_bench_artifacts_round_trip(tmp_path: Path) -> None:
    results = [
        BenchResult(
            encoder="dkvmn", batch_size=1, seq_len=50, device="cpu",
            median_ms=8.5, mean_ms=8.7, min_ms=8.1, max_ms=9.3, n_repeats=5,
        ),
        BenchResult(
            encoder="lstm", batch_size=128, seq_len=50, device="cuda",
            median_ms=4.2, mean_ms=4.3, min_ms=4.0, max_ms=4.8, n_repeats=5,
        ),
    ]
    json_path = tmp_path / "bench.json"
    md_path = tmp_path / "bench.md"
    write_bench_artifacts(
        results, json_path=json_path, markdown_path=md_path,
        extras={"torch_version": torch.__version__},
    )
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "results" in payload and len(payload["results"]) == 2
    assert payload["extras"]["torch_version"] == torch.__version__
    md = md_path.read_text(encoding="utf-8")
    assert "dkvmn" in md and "lstm" in md
    assert "cpu" in md and "cuda" in md


def test_invariance_check_requires_n_calls_at_least_two() -> None:
    frozen = _frozen_factory()
    q = torch.tensor([[1, 2, 3]], dtype=torch.long)
    r = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with pytest.raises(ValueError):
        no_grad_invariance_check(frozen, q, r, n_calls=1)
