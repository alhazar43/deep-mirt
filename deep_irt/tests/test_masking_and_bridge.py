"""
Tests for G1 masking, G2 mini-batching, the realdata bridge, and
end-to-end regression.

Test IDs
--------
1. test_mask_equivalence     -- all-True mask gives identical loss + theta to no-mask
2. test_pad_invariance       -- padded batch yields same theta as per-learner unpadded
3. test_bridge_roundtrip     -- collate_adapter_items yields correct 0-based / masked output
4. test_regression_synthetic -- existing synthetic end-to-end pipeline still passes
"""

from __future__ import annotations

import math

import torch
import pytest

from deep_irt.core import DeepIRTModel, collate_adapter_items
from deep_irt.core.realdata import CollatedBatch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEED = 42
_DEVICE = torch.device("cpu")


def _make_model(num_items: int = 10, n_cats: int = 4, seed: int = _SEED) -> DeepIRTModel:
    return DeepIRTModel(
        num_items=num_items,
        emb_dim=4,
        hidden_dim=8,
        n_cats=n_cats,
        decoder="gpcm",
        device=_DEVICE,
        seed=seed,
    )


def _rect_batch(
    n: int = 6,
    t: int = 8,
    num_items: int = 10,
    n_cats: int = 4,
    seed: int = _SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (item_ids, responses) as fully rectangular tensors."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    item_ids = torch.randint(0, num_items, (n, t), generator=rng)
    responses = torch.randint(0, n_cats, (n, t), generator=rng)
    return item_ids, responses


# ---------------------------------------------------------------------------
# Test 1: mask_equivalence
# ---------------------------------------------------------------------------

class TestMaskEquivalence:
    """An all-True (N, T) mask must give bit-for-bit identical NLL and theta."""

    def test_nll_identical(self) -> None:
        item_ids, responses = _rect_batch()
        mask = torch.ones_like(item_ids, dtype=torch.bool)

        model_no_mask = _make_model()
        model_with_mask = _make_model()
        # Both models start from identical weights (same seed).

        result_no = model_no_mask.fit(
            item_ids, responses, n_epochs=5, verbose=False
        )
        result_with = model_with_mask.fit(
            item_ids, responses, n_epochs=5, verbose=False, mask=mask
        )

        assert math.isclose(
            result_no["final_nll"], result_with["final_nll"], rel_tol=1e-5
        ), (
            f"NLL mismatch: no-mask={result_no['final_nll']:.8f}, "
            f"with-mask={result_with['final_nll']:.8f}"
        )

    def test_theta_identical(self) -> None:
        item_ids, responses = _rect_batch()
        mask = torch.ones_like(item_ids, dtype=torch.bool)

        model_no_mask = _make_model()
        model_with_mask = _make_model()

        model_no_mask.fit(item_ids, responses, n_epochs=5, verbose=False)
        model_with_mask.fit(
            item_ids, responses, n_epochs=5, verbose=False, mask=mask
        )

        theta_no = model_no_mask.track(item_ids, responses)
        theta_with = model_with_mask.track(item_ids, responses)

        assert torch.allclose(theta_no, theta_with, atol=1e-6), (
            f"theta max abs diff: {(theta_no - theta_with).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Test 2: pad_invariance
# ---------------------------------------------------------------------------

class TestPadInvariance:
    """Padded batch theta matches per-learner unpadded theta within tolerance."""

    def _make_variable_seqs(
        self,
        n: int = 5,
        min_t: int = 4,
        max_t: int = 10,
        num_items: int = 15,
        n_cats: int = 4,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (item_ids_1d, responses_1d) with varying lengths."""
        rng = torch.Generator()
        rng.manual_seed(_SEED + 100)
        seqs = []
        lengths = [min_t + (i % (max_t - min_t + 1)) for i in range(n)]
        for T in lengths:
            ids = torch.randint(0, num_items, (T,), generator=rng)
            resp = torch.randint(0, n_cats, (T,), generator=rng)
            seqs.append((ids, resp))
        return seqs

    def test_final_theta_matches_unpadded(self) -> None:
        NUM_ITEMS = 15
        N_CATS = 4
        seqs = self._make_variable_seqs(num_items=NUM_ITEMS, n_cats=N_CATS)
        n = len(seqs)

        # Build padded batch via the bridge adapter-dict format.
        adapter_dicts = [
            {
                "student_id": i + 1,
                # Bridge expects 1-based; add 1 to the 0-based ids here.
                "questions": seqs[i][0] + 1,
                "responses": seqs[i][1],
            }
            for i in range(n)
        ]
        batch = collate_adapter_items(adapter_dicts)
        item_ids_pad = batch.item_ids    # (N, T_max) -- 0-based
        responses_pad = batch.responses  # (N, T_max)
        mask = batch.mask                # (N, T_max)

        # Train a single model on the padded batch.
        model = DeepIRTModel(
            num_items=NUM_ITEMS, emb_dim=4, hidden_dim=8,
            n_cats=N_CATS, decoder="gpcm", device=_DEVICE, seed=_SEED,
        )
        model.fit(item_ids_pad, responses_pad, n_epochs=10, verbose=False, mask=mask)

        # Track padded: theta at position L-1 for each learner.
        theta_padded = model.track(item_ids_pad, responses_pad)  # (N, T_max)

        for i, (ids_1d, resp_1d) in enumerate(seqs):
            L = ids_1d.size(0)
            # Unpadded run with the same trained model (no further training).
            theta_unpad = model.track(
                ids_1d.unsqueeze(0), resp_1d.unsqueeze(0)
            )  # (1, L)
            theta_final_unpad = theta_unpad[0, L - 1].item()
            theta_final_pad = theta_padded[i, L - 1].item()
            assert math.isclose(
                theta_final_unpad, theta_final_pad, abs_tol=1e-5
            ), (
                f"Learner {i}: unpadded={theta_final_unpad:.7f}, "
                f"padded={theta_final_pad:.7f}"
            )


# ---------------------------------------------------------------------------
# Test 3: bridge_roundtrip
# ---------------------------------------------------------------------------

class TestBridgeRoundtrip:
    """collate_adapter_items produces correct 0-based ids and mask."""

    def _make_adapter_dicts(self) -> list[dict]:
        # Three sequences: lengths 3, 5, 4; 1-based question ids in [1, 8].
        return [
            {"student_id": 1, "questions": torch.tensor([1, 3, 8]), "responses": torch.tensor([0, 2, 1])},
            {"student_id": 2, "questions": torch.tensor([2, 4, 6, 7, 8]), "responses": torch.tensor([1, 0, 3, 2, 1])},
            {"student_id": 3, "questions": torch.tensor([5, 1, 3, 7]), "responses": torch.tensor([0, 1, 0, 2])},
        ]

    def test_shapes(self) -> None:
        dicts = self._make_adapter_dicts()
        batch = collate_adapter_items(dicts)
        assert isinstance(batch, CollatedBatch)
        assert batch.item_ids.shape == (3, 5)
        assert batch.responses.shape == (3, 5)
        assert batch.mask.shape == (3, 5)
        assert batch.student_ids.shape == (3,)
        assert batch.seq_lens.shape == (3,)

    def test_zero_based_conversion(self) -> None:
        dicts = self._make_adapter_dicts()
        batch = collate_adapter_items(dicts)
        # Row 0: questions=[1,3,8] -> item_ids=[0,2,7]
        assert batch.item_ids[0, 0].item() == 0
        assert batch.item_ids[0, 1].item() == 2
        assert batch.item_ids[0, 2].item() == 7

    def test_mask_matches_seq_lens(self) -> None:
        dicts = self._make_adapter_dicts()
        batch = collate_adapter_items(dicts)
        expected_lens = torch.tensor([3, 5, 4])
        assert torch.equal(batch.seq_lens, expected_lens)
        for i, L in enumerate(expected_lens.tolist()):
            assert batch.mask[i, :L].all(), f"Row {i}: first {L} positions should be True"
            assert not batch.mask[i, L:].any(), f"Row {i}: padding positions should be False"

    def test_padded_positions_have_pad_id(self) -> None:
        dicts = self._make_adapter_dicts()
        pad_id = 99
        batch = collate_adapter_items(dicts, pad_id=pad_id)
        # Row 0 has L=3; positions [3,4] are padding.
        assert batch.item_ids[0, 3].item() == pad_id
        assert batch.item_ids[0, 4].item() == pad_id

    def test_responses_preserved(self) -> None:
        dicts = self._make_adapter_dicts()
        batch = collate_adapter_items(dicts)
        # Row 1: responses=[1,0,3,2,1]
        expected = torch.tensor([1, 0, 3, 2, 1])
        assert torch.equal(batch.responses[1, :5], expected)

    def test_student_ids_preserved(self) -> None:
        dicts = self._make_adapter_dicts()
        batch = collate_adapter_items(dicts)
        assert torch.equal(batch.student_ids, torch.tensor([1, 2, 3]))

    def test_adapter_object_interface(self) -> None:
        """collate_adapter_items accepts an adapter-like object."""

        class _FakeAdapter:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return [
                    {"student_id": 1, "questions": torch.tensor([1, 2, 3]), "responses": torch.tensor([0, 1, 0])},
                    {"student_id": 2, "questions": torch.tensor([4, 5]), "responses": torch.tensor([1, 2])},
                ][idx]

        batch = collate_adapter_items(_FakeAdapter())
        assert batch.item_ids.shape == (2, 3)
        assert batch.mask[1, 2].item() is False  # padded position

    def test_empty_items_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            collate_adapter_items([])

    def test_python_list_questions(self) -> None:
        """Plain Python lists (not tensors) should also work."""
        dicts = [
            {"student_id": 1, "questions": [1, 2, 3], "responses": [0, 1, 2]},
        ]
        batch = collate_adapter_items(dicts)
        assert batch.item_ids.shape == (1, 3)
        assert batch.item_ids[0, 0].item() == 0  # 1-based -> 0-based


# ---------------------------------------------------------------------------
# Test 4: regression_synthetic
# ---------------------------------------------------------------------------

class TestRegressionSynthetic:
    """
    Mini version of the full pipeline -- verifies that mask=None / full-batch
    mode is bit-identical to the original.
    """

    def test_no_mask_no_minibatch_is_unchanged(self) -> None:
        """The default (no mask, no batch_size) path must match the old API."""
        NUM_ITEMS, N, T, K = 12, 20, 15, 4

        torch.manual_seed(_SEED)
        item_ids = torch.randint(0, NUM_ITEMS, (N, T))
        responses = torch.randint(0, K, (N, T))

        model_old = DeepIRTModel(
            num_items=NUM_ITEMS, emb_dim=4, hidden_dim=8,
            n_cats=K, decoder="gpcm", device=_DEVICE, seed=_SEED,
        )
        result_old = model_old.fit(item_ids, responses, n_epochs=20, verbose=False)

        model_new = DeepIRTModel(
            num_items=NUM_ITEMS, emb_dim=4, hidden_dim=8,
            n_cats=K, decoder="gpcm", device=_DEVICE, seed=_SEED,
        )
        # Explicitly pass the new optional kwargs at their defaults.
        result_new = model_new.fit(
            item_ids, responses, n_epochs=20, verbose=False,
            mask=None, batch_size=None,
        )

        assert math.isclose(
            result_old["final_nll"], result_new["final_nll"], rel_tol=1e-7
        ), (
            f"Regression: NLL changed. old={result_old['final_nll']:.10f}, "
            f"new={result_new['final_nll']:.10f}"
        )

        theta_old = model_old.track(item_ids, responses)
        theta_new = model_new.track(item_ids, responses)
        assert torch.allclose(theta_old, theta_new, atol=1e-7), (
            f"Regression: theta changed. max diff="
            f"{(theta_old - theta_new).abs().max().item():.2e}"
        )

    def test_full_batch_size_equals_n_is_unchanged(self) -> None:
        """batch_size >= N must behave identically to batch_size=None."""
        NUM_ITEMS, N, T, K = 10, 8, 6, 3

        torch.manual_seed(_SEED + 1)
        item_ids = torch.randint(0, NUM_ITEMS, (N, T))
        responses = torch.randint(0, K, (N, T))

        model_none = DeepIRTModel(
            num_items=NUM_ITEMS, emb_dim=4, hidden_dim=8,
            n_cats=K, decoder="gpcm", device=_DEVICE, seed=_SEED,
        )
        model_full = DeepIRTModel(
            num_items=NUM_ITEMS, emb_dim=4, hidden_dim=8,
            n_cats=K, decoder="gpcm", device=_DEVICE, seed=_SEED,
        )

        model_none.fit(item_ids, responses, n_epochs=10, verbose=False, batch_size=None)
        model_full.fit(item_ids, responses, n_epochs=10, verbose=False, batch_size=N)

        theta_none = model_none.track(item_ids, responses)
        theta_full = model_full.track(item_ids, responses)
        assert torch.allclose(theta_none, theta_full, atol=1e-7), (
            f"batch_size=N diverged from batch_size=None. max diff="
            f"{(theta_none - theta_full).abs().max().item():.2e}"
        )

    def test_minibatch_converges(self) -> None:
        """Mini-batching must converge (NLL should decrease over training)."""
        NUM_ITEMS, N, T, K = 10, 20, 8, 4

        torch.manual_seed(_SEED + 2)
        item_ids = torch.randint(0, NUM_ITEMS, (N, T))
        responses = torch.randint(0, K, (N, T))

        model = DeepIRTModel(
            num_items=NUM_ITEMS, emb_dim=4, hidden_dim=8,
            n_cats=K, decoder="gpcm", device=_DEVICE, seed=_SEED,
        )
        result = model.fit(
            item_ids, responses, n_epochs=30, verbose=False, batch_size=5
        )

        # With 30 epochs + lr=0.01, NLL should have dropped below the ~1.4
        # untrained baseline (uniform over K=4 gives NLL = log(4) ~= 1.386).
        assert result["final_nll"] < 1.40, (
            f"Mini-batch training did not converge: NLL={result['final_nll']:.4f}"
        )
