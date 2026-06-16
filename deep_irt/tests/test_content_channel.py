"""Tests for the Strand 3 content channel.

Contracts:

1. ContentItemEmbedding is a drop-in for nn.Embedding: callable
   (item_ids -> (..., emb_dim)) and exposes .weight (full table) +
   .num_embeddings, so it can replace encoder.item_emb without touching the core.
2. mode="content" has NO free per-item rows (only the projection trains); the
   .weight table is a pure function of frozen features + projection.
3. cold_start_repr computes a NEW item's representation from text ALONE
   (no responses, no trained row) and matches forward() on in-bank items.
4. Swapping the content channel into a DeepIRTModel leaves fit/track working;
   the content projection receives gradient (the channel actually trains).
5. mode="concat" adds a zero-initialised ID residual, so a fresh concat model
   equals the content-only forward at init, and cold-start ignores the residual.
6. The synthetic generator + cold-start recover a content-determined difficulty
   well in the POSITIVE control and collapse under the NEGATIVE (shuffled)
   control -- the mechanism proof in miniature.
"""

from __future__ import annotations

import numpy as np
import torch

from deep_irt.core.model import DeepIRTModel
from deep_irt.strand3_content.content_channel import ContentItemEmbedding
from deep_irt.strand3_content.content_model import ContentModel
from deep_irt.strand3_content import synthetic as syn


_DEVICE = torch.device("cpu")


def _feats(num_items=12, text_dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_items, text_dim, generator=g)


# ---------------------------------------------------------------------------
# Contract 1 + 2: drop-in interface, no free rows in content mode
# ---------------------------------------------------------------------------

def test_dropin_interface_and_shapes():
    feats = _feats()
    ch = ContentItemEmbedding(feats, emb_dim=4, mode="content")
    assert ch.num_embeddings == 12
    ids = torch.tensor([0, 5, 11])
    rep = ch(ids)
    assert rep.shape == (3, 4)
    # .weight is the full table and matches forward on all ids
    w = ch.weight
    assert w.shape == (12, 4)
    assert torch.allclose(w[ids], rep)


def test_content_mode_has_no_id_rows():
    feats = _feats()
    ch = ContentItemEmbedding(feats, emb_dim=4, mode="content")
    pnames = {n for n, _ in ch.named_parameters()}
    # Only the projection trains; no id_emb parameter.
    assert any(n.startswith("proj") for n in pnames)
    assert not any("id_emb" in n for n in pnames)
    # text_features is a buffer, not a parameter.
    assert "text_features" in dict(ch.named_buffers())


# ---------------------------------------------------------------------------
# Contract 3: cold-start from text alone
# ---------------------------------------------------------------------------

def test_cold_start_matches_forward_on_text():
    feats = _feats()
    ch = ContentItemEmbedding(feats, emb_dim=4, mode="content")
    # cold_start_repr on the SAME features must equal forward on those ids.
    ids = torch.tensor([1, 2, 3])
    fwd = ch(ids)
    cold = ch.cold_start_repr(feats[ids])
    assert torch.allclose(fwd, cold, atol=1e-6)

    # A brand-new feature vector (no row in the table) still yields a repr.
    new = torch.randn(2, ch.text_dim)
    rep_new = ch.cold_start_repr(new)
    assert rep_new.shape == (2, 4)


# ---------------------------------------------------------------------------
# Contract 4: swap into DeepIRTModel, trains, projection gets gradient
# ---------------------------------------------------------------------------

def test_content_model_trains_and_projection_gets_grad():
    data = syn.generate(n_items=20, n_learners=60, seq_len=20, text_dim=8,
                        K=3, seed=0)
    feats = torch.from_numpy(data.content)
    cm = ContentModel(text_features=feats, emb_dim=8, hidden_dim=16, n_cats=3,
                      mode="content", state_alpha=True, device=_DEVICE, seed=0)
    it = torch.from_numpy(data.item_ids)
    rp = torch.from_numpy(data.responses)
    info = cm.fit(it, rp, n_epochs=10, lr=0.02, verbose=False)
    assert np.isfinite(info["final_nll"])
    # The projection must carry gradient after a manual backward.
    cm.model.encoder.zero_grad()
    cm.model.decoder.zero_grad()
    loss = cm.model._compute_loss(it, rp)
    loss.backward()
    g = cm.content_emb.proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ---------------------------------------------------------------------------
# Contract 5: concat residual is zero at init
# ---------------------------------------------------------------------------

def test_concat_residual_zero_at_init():
    feats = _feats()
    ch_c = ContentItemEmbedding(feats, emb_dim=4, mode="content")
    ch_x = ContentItemEmbedding(feats, emb_dim=4, mode="concat")
    # Force the same projection weights so we isolate the residual.
    ch_x.proj.load_state_dict(ch_c.proj.state_dict())
    ids = torch.arange(12)
    assert torch.allclose(ch_c(ids), ch_x(ids), atol=1e-6)
    # cold-start uses text only in both modes.
    assert torch.allclose(
        ch_c.cold_start_repr(feats[:3]), ch_x.cold_start_repr(feats[:3]),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Contract 6: positive recovers, negative collapses (mechanism proof mini)
# ---------------------------------------------------------------------------

def test_positive_recovers_negative_collapses():
    data = syn.generate(n_items=80, n_learners=400, seq_len=30, text_dim=32,
                        K=3, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(data.n_items)
    base = np.sort(perm[:56])
    held = np.sort(perm[56:])

    def coldstart_corr(feats):
        base_g = np.sort(base)
        g2l = {int(g): j for j, g in enumerate(base_g)}
        recs = []
        for n in range(data.item_ids.shape[0]):
            pairs = [(g2l[int(q)], int(r))
                     for q, r in zip(data.item_ids[n], data.responses[n])
                     if int(q) in set(base.tolist())]
            if len(pairs) < 2:
                continue
            lq, lr = zip(*pairs)
            recs.append({
                "student_id": n + 1,
                "questions": torch.tensor(lq, dtype=torch.long) + 1,
                "responses": torch.tensor(lr, dtype=torch.long),
            })
        from deep_irt.core.realdata import collate_adapter_items
        b = collate_adapter_items(recs)
        cm = ContentModel(
            text_features=torch.from_numpy(feats[base_g]),
            emb_dim=16, hidden_dim=32, n_cats=3, mode="content",
            state_alpha=True, device=_DEVICE, seed=0,
        )
        cm.fit(b.item_ids, b.responses, n_epochs=120, lr=0.02, verbose=False,
               mask=b.mask, batch_size=256)
        cold = cm.cold_start_difficulty(torch.from_numpy(feats[held]))
        from scipy.stats import spearmanr
        return abs(float(spearmanr(cold, data.b_true[held])[0]))

    pos = coldstart_corr(data.content)
    neg = coldstart_corr(syn.shuffle_content(data, seed=2))
    assert pos > 0.5, f"positive cold-start too weak: {pos}"
    assert neg < 0.35, f"negative cold-start did not collapse: {neg}"
    assert pos - neg > 0.3, f"pos/neg gap too small: {pos - neg}"
