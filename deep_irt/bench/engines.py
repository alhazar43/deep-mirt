"""
engines.py -- uniform fit / predict / recover wrappers for the two deep-IRT engines.

Both engines see the SAME BenchDataset (same learners, same train/val split,
same ground truth) and are driven with a comparable budget (same epochs, same
Adam lr).  Each wrapper exposes:

    fit()                          -> trains on the train learners
    predict_heldout()             -> (probs (M,K), targets (M,)) on the held-out
                                     tail positions of the VAL learners
    recover()                     -> dict with a (Q,), b (Q,K-1), theta_final (N,)
                                     and (dynamic) theta_traj (N,T)
    cost()                        -> dict: n_params, train_time, infer_time

ENGINE A -- "deep_irt" (ours / vanilla):
    LSTMEncoder (plain nn.LSTM) + deep_irt GPCM / Binary / NRM decoder.
    Item ids 0-indexed.  theta is the per-step scalar from the encoder.

ENGINE B -- "ma-irt" (the PI's richer Chapter-0 model), FROZEN, run not edited:
    one of {dkvmn (MAGPCM), lstm_gpcm, transformer_gpcm} encoder, each paired
    with ma-irt's OWN GPCM decoder + IRTParameterExtractor.  Item ids 1-indexed.
    Recovery uses ma-irt's published convention: per-item alpha averaged over all
    (learner, step) occurrences; beta read per item; theta the per-step output.

The encoder-axis swap inside ma-irt (dkvmn vs lstm_gpcm vs transformer_gpcm) is a
clean controlled comparison: identical decoder, identical IRT head, only the
sequence encoder changes -- so it isolates exactly what the DKVMN memory buys
over a plain recurrent / attention encoder.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from deep_irt.core.model import DeepIRTModel


# ===========================================================================
# ENGINE A: deep_irt (ours / vanilla LSTM + deep_irt decoders)
# ===========================================================================

class DeepIRTEngine:
    """Plain-LSTM deep_irt engine with a swappable deep_irt decoder.

    The organic single-pass engine.  Two orthogonal levers:

      ``separate_theta`` -- theta source.  False (default) = the direct
        responsive stream over ``[item_emb(q_t), resp_emb(r_t)]`` (the reported
        dynamic track).  True = the item-blind stream (ma-irt's read-before-write
        ability pass).  Both obey the single-shift alignment contract.

      ``state_alpha`` -- a clean optional DECODER feature (gpcm/binary).  When
        True the discrimination is read from the prediction-aligned encoder
        state plus the item embedding (ma-irt's IRTParameterExtractor recipe)
        and occurrence-averaged per item at recovery.  No second encoder stream:
        one LSTM, one pass, one shift.

    ``emb_dim`` / ``hidden_dim`` are exposed so the engine can be run at a
    capacity matched to the ma-irt reference (a fair equal-params fight).

    The label is suffixed ``/direct`` or ``/separate`` (theta source) with a
    ``+sa`` marker when state_alpha is on.
    """

    def __init__(self, ds, decoder="gpcm", emb_dim=8, hidden_dim=32,
                 separate_theta=False, state_alpha=False, alpha_emb_dim=None,
                 alpha_log_scale=None, encoder="lstm", encoder_kwargs=None,
                 device="cpu", seed=0):
        self.ds = ds
        self.decoder = decoder
        self.separate_theta = separate_theta
        self.state_alpha = state_alpha
        self.alpha_emb_dim = alpha_emb_dim
        self.alpha_log_scale = alpha_log_scale
        self.encoder_kind = encoder
        self.device = torch.device(device)
        self.seed = seed
        variant = "separate" if separate_theta else "direct"
        sa = "+sa" if state_alpha else ""
        de = f"+ae{alpha_emb_dim}" if alpha_emb_dim is not None else ""
        xp = f"+exp{alpha_log_scale}" if alpha_log_scale is not None else ""
        bk = encoder.upper()
        self.label = f"deep_irt-{bk}/{decoder}/{variant}{sa}{de}{xp}"
        self.encoder_name = (
            f"{bk}({variant}{sa}{de}{xp},e{emb_dim}h{hidden_dim})")
        K = ds.cfg.n_cats
        self.model = DeepIRTModel(
            num_items=ds.cfg.n_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
            n_cats=K, decoder=decoder, separate_theta=separate_theta,
            state_alpha=state_alpha, alpha_emb_dim=alpha_emb_dim,
            alpha_log_scale=alpha_log_scale, decouple=False, encoder=encoder,
            encoder_kwargs=encoder_kwargs,
            device=self.device, seed=seed,
        )
        self._train_time = None

    def _tensor(self, idx):
        it = torch.tensor(self.ds.items0[idx], dtype=torch.long)
        rp = torch.tensor(self.ds.responses[idx], dtype=torch.long)
        return it, rp

    def fit(self, n_epochs=120, lr=1e-2, batch_size=None):
        it, rp = self._tensor(self.ds.train_idx)
        t0 = time.time()
        self.model.fit(it, rp, n_epochs=n_epochs, lr=lr, verbose=False,
                       batch_size=batch_size)
        self._train_time = time.time() - t0
        return self

    @torch.no_grad()
    def predict_heldout(self):
        """probs (M,K) and targets (M,) on the val learners' held-out tail."""
        K = self.ds.cfg.n_cats
        T = self.ds.cfg.seq_len
        h = self.ds.cfg.n_holdout
        it, rp = self._tensor(self.ds.val_idx)
        it = it.to(self.device); rp = rp.to(self.device)
        embs = self.model.encoder.item_emb(it)                     # (B,T,emb)

        if self.state_alpha:
            # ONE pass -> aligned theta (for the likelihood) and the alpha state.
            theta_in, state_in = self.model.encoder.aligned_theta_and_state(it, rp)
            # Decoupled wide alpha key (separate item table; not the LSTM input).
            if self.alpha_emb_dim is not None:
                alpha_embs = self.model.encoder.alpha_item_emb(it)      # (B,T,ae)
            else:
                alpha_embs = None
        else:
            # theta_in[t] = ability after history 0..t-1 -> predicts response[t]
            theta_in = self.model.encoder.theta_for_prediction(it, rp)  # (B,T)
            state_in = None
            alpha_embs = None

        probs_full = []
        for t in range(T):
            state_t = state_in[:, t, :] if state_in is not None else None
            alpha_t = alpha_embs[:, t, :] if alpha_embs is not None else None
            params = self.model.decoder.item_params(
                embs[:, t, :], state=state_t, alpha_emb=alpha_t)
            lp = self.model.decoder.log_probs(theta_in[:, t], **params)  # (B,K)
            probs_full.append(lp.exp().unsqueeze(1))
        probs_full = torch.cat(probs_full, dim=1)                  # (B,T,K)
        probs_tail = probs_full[:, T - h:, :].reshape(-1, K).cpu().numpy()
        targ_tail = rp[:, T - h:].reshape(-1).cpu().numpy()
        self._infer_probs = probs_full                            # cache for timing
        return probs_tail, targ_tail

    @torch.no_grad()
    def recover(self):
        if self.state_alpha:
            # State-conditioned alpha, occurrence-averaged over all sequences.
            it_all = torch.tensor(self.ds.items0, dtype=torch.long,
                                  device=self.device)
            rp_all = torch.tensor(self.ds.responses, dtype=torch.long,
                                  device=self.device)
            rec = self.model.recover_item_params(it_all, rp_all)
            theta_traj = self.model.track(it_all, rp_all).cpu().numpy()  # (N,T)
            return {"a": rec["a"], "b": rec["b"],
                    "theta_final": theta_traj[:, -1], "theta_traj": theta_traj,
                    "seen": rec["seen"]}

        params = self.model.recover_item_params()                 # all Q items
        if self.decoder == "nrm":
            # NRM exposes per-option a (Q,K) and a correct-option location beta
            # (Q,). It carries no ordinal step-thresholds, so item-param recovery
            # against the GPCM ground truth is not defined; run_bench scores NRM
            # on prediction + theta only and marks a/b as N/A.
            a_hat = params["a"]                                  # (Q,K)
            b_hat = params.get("beta")                           # (Q,)
        else:
            a_hat = params["a"]                                  # (Q,)
            b_hat = params["b"]                                  # (Q,K-1)
        # theta per learner: final-step theta over the FULL sequence
        it = torch.tensor(self.ds.items0, dtype=torch.long, device=self.device)
        rp = torch.tensor(self.ds.responses, dtype=torch.long, device=self.device)
        theta_traj = self.model.track(it, rp).cpu().numpy()       # (N,T)
        theta_final = theta_traj[:, -1]
        return {"a": a_hat, "b": b_hat, "theta_final": theta_final,
                "theta_traj": theta_traj}

    def cost(self):
        n = sum(p.numel() for p in self.model.encoder.parameters()) + \
            sum(p.numel() for p in self.model.decoder.parameters())
        return {"n_params": int(n), "train_time": self._train_time}


# ===========================================================================
# ENGINE B: ma-irt (DKVMN / LSTM / Transformer encoder + ma-irt GPCM decoder)
# ===========================================================================

class MaIrtEngine:
    """ma-irt deep-ordinal-IRT engine (frozen code, run not edited).

    encoder in {"dkvmn", "lstm_gpcm", "transformer_gpcm"}.  All three pair with
    ma-irt's GPCM decoder + IRTParameterExtractor, so the decoder is held fixed
    across the encoder-axis sweep.
    """

    _BUILDERS = {}   # filled lazily to avoid import at module load

    def __init__(self, ds, encoder="dkvmn", device="cpu", seed=0,
                 separate_theta=False, **model_kwargs):
        self.ds = ds
        self.encoder = encoder
        self.device = torch.device(device)
        self.seed = seed
        self.label = f"ma-irt-{encoder}/gpcm"
        self.encoder_name = encoder
        self._train_time = None
        K = ds.cfg.n_cats
        Q = ds.cfg.n_items

        torch.manual_seed(seed)
        model = self._build(encoder, Q, K, separate_theta, model_kwargs)
        self.model = model.to(self.device)

    @staticmethod
    def _build(encoder, Q, K, separate_theta, kw):
        # Imported here so the module imports cleanly even if ma-irt is absent.
        from models.magpcm import MAGPCM
        from models.lstm_gpcm import LSTMGPCM
        from models.transformer_gpcm import TransformerGPCM
        if encoder == "dkvmn":
            return MAGPCM(n_questions=Q, n_categories=K,
                          memory_size=kw.get("memory_size", 20),
                          key_dim=kw.get("key_dim", 32),
                          value_dim=kw.get("value_dim", 32),
                          summary_dim=kw.get("summary_dim", 32),
                          separate_theta=separate_theta,
                          item_conditioned=kw.get("item_conditioned", False))
        if encoder == "lstm_gpcm":
            return LSTMGPCM(n_questions=Q, n_categories=K,
                            d_model=kw.get("d_model", 32),
                            key_dim=kw.get("key_dim", 32),
                            value_dim=kw.get("value_dim", 32),
                            dropout_rate=kw.get("dropout_rate", 0.0),
                            separate_theta=separate_theta)
        if encoder == "transformer_gpcm":
            return TransformerGPCM(n_questions=Q, n_categories=K,
                                   d_model=kw.get("d_model", 32),
                                   n_heads=kw.get("n_heads", 4),
                                   n_layers=kw.get("n_layers", 2),
                                   d_ff=kw.get("d_ff", 64),
                                   max_seq_len=kw.get("max_seq_len", 256),
                                   key_dim=kw.get("key_dim", 32),
                                   value_dim=kw.get("value_dim", 32),
                                   dropout_rate=kw.get("dropout_rate", 0.0),
                                   separate_theta=separate_theta)
        raise ValueError(f"unknown ma-irt encoder {encoder!r}")

    def _q1(self, idx):
        """1-indexed questions + responses tensors for a learner subset."""
        q = torch.tensor(self.ds.items0[idx] + 1, dtype=torch.long)
        r = torch.tensor(self.ds.responses[idx], dtype=torch.long)
        return q, r

    def fit(self, n_epochs=120, lr=1e-2, batch_size=None):
        """Train to minimise mean cross-entropy over ALL positions of the train
        learners (the same objective deep_irt's fit minimises: per-token NLL).
        Same epochs and Adam lr as the deep_irt engine for a fair budget."""
        q, r = self._q1(self.ds.train_idx)
        q = q.to(self.device); r = r.to(self.device)
        N = q.size(0)
        bs = N if (batch_size is None or batch_size >= N) else batch_size
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(self.seed + 1)

        t0 = time.time()
        self.model.train()
        for _ in range(n_epochs):
            if bs >= N:
                opt.zero_grad()
                out = self.model(q, r)
                loss = self._nll(out["logits"], r)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
            else:
                perm = torch.randperm(N, generator=gen)
                for s in range(0, N, bs):
                    bi = perm[s:s + bs]
                    opt.zero_grad()
                    out = self.model(q[bi], r[bi])
                    loss = self._nll(out["logits"], r[bi])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    opt.step()
        self._train_time = time.time() - t0
        return self

    @staticmethod
    def _nll(logits, responses):
        B, S, K = logits.shape
        return F.cross_entropy(logits.reshape(B * S, K), responses.reshape(B * S))

    @torch.no_grad()
    def predict_heldout(self):
        """ma-irt predicts response[t] from history 0..t-1 (it reads memory
        before writing step t), so probs[:, t] is the next-step prediction for
        responses[:, t].  Same contract as the deep_irt engine."""
        K = self.ds.cfg.n_cats
        T = self.ds.cfg.seq_len
        h = self.ds.cfg.n_holdout
        q, r = self._q1(self.ds.val_idx)
        q = q.to(self.device); r = r.to(self.device)
        self.model.eval()
        out = self.model(q, r)
        probs = out["probs"]                                       # (B,T,K)
        probs_tail = probs[:, T - h:, :].reshape(-1, K).cpu().numpy()
        targ_tail = r[:, T - h:].reshape(-1).cpu().numpy()
        return probs_tail, targ_tail

    @torch.no_grad()
    def recover(self):
        """ma-irt recovery convention: per-item alpha averaged over all
        occurrences; beta read per item (constant given the item embedding);
        theta per-step trajectory."""
        Q = self.ds.cfg.n_items
        K = self.ds.cfg.n_cats
        q_all = torch.tensor(self.ds.items0 + 1, dtype=torch.long, device=self.device)
        r_all = torch.tensor(self.ds.responses, dtype=torch.long, device=self.device)
        self.model.eval()
        out = self.model(q_all, r_all)
        alpha = out["alpha"].squeeze(-1).cpu().numpy()             # (N,T)
        beta = out["beta"].cpu().numpy()                           # (N,T,K-1)
        theta = out["theta"].squeeze(-1).cpu().numpy()             # (N,T)
        items = self.ds.items0                                     # (N,T) 0-indexed

        a_sum = np.zeros(Q); a_cnt = np.zeros(Q)
        b_sum = np.zeros((Q, K - 1)); b_cnt = np.zeros(Q)
        flat_items = items.ravel()
        flat_alpha = alpha.ravel()
        flat_beta = beta.reshape(-1, K - 1)
        np.add.at(a_sum, flat_items, flat_alpha)
        np.add.at(a_cnt, flat_items, 1.0)
        np.add.at(b_sum, flat_items, flat_beta)
        np.add.at(b_cnt, flat_items, 1.0)
        a_hat = a_sum / np.maximum(a_cnt, 1.0)
        b_hat = b_sum / np.maximum(b_cnt[:, None], 1.0)
        seen = a_cnt > 0

        return {"a": a_hat, "b": b_hat, "theta_final": theta[:, -1],
                "theta_traj": theta, "seen": seen}

    def cost(self):
        n = sum(p.numel() for p in self.model.parameters())
        return {"n_params": int(n), "train_time": self._train_time}
