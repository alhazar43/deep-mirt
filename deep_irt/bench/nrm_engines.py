"""nrm_engines.py -- uniform fit / predict / recover wrappers for the NRM bench.

Two engines, head-to-head on the SAME NRM ground truth and the SAME train/val
split, driven with the same budget (epochs, Adam lr):

ENGINE A -- "deep_irt-NRM":
    plain LSTMEncoder + deep_irt ``NRMDecoder`` (Bock 1972).  Per-option slopes
    a_k and intercepts c_k are ITEM-ONLY linear maps of the item embedding.
    Item ids 0-indexed.  theta is the per-step scalar from the encoder.

ENGINE B -- "ma-irt-NRM":
    ma-irt encoder ({lstm, dkvmn}, separated ability pathway) + the bench's
    ``MaIrtNRM`` head.  theta is read from the item-BLIND student summary; the
    per-option slopes a_k are STATE-CONDITIONED on [joint_summary, item_embed]
    (the ma-irt edge this bench tests).  Item ids 1-indexed (0 = padding).

Both expose:
    fit()              -> trains on the train learners (per-token NLL).
    predict_heldout()  -> (probs (M,K), targets (M,)) on the held-out tail of
                          the VAL learners.
    recover()          -> dict with a (Q,K), c (Q,K), beta (Q,), theta_final (N,),
                          theta_traj (N,T), and (ma-irt) a `seen` mask.
    cost()             -> dict: n_params, train_time.

Prediction alignment (matches the GPCM bench)
---------------------------------------------
Both engines predict response[t] from history 0..t-1 plus the known item id at
position t.  deep_irt uses ``shifted_theta`` (theta_{t-1} predicts response[t])
with the current item embedding.  The ma-irt encoder right-shifts its value
stream internally (interaction_start token), so ``probs[:, t]`` is already the
next-step prediction for ``responses[:, t]`` -- same contract, no extra shift.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from deep_irt.core.model import DeepIRTModel
from deep_irt.core.nrm_ma_irt import MaIrtNRM


# ===========================================================================
# ENGINE A: deep_irt-NRM (plain LSTM + deep_irt item-only NRM decoder)
# ===========================================================================

class DeepIRTNRMEngine:
    def __init__(self, ds, emb_dim=8, hidden_dim=32, device="cpu", seed=0):
        self.ds = ds
        self.device = torch.device(device)
        self.seed = seed
        self.label = "deep_irt-LSTM/nrm"
        self.encoder_name = "LSTM(plain)"
        self.decoder = "nrm"
        K = ds.cfg.n_options
        self.model = DeepIRTModel(
            num_items=ds.cfg.n_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
            n_cats=K, decoder="nrm", correct_option=ds.cfg.correct_option,
            device=self.device, seed=seed,
        )
        self._train_time = None

    def _tensor(self, idx):
        it = torch.tensor(self.ds.items0[idx], dtype=torch.long)
        rp = torch.tensor(self.ds.responses[idx], dtype=torch.long)
        return it, rp

    def fit(self, n_epochs=150, lr=1e-2, batch_size=None):
        it, rp = self._tensor(self.ds.train_idx)
        t0 = time.time()
        self.model.fit(it, rp, n_epochs=n_epochs, lr=lr, verbose=False,
                       batch_size=batch_size)
        self._train_time = time.time() - t0
        return self

    @torch.no_grad()
    def predict_heldout(self):
        K = self.ds.cfg.n_options
        T = self.ds.cfg.seq_len
        h = self.ds.cfg.n_holdout
        it, rp = self._tensor(self.ds.val_idx)
        it = it.to(self.device); rp = rp.to(self.device)
        theta_in = self.model.encoder.shifted_theta(it, rp)        # (B,T)
        embs = self.model.encoder.item_emb(it)                     # (B,T,emb)
        probs_full = []
        for t in range(T):
            params = self.model.decoder.item_params(embs[:, t, :])
            lp = self.model.decoder.log_probs(theta_in[:, t], **params)  # (B,K)
            probs_full.append(lp.exp().unsqueeze(1))
        probs_full = torch.cat(probs_full, dim=1)                  # (B,T,K)
        probs_tail = probs_full[:, T - h:, :].reshape(-1, K).cpu().numpy()
        targ_tail = rp[:, T - h:].reshape(-1).cpu().numpy()
        return probs_tail, targ_tail

    @torch.no_grad()
    def recover(self):
        params = self.model.recover_item_params()                 # all Q items
        a_hat = params["a"]                                        # (Q,K)
        c_hat = params["c"]                                        # (Q,K)
        beta_hat = params["beta"]                                  # (Q,)
        it = torch.tensor(self.ds.items0, dtype=torch.long, device=self.device)
        rp = torch.tensor(self.ds.responses, dtype=torch.long, device=self.device)
        theta_traj = self.model.track(it, rp).cpu().numpy()        # (N,T)
        return {"a": a_hat, "c": c_hat, "beta": beta_hat,
                "theta_final": theta_traj[:, -1], "theta_traj": theta_traj}

    def cost(self):
        n = sum(p.numel() for p in self.model.encoder.parameters()) + \
            sum(p.numel() for p in self.model.decoder.parameters())
        return {"n_params": int(n), "train_time": self._train_time}


# ===========================================================================
# ENGINE B: ma-irt-NRM (ma-irt encoder + state-conditioned NRM head)
# ===========================================================================

class MaIrtNRMEngine:
    def __init__(self, ds, encoder="lstm", device="cpu", seed=0,
                 item_conditioned=False, slopes_mode="state_conditioned",
                 **model_kwargs):
        self.ds = ds
        self.encoder = encoder
        self.device = torch.device(device)
        self.seed = seed
        self.slopes_mode = slopes_mode
        # item_only slopes get a distinct label so the bench table separates
        # the two ma-irt NRM heads (state-conditioned vs item-only a_k).
        suffix = "/nrm-itemonly" if slopes_mode == "item_only" else "/nrm"
        self.label = f"ma-irt-{encoder}{suffix}"
        self.encoder_name = encoder
        self.decoder = "nrm"
        self._train_time = None
        K = ds.cfg.n_options
        Q = ds.cfg.n_items

        torch.manual_seed(seed)
        self.model = MaIrtNRM(
            n_questions=Q, n_options=K, encoder=encoder,
            correct_option=ds.cfg.correct_option,
            item_conditioned=item_conditioned, slopes_mode=slopes_mode,
            **model_kwargs,
        ).to(self.device)

    def _q1(self, idx):
        q = torch.tensor(self.ds.items0[idx] + 1, dtype=torch.long)   # 1-indexed
        r = torch.tensor(self.ds.responses[idx], dtype=torch.long)
        return q, r

    @staticmethod
    def _nll(logits, responses):
        B, S, K = logits.shape
        return F.cross_entropy(logits.reshape(B * S, K), responses.reshape(B * S))

    def fit(self, n_epochs=150, lr=1e-2, batch_size=None):
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

    @torch.no_grad()
    def predict_heldout(self):
        K = self.ds.cfg.n_options
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
        """ma-irt recovery convention: per-item a_k / c_k averaged over all
        (learner, step) occurrences (they are per-step due to state-conditioning),
        then re-centered across options and beta = -c_corr/a_corr computed from
        the per-item averages.  theta per-step trajectory.
        """
        Q = self.ds.cfg.n_items
        K = self.ds.cfg.n_options
        corr = self.ds.cfg.correct_option
        q_all = torch.tensor(self.ds.items0 + 1, dtype=torch.long, device=self.device)
        r_all = torch.tensor(self.ds.responses, dtype=torch.long, device=self.device)
        self.model.eval()
        out = self.model(q_all, r_all)
        a = out["a"].cpu().numpy()                                 # (N,T,K)
        c = out["c"].cpu().numpy()                                 # (N,T,K)
        theta = out["theta"].squeeze(-1).cpu().numpy()             # (N,T)
        items = self.ds.items0                                     # (N,T) 0-indexed

        flat_items = items.ravel()
        a_flat = a.reshape(-1, K)
        c_flat = c.reshape(-1, K)
        a_sum = np.zeros((Q, K)); c_sum = np.zeros((Q, K)); cnt = np.zeros(Q)
        np.add.at(a_sum, flat_items, a_flat)
        np.add.at(c_sum, flat_items, c_flat)
        np.add.at(cnt, flat_items, 1.0)
        denom = np.maximum(cnt, 1.0)[:, None]
        a_hat = a_sum / denom                                      # (Q,K)
        c_hat = c_sum / denom
        # Re-center the per-item averages (averaging preserves sum-to-zero, but
        # re-center defensively so the Bock representative is exact).
        a_hat = a_hat - a_hat.mean(axis=1, keepdims=True)
        c_hat = c_hat - c_hat.mean(axis=1, keepdims=True)
        # Correct-option location beta = -c_corr / a_corr (a_floor like deep_irt).
        a_corr = a_hat[:, corr]
        c_corr = c_hat[:, corr]
        a_floor = 0.1
        denom_b = np.sign(a_corr) * np.maximum(np.abs(a_corr), a_floor)
        denom_b = np.where(denom_b == 0, a_floor, denom_b)
        beta_hat = -c_corr / denom_b
        seen = cnt > 0

        return {"a": a_hat, "c": c_hat, "beta": beta_hat,
                "theta_final": theta[:, -1], "theta_traj": theta, "seen": seen}

    def cost(self):
        n = sum(p.numel() for p in self.model.parameters())
        return {"n_params": int(n), "train_time": self._train_time}
