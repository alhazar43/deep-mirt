"""_qm3_model.py -- blueprint model class (docs/qmirt_blueprint.md v1.1).

EMISSION (review-specified): eta_j(z) = sum_c Q[j,c] A[j,c] z_c;
category-k logit = k*eta - sum_{i<k} b[j,i]. Vector loadings on the Q
support, scalar composite-scale thresholds. Nests the qm2/Chapter-0
unidimensional GPCM exactly via A[j, c_j] = alpha_j, b[j,i] = alpha_j *
d[j,i] (converter provided; Gate A runs qm2 beds through this emission).

TRANSITION (mechanism; identification lemmas carried): all change fires on
practice events only; rho = 1, mu = 0 on monotone beds; own growth
(ceil_c - z)+ * lambda_n * gain_c * w_prac; transfer gap/headroom-gated
signed G scaled by gamma_n. Practice weight of a cross-loading item is
SPLIT across its tapped concepts (w_jc = Q_jc / n_loadings(j)); this
fractional-practice-credit rule is a stated design choice and gets its own
stress test before any cross-loading claim (it is an attribution channel).

PERSON PARAMETERS: z0 (N,C); lambda_n = exp(u_n - mean(u)) and gamma_n
likewise, so E-type constraints (geometric-mean one) hold by construction
(the blueprint's multiplicative-gauge resolution). Estimated either as
free parameters (encoder OFF; Gate A) or amortized by the LSTM recognition
network (encoder ON; Gate B): the encoder reads (item, response) pairs of
at most the first `enc_window` steps and outputs z0, u_lambda, u_gamma.
The encoder NEVER touches the state rollout; it only proposes person
parameters. Whether that channel fabricates transfer is exactly Gate B.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bank_from_qm2(items):
    """Convert a qm2 single-tagged bank (alpha, d on theta-d scale) to the
    vector-loading composite-scale form. Exact nesting."""
    J, K = items["J"], items["K"]
    C = int(items["concept"].max()) + 1
    A = np.zeros((J, C))
    A[np.arange(J), items["concept"]] = items["alpha"]
    b = items["alpha"][:, None] * items["d"]
    Q = (A > 0).astype(float)
    return {"A": A, "Q": Q, "b": b, "K": K, "J": J, "C": C,
            "primary": items["concept"],
            "pure": np.ones(J, dtype=bool)}


class QM3Model(nn.Module):
    def __init__(self, bank, N, inert_items=None, use_encoder=False,
                 enc_window=None, enc_hidden=32, enc_emb=16,
                 enc_gamma=True, enc_lambda=True, gain_features=False,
                 init_gain=0.05, frac_practice=True):
        super().__init__()
        J, C, K = bank["J"], bank["C"], bank["K"]
        self.J, self.C, self.K, self.N = J, C, K, N
        self.use_encoder = use_encoder
        self.enc_window = enc_window
        # frozen bank (registered as buffers; never trained here)
        self.register_buffer("A", torch.as_tensor(bank["A"],
                                                  dtype=torch.float))
        self.register_buffer("bth", torch.as_tensor(bank["b"],
                                                    dtype=torch.float))
        self.register_buffer("Qm", torch.as_tensor(bank["Q"],
                                                   dtype=torch.float))
        w = bank["Q"] / np.maximum(bank["Q"].sum(1, keepdims=True), 1.0) \
            if frac_practice else bank["Q"]
        if inert_items is not None:
            w = w * (~np.asarray(inert_items, dtype=bool))[:, None]
        self.register_buffer("Wprac", torch.as_tensor(w,
                                                      dtype=torch.float))
        # mechanism
        self.gain_raw = nn.Parameter(torch.full(
            (C,), float(np.log(np.expm1(init_gain)))))
        self.ceil = nn.Parameter(torch.full((C,), 2.0))
        self.floor = nn.Parameter(torch.full((C,), -2.0))
        self.G_raw = nn.Parameter(torch.zeros(C, C))
        self.register_buffer("G_mask", 1.0 - torch.eye(C))
        # person parameters (free-parameter path). Init carries small seed
        # noise so the model-seed replication dimension is REAL (review
        # finding 2026-07-05: deterministic zero inits + full-batch Adam
        # made model seeds near-copies, in qm2 too).
        self.z0_free = nn.Parameter(0.05 * torch.randn(N, C))
        self.ul_free = nn.Parameter(0.02 * torch.randn(N))
        self.ug_free = nn.Parameter(0.02 * torch.randn(N))
        # recognition network (amortized path)
        self.enc_gamma, self.enc_lambda = enc_gamma, enc_lambda
        # V2 event-conditioned gain (pre-registered form): the multiplier
        # exp(w_out*(y/(K-1)-0.5) + w_feat*(f_j - fbar)) scales the OWN
        # gain of the event that produced outcome y; responses touch the
        # own-concept transition only, never the cross-concept route.
        self.gain_features = bool(gain_features)
        feat = bank.get("feat")
        if feat is None:
            feat = np.zeros(J)
        prac_rows = np.asarray(bank.get("kind", ["x"] * J)) == "apool"
        fbar = float(feat[prac_rows].mean()) if prac_rows.any() else 0.0
        self.register_buffer("feat", torch.as_tensor(feat - fbar,
                                                     dtype=torch.float))
        self.w_out = nn.Parameter(torch.zeros(()))
        self.w_feat = nn.Parameter(torch.zeros(()))
        if self.gain_features:
            # ISOLATION GUARD (2026-07-05 review caveat): the outcome
            # multiplier is only leak-free while practice items are
            # single-concept; a cross-loading practice item would carry
            # forecast responses into a certified concept's state.
            live = self.Wprac.sum(1) > 0
            assert float((self.Wprac[live] > 0).sum(1).max()) <= 1.0, \
                "gain_features requires single-concept practice items"
        if use_encoder:
            self.item_emb = nn.Embedding(J, enc_emb)
            self.enc = nn.LSTM(enc_emb + K, enc_hidden, batch_first=True)
            self.head = nn.Linear(enc_hidden, C + 2)

    # ------------------------------------------------------------ persons
    def person_params(self, seq_eff=None, resp=None):
        # FULL-COHORT INVARIANT: the geometric-mean-1 gauges and the z0
        # row indexing assume every call passes the complete fixed-order
        # cohort; person-minibatching would silently break both.
        if not self.use_encoder:
            z0, ul, ug = self.z0_free, self.ul_free, self.ug_free
        else:
            W = self.enc_window or seq_eff.shape[1]
            x = torch.cat([self.item_emb(seq_eff[:, :W]),
                           F.one_hot(resp[:, :W],
                                     num_classes=self.K).float()], dim=-1)
            h, _ = self.enc(x)
            out = self.head(h[:, -1])
            z0, ul, ug = out[:, :self.C], out[:, self.C], out[:, self.C + 1]
            if not self.enc_lambda:
                ul = torch.zeros_like(ul)        # lambda pinned at 1
            if not self.enc_gamma:
                ug = torch.zeros_like(ug)        # gamma pinned at 1
                # (Gate B2: the transfer-ability wire is cut)
        lam = torch.exp(ul - ul.mean())          # geometric-mean-1
        gam = torch.exp(ug - ug.mean())
        return z0, lam, gam

    @property
    def G_signed(self):
        return self.G_raw * self.G_mask

    # ------------------------------------------------------------ rollout
    def unroll(self, seq_eff, resp=None, g_off_from=None, g_zero=False):
        N, T = seq_eff.shape
        z0, lam, gam = self.person_params(seq_eff, resp)
        gain = F.softplus(self.gain_raw)
        G = torch.zeros_like(self.G_signed) if g_zero else self.G_signed
        z = [z0[:N]]
        for t in range(T - 1):
            Gt = G
            if g_off_from is not None and t >= g_off_from:
                Gt = torch.zeros_like(G)
            zt = z[-1]
            prac = self.Wprac[seq_eff[:, t]]                  # (N, C)
            gap_up = (self.ceil - zt).clamp_min(0.0)
            gap_dn = (zt - self.floor).clamp_min(0.0)
            own = lam.unsqueeze(1) * gain * prac
            if self.gain_features and resp is not None:
                arg = (self.w_out * (resp[:, t].float()
                                     / (self.K - 1) - 0.5)
                       + self.w_feat * self.feat[seq_eff[:, t]])
                mult = torch.exp(arg.clamp(-8.0, 8.0))   # overflow guard
                own = own * mult.unsqueeze(1)
            x = own + gam.unsqueeze(1) * (prac @ Gt.T)
            grow = gap_up * x.clamp_min(0.0) - gap_dn * (-x).clamp_min(0.0)
            z.append(zt + grow)
        return torch.stack(z, dim=1)

    def logits(self, seq_eff, z):
        Aj = self.A[seq_eff]                                  # (N,T,C)
        eta = (Aj * z).sum(-1)                                # (N,T)
        steps = eta.unsqueeze(-1) - self.bth[seq_eff]
        csum = torch.cumsum(steps, dim=-1)
        return torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)

    def nll(self, seq_eff, resp, step_mask=None, g_off_from=None,
            g_zero=False):
        z = self.unroll(seq_eff, resp, g_off_from=g_off_from,
                        g_zero=g_zero)
        ll = F.log_softmax(self.logits(seq_eff, z), dim=-1)
        per = -torch.gather(ll, 2, resp.unsqueeze(-1)).squeeze(-1)
        if step_mask is None:
            return per.mean(), per
        m = step_mask.to(per.dtype)
        return (per * m).sum() / m.sum(), per

    # ------------------------------------------------------------ fitting
    def fit_params(self, stage):
        mech = [self.gain_raw, self.ceil, self.floor]
        if self.gain_features:
            mech += [p for p in (self.w_out, self.w_feat)
                     if p.requires_grad]
        person = list(self.enc.parameters()) + [self.head.weight,
                                                self.head.bias,
                                                self.item_emb.weight] \
            if self.use_encoder else [self.z0_free, self.ul_free,
                                      self.ug_free]
        if stage == 1:
            return mech + person
        return [self.G_raw] + mech + person


def fit_two_stage(model, seq_eff, resp, cond_mask, e1=500, e2=300,
                  lr=5e-3, l1_g=3e-3):
    for stage, epochs in ((1, e1), (2, e2)):
        opt = torch.optim.Adam(model.fit_params(stage), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss, _ = model.nll(seq_eff, resp, step_mask=cond_mask)
            if stage == 2:
                loss = loss + l1_g * model.G_signed.abs().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fit_params(stage), 5.0)
            opt.step()
    return model


def isolation_check(device="cpu", tol=1e-6):
    """Carried from qm2: with encoder OFF and G's row zero, an unpracticed
    concept stays flat; the G route moves its target (positive control)."""
    torch.manual_seed(0)
    C, K, T, N, J = 3, 5, 40, 7, 6
    bank = {"A": np.zeros((J, C)), "Q": np.zeros((J, C)),
            "b": np.tile(np.linspace(-.8, .8, K - 1), (J, 1)),
            "K": K, "J": J, "C": C,
            "primary": np.zeros(J, dtype=int),
            "pure": np.ones(J, dtype=bool)}
    bank["A"][:, 0] = 1.0
    bank["Q"][:, 0] = 1.0
    m = QM3Model(bank, N)
    with torch.no_grad():
        m.z0_free.normal_(0, 1.0)
        m.G_raw.zero_()
        m.G_raw[1, 0] = 0.7
    seq = torch.randint(0, J, (N, T))
    z = m.unroll(seq)
    dev2 = float((z[:, :, 2] - m.z0_free[:, 2:3]).abs().max())
    moved1 = float((z[:, -1, 1] - m.z0_free[:, 1]).abs().min())
    z0g = m.unroll(seq, g_zero=True)
    dev1 = float((z0g[:, :, 1] - m.z0_free[:, 1:2]).abs().max())
    dev = max(dev1, dev2)
    assert dev < tol, f"ISOLATION LEAK {dev}"
    assert moved1 > 1e-2, "G route inert"
    return dev


if __name__ == "__main__":
    print("qm3 isolation (asserted):", isolation_check())
    # nesting check: qm2 bank through qm3 emission reproduces probabilities
    import _qm2_datagen as q2
    rng = np.random.default_rng(0)
    items = q2.make_items(rng)
    bank = bank_from_qm2(items)
    z = rng.normal(0, 1, size=(5, 3))
    for j in (0, 13, 30):
        c = items["concept"][j]
        p2 = q2.gpcm_probs(items["alpha"][j], items["d"][j], z[:, c])
        from _qm3_datagen import mgpcm_probs
        p3 = mgpcm_probs(bank["A"][j], bank["b"][j], z)
        assert np.allclose(p2, p3, atol=1e-10), (j, p2, p3)
    print("nesting check: qm2 emission == qm3 emission on converted bank")
