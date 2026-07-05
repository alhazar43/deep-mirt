"""_qm2_model.py -- simple-structure dynamic GPCM (paper 2 core model).

Fresh build per docs/qmirt_paper_plan.md v1.1 (D1-D4). FB-OFF by design (R9):
responses NEVER enter the state transition; individual differences enter only
through the fitted per-learner initial state z0 (a static person parameter).
The sole cross-concept route is the zero-diagonal signed matrix G.

    z[:, 0]   = z0                                  (N, C), fitted
    gap_t      = relu(ceil - z[:, t])                remaining mastery gap
    z[:, t+1] = z[:, t]
                + (1 - rho) * (mu - z[:, t])         decay to FLOOR mu=0
                + gap_t * ( softplus(gain) * prac_t  own gain, gap-scaled
                          + prac_t @ G_signed.T )    transfer,  gap-scaled

LEMMA 3 (gain-form misfit launders into G; 2026-07-04 diagnostic): with a
CONSTANT per-practice own-gain and decelerating true gains, stage 2 sculpts
the residual out of practice-count bases (fabricated G[B,A] > 0 with
G[B,U] < 0 zigzag on the null twin). The own-gain family must be able to
express deceleration; the mastery-gap scaling (venue-4 M2, sanctioned by
plan v1.1 C2) does, and the same gap gates transfer. ceil_c is fitted but
PRACTICE-GATED (multiplies prac terms only), so Lemma 1 is not violated.

IDENTIFICATION RULES (2026-07-04 pilot findings, D2 amendments):
LEMMA 1 (free asymptote): mu is a FIXED buffer at 0, never fitted. A fitted
free asymptote is an always-on growth channel that (a) absorbs
practice-driven rise (destroys G identification; the null twin fabricates G)
and (b) keeps the no-G forecast arm rising (ablation shows nothing). Every
GROWTH route must be practice-gated.
LEMMA 2 (free persistence): rho is FROZEN at 1 unless the bed identifies it.
Under monotone data, fitted rho < 1 and G > 0 are collinear through the
A-practice-step count (decay-down-per-step + G-up-per-step cancel on the
conditioning window), so the null twin fabricates G as a decay compensator
and the no-G forecast arm spuriously decays. Decay/forgetting is only
identifiable from non-monotone data (dips during absence); the OU/decay
variant belongs to those beds (fit_rho=True), never to the monotone bridge.

    emission at step t: GPCM on the TAGGED concept only (simple structure),
    logits_k = cumsum_j<=k alpha_q * (z[i, t, c(q)] - d[q, j]),  category 0 = 0.
    Response at t is emitted from z[:, t] BEFORE the t -> t+1 update.

Two-stage training (campaign posture, carried):
    stage 1: fit {alpha, d, z0, gain, rho, mu}; G excluded (stays 0).
    stage 2: freeze {alpha, d}; fit {G} (+ dynamics stay trainable) + L1 on G.

Forecast arms: unroll with `g_off_from=T_cond` zeroes G's contribution from
step T_cond on (state at T_cond is the with-G state; venue-1 semantics).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(x: float) -> float:
    return float(np.log(np.expm1(x)))


class SimpleStructureGPCM(nn.Module):
    def __init__(self, J, C, K, N, item_concept, ou=True, fit_rho=False,
                 inert_items=None,
                 init_gain=0.05, init_rho=0.999999, d_spread=1.6):
        super().__init__()
        self.J, self.C, self.K, self.N = J, C, K, N
        self.ou = ou
        self.fit_rho = fit_rho
        self.register_buffer("item_concept",
                             torch.as_tensor(item_concept, dtype=torch.long))
        prac_ok = torch.ones(J)
        if inert_items is not None:
            prac_ok[torch.as_tensor(np.flatnonzero(inert_items))] = 0.0
        self.register_buffer("prac_ok", prac_ok)   # measurement state-inert
        self.log_alpha = nn.Parameter(torch.zeros(J))
        off = torch.linspace(-d_spread / 2, d_spread / 2, K - 1)
        self.d = nn.Parameter(off.repeat(J, 1).clone())
        self.z0 = nn.Parameter(torch.zeros(N, C))
        self.gain_raw = nn.Parameter(
            torch.full((C,), _inv_softplus(init_gain)))
        self.rho_raw = nn.Parameter(
            torch.full((C,), float(np.log(init_rho / (1 - init_rho)))))
        self.register_buffer("mu", torch.zeros(C))   # FIXED floor, never fit
        self.ceil = nn.Parameter(torch.full((C,), 2.0))
        self.floor = nn.Parameter(torch.full((C,), -2.0))
        self.G_raw = nn.Parameter(torch.zeros(C, C))
        self.register_buffer("G_mask",
                             1.0 - torch.eye(C))

    # ------------------------------------------------------------ pieces
    @property
    def G_signed(self):
        return self.G_raw * self.G_mask

    def dyn_params(self):
        """Trainable dynamics per the identification rules."""
        ps = [self.z0, self.gain_raw, self.ceil, self.floor]
        if self.fit_rho:
            ps.append(self.rho_raw)
        return ps

    def _prac(self, seq_eff):
        """(N, T, C) practice indicators; inert (reference) items fire no
        practice."""
        oh = F.one_hot(self.item_concept[seq_eff],
                       num_classes=self.C).float()
        return oh * self.prac_ok[seq_eff].unsqueeze(-1)

    def unroll(self, seq_eff, g_off_from=None, g_zero=False):
        """State trajectory z: (N, T, C). seq_eff: LongTensor (N, T)."""
        N, T = seq_eff.shape
        prac = self._prac(seq_eff)
        rho = torch.sigmoid(self.rho_raw) if self.ou \
            else torch.ones_like(self.rho_raw)
        gain = F.softplus(self.gain_raw)
        G = torch.zeros_like(self.G_signed) if g_zero else self.G_signed
        z = [self.z0[:N]]
        for t in range(T - 1):
            Gt = G
            if g_off_from is not None and t >= g_off_from:
                Gt = torch.zeros_like(G)
            zt = z[-1]
            gap_up = (self.ceil - zt).clamp_min(0.0)
            gap_dn = (zt - self.floor).clamp_min(0.0)
            x = gain * prac[:, t, :] + prac[:, t, :] @ Gt.T
            grow = gap_up * x.clamp_min(0.0) - gap_dn * (-x).clamp_min(0.0)
            znext = zt + (1.0 - rho) * (self.mu - zt) + grow
            z.append(znext)
        return torch.stack(z, dim=1)

    def logits(self, seq_eff, z):
        """GPCM logits (N, T, K) reading the tagged concept only."""
        alpha = torch.exp(self.log_alpha)[seq_eff]            # (N, T)
        d = self.d[seq_eff]                                   # (N, T, K-1)
        c = self.item_concept[seq_eff]                        # (N, T)
        z_tag = torch.gather(z, 2, c.unsqueeze(-1)).squeeze(-1)
        steps = alpha.unsqueeze(-1) * (z_tag.unsqueeze(-1) - d)
        csum = torch.cumsum(steps, dim=-1)
        return torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)

    def nll(self, seq_eff, resp, step_mask=None, g_off_from=None,
            g_zero=False):
        """Mean NLL over selected steps; also returns per-step NLL (N, T)."""
        z = self.unroll(seq_eff, g_off_from=g_off_from, g_zero=g_zero)
        lg = self.logits(seq_eff, z)
        ll = F.log_softmax(lg, dim=-1)
        per = -torch.gather(ll, 2, resp.unsqueeze(-1)).squeeze(-1)
        if step_mask is None:
            return per.mean(), per
        m = step_mask.to(per.dtype)
        return (per * m).sum() / m.sum(), per

    # ------------------------------------------------------------ fitting
    def fit_stage1(self, seq_eff, resp, epochs=80, lr=5e-3, verbose=False):
        params = [self.log_alpha, self.d] + self.dyn_params()
        opt = torch.optim.Adam(params, lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            loss, _ = self.nll(seq_eff, resp)
            loss.backward()
            opt.step()
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"    s1 ep{ep:3d} nll={loss.item():.4f}")
        return float(loss.item())

    def freeze_items(self):
        self.log_alpha.requires_grad_(False)
        self.d.requires_grad_(False)

    def fit_stage2(self, seq_eff, resp, epochs=60, lr=5e-3, l1_g=1e-2,
                   verbose=False):
        self.freeze_items()
        params = [self.G_raw] + self.dyn_params()
        opt = torch.optim.Adam(params, lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            loss, _ = self.nll(seq_eff, resp)
            total = loss + l1_g * self.G_signed.abs().sum()
            total.backward()
            opt.step()
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"    s2 ep{ep:3d} nll={loss.item():.4f} "
                      f"G={self.G_signed.detach().cpu().numpy().round(4)}")
        return float(loss.item())

    def fit_static(self, seq_eff, resp, epochs=120, lr=5e-3):
        """Anchor-first calibration: STATIC person (gain forced 0, rho=1),
        z0 free per learner. Item params from a motionless-person fit."""
        with torch.no_grad():
            self.gain_raw.fill_(_inv_softplus(1e-6))
            self.rho_raw.fill_(20.0)          # sigmoid -> ~1, OU pull off
        params = [self.log_alpha, self.d, self.z0]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss, _ = self.nll(seq_eff, resp)
            loss.backward()
            opt.step()
        return float(loss.item())


# ------------------------------------------------------- refit discrepancy
def refit_items(model, seq_eff, resp, epochs=300, lr=2e-2):
    """Per-item refit of (alpha, d) holding the model's own states fixed.
    Separable across items given z, so one joint Adam == per-item refits.
    Returns refit log_alpha (J,) numpy."""
    with torch.no_grad():
        z = model.unroll(seq_eff)
    log_a = model.log_alpha.detach().clone().requires_grad_(True)
    d = model.d.detach().clone().requires_grad_(True)
    c = model.item_concept[seq_eff]
    z_tag = torch.gather(z, 2, c.unsqueeze(-1)).squeeze(-1).detach()
    opt = torch.optim.Adam([log_a, d], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        alpha = torch.exp(log_a)[seq_eff]
        dd = d[seq_eff]
        steps = alpha.unsqueeze(-1) * (z_tag.unsqueeze(-1) - dd)
        csum = torch.cumsum(steps, dim=-1)
        lg = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
        ll = F.log_softmax(lg, dim=-1)
        loss = -torch.gather(ll, 2, resp.unsqueeze(-1)).squeeze(-1).mean()
        loss.backward()
        opt.step()
    return log_a.detach().cpu().numpy()


# ----------------------------------------------------------- structural QA
def isolation_check(device="cpu", tol=1e-6):
    """Real leak test (2026-07-05 review rewrite). Setup: exact rho=1
    (ou=False), all items tag concept 0, G[1,0] = 0.7 NONZERO and row 2 all
    zero. Assertions: (a) concept 2 (zero G-row, unpracticed) stays exactly
    flat WITH G active -- the isolation guarantee proper; (b) concept 1
    moves through G (positive control, the check can fail); (c) with
    g_zero=True concept 1 is flat too. Returns max flat-leg deviation."""
    torch.manual_seed(0)
    C, K, T, N = 3, 5, 40, 7
    J = 6
    item_concept = np.zeros(J, dtype=int)          # ALL items tag concept 0
    m = SimpleStructureGPCM(J, C, K, N, item_concept, ou=False).to(device)
    with torch.no_grad():
        m.z0.normal_(0, 1.0)
        m.G_raw.zero_()
        m.G_raw[1, 0] = 0.7                         # the live leak route
    seq = torch.randint(0, J, (N, T), device=device)
    z = m.unroll(seq)                                # G ACTIVE
    dev2 = float((z[:, :, 2] - m.z0[:, 2:3]).abs().max())      # (a)
    moved1 = float((z[:, -1, 1] - m.z0[:, 1]).abs().min())     # (b)
    z0g = m.unroll(seq, g_zero=True)
    dev1 = float((z0g[:, :, 1] - m.z0[:, 1:2]).abs().max())    # (c)
    dev = max(dev2, dev1)
    assert dev < tol, f"ISOLATION LEAK: {dev} >= {tol}"
    assert moved1 > 1e-2, "positive control failed: G route inert"
    return dev


if __name__ == "__main__":
    print("isolation deviation (asserted < 1e-6):", isolation_check())
