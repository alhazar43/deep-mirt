"""D=3 signed-edge generator for the CT0 power precondition
(`_planning/design/a1_design.md` v1.1, sections 2.1.1, 3.1, 3.2).

Extends `growth/synth.py`: the certified A4 substrate (KC popularity,
learner-KC incidence, per-learner heterogeneity, item bank, calibrated
difficulties, matched-twin discipline) is reused VERBATIM via
`growth.synth._build_substrate`; the ONE addition is a known sparse signed
``G_true`` injected into the per-slice trajectory so a target KC's state
responds to source-KC practice through the SAME practice-gated,
sign-asymmetric gate the model uses (`transfer.model.sign_asymmetric_
transfer`). Generator and model share the gate (the qmirt discipline).

Scope: this builds ONLY the two twins CT0 reads -- SYN-T-KG (known signed
edges) and its matched SYN-T-NG null (G_true = 0, own-growth identical) --
at D = 3, with an N-learners knob and a decoupling knob. The confound
twins (CNT/CNT-DYN/ENDO/CO/NS/SAT) and D = 5/8/full-K are later stages.

The DECOUPLING knob (design's positivity quantity, section 3.1). Per-edge
sign is identified only when the cumulative SOURCE-practice count that a
target has seen decorrelates from the target's OWN opportunity index;
otherwise transfer is collinear with own-gain (the SYN-T-CO boundary).
``decoupling`` in [0, 1] is the fraction of SOURCE-practice slots placed in
a slot DECOUPLED from the target: a decoupled source event keeps an
independent arrival time (free-floating, so its running count decorrelates
from the target index across learners), while a coupled source event is
glued immediately before a randomly chosen target opportunity (forcing its
running count to track the target index). ``decoupling = 1`` is the clean
identifiable regime, ``decoupling = 0`` the fully coupled boundary; the
per-edge EFFECTIVE SAMPLE (learners contributing a decoupled
source-before-target observation) grows with both N and decoupling, which
is exactly the CT0 power axis (section 2.1.1).

Matched-twin discipline: the schedule (arrival times, item draws, decouple
flags, gluing) is drawn from a schedule RNG whose seed does NOT depend on
the twin and is built from the KG edge SUPPORT for BOTH twins, so SYN-T-NG
shares SYN-T-KG's exact schedule and differs only by ``G_true = 0``. The
response RNG is likewise shared, so the two twins consume the same Bernoulli
draws against different success probabilities (the A4 matched-twin rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from kt_mirt.growth.synth import DensityProfile, ItemBank, LearnerLog, _build_substrate
from kt_mirt.transfer.model import _sign_asymmetric_np, ceiling_init, floor_init

SignedTwinName = Literal["syn_t_kg", "syn_t_ng"]

# Schedule / response RNG salts (twin-INVARIANT, so KG and NG match).
_SCHED_SALT = 7001
_RESP_SALT = 7002
_GLUE_EPS = 1e-6


# ---------------------------------------------------------------------------
# D=3 density profiles (the A4 profile shape at KC-count 3)
# ---------------------------------------------------------------------------


def d3_profile(density: str = "kdd", n_learners: int = 500) -> DensityProfile:
    """A D = 3 `DensityProfile` in the A4 KDD- or EdNet-matched density
    shape, with ``n_learners`` the CT0 N-knob. ``kcs_per_learner`` is set
    high enough that every learner practices all 3 KCs (so every edge draws
    from the full cohort; the per-edge effective sample is then governed by
    the decoupling knob and N, not by incidence)."""
    if density == "kdd":
        return DensityProfile(
            name=f"d3_kdd_N{n_learners}", n_kcs=3, n_learners=n_learners,
            opp_q1=4.0, opp_median=8.0, opp_q3=16.0, opp_clip=(1, 60), opp_frac_ge10=0.42,
            rate_q1=0.75, rate_median=0.855, rate_q3=0.925, kcs_per_learner=6.0,
            item_arity_mean=1.0, item_arity_max=1, rise_band=(0.08, 0.16),
        )
    if density == "ednet":
        return DensityProfile(
            name=f"d3_ednet_N{n_learners}", n_kcs=3, n_learners=n_learners,
            opp_q1=1.0, opp_median=2.0, opp_q3=5.0, opp_clip=(1, 60), opp_frac_ge10=0.14,
            rate_q1=0.61, rate_median=0.68, rate_q3=0.72, kcs_per_learner=6.0,
            item_arity_mean=2.2, item_arity_max=3, rise_band=(0.09, 0.17),
        )
    raise ValueError(f"unknown density {density!r}, expected 'kdd' or 'ednet'")


# ---------------------------------------------------------------------------
# The reference signed edge set
# ---------------------------------------------------------------------------


def default_G_true(n_kcs: int, g_pos: float = 0.05, g_neg: float = 0.02) -> np.ndarray:
    """The reference sparse signed ``G_true`` (design section 3.1, "one +,
    one -, one zero-row control KC" at D = 3, scaled with D). Row = target
    ``c``, column = source ``a``; zero-diagonal.

    ``g_pos`` and ``g_neg`` are the injected COEFFICIENTS (state-coefficient
    units). ``g_pos > g_neg`` by default reflects the gate-room asymmetry at
    the calibrated D=3 difficulties (positive ceiling-room ~1.4 vs negative
    floor-room ~6), chosen so a positive and a negative edge land at a
    comparable per-opportunity logit push (design's score-anchored dose,
    section 3.1); the two halves are still scored SEPARATELY (section 4.1).

    At D = 3: ``G[0, 1] = +g_pos`` (practicing KC1 helps KC0), ``G[1, 2] =
    -g_neg`` (practicing KC2 hurts KC1), KC2 a zero-row control (no incoming
    edge). For D > 3 the positive and negative edges scale ~ D/3 over
    distinct source/target cells, leaving >= 1 zero-row control KC."""
    G = np.zeros((n_kcs, n_kcs), dtype=float)
    if n_kcs < 3:
        raise ValueError("default_G_true needs at least 3 KCs")
    n_edge = max(1, n_kcs // 3)
    # Positive edges c<-a on cells (2k, 2k+1); negative edges on (2k+1, 2k+2),
    # keeping the highest-indexed KC a zero-row control.
    for k in range(n_edge):
        c_p, a_p = (2 * k) % n_kcs, (2 * k + 1) % n_kcs
        if c_p != a_p and c_p != n_kcs - 1:
            G[c_p, a_p] = g_pos
    for k in range(n_edge):
        c_n, a_n = (2 * k + 1) % n_kcs, (2 * k + 2) % n_kcs
        if c_n != a_n and c_n != n_kcs - 1 and G[c_n, a_n] == 0.0:
            G[c_n, a_n] = -g_neg
    np.fill_diagonal(G, 0.0)
    return G


# ---------------------------------------------------------------------------
# Ground truth / output containers
# ---------------------------------------------------------------------------


@dataclass
class TransferGroundTruth:
    G_true: np.ndarray  # (C, C) injected signed matrix, zero-diagonal
    M_gen: float  # positive-transfer ceiling used in the generator gate
    floor: float  # negative-transfer floor used in the generator gate
    b_true: np.ndarray  # (n_items,) frozen difficulties (the CT0 "bank")
    decoupling: float
    eff_sample: np.ndarray  # (C, C) per-edge effective sample (decoupled a-before-c learners)
    realized_displacement: np.ndarray  # (C, C) mean target-state displacement attributable to source
    off_diag: np.ndarray = field(default=None)  # (C, C) bool, True off the diagonal

    def __post_init__(self):
        C = self.G_true.shape[0]
        self.off_diag = ~np.eye(C, dtype=bool)


@dataclass
class SignedTwin:
    twin: str
    profile: DensityProfile
    seed: int
    n_kcs: int
    n_learners: int
    decoupling: float
    item_bank: ItemBank
    learners: list[LearnerLog]
    truth: TransferGroundTruth


# ---------------------------------------------------------------------------
# Own-growth (A4 standard bounded-exponential, as a function of own index)
# ---------------------------------------------------------------------------


def _own_state(theta0: float, m: float, r: float, lam: float, k: int) -> float:
    """A4 standard own-growth state after ``k`` own-practices (``k = 0`` is
    ``theta0``): ``m - (m - theta0) * exp(-r * lam * k)`` (the closed form
    of `growth.synth._standard_theta_matrix`, indexed by the KC's OWN
    opportunity count). Transfer is layered ON TOP of this per design 3.1."""
    return m - (m - theta0) * np.exp(-r * lam * k)


# ---------------------------------------------------------------------------
# Schedule construction (twin-invariant; drives the decoupling knob)
# ---------------------------------------------------------------------------


@dataclass
class _Event:
    kc: int  # actual KC id practiced
    time: float  # arrival time (adjusted by gluing)
    item: int  # item id drawn for this opportunity
    decoupled: bool  # True iff this is a decoupled SOURCE event


def _edge_support(G_true: np.ndarray) -> tuple[list[int], dict[int, list[int]]]:
    """Return ``(sources, targets_of)``: the source KCs (columns with any
    nonzero) and, per source ``a``, the list of target KCs ``c`` it
    influences (rows with ``G[c, a] != 0``)."""
    C = G_true.shape[0]
    targets_of: dict[int, list[int]] = {}
    sources: list[int] = []
    for a in range(C):
        tgt = [c for c in range(C) if c != a and G_true[c, a] != 0.0]
        if tgt:
            sources.append(a)
            targets_of[a] = tgt
    return sources, targets_of


def _build_learner_schedule(
    kcs: np.ndarray,
    Ts: np.ndarray,
    item_bank: ItemBank,
    targets_of: dict[int, list[int]],
    decoupling: float,
    rng: np.random.Generator,
) -> tuple[list[_Event], dict[int, np.ndarray]]:
    """Build one learner's interleaved event sequence under the decoupling
    knob. Depends ONLY on the KG edge SUPPORT (``targets_of``), never on
    edge magnitudes/signs, so SYN-T-KG and SYN-T-NG share it exactly.

    Each KC's opportunities get independent U(0,1) base times. Every SOURCE
    event is then, with probability ``1 - decoupling``, glued immediately
    before a random target opportunity (its time set just under that target
    event's base time); otherwise it stays decoupled (free base time). The
    sequence is the events sorted by adjusted time."""
    events: list[_Event] = []
    drawn: dict[int, np.ndarray] = {}
    base_time: dict[int, list[float]] = {}
    for local_idx, c in enumerate(kcs):
        c = int(c)
        T = int(Ts[local_idx])
        pool = item_bank.items_by_primary[c]
        items = rng.choice(pool, size=T, replace=True)
        drawn[c] = items
        times = rng.random(T)
        base_time[c] = list(times)
        for k in range(T):
            events.append(_Event(kc=c, time=float(times[k]), item=int(items[k]), decoupled=False))

    # Snapshot target base times BEFORE gluing (so chained gluing anchors on
    # the un-glued target schedule, deterministically).
    for e in events:
        if e.kc in targets_of:  # e.kc is a source
            if rng.random() < decoupling:
                e.decoupled = True
            else:
                tgt_choices = [c for c in targets_of[e.kc] if base_time.get(c)]
                if not tgt_choices:
                    e.decoupled = True  # no target to couple to; treat as decoupled
                    continue
                c_tgt = int(rng.choice(tgt_choices))
                t_anchor = float(rng.choice(base_time[c_tgt]))
                e.time = t_anchor - _GLUE_EPS * (1.0 + 0.1 * rng.random())
                e.decoupled = False

    events.sort(key=lambda ev: ev.time)
    return events, drawn


# ---------------------------------------------------------------------------
# Response realization (injects G_true through the shared gate)
# ---------------------------------------------------------------------------


def _realize_twin(
    profile: DensityProfile,
    seed: int,
    G_gate: np.ndarray,
    G_support: np.ndarray,
    M_gen: float,
    floor: float,
    decoupling: float,
) -> tuple[list[LearnerLog], np.ndarray, np.ndarray]:
    """Realize one twin's learner logs. Returns ``(learners, eff_sample,
    realized_displacement)``.

    ``G_support`` drives the SCHEDULE (which cells are edges under test, so
    which source events glue to which targets); ``G_gate`` supplies the
    signed COEFFICIENTS the transfer gate actually applies. SYN-T-KG passes
    the same signed matrix for both; SYN-T-NG passes the KG matrix as
    ``G_support`` (identical schedule and effective sample) but an all-zero
    ``G_gate`` (no transfer realized). This is the matched-twin discipline
    made explicit -- no support hacks."""
    substrate = _build_substrate(profile, seed)
    C, N = profile.n_kcs, profile.n_learners
    item_bank = substrate.item_bank
    sources, targets_of = _edge_support(G_support)

    sched_rng = np.random.default_rng((seed, _SCHED_SALT))
    resp_rng = np.random.default_rng((seed, _RESP_SALT))

    learners: list[LearnerLog] = []
    eff_sample = np.zeros((C, C), dtype=float)
    disp_accum = np.zeros((C, C), dtype=float)
    disp_count = np.zeros((C, C), dtype=float)

    for i in range(N):
        kcs = substrate.learner_kc_ids[i]
        Ts = substrate.learner_kc_T[i]
        # Per-KC own-growth params for this learner, keyed by actual KC id.
        theta0 = {int(c): float(substrate.theta0[i][j]) for j, c in enumerate(kcs)}
        m = {int(c): float(substrate.theta0[i][j] + substrate.gap[i][j]) for j, c in enumerate(kcs)}
        r = {int(c): float(substrate.r_base[int(c)]) for c in kcs}
        lam = float(substrate.lambda_i[i])

        events, _ = _build_learner_schedule(kcs, Ts, item_bank, targets_of, decoupling, sched_rng)

        own_k: dict[int, int] = {int(c): 0 for c in kcs}
        transfer: dict[int, float] = {int(c): 0.0 for c in kcs}

        item_ids = np.empty(len(events), dtype=np.int64)
        responses = np.empty(len(events), dtype=np.int8)
        for pos, e in enumerate(events):
            c = e.kc
            k = own_k[c]
            z_pre = _own_state(theta0[c], m[c], r[c], lam, k) + transfer[c]
            b = float(item_bank.b_true[e.item])
            p = 1.0 / (1.0 + np.exp(-(z_pre - b)))
            responses[pos] = int(resp_rng.random() < p)
            item_ids[pos] = e.item
            # c is now a source: push its targets from THEIR pre-update state.
            if c in targets_of:
                for t in targets_of[c]:
                    if t not in own_k:
                        continue
                    z_t = _own_state(theta0[t], m[t], r[t], lam, own_k[t]) + transfer[t]
                    push = float(_sign_asymmetric_np(G_gate[t, c], z_t, M_gen, floor))
                    transfer[t] += push
                    disp_accum[t, c] += push
                    disp_count[t, c] += 1.0
            own_k[c] += 1

        # Effective sample per edge (c<-a): the number of DECOUPLED
        # a-practice OBSERVATIONS that precede at least one c-measurement
        # (a practiced, then measured on c, in a decoupled slot -- design
        # section 2.1.1's per-edge learner-observation count, taken at the
        # observation grain so the axis spans identifiability linearly in
        # decoupling and N, not saturating at the learner count).
        c_times: dict[int, list[float]] = {int(cc): [] for cc in kcs}
        decoupled_src_times: dict[int, list[float]] = {int(cc): [] for cc in kcs}
        for e in events:
            c_times[e.kc].append(e.time)
            if e.decoupled:
                decoupled_src_times[e.kc].append(e.time)
        for a in sources:
            src_ts = decoupled_src_times.get(a, [])
            if not src_ts:
                continue
            for c in targets_of[a]:
                c_ts = c_times.get(c, [])
                if not c_ts:
                    continue
                latest_c = max(c_ts)
                eff_sample[c, a] += float(sum(1 for t in src_ts if t < latest_c))

        tag_ids = item_bank.tag_ids[item_ids]
        tag_mask = item_bank.tag_mask[item_ids]
        learners.append(
            LearnerLog(learner=i, item_ids=item_ids, responses=responses, tag_ids=tag_ids, tag_mask=tag_mask)
        )

    realized_disp = np.divide(disp_accum, disp_count, out=np.zeros_like(disp_accum), where=disp_count > 0)
    return learners, eff_sample, realized_disp


def generate_signed_twin(
    twin: SignedTwinName,
    density: str = "kdd",
    seed: int = 0,
    n_learners: int = 500,
    decoupling: float = 0.8,
    g_pos: float = 0.05,
    g_neg: float = 0.02,
    G_true: Optional[np.ndarray] = None,
) -> SignedTwin:
    """Build one CT0 twin. ``twin='syn_t_kg'`` injects the reference signed
    ``G_true``; ``twin='syn_t_ng'`` is the matched null (same schedule and
    RNG streams, ``G_true = 0``). The SCHEDULE is always built from the KG
    support so the two twins match exactly (the design's matched-twin
    discipline). Returns a `SignedTwin` carrying the learner logs, the frozen
    difficulties, the per-edge effective sample, and the realized transfer
    displacement (for generator acceptance)."""
    if twin not in ("syn_t_kg", "syn_t_ng"):
        raise ValueError(f"unknown twin {twin!r}, expected 'syn_t_kg' or 'syn_t_ng'")
    profile = d3_profile(density, n_learners)
    substrate = _build_substrate(profile, seed)
    C = profile.n_kcs
    b_true = substrate.item_bank.b_true
    M_gen = ceiling_init(b_true)
    floor = floor_init(b_true)

    G_ref = default_G_true(C, g_pos, g_neg) if G_true is None else np.asarray(G_true, dtype=float).copy()
    np.fill_diagonal(G_ref, 0.0)
    # G_support = the KG edge set (drives the schedule for BOTH twins);
    # G_gate = the signed coefficients actually applied (KG: G_ref; NG: 0).
    G_gate = G_ref if twin == "syn_t_kg" else np.zeros((C, C))
    learners, eff_sample, realized_disp = _realize_twin(
        profile, seed, G_gate, G_ref, M_gen, floor, decoupling
    )
    truth = TransferGroundTruth(
        G_true=G_ref if twin == "syn_t_kg" else np.zeros((C, C)),
        M_gen=M_gen, floor=floor, b_true=b_true, decoupling=decoupling,
        eff_sample=eff_sample, realized_displacement=realized_disp,
    )
    return SignedTwin(
        twin=twin, profile=profile, seed=seed, n_kcs=C, n_learners=profile.n_learners,
        decoupling=decoupling, item_bank=substrate.item_bank, learners=learners, truth=truth,
    )


# ---------------------------------------------------------------------------
# Generator acceptance (design section 3.1's two new checks)
# ---------------------------------------------------------------------------


def realized_transfer_ok(twin: SignedTwin, rel_tol: float = 0.5) -> dict:
    """The two new generator acceptance checks (design section 3.1): on the
    KG twin the realized target-state displacement attributable to source
    practice must be sign-correct on every true edge (and non-trivial in
    magnitude); on the NG twin it must be exactly zero. Returns a dict of
    per-edge realized displacement and a boolean ``passed``."""
    G = twin.truth.G_true
    disp = twin.truth.realized_displacement
    off = twin.truth.off_diag
    if twin.twin == "syn_t_ng":
        passed = bool(np.all(disp[off] == 0.0))
        return {"twin": twin.twin, "passed": passed, "max_abs_disp": float(np.max(np.abs(disp)))}
    ok = True
    per_edge = {}
    for c in range(G.shape[0]):
        for a in range(G.shape[1]):
            if c != a and G[c, a] != 0.0:
                same_sign = np.sign(disp[c, a]) == np.sign(G[c, a])
                nontrivial = abs(disp[c, a]) > 0.0
                per_edge[(c, a)] = float(disp[c, a])
                ok = ok and bool(same_sign and nontrivial)
    return {"twin": twin.twin, "passed": bool(ok), "per_edge_displacement": per_edge}


__all__ = [
    "SignedTwinName",
    "d3_profile",
    "default_G_true",
    "TransferGroundTruth",
    "SignedTwin",
    "generate_signed_twin",
    "realized_transfer_ok",
]
