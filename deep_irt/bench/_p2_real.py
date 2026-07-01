"""_p2_real.py -- real-data split-half reliability driver (Phase-2 step 6b).

The real-data leg has NO ground-truth item parameters, so recovery is scored by
RELIABILITY (blueprint section D): split the learners into halves, fit an
independent model on each, correlate the per-item recovered parameters on the
items both halves saw, Spearman-Brown correct.  This module is the driver that
turns a {dataset, decoder, arm} cell into those numbers.

Reuse (not reinvention)
  * data loading -- the verified workshop real-data loaders:
      ``_ednet_reliability`` (EdNet-KT1: answer key, item bank, learner
      sequences, the K=3 elapsed-time coerced-ordinal coder) and
      ``_kdd_reliability`` (KDD Cup 2010 Algebra: chunked TSV loader, the
      attempts/hints K=3 coder).  Read-only ``_ednet_ot`` is the reference for
      the EdNet option-tracing (NRM) label; its fixed-length loader is
      re-implemented here parametrically (that file bakes its sizes as module
      constants and must not be edited).
  * reliability + accuracy -- ``_p2_reliability.split_half_reliability`` and
      ``_p2_reliability.predictive_accuracy``, both driving the Phase-2
      ``_P2Engine`` (proper GPCM ordinal CE, Option-A decoupled routing, masked
      variable-length prediction).

Arms (the {shared vs the decoupling fix} contrast, per decoder)
  * 2PL / GPCM: ``shared`` = thin-value only (item_key_dim=None, emb 64);
      ``fix`` = STATIC decoupled, slope on its own wide item key via Option A
      (item_key_dim=64, emb 8, state_alpha=False).
  * NRM: ``shared`` = all readouts on the thin value; ``fix`` = ``c_only_dec``
      (the intercept c_k on its own wide key -- the blueprint's expected
      minimal-sufficient NRM channel; NRM is the boundary case where the
      OPPOSITE parameter to GPCM's slope needs the channel).

Datasets present on disk: EdNet-KT1 (ednet), KDD Cup Algebra (kdd).  ASSISTments
raw data is NOT on disk (only legacy figures); those cells are unavailable.

Scratch file: ``_``-prefixed, gitignored.  CLI:
    python -m deep_irt.bench._p2_real --config <cfg.yaml> [--smoke] [--device cuda]
    python -m deep_irt.bench._p2_real --dataset ednet --decoder 2pl --arm shared --smoke
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from deep_irt.bench._p2_engine import _P2Engine
from deep_irt.bench._p2_reliability import (
    split_half_reliability, predictive_accuracy,
)

ROOT = Path("C:/Users/steph/documents/deep-mirt")
OUT_DIR = ROOT / "deep_irt" / "bench" / "outputs" / "p2_real"

# decoder token -> core decoder name + K
_DECODER = {"2pl": "binary", "gpcm": "gpcm", "nrm": "nrm"}
_NCATS = {"2pl": 2, "gpcm": 3, "nrm": 4}


# ---------------------------------------------------------------------------
# Cell config (a real-data reliability cell)
# ---------------------------------------------------------------------------

@dataclass
class RealCell:
    """One {dataset, decoder, arm} real-data reliability cell."""
    dataset: str = "ednet"          # "ednet" | "kdd"
    decoder: str = "2pl"            # "2pl" | "gpcm" | "nrm"
    arm: str = "shared"            # "shared" | "fix"

    # data sizing (reuses the workshop loader defaults)
    n_learners: int = 3000
    n_items: int = 250
    max_seq_len: int = 200
    min_answers: int = 20
    scan_files: int = 30000         # EdNet: how many user files to scan/rank

    # training
    n_epochs: int = 150
    lr: float = 1e-2
    batch_size: Optional[int] = 512
    hidden_dim: int = 32
    n_holdout: int = 10

    # reliability
    n_splits: int = 4
    seed: int = 42

    name: str = ""                  # cell name for the output dir (auto if empty)

    def resolved_name(self) -> str:
        return self.name or f"{self.dataset}_{self.decoder}_{self.arm}"


def load_real_config(path: str | Path) -> RealCell:
    """Load a RealCell from a small YAML (unknown keys dropped)."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    valid = {f.name for f in dataclasses.fields(RealCell)}
    return RealCell(**{k: v for k, v in raw.items() if k in valid})


# ---------------------------------------------------------------------------
# Engine kwargs for the {shared, fix} arms
# ---------------------------------------------------------------------------

def engine_kwargs_for(cell: RealCell, device: str) -> dict:
    """Map (decoder, arm) to _P2Engine constructor kwargs (blueprint arms)."""
    dec = _DECODER[cell.decoder]
    ek: dict = dict(
        decoder=dec, hidden_dim=cell.hidden_dim, encoder="lstm",
        device=device, gpcm_loss="ordinal_ce", state_alpha=False,
    )
    if cell.decoder in ("2pl", "gpcm"):
        if cell.arm == "shared":
            ek.update(emb_dim=64, item_key_dim=None)
        elif cell.arm == "fix":
            ek.update(emb_dim=8, item_key_dim=64)       # Option A static decouple
        else:
            raise ValueError(f"arm must be shared|fix, got {cell.arm!r}")
    elif cell.decoder == "nrm":
        if cell.arm == "shared":
            ek.update(emb_dim=8, item_key_dim=None, nrm_channel="shared")
        elif cell.arm == "fix":
            ek.update(emb_dim=8, item_key_dim=64, nrm_channel="c_only_dec")
        else:
            raise ValueError(f"arm must be shared|fix, got {cell.arm!r}")
    else:
        raise ValueError(f"unknown decoder {cell.decoder!r}")
    return ek


def train_kwargs_for(cell: RealCell) -> dict:
    """_P2Engine.fit kwargs -- 150 epochs, no early stop (full-length reliability)."""
    return dict(n_epochs=cell.n_epochs, lr=cell.lr,
                batch_size=cell.batch_size, patience=None)


# ---------------------------------------------------------------------------
# Data loading (reuse the workshop loaders)
# ---------------------------------------------------------------------------

def _load_ednet(cell: RealCell):
    """EdNet-KT1: (items, responses, mask) for 2pl (binary) or gpcm (K=3 ordinal).

    Reuses ``_ednet_reliability`` verbatim for the answer key, item bank, learner
    sequences and (gpcm) the elapsed-time coerced-ordinal coder.
    """
    from deep_irt.bench import _ednet_reliability as er
    correct = er.load_correct_answers()
    bank = er.build_item_bank(correct, cell.scan_files, cell.n_items)
    items, corr, elapsed, _uids = er.load_learners(
        correct, bank, cell.scan_files, cell.min_answers,
        cell.n_learners, cell.max_seq_len,
    )
    mask = items >= 0
    if cell.decoder == "2pl":
        resp = np.where(mask, corr, 0).astype(np.int64)
        return items, resp, mask
    if cell.decoder == "gpcm":
        all_train = np.ones(items.shape[0], dtype=bool)
        cats = er.build_ordinal_responses(items, corr, elapsed, all_train)
        resp = np.where(mask, cats, 0).astype(np.int64)
        return items, resp, mask
    if cell.decoder == "nrm":
        return _load_ednet_nrm(cell)
    raise ValueError(f"ednet: unsupported decoder {cell.decoder!r}")


def _load_ednet_nrm(cell: RealCell):
    """EdNet-KT1 option tracing (K=4): label = chosen option a/b/c/d.

    Mirrors the read-only ``_ednet_ot`` label (Ghosh et al. AIED 2021), but
    parametric so the cell's sizes/subsample control it.  Reuses
    ``_ednet_reliability.load_correct_answers`` for the answer key and the same
    file glob; builds fixed-length (max_seq_len) option sequences with a validity
    mask so short learners are padded, not dropped.
    """
    from deep_irt.bench import _ednet_reliability as er
    opt = {"a": 0, "b": 1, "c": 2, "d": 3}
    correct = er.load_correct_answers()  # {qid: correct_char}; only need the keys

    files = sorted(er.EDNET_DIR.glob("u*.csv"),
                   key=lambda p: int(p.stem[1:]))[:cell.scan_files]
    qvocab: dict[str, int] = {}
    rows: list[tuple[list[int], list[int]]] = []
    for f in files:
        if len(rows) >= cell.n_learners:
            break
        seq_q, seq_o = [], []
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                qid = row["question_id"].strip()
                ua = row["user_answer"].strip().lower()
                if qid not in correct or ua not in opt:
                    continue
                seq_q.append(qid)
                seq_o.append(opt[ua])
                if len(seq_q) >= cell.max_seq_len:
                    break
        if len(seq_q) >= cell.min_answers:
            rows.append((seq_q, seq_o))

    N = len(rows)
    T = max(len(q) for q, _ in rows)
    items = np.full((N, T), -1, dtype=np.int64)
    resp = np.zeros((N, T), dtype=np.int64)
    for i, (sq, so) in enumerate(rows):
        for t, (qid, o) in enumerate(zip(sq, so)):
            if qid not in qvocab:
                qvocab[qid] = len(qvocab)
            items[i, t] = qvocab[qid]
            resp[i, t] = o
    mask = items >= 0
    return items, resp, mask


def _load_kdd(cell: RealCell):
    """KDD Cup 2010 Algebra: (items, responses, mask) for 2pl or gpcm.

    Reuses ``_kdd_reliability`` for the streamed item bank + learner sequences
    (attempts/hints K=3 coder).  2PL correctness = correct-first-attempt (cat 2).
    """
    from deep_irt.bench import _kdd_reliability as kr
    bank = kr.build_item_bank(cell.n_items)
    items, cats = kr.load_learners(
        bank, cell.min_answers, cell.n_learners, cell.max_seq_len,
    )
    mask = items >= 0
    if cell.decoder == "2pl":
        resp = np.where(mask, (cats == 2).astype(np.int64), 0)
        return items, resp, mask
    if cell.decoder == "gpcm":
        resp = np.where(mask, cats, 0).astype(np.int64)
        return items, resp, mask
    raise ValueError(f"kdd: unsupported decoder {cell.decoder!r} (no options)")


def load_data(cell: RealCell):
    if cell.dataset == "ednet":
        return _load_ednet(cell)
    if cell.dataset == "kdd":
        return _load_kdd(cell)
    raise ValueError(
        f"unknown/absent dataset {cell.dataset!r}. Present on disk: ednet, kdd. "
        f"ASSISTments raw data is not available."
    )


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

def _coverage_counts(items: np.ndarray, mask: np.ndarray, Q: int) -> np.ndarray:
    """(Q,) responses seen per item (valid positions only)."""
    ids = np.asarray(items)[np.asarray(mask, dtype=bool)]
    ids = ids[ids >= 0]
    cov = np.zeros(Q, dtype=np.int64)
    if ids.size:
        u, c = np.unique(ids, return_counts=True)
        cov[u] = c
    return cov


def run_real_cell(cell: RealCell, device: str = "cpu",
                  smoke: bool = False) -> dict:
    """Load the dataset, run split-half reliability + the accuracy guard."""
    if smoke:
        # Small, fast: subsample learners/items and cut epochs/splits.
        cell = dataclasses.replace(
            cell, n_learners=min(cell.n_learners, 400),
            n_items=min(cell.n_items, 150), max_seq_len=min(cell.max_seq_len, 100),
            scan_files=min(cell.scan_files, 6000), min_answers=15,
            n_epochs=20, n_splits=2, batch_size=256,
        )

    t0 = time.time()
    items, resp, mask = load_data(cell)
    N, T = items.shape
    Q = int(items[items >= 0].max()) + 1 if (items >= 0).any() else 0
    cov = _coverage_counts(items, mask, Q)
    t_load = time.time() - t0

    n_cats = _NCATS[cell.decoder]
    ek = engine_kwargs_for(cell, device)
    tk = train_kwargs_for(cell)

    t1 = time.time()
    rel = split_half_reliability(
        resp, items, _P2Engine, ek, tk,
        n_splits=cell.n_splits, seed=cell.seed, mask=mask,
        n_cats=n_cats, n_holdout=cell.n_holdout,
    )
    t_rel = time.time() - t1

    t2 = time.time()
    acc = predictive_accuracy(
        resp, items, _P2Engine, ek, tk,
        decoder=cell.decoder, n_cats=n_cats, n_holdout=cell.n_holdout,
        seed=cell.seed, mask=mask,
    )
    t_acc = time.time() - t2

    return {
        "cell": cell.resolved_name(),
        "config": dataclasses.asdict(cell),
        "smoke": bool(smoke),
        "device": device,
        "data": {
            "n_learners": int(N), "seq_len": int(T), "n_items": int(Q),
            "n_cats": int(n_cats),
            "coverage_min": int(cov[cov > 0].min()) if (cov > 0).any() else 0,
            "coverage_median": float(np.median(cov[cov > 0])) if (cov > 0).any() else 0.0,
        },
        "engine_kwargs": {k: v for k, v in ek.items() if k != "device"},
        "reliability": {
            "a_reliability": rel["a_reliability"],
            "a_reliability_std": rel["a_reliability_std"],
            "b_reliability": rel["b_reliability"],
            "b_reliability_std": rel["b_reliability_std"],
            "a_split": rel["a_split"],
            "b_split": rel["b_split"],
        },
        "accuracy": acc,
        "timing_s": {"load": round(t_load, 1), "reliability": round(t_rel, 1),
                     "accuracy": round(t_acc, 1), "total": round(time.time() - t0, 1)},
    }


def _print_result(res: dict) -> None:
    r = res["reliability"]
    a = res["accuracy"]
    d = res["data"]
    slope_lbl = "a_k" if res["config"]["decoder"] == "nrm" else "alpha"
    loc_lbl = "c_k" if res["config"]["decoder"] == "nrm" else "beta"
    print("=" * 68)
    print(f"CELL: {res['cell']}   (device={res['device']}, smoke={res['smoke']})")
    print(f"  data: N={d['n_learners']} T={d['seq_len']} Q={d['n_items']} "
          f"K={d['n_cats']}  coverage min/med={d['coverage_min']}/{d['coverage_median']:.0f}")
    print(f"  reliability (Spearman-Brown, |r|):")
    print(f"    slope  {slope_lbl:5s} (Spearman): {r['a_reliability']:.3f} "
          f"+/- {r['a_reliability_std']:.3f}   splits={_fmt(r['a_split'])}")
    print(f"    loc    {loc_lbl:5s} (Pearson) : {r['b_reliability']:.3f} "
          f"+/- {r['b_reliability_std']:.3f}   splits={_fmt(r['b_split'])}")
    acc_str = "  ".join(f"{k}={a[k]:.3f}" for k in ("acc", "qwk", "auc")
                        if k in a and a[k] == a[k])
    print(f"  accuracy guard: {acc_str}   (n_scored={a.get('n_scored')})")
    print(f"  timing: {res['timing_s']}")
    print("=" * 68)


def _fmt(xs) -> str:
    return "[" + ", ".join(f"{x:.3f}" if x == x else "nan" for x in xs) + "]"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase-2 real-data reliability cell")
    p.add_argument("--config", default=None, help="path to a real-cell YAML")
    p.add_argument("--dataset", default="ednet", choices=["ednet", "kdd"])
    p.add_argument("--decoder", default="2pl", choices=["2pl", "gpcm", "nrm"])
    p.add_argument("--arm", default="shared", choices=["shared", "fix"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="subsample learners/items, cut epochs+splits for a fast check")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    if args.config:
        cell = load_real_config(args.config)
    else:
        cell = RealCell(dataset=args.dataset, decoder=args.decoder, arm=args.arm)

    res = run_real_cell(cell, device=args.device, smoke=args.smoke)
    _print_result(res)

    if not args.no_save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = "_smoke" if args.smoke else ""
        out = OUT_DIR / f"{cell.resolved_name()}{tag}.json"
        out.write_text(json.dumps(res, indent=2, default=_json_default),
                       encoding="utf-8")
        print(f"saved -> {out}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


if __name__ == "__main__":
    main()
