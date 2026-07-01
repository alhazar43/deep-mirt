"""
_p2_config.py -- Configuration schema for Phase 2 experiments.

SCAFFOLD: intentionally near-complete; this file is the interface that all
other _p2_* modules import. Fill-in of experiment logic is elsewhere.

Role:
  - P2Config dataclass with four sub-blocks (data / model / train / eval)
    and a report block. Every field has a sensible default so a YAML can
    specify only what differs from the baseline.
  - load_config(path): YAML loader (PyYAML) with JSON fallback.
  - config_sha256(cfg): canonical sha256 of the experiment identity
    (data + model + train + eval only; report, description, and experiment
    tag are excluded so renaming a cell does not invalidate its result).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-block dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Synthetic-data generation parameters."""
    kind: str = "static"            # "static" | "dynamic" | "binary"
    n_learners: int = 800           # N
    n_items: int = 60               # Q
    seq_len: int = 60               # T
    n_cats: int = 4                 # K
    drift_sigma: float = 0.15       # per-step std for dynamic random walk
    alpha_theta_slope: float = 0.0  # 0.0 = static alpha (null condition)
    n_holdout: int = 10             # tail positions scored for prediction
    train_frac: float = 0.8
    # alpha ~ LogNormal(0, alpha_sigma); blueprint prior uses sigma=0.3
    alpha_sigma: float = 0.3
    # difficulty spread: b ~ N(0, beta_sigma); used in E-budget kappa sweep
    # NOTE: datagen.generate() currently hardcodes lognormal(0, 0.3) and N(0,1)
    # for b. When datagen exposes these as BenchDataConfig fields the values
    # here can be threaded through run_cell directly.
    beta_sigma: float = 1.0

    # -- regime / administration block (step-6a realistic bed; CORRECTED spec) --
    # "clean"     : the rectangular clean-bed generator (datagen.generate); the
    #               fields below are inert.
    # "realistic" : the step-6a ma-irt-aligned generator (_p2_datagen_realistic,
    #               a separate build) consumes these.  Administration is the ONLY
    #               noise: each learner is administered a random, theta-INDEPENDENT
    #               subset of DISTINCT items (each answered once, clean IRT draw),
    #               the count ~ Uniform[admin_min, admin_max].  run_cell routes
    #               here only when regime == "realistic"; the clean path ignores
    #               these, so the clean benchmark cells are byte-identical.
    regime: str = "clean"           # "clean" | "realistic"
    admin_min: int = 40             # min distinct items administered per learner
    admin_max: int = 80             # max distinct items administered per learner;
                                    # count ~ Uniform[admin_min, admin_max] (incl.,
                                    # clamped to Q).  ma-irt's range is 40..80.
    dense: bool = False             # dense control: every learner sees ALL Q items
                                    # (rectangular; requires seq_len >= Q)
    popularity: str = "uniform"     # "uniform" | "zipf" item-selection weighting,
                                    # theta- AND parameter-INDEPENDENT (exposure
                                    # stratification only, not adaptivity)
    holdout_frac: float = 0.2       # per-learner holdout = min(n_holdout,
                                    # floor(holdout_frac * count))


@dataclass
class ModelConfig:
    """Engine and architecture parameters."""
    engine: str = "deep_irt"        # "deep_irt" | "ma_irt"
    encoder: str = "lstm"           # "lstm" | "transformer" | "dkvmn"
    decoder: str = "gpcm"           # "gpcm" | "nrm" | "binary"
    emb_dim: int = 8
    hidden_dim: int = 32
    state_alpha: bool = False       # dynamic discrimination head (E-levers)
    state_beta: bool = False        # dynamic threshold head
    item_key_dim: Optional[int] = None  # decoupled item key width (E2b)
    # ma_irt engine extra kwargs (ignored when engine == "deep_irt")
    ma_irt_kwargs: Dict[str, Any] = field(default_factory=dict)
    # E2(a) coupling flag: whether the theta stream is item-conditioned
    # (sets separate_theta=False in MaIrtEngine; see _p2_coupled_theta.py)
    coupled_theta: bool = False


@dataclass
class TrainConfig:
    """Optimizer and schedule parameters."""
    optimizer: str = "adam"         # "adam" | "sgd" (sgd = plain GD for E-budget)
    n_epochs: int = 120
    lr: float = 1e-2
    batch_size: Optional[int] = None
    grad_clip_norm: Optional[float] = None
    early_stop_patience: Optional[int] = None
    inner_val_frac: float = 0.15    # fraction of TRAIN learners carved as the
                                    # inner-val slice for the early-stop monitor
                                    # (kept out of the CV held-out fold: no leakage)
    # GPCM training loss (blueprint section A + LOCKED build order step 1).
    #   "ordinal_ce" (default) : the strictly-proper cumulative-link ordinal CE
    #                            (_p2_ordinal_ce.OrdinalCELoss); the recovery
    #                            claim requires a proper scoring rule.
    #   "wol"                  : the legacy WeightedOrdinalLoss (improper; kept
    #                            only for the ablation contrast).
    # Ignored for the 2PL (BCE) and NRM (softmax CE) decoders -- their losses are
    # already the strictly-proper scoring rule for their response type.
    gpcm_loss: str = "ordinal_ce"   # "ordinal_ce" | "wol"


@dataclass
class EvalConfig:
    """Seed multiplier and statistical-test parameters."""
    n_data_seeds: int = 5           # separate data RNG seeds (fresh ground-truth
                                    # bank per seed); pooled with the folds below
    n_init_seeds: int = 3           # separate weight-init RNG seeds
    n_folds: int = 5                # LOCKED decision 1: 5-fold LEARNER CV.  Each
                                    # fold is one fit; recovery is the fold-mean +/-
                                    # bootstrap CI over the rotated held-out folds.
    n_bootstrap: int = 2000         # bootstrap resamples for 95% CI
    gauge: str = "affine"           # "affine" | "none"
                                    # "affine": one global linear map aligns
                                    # recovered to true scale (never per-item)
    paired_test: str = "wilcoxon"   # "wilcoxon" | "none"


@dataclass
class ReportConfig:
    """Output paths and smoke-test flag."""
    output_dir: str = "outputs/p2"
    cell_name: str = "unnamed_cell"
    quick: bool = False             # True: 1 data seed x 1 init seed (smoke)


@dataclass
class P2Config:
    """Root config for one Phase 2 experiment cell.

    Identity (hashed): data + model + train + eval.
    Non-identity (excluded from hash): report, description, experiment.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    experiment: str = "unnamed"
    description: str = ""


# ---------------------------------------------------------------------------
# YAML / JSON loader
# ---------------------------------------------------------------------------

def _load_subblock(dc_cls, raw):
    """Populate a flat dataclass from a dict; unknown keys are silently dropped."""
    if not isinstance(raw, dict):
        return dc_cls()
    valid = {f.name for f in dataclasses.fields(dc_cls)}
    return dc_cls(**{k: v for k, v in raw.items() if k in valid})


def load_config(path: str | Path) -> P2Config:
    """Load a P2Config from a YAML file (with .json fallback).

    Missing fields take their dataclass defaults. Unknown fields are ignored.
    Raises FileNotFoundError if path does not exist.
    Raises ImportError if PyYAML is missing and path is .yaml / .yml.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                raw: dict = yaml.safe_load(fh) or {}
        except ImportError as exc:
            raise ImportError(
                "PyYAML required for .yaml configs. Install with: pip install pyyaml"
            ) from exc
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raise ValueError(f"unsupported config extension: {path.suffix!r}")

    return P2Config(
        data=_load_subblock(DataConfig, raw.get("data", {})),
        model=_load_subblock(ModelConfig, raw.get("model", {})),
        train=_load_subblock(TrainConfig, raw.get("train", {})),
        eval=_load_subblock(EvalConfig, raw.get("eval", {})),
        report=_load_subblock(ReportConfig, raw.get("report", {})),
        experiment=raw.get("experiment", "unnamed"),
        description=raw.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Config hash (resume gate + provenance)
# ---------------------------------------------------------------------------

def _identity_dict(cfg: P2Config) -> dict:
    """Return the experiment-identity dict: data + model + train + eval only.

    report, description, and experiment tag are excluded so that renaming a
    cell or moving its output directory does not change its hash.
    """
    d = asdict(cfg)
    for key in ("report", "description", "experiment"):
        d.pop(key, None)
    return d


def config_sha256(cfg: P2Config) -> str:
    """Canonical sha256 of the experiment identity.

    Serialised as JSON with sorted keys and no whitespace so the hash is
    deterministic across Python versions and dict insertion orders.
    """
    canonical = json.dumps(
        _identity_dict(cfg), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
