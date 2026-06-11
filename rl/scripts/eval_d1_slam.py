"""D1 SLAM binary-collapsed evaluation: AUC and log-loss.

For the SLAM K=3 ordinal model, binary collapse is:
  positive class = category 2 (all-correct exercise)
  negative class = categories 0 or 1 (at least one mistake)

P(positive) = model probability of category 2 (index 2 in [0,1,2]).

This mirrors the official SLAM 2018 shared-task evaluation, which
reports AUC on binary correct/incorrect at the token level.  Our
collapse is at the exercise level; the caveat is noted in the results.

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \\
        python rl/scripts/eval_d1_slam.py \\
            --config ma-irt/configs/ordrec_slam_k3.yaml \\
            --checkpoint ma-irt/outputs/ordrec_slam_k3/best.pt \\
            --data-dir ma-irt/data/ordrec_slam_k3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

# Allow running from repo root with PYTHONPATH="rl/src;ma-irt"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ma-irt"))

from models import build_model
from utils.config import load_config
from utils.dataloader import DataModule


def run_binary_eval(
    checkpoint_path: Path,
    config_path: Path,
    data_dir: Path,
) -> dict[str, float]:
    """Return binary-collapsed AUC and log-loss over the test split.

    Args:
        checkpoint_path: Path to best.pt from ma-irt training.
        config_path:     Path to the YAML experiment config.
        data_dir:        Directory with sequences.json / metadata.json.

    Returns:
        Dict with keys: auc, log_loss, n_observations.
    """
    cfg = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (DataModule updates cfg.model.n_questions/n_categories from metadata)
    # data_dir is the dataset leaf dir (e.g., ma-irt/data/ordrec_slam_k3).
    # DataModule expects base_dir = parent of dataset_name.
    cfg.data.dataset_name = data_dir.name
    dm = DataModule(cfg, base_dir=str(data_dir.parent))
    dm.build()

    # Build model using unpacked ModelConfig kwargs (same path as train.py)
    model = build_model(cfg, device)
    ck = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    test_loader = dm.test_loader

    needs_sid = getattr(model, "needs_student_id", False)

    all_trues: list[int] = []
    all_p_correct: list[float] = []  # P(cat=2)

    with torch.no_grad():
        for batch in test_loader:
            questions = batch[0].to(device)
            responses = batch[1].to(device)
            mask = batch[2].to(device)
            student_ids = batch[3].to(device)

            if needs_sid:
                out = model(student_ids, questions, responses)
            else:
                out = model(questions, responses)

            probs = out["probs"]          # (B, S, K)
            p_top = probs[..., -1]        # P(cat = K-1 = 2)

            m_np = mask.cpu().numpy()
            r_np = responses.cpu().numpy()
            p_np = p_top.cpu().numpy()

            B, S = m_np.shape
            for b in range(B):
                for t in range(S):
                    if m_np[b, t]:
                        all_trues.append(int(r_np[b, t]))
                        all_p_correct.append(float(p_np[b, t]))

    y_true = np.array(all_trues)
    y_score = np.array(all_p_correct)

    # Binary collapse: 1 if cat==2, else 0
    y_bin = (y_true == 2).astype(int)

    auc = float("nan")
    ll = float("nan")

    if len(np.unique(y_bin)) >= 2:
        try:
            auc = float(roc_auc_score(y_bin, y_score))
        except ValueError:
            pass
        try:
            # log_loss expects P(class=1) when binary
            ll = float(log_loss(y_bin, y_score))
        except ValueError:
            pass

    return {
        "auc": auc,
        "log_loss": ll,
        "n_observations": int(len(y_bin)),
        "frac_positive": float(y_bin.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D1 SLAM binary-collapsed AUC and log-loss."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir:   {args.data_dir}")
    print()

    results = run_binary_eval(args.checkpoint, args.config, args.data_dir)

    print("=== D1 SLAM binary-collapsed metrics (exercise level) ===")
    print(f"  AUC              = {results['auc']:.4f}")
    print(f"  Log-loss         = {results['log_loss']:.4f}")
    print(f"  N observations   = {results['n_observations']:,}")
    print(f"  Frac positive    = {results['frac_positive']:.4f}  (cat-2 rate)")
    print()
    print("Caveat: AUC/log-loss collapse K=3 categories to binary")
    print("(cat 2 = all-correct exercise, cats 0+1 = any mistake).")
    print("SLAM 2018 baseline AUC is token-level; these are exercise-level.")

    out = args.output
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
