"""Quick evaluation script: load checkpoint, compute all metrics including Kendall's τ.

Usage (from kt-gpcm/):
    PYTHONPATH=src python scripts/eval_metrics.py --config configs/deepgpcm_k5_s42.yaml \
        --checkpoint outputs/deepgpcm_k5_s42/best.pt

Metrics reported
----------------
Standard predictive metrics (via compute_metrics):
    acc, qwk, spearman, kendall_tau, mae

IRT recovery diagnostic:
    pct_disordered_betas — percentage of items whose per-item average β vector
    has at least one disordered adjacent pair (β_k > β_{k+1}).  For a model
    trained with monotonic_betas=True this is always 0%.  For unconstrained
    models (monotonic_betas=False) this measures how often the model fails to
    respect threshold ordering.  Requires K >= 3; reported as "N/A" for K=2.

    Method: for each (student, timestep, item) observation, accumulate the
    predicted beta vector and the question ID.  After the full validation pass,
    compute the per-item mean beta by averaging across all (B*S) observations
    that share the same question ID.  Then count items with any disorder.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule
from kt_gpcm.models.kt_gpcm import DeepGPCM
from kt_gpcm.models.dkvmn_softmax import DKVMNSoftmax
from kt_gpcm.models.static_gpcm import StaticGPCM
from kt_gpcm.models.dynamic_gpcm import DynamicGPCM
from kt_gpcm.utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")

    data_mgr = DataModule(cfg)
    _, val_loader = data_mgr.build()

    model_kwargs = {k: v for k, v in vars(cfg.model).items() if k != "model_type"}
    model_type = getattr(cfg.model, "model_type", "deepgpcm")
    if model_type == "dkvmn_softmax":
        model = DKVMNSoftmax(**model_kwargs).to(device)
    elif model_type == "static_gpcm":
        model = StaticGPCM(n_students=data_mgr.n_students, **model_kwargs).to(device)
        model._model_type = "static_gpcm"
    elif model_type == "dynamic_gpcm":
        model = DynamicGPCM(n_students=data_mgr.n_students, **model_kwargs).to(device)
        model._model_type = "dynamic_gpcm"
    else:
        model = DeepGPCM(**model_kwargs).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_probs, all_targets, all_masks = [], [], []
    # For disordered-beta metric: accumulate per-item beta sums and counts.
    # beta_sum[q] and beta_count[q] indexed by 1-based question ID.
    n_q = cfg.model.n_questions
    K = cfg.model.n_categories
    beta_sum = torch.zeros(n_q + 1, K - 1)   # index 0 unused (padding)
    beta_count = torch.zeros(n_q + 1)

    with torch.no_grad():
        for batch in val_loader:
            questions, responses, mask = batch[0], batch[1], batch[2]
            student_ids = batch[3] if len(batch) > 3 else None
            questions = questions.to(device)
            responses = responses.to(device)
            if model_type in ("static_gpcm", "dynamic_gpcm") and student_ids is not None:
                out = model(student_ids.to(device), questions, responses)
            else:
                out = model(questions, responses)
            all_probs.append(out["probs"].cpu())
            all_targets.append(responses.cpu())
            all_masks.append(mask.cpu())

            # Accumulate beta vectors per item if the model produces them.
            if "beta" in out and K >= 3:
                beta_cpu = out["beta"].cpu()          # (B, S, K-1)
                q_cpu = questions.cpu()               # (B, S) — 1-based IDs
                B_b, S_b = q_cpu.shape
                beta_flat = beta_cpu.view(B_b * S_b, K - 1)
                q_flat = q_cpu.view(B_b * S_b)
                # Exclude padding (question ID 0)
                valid_mask = q_flat > 0
                q_valid = q_flat[valid_mask]
                b_valid = beta_flat[valid_mask]
                beta_sum.index_add_(0, q_valid, b_valid)
                beta_count.index_add_(0, q_valid, torch.ones(q_valid.shape[0]))

    # Pad to uniform length
    max_len = max(p.shape[1] for p in all_probs)
    K = all_probs[0].shape[-1]

    def pad(t, fill=0):
        B, S = t.shape[:2]
        if S == max_len:
            return t
        pad_shape = (B, max_len - S) + t.shape[2:]
        return torch.cat([t, torch.full(pad_shape, fill, dtype=t.dtype)], dim=1)

    probs_cat = torch.cat([pad(p, 0.0) for p in all_probs], dim=0)
    targets_cat = torch.cat([pad(t, 0) for t in all_targets], dim=0)
    masks_cat = torch.cat([pad(m, False) for m in all_masks], dim=0)

    m = compute_metrics(probs_cat, targets_cat, masks_cat)

    # ---- Disordered-beta metric -----------------------------------------------
    # Compute the fraction of items whose mean predicted β vector has at least one
    # disordered adjacent pair (β_k > β_{k+1}).  Defined only for K >= 3.
    disorder_str = ""
    if K >= 3:
        # Items seen at least once (skip padding slot 0)
        seen = beta_count[1:] > 0                           # (n_q,)
        if seen.any():
            # Per-item mean beta: (n_items_seen, K-1)
            mean_beta = beta_sum[1:][seen] / beta_count[1:][seen].unsqueeze(-1)
            # Disordered if any gap β_{k+1} - β_k <= 0  (i.e., not strictly increasing)
            gaps = mean_beta[:, 1:] - mean_beta[:, :-1]    # (n_items_seen, K-2)
            n_disordered = int((gaps <= 0).any(dim=-1).sum().item())
            n_seen = int(seen.sum().item())
            pct = 100.0 * n_disordered / n_seen
            disorder_str = f"  pct_disordered_betas={pct:.1f}% ({n_disordered}/{n_seen} items)"
        else:
            disorder_str = "  pct_disordered_betas=N/A (no items seen)"
    else:
        disorder_str = "  pct_disordered_betas=N/A (K<3)"

    # AUC for K=2 (binary case)
    auc_str = ""
    if K == 2:
        valid = masks_cat.view(-1).bool()
        scores = probs_cat.view(-1, K)[valid, 1]
        labels = targets_cat.view(-1)[valid].long()
        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        if n_pos > 0 and n_neg > 0:
            order = scores.argsort(descending=True)
            sl = labels[order].float()
            tp = sl.cumsum(0)
            fp = (1 - sl).cumsum(0)
            tpr = torch.cat([torch.zeros(1), tp / n_pos])
            fpr = torch.cat([torch.zeros(1), fp / n_neg])
            auc = float(torch.trapz(tpr, fpr).item())
            auc_str = f"  auc={auc:.4f}"

    print(f"acc={m['categorical_accuracy']:.4f}  qwk={m['qwk']:.4f}  "
          f"spearman={m['spearman']:.4f}  kendall_tau={m['kendall_tau']:.4f}  "
          f"mae={m['mae']:.4f}{auc_str}{disorder_str}")


if __name__ == "__main__":
    main()
