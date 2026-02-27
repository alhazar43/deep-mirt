"""Quick evaluation script: load checkpoint, compute all metrics including Kendall's τ.

Usage (from kt-gpcm/):
    PYTHONPATH=src python scripts/eval_metrics.py --config configs/deepgpcm_k5_s42.yaml \
        --checkpoint outputs/deepgpcm_k5_s42/best.pt
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
          f"mae={m['mae']:.4f}{auc_str}")


if __name__ == "__main__":
    main()
