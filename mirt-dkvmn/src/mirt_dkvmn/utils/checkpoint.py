"""Checkpoint utilities."""

from typing import Any, Dict, Optional
import torch


def save_checkpoint(path: str, model, optimizer, step: int, metrics: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload
