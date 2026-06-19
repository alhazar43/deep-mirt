"""train.py -- Train DeepIRTModel on the EdNet cohort and recover per-learner rates.

Pads sequences to a common max length, trains with a boolean mask, then
recovers the prediction-aligned theta trajectory and fits a learning-curve
rate per learner via fit_rate.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from deep_irt.core.model import DeepIRTModel
from deep_irt.traj_synth.metrics import fit_rate


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def _pad_sequences(
    seqs: list[np.ndarray],
    pad_val: int = 0,
    dtype=np.int32,
    max_len: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad a list of 1D arrays.  Returns (N, T) data and (N, T) mask."""
    T = max_len if max_len is not None else max(len(s) for s in seqs)
    N = len(seqs)
    data = np.full((N, T), pad_val, dtype=dtype)
    mask = np.zeros((N, T), dtype=bool)
    for i, s in enumerate(seqs):
        L = min(len(s), T)
        data[i, :L] = s[:L]
        mask[i, :L] = True
    return data, mask


# ---------------------------------------------------------------------------
# Train and recover
# ---------------------------------------------------------------------------

def train_and_recover(
    seqs_item:  list[np.ndarray],
    seqs_resp:  list[np.ndarray],
    n_items:    int,
    device:     torch.device,
    n_epochs:   int = 300,
    batch_size: int = 256,
    max_seq_len: Optional[int] = None,
    lr: float = 1e-2,
    verbose: bool = True,
) -> tuple[DeepIRTModel, np.ndarray, np.ndarray]:
    """Train DeepIRTModel and return (model, theta_hat, r_hat).

    theta_hat : (N, T_pad) float -- prediction-aligned theta (padded with 0)
    r_hat     : (N,) float -- per-learner fitted rate (nan if fit fails)
    """
    N = len(seqs_item)
    T_max = max_seq_len or max(len(s) for s in seqs_item)

    if verbose:
        print(f"[train] N={N} learners, T_max={T_max}, n_items={n_items}")

    items_np, mask_np = _pad_sequences(seqs_item, pad_val=0,
                                       dtype=np.int32, max_len=T_max)
    resp_np,  _       = _pad_sequences(seqs_resp,  pad_val=0,
                                       dtype=np.int32, max_len=T_max)

    items_t = torch.tensor(items_np, dtype=torch.long)
    resp_t  = torch.tensor(resp_np,  dtype=torch.long)
    mask_t  = torch.tensor(mask_np,  dtype=torch.bool)

    model = DeepIRTModel(
        num_items=n_items,
        n_cats=4,
        decoder="gpcm",
        encoder="lstm",
        decouple=True,
        device=device,
        seed=42,
    )

    t0 = time.time()
    result = model.fit(
        items_t, resp_t,
        n_epochs=n_epochs,
        lr=lr,
        verbose=verbose,
        mask=mask_t,
        batch_size=batch_size,
    )
    wall = time.time() - t0

    if verbose:
        print(f"[train] final loss={result['final_loss']:.4f}  "
              f"train_time={wall:.1f}s")
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_gb = mem.used / 1e9
            print(f"[train] VRAM used ~ {vram_gb:.2f} GB")
        except Exception:
            if device.type == "cuda":
                vram_gb = torch.cuda.memory_allocated(device) / 1e9
                print(f"[train] VRAM allocated ~ {vram_gb:.2f} GB")

    # --- Recover aligned theta in batches to avoid OOM ----------------------
    model.encoder.eval()
    theta_all = np.zeros((N, T_max), dtype=np.float32)
    chunk = batch_size * 2
    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            ids_b  = items_t[start:end].to(device)
            resp_b = resp_t[start:end].to(device)
            th, _ = model.encoder.aligned_theta_and_state(ids_b, resp_b)
            theta_all[start:end] = th.cpu().numpy()

    # Zero out padded positions
    theta_all[~mask_np] = 0.0

    # --- Fit rate per learner -----------------------------------------------
    actual_lens = mask_np.sum(axis=1)  # (N,)
    r_hat = np.full(N, np.nan, dtype=np.float64)
    for i, L in enumerate(actual_lens):
        if L < 20:
            continue
        th_i = theta_all[i, :L].astype(float)
        t_i  = np.arange(L, dtype=float)
        r_hat[i], _, _ = fit_rate(th_i, robust=True, smooth=1, t=t_i)

    frac_fit = np.isfinite(r_hat).mean()
    if verbose:
        print(f"[train] r_hat: {np.isfinite(r_hat).sum()}/{N} fits succeeded "
              f"({frac_fit*100:.1f}%)")
        r_ok = r_hat[np.isfinite(r_hat)]
        if len(r_ok):
            print(f"[train] r_hat median={np.median(r_ok):.4f}  "
                  f"mean={np.mean(r_ok):.4f}  "
                  f"p5={np.percentile(r_ok,5):.4f}  "
                  f"p95={np.percentile(r_ok,95):.4f}")

    return model, theta_all, r_hat
