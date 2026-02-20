"""Training and evaluation loop for DeepGPCM.

The ``Trainer`` class wraps a ``DeepGPCM`` model with mask-aware loss
computation, gradient clipping with norm monitoring, NaN/Inf detection,
and optional regularisation penalties.

Design choices:
- ``train_epoch`` and ``evaluate_epoch`` return structured dicts —
  the training script decides what to log/persist.
- Regularisation penalties are separate ``_penalty_*`` methods so they
  can be tested and ablated individually.
- Scheduler step is driven by QWK (``mode='max'``) externally from the
  training script (after ``evaluate_epoch``).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils.metrics import compute_metrics

log = logging.getLogger(__name__)


class Trainer:
    """Manages one training + evaluation epoch for ``DeepGPCM``.

    Args:
        model: ``DeepGPCM`` instance.
        optimizer: PyTorch optimiser (e.g. ``Adam``).
        scheduler: LR scheduler (e.g. ``ReduceLROnPlateau``).
        loss_fn: ``CombinedLoss`` instance (accepts logits + flat targets).
        device: ``torch.device`` to move tensors to.
        grad_clip: Max gradient norm for clipping (default 1.0).
        attention_entropy_weight: Weight for attention entropy penalty.
        theta_norm_weight: Weight for θ ~ N(0,1) penalty.
        alpha_prior_weight: Weight for log(α) ~ N(0, 0.3) penalty.
        beta_prior_weight: Weight for β ~ N(0, 1) penalty.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        loss_fn: nn.Module,
        device: torch.device,
        grad_clip: float = 1.0,
        attention_entropy_weight: float = 0.0,
        theta_norm_weight: float = 0.0,
        alpha_prior_weight: float = 0.0,
        beta_prior_weight: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.grad_clip = grad_clip
        self.attention_entropy_weight = attention_entropy_weight
        self.theta_norm_weight = theta_norm_weight
        self.alpha_prior_weight = alpha_prior_weight
        self.beta_prior_weight = beta_prior_weight

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Run one training epoch.

        Args:
            dataloader: Yields ``(questions, responses, mask)`` tuples
                where each is a ``(B, S)`` tensor.

        Returns:
            Dict with keys:
                ``loss``       float — mean batch loss
                ``accuracy``   float — categorical accuracy on valid tokens
                ``grad_norm``  float — mean clipped gradient norm
        """
        self.model.train()
        total_loss = 0.0
        correct = total = 0
        grad_norms: list[float] = []
        n_batches = 0

        for batch in dataloader:
            questions, responses, mask = self._unpack(batch)
            self.optimizer.zero_grad()

            out = self.model(questions, responses)
            logits: Tensor = out["logits"]   # (B, S, K)
            probs: Tensor = out["probs"]     # (B, S, K)

            # ---- Flatten + mask -------------------------------------------
            valid_logits, valid_targets = self._flatten_mask(logits, responses, mask)
            if valid_logits.numel() == 0:
                continue

            # ---- Primary loss (expects pre-flattened logits) ---------------
            loss = self.loss_fn(valid_logits, valid_targets)

            # ---- Regularisation penalties ----------------------------------
            if self.attention_entropy_weight > 0.0 and self.model.last_attention is not None:
                loss = loss + self.attention_entropy_weight * self._attention_entropy_penalty(
                    self.model.last_attention
                )
            if self.theta_norm_weight > 0.0:
                loss = loss + self.theta_norm_weight * self._theta_norm_penalty(out["theta"])
            if self.alpha_prior_weight > 0.0 or self.beta_prior_weight > 0.0:
                loss = loss + self._item_prior_penalty(out["alpha"], out["beta"])

            # ---- NaN / Inf guard ------------------------------------------
            if not torch.isfinite(loss):
                log.warning("Non-finite loss (%s) — skipping batch", loss.item())
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # ---- Gradient clipping + norm monitoring -----------------------
            raw_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            if not torch.isfinite(raw_norm):
                log.warning("Non-finite grad norm — skipping step")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            grad_norms.append(float(raw_norm.item()))
            self.optimizer.step()

            # ---- Running stats --------------------------------------------
            total_loss += loss.item()
            n_batches += 1

            valid_probs_flat = self._flatten_mask_probs(probs, mask)
            preds = valid_probs_flat.argmax(dim=-1)
            correct += int((preds == valid_targets).sum().item())
            total += valid_targets.numel()

        mean_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        mean_grad = float(sum(grad_norms) / max(len(grad_norms), 1))

        return {"loss": mean_loss, "accuracy": accuracy, "grad_norm": mean_grad}

    # ------------------------------------------------------------------
    # Evaluation epoch
    # ------------------------------------------------------------------

    def evaluate_epoch(self, dataloader: DataLoader) -> dict:
        """Run one evaluation epoch (no gradient computation).

        Args:
            dataloader: Same format as ``train_epoch``.

        Returns:
            Dict with keys:
                ``loss``     float
                ``metrics``  dict from :func:`compute_metrics`
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_probs: list[Tensor] = []
        all_targets: list[Tensor] = []
        all_masks: list[Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                questions, responses, mask = self._unpack(batch)
                out = self.model(questions, responses)
                logits: Tensor = out["logits"]
                probs: Tensor = out["probs"]

                valid_logits, valid_targets = self._flatten_mask(logits, responses, mask)
                if valid_logits.numel() == 0:
                    continue

                loss = self.loss_fn(valid_logits, valid_targets)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    n_batches += 1

                # Collect full (B, S, K) for metric computation
                all_probs.append(probs.cpu())
                all_targets.append(responses.cpu())
                all_masks.append(mask.cpu())

        mean_loss = total_loss / max(n_batches, 1)

        if all_probs:
            # Pad to uniform sequence length before cat
            probs_cat, targets_cat, mask_cat = self._pad_and_cat(
                all_probs, all_targets, all_masks
            )
            metrics = compute_metrics(probs_cat, targets_cat, mask_cat)
        else:
            K = self.model.n_categories
            metrics = {
                "categorical_accuracy": 0.0,
                "ordinal_accuracy": 0.0,
                "qwk": 0.0,
                "mae": 0.0,
                "spearman": 0.0,
                "confusion_matrix": torch.zeros(K, K, dtype=torch.long),
            }

        return {"loss": mean_loss, "metrics": metrics}

    # ------------------------------------------------------------------
    # Regularisation penalties
    # ------------------------------------------------------------------

    def _attention_entropy_penalty(self, attention: Tensor) -> Tensor:
        """Encourage focused memory reads by penalising high entropy.

        Penalty = mean(-sum(w * log(w + 1e-8))).  High entropy
        (uniform weights) is penalised; focused (peaked) weights are
        rewarded.

        Args:
            attention: ``(B, S, M)`` attention weight tensor.

        Returns:
            Scalar penalty tensor.
        """
        eps = 1e-8
        entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)  # (B, S)
        return entropy.mean()

    def _theta_norm_penalty(self, theta: Tensor) -> Tensor:
        """Regularise theta towards a standard normal prior N(0, 1).

        Penalises deviation of the *population distribution* from N(0,1),
        not individual magnitudes.  Two sub-terms:
            mean_pen = E[θ]²         — pulls mean toward 0
            var_pen  = (Var[θ] − 1)² — pulls variance toward 1

        Enforcing unit variance breaks the α-θ scale ambiguity: the model
        cannot set α≡1 and absorb all scale into θ, because θ is anchored
        to unit variance.

        Args:
            theta: ``(B, S, D)`` student ability tensor.

        Returns:
            Scalar penalty tensor.
        """
        theta_flat = theta.reshape(-1)  # (B*S*D,)
        mean_pen = theta_flat.mean() ** 2
        var_pen = (theta_flat.var(unbiased=False) - 1.0) ** 2
        return mean_pen + var_pen

    def _item_prior_penalty(self, alpha: Tensor, beta: Tensor) -> Tensor:
        """Regularise alpha towards log-N(0, 0.3) and beta towards N(0, 1).

        log(alpha) ~ N(0, 0.3) → penalise (log(alpha))^2 / (2 * 0.3^2)
        beta ~ N(0, 1) → penalise beta^2 / 2

        Args:
            alpha: ``(B, S, D)`` discrimination (positive).
            beta:  ``(B, S, K-1)`` thresholds.

        Returns:
            Scalar combined penalty tensor.
        """
        penalty = torch.zeros(1, device=alpha.device, dtype=alpha.dtype).squeeze()

        if self.alpha_prior_weight > 0.0:
            log_alpha = torch.log(alpha.clamp(min=1e-6))
            penalty = penalty + self.alpha_prior_weight * (log_alpha ** 2 / (2 * 0.3 ** 2)).mean()

        if self.beta_prior_weight > 0.0:
            penalty = penalty + self.beta_prior_weight * (beta ** 2 / 2).mean()

        return penalty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack(
        self, batch: object
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Unpack a batch into (questions, responses, mask) on device."""
        if isinstance(batch, (list, tuple)):
            questions, responses, mask = batch[0], batch[1], batch[2]
        else:
            # Dict-style batch (from SequenceDataset / collate_sequences)
            questions = batch["questions"]
            responses = batch["responses"]
            mask = batch["mask"]
        return (
            questions.to(self.device),
            responses.to(self.device),
            mask.to(self.device),
        )

    @staticmethod
    def _flatten_mask(
        logits: Tensor, targets: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Flatten (B, S, K) / (B, S) and select valid positions.

        Returns:
            Tuple ``(valid_logits, valid_targets)`` of shapes
            ``(V, K)`` and ``(V,)`` where V = number of valid tokens.
        """
        B, S, K = logits.shape
        logits_flat = logits.view(-1, K)
        targets_flat = targets.view(-1)
        valid = mask.view(-1).bool()
        return logits_flat[valid], targets_flat[valid]

    @staticmethod
    def _flatten_mask_probs(probs: Tensor, mask: Tensor) -> Tensor:
        """Flatten probs (B, S, K) and select valid positions."""
        K = probs.shape[-1]
        probs_flat = probs.view(-1, K)
        valid = mask.view(-1).bool()
        return probs_flat[valid]

    @staticmethod
    def _pad_and_cat(
        probs_list: list[Tensor],
        targets_list: list[Tensor],
        masks_list: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pad batches to uniform S before concatenating.

        Different batches may have different max sequence lengths due to
        variable-length padding inside ``collate_sequences``.  We pad to
        the global maximum to enable torch.cat along batch dim.
        """
        K = probs_list[0].shape[-1]
        max_s = max(p.shape[1] for p in probs_list)

        padded_probs, padded_targets, padded_masks = [], [], []
        for p, t, m in zip(probs_list, targets_list, masks_list):
            b, s, k = p.shape
            if s < max_s:
                pad_s = max_s - s
                p = torch.cat([p, torch.zeros(b, pad_s, k)], dim=1)
                t = torch.cat([t, torch.zeros(b, pad_s, dtype=torch.long)], dim=1)
                m = torch.cat([m, torch.zeros(b, pad_s, dtype=torch.bool)], dim=1)
            padded_probs.append(p)
            padded_targets.append(t)
            padded_masks.append(m)

        return (
            torch.cat(padded_probs, dim=0),
            torch.cat(padded_targets, dim=0),
            torch.cat(padded_masks, dim=0),
        )
