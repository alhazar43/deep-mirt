"""Trainer skeleton."""

from typing import Iterable, Tuple

import torch

from mirt_dkvmn.utils.metrics import compute_metrics, confusion_matrix


class Trainer:
    """Minimal trainer with checkpoint-friendly structure."""

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cpu",
        attention_entropy_weight: float = 0.0,
        theta_norm_weight: float = 0.0,
        alpha_prior_weight: float = 0.0,
        beta_prior_weight: float = 0.0,
        alpha_norm_weight: float = 0.0,
        alpha_norm_target: float = 1.0,
        alpha_ortho_weight: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.attention_entropy_weight = attention_entropy_weight
        self.theta_norm_weight = theta_norm_weight
        self.alpha_prior_weight = alpha_prior_weight
        self.beta_prior_weight = beta_prior_weight
        self.alpha_norm_weight = alpha_norm_weight
        self.alpha_norm_target = alpha_norm_target
        self.alpha_ortho_weight = alpha_ortho_weight

    def train_epoch(self, dataloader: Iterable) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0

        for batch in dataloader:
            questions, responses = self._unpack_batch(batch)
            questions = questions.to(self.device)
            responses = responses.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(questions, responses)
            theta = outputs[0]
            beta = outputs[1]
            alpha = outputs[2]
            probs = outputs[-1]
            logits = torch.log(probs + 1e-8)
            loss = self.loss_fn(logits, responses)
            loss = loss + self._attention_entropy_penalty()
            loss = loss + self._theta_norm_penalty(theta)
            loss = loss + self._item_prior_penalty(alpha, beta)
            loss = loss + self._alpha_norm_penalty(alpha)
            loss = loss + self._alpha_ortho_penalty(alpha)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / max(batches, 1)

    @torch.no_grad()
    def evaluate_epoch(self, dataloader: Iterable) -> tuple[float, dict, torch.Tensor]:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        metrics_accum = {}
        metrics_count = 0
        conf_sum = None

        for batch in dataloader:
            questions, responses = self._unpack_batch(batch)
            questions = questions.to(self.device)
            responses = responses.to(self.device)

            outputs = self.model(questions, responses)
            theta = outputs[0]
            beta = outputs[1]
            alpha = outputs[2]
            probs = outputs[-1]
            logits = torch.log(probs + 1e-8)
            loss = self.loss_fn(logits, responses)
            loss = loss + self._attention_entropy_penalty()
            loss = loss + self._theta_norm_penalty(theta)
            loss = loss + self._item_prior_penalty(alpha, beta)
            loss = loss + self._alpha_norm_penalty(alpha)
            loss = loss + self._alpha_ortho_penalty(alpha)

            total_loss += loss.item()
            batches += 1

            mask = batch.get("mask") if isinstance(batch, dict) else None
            if mask is not None:
                mask = mask.to(self.device)
            batch_metrics = compute_metrics(probs, responses, mask)
            for key, value in batch_metrics.items():
                if not torch.isfinite(torch.tensor(value)):
                    continue
                metrics_accum[key] = metrics_accum.get(key, 0.0) + value
            metrics_count += 1
            batch_conf = confusion_matrix(
                probs.argmax(dim=-1), responses, probs.size(-1), mask
            )
            if conf_sum is None:
                conf_sum = torch.tensor(batch_conf, dtype=torch.long)
            else:
                conf_sum += torch.tensor(batch_conf, dtype=torch.long)

        avg_loss = total_loss / max(batches, 1)
        avg_metrics = {key: value / max(metrics_count, 1) for key, value in metrics_accum.items()}
        if conf_sum is None:
            conf_sum = torch.zeros((probs.size(-1), probs.size(-1)), dtype=torch.long)
        return avg_loss, avg_metrics, conf_sum

    def _attention_entropy_penalty(self) -> torch.Tensor:
        if self.attention_entropy_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        attn = getattr(self.model, "last_attention", None)
        if attn is None:
            return torch.tensor(0.0, device=self.device)
        eps = 1e-8
        ent = -(attn * torch.log(attn + eps)).sum(dim=-1).mean()
        return self.attention_entropy_weight * ent

    def _theta_norm_penalty(self, theta: torch.Tensor) -> torch.Tensor:
        if self.theta_norm_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        mean = theta.mean(dim=(0, 1))
        std = theta.std(dim=(0, 1))
        penalty = torch.mean(mean.pow(2)) + torch.mean((std - 1.0).pow(2))
        return self.theta_norm_weight * penalty

    def _item_prior_penalty(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if self.alpha_prior_weight <= 0 and self.beta_prior_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)

        if self.alpha_prior_weight > 0:
            log_alpha = torch.log(alpha + 1e-8)
            mean = log_alpha.mean()
            std = log_alpha.std()
            penalty = penalty + self.alpha_prior_weight * (mean.pow(2) + (std - 0.3).pow(2))

        if self.beta_prior_weight > 0:
            mean = beta.mean()
            std = beta.std()
            penalty = penalty + self.beta_prior_weight * (mean.pow(2) + (std - 1.0).pow(2))

        return penalty

    def _alpha_norm_penalty(self, alpha: torch.Tensor) -> torch.Tensor:
        if self.alpha_norm_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        norms = torch.linalg.norm(alpha, dim=-1)
        target = torch.tensor(self.alpha_norm_target, device=self.device)
        return self.alpha_norm_weight * (norms.mean() - target).pow(2)

    def _alpha_ortho_penalty(self, alpha: torch.Tensor) -> torch.Tensor:
        if self.alpha_ortho_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        flat = alpha.reshape(-1, alpha.shape[-1])
        flat = flat - flat.mean(dim=0, keepdim=True)
        cov = (flat.T @ flat) / max(flat.shape[0] - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return self.alpha_ortho_weight * torch.mean(off_diag.pow(2))

    @staticmethod
    def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["questions"], batch["responses"]
        return batch[0], batch[1]
