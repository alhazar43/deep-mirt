"""Trainer skeleton."""

from typing import Iterable, Tuple

import torch

from mirt_dkvmn.utils.metrics import compute_metrics


class Trainer:
    """Minimal trainer with checkpoint-friendly structure."""

    def __init__(self, model, optimizer, loss_fn, device: str = "cpu") -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

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
            probs = outputs[-1]
            logits = torch.log(probs + 1e-8)
            loss = self.loss_fn(logits, responses)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / max(batches, 1)

    @torch.no_grad()
    def evaluate_epoch(self, dataloader: Iterable) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        metrics_accum = {}
        metrics_count = 0

        for batch in dataloader:
            questions, responses = self._unpack_batch(batch)
            questions = questions.to(self.device)
            responses = responses.to(self.device)

            outputs = self.model(questions, responses)
            probs = outputs[-1]
            logits = torch.log(probs + 1e-8)
            loss = self.loss_fn(logits, responses)

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

        avg_loss = total_loss / max(batches, 1)
        avg_metrics = {key: value / max(metrics_count, 1) for key, value in metrics_accum.items()}
        return avg_loss, avg_metrics

    @staticmethod
    def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["questions"], batch["responses"]
        return batch[0], batch[1]
