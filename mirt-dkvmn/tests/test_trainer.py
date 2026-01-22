"""Trainer smoke test."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT
from mirt_dkvmn.training.losses import OrdinalCrossEntropy
from mirt_dkvmn.training.trainer import Trainer


def test_trainer_epoch_runs():
    model = DKVMNMIRT(n_questions=10, n_cats=4, n_traits=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = OrdinalCrossEntropy()

    questions = torch.randint(1, 10, (4, 6))
    responses = torch.randint(0, 4, (4, 6))
    loader = DataLoader(TensorDataset(questions, responses), batch_size=2)

    trainer = Trainer(model, optimizer, loss_fn, device="cpu")
    loss = trainer.train_epoch(loader)
    val_loss, metrics = trainer.evaluate_epoch(loader)

    assert isinstance(loss, float)
    assert isinstance(val_loss, float)
    assert "qwk" in metrics

