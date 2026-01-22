"""Shape checks for model output."""

import torch
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT


def test_output_shapes():
    model = DKVMNMIRT(n_questions=10, n_cats=4, n_traits=3)
    questions = torch.randint(1, 10, (2, 5))
    responses = torch.randint(0, 4, (2, 5))

    theta, beta, alpha, probs = model(questions, responses)

    assert theta.shape == (2, 5, 3)
    assert alpha.shape == (2, 5, 3)
    assert beta.shape == (2, 5, 3)
    assert probs.shape == (2, 5, 4)


def test_output_shapes_concept_aligned():
    model = DKVMNMIRT(
        n_questions=10,
        n_cats=4,
        n_traits=2,
        memory_size=5,
        value_dim=2,
        concept_aligned_memory=True,
    )
    questions = torch.randint(1, 10, (2, 5))
    responses = torch.randint(0, 4, (2, 5))

    theta, beta, alpha, probs = model(questions, responses)

    assert theta.shape == (2, 5, 2)
    assert alpha.shape == (2, 5, 2)
    assert beta.shape == (2, 5, 3)
    assert probs.shape == (2, 5, 4)
