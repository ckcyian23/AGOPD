from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_relation_disagreement_allows_hidden_dim_mismatch():
    from agopd.lr_tip import compute_relation_disagreement

    teacher = torch.randn(2, 5, 8)
    student = torch.randn(2, 5, 3)
    scores = compute_relation_disagreement(teacher, student, window=2)

    assert scores.shape == (2, 5)
    assert torch.allclose(scores[:, 0], torch.zeros(2))
    assert torch.isfinite(scores).all()


def test_lr_tip_respects_mask_and_shape():
    from agopd.lr_tip import compute_lr_tip

    teacher_hidden = torch.randn(1, 4, 8)
    student_hidden = torch.randn(1, 4, 6)
    teacher_logits = torch.randn(1, 4, 11)
    student_logits = torch.randn(1, 4, 11)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    scores = compute_lr_tip(
        teacher_hidden,
        student_hidden,
        teacher_logits,
        student_logits,
        attention_mask=mask,
        window=2,
        shift_output_to_labels=True,
    )

    assert scores.importance.shape == (1, 4)
    assert scores.importance[0, 3].item() == 0.0
    assert scores.output_disagreement[0, 0].item() == 0.0
