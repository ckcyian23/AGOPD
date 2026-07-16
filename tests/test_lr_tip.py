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
    assert scores.student_entropy.shape == (1, 4)
    assert scores.tip_soft_or.shape == (1, 4)
    assert scores.lr_tip_soft_or.shape == (1, 4)
    assert scores.importance[0, 3].item() == 0.0
    assert scores.output_disagreement[0, 0].item() == 0.0
    assert scores.student_entropy[0, 0].item() == 0.0
    assert scores.tip_soft_or[0, 3].item() == 0.0
    assert scores.lr_tip_soft_or[0, 3].item() == 0.0
    assert torch.all((scores.tip_soft_or >= 0.0) & (scores.tip_soft_or <= 1.0))
    assert torch.all((scores.lr_tip_soft_or >= 0.0) & (scores.lr_tip_soft_or <= 1.0))


def test_soft_or_is_parameter_free_union_score():
    from agopd.lr_tip import soft_or

    a = torch.tensor([[0.0, 0.5, 1.0]])
    b = torch.tensor([[0.0, 0.5, 1.0]])

    scores = soft_or(a, b)

    assert torch.allclose(scores, torch.tensor([[0.0, 0.75, 1.0]]))


def test_lr_tip_full_modes_are_selectable():
    from agopd.lr_tip import compute_lr_tip

    teacher_hidden = torch.randn(1, 5, 8)
    student_hidden = torch.randn(1, 5, 6)
    teacher_logits = torch.randn(1, 5, 13)
    student_logits = torch.randn(1, 5, 13)
    mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)

    soft_or = compute_lr_tip(
        teacher_hidden,
        student_hidden,
        teacher_logits,
        student_logits,
        attention_mask=mask,
        lr_tip_full_mode="soft_or",
    )
    add = compute_lr_tip(
        teacher_hidden,
        student_hidden,
        teacher_logits,
        student_logits,
        attention_mask=mask,
        lr_tip_full_mode="add",
        relation_lambda=0.5,
    )
    gated = compute_lr_tip(
        teacher_hidden,
        student_hidden,
        teacher_logits,
        student_logits,
        attention_mask=mask,
        lr_tip_full_mode="gated",
        relation_lambda=0.5,
    )

    assert torch.allclose(soft_or.lr_tip_full, soft_or.lr_tip_soft_or)
    assert torch.allclose(add.lr_tip_full, add.lr_tip_add)
    assert torch.allclose(gated.lr_tip_full, gated.lr_tip_gated)
    assert soft_or.lr_tip_full[0, 4].item() == 0.0
    assert add.lr_tip_full[0, 4].item() == 0.0
    assert gated.lr_tip_full[0, 4].item() == 0.0

    with pytest.raises(ValueError):
        compute_lr_tip(
            teacher_hidden,
            student_hidden,
            teacher_logits,
            student_logits,
            lr_tip_full_mode="bad",
        )


def test_output_disagreement_supports_kl_direction():
    from agopd.lr_tip import compute_output_disagreement

    teacher_logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    student_logits = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])

    forward = compute_output_disagreement(
        teacher_logits, student_logits, direction="forward"
    )
    reverse = compute_output_disagreement(
        teacher_logits, student_logits, direction="reverse"
    )

    assert forward.shape == (1, 2)
    assert reverse.shape == (1, 2)
    assert torch.isfinite(forward).all()
    assert torch.isfinite(reverse).all()
    with pytest.raises(ValueError):
        compute_output_disagreement(teacher_logits, student_logits, direction="bad")


def test_select_prompt_shard_interleaves_prompts():
    from agopd.experiments.offline_lr_tip_eval import select_prompt_shard

    prompts = [f"prompt-{idx}" for idx in range(7)]

    assert select_prompt_shard(prompts, num_shards=2, shard_index=0) == [
        "prompt-0",
        "prompt-2",
        "prompt-4",
        "prompt-6",
    ]
    assert select_prompt_shard(prompts, num_shards=2, shard_index=1) == [
        "prompt-1",
        "prompt-3",
        "prompt-5",
    ]


def test_select_prompt_shard_validates_index():
    from agopd.experiments.offline_lr_tip_eval import select_prompt_shard

    with pytest.raises(ValueError):
        select_prompt_shard(["a"], num_shards=2, shard_index=2)
