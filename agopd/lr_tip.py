"""LR-TIP token importance scoring.

The scorer is intentionally independent from any training framework so it can
be used both in a standalone offline validation script and inside an OPD loss
pipeline later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LRTipScores:
    """Container for the raw and fused LR-TIP signals."""

    importance: Any
    output_disagreement: Any
    relation_disagreement: Any
    student_entropy: Any
    output_z: Any
    relation_z: Any
    entropy_norm: Any
    output_norm: Any
    relation_norm: Any
    tip_soft_or: Any
    lr_tip_soft_or: Any


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "LR-TIP scoring requires torch. Install project dependencies with "
            "`uv sync` or install torch manually in the runtime environment."
        ) from exc
    return torch, F


def _full_mask_like(values: Any):
    torch, _ = _require_torch()
    return torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)


def shift_scores_to_label_positions(scores: Any) -> Any:
    """Move causal-LM logits-position scores onto the labels they predict."""

    torch, _ = _require_torch()
    shifted = torch.zeros_like(scores)
    shifted[:, 1:] = scores[:, :-1]
    return shifted


def masked_zscore(values: Any, mask: Any | None = None, eps: float = 1e-6) -> Any:
    """Z-normalize scores over valid tokens in the current batch."""

    torch, _ = _require_torch()
    if mask is None:
        mask = _full_mask_like(values)
    mask = mask.to(device=values.device, dtype=torch.bool)
    values_f = values.float()
    valid = values_f[mask]
    if valid.numel() == 0:
        return torch.zeros_like(values_f)

    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    z = (values_f - mean) / std
    return z.masked_fill(~mask, 0.0)


def masked_minmax(
    values: Any,
    mask: Any | None = None,
    eps: float = 1e-6,
    clip_quantile: float | None = None,
) -> Any:
    """Min-max normalize scores over valid tokens in the current batch."""

    torch, _ = _require_torch()
    if mask is None:
        mask = _full_mask_like(values)
    mask = mask.to(device=values.device, dtype=torch.bool)
    values_f = values.float()
    valid = values_f[mask]
    if valid.numel() == 0:
        return torch.zeros_like(values_f)

    clipped = values_f
    if clip_quantile is not None:
        if not 0.0 < clip_quantile <= 1.0:
            raise ValueError("clip_quantile must be in (0, 1].")
        upper = torch.quantile(valid, clip_quantile)
        clipped = values_f.clamp_max(upper)
        valid = clipped[mask]

    min_value = valid.min()
    max_value = valid.max()
    norm = (clipped - min_value) / (max_value - min_value).clamp_min(eps)
    return norm.masked_fill(~mask, 0.0)


def soft_or(*scores: Any) -> Any:
    """Parameter-free Soft-OR over scores normalized to [0, 1]."""

    if not scores:
        raise ValueError("At least one score tensor is required.")
    result = scores[0]
    for score in scores[1:]:
        result = result + score - result * score
    return result


def compute_output_disagreement(
    logits_teacher: Any,
    logits_student: Any,
    attention_mask: Any | None = None,
    chunk_size: int | None = None,
    direction: str = "forward",
) -> Any:
    """Compute token-level teacher/student KL divergence.

    Args:
        logits_teacher: Teacher logits with shape [B, T, V].
        logits_student: Student logits with shape [B, T, V].
        attention_mask: Optional [B, T] mask.
        chunk_size: Optional number of sequence positions to score per chunk.
            This reduces peak memory when V is large.
        direction: ``forward`` computes KL(P_teacher || P_student), while
            ``reverse`` computes KL(P_student || P_teacher).
    """

    torch, F = _require_torch()
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'.")
    if logits_teacher.shape != logits_student.shape:
        raise ValueError(
            "Teacher and student logits must have the same shape. "
            f"Got {tuple(logits_teacher.shape)} and {tuple(logits_student.shape)}."
        )

    bsz, seq_len, _ = logits_teacher.shape
    if chunk_size is None or chunk_size <= 0:
        chunk_size = seq_len

    scores = torch.empty(
        (bsz, seq_len), dtype=torch.float32, device=logits_teacher.device
    )
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        t_logits = logits_teacher[:, start:end, :].float()
        s_logits = logits_student[:, start:end, :].float()
        log_p_t = F.log_softmax(t_logits, dim=-1)
        log_p_s = F.log_softmax(s_logits, dim=-1)
        if direction == "forward":
            p_t = log_p_t.exp()
            scores[:, start:end] = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
        else:
            p_s = log_p_s.exp()
            scores[:, start:end] = (p_s * (log_p_s - log_p_t)).sum(dim=-1)

    if attention_mask is not None:
        mask = attention_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, 0.0)
    return scores


def compute_student_entropy(
    logits_student: Any,
    attention_mask: Any | None = None,
    chunk_size: int | None = None,
) -> Any:
    """Compute token-level student entropy H(P_student)."""

    torch, F = _require_torch()
    bsz, seq_len, _ = logits_student.shape
    if chunk_size is None or chunk_size <= 0:
        chunk_size = seq_len

    scores = torch.empty(
        (bsz, seq_len), dtype=torch.float32, device=logits_student.device
    )
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = logits_student[:, start:end, :].float()
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        scores[:, start:end] = -(p * log_p).sum(dim=-1)

    if attention_mask is not None:
        mask = attention_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, 0.0)
    return scores


def compute_relation_disagreement(
    hidden_teacher: Any,
    hidden_student: Any,
    attention_mask: Any | None = None,
    window: int = 16,
    huber_delta: float = 1.0,
) -> Any:
    """Compute local relational discrepancy D_R.

    Hidden dimensions may differ between teacher and student. Cosine relation
    profiles are computed within each model's own representation space, then
    compared at the scalar relation level.
    """

    torch, F = _require_torch()
    if hidden_teacher.ndim != 3 or hidden_student.ndim != 3:
        raise ValueError("Hidden states must have shape [B, T, D].")
    if hidden_teacher.shape[:2] != hidden_student.shape[:2]:
        raise ValueError(
            "Teacher and student hidden states must share [B, T]. "
            f"Got {tuple(hidden_teacher.shape)} and {tuple(hidden_student.shape)}."
        )
    if window < 1:
        raise ValueError("window must be >= 1.")

    bsz, seq_len = hidden_teacher.shape[:2]
    device = hidden_teacher.device
    if hidden_student.device != device:
        hidden_student = hidden_student.to(device)

    h_t = F.layer_norm(hidden_teacher.float(), hidden_teacher.shape[-1:])
    h_s = F.layer_norm(hidden_student.float(), hidden_student.shape[-1:])
    h_t = F.normalize(h_t, p=2, dim=-1)
    h_s = F.normalize(h_s, p=2, dim=-1)

    scores = torch.zeros((bsz, seq_len), dtype=torch.float32, device=device)
    for idx in range(1, seq_len):
        start = max(0, idx - window)
        teacher_rel = (h_t[:, start:idx, :] * h_t[:, idx : idx + 1, :]).sum(dim=-1)
        student_rel = (h_s[:, start:idx, :] * h_s[:, idx : idx + 1, :]).sum(dim=-1)
        diff = F.huber_loss(
            teacher_rel, student_rel, reduction="none", delta=huber_delta
        )
        scores[:, idx] = diff.mean(dim=-1)

    if attention_mask is not None:
        mask = attention_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, 0.0)
    return scores


def compute_lr_tip(
    hidden_teacher: Any,
    hidden_student: Any,
    logits_teacher: Any,
    logits_student: Any,
    attention_mask: Any | None = None,
    window: int = 16,
    alpha: float = 0.5,
    beta: float = 0.5,
    output_chunk_size: int | None = None,
    shift_output_to_labels: bool = False,
    huber_delta: float = 1.0,
    entropy_clip_quantile: float | None = 0.98,
    output_kl_direction: str = "forward",
) -> LRTipScores:
    """Compute fused LR-TIP importance.

    Set ``shift_output_to_labels=True`` for causal language models when the
    token mask is expressed over label/input token positions rather than logits
    positions.
    """

    do = compute_output_disagreement(
        logits_teacher,
        logits_student,
        attention_mask=attention_mask,
        chunk_size=output_chunk_size,
        direction=output_kl_direction,
    )
    if shift_output_to_labels:
        do = shift_scores_to_label_positions(do)

    entropy = compute_student_entropy(
        logits_student,
        attention_mask=attention_mask,
        chunk_size=output_chunk_size,
    )
    if shift_output_to_labels:
        entropy = shift_scores_to_label_positions(entropy)

    dr = compute_relation_disagreement(
        hidden_teacher,
        hidden_student,
        attention_mask=attention_mask,
        window=window,
        huber_delta=huber_delta,
    )

    output_z = masked_zscore(do, attention_mask)
    relation_z = masked_zscore(dr, attention_mask)
    importance = alpha * output_z + beta * relation_z
    entropy_norm = masked_minmax(
        entropy, attention_mask, clip_quantile=entropy_clip_quantile
    )
    output_norm = masked_minmax(do, attention_mask)
    relation_norm = masked_minmax(dr, attention_mask)
    tip_soft_or = soft_or(entropy_norm, output_norm)
    lr_tip_soft_or = soft_or(entropy_norm, output_norm, relation_norm)
    if attention_mask is not None:
        torch, _ = _require_torch()
        mask = attention_mask.to(device=importance.device, dtype=torch.bool)
        importance = importance.masked_fill(~mask, 0.0)
        tip_soft_or = tip_soft_or.masked_fill(~mask, 0.0)
        lr_tip_soft_or = lr_tip_soft_or.masked_fill(~mask, 0.0)

    return LRTipScores(
        importance=importance,
        output_disagreement=do,
        relation_disagreement=dr,
        student_entropy=entropy,
        output_z=output_z,
        relation_z=relation_z,
        entropy_norm=entropy_norm,
        output_norm=output_norm,
        relation_norm=relation_norm,
        tip_soft_or=tip_soft_or,
        lr_tip_soft_or=lr_tip_soft_or,
    )
