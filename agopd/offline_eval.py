"""Offline metrics for LR-TIP MVP validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BudgetMetrics:
    budget_fraction: float
    selected_tokens: int
    random_do_mean: float
    random_dr_mean: float
    random_entropy_mean: float
    entropy_only_do_mean: float
    entropy_only_dr_mean: float
    entropy_only_entropy_mean: float
    div_only_do_mean: float
    div_only_dr_mean: float
    div_only_entropy_mean: float
    tip_do_mean: float
    tip_dr_mean: float
    tip_entropy_mean: float
    lr_tip_do_mean: float
    lr_tip_dr_mean: float
    lr_tip_entropy_mean: float
    lr_tip_full_do_mean: float
    lr_tip_full_dr_mean: float
    lr_tip_full_entropy_mean: float
    entropy_tip_overlap: float
    div_tip_overlap: float
    tip_lr_tip_overlap: float
    tip_lr_tip_full_overlap: float


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Offline evaluation requires torch. Install project dependencies "
            "with `uv sync` or install torch manually."
        ) from exc
    return torch


def pearson_corr(x: Any, y: Any, mask: Any | None = None) -> float:
    torch = _require_torch()
    x = x.float()
    y = y.float().to(x.device)
    if mask is not None:
        mask = mask.to(device=x.device, dtype=torch.bool)
        x = x[mask]
        y = y[mask]
    else:
        x = x.reshape(-1)
        y = y.reshape(-1)
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-12))
    return float((x * y).sum().div(denom).cpu())


def topk_mask(scores: Any, mask: Any, fraction: float) -> Any:
    torch = _require_torch()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")

    scores = scores.float()
    mask = mask.to(device=scores.device, dtype=torch.bool)
    selected = torch.zeros_like(mask)
    for row_idx in range(scores.shape[0]):
        valid_count = int(mask[row_idx].sum().item())
        if valid_count == 0:
            continue
        k = max(1, int(round(valid_count * fraction)))
        row_scores = scores[row_idx].masked_fill(~mask[row_idx], float("-inf"))
        indices = torch.topk(row_scores, k=k).indices
        selected[row_idx, indices] = True
    return selected


def random_mask(mask: Any, fraction: float, seed: int = 0) -> Any:
    torch = _require_torch()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")

    mask = mask.bool()
    selected = torch.zeros_like(mask)
    generator = torch.Generator(device=mask.device)
    generator.manual_seed(seed)
    for row_idx in range(mask.shape[0]):
        valid_indices = torch.where(mask[row_idx])[0]
        valid_count = int(valid_indices.numel())
        if valid_count == 0:
            continue
        k = max(1, int(round(valid_count * fraction)))
        perm = torch.randperm(valid_count, generator=generator, device=mask.device)
        selected[row_idx, valid_indices[perm[:k]]] = True
    return selected


def masked_mean(values: Any, mask: Any) -> float:
    torch = _require_torch()
    mask = mask.to(device=values.device, dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        return float("nan")
    return float(values.float()[mask].mean().cpu())


def overlap_ratio(mask_a: Any, mask_b: Any) -> float:
    torch = _require_torch()
    mask_a = mask_a.bool()
    mask_b = mask_b.to(device=mask_a.device, dtype=torch.bool)
    denom = int(torch.minimum(mask_a.sum(), mask_b.sum()).item())
    if denom == 0:
        return float("nan")
    return float((mask_a & mask_b).sum().div(denom).cpu())


def evaluate_budgets(
    output_disagreement: Any,
    relation_disagreement: Any,
    importance: Any,
    token_mask: Any,
    budget_fractions: list[float],
    random_seed: int = 0,
    student_entropy: Any | None = None,
    tip_importance: Any | None = None,
    lr_tip_full_importance: Any | None = None,
) -> list[BudgetMetrics]:
    torch = _require_torch()
    if student_entropy is None:
        student_entropy = torch.zeros_like(output_disagreement)
    if tip_importance is None:
        tip_importance = output_disagreement
    if lr_tip_full_importance is None:
        lr_tip_full_importance = importance

    results: list[BudgetMetrics] = []
    for idx, fraction in enumerate(budget_fractions):
        rand = random_mask(token_mask, fraction, seed=random_seed + idx)
        entropy_only = topk_mask(student_entropy, token_mask, fraction)
        div_only = topk_mask(output_disagreement, token_mask, fraction)
        tip = topk_mask(tip_importance, token_mask, fraction)
        lr_tip = topk_mask(importance, token_mask, fraction)
        lr_tip_full = topk_mask(lr_tip_full_importance, token_mask, fraction)
        results.append(
            BudgetMetrics(
                budget_fraction=fraction,
                selected_tokens=int(lr_tip.sum().item()),
                random_do_mean=masked_mean(output_disagreement, rand),
                random_dr_mean=masked_mean(relation_disagreement, rand),
                random_entropy_mean=masked_mean(student_entropy, rand),
                entropy_only_do_mean=masked_mean(output_disagreement, entropy_only),
                entropy_only_dr_mean=masked_mean(relation_disagreement, entropy_only),
                entropy_only_entropy_mean=masked_mean(student_entropy, entropy_only),
                div_only_do_mean=masked_mean(output_disagreement, div_only),
                div_only_dr_mean=masked_mean(relation_disagreement, div_only),
                div_only_entropy_mean=masked_mean(student_entropy, div_only),
                tip_do_mean=masked_mean(output_disagreement, tip),
                tip_dr_mean=masked_mean(relation_disagreement, tip),
                tip_entropy_mean=masked_mean(student_entropy, tip),
                lr_tip_do_mean=masked_mean(output_disagreement, lr_tip),
                lr_tip_dr_mean=masked_mean(relation_disagreement, lr_tip),
                lr_tip_entropy_mean=masked_mean(student_entropy, lr_tip),
                lr_tip_full_do_mean=masked_mean(output_disagreement, lr_tip_full),
                lr_tip_full_dr_mean=masked_mean(relation_disagreement, lr_tip_full),
                lr_tip_full_entropy_mean=masked_mean(student_entropy, lr_tip_full),
                entropy_tip_overlap=overlap_ratio(entropy_only, tip),
                div_tip_overlap=overlap_ratio(div_only, tip),
                tip_lr_tip_overlap=overlap_ratio(tip, lr_tip),
                tip_lr_tip_full_overlap=overlap_ratio(tip, lr_tip_full),
            )
        )
    return results
