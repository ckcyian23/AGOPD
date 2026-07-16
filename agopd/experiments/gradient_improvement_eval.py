"""Evaluate whether LR-TIP scores predict gradient-norm distillation gain.

This is intentionally a pilot-scale script. The exact proxy computes one
student backward pass per scored token, so running it over every token in a
large benchmark is expensive.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from agopd.experiments.offline_lr_tip_eval import (
    forward_tensors,
    generate_rollout,
    import_runtime,
    load_model,
    load_prompts,
    make_tokenizer,
    model_device,
    parse_budgets,
    resolve_device,
    resolve_dtype,
    select_prompt_shard,
)
from agopd.lr_tip import compute_lr_tip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default="models/Qwen/Qwen3-1.7B")
    parser.add_argument("--student-model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts-file", type=Path, default=Path("data/real_prompts_512.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lr_tip_grad_improvement"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tokens-per-prompt", type=int, default=16)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--layer-index", type=int, default=-2)
    parser.add_argument("--budgets", default="0.05,0.1,0.2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--device-map",
        default="none",
        help="Pass e.g. 'auto' for accelerate device_map. Use 'none' for .to(device).",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-chunk-size", type=int, default=8)
    parser.add_argument(
        "--output-kl-direction",
        default="forward",
        choices=["forward", "reverse"],
    )
    parser.add_argument(
        "--lr-tip-full-mode",
        default="soft_or",
        choices=["soft_or", "add", "gated"],
    )
    parser.add_argument("--relation-gamma", type=float, default=1.0)
    parser.add_argument("--relation-lambda", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    x_centered = [x - x_mean for x, _ in pairs]
    y_centered = [y - y_mean for _, y in pairs]
    denom = math.sqrt(sum(x * x for x in x_centered) * sum(y * y for y in y_centered))
    if denom <= 1e-12:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denom


def sample_token_positions(
    token_mask: Any,
    tokens_per_prompt: int,
    seed: int,
) -> list[int]:
    valid = [idx for idx, keep in enumerate(token_mask[0].tolist()) if keep and idx > 0]
    if tokens_per_prompt <= 0 or len(valid) <= tokens_per_prompt:
        return valid
    rng = random.Random(seed)
    return sorted(rng.sample(valid, tokens_per_prompt))


def gradient_norm_for_label_position(
    torch: Any,
    student: Any,
    input_ids: Any,
    attention_mask: Any,
    teacher_logits: Any,
    label_position: int,
) -> float:
    import torch.nn.functional as F

    if label_position < 1:
        return 0.0

    pred_position = label_position - 1
    device = model_device(student)
    prefix_ids = input_ids[:, : label_position].to(device)
    prefix_mask = attention_mask[:, : label_position].to(device)
    teacher_token_logits = teacher_logits[:, pred_position, :].to(device)

    student.zero_grad(set_to_none=True)
    output = student(
        input_ids=prefix_ids,
        attention_mask=prefix_mask,
        use_cache=False,
    )
    student_logp = F.log_softmax(output.logits[:, -1, :].float(), dim=-1)
    teacher_p = F.softmax(teacher_token_logits.float(), dim=-1)
    loss = F.kl_div(student_logp, teacher_p, reduction="batchmean")
    loss.backward()

    total = 0.0
    for param in student.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().square().sum().cpu())
    student.zero_grad(set_to_none=True)
    return total


def select_top_mean(rows: list[dict[str, Any]], score_key: str, value_key: str, fraction: float) -> float:
    rows = [row for row in rows if score_key in row and value_key in row]
    if not rows:
        return float("nan")
    k = max(1, int(round(len(rows) * fraction)))
    selected = sorted(rows, key=lambda item: item[score_key], reverse=True)[:k]
    return sum(float(item[value_key]) for item in selected) / len(selected)


def random_mean(rows: list[dict[str, Any]], value_key: str, fraction: float, seed: int) -> float:
    if not rows:
        return float("nan")
    k = max(1, int(round(len(rows) * fraction)))
    rng = random.Random(seed)
    selected = rng.sample(rows, k)
    return sum(float(item[value_key]) for item in selected) / len(selected)


def aggregate_rows(rows: list[dict[str, Any]], budgets: list[float], seed: int) -> dict[str, Any]:
    improvement = [row["grad_improvement"] for row in rows]
    aggregate: dict[str, Any] = {
        "num_tokens": len(rows),
        "num_prompts": len(
            {
                (row.get("merge_source_index", 0), row.get("shard_index", 0), row["prompt_index"])
                for row in rows
            }
        ),
        "mean_grad_improvement": (
            sum(improvement) / len(improvement) if improvement else None
        ),
        "corr_output_improvement": pearson(
            [row["output_disagreement"] for row in rows], improvement
        ),
        "corr_relation_improvement": pearson(
            [row["relation_disagreement"] for row in rows], improvement
        ),
        "corr_entropy_improvement": pearson(
            [row["student_entropy"] for row in rows], improvement
        ),
        "corr_tip_improvement": pearson([row["tip_soft_or"] for row in rows], improvement),
        "corr_lr_tip_soft_or_improvement": pearson(
            [row.get("lr_tip_soft_or", row["lr_tip_full"]) for row in rows],
            improvement,
        ),
        "corr_lr_tip_add_improvement": pearson(
            [row.get("lr_tip_add", row["lr_tip_full"]) for row in rows],
            improvement,
        ),
        "corr_lr_tip_gated_improvement": pearson(
            [row.get("lr_tip_gated", row["lr_tip_full"]) for row in rows],
            improvement,
        ),
        "corr_lr_tip_full_improvement": pearson(
            [row["lr_tip_full"] for row in rows], improvement
        ),
        "budgets": {},
    }
    for idx, budget in enumerate(budgets):
        aggregate["budgets"][str(budget)] = {
            "random_grad_improvement_mean": random_mean(
                rows, "grad_improvement", budget, seed + idx
            ),
            "entropy_grad_improvement_mean": select_top_mean(
                rows, "student_entropy", "grad_improvement", budget
            ),
            "div_grad_improvement_mean": select_top_mean(
                rows, "output_disagreement", "grad_improvement", budget
            ),
            "relation_grad_improvement_mean": select_top_mean(
                rows, "relation_disagreement", "grad_improvement", budget
            ),
            "tip_grad_improvement_mean": select_top_mean(
                rows, "tip_soft_or", "grad_improvement", budget
            ),
            "lr_tip_soft_or_grad_improvement_mean": select_top_mean(
                rows, "lr_tip_soft_or", "grad_improvement", budget
            ),
            "lr_tip_add_grad_improvement_mean": select_top_mean(
                rows, "lr_tip_add", "grad_improvement", budget
            ),
            "lr_tip_gated_grad_improvement_mean": select_top_mean(
                rows, "lr_tip_gated", "grad_improvement", budget
            ),
            "lr_tip_full_grad_improvement_mean": select_top_mean(
                rows, "lr_tip_full", "grad_improvement", budget
            ),
        }
    return aggregate


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    torch, model_cls, tokenizer_cls = import_runtime()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    tokenizer = make_tokenizer(tokenizer_cls, args.student_model, args.local_files_only)

    prompts = load_prompts(args.prompts_file, args.limit)
    prompts = select_prompt_shard(prompts, args.num_shards, args.shard_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    student = load_model(
        model_cls,
        args.student_model,
        dtype,
        device,
        args.device_map,
        args.local_files_only,
    )
    teacher = load_model(
        model_cls,
        args.teacher_model,
        dtype,
        device,
        args.device_map,
        args.local_files_only,
    )
    teacher.requires_grad_(False)

    rows: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        if args.progress_every > 0 and (
            prompt_idx == 0
            or prompt_idx + 1 == len(prompts)
            or (prompt_idx + 1) % args.progress_every == 0
        ):
            print(f"[prompt] {prompt_idx + 1}/{len(prompts)}", flush=True)

        sample = generate_rollout(
            torch,
            tokenizer,
            student,
            prompt,
            args.max_prompt_tokens,
            args.max_new_tokens,
        )
        student_logits, student_hidden = forward_tensors(
            torch, student, sample["input_ids"], sample["attention_mask"], args.layer_index
        )
        teacher_logits, teacher_hidden = forward_tensors(
            torch, teacher, sample["input_ids"], sample["attention_mask"], args.layer_index
        )
        scores = compute_lr_tip(
            hidden_teacher=teacher_hidden,
            hidden_student=student_hidden,
            logits_teacher=teacher_logits,
            logits_student=student_logits,
            attention_mask=sample["attention_mask"],
            window=args.window,
            output_chunk_size=args.output_chunk_size,
            shift_output_to_labels=True,
            output_kl_direction=args.output_kl_direction,
            relation_gamma=args.relation_gamma,
            relation_lambda=args.relation_lambda,
            lr_tip_full_mode=args.lr_tip_full_mode,
        )
        positions = sample_token_positions(
            sample["target_mask"],
            args.tokens_per_prompt,
            seed=args.seed + prompt_idx,
        )
        for pos_idx, label_position in enumerate(positions):
            grad_improvement = gradient_norm_for_label_position(
                torch,
                student,
                sample["input_ids"],
                sample["attention_mask"],
                teacher_logits,
                label_position,
            )
            token_id = int(sample["input_ids"][0, label_position].item())
            rows.append(
                {
                    "prompt_index": prompt_idx,
                    "shard_index": args.shard_index,
                    "token_index": pos_idx,
                    "label_position": label_position,
                    "token_id": token_id,
                    "token": tokenizer.decode([token_id], skip_special_tokens=False),
                    "grad_improvement": grad_improvement,
                    "output_disagreement": float(
                        scores.output_disagreement[0, label_position].item()
                    ),
                    "relation_disagreement": float(
                        scores.relation_disagreement[0, label_position].item()
                    ),
                    "student_entropy": float(
                        scores.student_entropy[0, label_position].item()
                    ),
                    "entropy_norm": float(scores.entropy_norm[0, label_position].item()),
                    "output_norm": float(scores.output_norm[0, label_position].item()),
                    "relation_norm": float(
                        scores.relation_norm[0, label_position].item()
                    ),
                    "tip_soft_or": float(scores.tip_soft_or[0, label_position].item()),
                    "lr_tip_soft_or": float(
                        scores.lr_tip_soft_or[0, label_position].item()
                    ),
                    "lr_tip_add": float(scores.lr_tip_add[0, label_position].item()),
                    "lr_tip_gated": float(
                        scores.lr_tip_gated[0, label_position].item()
                    ),
                    "lr_tip_full": float(scores.lr_tip_full[0, label_position].item()),
                }
            )

    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {"budgets": budgets},
        "aggregate": aggregate_rows(rows, budgets, args.seed),
        "per_token": rows,
    }
    output_path = args.output_dir / "report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
