"""Small-scale distillation uplift check for LR-TIP token selection."""

from __future__ import annotations

import argparse
import json
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
)
from agopd.lr_tip import compute_lr_tip
from agopd.offline_eval import random_mask, topk_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default="models/Qwen/Qwen3-1.7B")
    parser.add_argument("--student-model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts-file", type=Path, default=Path("data/real_prompts_512.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lr_tip_distill_uplift"))
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--layer-index", type=int, default=-2)
    parser.add_argument(
        "--selector",
        default="tip",
        choices=[
            "random",
            "entropy",
            "divergence",
            "relation",
            "tip",
            "lr_tip_soft_or",
            "lr_tip_add",
            "lr_tip_gated",
            "lr_tip_product",
            "lr_tip_product_add",
        ],
    )
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--relation-gamma", type=float, default=1.0)
    parser.add_argument("--relation-lambda", type=float, default=1.0)
    parser.add_argument(
        "--relation-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary relation-profile loss weight used during training.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--device-map", default="none")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-chunk-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--save-student-dir",
        type=Path,
        default=None,
        help="Optional directory for saving the trained student checkpoint.",
    )
    parser.add_argument(
        "--eval-regenerate-after",
        action="store_true",
        help="Regenerate eval rollouts with the trained student and report KL/relation metrics.",
    )
    parser.add_argument(
        "--save-rollouts",
        action="store_true",
        help="Write eval rollout prompts/responses before and after training as JSONL.",
    )
    return parser.parse_args()


def select_scores(scores: Any, selector: str, relation_lambda: float) -> Any:
    if selector == "entropy":
        return scores.student_entropy
    if selector == "divergence":
        return scores.output_disagreement
    if selector == "relation":
        return scores.relation_disagreement
    if selector == "tip":
        return scores.tip_soft_or
    if selector == "lr_tip_soft_or":
        return scores.lr_tip_soft_or
    if selector == "lr_tip_add":
        return (scores.tip_soft_or + relation_lambda * scores.relation_norm).clamp(0.0, 1.0)
    if selector == "lr_tip_gated":
        return (
            scores.tip_soft_or
            + relation_lambda * scores.relation_norm * (1.0 - scores.tip_soft_or)
        ).clamp(0.0, 1.0)
    if selector == "lr_tip_product":
        return scores.lr_tip_product
    if selector == "lr_tip_product_add":
        return scores.lr_tip_product_add
    raise ValueError(f"Unsupported score selector: {selector}")


def choose_token_mask(
    torch: Any,
    scores: Any,
    token_mask: Any,
    args: argparse.Namespace,
    sample_index: int,
) -> Any:
    if args.selector == "random":
        return random_mask(token_mask, args.budget, seed=args.seed + sample_index)
    score_values = select_scores(scores, args.selector, args.relation_lambda)
    return topk_mask(score_values, token_mask, args.budget)


def prepare_sample(
    torch: Any,
    tokenizer: Any,
    student: Any,
    teacher: Any,
    prompt: str,
    args: argparse.Namespace,
    sample_index: int,
    with_selection: bool,
) -> dict[str, Any]:
    sample = generate_rollout(
        torch,
        tokenizer,
        student,
        prompt,
        args.max_prompt_tokens,
        args.max_new_tokens,
    )
    if not with_selection:
        return sample

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
        relation_gamma=args.relation_gamma,
        relation_lambda=args.relation_lambda,
    )
    selected = choose_token_mask(
        torch, scores, sample["target_mask"], args, sample_index
    )
    sample["selected_mask"] = selected.cpu()
    sample["selected_tokens"] = int(selected.sum().item())
    sample["target_tokens"] = int(sample["target_mask"].sum().item())
    return sample


def kl_loss_on_mask(
    torch: Any,
    student: Any,
    teacher: Any,
    sample: dict[str, Any],
    label_mask: Any,
    train: bool,
) -> Any:
    import torch.nn.functional as F

    device = model_device(student)
    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)
    label_mask = label_mask.to(device=device, dtype=torch.bool)
    pred_mask = label_mask[:, 1:]
    if int(pred_mask.sum().item()) == 0:
        return None

    with torch.no_grad():
        teacher_output = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        teacher_p = F.softmax(teacher_output.logits[:, :-1, :].float(), dim=-1)

    if train:
        student_output = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    else:
        with torch.no_grad():
            student_output = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    student_logp = F.log_softmax(student_output.logits[:, :-1, :].float(), dim=-1)
    per_token = F.kl_div(student_logp, teacher_p, reduction="none").sum(dim=-1)
    return per_token[pred_mask].mean()


def relation_profile_loss_on_mask(
    torch: Any,
    student: Any,
    teacher: Any,
    sample: dict[str, Any],
    label_mask: Any,
    args: argparse.Namespace,
    train: bool,
) -> Any:
    import torch.nn.functional as F

    device = model_device(student)
    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)
    label_mask = label_mask.to(device=device, dtype=torch.bool)
    selected_positions = [
        int(idx)
        for idx in torch.where(label_mask[0])[0].detach().cpu().tolist()
        if int(idx) > 0
    ]
    if not selected_positions:
        return None

    with torch.no_grad():
        teacher_output = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        teacher_hidden = teacher_output.hidden_states[args.layer_index].float()
        teacher_hidden = F.layer_norm(teacher_hidden, teacher_hidden.shape[-1:])
        teacher_hidden = F.normalize(teacher_hidden, p=2, dim=-1)

    if train:
        student_output = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    else:
        with torch.no_grad():
            student_output = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    student_hidden = student_output.hidden_states[args.layer_index].float()
    student_hidden = F.layer_norm(student_hidden, student_hidden.shape[-1:])
    student_hidden = F.normalize(student_hidden, p=2, dim=-1)

    losses = []
    for idx in selected_positions:
        start = max(0, idx - args.window)
        if start == idx:
            continue
        teacher_rel = (
            teacher_hidden[:, start:idx, :] * teacher_hidden[:, idx : idx + 1, :]
        ).sum(dim=-1)
        student_rel = (
            student_hidden[:, start:idx, :] * student_hidden[:, idx : idx + 1, :]
        ).sum(dim=-1)
        losses.append(F.huber_loss(student_rel, teacher_rel, reduction="mean"))
    if not losses:
        return None
    return torch.stack(losses).mean()


def evaluate_kl(torch: Any, student: Any, teacher: Any, samples: list[dict[str, Any]]) -> float:
    losses = []
    for sample in samples:
        loss = kl_loss_on_mask(
            torch,
            student,
            teacher,
            sample,
            sample["target_mask"],
            train=False,
        )
        if loss is not None:
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else float("nan")


def evaluate_relation(
    torch: Any,
    student: Any,
    teacher: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> float:
    losses = []
    for sample in samples:
        loss = relation_profile_loss_on_mask(
            torch,
            student,
            teacher,
            sample,
            sample["target_mask"],
            args,
            train=False,
        )
        if loss is not None:
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else float("nan")


def train_selected(
    torch: Any,
    student: Any,
    teacher: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[float]:
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr)
    losses = []
    for epoch in range(args.epochs):
        for idx, sample in enumerate(samples):
            optimizer.zero_grad(set_to_none=True)
            loss = kl_loss_on_mask(
                torch,
                student,
                teacher,
                sample,
                sample["selected_mask"],
                train=True,
            )
            if loss is None:
                continue
            relation_loss = None
            if args.relation_loss_weight > 0.0:
                relation_loss = relation_profile_loss_on_mask(
                    torch,
                    student,
                    teacher,
                    sample,
                    sample["selected_mask"],
                    args,
                    train=True,
                )
                if relation_loss is not None:
                    loss = loss + args.relation_loss_weight * relation_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if args.progress_every > 0 and (
                idx == 0 or idx + 1 == len(samples) or (idx + 1) % args.progress_every == 0
            ):
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} sample={idx + 1}/{len(samples)} loss={losses[-1]:.6f}",
                    flush=True,
                )
    return losses


def prepare_eval_samples(
    torch: Any,
    tokenizer: Any,
    student: Any,
    prompts: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    samples = []
    for idx, prompt in enumerate(prompts):
        if args.progress_every > 0:
            print(f"[prepare eval] {idx + 1}/{len(prompts)}", flush=True)
        samples.append(
            prepare_sample(torch, tokenizer, student, None, prompt, args, idx, False)
        )
    return samples


def write_rollouts(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples):
            item = {
                "index": idx,
                "prompt": sample["prompt"],
                "response": sample["response"],
                "prompt_len": sample["prompt_len"],
                "num_target_tokens": int(sample["target_mask"].sum().item()),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.budget <= 1.0:
        raise ValueError("--budget must be in (0, 1].")
    random.seed(args.seed)
    torch, model_cls, tokenizer_cls = import_runtime()
    torch.manual_seed(args.seed)
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    tokenizer = make_tokenizer(tokenizer_cls, args.student_model, args.local_files_only)

    all_prompts = load_prompts(args.prompts_file, 0)
    start = args.prompt_offset
    train_prompts = all_prompts[start : start + args.train_limit]
    eval_start = start + args.train_limit
    eval_prompts = all_prompts[eval_start : eval_start + args.eval_limit]
    if len(train_prompts) < args.train_limit or len(eval_prompts) < args.eval_limit:
        raise ValueError("Not enough prompts for requested train/eval split.")

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

    train_samples = []
    for idx, prompt in enumerate(train_prompts):
        if args.progress_every > 0:
            print(f"[prepare train] {idx + 1}/{len(train_prompts)}", flush=True)
        train_samples.append(
            prepare_sample(torch, tokenizer, student, teacher, prompt, args, idx, True)
        )
    eval_samples = []
    eval_samples = prepare_eval_samples(torch, tokenizer, student, eval_prompts, args)
    if args.save_rollouts or args.eval_regenerate_after:
        write_rollouts(args.output_dir / "eval_rollouts_before.jsonl", eval_samples)

    eval_before = evaluate_kl(torch, student, teacher, eval_samples)
    eval_relation_before = evaluate_relation(torch, student, teacher, eval_samples, args)
    train_losses = train_selected(torch, student, teacher, train_samples, args)
    eval_after = evaluate_kl(torch, student, teacher, eval_samples)
    eval_relation_after = evaluate_relation(torch, student, teacher, eval_samples, args)
    eval_regenerated_kl_after = None
    eval_regenerated_relation_after = None
    regenerated_rollouts_path = None
    if args.eval_regenerate_after:
        regenerated_eval_samples = prepare_eval_samples(
            torch, tokenizer, student, eval_prompts, args
        )
        regenerated_rollouts_path = args.output_dir / "eval_rollouts_after.jsonl"
        write_rollouts(regenerated_rollouts_path, regenerated_eval_samples)
        eval_regenerated_kl_after = evaluate_kl(
            torch, student, teacher, regenerated_eval_samples
        )
        eval_regenerated_relation_after = evaluate_relation(
            torch, student, teacher, regenerated_eval_samples, args
        )

    if args.save_student_dir is not None:
        args.save_student_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(args.save_student_dir)
        tokenizer.save_pretrained(args.save_student_dir)

    selected_tokens = sum(sample["selected_tokens"] for sample in train_samples)
    target_tokens = sum(sample["target_tokens"] for sample in train_samples)
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "metrics": {
            "eval_kl_before": eval_before,
            "eval_kl_after": eval_after,
            "eval_kl_delta": eval_after - eval_before,
            "relative_eval_kl_delta": (
                (eval_after - eval_before) / eval_before if eval_before else None
            ),
            "eval_relation_before": eval_relation_before,
            "eval_relation_after": eval_relation_after,
            "eval_relation_delta": eval_relation_after - eval_relation_before,
            "relative_eval_relation_delta": (
                (eval_relation_after - eval_relation_before) / eval_relation_before
                if eval_relation_before
                else None
            ),
            "eval_regenerated_kl_after": eval_regenerated_kl_after,
            "eval_regenerated_relation_after": eval_regenerated_relation_after,
            "eval_rollouts_before": (
                str(args.output_dir / "eval_rollouts_before.jsonl")
                if args.save_rollouts or args.eval_regenerate_after
                else None
            ),
            "eval_rollouts_after": (
                str(regenerated_rollouts_path) if regenerated_rollouts_path else None
            ),
            "train_loss_first": train_losses[0] if train_losses else None,
            "train_loss_last": train_losses[-1] if train_losses else None,
            "selected_tokens": selected_tokens,
            "target_tokens": target_tokens,
            "selected_fraction": selected_tokens / target_tokens if target_tokens else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "report.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
