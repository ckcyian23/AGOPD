"""Run the LR-TIP offline MVP validation.

This script deliberately avoids training. It scores tokens from student
rollouts and reports whether the LR-TIP relation signal is complementary to
TIP's output-disagreement signal.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agopd.lr_tip import compute_lr_tip
from agopd.offline_eval import evaluate_budgets, pearson_corr


DEFAULT_PROMPTS = [
    "Solve 3x + 5 = 20. Show the reasoning.",
    "A train travels 120 km in 2 hours. What is its average speed?",
    "If a rectangle has length 8 and width 5, what is its area?",
    "Explain why the sum of two odd numbers is even.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default="models/Qwen/Qwen3-1.7B")
    parser.add_argument("--student-model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lr_tip_mvp"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--layer-index", type=int, default=-2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--budgets", default="0.1,0.2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split prompts into this many interleaved shards for multi-GPU runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Run only this zero-based prompt shard.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--execution-mode",
        default="student-cache",
        choices=["student-cache", "joint"],
        help=(
            "student-cache loads student first, caches CPU tensors, frees it, then "
            "loads teacher. joint keeps both models loaded and is faster on large GPUs."
        ),
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
        help="Only load models/tokenizers from local paths by default.",
    )
    parser.add_argument(
        "--output-chunk-size",
        type=int,
        default=8,
        help="Sequence positions per KL chunk; lower uses less peak memory.",
    )
    return parser.parse_args()


def load_prompts(path: Path | None, limit: int) -> list[str]:
    if path is None:
        prompts = DEFAULT_PROMPTS
    elif path.suffix.lower() == ".jsonl":
        prompts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompts.append(item.get("prompt") or item.get("question") or item["text"])
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        prompts = [
            item.get("prompt") or item.get("question") or item["text"]
            for item in data
        ]
    else:
        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return prompts[:limit] if limit > 0 else prompts


def select_prompt_shard(prompts: list[str], num_shards: int, shard_index: int) -> list[str]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}). Got {shard_index}."
        )
    return prompts[shard_index::num_shards]


def parse_budgets(raw: str) -> list[float]:
    budgets = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not budgets:
        raise ValueError("At least one budget fraction is required.")
    for budget in budgets:
        if not 0.0 < budget <= 1.0:
            raise ValueError(f"Invalid budget fraction: {budget}")
    return budgets


def import_runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Model evaluation requires torch and transformers. Run `uv sync` "
            "on the experiment machine before launching this script."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def resolve_dtype(torch: Any, requested: str, device: str):
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def model_device(model: Any):
    return next(model.parameters()).device


def load_model(
    model_cls: Any,
    model_path: str,
    torch_dtype: Any,
    device: str,
    device_map: str,
    local_files_only: bool,
):
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map != "none":
        kwargs["device_map"] = device_map
    model = model_cls.from_pretrained(model_path, **kwargs)
    if device_map == "none":
        model = model.to(device)
    model.eval()
    return model


def make_tokenizer(tokenizer_cls: Any, model_path: str, local_files_only: bool):
    tokenizer = tokenizer_cls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def generate_rollout(
    torch: Any,
    tokenizer: Any,
    student: Any,
    prompt: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    prompt_len = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(model_device(student)) for key, value in encoded.items()}
    with torch.no_grad():
        generated = student.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_ids = generated.detach().cpu()
    response_ids = full_ids[:, prompt_len:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    attention_mask = torch.ones_like(full_ids, dtype=torch.long)
    target_mask = attention_mask.bool()
    target_mask[:, :prompt_len] = False
    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "response": response_text,
    }


def forward_tensors(torch: Any, model: Any, input_ids: Any, attention_mask: Any, layer_idx: int):
    input_ids = input_ids.to(model_device(model))
    attention_mask = attention_mask.to(model_device(model))
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = output.hidden_states[layer_idx].detach().cpu()
    logits = output.logits.detach().cpu()
    return logits, hidden


def score_one(
    sample: dict[str, Any],
    teacher_logits: Any,
    teacher_hidden: Any,
    student_logits: Any,
    student_hidden: Any,
    args: argparse.Namespace,
    budgets: list[float],
):
    scores = compute_lr_tip(
        hidden_teacher=teacher_hidden,
        hidden_student=student_hidden,
        logits_teacher=teacher_logits,
        logits_student=student_logits,
        attention_mask=sample["attention_mask"],
        window=args.window,
        alpha=args.alpha,
        beta=args.beta,
        output_chunk_size=args.output_chunk_size,
        shift_output_to_labels=True,
    )
    token_mask = sample["target_mask"]
    corr = pearson_corr(
        scores.output_disagreement, scores.relation_disagreement, token_mask
    )
    budget_metrics = evaluate_budgets(
        scores.output_disagreement,
        scores.relation_disagreement,
        scores.importance,
        token_mask,
        budgets,
        random_seed=args.seed,
    )
    return {
        "prompt": sample["prompt"],
        "response": sample["response"],
        "num_target_tokens": int(token_mask.sum().item()),
        "do_dr_corr": corr,
        "budgets": [asdict(item) for item in budget_metrics],
    }


def aggregate_results(per_prompt: list[dict[str, Any]], budgets: list[float]):
    aggregate: dict[str, Any] = {
        "num_prompts": len(per_prompt),
        "mean_do_dr_corr": None,
        "budgets": {},
    }
    corrs = [item["do_dr_corr"] for item in per_prompt if item["do_dr_corr"] == item["do_dr_corr"]]
    aggregate["mean_do_dr_corr"] = sum(corrs) / len(corrs) if corrs else None

    for budget in budgets:
        rows = [
            metric
            for item in per_prompt
            for metric in item["budgets"]
            if metric["budget_fraction"] == budget
        ]
        if not rows:
            continue
        keys = [key for key in rows[0] if key != "budget_fraction"]
        aggregate["budgets"][str(budget)] = {
            key: sum(row[key] for row in rows) / len(rows) for key in keys
        }
    return aggregate


def run_student_cache(args: argparse.Namespace, prompts: list[str], budgets: list[float]):
    torch, model_cls, tokenizer_cls = import_runtime()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    tokenizer = make_tokenizer(tokenizer_cls, args.student_model, args.local_files_only)

    student = load_model(
        model_cls,
        args.student_model,
        dtype,
        device,
        args.device_map,
        args.local_files_only,
    )
    cached = []
    for prompt in prompts:
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
        sample["student_logits"] = student_logits
        sample["student_hidden"] = student_hidden
        cached.append(sample)

    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = load_model(
        model_cls,
        args.teacher_model,
        dtype,
        device,
        args.device_map,
        args.local_files_only,
    )
    results = []
    for sample in cached:
        teacher_logits, teacher_hidden = forward_tensors(
            torch, teacher, sample["input_ids"], sample["attention_mask"], args.layer_index
        )
        results.append(
            score_one(
                sample,
                teacher_logits,
                teacher_hidden,
                sample["student_logits"],
                sample["student_hidden"],
                args,
                budgets,
            )
        )
    return results


def run_joint(args: argparse.Namespace, prompts: list[str], budgets: list[float]):
    torch, model_cls, tokenizer_cls = import_runtime()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    tokenizer = make_tokenizer(tokenizer_cls, args.student_model, args.local_files_only)
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

    results = []
    for prompt in prompts:
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
        results.append(
            score_one(
                sample,
                teacher_logits,
                teacher_hidden,
                student_logits,
                student_hidden,
                args,
                budgets,
            )
        )
    return results


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    prompts = load_prompts(args.prompts_file, args.limit)
    prompts = select_prompt_shard(prompts, args.num_shards, args.shard_index)
    default_output_dir = Path("outputs/lr_tip_mvp")
    if args.num_shards > 1 and args.output_dir == default_output_dir:
        args.output_dir = args.output_dir / f"shard_{args.shard_index}_of_{args.num_shards}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.execution_mode == "student-cache":
        per_prompt = run_student_cache(args, prompts, budgets)
    else:
        per_prompt = run_joint(args, prompts, budgets)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "config": config | {"budgets": budgets},
        "aggregate": aggregate_results(per_prompt, budgets),
        "per_prompt": per_prompt,
    }
    output_path = args.output_dir / "report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
