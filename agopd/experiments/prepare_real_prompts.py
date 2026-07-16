"""Prepare benchmark prompts for LR-TIP offline feasibility experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/real_prompts_512.jsonl"))
    parser.add_argument("--gsm-count", type=int, default=256)
    parser.add_argument("--math-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Preparing benchmark prompts requires `datasets`. Run `uv sync` first."
        ) from exc
    return load_dataset


def load_first_available_split(
    dataset_name: str,
    config_name: str | None,
    split_candidates: list[str],
):
    load_dataset = require_datasets()
    errors: list[str] = []
    for split in split_candidates:
        try:
            if config_name is None:
                return load_dataset(dataset_name, split=split), split
            return load_dataset(dataset_name, config_name, split=split), split
        except Exception as exc:  # pragma: no cover - depends on remote datasets
            errors.append(f"{split}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Could not load {dataset_name}. Tried:\n{joined}")


def sample_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be >= 0.")
    if count == 0:
        return []
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows, but only {len(rows)} are available.")
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[idx] for idx in indices[:count]]


def gsm8k_records(count: int, seed: int) -> list[dict[str, Any]]:
    dataset, split = load_first_available_split(
        "openai/gsm8k", "main", ["test", "validation", "train"]
    )
    sampled = sample_rows(list(dataset), count, seed)
    return [
        {
            "dataset": "openai/gsm8k",
            "split": split,
            "prompt": (
                "Solve the following grade-school math problem. "
                "Show the reasoning step by step.\n\n"
                f"Problem: {row['question']}"
            ),
            "answer": row.get("answer"),
        }
        for row in sampled
    ]


def math500_records(count: int, seed: int) -> list[dict[str, Any]]:
    dataset, split = load_first_available_split(
        "HuggingFaceH4/MATH-500", None, ["test", "train", "validation"]
    )
    sampled = sample_rows(list(dataset), count, seed)
    return [
        {
            "dataset": "HuggingFaceH4/MATH-500",
            "split": split,
            "prompt": (
                "Solve the following competition math problem. "
                "Show the reasoning step by step.\n\n"
                f"Problem: {row['problem']}"
            ),
            "answer": row.get("answer"),
            "subject": row.get("subject"),
            "level": row.get("level"),
        }
        for row in sampled
    ]


def main() -> None:
    args = parse_args()
    records = []
    records.extend(gsm8k_records(args.gsm_count, args.seed))
    records.extend(math500_records(args.math_count, args.seed + 1))
    rng = random.Random(args.seed + 2)
    rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} prompts to {args.output}")


if __name__ == "__main__":
    main()
