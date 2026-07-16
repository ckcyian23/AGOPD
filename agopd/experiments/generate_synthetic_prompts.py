"""Generate deterministic synthetic math/reasoning prompts for offline eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_prompts(target_count: int) -> list[str]:
    prompts: list[str] = []

    for a in range(2, 42):
        for b in range(1, 18):
            c = a * b + (a + b)
            prompts.append(
                f"Solve {a}x + {a + b} = {c}. Show each algebraic step."
            )
            prompts.append(
                f"A store sells {a} items for {c} dollars. What is the cost of {b + 3} items at the same rate?"
            )
            prompts.append(
                f"If a rectangle has width {a} and area {a * (b + 5)}, what is its length? Explain."
            )
            prompts.append(
                f"The average of {a}, {b}, and x is {a + b}. Find x and justify the calculation."
            )
            if len(prompts) >= target_count:
                return prompts[:target_count]

    for n in range(3, 200):
        prompts.append(f"Prove that if n = {n} is odd or even, then n^2 has the same parity.")
        prompts.append(
            f"Find the prime factorization of {n * (n + 1)} and explain the steps."
        )
        prompts.append(
            f"A sequence starts at {n} and increases by {n % 7 + 2}. What is the 25th term?"
        )
        prompts.append(
            f"If {n}% of a number is {n * 3}, what is the original number?"
        )
        if len(prompts) >= target_count:
            return prompts[:target_count]

    return prompts[:target_count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/prompts_1024.jsonl"))
    parser.add_argument("--count", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be >= 1.")
    prompts = build_prompts(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
