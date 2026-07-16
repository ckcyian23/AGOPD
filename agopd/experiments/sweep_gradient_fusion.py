"""Sweep LR-TIP fusion hyperparameters on a gradient-improvement report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--budgets", default="0.05,0.1,0.2")
    parser.add_argument("--gammas", default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--lambdas", default="0.1,0.25,0.5,0.75,1.0,1.5,2.0")
    return parser.parse_args()


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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
    xs_c = [x - x_mean for x, _ in pairs]
    ys_c = [y - y_mean for _, y in pairs]
    denom = math.sqrt(sum(x * x for x in xs_c) * sum(y * y for y in ys_c))
    if denom <= 1e-12:
        return None
    return sum(x * y for x, y in zip(xs_c, ys_c)) / denom


def soft_or(a: float, b: float) -> float:
    return a + b - a * b


def top_mean(rows: list[dict[str, Any]], score_key: str, budget: float) -> float:
    k = max(1, int(round(len(rows) * budget)))
    selected = sorted(rows, key=lambda item: item[score_key], reverse=True)[:k]
    return sum(float(item["grad_improvement"]) for item in selected) / len(selected)


def score_rows(
    rows: list[dict[str, Any]],
    mode: str,
    value: float,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        tip = float(row["tip_soft_or"])
        relation = float(row["relation_norm"])
        if mode == "soft_or":
            score = soft_or(tip, min(1.0, value * relation))
        elif mode == "add":
            score = min(1.0, tip + value * relation)
        elif mode == "gated":
            score = min(1.0, tip + value * relation * (1.0 - tip))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        scored.append(row | {"sweep_score": score})
    return scored


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["per_token"]
    budgets = parse_floats(args.budgets)
    gammas = parse_floats(args.gammas)
    lambdas = parse_floats(args.lambdas)
    improvement = [float(row["grad_improvement"]) for row in rows]

    results: list[dict[str, Any]] = []
    for mode, values in [
        ("baseline", [0.0]),
        ("soft_or", gammas),
        ("add", lambdas),
        ("gated", lambdas),
    ]:
        for value in values:
            if mode == "baseline":
                scored = rows
                label = "tip"
                score_key = "tip_soft_or"
            else:
                scored = score_rows(rows, mode, value)
                label = f"{mode}_{value:g}"
                score_key = "sweep_score"
            item: dict[str, Any] = {
                "label": label,
                "mode": mode,
                "value": value,
                "corr": pearson([float(row[score_key]) for row in scored], improvement),
            }
            for budget in budgets:
                item[f"top_{budget:g}_grad_improvement_mean"] = top_mean(
                    scored, score_key, budget
                )
            results.append(item)

    output = {"source": str(args.report), "num_tokens": len(rows), "results": results}
    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")

    for budget in budgets:
        key = f"top_{budget:g}_grad_improvement_mean"
        print(f"\nBudget {budget:g}")
        for item in sorted(results, key=lambda row: row[key], reverse=True)[:10]:
            print(
                f"{item['label']}\tcorr={item['corr']:.4f}\t{key}={item[key]:.2f}"
            )


if __name__ == "__main__":
    main()
