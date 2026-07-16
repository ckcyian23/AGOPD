"""Merge sharded gradient-improvement reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agopd.experiments.gradient_improvement_eval import aggregate_rows, parse_budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/lr_tip_grad_improvement/merged/report.json"))
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def resolve_report(path: Path) -> Path:
    return path / "report.json" if path.is_dir() else path


def load_report(path: Path) -> dict[str, Any]:
    report_path = resolve_report(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def infer_budgets(reports: list[dict[str, Any]], raw: str | None) -> list[float]:
    if raw is not None:
        return parse_budgets(raw)
    seen: set[float] = set()
    for report in reports:
        for budget in report.get("config", {}).get("budgets", []):
            seen.add(float(budget))
        for budget in report.get("aggregate", {}).get("budgets", {}):
            seen.add(float(budget))
    if not seen:
        raise ValueError("No budgets found. Pass --budgets.")
    return sorted(seen)


def main() -> None:
    args = parse_args()
    reports = [load_report(path) for path in args.reports]
    budgets = infer_budgets(reports, args.budgets)
    rows = []
    for report_idx, report in enumerate(reports):
        for row in report.get("per_token", []):
            rows.append(row | {"merge_source_index": report_idx})
    if not rows:
        raise ValueError("Input reports contain no per_token rows.")

    merged = {
        "config": {
            "merged_from": [str(resolve_report(path)) for path in args.reports],
            "num_shards": len(reports),
            "budgets": budgets,
        },
        "aggregate": aggregate_rows(rows, budgets, args.seed),
        "per_token": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(merged["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
