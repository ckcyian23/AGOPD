"""Merge sharded LR-TIP offline-eval reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agopd.experiments.offline_lr_tip_eval import aggregate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Shard report.json paths, or directories containing report.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lr_tip_merged/report.json"),
        help="Merged report output path.",
    )
    return parser.parse_args()


def resolve_report(path: Path) -> Path:
    return path / "report.json" if path.is_dir() else path


def load_report(path: Path) -> dict[str, Any]:
    report_path = resolve_report(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def collect_budgets(reports: list[dict[str, Any]]) -> list[float]:
    seen: set[float] = set()
    for report in reports:
        for budget in report.get("config", {}).get("budgets", []):
            seen.add(float(budget))
        for budget in report.get("aggregate", {}).get("budgets", {}):
            seen.add(float(budget))
    if not seen:
        raise ValueError("No budget fractions found in input reports.")
    return sorted(seen)


def main() -> None:
    args = parse_args()
    reports = [load_report(path) for path in args.reports]
    budgets = collect_budgets(reports)
    per_prompt = [
        item for report in reports for item in report.get("per_prompt", [])
    ]
    if not per_prompt:
        raise ValueError("Input reports contain no per_prompt rows.")

    sources = [str(resolve_report(path)) for path in args.reports]
    config = {
        "merged_from": sources,
        "num_shards": len(reports),
        "budgets": budgets,
    }
    merged = {
        "config": config,
        "aggregate": aggregate_results(per_prompt, budgets),
        "per_prompt": per_prompt,
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
