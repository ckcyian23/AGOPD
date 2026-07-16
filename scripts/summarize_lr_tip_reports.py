"""Print compact summaries for LR-TIP report.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--budgets", default="0.05,0.1,0.2")
    return parser.parse_args()


def resolve_report(path: Path) -> Path:
    return path / "report.json" if path.is_dir() else path


def ratio(numerator: Any, denominator: Any) -> float | None:
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    budgets = [item.strip() for item in args.budgets.split(",") if item.strip()]

    header = [
        "report",
        "prompts",
        "do_dr_corr",
        "entropy_do_corr",
        "entropy_dr_corr",
    ]
    for budget in budgets:
        header.extend(
            [
                f"{budget}_tip_dr",
                f"{budget}_full_dr",
                f"{budget}_full/tip",
                f"{budget}_overlap",
            ]
        )
    print("\t".join(header))

    for raw_path in args.reports:
        report_path = resolve_report(raw_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        aggregate = report["aggregate"]
        row = [
            str(report_path),
            fmt(aggregate.get("num_prompts"), 0),
            fmt(aggregate.get("mean_do_dr_corr")),
            fmt(aggregate.get("mean_entropy_do_corr")),
            fmt(aggregate.get("mean_entropy_dr_corr")),
        ]
        for budget in budgets:
            metrics = aggregate["budgets"][budget]
            tip_dr = metrics["tip_dr_mean"]
            full_dr = metrics["lr_tip_full_dr_mean"]
            row.extend(
                [
                    fmt(tip_dr),
                    fmt(full_dr),
                    fmt(ratio(full_dr, tip_dr), 2),
                    fmt(metrics["tip_lr_tip_full_overlap"]),
                ]
            )
        print("\t".join(row))


if __name__ == "__main__":
    main()
