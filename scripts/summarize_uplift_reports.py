"""Print compact summaries for distillation uplift report.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    return parser.parse_args()


def resolve_report(path: Path) -> Path:
    return path / "report.json" if path.is_dir() else path


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    header = [
        "report",
        "selector",
        "budget",
        "rel_weight",
        "train_eval",
        "kl_before",
        "kl_after",
        "kl_delta",
        "rel_delta",
        "regen_kl_after",
        "regen_rel_after",
        "selected",
    ]
    print("\t".join(header))
    for raw_path in args.reports:
        report_path = resolve_report(raw_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report["config"]
        metrics = report["metrics"]
        row = [
            str(report_path),
            config.get("selector"),
            fmt(config.get("budget"), 3),
            fmt(config.get("relation_loss_weight"), 1),
            f"{config.get('train_limit')}/{config.get('eval_limit')}",
            fmt(metrics.get("eval_kl_before")),
            fmt(metrics.get("eval_kl_after")),
            fmt(metrics.get("eval_kl_delta")),
            fmt(metrics.get("eval_relation_delta")),
            fmt(metrics.get("eval_regenerated_kl_after")),
            fmt(metrics.get("eval_regenerated_relation_after")),
            f"{metrics.get('selected_tokens')}/{metrics.get('target_tokens')}",
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
