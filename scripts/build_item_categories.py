#!/usr/bin/env python3
"""Build a point-in-time item-to-category lookup from Retailrocket metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def maximum_event_timestamp(path: Path) -> int:
    maximum = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            maximum = max(maximum, int(row["timestamp"]))
    if not maximum:
        raise ValueError(f"No events found in {path}")
    return maximum


def extract_categories(paths: list[Path], as_of_timestamp: int) -> dict[int, tuple[int, int]]:
    categories: dict[int, tuple[int, int]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["property"] != "categoryid":
                    continue
                timestamp = int(row["timestamp"])
                if timestamp > as_of_timestamp:
                    continue
                try:
                    item_id = int(row["itemid"])
                    category_id = int(row["value"])
                except ValueError:
                    continue
                previous = categories.get(item_id)
                if previous is None or timestamp > previous[1]:
                    categories[item_id] = (category_id, timestamp)
    return categories


def write_categories(categories: dict[int, tuple[int, int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "category_id", "source_timestamp"])
        for item_id in sorted(categories):
            category_id, timestamp = categories[item_id]
            writer.writerow([item_id, category_id, timestamp])


def load_categories(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["item_id"]): int(row["category_id"])
            for row in csv.DictReader(handle)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--events", type=Path, default=Path("data/processed/events_train.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/item_categories.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/item_categories.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of = maximum_event_timestamp(args.events)
    sources = [
        args.raw_dir / "item_properties_part1.csv",
        args.raw_dir / "item_properties_part2.csv",
    ]
    categories = extract_categories(sources, as_of)
    write_categories(categories, args.output)
    report = {
        "as_of_timestamp": as_of,
        "items_with_category": len(categories),
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
