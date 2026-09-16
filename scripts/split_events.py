#!/usr/bin/env python3
"""Create leakage-safe chronological splits from Retailrocket events."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_HEADER = [
    "timestamp",
    "user_id",
    "item_id",
    "event_type",
    "event_weight",
    "transaction_id",
]
EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}


def scan_timestamp_range(path: Path) -> tuple[int, int]:
    minimum: int | None = None
    maximum: int | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["timestamp"])
            minimum = timestamp if minimum is None else min(minimum, timestamp)
            maximum = timestamp if maximum is None else max(maximum, timestamp)
    if minimum is None or maximum is None:
        raise ValueError(f"No event rows found in {path}")
    return minimum, maximum


def calculate_boundaries(
    minimum: int, maximum: int, train_ratio: float, validation_ratio: float
) -> tuple[int, int]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Ratios must be positive and sum to less than 1")
    span = maximum - minimum + 1
    train_end = minimum + int(span * train_ratio)
    validation_end = minimum + int(span * (train_ratio + validation_ratio))
    return train_end, validation_end


def split_name(timestamp: int, train_end: int, validation_end: int) -> str:
    if timestamp < train_end:
        return "train"
    if timestamp < validation_end:
        return "validation"
    return "test"


def iso_timestamp(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def split_events(
    source: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict[str, object]:
    minimum, maximum = scan_timestamp_range(source)
    train_end, validation_end = calculate_boundaries(
        minimum, maximum, train_ratio, validation_ratio
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: output_dir / f"events_{name}.csv" for name in ("train", "validation", "test")}
    handles = {name: path.open("w", newline="", encoding="utf-8") for name, path in paths.items()}
    writers = {name: csv.writer(handle) for name, handle in handles.items()}
    counts = Counter()
    event_counts = {name: Counter() for name in paths}
    bounds: dict[str, dict[str, int | None]] = {
        name: {"min_timestamp": None, "max_timestamp": None} for name in paths
    }

    try:
        for writer in writers.values():
            writer.writerow(OUTPUT_HEADER)

        with source.open(newline="", encoding="utf-8") as source_handle:
            for row in csv.DictReader(source_handle):
                timestamp = int(row["timestamp"])
                event = row["event"]
                if event not in EVENT_WEIGHTS:
                    raise ValueError(f"Unsupported event type: {event}")
                name = split_name(timestamp, train_end, validation_end)
                writers[name].writerow(
                    [
                        timestamp,
                        row["visitorid"],
                        row["itemid"],
                        event,
                        EVENT_WEIGHTS[event],
                        row["transactionid"],
                    ]
                )
                counts[name] += 1
                event_counts[name][event] += 1
                current = bounds[name]
                current["min_timestamp"] = (
                    timestamp
                    if current["min_timestamp"] is None
                    else min(current["min_timestamp"], timestamp)
                )
                current["max_timestamp"] = (
                    timestamp
                    if current["max_timestamp"] is None
                    else max(current["max_timestamp"], timestamp)
                )
    finally:
        for handle in handles.values():
            handle.close()

    split_report = {}
    total = sum(counts.values())
    for name in paths:
        split_report[name] = {
            "rows": counts[name],
            "fraction": counts[name] / total if total else 0,
            "event_counts": dict(sorted(event_counts[name].items())),
            "min_timestamp_utc": iso_timestamp(bounds[name]["min_timestamp"]),
            "max_timestamp_utc": iso_timestamp(bounds[name]["max_timestamp"]),
            "path": str(paths[name]),
        }

    return {
        "strategy": "global_time_window_70_15_15",
        "event_weights": EVENT_WEIGHTS,
        "source_rows": total,
        "source_min_timestamp_utc": iso_timestamp(minimum),
        "source_max_timestamp_utc": iso_timestamp(maximum),
        "train_cutoff_utc": iso_timestamp(train_end),
        "validation_cutoff_utc": iso_timestamp(validation_end),
        "splits": split_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/events.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/split_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = split_events(args.input, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
