#!/usr/bin/env python3
"""Stream and validate the Retailrocket source files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXPECTED_HEADERS = {
    "events.csv": ["timestamp", "visitorid", "event", "itemid", "transactionid"],
    "category_tree.csv": ["categoryid", "parentid"],
    "item_properties_part1.csv": ["timestamp", "itemid", "property", "value"],
    "item_properties_part2.csv": ["timestamp", "itemid", "property", "value"],
}
VALID_EVENTS = {"view", "addtocart", "transaction"}


def require_files(data_dir: Path) -> None:
    missing = [name for name in EXPECTED_HEADERS if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")


def validate_headers(data_dir: Path) -> dict[str, list[str]]:
    headers: dict[str, list[str]] = {}
    for name, expected in EXPECTED_HEADERS.items():
        with (data_dir / name).open(newline="", encoding="utf-8") as handle:
            actual = next(csv.reader(handle))
        if actual != expected:
            raise ValueError(f"{name}: expected header {expected}, found {actual}")
        headers[name] = actual
    return headers


def iter_limited(rows: Iterable[dict[str, str]], max_rows: int | None):
    for index, row in enumerate(rows, start=1):
        if max_rows is not None and index > max_rows:
            break
        yield index, row


def iso_timestamp(milliseconds: int | None) -> str | None:
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def profile_events(path: Path, max_rows: int | None) -> dict[str, object]:
    event_counts: Counter[str] = Counter()
    users: set[int] = set()
    items: set[int] = set()
    errors: Counter[str] = Counter()
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    transactions_with_id = 0
    rows_read = 0

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for rows_read, row in iter_limited(reader, max_rows):
            try:
                timestamp = int(row["timestamp"])
                visitor_id = int(row["visitorid"])
                item_id = int(row["itemid"])
            except (TypeError, ValueError):
                errors["invalid_numeric_field"] += 1
                continue

            event = row["event"]
            if event not in VALID_EVENTS:
                errors["invalid_event"] += 1
            if event == "transaction" and not row["transactionid"]:
                errors["transaction_missing_id"] += 1
            if event != "transaction" and row["transactionid"]:
                errors["non_transaction_has_id"] += 1
            if row["transactionid"]:
                transactions_with_id += 1

            event_counts[event] += 1
            users.add(visitor_id)
            items.add(item_id)
            min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
            max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)

    return {
        "rows_read": rows_read,
        "event_counts": dict(sorted(event_counts.items())),
        "unique_visitors": len(users),
        "unique_items": len(items),
        "transactions_with_id": transactions_with_id,
        "timestamp_min_utc": iso_timestamp(min_timestamp),
        "timestamp_max_utc": iso_timestamp(max_timestamp),
        "validation_errors": dict(sorted(errors.items())),
        "is_sample": max_rows is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")

    require_files(args.data_dir)
    result = {
        "dataset": "retailrocket",
        "headers": validate_headers(args.data_dir),
        "events": profile_events(args.data_dir / "events.csv", args.max_rows),
    }
    rendered = json.dumps(result, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote profile to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
