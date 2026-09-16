#!/usr/bin/env python3
"""Build point-in-time offline features with Spark SQL.

Each run is immutable under ``<output>/<feature_version>``. Incremental runs only
consume events newer than the last successful watermark and record their parent
version, so consumers can replay the version chain without rewriting history.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from contracts import IncrementalState, render_sql, validate_manifest

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["data/processed/events_train.csv"])
    parser.add_argument("--input-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--version", help="Immutable feature version; defaults to UTC timestamp")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--state-file", default="data/features/_state.json")
    parser.add_argument("--sql-dir", default="sql")
    parser.add_argument("--partitions", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pyspark.sql import SparkSession, functions as F, types as T
    except ImportError as exc:
        raise SystemExit(
            "PySpark is required for this production job. Install the 'spark' extra "
            "or run it in the Docker Spark profile."
        ) from exc

    version = args.version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root, state_path = Path(args.output), Path(args.state_file)
    version_path = output_root / version
    if version_path.exists():
        raise SystemExit(f"Refusing to overwrite immutable version: {version_path}")

    state = IncrementalState.load(state_path)
    prior_watermark = state.watermark_ms if args.incremental else -1
    schema = T.StructType([
        T.StructField("timestamp", T.LongType(), False),
        T.StructField("user_id", T.LongType(), False),
        T.StructField("item_id", T.LongType(), False),
        T.StructField("event_type", T.StringType(), False),
        T.StructField("event_weight", T.DoubleType(), False),
        T.StructField("transaction_id", T.StringType(), True),
    ])
    spark = SparkSession.builder.appName("personalize-offline-features").getOrCreate()
    if args.input_format == "parquet":
        events = spark.read.parquet(*args.input).select(*schema.fieldNames())
    else:
        events = spark.read.option("header", True).schema(schema).csv(args.input)
    events = events.filter(F.col("timestamp") > prior_watermark)
    invalid = events.filter(
        F.col("user_id").isNull() | F.col("item_id").isNull() |
        ~F.col("event_type").isin("view", "addtocart", "transaction") |
        (F.col("event_weight") <= 0)
    ).limit(1).count()
    if invalid:
        raise ValueError("Input failed event schema/domain validation")
    if events.limit(1).count() == 0:
        print("No events newer than the stored watermark; no version created.")
        spark.stop()
        return

    as_of_ms = int(events.agg(F.max("timestamp")).first()[0])
    events.createOrReplaceTempView("events")
    counts = {}
    for entity in ("user", "item", "interaction"):
        frame = spark.sql(render_sql(
            Path(args.sql_dir) / f"{entity}_features.sql", "events", as_of_ms
        ))
        counts[entity] = frame.count()
        frame.repartition(args.partitions).write.mode("errorifexists").parquet(
            str(version_path / f"{entity}_features")
        )

    input_count = events.count()
    manifest = {
        "feature_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "parquet",
        "input_paths": args.input,
        "incremental": args.incremental,
        "parent_version": state.latest_version if args.incremental else None,
        "prior_watermark_ms": prior_watermark,
        "watermark_ms": as_of_ms,
        "input_event_count": input_count,
        "row_counts": counts,
    }
    version_path.mkdir(parents=True, exist_ok=True)
    failures = validate_manifest(manifest)
    if failures:
        raise ValueError("; ".join(failures))
    (version_path / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    IncrementalState(as_of_ms, version).dump(state_path)
    print(json.dumps(manifest, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
