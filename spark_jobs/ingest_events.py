#!/usr/bin/env python3
"""Incrementally ingest Retailrocket CSV into a versioned bronze Parquet layer."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from contracts import EVENT_WEIGHTS, IncrementalState


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/events.csv")
    parser.add_argument("--output", default="data/bronze/events")
    parser.add_argument("--state-file", default="data/bronze/_ingestion_state.json")
    parser.add_argument("--version")
    parser.add_argument("--incremental", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    from pyspark.sql import SparkSession, functions as F, types as T

    version = args.version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(args.output) / f"version={version}"
    if target.exists():
        raise SystemExit(f"Refusing to overwrite immutable ingestion version: {target}")
    state_path = Path(args.state_file)
    state = IncrementalState.load(state_path)
    watermark = state.watermark_ms if args.incremental else -1
    schema = T.StructType([
        T.StructField("timestamp", T.LongType(), False),
        T.StructField("visitorid", T.LongType(), False),
        T.StructField("event", T.StringType(), False),
        T.StructField("itemid", T.LongType(), False),
        T.StructField("transactionid", T.StringType(), True),
    ])
    spark = SparkSession.builder.appName("personalize-event-ingestion").getOrCreate()
    raw = spark.read.option("header", True).schema(schema).csv(args.input)
    events = (
        raw.filter(F.col("timestamp") > watermark)
        .filter(F.col("event").isin(*EVENT_WEIGHTS))
        .select(
            "timestamp",
            F.col("visitorid").alias("user_id"),
            F.col("itemid").alias("item_id"),
            F.col("event").alias("event_type"),
            F.create_map(*sum(([F.lit(k), F.lit(v)] for k, v in EVENT_WEIGHTS.items()), [])).getItem(F.col("event")).alias("event_weight"),
            F.col("transactionid").alias("transaction_id"),
        )
        .dropDuplicates(["timestamp", "user_id", "item_id", "event_type", "transaction_id"])
        .withColumn("event_date", F.to_date(F.from_unixtime(F.col("timestamp") / 1000)))
    )
    if events.limit(1).count() == 0:
        print("No events beyond the current watermark.")
        spark.stop()
        return
    new_watermark = int(events.agg(F.max("timestamp")).first()[0])
    row_count = events.count()
    events.write.mode("errorifexists").partitionBy("event_date").parquet(str(target))
    manifest = {
        "ingestion_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.input,
        "incremental": args.incremental,
        "parent_version": state.latest_version if args.incremental else None,
        "prior_watermark_ms": watermark,
        "watermark_ms": new_watermark,
        "row_count": row_count,
    }
    (target / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    IncrementalState(new_watermark, version).dump(state_path)
    print(json.dumps(manifest, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
