#!/usr/bin/env python3
"""Publish a validated offline feature version to Redis for online inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version_path")
    parser.add_argument("--redis-url", default="redis://redis:6379/0")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    root = Path(args.version_path)
    manifest = json.loads((root / "_manifest.json").read_text())
    quality = json.loads((root / "_quality.json").read_text())
    if not quality.get("passed"):
        raise SystemExit("Refusing to publish a feature version that failed quality checks")
    if manifest.get("snapshot_type") != "full":
        raise SystemExit(
            "Refusing to activate an incremental delta. Compact/rebuild a full snapshot first."
        )
    version = manifest["feature_version"]

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("personalize-online-publisher").getOrCreate()

    import redis

    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    published_counts: dict[str, int] = {}
    batch_size = args.batch_size
    for entity, key_columns in (
        ("user", ["user_id"]),
        ("item", ["item_id"]),
        ("interaction", ["user_id", "item_id"]),
    ):
        frame = spark.read.parquet(str(root / f"{entity}_features"))
        pipe, pending, published = client.pipeline(transaction=False), 0, 0
        for row in frame.toLocalIterator(prefetchPartitions=True):
            value = row.asDict(recursive=True)
            identifier = ":".join(str(value[key]) for key in key_columns)
            pipe.set(f"features:{version}:{entity}:{identifier}", json.dumps(value))
            if entity == "item":
                pipe.zadd(
                    f"features:{version}:item_popularity",
                    {str(value["item_id"]): float(value["weighted_popularity"])},
                )
            pending += 1
            published += 1
            if pending >= batch_size:
                pipe.execute()
                pipe, pending = client.pipeline(transaction=False), 0
        if pending:
            pipe.execute()
        expected = int(manifest["row_counts"][entity])
        if published != expected:
            raise RuntimeError(
                f"Published {published} {entity} rows, expected {expected}; refusing activation"
            )
        published_counts[entity] = published

    client.set("features:active_version", version)
    client.set(f"features:{version}:manifest", json.dumps(manifest))
    client.set(f"features:{version}:published_counts", json.dumps(published_counts))
    print(json.dumps({"published_version": version, "row_counts": published_counts}, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
