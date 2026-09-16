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

    redis_url, batch_size = args.redis_url, args.batch_size
    for entity, key_columns in (
        ("user", ["user_id"]),
        ("item", ["item_id"]),
        ("interaction", ["user_id", "item_id"]),
    ):
        frame = spark.read.parquet(str(root / f"{entity}_features"))

        def publish_partition(rows, entity_name=entity, keys=key_columns):
            import redis
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            pipe, pending = client.pipeline(transaction=False), 0
            for row in rows:
                value = row.asDict(recursive=True)
                identifier = ":".join(str(value[key]) for key in keys)
                pipe.set(f"features:{version}:{entity_name}:{identifier}", json.dumps(value))
                pending += 1
                if pending >= batch_size:
                    pipe.execute()
                    pipe, pending = client.pipeline(transaction=False), 0
            if pending:
                pipe.execute()

        frame.foreachPartition(publish_partition)

        if entity == "item":
            def publish_popularity(rows):
                import redis
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                mapping = {
                    str(row["item_id"]): float(row["weighted_popularity"])
                    for row in rows
                }
                if mapping:
                    client.zadd(f"features:{version}:item_popularity", mapping)

            frame.select("item_id", "weighted_popularity").foreachPartition(publish_popularity)

    import redis
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    client.set("features:active_version", version)
    client.set(f"features:{version}:manifest", json.dumps(manifest))
    print(json.dumps({"published_version": version, "row_counts": manifest["row_counts"]}, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
