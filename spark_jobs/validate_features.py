#!/usr/bin/env python3
"""Fail a feature build when schema, uniqueness, domain, or freshness checks fail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import validate_manifest


KEYS = {"user": ["user_id"], "item": ["item_id"], "interaction": ["user_id", "item_id"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version_path")
    args = parser.parse_args()
    root = Path(args.version_path)
    manifest = json.loads((root / "_manifest.json").read_text())
    failures = validate_manifest(manifest)
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.appName("personalize-feature-quality").getOrCreate()
    for entity, keys in KEYS.items():
        frame = spark.read.parquet(str(root / f"{entity}_features"))
        actual = frame.count()
        expected = manifest["row_counts"][entity]
        if actual != expected:
            failures.append(f"{entity}: manifest={expected}, actual={actual}")
        if frame.dropDuplicates(keys).count() != actual:
            failures.append(f"{entity}: duplicate primary keys")
        if frame.filter(F.expr(" OR ".join(f"{key} IS NULL" for key in keys))).limit(1).count():
            failures.append(f"{entity}: null primary key")
        if "recency_days" in frame.columns and frame.filter(F.col("recency_days") < 0).limit(1).count():
            failures.append(f"{entity}: negative recency")
    spark.stop()
    report = {"passed": not failures, "failures": failures}
    (root / "_quality.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
