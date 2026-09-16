#!/usr/bin/env python3
"""Measure cached and uncached API latency and emit machine-readable percentiles."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def request_ms(url: str) -> float:
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        response.read()
    return (time.perf_counter() - started) * 1000


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "requests": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output", default="data/reports/latency_benchmark.json")
    args = parser.parse_args()
    endpoint = f"{args.base_url}/v1/recommendations"
    request_ms(f"{endpoint}/{args.user_id}")
    cached = [request_ms(f"{endpoint}/{args.user_id}") for _ in range(args.requests)]
    uncached = [request_ms(f"{endpoint}/{args.user_id + i + 1}") for i in range(args.requests)]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "cached": summarize(cached),
        "uncached": summarize(uncached),
    }
    report["p95_cache_improvement_pct"] = (
        100 * (1 - report["cached"]["p95_ms"] / report["uncached"]["p95_ms"])
        if report["uncached"]["p95_ms"] else 0.0
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
