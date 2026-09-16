#!/usr/bin/env python3
"""Measure cached and uncached API latency and emit machine-readable percentiles."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


def summarize(values: list[float], wall_seconds: float) -> dict[str, float]:
    return {
        "requests": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "throughput_requests_per_second": len(values) / wall_seconds,
    }


def run_load(urls: list[str], concurrency: int) -> tuple[list[float], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        values = list(executor.map(request_ms, urls))
    return values, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output", default="data/reports/latency_benchmark.json")
    args = parser.parse_args()
    endpoint = f"{args.base_url}/v1/recommendations"
    request_ms(f"{endpoint}/{args.user_id}")
    cached, cached_wall = run_load(
        [f"{endpoint}/{args.user_id}"] * args.requests, args.concurrency
    )
    uncached, uncached_wall = run_load(
        [f"{endpoint}/{args.user_id + i + 1}" for i in range(args.requests)],
        args.concurrency,
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "cached": summarize(cached, cached_wall),
        "uncached": summarize(uncached, uncached_wall),
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
