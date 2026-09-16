#!/usr/bin/env python3
"""Measure cached and uncached API latency and emit machine-readable percentiles."""
from __future__ import annotations

import argparse
import json
import math
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


def request_ms(url: str, api_key: str = "") -> float:
    started = time.perf_counter()
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(request, timeout=10) as response:
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


def run_load(urls: list[str], concurrency: int, api_key: str) -> tuple[list[float], float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        values = list(executor.map(lambda url: request_ms(url, api_key), urls))
    return values, time.perf_counter() - started


def confidence_interval_95(values: list[float]) -> list[float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return [value, value]
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    mean = statistics.fmean(values)
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - margin, mean + margin]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", default="data/reports/latency_benchmark.json")
    args = parser.parse_args()
    endpoint = f"{args.base_url}/v1/recommendations"
    if args.repetitions < 2:
        raise SystemExit("--repetitions must be at least 2 for a confidence interval")
    request_ms(f"{endpoint}/{args.user_id}", args.api_key)
    cached, uncached = [], []
    cached_wall = uncached_wall = 0.0
    repetitions = []
    for repetition in range(args.repetitions):
        cached_values, cached_elapsed = run_load(
            [f"{endpoint}/{args.user_id}"] * args.requests,
            args.concurrency,
            args.api_key,
        )
        offset = repetition * args.requests
        uncached_values, uncached_elapsed = run_load(
            [f"{endpoint}/{args.user_id + offset + i + 1}" for i in range(args.requests)],
            args.concurrency,
            args.api_key,
        )
        cached.extend(cached_values)
        uncached.extend(uncached_values)
        cached_wall += cached_elapsed
        uncached_wall += uncached_elapsed
        repetitions.append(
            {
                "repetition": repetition + 1,
                "cached": summarize(cached_values, cached_elapsed),
                "uncached": summarize(uncached_values, uncached_elapsed),
            }
        )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "requests_per_repetition": args.requests,
        "repetition_count": args.repetitions,
        "cached": summarize(cached, cached_wall),
        "uncached": summarize(uncached, uncached_wall),
        "repetitions": repetitions,
    }
    report["cached"]["p95_ms_95ci_across_repetitions"] = confidence_interval_95(
        [value["cached"]["p95_ms"] for value in repetitions]
    )
    report["uncached"]["p95_ms_95ci_across_repetitions"] = confidence_interval_95(
        [value["uncached"]["p95_ms"] for value in repetitions]
    )
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
