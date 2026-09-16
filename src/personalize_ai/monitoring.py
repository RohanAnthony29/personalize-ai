from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "recommendation_requests_total", "Recommendation requests", ["endpoint", "status"]
)
LATENCY = Histogram(
    "recommendation_latency_seconds",
    "End-to-end recommendation latency",
    ["endpoint", "cache"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
CACHE = Counter("recommendation_cache_total", "Cache outcomes", ["outcome"])
RECOMMENDATION_COUNT = Histogram(
    "recommendation_result_count", "Number of recommendations returned", buckets=(0, 1, 5, 10, 20, 50, 100)
)
CHALLENGER_LATENCY = Histogram(
    "challenger_scoring_seconds", "PyTorch challenger scoring latency"
)
CHALLENGER_LOADED = Counter(
    "challenger_load_total", "Challenger model load outcomes", ["outcome"]
)
