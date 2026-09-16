# Production evidence

Measured in the free-tier GitHub Codespace on 2026-09-16 using the untouched
chronological test split and the full versioned feature snapshot.

## Offline relevance

| Comparison | Cohort | Baseline | Hybrid/challenger | Relative lift |
|---|---:|---:|---:|---:|
| Recall@50, item co-occurrence vs hybrid candidates | 580 warm users | 0.07102 | 0.10035 | **+41.30%** |
| Recall@100, item co-occurrence vs hybrid candidates | 580 warm users | 0.09931 | 0.12268 | **+23.54%** |
| NDCG@10, retrieval vs BPR neural ranker | 92 retrieved-positive groups | 0.23469 | 0.23516 | +0.20% |

The hybrid candidate generator demonstrates a defensible relevance gain. The
neural model did not meet its promotion gate, so it remains a shadow challenger
while reciprocal-rank-fusion retrieval remains the production champion.

## Online platform

- Spark SQL full snapshot: 1,218,519 users, 220,512 items, and 1,856,768
  interaction rows (3,295,799 total rows).
- Data-quality validation: passed with no failures.
- Redis: 3,295,803 keys, 1.13 GiB used, no eviction, active version
  `20260916-full`.
- FastAPI: hybrid champion with `bpr_ranker.pt` loaded for shadow scoring.
- Docker services: Redis, FastAPI, MLflow, and Prometheus.
- Automated tests: 38/38 passed.

## Latency benchmark

The benchmark used five repetitions of 50 requests per cohort at concurrency 10
against the populated Docker stack. Confidence intervals describe variation in
the per-repetition p95 measurement.

| Path | Mean | p50 | p95 | p95 95% CI | p99 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Redis-cached | 24.36 ms | 21.94 ms | 41.38 ms | 26.15–48.29 ms | 52.09 ms | 376.68 req/s |
| Uncached | 67.27 ms | 66.14 ms | 102.42 ms | 73.42–114.66 ms | 120.25 ms | 139.59 req/s |

Caching reduced measured p95 latency by **59.60%**. These numbers describe this
specific Codespace run; rerun `scripts/benchmark_api.py` for each deployment
environment rather than treating them as universal performance guarantees.
