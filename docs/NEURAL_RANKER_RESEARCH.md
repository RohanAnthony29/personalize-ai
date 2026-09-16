# Neural ranker promotion status

The PyTorch BPR ranker remains a shadow challenger. On the held-out evaluation,
it reached NDCG@10 of `0.235163`, compared with `0.234687` for the hybrid
retrieval ranking: a `0.203%` relative lift. That is positive, but below the 5%
offline promotion gate, so the hybrid reciprocal-rank-fusion system correctly
remains the champion.

Experiments already completed include rolling temporal examples, naturally
retrieved positives, feature ablations, a feature-only pairwise baseline, and
full-list listwise training across temporal folds. Injected positives were
removed from the final listwise experiment because they represent items the
candidate generator would not serve.

The next promotion-oriented experiments should focus on:

1. Learning a session encoder over the most recent event sequence rather than a
   pooled long-term history.
2. Expanding natural-positive coverage with more temporal folds and diversified
   candidate sources, while keeping each fold leakage-free.
3. Calibrating the neural correction against the strong hybrid retrieval score
   and evaluating improvements by user-history and item-popularity segments.
4. Requiring both the 5% NDCG@10 gate and no material regression in candidate
   coverage, p95 latency, or cold-start behavior before promotion.

Until that evidence exists, the model is loaded by FastAPI only as a challenger
and does not control production ordering.
