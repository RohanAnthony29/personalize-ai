# PersonalizeAI

A production-style recommendation platform built from the Retailrocket e-commerce
interaction dataset. The project will cover data validation, feature engineering,
candidate generation, neural ranking, offline evaluation, real-time inference,
caching, experiment tracking, and containerized deployment.

## Dataset

Download the Retailrocket dataset and place these files in `data/raw/`:

- `events.csv`
- `category_tree.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`

Raw data is intentionally excluded from Git.

## First data-quality check

Run a quick sample profile:

```bash
python3 scripts/profile_data.py --data-dir data/raw --max-rows 100000
```

Run the complete profile and save the result:

```bash
make profile
```

The command validates file headers, event values, identifiers, timestamps, and
transaction IDs while streaming rows to keep memory usage bounded.

## Chronological dataset split

Create non-overlapping train, validation, and test windows:

```bash
make split
```

The outputs use standardized column names and attach implicit-feedback weights:
`view=1`, `addtocart=3`, and `transaction=5`. The split is based on event time,
not CSV row order, so future interactions cannot leak into earlier windows.

## Popularity baseline

Train and evaluate the weighted global-popularity recommender:

```bash
make baseline
```

The validation backtest uses only training history. The final test backtest uses
training plus validation history. Relevant outcomes are unique, previously unseen
items receiving an `addtocart` or `transaction` event. The report contains
Precision@K, Recall@K, Hit Rate@K, and NDCG@K for K=5, 10, and 20.

## Item-to-item candidate generation

Build cosine-normalized co-occurrence neighbors and compare them with popularity
on the identical warm-user cohort:

```bash
make cooccurrence
```

The generator filters previously seen products, limits bot-like histories to the
50 most recent unique items, retains 100 neighbors per item, and uses popularity
as a fallback when item neighbors cannot fill the requested candidate list.
The command also exports `artifacts/item_neighbors.csv.gz`, a serving-ready lookup
table that can feed candidates into the upcoming neural ranking stage.

## Neural ranking

Generate point-in-time labeled candidate examples, train the PyTorch embedding
ranker, and evaluate candidate reordering on the untouched test window:

```bash
make ranker
```

The model combines candidate embeddings, pooled user-history embeddings, embedding
interactions, co-occurrence score, popularity, retrieval rank, source coverage,
and history length. A bounded residual architecture preserves the retrieval prior
when supervised positives are sparse. Artifacts and generated training examples
are excluded from Git.

### Rolling-window supervision

Generate multiple point-in-time examples from the training period:

```bash
make rolling-examples
```

The first 70% of training time builds a frozen candidate graph. New cart and
transaction events in the remaining 30% become positive prediction points using
only each user's preceding history. Every positive is paired with 20 hard
candidate negatives, and the final 20% of prediction time is reserved for early
stopping.
Chronological validation retains the complete 100-item candidate pool so the
early-stopping task matches the serving-time ranking problem.

The default `make ranker` command consumes these groups with Bayesian Personalized
Ranking loss. It evaluates the untrained retrieval prior at epoch zero, monitors
chronological validation NDCG@10, stops after three stale epochs, restores the
best checkpoint, and evaluates that checkpoint once on the test window.
A learned checkpoint must improve validation NDCG@10 by at least 5% relative to
the retrieval prior; otherwise the exported model safely reverts to epoch zero.
Training uses every naturally retrieved positive plus a deterministic 10% sample
of injected positives at one-quarter pair weight. Candidate features also include
category affinity, recency-weighted evidence, and maximum source-item similarity.

## Hybrid candidate generation

Evaluate diversified candidates and export serving lookups:

```bash
make hybrid
```

The hybrid combines long-term co-occurrence, directed transitions occurring inside
30-minute visitor sessions, recent-category popularity, and global popularity with
weighted reciprocal-rank fusion. Candidate Recall@50/100 and user coverage are
measured against the original co-occurrence candidate source.

Rolling examples use this same hybrid pool and record its normalized fusion score
plus four binary source-provenance features for downstream ranking.

## Ranker diagnosis

Run controlled feature-only BPR experiments:

```bash
make diagnose-ranker
```

The diagnostic compares linear and small-MLP corrections, then removes category,
recency, maximum-similarity, and source-provenance features one group at a time.
All variants share the same temporal validation cohort; only the winning variant
is evaluated once on the untouched test set.

## Multi-fold listwise experiment

```bash
make temporal-folds
make listwise-ranker
```

Four expanding temporal folds use disjoint outcome windows. The first three train
the model and the fourth controls early stopping. Only naturally retrieved targets
are retained, every group contains all 100 candidates, and softmax cross-entropy
optimizes the complete ranked list rather than sampled positive-negative pairs.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Roadmap

1. Validate and profile raw interactions.
2. Create chronological train, validation, and test splits.
3. Establish popularity and item-to-item baselines.
4. Train PyTorch retrieval and ranking models.
5. Serve recommendations with FastAPI and Redis.
6. Track experiments and package the stack with MLflow and Docker.
