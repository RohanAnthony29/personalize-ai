#!/usr/bin/env python3
"""Train and evaluate a weighted global-popularity recommender."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


RELEVANT_EVENTS = {"addtocart", "transaction"}


def load_history(paths: list[Path]) -> tuple[Counter[int], dict[int, set[int]], int]:
    """Build weighted item popularity and per-user seen-item history."""
    popularity: Counter[int] = Counter()
    seen: dict[int, set[int]] = defaultdict(set)
    row_count = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                user_id = int(row["user_id"])
                item_id = int(row["item_id"])
                popularity[item_id] += int(row["event_weight"])
                seen[user_id].add(item_id)
                row_count += 1
    return popularity, dict(seen), row_count


def load_ground_truth(
    path: Path,
    history_seen: dict[int, set[int]],
) -> tuple[dict[int, set[int]], dict[str, int]]:
    """Load unique, relevant held-out items that were not already seen."""
    relevant: dict[int, set[int]] = defaultdict(set)
    candidate_users: set[int] = set()
    relevant_rows = 0
    repeated_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] not in RELEVANT_EVENTS:
                continue
            relevant_rows += 1
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            candidate_users.add(user_id)
            if item_id in history_seen.get(user_id, set()):
                repeated_rows += 1
                continue
            relevant[user_id].add(item_id)
    stats = {
        "relevant_event_rows": relevant_rows,
        "users_with_relevant_events": len(candidate_users),
        "repeated_seen_event_rows_excluded": repeated_rows,
        "eligible_users": len(relevant),
    }
    return dict(relevant), stats


def rank_items(popularity: Counter[int]) -> list[int]:
    """Use item ID as a deterministic tie-breaker."""
    return [item for item, _ in sorted(popularity.items(), key=lambda pair: (-pair[1], pair[0]))]


def recommend(ranked_items: list[int], seen_items: set[int], k: int) -> list[int]:
    recommendations: list[int] = []
    for item_id in ranked_items:
        if item_id not in seen_items:
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
    return recommendations


def user_metrics(recommendations: list[int], relevant: set[int], k: int) -> dict[str, float]:
    hits = [1 if item in relevant else 0 for item in recommendations[:k]]
    hit_count = sum(hits)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return {
        "precision": hit_count / k,
        "recall": hit_count / len(relevant),
        "hit_rate": float(hit_count > 0),
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def evaluate(
    popularity: Counter[int],
    history_seen: dict[int, set[int]],
    ground_truth: dict[int, set[int]],
    k_values: list[int],
) -> dict[str, object]:
    ranked_items = rank_items(popularity)
    totals = {k: Counter() for k in k_values}
    warm_users = 0
    cold_users = 0

    for user_id, relevant in ground_truth.items():
        seen_items = history_seen.get(user_id, set())
        if user_id in history_seen:
            warm_users += 1
        else:
            cold_users += 1
        recommendations = recommend(ranked_items, seen_items, max(k_values))
        for k in k_values:
            totals[k].update(user_metrics(recommendations, relevant, k))

    user_count = len(ground_truth)
    metrics = {
        f"@{k}": {name: value / user_count if user_count else 0.0 for name, value in totals[k].items()}
        for k in k_values
    }
    return {
        "metrics": metrics,
        "evaluated_users": user_count,
        "warm_users": warm_users,
        "cold_users": cold_users,
        "catalog_items_in_model": len(popularity),
        "top_10_items": ranked_items[:10],
    }


def run_backtest(processed_dir: Path, k_values: list[int]) -> dict[str, object]:
    train = processed_dir / "events_train.csv"
    validation = processed_dir / "events_validation.csv"
    test = processed_dir / "events_test.csv"
    for path in (train, validation, test):
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")

    train_popularity, train_seen, train_rows = load_history([train])
    validation_truth, validation_stats = load_ground_truth(validation, train_seen)
    validation_result = evaluate(train_popularity, train_seen, validation_truth, k_values)
    validation_result["ground_truth"] = validation_stats
    validation_result["history_rows"] = train_rows

    final_popularity, final_seen, final_history_rows = load_history([train, validation])
    test_truth, test_stats = load_ground_truth(test, final_seen)
    test_result = evaluate(final_popularity, final_seen, test_truth, k_values)
    test_result["ground_truth"] = test_stats
    test_result["history_rows"] = final_history_rows

    return {
        "model": "weighted_global_popularity",
        "popularity_score": "sum(event_weight), where view=1, addtocart=3, transaction=5",
        "relevance_definition": "unique unseen items with addtocart or transaction events",
        "evaluation_protocol": {
            "validation_history": "train",
            "test_history": "train + validation",
            "seen_item_filtering": True,
            "averaging": "macro average over eligible users",
            "k_values": k_values,
        },
        "validation": validation_result,
        "test": test_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/popularity_baseline.json")
    )
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = sorted(set(args.k))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("All K values must be positive")
    report = run_backtest(args.processed_dir, k_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
