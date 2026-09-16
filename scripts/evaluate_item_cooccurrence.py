#!/usr/bin/env python3
"""Build and evaluate an item-to-item co-occurrence candidate generator."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


RELEVANT_EVENTS = {"addtocart", "transaction"}


def load_history(
    paths: list[Path], max_history: int
) -> tuple[
    dict[int, list[tuple[int, int, int]]], dict[int, set[int]], Counter[int], int, int
]:
    """Return recent unique (item, weight, timestamp) histories and popularity."""
    raw: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
    popularity: Counter[int] = Counter()
    rows = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                user_id = int(row["user_id"])
                item_id = int(row["item_id"])
                timestamp = int(row["timestamp"])
                weight = int(row["event_weight"])
                popularity[item_id] += weight
                previous = raw[user_id].get(item_id)
                if previous is None:
                    raw[user_id][item_id] = (weight, timestamp)
                else:
                    raw[user_id][item_id] = (previous[0] + weight, max(previous[1], timestamp))
                rows += 1

    capped_users = 0
    histories: dict[int, list[tuple[int, int, int]]] = {}
    full_seen: dict[int, set[int]] = {}
    for user_id, item_values in raw.items():
        full_seen[user_id] = set(item_values)
        history = [(item, weight, timestamp) for item, (weight, timestamp) in item_values.items()]
        history.sort(key=lambda value: (-value[2], value[0]))
        if len(history) > max_history:
            capped_users += 1
            history = history[:max_history]
        histories[user_id] = history
    return histories, full_seen, popularity, rows, capped_users


def build_neighbors(
    histories: dict[int, list[tuple[int, int, int]]], neighbors_per_item: int
) -> tuple[dict[int, list[tuple[int, float]]], int]:
    """Build cosine-normalized item neighbors from unique user histories."""
    item_users: Counter[int] = Counter()
    pairs: dict[int, Counter[int]] = defaultdict(Counter)
    pair_updates = 0
    for history in histories.values():
        items = [item for item, _, _ in history]
        item_users.update(items)
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                pairs[left][right] += 1
                pairs[right][left] += 1
                pair_updates += 1

    neighbors: dict[int, list[tuple[int, float]]] = {}
    for item, counts in pairs.items():
        scored = [
            (other, count / math.sqrt(item_users[item] * item_users[other]))
            for other, count in counts.items()
        ]
        scored.sort(key=lambda value: (-value[1], value[0]))
        neighbors[item] = scored[:neighbors_per_item]
    return neighbors, pair_updates


def load_warm_ground_truth(
    path: Path, full_seen: dict[int, set[int]]
) -> tuple[dict[int, set[int]], dict[str, int]]:
    truth: dict[int, set[int]] = defaultdict(set)
    relevant_rows = 0
    repeated_rows = 0
    cold_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] not in RELEVANT_EVENTS:
                continue
            relevant_rows += 1
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            if user_id not in full_seen:
                cold_rows += 1
            elif item_id in full_seen[user_id]:
                repeated_rows += 1
            else:
                truth[user_id].add(item_id)
    return dict(truth), {
        "relevant_event_rows": relevant_rows,
        "cold_user_event_rows_excluded": cold_rows,
        "repeated_seen_event_rows_excluded": repeated_rows,
        "eligible_warm_users": len(truth),
    }


def popularity_recommendations(
    ranked_popularity: list[int], seen: set[int], k: int
) -> list[int]:
    recommendations = []
    for item in ranked_popularity:
        if item not in seen:
            recommendations.append(item)
            if len(recommendations) == k:
                break
    return recommendations


def cooccurrence_recommendations(
    history: list[tuple[int, int, int]],
    full_seen: set[int],
    neighbors: dict[int, list[tuple[int, float]]],
    ranked_popularity: list[int],
    k: int,
) -> tuple[list[int], int]:
    scores: Counter[int] = Counter()
    for source_item, source_weight, _ in history:
        source_strength = 1.0 + math.log1p(source_weight)
        for candidate, similarity in neighbors.get(source_item, []):
            if candidate not in full_seen:
                scores[candidate] += source_strength * similarity
    ranked = [item for item, _ in sorted(scores.items(), key=lambda value: (-value[1], value[0]))]
    cooccurrence_count = min(len(ranked), k)
    selected = ranked[:k]
    selected_set = set(selected)
    if len(selected) < k:
        for item in ranked_popularity:
            if item not in full_seen and item not in selected_set:
                selected.append(item)
                selected_set.add(item)
                if len(selected) == k:
                    break
    return selected, cooccurrence_count


def metrics(recommendations: list[int], relevant: set[int], k: int) -> dict[str, float]:
    hits = [int(item in relevant) for item in recommendations[:k]]
    hit_count = sum(hits)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return {
        "precision": hit_count / k,
        "recall": hit_count / len(relevant),
        "hit_rate": float(hit_count > 0),
        "ndcg": dcg / ideal,
    }


def evaluate_window(
    history_paths: list[Path],
    target_path: Path,
    k_values: list[int],
    max_history: int,
    neighbors_per_item: int,
) -> dict[str, object]:
    histories, full_seen, popularity, history_rows, capped_users = load_history(
        history_paths, max_history
    )
    neighbors, pair_updates = build_neighbors(histories, neighbors_per_item)
    truth, truth_stats = load_warm_ground_truth(target_path, full_seen)
    ranked_popularity = [
        item for item, _ in sorted(popularity.items(), key=lambda value: (-value[1], value[0]))
    ]
    totals = {
        model: {k: Counter() for k in k_values} for model in ("popularity", "cooccurrence")
    }
    users_with_cooccurrence_candidates = 0
    max_k = max(k_values)
    for user_id, relevant in truth.items():
        history = histories[user_id]
        seen = full_seen[user_id]
        popularity_recs = popularity_recommendations(ranked_popularity, seen, max_k)
        cooccurrence_recs, generated = cooccurrence_recommendations(
            history, seen, neighbors, ranked_popularity, max_k
        )
        users_with_cooccurrence_candidates += int(generated > 0)
        for k in k_values:
            totals["popularity"][k].update(metrics(popularity_recs, relevant, k))
            totals["cooccurrence"][k].update(metrics(cooccurrence_recs, relevant, k))

    user_count = len(truth)
    rendered_metrics = {}
    for model in totals:
        rendered_metrics[model] = {
            f"@{k}": {
                name: value / user_count if user_count else 0.0
                for name, value in totals[model][k].items()
            }
            for k in k_values
        }
    improvements = {}
    for k in k_values:
        improvements[f"@{k}"] = {}
        for name, cooccurrence_value in rendered_metrics["cooccurrence"][f"@{k}"].items():
            popularity_value = rendered_metrics["popularity"][f"@{k}"][name]
            improvements[f"@{k}"][f"{name}_relative_pct"] = (
                (cooccurrence_value / popularity_value - 1) * 100 if popularity_value else None
            )

    return {
        "metrics": rendered_metrics,
        "relative_improvement": improvements,
        "ground_truth": truth_stats,
        "history_rows": history_rows,
        "history_users": len(histories),
        "users_capped_at_max_history": capped_users,
        "pair_updates": pair_updates,
        "items_with_neighbors": len(neighbors),
        "users_with_cooccurrence_candidates": users_with_cooccurrence_candidates,
        "cooccurrence_candidate_coverage": (
            users_with_cooccurrence_candidates / user_count if user_count else 0.0
        ),
    }


def write_neighbor_artifact(
    neighbors: dict[int, list[tuple[int, float]]], output: Path
) -> int:
    """Write a compact, language-agnostic candidate lookup artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "neighbor_item_id", "similarity", "rank"])
        for item_id in sorted(neighbors):
            for rank, (neighbor_id, similarity) in enumerate(neighbors[item_id], start=1):
                writer.writerow([item_id, neighbor_id, f"{similarity:.8f}", rank])
                rows += 1
    return rows


def run_backtest(
    processed_dir: Path, k_values: list[int], max_history: int, neighbors_per_item: int
) -> dict[str, object]:
    train = processed_dir / "events_train.csv"
    validation = processed_dir / "events_validation.csv"
    test = processed_dir / "events_test.csv"
    return {
        "model": "item_to_item_cosine_cooccurrence",
        "protocol": {
            "relevance": "unique unseen addtocart or transaction items for warm users",
            "max_unique_history_items": max_history,
            "neighbors_per_item": neighbors_per_item,
            "fallback": "weighted global popularity",
            "k_values": k_values,
        },
        "validation": evaluate_window(
            [train], validation, k_values, max_history, neighbors_per_item
        ),
        "test": evaluate_window(
            [train, validation], test, k_values, max_history, neighbors_per_item
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/item_cooccurrence.json")
    )
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--neighbors-per-item", type=int, default=100)
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/item_neighbors.csv.gz")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = sorted(set(args.k))
    if any(value <= 0 for value in [*k_values, args.max_history, args.neighbors_per_item]):
        raise ValueError("K, max history, and neighbor count must be positive")
    report = run_backtest(
        args.processed_dir, k_values, args.max_history, args.neighbors_per_item
    )
    train = args.processed_dir / "events_train.csv"
    validation = args.processed_dir / "events_validation.csv"
    histories, _, _, artifact_history_rows, _ = load_history(
        [train, validation], args.max_history
    )
    neighbors, _ = build_neighbors(histories, args.neighbors_per_item)
    artifact_rows = write_neighbor_artifact(neighbors, args.artifact)
    report["candidate_artifact"] = {
        "path": str(args.artifact),
        "training_history": "train + validation",
        "history_rows": artifact_history_rows,
        "items_with_neighbors": len(neighbors),
        "neighbor_rows": artifact_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
