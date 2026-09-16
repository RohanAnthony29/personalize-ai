#!/usr/bin/env python3
"""Evaluate a diversified, session-aware candidate generation layer."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_item_cooccurrence import (
    build_neighbors,
    cooccurrence_recommendations,
    load_history,
    load_warm_ground_truth,
)
from build_item_categories import load_categories


def build_session_transitions(
    paths: list[Path], session_gap_minutes: int, max_events_per_user: int
) -> tuple[dict[int, list[tuple[int, float]]], dict[str, int]]:
    events: dict[int, list[tuple[int, int]]] = defaultdict(list)
    rows = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                events[int(row["user_id"])].append(
                    (int(row["timestamp"]), int(row["item_id"]))
                )
                rows += 1

    neighbors, stats = build_session_transitions_from_events(
        events, session_gap_minutes, max_events_per_user
    )
    stats["source_rows"] = rows
    return neighbors, stats


def build_session_transitions_from_events(
    events: dict[int, list[tuple[int, int]]],
    session_gap_minutes: int,
    max_events_per_user: int,
) -> tuple[dict[int, list[tuple[int, float]]], dict[str, int]]:

    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    outgoing: Counter[int] = Counter()
    incoming: Counter[int] = Counter()
    transition_updates = 0
    capped_users = 0
    gap_ms = session_gap_minutes * 60_000
    for user_events in events.values():
        user_events.sort(key=lambda event: (event[0], event[1]))
        if len(user_events) > max_events_per_user:
            capped_users += 1
            user_events = user_events[-max_events_per_user:]
        for (left_time, left_item), (right_time, right_item) in zip(
            user_events, user_events[1:]
        ):
            if right_time - left_time > gap_ms or left_item == right_item:
                continue
            transitions[left_item][right_item] += 1
            outgoing[left_item] += 1
            incoming[right_item] += 1
            transition_updates += 1

    normalized = {}
    for item, counts in transitions.items():
        scores = [
            (candidate, count / math.sqrt(outgoing[item] * incoming[candidate]))
            for candidate, count in counts.items()
        ]
        scores.sort(key=lambda value: (-value[1], value[0]))
        normalized[item] = scores[:100]
    return normalized, {
        "transition_updates": transition_updates,
        "items_with_outgoing_transitions": len(normalized),
        "users_capped_at_max_events": capped_users,
    }


def build_category_popularity(
    popularity: Counter[int], categories: dict[int, int], items_per_category: int
) -> dict[int, list[int]]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for item, score in popularity.items():
        if item in categories:
            grouped[categories[item]].append((item, score))
    result = {}
    for category, values in grouped.items():
        values.sort(key=lambda value: (-value[1], value[0]))
        result[category] = [item for item, _ in values[:items_per_category]]
    return result


def source_rankings(
    history: list[tuple[int, int, int]],
    seen: set[int],
    item_neighbors: dict[int, list[tuple[int, float]]],
    session_neighbors: dict[int, list[tuple[int, float]]],
    category_popularity: dict[int, list[int]],
    categories: dict[int, int],
    ranked_popularity: list[int],
    source_limit: int = 100,
) -> dict[str, list[int]]:
    item_scores: Counter[int] = Counter()
    session_scores: Counter[int] = Counter()
    for position, (source_item, source_weight, _) in enumerate(history):
        strength = (1 + math.log1p(source_weight)) / (position + 1)
        for candidate, similarity in item_neighbors.get(source_item, []):
            if candidate not in seen:
                item_scores[candidate] += strength * similarity
        # Directed transitions are strongest for the most recent source items.
        if position < 10:
            for candidate, similarity in session_neighbors.get(source_item, []):
                if candidate not in seen:
                    session_scores[candidate] += strength * similarity

    recent_categories = []
    for item, _, _ in history:
        category = categories.get(item)
        if category is not None and category not in recent_categories:
            recent_categories.append(category)
        if len(recent_categories) == 5:
            break
    category_items = []
    category_seen = set()
    for category in recent_categories:
        for item in category_popularity.get(category, []):
            if item not in seen and item not in category_seen:
                category_items.append(item)
                category_seen.add(item)

    return {
        "item_cooccurrence": sorted(
            item_scores, key=lambda item: (-item_scores[item], item)
        )[:source_limit],
        "session_transition": sorted(
            session_scores, key=lambda item: (-session_scores[item], item)
        )[:source_limit],
        "category_popularity": category_items[:source_limit],
        "global_popularity": limited_unseen_items(
            ranked_popularity, seen, source_limit
        ),
    }


def limited_unseen_items(
    ranked_items: list[int], seen: set[int], limit: int
) -> list[int]:
    result = []
    for item in ranked_items:
        if item not in seen:
            result.append(item)
            if len(result) == limit:
                break
    return result


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]],
    weights: dict[str, float],
    candidate_count: int,
    rank_constant: int = 60,
) -> tuple[list[int], dict[str, int]]:
    fused, contribution, _ = reciprocal_rank_scores(
        rankings, weights, candidate_count, rank_constant
    )
    return fused, contribution


def reciprocal_rank_scores(
    rankings: dict[str, list[int]],
    weights: dict[str, float],
    candidate_count: int,
    rank_constant: int = 60,
) -> tuple[list[int], dict[str, int], Counter[int]]:
    scores: Counter[int] = Counter()
    source_candidates: dict[str, set[int]] = {}
    for source, ranking in rankings.items():
        source_candidates[source] = set(ranking[:candidate_count])
        for rank, item in enumerate(ranking[:candidate_count], start=1):
            scores[item] += weights[source] / (rank_constant + rank)
    fused = sorted(scores, key=lambda item: (-scores[item], item))[:candidate_count]
    contribution = {
        source: sum(item in items for item in fused)
        for source, items in source_candidates.items()
    }
    return fused, contribution, scores


def candidate_metrics(candidates: list[int], relevant: set[int], k: int) -> dict[str, float]:
    hits = len(set(candidates[:k]) & relevant)
    return {
        "recall": hits / len(relevant),
        "coverage": float(hits > 0),
        "hits": float(hits),
    }


def evaluate_window(
    history_paths: list[Path],
    target_path: Path,
    categories: dict[int, int],
    k_values: list[int],
    max_history: int,
    session_gap_minutes: int,
    max_events_per_user: int,
) -> tuple[dict[str, object], dict[int, list[tuple[int, float]]], dict[int, list[int]]]:
    histories, full_seen, popularity, history_rows, capped_history_users = load_history(
        history_paths, max_history
    )
    item_neighbors, pair_updates = build_neighbors(histories, 100)
    session_neighbors, session_stats = build_session_transitions(
        history_paths, session_gap_minutes, max_events_per_user
    )
    category_popularity = build_category_popularity(popularity, categories, 100)
    truth, truth_stats = load_warm_ground_truth(target_path, full_seen)
    ranked_popularity = [
        item for item, _ in sorted(popularity.items(), key=lambda value: (-value[1], value[0]))
    ]
    weights = {
        "item_cooccurrence": 1.0,
        "session_transition": 1.5,
        "category_popularity": 0.75,
        "global_popularity": 0.25,
    }
    totals = {
        model: {k: Counter() for k in k_values}
        for model in ("item_cooccurrence", "hybrid")
    }
    source_totals = Counter()
    source_nonempty_users = Counter()
    for user_id, relevant in truth.items():
        history = histories[user_id]
        seen = full_seen[user_id]
        baseline, _ = cooccurrence_recommendations(
            history, seen, item_neighbors, ranked_popularity, max(k_values)
        )
        rankings = source_rankings(
            history,
            seen,
            item_neighbors,
            session_neighbors,
            category_popularity,
            categories,
            ranked_popularity,
        )
        hybrid, contribution = reciprocal_rank_fusion(rankings, weights, max(k_values))
        for source, ranking in rankings.items():
            source_nonempty_users[source] += int(bool(ranking))
        source_totals.update(contribution)
        for k in k_values:
            totals["item_cooccurrence"][k].update(candidate_metrics(baseline, relevant, k))
            totals["hybrid"][k].update(candidate_metrics(hybrid, relevant, k))

    users = len(truth)
    rendered = {
        model: {
            f"@{k}": {
                metric: value / users if users else 0.0
                for metric, value in totals[model][k].items()
            }
            for k in k_values
        }
        for model in totals
    }
    return (
        {
            "eligible_warm_users": users,
            "metrics": rendered,
            "ground_truth": truth_stats,
            "history_rows": history_rows,
            "history_users_capped": capped_history_users,
            "item_pair_updates": pair_updates,
            "session_graph": session_stats,
            "categories_with_popular_items": len(category_popularity),
            "source_nonempty_user_fraction": {
                source: count / users if users else 0.0
                for source, count in source_nonempty_users.items()
            },
            "average_fused_slots_by_source": {
                source: count / users if users else 0.0
                for source, count in source_totals.items()
            },
            "fusion_weights": weights,
        },
        session_neighbors,
        category_popularity,
    )


def write_session_artifact(
    neighbors: dict[int, list[tuple[int, float]]], path: Path
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "next_item_id", "transition_score", "rank"])
        for item in sorted(neighbors):
            for rank, (candidate, score) in enumerate(neighbors[item], start=1):
                writer.writerow([item, candidate, f"{score:.8f}", rank])
                rows += 1
    return rows


def write_category_artifact(category_items: dict[int, list[int]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category_id", "item_id", "rank"])
        for category in sorted(category_items):
            for rank, item in enumerate(category_items[category], start=1):
                writer.writerow([category, item, rank])
                rows += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--categories", type=Path, default=Path("data/processed/item_categories.csv")
    )
    parser.add_argument("--report", type=Path, default=Path("data/reports/hybrid_candidates.json"))
    parser.add_argument("--session-artifact", type=Path, default=Path("artifacts/session_neighbors.csv.gz"))
    parser.add_argument("--category-artifact", type=Path, default=Path("artifacts/category_popular.csv.gz"))
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--max-events-per-user", type=int, default=200)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = sorted(set(args.k))
    categories = load_categories(args.categories)
    train = args.processed_dir / "events_train.csv"
    validation = args.processed_dir / "events_validation.csv"
    test = args.processed_dir / "events_test.csv"
    validation_report, _, _ = evaluate_window(
        [train],
        validation,
        categories,
        k_values,
        args.max_history,
        args.session_gap_minutes,
        args.max_events_per_user,
    )
    test_report, session_neighbors, category_popularity = evaluate_window(
        [train, validation],
        test,
        categories,
        k_values,
        args.max_history,
        args.session_gap_minutes,
        args.max_events_per_user,
    )
    session_rows = write_session_artifact(session_neighbors, args.session_artifact)
    category_rows = write_category_artifact(category_popularity, args.category_artifact)
    report = {
        "model": "hybrid_reciprocal_rank_fusion_candidates",
        "session_gap_minutes": args.session_gap_minutes,
        "validation": validation_report,
        "test": test_report,
        "serving_artifacts": {
            "session_neighbors": {"path": str(args.session_artifact), "rows": session_rows},
            "category_popular": {"path": str(args.category_artifact), "rows": category_rows},
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
