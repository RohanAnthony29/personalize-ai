#!/usr/bin/env python3
"""Generate leakage-safe rolling-window ranker examples with hard negatives."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from evaluate_item_cooccurrence import build_neighbors
from evaluate_hybrid_candidates import (
    build_category_popularity,
    build_session_transitions_from_events,
    reciprocal_rank_scores,
    source_rankings,
)


RELEVANT_EVENTS = {"addtocart", "transaction"}
OUTPUT_HEADER = [
    "example_id",
    "split",
    "prediction_timestamp",
    "user_id",
    "candidate_item_id",
    "label",
    "retrieved_positive",
    "history_item_ids",
    "cooccurrence_score",
    "popularity_score",
    "reciprocal_candidate_rank",
    "source_coverage",
    "normalized_history_length",
    "recency_weighted_score",
    "max_source_similarity",
    "category_affinity",
    "hybrid_rrf_score",
    "from_item_cooccurrence",
    "from_session_transition",
    "from_category_popularity",
    "from_global_popularity",
]


def timestamp_range(path: Path) -> tuple[int, int]:
    minimum: int | None = None
    maximum: int | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["timestamp"])
            minimum = timestamp if minimum is None else min(minimum, timestamp)
            maximum = timestamp if maximum is None else max(maximum, timestamp)
    if minimum is None or maximum is None:
        raise ValueError("Training split contains no events")
    return minimum, maximum


def time_boundary(minimum: int, maximum: int, ratio: float) -> int:
    if not 0 < ratio < 1:
        raise ValueError("Time ratio must be between zero and one")
    return minimum + int((maximum - minimum + 1) * ratio)


def load_temporal_partitions(
    path: Path, graph_cutoff: int
) -> tuple[
    dict[int, dict[int, tuple[int, int]]],
    dict[int, list[tuple[int, int, str, int]]],
    dict[int, list[tuple[int, int]]],
    Counter[int],
    int,
    int,
]:
    early: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
    rolling: dict[int, list[tuple[int, int, str, int]]] = defaultdict(list)
    early_events: dict[int, list[tuple[int, int]]] = defaultdict(list)
    popularity: Counter[int] = Counter()
    early_rows = 0
    rolling_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["timestamp"])
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            event = row["event_type"]
            weight = int(row["event_weight"])
            if timestamp < graph_cutoff:
                previous = early[user_id].get(item_id)
                if previous is None:
                    early[user_id][item_id] = (weight, timestamp)
                else:
                    early[user_id][item_id] = (
                        previous[0] + weight,
                        max(previous[1], timestamp),
                    )
                popularity[item_id] += weight
                early_events[user_id].append((timestamp, item_id))
                early_rows += 1
            else:
                rolling[user_id].append((timestamp, item_id, event, weight))
                rolling_rows += 1
    return (
        dict(early),
        dict(rolling),
        dict(early_events),
        popularity,
        early_rows,
        rolling_rows,
    )


def capped_history(
    state: dict[int, tuple[int, int]], max_history: int
) -> list[tuple[int, int, int]]:
    values = [(item, weight, timestamp) for item, (weight, timestamp) in state.items()]
    values.sort(key=lambda value: (-value[2], value[0]))
    return values[:max_history]


def candidate_pool(
    history: list[tuple[int, int, int]],
    full_seen: set[int],
    neighbors: dict[int, list[tuple[int, float]]],
    popularity: Counter[int],
    ranked_popularity: list[int],
    pool_size: int,
    prediction_timestamp: int,
    categories: dict[int, int],
    max_popularity_value: int | None = None,
) -> tuple[list[int], dict[int, list[float]]]:
    scores: Counter[int] = Counter()
    recency_scores: Counter[int] = Counter()
    sources: Counter[int] = Counter()
    max_similarities: Counter[int] = Counter()
    for source_item, source_weight, source_timestamp in history:
        strength = 1.0 + math.log1p(source_weight)
        age_days = max(prediction_timestamp - source_timestamp, 0) / 86_400_000
        recency_weight = math.exp(-age_days / 30.0)
        for candidate, similarity in neighbors.get(source_item, []):
            if candidate not in full_seen:
                scores[candidate] += strength * similarity
                recency_scores[candidate] += strength * similarity * recency_weight
                sources[candidate] += 1
                max_similarities[candidate] = max(max_similarities[candidate], similarity)
    candidates = sorted(scores, key=lambda item: (-scores[item], item))[:pool_size]
    selected = set(candidates)
    for item in ranked_popularity:
        if len(candidates) == pool_size:
            break
        if item not in full_seen and item not in selected:
            candidates.append(item)
            selected.add(item)

    max_score = max(scores.values(), default=1.0)
    max_recency_score = max(recency_scores.values(), default=1.0)
    max_popularity = (
        max_popularity_value
        if max_popularity_value is not None
        else max(popularity.values(), default=1)
    )
    history_categories = [categories[item] for item, _, _ in history if item in categories]
    category_counts = Counter(history_categories)
    features = {}
    for rank, item in enumerate(candidates, start=1):
        features[item] = [
            scores[item] / max_score,
            math.log1p(popularity[item]) / math.log1p(max_popularity),
            1.0 / rank,
            sources[item] / max(len(history), 1),
            math.log1p(len(full_seen)) / math.log1p(50),
            recency_scores[item] / max_recency_score,
            max_similarities[item],
            category_counts[categories[item]] / len(history_categories)
            if item in categories and history_categories
            else 0.0,
        ]
    return candidates, features


def positive_features(
    item_id: int,
    popularity: Counter[int],
    full_seen_count: int,
    categories: dict[int, int],
    history: list[tuple[int, int, int]],
    max_popularity_value: int | None = None,
) -> list[float]:
    max_popularity = (
        max_popularity_value
        if max_popularity_value is not None
        else max(popularity.values(), default=1)
    )
    history_categories = [categories[item] for item, _, _ in history if item in categories]
    category_counts = Counter(history_categories)
    return [
        0.0,
        math.log1p(popularity[item_id]) / math.log1p(max_popularity),
        0.0,
        0.0,
        math.log1p(full_seen_count) / math.log1p(50),
        0.0,
        0.0,
        category_counts[categories[item_id]] / len(history_categories)
        if item_id in categories and history_categories
        else 0.0,
    ]


def load_categories(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["item_id"]): int(row["category_id"])
            for row in csv.DictReader(handle)
        }


def update_state(
    state: dict[int, tuple[int, int]], item: int, weight: int, timestamp: int
) -> None:
    previous = state.get(item)
    state[item] = (
        weight if previous is None else previous[0] + weight,
        timestamp if previous is None else max(previous[1], timestamp),
    )


def iso_timestamp(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def generate(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    graph_ratio: float,
    example_train_ratio: float,
    max_history: int,
    neighbors_per_item: int,
    candidate_pool_size: int,
    negatives_per_positive: int,
    category_path: Path,
) -> dict[str, object]:
    minimum, maximum = timestamp_range(input_path)
    graph_cutoff = time_boundary(minimum, maximum, graph_ratio)
    example_validation_cutoff = time_boundary(graph_cutoff, maximum, example_train_ratio)
    early, rolling, early_events, popularity, early_rows, rolling_rows = (
        load_temporal_partitions(input_path, graph_cutoff)
    )
    graph_histories = {
        user: capped_history(state, max_history) for user, state in early.items()
    }
    neighbors, pair_updates = build_neighbors(graph_histories, neighbors_per_item)
    session_neighbors, session_stats = build_session_transitions_from_events(
        early_events, session_gap_minutes=30, max_events_per_user=200
    )
    ranked_popularity = [
        item for item, _ in sorted(popularity.items(), key=lambda value: (-value[1], value[0]))
    ]
    max_popularity_value = max(popularity.values(), default=1)
    categories = load_categories(category_path)
    category_popularity = build_category_popularity(popularity, categories, 100)
    fusion_weights = {
        "item_cooccurrence": 1.0,
        "session_transition": 1.5,
        "category_popularity": 0.75,
        "global_popularity": 0.25,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = Counter()
    row_counts = Counter()
    retrieved_positives = Counter()
    item_only_retrieved_positives = Counter()
    skipped_no_history = 0
    skipped_repeat_target = 0
    example_id = 0
    with gzip.open(output_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_HEADER)
        for user_id in sorted(rolling):
            state = dict(early.get(user_id, {}))
            events = sorted(rolling[user_id], key=lambda value: (value[0], value[1], value[2]))
            for timestamp, target_item, event, weight in events:
                if event in RELEVANT_EVENTS and target_item not in state:
                    if not state:
                        skipped_no_history += 1
                    else:
                        history = capped_history(state, max_history)
                        seen = set(state)
                        item_pool, base_feature_map = candidate_pool(
                            history,
                            seen,
                            neighbors,
                            popularity,
                            ranked_popularity,
                            candidate_pool_size,
                            timestamp,
                            categories,
                            max_popularity_value,
                        )
                        rankings = source_rankings(
                            history,
                            seen,
                            neighbors,
                            session_neighbors,
                            category_popularity,
                            categories,
                            ranked_popularity,
                            candidate_pool_size,
                        )
                        pool, _, fusion_scores = reciprocal_rank_scores(
                            rankings,
                            fusion_weights,
                            candidate_pool_size,
                        )
                        source_sets = {
                            source: set(items) for source, items in rankings.items()
                        }
                        max_fusion_score = max(fusion_scores.values(), default=1.0)
                        feature_map = {}
                        for candidate in pool:
                            base_features = base_feature_map.get(candidate)
                            if base_features is None:
                                base_features = positive_features(
                                    candidate,
                                    popularity,
                                    len(seen),
                                    categories,
                                    history,
                                    max_popularity_value,
                                )
                            feature_map[candidate] = [
                                *base_features,
                                fusion_scores[candidate] / max_fusion_score,
                                float(candidate in source_sets["item_cooccurrence"]),
                                float(candidate in source_sets["session_transition"]),
                                float(candidate in source_sets["category_popularity"]),
                                float(candidate in source_sets["global_popularity"]),
                            ]
                        retrieved = target_item in feature_map
                        split = (
                            "train" if timestamp < example_validation_cutoff else "validation"
                        )
                        if split == "train":
                            negatives = [item for item in pool if item != target_item][
                                :negatives_per_positive
                            ]
                            candidates = [target_item, *negatives]
                        elif retrieved:
                            # Match serving evaluation: retain the complete retrieved pool.
                            candidates = pool
                        else:
                            # Keep the target for auditing, but these groups are not used
                            # for conditional ranker early stopping.
                            candidates = [target_item, *pool]
                        example_id += 1
                        examples[split] += 1
                        retrieved_positives[split] += int(retrieved)
                        item_only_retrieved_positives[split] += int(
                            target_item in item_pool
                        )
                        history_text = " ".join(str(item) for item, _, _ in history)
                        for candidate in candidates:
                            label = int(candidate == target_item)
                            features = feature_map.get(
                                candidate,
                                positive_features(
                                    candidate,
                                    popularity,
                                    len(seen),
                                    categories,
                                    history,
                                    max_popularity_value,
                                )
                                + [0.0, 0.0, 0.0, 0.0, 0.0],
                            )
                            writer.writerow(
                                [
                                    example_id,
                                    split,
                                    timestamp,
                                    user_id,
                                    candidate,
                                    label,
                                    int(retrieved) if label else 0,
                                    history_text,
                                    *[f"{value:.8f}" for value in features],
                                ]
                            )
                            row_counts[split] += 1
                elif event in RELEVANT_EVENTS:
                    skipped_repeat_target += 1
                update_state(state, target_item, weight, timestamp)

    report = {
        "strategy": "frozen_early_graph_with_rolling_point_in_time_examples",
        "source": str(input_path),
        "output": str(output_path),
        "source_min_utc": iso_timestamp(minimum),
        "source_max_utc": iso_timestamp(maximum),
        "graph_cutoff_utc": iso_timestamp(graph_cutoff),
        "example_validation_cutoff_utc": iso_timestamp(example_validation_cutoff),
        "early_graph_rows": early_rows,
        "rolling_source_rows": rolling_rows,
        "graph_users": len(graph_histories),
        "items_with_neighbors": len(neighbors),
        "pair_updates": pair_updates,
        "examples": dict(examples),
        "output_rows": dict(row_counts),
        "retrieved_positives": dict(retrieved_positives),
        "item_only_retrieved_positives": dict(item_only_retrieved_positives),
        "skipped_positive_events_without_history": skipped_no_history,
        "skipped_repeat_positive_events": skipped_repeat_target,
        "max_history": max_history,
        "neighbors_per_item": neighbors_per_item,
        "candidate_pool_size": candidate_pool_size,
        "negatives_per_positive": negatives_per_positive,
        "items_with_category": len(categories),
        "session_graph": session_stats,
        "fusion_weights": fusion_weights,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/events_train.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/rolling_ranker_examples.csv.gz")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/reports/rolling_examples.json")
    )
    parser.add_argument("--graph-ratio", type=float, default=0.70)
    parser.add_argument("--example-train-ratio", type=float, default=0.80)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--neighbors-per-item", type=int, default=100)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--negatives-per-positive", type=int, default=20)
    parser.add_argument(
        "--categories", type=Path, default=Path("data/processed/item_categories.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate(
        args.input,
        args.output,
        args.report,
        args.graph_ratio,
        args.example_train_ratio,
        args.max_history,
        args.neighbors_per_item,
        args.candidate_pool_size,
        args.negatives_per_positive,
        args.categories,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
