#!/usr/bin/env python3
"""Generate expanding-window hybrid candidate folds for listwise ranking."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from build_item_categories import load_categories
from evaluate_hybrid_candidates import (
    build_category_popularity,
    build_session_transitions_from_events,
    reciprocal_rank_scores,
    source_rankings,
)
from evaluate_item_cooccurrence import build_neighbors
from generate_rolling_examples import (
    OUTPUT_HEADER,
    RELEVANT_EVENTS,
    candidate_pool,
    capped_history,
    load_temporal_partitions,
    positive_features,
    time_boundary,
    timestamp_range,
    update_state,
)


FOLDS = [(0.45, 0.60), (0.60, 0.75), (0.75, 0.90), (0.90, 1.00)]
FUSION_WEIGHTS = {
    "item_cooccurrence": 1.0,
    "session_transition": 1.5,
    "category_popularity": 0.75,
    "global_popularity": 0.25,
}


def enriched_hybrid_pool(
    history,
    seen,
    item_neighbors,
    session_neighbors,
    category_popularity,
    categories,
    popularity,
    ranked_popularity,
    timestamp,
    candidate_count,
    max_popularity_value,
):
    _, base_feature_map = candidate_pool(
        history,
        seen,
        item_neighbors,
        popularity,
        ranked_popularity,
        candidate_count,
        timestamp,
        categories,
        max_popularity_value,
    )
    rankings = source_rankings(
        history,
        seen,
        item_neighbors,
        session_neighbors,
        category_popularity,
        categories,
        ranked_popularity,
        candidate_count,
    )
    candidates, _, fusion_scores = reciprocal_rank_scores(
        rankings, FUSION_WEIGHTS, candidate_count
    )
    source_sets = {source: set(items) for source, items in rankings.items()}
    max_fusion_score = max(fusion_scores.values(), default=1.0)
    features = {}
    for candidate in candidates:
        base = base_feature_map.get(candidate)
        if base is None:
            base = positive_features(
                candidate,
                popularity,
                len(seen),
                categories,
                history,
                max_popularity_value,
            )
        features[candidate] = [
            *base,
            fusion_scores[candidate] / max_fusion_score,
            float(candidate in source_sets["item_cooccurrence"]),
            float(candidate in source_sets["session_transition"]),
            float(candidate in source_sets["category_popularity"]),
            float(candidate in source_sets["global_popularity"]),
        ]
    return candidates, features


def generate_fold(
    input_path: Path,
    categories: dict[int, int],
    minimum: int,
    maximum: int,
    graph_ratio: float,
    window_end_ratio: float,
    fold_index: int,
    max_history: int,
    candidate_count: int,
):
    graph_cutoff = time_boundary(minimum, maximum, graph_ratio)
    window_end = maximum + 1 if window_end_ratio == 1 else time_boundary(
        minimum, maximum, window_end_ratio
    )
    early, rolling, early_events, popularity, early_rows, _ = load_temporal_partitions(
        input_path, graph_cutoff
    )
    histories = {user: capped_history(state, max_history) for user, state in early.items()}
    item_neighbors, pair_updates = build_neighbors(histories, 100)
    session_neighbors, session_stats = build_session_transitions_from_events(
        early_events, 30, 200
    )
    category_popularity = build_category_popularity(popularity, categories, 100)
    ranked_popularity = [
        item for item, _ in sorted(popularity.items(), key=lambda value: (-value[1], value[0]))
    ]
    max_popularity_value = max(popularity.values(), default=1)
    role = "validation" if fold_index == len(FOLDS) - 1 else "train"
    groups = []
    skipped_not_retrieved = 0
    skipped_no_history = 0
    repeated_targets = 0
    for user_id in sorted(rolling):
        state = dict(early.get(user_id, {}))
        for timestamp, target_item, event, weight in sorted(
            rolling[user_id], key=lambda value: (value[0], value[1], value[2])
        ):
            if timestamp >= window_end:
                break
            if event in RELEVANT_EVENTS and target_item not in state:
                if not state:
                    skipped_no_history += 1
                else:
                    history = capped_history(state, max_history)
                    candidates, features = enriched_hybrid_pool(
                        history,
                        set(state),
                        item_neighbors,
                        session_neighbors,
                        category_popularity,
                        categories,
                        popularity,
                        ranked_popularity,
                        timestamp,
                        candidate_count,
                        max_popularity_value,
                    )
                    if target_item in features:
                        groups.append(
                            {
                                "fold": fold_index,
                                "role": role,
                                "timestamp": timestamp,
                                "user_id": user_id,
                                "target": target_item,
                                "history": [item for item, _, _ in history],
                                "candidates": candidates,
                                "features": features,
                            }
                        )
                    else:
                        skipped_not_retrieved += 1
            elif event in RELEVANT_EVENTS:
                repeated_targets += 1
            update_state(state, target_item, weight, timestamp)
    stats = {
        "fold": fold_index,
        "role": role,
        "graph_ratio": graph_ratio,
        "window_end_ratio": window_end_ratio,
        "graph_cutoff": graph_cutoff,
        "window_end": window_end,
        "early_graph_rows": early_rows,
        "retrieved_groups": len(groups),
        "skipped_not_retrieved": skipped_not_retrieved,
        "skipped_no_history": skipped_no_history,
        "repeated_targets": repeated_targets,
        "item_pair_updates": pair_updates,
        "session_graph": session_stats,
    }
    return groups, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/events_train.csv"))
    parser.add_argument(
        "--categories", type=Path, default=Path("data/processed/item_categories.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/listwise_temporal_folds.csv.gz")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/reports/temporal_folds.json")
    )
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--candidate-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = load_categories(args.categories)
    minimum, maximum = timestamp_range(args.input)
    all_groups = []
    fold_reports = []
    for fold_index, (graph_ratio, window_end_ratio) in enumerate(FOLDS):
        groups, stats = generate_fold(
            args.input,
            categories,
            minimum,
            maximum,
            graph_ratio,
            window_end_ratio,
            fold_index,
            args.max_history,
            args.candidate_count,
        )
        all_groups.extend(groups)
        fold_reports.append(stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with gzip.open(args.output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", *OUTPUT_HEADER])
        for example_id, group in enumerate(all_groups, start=1):
            history = " ".join(map(str, group["history"]))
            for candidate in group["candidates"]:
                writer.writerow(
                    [
                        group["fold"],
                        example_id,
                        group["role"],
                        group["timestamp"],
                        group["user_id"],
                        candidate,
                        int(candidate == group["target"]),
                        int(candidate == group["target"]),
                        history,
                        *[f"{value:.8f}" for value in group["features"][candidate]],
                    ]
                )
                row_count += 1
    report = {
        "strategy": "four_expanding_temporal_folds",
        "fold_boundaries": FOLDS,
        "injected_positives": 0,
        "candidate_count": args.candidate_count,
        "folds": fold_reports,
        "training_groups": sum(len_group["retrieved_groups"] for len_group in fold_reports[:-1]),
        "validation_groups": fold_reports[-1]["retrieved_groups"],
        "output_rows": row_count,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
