#!/usr/bin/env python3
"""Train a residual neural ranker with BPR loss and NDCG early stopping."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_item_cooccurrence import build_neighbors, load_history, load_warm_ground_truth
from evaluate_hybrid_candidates import (
    build_category_popularity,
    build_session_transitions,
    reciprocal_rank_scores,
    source_rankings,
)
from build_item_categories import load_categories
from generate_rolling_examples import candidate_pool
from train_neural_ranker import CandidateGroup


FEATURE_COLUMNS = [
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


@dataclass
class RollingGroup:
    example_id: int
    split: str
    timestamp: int
    history: list[int]
    candidates: list[int]
    features: list[list[float]]
    labels: list[int]
    retrieved_positive: bool


def load_rolling_groups(path: Path) -> list[RollingGroup]:
    grouped: dict[int, RollingGroup] = {}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            example_id = int(row["example_id"])
            if example_id not in grouped:
                grouped[example_id] = RollingGroup(
                    example_id=example_id,
                    split=row["split"],
                    timestamp=int(row["prediction_timestamp"]),
                    history=[int(item) for item in row["history_item_ids"].split()],
                    candidates=[],
                    features=[],
                    labels=[],
                    retrieved_positive=False,
                )
            group = grouped[example_id]
            label = int(row["label"])
            group.candidates.append(int(row["candidate_item_id"]))
            group.features.append([float(row[column]) for column in FEATURE_COLUMNS])
            group.labels.append(label)
            if label:
                group.retrieved_positive = bool(int(row["retrieved_positive"]))
    groups = list(grouped.values())
    groups.sort(key=lambda group: (group.timestamp, group.example_id))
    return groups


def build_item_mapping(history_paths: list[Path]) -> dict[int, int]:
    items = set()
    for path in history_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                items.add(int(row["item_id"]))
    return {item: index for index, item in enumerate(sorted(items), start=1)}


def build_enriched_test_groups(
    history_paths: list[Path],
    target_path: Path,
    categories: dict[int, int],
    max_history: int,
    neighbors_per_item: int,
    candidate_count: int,
) -> tuple[list[CandidateGroup], dict[str, int]]:
    histories, full_seen, popularity, _, _ = load_history(history_paths, max_history)
    neighbors, _ = build_neighbors(histories, neighbors_per_item)
    session_neighbors, _ = build_session_transitions(history_paths, 30, 200)
    category_popularity = build_category_popularity(popularity, categories, 100)
    truth, truth_stats = load_warm_ground_truth(target_path, full_seen)
    target_timestamps: dict[int, int] = {}
    with target_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            if user_id in truth and item_id in truth[user_id]:
                timestamp = int(row["timestamp"])
                target_timestamps[user_id] = min(
                    timestamp, target_timestamps.get(user_id, timestamp)
                )
    ranked_popularity = [
        item for item, _ in sorted(popularity.items(), key=lambda value: (-value[1], value[0]))
    ]
    max_popularity_value = max(popularity.values(), default=1)
    fusion_weights = {
        "item_cooccurrence": 1.0,
        "session_transition": 1.5,
        "category_popularity": 0.75,
        "global_popularity": 0.25,
    }
    groups = []
    for user_id, relevant in truth.items():
        _, base_feature_map = candidate_pool(
            histories[user_id],
            full_seen[user_id],
            neighbors,
            popularity,
            ranked_popularity,
            candidate_count,
            target_timestamps[user_id],
            categories,
            max_popularity_value,
        )
        rankings = source_rankings(
            histories[user_id],
            full_seen[user_id],
            neighbors,
            session_neighbors,
            category_popularity,
            categories,
            ranked_popularity,
            candidate_count,
        )
        candidates, _, fusion_scores = reciprocal_rank_scores(
            rankings, fusion_weights, candidate_count
        )
        source_sets = {source: set(items) for source, items in rankings.items()}
        max_fusion_score = max(fusion_scores.values(), default=1.0)
        features = []
        for candidate in candidates:
            base = base_feature_map.get(candidate, [0.0] * 8)
            features.append(
                [
                    *base,
                    fusion_scores[candidate] / max_fusion_score,
                    float(candidate in source_sets["item_cooccurrence"]),
                    float(candidate in source_sets["session_transition"]),
                    float(candidate in source_sets["category_popularity"]),
                    float(candidate in source_sets["global_popularity"]),
                ]
            )
        groups.append(
            CandidateGroup(
                user_id=user_id,
                history=[item for item, _, _ in histories[user_id]],
                candidates=candidates,
                features=features,
                labels=[int(item in relevant) for item in candidates],
            )
        )
    truth_stats["candidate_groups"] = len(groups)
    truth_stats["groups_with_retrieved_positive"] = sum(any(group.labels) for group in groups)
    truth_stats["candidate_rows"] = sum(len(group.candidates) for group in groups)
    return groups, truth_stats


def pad_history(history: list[int], mapping: dict[int, int], max_history: int) -> list[int]:
    encoded = [mapping.get(item, 0) for item in history[:max_history]]
    return encoded + [0] * (max_history - len(encoded))


def build_pair_dataset(
    groups: list[RollingGroup], mapping: dict[int, int], max_history: int
) -> TensorDataset:
    positives = []
    negatives = []
    histories = []
    positive_features = []
    negative_features = []
    pair_weights = []
    for group in groups:
        positive_indexes = [index for index, label in enumerate(group.labels) if label]
        negative_indexes = [index for index, label in enumerate(group.labels) if not label]
        if len(positive_indexes) != 1:
            raise ValueError(f"Example {group.example_id} must contain exactly one positive")
        positive_index = positive_indexes[0]
        encoded_history = pad_history(group.history, mapping, max_history)
        for negative_index in negative_indexes:
            positives.append(mapping.get(group.candidates[positive_index], 0))
            negatives.append(mapping.get(group.candidates[negative_index], 0))
            histories.append(encoded_history)
            positive_features.append(group.features[positive_index])
            negative_features.append(group.features[negative_index])
            pair_weights.append(1.0 if group.retrieved_positive else 0.25)
    return TensorDataset(
        torch.tensor(positives, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
        torch.tensor(histories, dtype=torch.long),
        torch.tensor(positive_features, dtype=torch.float32),
        torch.tensor(negative_features, dtype=torch.float32),
        torch.tensor(pair_weights, dtype=torch.float32),
    )


class BPRRanker(nn.Module):
    def __init__(self, item_count: int, embedding_dim: int = 16, feature_count: int = 13):
        super().__init__()
        self.item_embedding = nn.Embedding(item_count + 1, embedding_dim, padding_idx=0)
        self.correction = nn.Sequential(
            nn.Linear(embedding_dim * 4 + feature_count, 32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(
        self, candidate: torch.Tensor, history: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        candidate_embedding = self.item_embedding(candidate)
        history_embeddings = self.item_embedding(history)
        mask = (history != 0).unsqueeze(-1)
        history_embedding = (history_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        inputs = torch.cat(
            [
                candidate_embedding,
                history_embedding,
                candidate_embedding * history_embedding,
                torch.abs(candidate_embedding - history_embedding),
                features,
            ],
            dim=1,
        )
        prior_index = 8 if features.shape[1] > 8 else 0
        retrieval_prior = 2.0 * features[:, prior_index] + 0.02 * features[:, 2]
        return retrieval_prior + self.correction(inputs).squeeze(1)


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = nn.functional.softplus(-(positive_scores - negative_scores))
    return losses.mean() if weights is None else (losses * weights).sum() / weights.sum()


def score_group(
    model: BPRRanker,
    group: RollingGroup | CandidateGroup,
    mapping: dict[int, int],
    max_history: int,
) -> list[float]:
    history = pad_history(group.history, mapping, max_history)
    candidates = torch.tensor([mapping.get(item, 0) for item in group.candidates], dtype=torch.long)
    histories = torch.tensor([history] * len(group.candidates), dtype=torch.long)
    features = torch.tensor(group.features, dtype=torch.float32)
    return model(candidates, histories, features).tolist()


def ranking_metrics(ranked_labels: list[int], k: int) -> dict[str, float]:
    hits = ranked_labels[:k]
    hit_count = sum(hits)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    relevant_count = sum(ranked_labels)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(relevant_count, k)))
    return {
        "precision": hit_count / k,
        "recall": hit_count / relevant_count,
        "hit_rate": float(hit_count > 0),
        "ndcg": dcg / ideal,
    }


def evaluate(
    model: BPRRanker,
    groups: list[RollingGroup] | list[CandidateGroup],
    mapping: dict[int, int],
    max_history: int,
    k_values: list[int],
    require_retrieved: bool,
) -> dict[str, object]:
    totals = {
        model_name: {k: {metric: 0.0 for metric in ("precision", "recall", "hit_rate", "ndcg")}
        for k in k_values}
        for model_name in ("cooccurrence", "neural")
    }
    evaluated = 0
    model.eval()
    with torch.no_grad():
        for group in groups:
            if require_retrieved and isinstance(group, RollingGroup) and not group.retrieved_positive:
                continue
            if not any(group.labels):
                continue
            prior_index = 8 if len(group.features[0]) > 8 else 0
            baseline_order = sorted(
                range(len(group.candidates)),
                key=lambda index: (
                    -group.features[index][prior_index],
                    -group.features[index][2],
                    group.candidates[index],
                ),
            )
            scores = score_group(model, group, mapping, max_history)
            neural_order = sorted(
                range(len(group.candidates)),
                key=lambda index: (-scores[index], group.candidates[index]),
            )
            for name, order in (("cooccurrence", baseline_order), ("neural", neural_order)):
                labels = [group.labels[index] for index in order]
                for k in k_values:
                    values = ranking_metrics(labels, k)
                    for metric, value in values.items():
                        totals[name][k][metric] += value
            evaluated += 1
    rendered = {
        name: {
            f"@{k}": {
                metric: value / evaluated if evaluated else 0.0
                for metric, value in totals[name][k].items()
            }
            for k in k_values
        }
        for name in totals
    }
    return {"evaluated_groups": evaluated, "metrics": rendered}


def train_with_early_stopping(
    model: BPRRanker,
    dataset: TensorDataset,
    validation_groups: list[RollingGroup],
    mapping: dict[int, int],
    max_history: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    minimum_relative_lift: float,
) -> tuple[int, int, bool, list[dict[str, float | int | None]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    history = []
    initial = evaluate(model, validation_groups, mapping, max_history, [10], True)
    best_ndcg = initial["metrics"]["neural"]["@10"]["ndcg"]
    history.append({"epoch": 0, "train_bpr_loss": None, "validation_ndcg_at_10": best_ndcg})
    best_epoch = 0
    initial_state = copy.deepcopy(model.state_dict())
    best_state = copy.deepcopy(initial_state)
    initial_ndcg = best_ndcg
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        rows = 0
        for positive, negative, histories, positive_features, negative_features, weights in loader:
            optimizer.zero_grad()
            positive_scores = model(positive, histories, positive_features)
            negative_scores = model(negative, histories, negative_features)
            loss = bpr_loss(positive_scores, negative_scores, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(positive)
            rows += len(positive)
        validation = evaluate(model, validation_groups, mapping, max_history, [10], True)
        ndcg = validation["metrics"]["neural"]["@10"]["ndcg"]
        history.append(
            {"epoch": epoch, "train_bpr_loss": total_loss / rows, "validation_ndcg_at_10": ndcg}
        )
        if ndcg > best_ndcg + 1e-6:
            best_ndcg = ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    relative_lift = (best_ndcg / initial_ndcg - 1) if initial_ndcg else 0.0
    promoted = best_epoch > 0 and relative_lift >= minimum_relative_lift
    selected_epoch = best_epoch if promoted else 0
    model.load_state_dict(best_state if promoted else initial_state)
    return selected_epoch, best_epoch, promoted, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/processed/rolling_ranker_examples.csv.gz")
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/bpr_ranker.pt"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/bpr_ranker.json"))
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--minimum-relative-lift", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injected-fraction", type=float, default=0.10)
    parser.add_argument(
        "--categories", type=Path, default=Path("data/processed/item_categories.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    groups = load_rolling_groups(args.examples)
    training_groups = [group for group in groups if group.split == "train"]
    natural_training_groups = [group for group in training_groups if group.retrieved_positive]
    injected_training_groups = [
        group
        for group in training_groups
        if not group.retrieved_positive
        and (group.example_id * 2654435761 % 10_000) < int(args.injected_fraction * 10_000)
    ]
    selected_training_groups = natural_training_groups + injected_training_groups
    validation_groups = [group for group in groups if group.split == "validation"]
    train_path = args.processed_dir / "events_train.csv"
    validation_path = args.processed_dir / "events_validation.csv"
    test_path = args.processed_dir / "events_test.csv"
    mapping = build_item_mapping([train_path, validation_path])
    dataset = build_pair_dataset(selected_training_groups, mapping, args.max_history)
    model = BPRRanker(len(mapping), args.embedding_dim, len(FEATURE_COLUMNS))
    selected_epoch, best_candidate_epoch, promoted, epoch_history = train_with_early_stopping(
        model,
        dataset,
        validation_groups,
        mapping,
        args.max_history,
        args.max_epochs,
        args.patience,
        args.batch_size,
        args.learning_rate,
        args.seed,
        args.minimum_relative_lift,
    )
    validation_evaluation = evaluate(
        model, validation_groups, mapping, args.max_history, [5, 10, 20], True
    )
    categories = load_categories(args.categories)
    test_groups, test_stats = build_enriched_test_groups(
        [train_path, validation_path], test_path, categories, args.max_history, 100, 100
    )
    test_evaluation = evaluate(model, test_groups, mapping, args.max_history, [5, 10, 20], False)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "item_to_index": mapping,
            "config": {
                "embedding_dim": args.embedding_dim,
                "feature_count": len(FEATURE_COLUMNS),
                "max_history": args.max_history,
                "architecture": "residual_bpr_ranker",
            },
        },
        args.model,
    )
    report = {
        "model": "residual_embedding_ranker_with_bpr_loss",
        "training_groups": len(training_groups),
        "selected_natural_training_groups": len(natural_training_groups),
        "selected_injected_training_groups": len(injected_training_groups),
        "injected_pair_weight": 0.25,
        "training_pairs": len(dataset),
        "chronological_validation_groups": len(validation_groups),
        "retrieved_validation_groups": sum(group.retrieved_positive for group in validation_groups),
        "item_vocabulary_size": len(mapping),
        "best_candidate_epoch": best_candidate_epoch,
        "selected_epoch": selected_epoch,
        "candidate_promoted": promoted,
        "minimum_relative_validation_lift": args.minimum_relative_lift,
        "early_stopping_patience": args.patience,
        "epoch_history": epoch_history,
        "validation_evaluation": validation_evaluation,
        "test_candidate_stats": test_stats,
        "test_evaluation": test_evaluation,
        "model_artifact": str(args.model),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
