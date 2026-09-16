#!/usr/bin/env python3
"""Create point-in-time candidate examples and train a PyTorch neural ranker."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_item_cooccurrence import (
    build_neighbors,
    load_history,
    load_warm_ground_truth,
    metrics,
)


@dataclass
class CandidateGroup:
    user_id: int
    history: list[int]
    candidates: list[int]
    features: list[list[float]]
    labels: list[int]


def generate_candidates(
    history: list[tuple[int, int, int]],
    full_seen: set[int],
    neighbors: dict[int, list[tuple[int, float]]],
    popularity: Counter[int],
    candidate_count: int,
) -> tuple[list[int], list[list[float]]]:
    scores: Counter[int] = Counter()
    contributing_sources: Counter[int] = Counter()
    for source_item, source_weight, _ in history:
        source_strength = 1.0 + math.log1p(source_weight)
        for candidate, similarity in neighbors.get(source_item, []):
            if candidate not in full_seen:
                scores[candidate] += source_strength * similarity
                contributing_sources[candidate] += 1

    ranked_cooccurrence = sorted(scores, key=lambda item: (-scores[item], item))
    ranked_popularity = [
        item for item, _ in sorted(popularity.items(), key=lambda value: (-value[1], value[0]))
    ]
    candidates = ranked_cooccurrence[:candidate_count]
    selected = set(candidates)
    for item in ranked_popularity:
        if len(candidates) == candidate_count:
            break
        if item not in full_seen and item not in selected:
            candidates.append(item)
            selected.add(item)

    max_score = max(scores.values(), default=1.0)
    max_popularity = max(popularity.values(), default=1)
    features = []
    for rank, item in enumerate(candidates, start=1):
        features.append(
            [
                scores[item] / max_score,
                math.log1p(popularity[item]) / math.log1p(max_popularity),
                1.0 / rank,
                contributing_sources[item] / max(len(history), 1),
                math.log1p(len(full_seen)) / math.log1p(50),
            ]
        )
    return candidates, features


def build_groups(
    history_paths: list[Path],
    target_path: Path,
    max_history: int,
    neighbors_per_item: int,
    candidate_count: int,
) -> tuple[list[CandidateGroup], dict[str, int], dict[int, list[tuple[int, float]]]]:
    histories, full_seen, popularity, _, _ = load_history(history_paths, max_history)
    neighbors, _ = build_neighbors(histories, neighbors_per_item)
    truth, truth_stats = load_warm_ground_truth(target_path, full_seen)
    groups = []
    groups_with_positive = 0
    positives = 0
    for user_id, relevant in truth.items():
        candidates, features = generate_candidates(
            histories[user_id], full_seen[user_id], neighbors, popularity, candidate_count
        )
        labels = [int(item in relevant) for item in candidates]
        positives += sum(labels)
        groups_with_positive += int(any(labels))
        groups.append(
            CandidateGroup(
                user_id=user_id,
                history=[item for item, _, _ in histories[user_id]],
                candidates=candidates,
                features=features,
                labels=labels,
            )
        )
    stats = {
        **truth_stats,
        "candidate_groups": len(groups),
        "groups_with_retrieved_positive": groups_with_positive,
        "retrieved_positive_rows": positives,
        "candidate_rows": sum(len(group.candidates) for group in groups),
    }
    return groups, stats, neighbors


def write_examples(groups: list[CandidateGroup], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "user_id",
                "candidate_item_id",
                "label",
                "history_item_ids",
                "cooccurrence_score",
                "popularity_score",
                "reciprocal_candidate_rank",
                "source_coverage",
                "normalized_history_length",
            ]
        )
        for group in groups:
            history = " ".join(map(str, group.history))
            for item, label, features in zip(group.candidates, group.labels, group.features):
                writer.writerow([group.user_id, item, label, history, *[f"{x:.8f}" for x in features]])
                rows += 1
    return rows


def build_item_mapping(*group_sets: list[CandidateGroup]) -> dict[int, int]:
    item_ids = set()
    for groups in group_sets:
        for group in groups:
            item_ids.update(group.history)
            item_ids.update(group.candidates)
    return {item: index for index, item in enumerate(sorted(item_ids), start=1)}


class CandidateDataset(Dataset):
    def __init__(self, groups: list[CandidateGroup], item_to_index: dict[int, int], max_history: int):
        self.rows = []
        for group_index, group in enumerate(groups):
            history = [item_to_index.get(item, 0) for item in group.history[:max_history]]
            history += [0] * (max_history - len(history))
            for item, feature, label in zip(group.candidates, group.features, group.labels):
                self.rows.append(
                    (item_to_index.get(item, 0), history, feature, float(label), group_index)
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        item, history, features, label, group = self.rows[index]
        return (
            torch.tensor(item, dtype=torch.long),
            torch.tensor(history, dtype=torch.long),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(group, dtype=torch.long),
        )


class NeuralRanker(nn.Module):
    def __init__(self, item_count: int, embedding_dim: int, feature_count: int):
        super().__init__()
        self.item_embedding = nn.Embedding(item_count + 1, embedding_dim, padding_idx=0)
        input_size = embedding_dim * 4 + feature_count
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, candidate: torch.Tensor, history: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        candidate_embedding = self.item_embedding(candidate)
        history_embeddings = self.item_embedding(history)
        mask = (history != 0).unsqueeze(-1)
        history_embedding = (history_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        combined = torch.cat(
            [
                candidate_embedding,
                history_embedding,
                candidate_embedding * history_embedding,
                torch.abs(candidate_embedding - history_embedding),
                features,
            ],
            dim=1,
        )
        # Preserve the strong retrieval prior and learn only a bounded correction.
        # This is important when supervised positives are sparse.
        retrieval_prior = 8.0 * (features[:, 0] + 0.01 * features[:, 2]) - 4.0
        neural_adjustment = 0.25 * torch.tanh(self.network(combined).squeeze(1))
        return retrieval_prior + neural_adjustment


def train_model(
    model: NeuralRanker,
    dataset: CandidateDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    torch.manual_seed(seed)
    positives = sum(row[3] for row in dataset.rows)
    negatives = len(dataset) - positives
    positive_weight = torch.tensor(negatives / max(positives, 1))
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    losses = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_rows = 0
        for candidate, history, features, labels, _ in loader:
            optimizer.zero_grad()
            logits = model(candidate, history, features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_rows += len(labels)
        losses.append(total_loss / total_rows)
    return losses


def evaluate_groups(
    model: NeuralRanker,
    groups: list[CandidateGroup],
    item_to_index: dict[int, int],
    max_history: int,
    k_values: list[int],
) -> dict[str, object]:
    totals = {name: {k: Counter() for k in k_values} for name in ("cooccurrence", "neural")}
    model.eval()
    with torch.no_grad():
        for group in groups:
            relevant = {
                item for item, label in zip(group.candidates, group.labels) if label
            }
            if not relevant:
                continue
            history = [item_to_index.get(item, 0) for item in group.history[:max_history]]
            history += [0] * (max_history - len(history))
            candidates = torch.tensor(
                [item_to_index.get(item, 0) for item in group.candidates], dtype=torch.long
            )
            histories = torch.tensor([history] * len(group.candidates), dtype=torch.long)
            features = torch.tensor(group.features, dtype=torch.float32)
            scores = model(candidates, histories, features).tolist()
            neural_ranking = [
                item
                for item, _ in sorted(
                    zip(group.candidates, scores), key=lambda value: (-value[1], value[0])
                )
            ]
            for k in k_values:
                totals["cooccurrence"][k].update(metrics(group.candidates, relevant, k))
                totals["neural"][k].update(metrics(neural_ranking, relevant, k))

    evaluated_users = sum(any(group.labels) for group in groups)
    rendered = {
        name: {
            f"@{k}": {
                metric: value / evaluated_users if evaluated_users else 0.0
                for metric, value in totals[name][k].items()
            }
            for k in k_values
        }
        for name in totals
    }
    return {"evaluated_users_with_retrieved_positive": evaluated_users, "metrics": rendered}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--examples", type=Path, default=Path("data/processed/ranker_examples.csv.gz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/neural_ranker.pt"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/neural_ranker.json"))
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--neighbors-per-item", type=int, default=100)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = args.processed_dir / "events_train.csv"
    validation = args.processed_dir / "events_validation.csv"
    test = args.processed_dir / "events_test.csv"
    training_groups, training_stats, _ = build_groups(
        [train], validation, args.max_history, args.neighbors_per_item, args.candidate_count
    )
    test_groups, test_stats, _ = build_groups(
        [train, validation], test, args.max_history, args.neighbors_per_item, args.candidate_count
    )
    example_rows = write_examples(training_groups, args.examples)
    item_to_index = build_item_mapping(training_groups, test_groups)
    dataset = CandidateDataset(training_groups, item_to_index, args.max_history)
    model = NeuralRanker(len(item_to_index), args.embedding_dim, 5)
    losses = train_model(
        model, dataset, args.epochs, args.batch_size, args.learning_rate, args.seed
    )
    evaluation = evaluate_groups(
        model, test_groups, item_to_index, args.max_history, sorted(set(args.k))
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "item_to_index": item_to_index,
            "config": {
                "embedding_dim": args.embedding_dim,
                "feature_count": 5,
                "max_history": args.max_history,
                "candidate_count": args.candidate_count,
            },
        },
        args.model,
    )
    report = {
        "model": "embedding_history_mlp_ranker",
        "training_protocol": "train history -> validation outcomes",
        "test_protocol": "train + validation history -> test outcomes",
        "training_examples": str(args.examples),
        "model_artifact": str(args.model),
        "training_example_rows": example_rows,
        "training_stats": training_stats,
        "test_stats": test_stats,
        "item_vocabulary_size": len(item_to_index),
        "epoch_losses": losses,
        "test_evaluation": evaluation,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
