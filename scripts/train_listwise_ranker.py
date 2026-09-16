#!/usr/bin/env python3
"""Train a feature-only hybrid ranker with full-list softmax loss."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_item_categories import load_categories
from train_bpr_ranker import (
    FEATURE_COLUMNS,
    build_enriched_test_groups,
    evaluate,
    load_rolling_groups,
)


SELECTED_FEATURES = list(range(9))  # Exclude source flags, which hurt in ablation.


class ListwiseFeatureRanker(nn.Module):
    def __init__(self, feature_indexes: list[int] = SELECTED_FEATURES):
        super().__init__()
        self.feature_indexes = feature_indexes
        self.correction = nn.Sequential(
            nn.Linear(len(feature_indexes), 16),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, candidate, history, features):
        del candidate, history
        prior = 2.0 * features[..., 8] + 0.02 * features[..., 2]
        return prior + self.correction(features[..., self.feature_indexes]).squeeze(-1)


def build_list_dataset(groups) -> TensorDataset:
    features = []
    targets = []
    for group in groups:
        if len(group.candidates) != 100:
            raise ValueError(f"Example {group.example_id} does not have 100 candidates")
        positives = [index for index, label in enumerate(group.labels) if label]
        if len(positives) != 1:
            raise ValueError(f"Example {group.example_id} must have exactly one positive")
        features.append(group.features)
        targets.append(positives[0])
    return TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )


def ndcg_at_10(model, groups) -> float:
    result = evaluate(model, groups, {}, 1, [10], False)
    return result["metrics"]["neural"]["@10"]["ndcg"]


def train(model, dataset, validation_groups, max_epochs, patience, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    initial_state = copy.deepcopy(model.state_dict())
    best_state = copy.deepcopy(initial_state)
    best_ndcg = ndcg_at_10(model, validation_groups)
    initial_ndcg = best_ndcg
    best_epoch = 0
    stale = 0
    history = [{"epoch": 0, "training_loss": None, "validation_ndcg_at_10": best_ndcg}]
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        rows = 0
        for features, targets in loader:
            optimizer.zero_grad()
            scores = model(None, None, features)
            loss = nn.functional.cross_entropy(scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            rows += len(targets)
        validation_ndcg = ndcg_at_10(model, validation_groups)
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / rows,
                "validation_ndcg_at_10": validation_ndcg,
            }
        )
        if validation_ndcg > best_ndcg + 1e-6:
            best_ndcg = validation_ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    relative_lift = best_ndcg / initial_ndcg - 1 if initial_ndcg else 0.0
    promoted = best_epoch > 0 and relative_lift >= 0.05
    model.load_state_dict(best_state if promoted else initial_state)
    return best_epoch if promoted else 0, best_epoch, promoted, history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/processed/listwise_temporal_folds.csv.gz")
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--categories", type=Path, default=Path("data/processed/item_categories.csv")
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/listwise_ranker.pt"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/listwise_ranker.json"))
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    groups = load_rolling_groups(args.examples)
    training_groups = [group for group in groups if group.split == "train"]
    validation_groups = [group for group in groups if group.split == "validation"]
    dataset = build_list_dataset(training_groups)
    model = ListwiseFeatureRanker()
    selected_epoch, best_epoch, promoted, history = train(
        model, dataset, validation_groups, args.max_epochs, args.patience, args.seed
    )
    validation_evaluation = evaluate(model, validation_groups, {}, 1, [5, 10, 20], False)

    train_path = args.processed_dir / "events_train.csv"
    validation_path = args.processed_dir / "events_validation.csv"
    test_path = args.processed_dir / "events_test.csv"
    test_groups, test_stats = build_enriched_test_groups(
        [train_path, validation_path],
        test_path,
        load_categories(args.categories),
        50,
        100,
        100,
    )
    test_evaluation = evaluate(model, test_groups, {}, 1, [5, 10, 20], False)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "selected_feature_indexes": SELECTED_FEATURES,
            "selected_epoch": selected_epoch,
            "promoted": promoted,
        },
        args.model,
    )
    report = {
        "model": "feature_only_listwise_softmax_ranker",
        "training_groups": len(training_groups),
        "validation_groups": len(validation_groups),
        "candidates_per_group": 100,
        "injected_positives": 0,
        "selected_features": [FEATURE_COLUMNS[index] for index in SELECTED_FEATURES],
        "best_candidate_epoch": best_epoch,
        "selected_epoch": selected_epoch,
        "promoted": promoted,
        "promotion_threshold_relative": 0.05,
        "epoch_history": history,
        "validation_evaluation": validation_evaluation,
        "test_candidate_stats": test_stats,
        "single_final_test_evaluation": test_evaluation,
        "model_artifact": str(args.model),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
