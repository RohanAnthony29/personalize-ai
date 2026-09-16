#!/usr/bin/env python3
"""Diagnose ranker overfitting with feature-only models and ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from build_item_categories import load_categories
from train_bpr_ranker import (
    FEATURE_COLUMNS,
    build_enriched_test_groups,
    build_pair_dataset,
    evaluate,
    load_rolling_groups,
    train_with_early_stopping,
)


class FeatureOnlyRanker(nn.Module):
    def __init__(self, feature_indexes: list[int], architecture: str):
        super().__init__()
        self.feature_indexes = feature_indexes
        if architecture == "linear":
            self.correction = nn.Linear(len(feature_indexes), 1)
        elif architecture == "mlp":
            self.correction = nn.Sequential(
                nn.Linear(len(feature_indexes), 16),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(16, 1),
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        final_layer = self.correction if architecture == "linear" else self.correction[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self, candidate: torch.Tensor, history: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        del candidate, history
        retrieval_prior = 2.0 * features[:, 8] + 0.02 * features[:, 2]
        selected = features[:, self.feature_indexes]
        return retrieval_prior + self.correction(selected).squeeze(1)


def variant_definitions() -> dict[str, tuple[str, list[int]]]:
    all_indexes = list(range(len(FEATURE_COLUMNS)))
    return {
        "linear_all_features": ("linear", all_indexes),
        "mlp_all_features": ("mlp", all_indexes),
        "mlp_without_category": ("mlp", [i for i in all_indexes if i != 7]),
        "mlp_without_recency": ("mlp", [i for i in all_indexes if i != 5]),
        "mlp_without_max_similarity": ("mlp", [i for i in all_indexes if i != 6]),
        "mlp_without_source_flags": ("mlp", [i for i in all_indexes if i not in {9, 10, 11, 12}]),
        "mlp_retrieval_core": ("mlp", [0, 1, 2, 3, 4, 8]),
        "mlp_hybrid_features": ("mlp", [8, 9, 10, 11, 12]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/processed/rolling_ranker_examples.csv.gz")
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--categories", type=Path, default=Path("data/processed/item_categories.csv")
    )
    parser.add_argument("--report", type=Path, default=Path("data/reports/ranker_diagnosis.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/feature_ranker.pt"))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = load_rolling_groups(args.examples)
    all_training_groups = [group for group in groups if group.split == "train"]
    natural_groups = [group for group in all_training_groups if group.retrieved_positive]
    injected_groups = [
        group
        for group in all_training_groups
        if not group.retrieved_positive
        and (group.example_id * 2654435761 % 10_000) < 1000
    ]
    training_groups = natural_groups + injected_groups
    validation_groups = [group for group in groups if group.split == "validation"]
    empty_mapping: dict[int, int] = {}
    dataset = build_pair_dataset(training_groups, empty_mapping, 50)

    results = {}
    best_name = None
    best_ndcg = -1.0
    best_model = None
    for offset, (name, (architecture, indexes)) in enumerate(variant_definitions().items()):
        torch.manual_seed(args.seed + offset)
        model = FeatureOnlyRanker(indexes, architecture)
        selected_epoch, best_epoch, _, epoch_history = train_with_early_stopping(
            model,
            dataset,
            validation_groups,
            empty_mapping,
            50,
            args.max_epochs,
            args.patience,
            2048,
            0.001,
            args.seed + offset,
            0.0,
        )
        validation = evaluate(model, validation_groups, empty_mapping, 50, [10], True)
        ndcg = validation["metrics"]["neural"]["@10"]["ndcg"]
        baseline = validation["metrics"]["cooccurrence"]["@10"]["ndcg"]
        results[name] = {
            "architecture": architecture,
            "features": [FEATURE_COLUMNS[index] for index in indexes],
            "selected_epoch": selected_epoch,
            "best_candidate_epoch": best_epoch,
            "validation_ndcg_at_10": ndcg,
            "baseline_ndcg_at_10": baseline,
            "relative_lift_pct": (ndcg / baseline - 1) * 100 if baseline else None,
            "epoch_history": epoch_history,
        }
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_name = name
            best_model = model

    assert best_name is not None and best_model is not None
    baseline_ndcg = results[best_name]["baseline_ndcg_at_10"]
    promoted = best_ndcg >= baseline_ndcg * 1.05
    train = args.processed_dir / "events_train.csv"
    validation = args.processed_dir / "events_validation.csv"
    test = args.processed_dir / "events_test.csv"
    categories = load_categories(args.categories)
    test_groups, test_stats = build_enriched_test_groups(
        [train, validation], test, categories, 50, 100, 100
    )
    test_evaluation = evaluate(
        best_model, test_groups, empty_mapping, 50, [5, 10, 20], False
    )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    best_architecture, best_indexes = variant_definitions()[best_name]
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "selected_feature_indexes": best_indexes,
            "architecture": best_architecture,
            "promoted": promoted,
        },
        args.model,
    )
    report = {
        "experiment": "feature_only_bpr_ablation",
        "training_groups": len(training_groups),
        "natural_training_groups": len(natural_groups),
        "injected_training_groups": len(injected_groups),
        "training_pairs": len(dataset),
        "validation_groups": sum(group.retrieved_positive for group in validation_groups),
        "variants": results,
        "best_validation_variant": best_name,
        "best_validation_ndcg_at_10": best_ndcg,
        "promotion_threshold_relative": 0.05,
        "promoted": promoted,
        "test_candidate_stats": test_stats,
        "single_final_test_evaluation": test_evaluation,
        "model_artifact": str(args.model),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
