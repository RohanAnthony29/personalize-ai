from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .features import FeatureStore


def _load_ranked(path: Path, source: str, target: str, score: str) -> dict[int, list[tuple[int, float]]]:
    result: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return result
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[int(row[source])].append((int(row[target]), float(row[score])))
    return dict(result)


class HybridRecommender:
    def __init__(self, feature_store: FeatureStore, artifact_dir: Path) -> None:
        self.features = feature_store
        self.item_neighbors = _load_ranked(
            artifact_dir / "item_neighbors.csv.gz", "item_id", "neighbor_item_id", "similarity"
        )
        self.session_neighbors = _load_ranked(
            artifact_dir / "session_neighbors.csv.gz", "item_id", "next_item_id", "transition_score"
        )

    @staticmethod
    def _history(user: dict | None) -> list[int]:
        if not user:
            return []
        values = user.get("recent_items", [])
        result = []
        for value in values:
            item = value.get("item_id") if isinstance(value, dict) else value
            if item is not None and int(item) not in result:
                result.append(int(item))
        return result

    @staticmethod
    def _rrf(rankings: Iterable[tuple[float, list[int]]], count: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for weight, ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                scores[item] += weight / (60 + rank)
        return sorted(scores.items(), key=lambda value: (-value[1], value[0]))[:count]

    def recommend(self, user_id: int, count: int = 10) -> tuple[list[dict], str]:
        version = self.features.active_version()
        history = self._history(self.features.user(user_id))
        seen = set(history)
        item_ranking, session_ranking = [], []
        for source in history[:50]:
            item_ranking.extend(item for item, _ in self.item_neighbors.get(source, []))
        for source in history[:10]:
            session_ranking.extend(item for item, _ in self.session_neighbors.get(source, []))

        # Item features are the authoritative popularity fallback from the active snapshot.
        popularity_ranking = self.features.popular_items(max(100, count + len(seen)))
        rankings = [(1.0, item_ranking), (1.5, session_ranking), (0.25, popularity_ranking)]
        fused = [(item, score) for item, score in self._rrf(rankings, count + len(seen)) if item not in seen][:count]
        item_features = self.features.items([item for item, _ in fused])
        recommendations = [
            {"item_id": item, "score": score, "features": item_features.get(item, {})}
            for item, score in fused
        ]
        return recommendations, version
