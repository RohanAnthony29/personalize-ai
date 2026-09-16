from __future__ import annotations

import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Protocol

from .features import FeatureStore


class Ranker(Protocol):
    model_version: str
    def score(self, candidates: list[int], history: list[int], features: list[list[float]]) -> list[float]: ...


def _load_ranked(path: Path, source: str, target: str, score: str) -> dict[int, list[tuple[int, float]]]:
    result: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return result
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[int(row[source])].append((int(row[target]), float(row[score])))
    return dict(result)


class HybridRecommender:
    def __init__(
        self,
        feature_store: FeatureStore,
        artifact_dir: Path,
        challenger: Ranker | None = None,
        ranking_mode: str = "champion",
    ) -> None:
        self.features = feature_store
        self.challenger = challenger
        self.ranking_mode = ranking_mode
        self.item_neighbors = _load_ranked(
            artifact_dir / "item_neighbors.csv.gz", "item_id", "neighbor_item_id", "similarity"
        )
        self.session_neighbors = _load_ranked(
            artifact_dir / "session_neighbors.csv.gz", "item_id", "next_item_id", "transition_score"
        )

    @property
    def cache_namespace(self) -> str:
        model = self.challenger.model_version if self.challenger else "none"
        return f"{self.ranking_mode}:{model}"

    @staticmethod
    def _history(user: dict | None) -> list[dict]:
        if not user:
            return []
        values = user.get("recent_items", [])
        result: list[dict] = []
        seen = set()
        for value in values:
            item = value.get("item_id") if isinstance(value, dict) else value
            if item is not None and int(item) not in seen:
                result.append(value if isinstance(value, dict) else {"item_id": int(item)})
                seen.add(int(item))
        return result

    @staticmethod
    def _rrf(rankings: Iterable[tuple[float, list[int]]], count: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for weight, ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                scores[item] += weight / (60 + rank)
        return sorted(scores.items(), key=lambda value: (-value[1], value[0]))[:count]

    def recommend(self, user_id: int, count: int = 10) -> tuple[list[dict], str, dict]:
        version = self.features.active_version()
        history_values = self._history(self.features.user(user_id))
        history = [int(value["item_id"]) for value in history_values]
        seen = set(history)
        item_scores: dict[int, float] = defaultdict(float)
        session_scores: dict[int, float] = defaultdict(float)
        max_similarity: dict[int, float] = defaultdict(float)
        for position, source in enumerate(history_values[:50]):
            strength = (1 + math.log1p(float(source.get("event_weight", 1)))) / (position + 1)
            for item, similarity in self.item_neighbors.get(int(source["item_id"]), []):
                if item not in seen:
                    item_scores[item] += strength * similarity
                    max_similarity[item] = max(max_similarity[item], similarity)
        for position, source in enumerate(history_values[:10]):
            strength = (1 + math.log1p(float(source.get("event_weight", 1)))) / (position + 1)
            for item, similarity in self.session_neighbors.get(int(source["item_id"]), []):
                if item not in seen:
                    session_scores[item] += strength * similarity
                    max_similarity[item] = max(max_similarity[item], similarity)
        item_ranking = sorted(item_scores, key=lambda item: (-item_scores[item], item))[:100]
        session_ranking = sorted(session_scores, key=lambda item: (-session_scores[item], item))[:100]

        # Item features are the authoritative popularity fallback from the active snapshot.
        popularity_ranking = self.features.popular_items(max(100, count + len(seen)))
        weighted_rankings = [(1.0, item_ranking), (1.5, session_ranking), (0.25, popularity_ranking)]
        fused = [(item, score) for item, score in self._rrf(weighted_rankings, max(100, count)) if item not in seen]
        candidate_ids = [item for item, _ in fused]
        item_features = self.features.items(candidate_ids)
        max_item = max(item_scores.values(), default=1.0)
        max_session = max(session_scores.values(), default=1.0)
        max_popularity = max(
            (float(value.get("weighted_popularity", 0)) for value in item_features.values()),
            default=1.0,
        ) or 1.0
        max_rrf = max((score for _, score in fused), default=1.0) or 1.0
        source_sets = [set(item_ranking), set(session_ranking), set(), set(popularity_ranking)]
        model_features = []
        for rank, (item, rrf_score) in enumerate(fused, start=1):
            source_flags = [float(item in values) for values in source_sets]
            model_features.append([
                item_scores.get(item, 0.0) / max_item,
                float(item_features.get(item, {}).get("weighted_popularity", 0)) / max_popularity,
                1.0 / rank,
                sum(source_flags) / 4.0,
                min(len(history) / 50.0, 1.0),
                session_scores.get(item, 0.0) / max_session,
                max_similarity.get(item, 0.0),
                0.0,
                rrf_score / max_rrf,
                *source_flags,
            ])
        challenger_scores = (
            self.challenger.score(candidate_ids, history, model_features)
            if self.challenger and candidate_ids
            else [None] * len(candidate_ids)
        )
        rows = [
            {
                "item_id": item,
                "score": rrf_score,
                "challenger_score": challenger_score,
                "features": item_features.get(item, {}),
            }
            for (item, rrf_score), challenger_score in zip(fused, challenger_scores)
        ]
        if self.ranking_mode == "challenger" and self.challenger:
            rows.sort(key=lambda row: (-float(row["challenger_score"]), row["item_id"]))
        metadata = {
            "ranking_mode": self.ranking_mode if self.challenger else "champion",
            "challenger_loaded": self.challenger is not None,
            "model_version": self.challenger.model_version if self.challenger else None,
        }
        return rows[:count], version, metadata
