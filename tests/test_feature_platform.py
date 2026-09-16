import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from personalize_ai.cache import MemoryTTLCache
from personalize_ai.features import DictionaryFeatureStore
from personalize_ai.recommender import HybridRecommender
from spark_jobs.contracts import IncrementalState, render_sql, validate_manifest


class FeatureContractTests(unittest.TestCase):
    def test_state_round_trip_and_atomic_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            IncrementalState(123, "v1").dump(path)
            self.assertEqual(IncrementalState.load(path), IncrementalState(123, "v1"))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_incremental_manifest_requires_parent(self):
        manifest = {
            "feature_version": "v2",
            "watermark_ms": 2,
            "row_counts": {"user": 1},
            "format": "parquet",
            "snapshot_type": "delta",
            "incremental": True,
            "parent_version": None,
        }
        self.assertIn("incremental snapshot must identify parent_version", validate_manifest(manifest))

    def test_sql_templates_are_rendered_without_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.sql"
            path.write_text("SELECT * FROM {{source_view}} WHERE timestamp <= {{as_of_ms}}")
            self.assertEqual(render_sql(path, "events", 42), "SELECT * FROM events WHERE timestamp <= 42")


class ServingTests(unittest.TestCase):
    def test_memory_cache(self):
        cache = MemoryTTLCache()
        cache.set("key", {"ok": True}, 10)
        self.assertEqual(cache.get("key"), {"ok": True})
        self.assertIsNone(cache.get("missing"))

    def test_hybrid_recommender_uses_versioned_features_and_filters_seen(self):
        users = {7: {"recent_items": [{"item_id": 1}, {"item_id": 2}]}}
        items = {
            1: {"weighted_popularity": 10},
            2: {"weighted_popularity": 9},
            3: {"weighted_popularity": 8},
            4: {"weighted_popularity": 7},
        }
        store = DictionaryFeatureStore("20260916T120000Z", users, items)
        with tempfile.TemporaryDirectory() as directory:
            recommender = HybridRecommender(store, Path(directory))
            values, version = recommender.recommend(7, 2)
        self.assertEqual(version, "20260916T120000Z")
        self.assertEqual([row["item_id"] for row in values], [3, 4])
        self.assertTrue(all(row["item_id"] not in {1, 2} for row in values))


if __name__ == "__main__":
    unittest.main()
