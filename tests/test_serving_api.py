import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ServingAPITests(unittest.TestCase):
    def test_recommendation_is_cached_and_versioned(self):
        try:
            from fastapi.testclient import TestClient
            from personalize_ai.api import create_app
        except ImportError as exc:
            self.skipTest(f"serving dependencies unavailable: {exc}")
        from personalize_ai.cache import MemoryTTLCache
        from personalize_ai.features import DictionaryFeatureStore

        store = DictionaryFeatureStore(
            "v-test",
            {10: {"recent_items": [{"item_id": 1}]}},
            {1: {"weighted_popularity": 10}, 2: {"weighted_popularity": 9}},
        )
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(store, MemoryTTLCache(), Path(directory))
            client = TestClient(app)
            first = client.get("/v1/recommendations/10?count=1")
            second = client.get("/v1/recommendations/10?count=1")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["feature_version"], "v-test")
        self.assertEqual(first.json()["cache"], "miss")
        self.assertEqual(second.json()["cache"], "hit")
        self.assertEqual(second.json()["recommendations"][0]["item_id"], 2)


if __name__ == "__main__":
    unittest.main()
