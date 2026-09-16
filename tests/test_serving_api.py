import sys
import tempfile
import unittest
from unittest.mock import patch
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
        self.assertEqual(first.json()["ranking_mode"], "champion")
        self.assertEqual(second.json()["cache"], "hit")
        self.assertEqual(second.json()["recommendations"][0]["item_id"], 2)

    def test_api_key_rate_limit_and_input_validation(self):
        try:
            from fastapi.testclient import TestClient
            from personalize_ai.api import create_app
        except ImportError as exc:
            self.skipTest(f"serving dependencies unavailable: {exc}")
        from personalize_ai.cache import MemoryTTLCache
        from personalize_ai.features import DictionaryFeatureStore

        store = DictionaryFeatureStore(
            "v-security",
            {10: {"recent_items": []}},
            {1: {"weighted_popularity": 1}},
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"API_KEY": "test-key", "RATE_LIMIT_REQUESTS_PER_MINUTE": "2"},
        ):
            client = TestClient(create_app(store, MemoryTTLCache(), Path(directory)))
            self.assertEqual(client.get("/v1/model").status_code, 401)
            headers = {"X-API-Key": "test-key"}
            self.assertEqual(client.get("/v1/model", headers=headers).status_code, 200)
            self.assertEqual(
                client.get("/v1/recommendations/0", headers=headers).status_code,
                422,
            )
            limited = client.get("/v1/model", headers=headers)
            self.assertEqual(limited.status_code, 429)
            self.assertEqual(limited.headers["Retry-After"], "60")


if __name__ == "__main__":
    unittest.main()
