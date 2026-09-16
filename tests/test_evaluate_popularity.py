import importlib.util
import math
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_popularity.py"
SPEC = importlib.util.spec_from_file_location("evaluate_popularity", SCRIPT)
baseline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(baseline)


class PopularityBaselineTests(unittest.TestCase):
    def test_ranking_is_deterministic_for_ties(self) -> None:
        popularity = Counter({20: 5, 10: 5, 30: 2})
        self.assertEqual(baseline.rank_items(popularity), [10, 20, 30])

    def test_recommendation_filters_seen_items(self) -> None:
        result = baseline.recommend([1, 2, 3, 4], {1, 3}, 2)
        self.assertEqual(result, [2, 4])

    def test_metrics_at_k(self) -> None:
        result = baseline.user_metrics([10, 20, 30], {20, 30}, 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["hit_rate"], 1.0)
        expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
        expected_idcg = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(result["ndcg"], expected_dcg / expected_idcg)


if __name__ == "__main__":
    unittest.main()
