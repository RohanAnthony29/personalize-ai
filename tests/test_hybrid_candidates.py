import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import evaluate_hybrid_candidates as hybrid


class HybridCandidateTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_combines_sources(self) -> None:
        rankings = {
            "item": [1, 2, 3],
            "session": [2, 4, 1],
        }
        fused, contribution = hybrid.reciprocal_rank_fusion(
            rankings, {"item": 1.0, "session": 2.0}, 3, rank_constant=1
        )
        self.assertEqual(fused[0], 2)
        self.assertEqual(contribution["session"], 3)

    def test_candidate_metrics_support_multiple_targets(self) -> None:
        result = hybrid.candidate_metrics([1, 2, 3], {2, 4}, 3)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["coverage"], 1.0)

    def test_popularity_source_stops_at_limit(self) -> None:
        self.assertEqual(
            hybrid.limited_unseen_items([1, 2, 3, 4, 5], {1, 3}, 2),
            [2, 4],
        )


if __name__ == "__main__":
    unittest.main()
