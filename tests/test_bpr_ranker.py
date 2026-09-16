import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import train_bpr_ranker as bpr


class BPRRankerTests(unittest.TestCase):
    def test_bpr_rewards_positive_score_margin(self) -> None:
        poor = bpr.bpr_loss(torch.tensor([0.0]), torch.tensor([1.0]))
        good = bpr.bpr_loss(torch.tensor([2.0]), torch.tensor([0.0]))
        self.assertLess(good.item(), poor.item())

    def test_model_starts_from_retrieval_prior(self) -> None:
        model = bpr.BPRRanker(item_count=10, embedding_dim=4, feature_count=13)
        features = torch.zeros((2, 13), dtype=torch.float32)
        features[0, 8] = 0.9
        features[0, 2] = 1.0
        features[1, 8] = 0.1
        features[1, 2] = 0.5
        scores = model(
            torch.tensor([1, 2]),
            torch.tensor([[3, 0], [3, 0]]),
            features,
        )
        self.assertGreater(scores[0].item(), scores[1].item())

    def test_single_positive_ndcg(self) -> None:
        top = bpr.ranking_metrics([1, 0, 0], 3)
        second = bpr.ranking_metrics([0, 1, 0], 3)
        self.assertEqual(top["ndcg"], 1.0)
        self.assertLess(second["ndcg"], top["ndcg"])

    def test_recall_handles_multiple_relevant_items(self) -> None:
        result = bpr.ranking_metrics([1, 0, 0, 1], 2)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
