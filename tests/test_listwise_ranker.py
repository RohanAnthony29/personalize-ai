import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import train_listwise_ranker as listwise


class ListwiseRankerTests(unittest.TestCase):
    def test_model_scores_complete_lists(self) -> None:
        model = listwise.ListwiseFeatureRanker()
        features = torch.zeros((2, 100, 13))
        scores = model(None, None, features)
        self.assertEqual(tuple(scores.shape), (2, 100))

    def test_softmax_loss_rewards_correct_top_item(self) -> None:
        target = torch.tensor([0])
        good = torch.nn.functional.cross_entropy(torch.tensor([[3.0, 0.0]]), target)
        bad = torch.nn.functional.cross_entropy(torch.tensor([[0.0, 3.0]]), target)
        self.assertLess(good.item(), bad.item())


if __name__ == "__main__":
    unittest.main()
