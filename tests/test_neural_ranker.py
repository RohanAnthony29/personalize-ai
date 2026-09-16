import sys
import unittest
from collections import Counter
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import train_neural_ranker as ranker


class NeuralRankerTests(unittest.TestCase):
    def test_candidate_features_and_seen_filtering(self) -> None:
        candidates, features = ranker.generate_candidates(
            history=[(10, 1, 1)],
            full_seen={10},
            neighbors={10: [(20, 0.8), (30, 0.4)]},
            popularity=Counter({10: 100, 40: 50, 20: 2, 30: 1}),
            candidate_count=3,
        )
        self.assertEqual(candidates, [20, 30, 40])
        self.assertEqual(len(features), 3)
        self.assertEqual(len(features[0]), 5)

    def test_model_output_shape(self) -> None:
        model = ranker.NeuralRanker(item_count=10, embedding_dim=4, feature_count=5)
        output = model(
            torch.tensor([1, 2]),
            torch.tensor([[3, 4, 0], [5, 0, 0]]),
            torch.zeros((2, 5)),
        )
        self.assertEqual(tuple(output.shape), (2,))

    def test_ranker_preserves_strong_retrieval_prior_at_initialization(self) -> None:
        torch.manual_seed(1)
        model = ranker.NeuralRanker(item_count=10, embedding_dim=4, feature_count=5)
        features = torch.tensor([[0.9, 0, 1, 0, 0], [0.1, 0, 0.5, 0, 0]], dtype=torch.float32)
        scores = model(
            torch.tensor([1, 2]),
            torch.tensor([[3, 0], [3, 0]]),
            features,
        )
        self.assertGreater(scores[0].item(), scores[1].item())


if __name__ == "__main__":
    unittest.main()
