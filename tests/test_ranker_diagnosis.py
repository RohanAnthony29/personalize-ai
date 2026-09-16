import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import diagnose_ranker as diagnosis


class RankerDiagnosisTests(unittest.TestCase):
    def test_feature_only_model_ignores_identifiers(self) -> None:
        model = diagnosis.FeatureOnlyRanker(list(range(13)), "linear")
        features = torch.zeros((2, 13))
        features[:, 8] = torch.tensor([0.8, 0.2])
        first = model(torch.tensor([1, 2]), torch.tensor([[3], [4]]), features)
        second = model(torch.tensor([9, 8]), torch.tensor([[7], [6]]), features)
        self.assertTrue(torch.equal(first, second))

    def test_ablation_indexes_are_valid(self) -> None:
        for _, indexes in diagnosis.variant_definitions().values():
            self.assertTrue(indexes)
            self.assertTrue(all(0 <= index < 13 for index in indexes))


if __name__ == "__main__":
    unittest.main()
