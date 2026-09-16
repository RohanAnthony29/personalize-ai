import importlib.util
import csv
import gzip
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_item_cooccurrence.py"
SPEC = importlib.util.spec_from_file_location("evaluate_item_cooccurrence", SCRIPT)
cooccurrence = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cooccurrence)


class ItemCooccurrenceTests(unittest.TestCase):
    def test_neighbor_scores_use_cosine_normalization(self) -> None:
        histories = {
            1: [(10, 1, 3), (20, 1, 2)],
            2: [(10, 1, 2), (20, 1, 1)],
            3: [(10, 1, 1), (30, 1, 0)],
        }
        neighbors, pairs = cooccurrence.build_neighbors(histories, 10)
        self.assertEqual(pairs, 3)
        self.assertEqual(neighbors[10][0][0], 20)
        self.assertAlmostEqual(neighbors[10][0][1], 2 / (3 * 2) ** 0.5)

    def test_candidates_filter_seen_and_backfill_popularity(self) -> None:
        history = [(10, 1, 1)]
        neighbors = {10: [(20, 0.8), (30, 0.4)]}
        recommendations, generated = cooccurrence.cooccurrence_recommendations(
            history, {10}, neighbors, [10, 40, 50], 4
        )
        self.assertEqual(recommendations, [20, 30, 40, 50])
        self.assertEqual(generated, 2)

    def test_metrics(self) -> None:
        result = cooccurrence.metrics([1, 2, 3], {2}, 3)
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["hit_rate"], 1.0)

    def test_neighbor_artifact_is_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "neighbors.csv.gz"
            rows = cooccurrence.write_neighbor_artifact(
                {10: [(20, 0.8), (30, 0.4)]}, output
            )
            with gzip.open(output, "rt", newline="", encoding="utf-8") as handle:
                values = list(csv.DictReader(handle))
        self.assertEqual(rows, 2)
        self.assertEqual(values[0]["neighbor_item_id"], "20")
        self.assertEqual(values[1]["rank"], "2")


if __name__ == "__main__":
    unittest.main()
