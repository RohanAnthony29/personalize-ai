import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import generate_rolling_examples as rolling


class RollingExamplesTests(unittest.TestCase):
    def test_time_boundary(self) -> None:
        self.assertEqual(rolling.time_boundary(0, 99, 0.70), 70)

    def test_candidate_pool_filters_seen_items(self) -> None:
        candidates, features = rolling.candidate_pool(
            history=[(10, 1, 1)],
            full_seen={10},
            neighbors={10: [(20, 0.8), (30, 0.4)]},
            popularity=Counter({10: 100, 40: 50, 20: 2, 30: 1}),
            ranked_popularity=[10, 40, 20, 30],
            pool_size=3,
            prediction_timestamp=1000,
            categories={20: 1, 30: 2, 40: 1},
        )
        self.assertEqual(candidates, [20, 30, 40])
        self.assertNotIn(10, features)
        self.assertEqual(len(features[20]), 8)

    def test_state_uses_only_events_processed_so_far(self) -> None:
        state = {10: (1, 100)}
        rolling.update_state(state, 20, 3, 200)
        self.assertEqual(state[20], (3, 200))
        rolling.update_state(state, 20, 5, 300)
        self.assertEqual(state[20], (8, 300))


if __name__ == "__main__":
    unittest.main()
