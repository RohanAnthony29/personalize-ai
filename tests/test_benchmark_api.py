import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_api import confidence_interval_95, percentile, summarize


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_and_summary(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(percentile(values, 0.50), 3.0)
        result = summarize(values, 0.5)
        self.assertEqual(result["requests"], 5)
        self.assertEqual(result["throughput_requests_per_second"], 10.0)

    def test_confidence_interval_contains_mean(self):
        values = [10.0, 11.0, 9.0, 10.5, 9.5]
        lower, upper = confidence_interval_95(values)
        self.assertLess(lower, 10.0)
        self.assertGreater(upper, 10.0)


if __name__ == "__main__":
    unittest.main()
