import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "split_events.py"
SPEC = importlib.util.spec_from_file_location("split_events", SCRIPT)
split_events = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(split_events)


class SplitEventsTests(unittest.TestCase):
    def test_boundaries_do_not_overlap(self) -> None:
        train_end, validation_end = split_events.calculate_boundaries(0, 99, 0.70, 0.15)
        self.assertEqual(split_events.split_name(train_end - 1, train_end, validation_end), "train")
        self.assertEqual(split_events.split_name(train_end, train_end, validation_end), "validation")
        self.assertEqual(split_events.split_name(validation_end, train_end, validation_end), "test")

    def test_split_writes_normalized_rows_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "visitorid", "event", "itemid", "transactionid"])
                writer.writerows(
                    [
                        [0, 1, "view", 10, ""],
                        [69, 1, "addtocart", 10, ""],
                        [70, 2, "view", 20, ""],
                        [84, 2, "view", 21, ""],
                        [85, 3, "transaction", 30, 7],
                        [99, 3, "view", 31, ""],
                    ]
                )

            report = split_events.split_events(source, root / "processed")

            self.assertEqual(report["source_rows"], 6)
            self.assertEqual(report["splits"]["train"]["rows"], 2)
            self.assertEqual(report["splits"]["validation"]["rows"], 2)
            self.assertEqual(report["splits"]["test"]["rows"], 2)
            with (root / "processed/events_test.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["event_weight"], "5")
            self.assertEqual(rows[0]["event_type"], "transaction")


if __name__ == "__main__":
    unittest.main()
