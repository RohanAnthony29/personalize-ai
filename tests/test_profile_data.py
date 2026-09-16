import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "profile_data.py"
SPEC = importlib.util.spec_from_file_location("profile_data", SCRIPT)
profile_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(profile_data)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class ProfileDataTests(unittest.TestCase):
    def test_profile_events_counts_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            write_csv(
                path,
                profile_data.EXPECTED_HEADERS["events.csv"],
                [
                    [1433221332117, 1, "view", 10, ""],
                    [1433224214164, 1, "addtocart", 10, ""],
                    [1433225214164, 2, "transaction", 11, 99],
                ],
            )

            result = profile_data.profile_events(path, max_rows=None)

        self.assertEqual(result["rows_read"], 3)
        self.assertEqual(
            result["event_counts"], {"addtocart": 1, "transaction": 1, "view": 1}
        )
        self.assertEqual(result["unique_visitors"], 2)
        self.assertEqual(result["unique_items"], 2)
        self.assertEqual(result["validation_errors"], {})

    def test_header_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, header in profile_data.EXPECTED_HEADERS.items():
                write_csv(path / name, header, [])
            write_csv(path / "events.csv", ["wrong"], [])

            with self.assertRaisesRegex(ValueError, "expected header"):
                profile_data.validate_headers(path)


if __name__ == "__main__":
    unittest.main()
