import csv
import json
from pathlib import Path
import tempfile
import unittest

from bias_dataset.build import build


class BuildTests(unittest.TestCase):
    def test_off_domain_and_short_records_are_rejected(self) -> None:
        base = {
            "record_id": "one",
            "source_id": "example",
            "source_name": "Example",
            "source_domain": "example.com",
            "headline": "Senate advances bipartisan election legislation",
            "feed_summary": "Summary",
            "feed_summary_truncated": False,
            "subheadline": "",
            "subheadline_provenance": "not_collected",
            "canonical_url": "https://example.com/story",
            "published_at": "2026-08-26T12:00:00Z",
            "observed_at": "2026-08-26T12:01:00Z",
            "headline_hash": "hash1",
            "text_hash": "text1",
            "weak_label": "Center",
            "label_scope": "outlet",
            "label_provider": "AllSides",
            "rating_scope": "online_written_news",
            "rating_url": "https://ratings/example",
            "rating_checked_at": "2026-08-27",
            "holdout_outlet": False,
        }
        off_domain = {**base, "record_id": "two", "canonical_url": "https://other.test/story"}
        short = {**base, "record_id": "three", "headline": "Too short"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in (base, off_domain, short)), encoding="utf-8")
            report = build(input_path, root / "output.csv", root / "report.json")
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(report["rejected"], {"off_domain_url": 1, "headline_too_short": 1})
            with (root / "output.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["label_scope"], "outlet")


if __name__ == "__main__":
    unittest.main()
