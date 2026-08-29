from pathlib import Path
import tempfile
import unittest

from bias_dataset.registry import load_registry


HEADER = "source_id,source_name,domain,weak_label,rating_scope,rating_url,rating_checked_at,feed_url,feed_kind,enabled,holdout\n"


class RegistryTests(unittest.TestCase):
    def test_project_registry_has_minimum_source_diversity(self) -> None:
        outlets = load_registry(Path("config/outlets.csv"))
        enabled = [outlet for outlet in outlets if outlet.enabled]
        counts: dict[str, int] = {}
        for outlet in enabled:
            counts[outlet.weak_label] = counts.get(outlet.weak_label, 0) + 1
        self.assertGreaterEqual(min(counts.values()), 3)
        self.assertEqual(sum(outlet.holdout for outlet in enabled), 5)

    def test_invalid_label_is_rejected(self) -> None:
        rows = []
        for index, label in enumerate(("Left", "Lean Left", "Center", "Lean Right", "Unknown")):
            rows.append(f"s{index},Source {index},example.com,{label},news,https://rating/{index},2026-08-27,https://example.com/feed/{index},rss,true,false\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outlets.csv"
            path.write_text(HEADER + "".join(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid label"):
                load_registry(path)


if __name__ == "__main__":
    unittest.main()
