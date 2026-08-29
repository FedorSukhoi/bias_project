from pathlib import Path
import unittest

from bias_dataset.feeds import canonicalize_url, parse_feed, plain_text
from bias_dataset.collect import paged_feed_url


FIXTURES = Path(__file__).parent / "fixtures"


class FeedParsingTests(unittest.TestCase):
    def test_rss_items_are_normalized(self) -> None:
        records = parse_feed((FIXTURES / "sample_rss.xml").read_bytes())
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["headline"], "Senate advances major climate package")
        self.assertEqual(records[0]["feed_summary"], "Lawmakers voted after a lengthy debate.")
        self.assertEqual(records[0]["canonical_url"], "https://example.com/politics/climate-package")
        self.assertEqual(records[0]["published_at"], "2026-08-26T13:45:00Z")
        self.assertEqual(len(records[0]["headline_hash"]), 64)

    def test_url_cleanup_preserves_non_tracking_parameters(self) -> None:
        cleaned = canonicalize_url("HTTPS://Example.COM/a//b/?id=7&utm_medium=email#fragment")
        self.assertEqual(cleaned, "https://example.com/a/b?id=7")

    def test_html_is_removed_from_summary(self) -> None:
        self.assertEqual(plain_text("<b>A &amp; B</b>"), "A & B")

    def test_feed_pagination_preserves_existing_query(self) -> None:
        url = paged_feed_url("https://example.com/feed/?category=politics", 3)
        self.assertIn("category=politics", url)
        self.assertIn("paged=3", url)


if __name__ == "__main__":
    unittest.main()
