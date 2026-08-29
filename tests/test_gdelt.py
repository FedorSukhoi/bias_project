from datetime import date
import unittest

from bias_dataset.backfill_gdelt import article_item, gdelt_timestamp, query_url
from bias_dataset.registry import Outlet


class GdeltBackfillTests(unittest.TestCase):
    def test_article_conversion(self) -> None:
        item = article_item({
            "title": "Senate votes on election legislation",
            "url": "https://example.com/politics/story?utm_source=x",
            "seendate": "20260820T134500Z",
        })
        self.assertIsNotNone(item)
        self.assertEqual(item["canonical_url"], "https://example.com/politics/story")
        self.assertEqual(item["published_at"], "2026-08-20T13:45:00Z")

    def test_query_is_domain_and_time_bounded(self) -> None:
        outlet = Outlet("example", "Example", "example.com", "Center", "news", "https://rating", "2026-08-27", "https://example.com/feed", "rss", True, False)
        url = query_url(outlet, date(2026, 7, 1), date(2026, 7, 31), 250)
        self.assertIn("domainis%3Aexample.com", url)
        self.assertIn("startdatetime=20260701000000", url)
        self.assertIn("enddatetime=20260731235959", url)

    def test_invalid_timestamp_is_blank(self) -> None:
        self.assertEqual(gdelt_timestamp("bad"), "")


if __name__ == "__main__":
    unittest.main()

