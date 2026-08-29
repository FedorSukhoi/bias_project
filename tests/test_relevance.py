import unittest

from bias_dataset.relevance import assess_political_relevance


class PoliticalRelevanceTests(unittest.TestCase):
    def test_user_examples_are_excluded(self) -> None:
        examples = (
            ("The 10 best Amazon deals to shop this week", "https://cnn.com/cnn-underscored/deals/amazon"),
            ("The beloved Dyson Supersonic hair dryer is at its lowest price ever", "https://cnn.com/cnn-underscored/deals/dyson"),
            ("Mother's Day is around the corner. Here are 50+ thoughtful gifts she'll love", "https://cnn.com/cnn-underscored/gifts/mothers-day"),
        )
        for headline, url in examples:
            with self.subTest(headline=headline):
                self.assertFalse(assess_political_relevance(headline, "", url).relevant)

    def test_political_news_and_analysis_are_included(self) -> None:
        examples = (
            ("Senate advances border legislation after bipartisan talks", "", "https://example.com/news/story"),
            ("Why the administration's tariff policy may reshape the election", "", "https://example.com/analysis/story"),
            ("Court weighs challenge to presidential immigration order", "", "https://example.com/politics/court-case"),
        )
        for headline, summary, url in examples:
            with self.subTest(headline=headline):
                self.assertTrue(assess_political_relevance(headline, summary, url).relevant)

    def test_single_ambiguous_term_is_not_enough(self) -> None:
        result = assess_political_relevance(
            "Local judge wins community cooking contest", "", "https://example.com/community/story"
        )
        self.assertFalse(result.relevant)
        self.assertEqual(result.reason, "single_ambiguous_signal")


if __name__ == "__main__":
    unittest.main()

