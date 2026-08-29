import unittest

import pandas as pd

from bias_dataset.modeling import (
    LABEL_ORDER,
    assert_split_integrity,
    build_word_logistic,
    make_pilot_split,
    prediction_metrics,
)


class ModelingTests(unittest.TestCase):
    def sample_frame(self):
        rows = []
        for class_index, label in enumerate(LABEL_ORDER):
            for holdout in (False, True):
                source = f"source_{class_index}_{int(holdout)}"
                for item_index in range(5):
                    rows.append(
                        {
                            "record_id": f"{source}_{item_index}",
                            "source_id": source,
                            "weak_label": label,
                            "headline_hash": f"hash_{source}_{item_index}",
                            "model_timestamp": pd.Timestamp("2026-01-01", tz="UTC")
                            + pd.Timedelta(days=item_index),
                            "holdout_outlet": holdout,
                        }
                    )
        return pd.DataFrame(rows)

    def test_split_keeps_holdout_outlets_disjoint(self):
        manifest = make_pilot_split(self.sample_frame())
        assert_split_integrity(manifest)
        development = set(manifest.loc[manifest["split"].isin(["train", "validation"]), "source_id"])
        test = set(manifest.loc[manifest["split"].eq("test_ood"), "source_id"])
        self.assertTrue(development.isdisjoint(test))

    def test_split_groups_exact_development_duplicates(self):
        frame = self.sample_frame()
        frame.loc[frame["record_id"].eq("source_0_0_0"), "headline_hash"] = "shared"
        frame.loc[frame["record_id"].eq("source_0_0_4"), "headline_hash"] = "shared"
        manifest = make_pilot_split(frame)
        splits = set(manifest.loc[manifest["headline_hash"].eq("shared"), "split"])
        self.assertEqual(splits, {"validation"})

    def test_ordinal_metrics_penalize_distance(self):
        metrics = prediction_metrics(LABEL_ORDER, LABEL_ORDER[1:] + LABEL_ORDER[:1])
        self.assertAlmostEqual(metrics["accuracy"], 0.0)
        self.assertAlmostEqual(metrics["ordinal_mae"], 1.6)
        self.assertAlmostEqual(metrics["within_one_class_accuracy"], 0.8)

    def test_word_model_exposes_coefficients(self):
        model = build_word_logistic(min_df=1)
        text = [f"shared political words unique{index}" for index in range(10)]
        labels = [LABEL_ORDER[index % len(LABEL_ORDER)] for index in range(10)]
        model.fit(text, labels)
        self.assertEqual(model.named_steps["classifier"].coef_.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
