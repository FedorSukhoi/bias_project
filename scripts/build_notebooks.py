"""Generate the narrated modeling notebooks from small, reviewable cell sources."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3 (bias-project)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_DIR / name)


COMMON_SETUP = """
from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_PATH = ROOT / "data/processed/publication_proxy_headlines.csv"
"""


write_notebook(
    "01_data_audit.ipynb",
    [
        md("""
        # 01 — Data audit before modeling

        This notebook establishes what the first model is allowed to learn from. The target is the
        **publication's weak AllSides label**, not a human judgment of each headline. We use only
        `headline` as model text; RSS summaries are reserved for a later ablation because they are
        not consistently true subheadlines.

        Run notebooks in numeric order. Each code cell performs one visible audit step.
        """),
        md("""
        ## Reproducibility setup

        `ROOT` makes the notebook work whether Jupyter starts in the repository or `notebooks/`.
        No random operation occurs in this audit.
        """),
        code(COMMON_SETUP),
        code("""
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        from bias_dataset.modeling import LABEL_ORDER, load_modeling_data

        sns.set_theme(style="whitegrid")
        pd.set_option("display.max_colwidth", 100)
        """),
        md("## Load the frozen processed snapshot"),
        code("""
        data = load_modeling_data(DATA_PATH)
        print(f"Rows: {len(data):,}")
        print(f"Columns: {data.shape[1]}")
        data[["record_id", "source_id", "headline", "weak_label"]].head()
        """),
        md("""
        ## Validate identity and label semantics

        Duplicate record IDs would make the same observation count more than once. `label_scope`
        must remain `outlet`, which prevents us from accidentally describing these weak labels as
        headline-level ground truth.
        """),
        code("""
        identity_checks = pd.Series({
            "unique_record_ids": data["record_id"].nunique(),
            "duplicate_record_ids": data["record_id"].duplicated().sum(),
            "blank_headlines": data["model_text"].eq("").sum(),
            "outlet_scope_rows": data["label_scope"].eq("outlet").sum(),
        })
        identity_checks
        """),
        md("## Class distribution"),
        code("""
        class_counts = data["weak_label"].value_counts().reindex(LABEL_ORDER)
        class_counts.to_frame("records")
        """),
        code("""
        ax = class_counts.plot.bar(color="#4C78A8", figsize=(8, 4))
        ax.set(title="Weak-label class distribution", xlabel="Outlet label", ylabel="Headlines")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        The class imbalance is visible before modeling. We retain the natural archive and use
        `class_weight='balanced'` in Logistic Regression rather than synthesizing TF-IDF vectors.
        """),
        md("## Publication concentration inside each class"),
        code("""
        outlet_counts = (
            data.groupby(["weak_label", "source_id"], observed=True)
            .size().rename("records").reset_index()
        )
        outlet_counts["weak_label"] = pd.Categorical(
            outlet_counts["weak_label"], categories=LABEL_ORDER, ordered=True
        )
        outlet_counts.sort_values(["weak_label", "records"], ascending=[True, False])
        """),
        code("""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=outlet_counts, y="source_id", x="records", hue="weak_label", hue_order=LABEL_ORDER)
        plt.title("Records per publication")
        plt.xlabel("Headlines")
        plt.ylabel("Publication ID")
        plt.tight_layout()
        plt.show()
        """),
        md("## Headline length and vocabulary surface"),
        code("""
        data["headline_words"] = data["model_text"].str.split().str.len()
        data["headline_chars"] = data["model_text"].str.len()
        data[["headline_words", "headline_chars"]].describe(percentiles=[.05, .25, .5, .75, .95]).round(1)
        """),
        code("""
        plt.figure(figsize=(8, 4))
        sns.histplot(data=data, x="headline_words", hue="weak_label", hue_order=LABEL_ORDER,
                     element="step", stat="density", common_norm=False, bins=30)
        plt.title("Headline word-count distributions")
        plt.tight_layout()
        plt.show()
        """),
        md("## Time coverage"),
        code("""
        time_summary = data.groupby("source_id").agg(
            first_record=("model_timestamp", "min"),
            last_record=("model_timestamp", "max"),
            records=("record_id", "size"),
        )
        time_summary.sort_values("first_record")
        """),
        md("""
        Archive depth differs greatly between sources. Notebook 02 therefore makes the validation
        tail **within each development outlet**, avoiding a global date cutoff that could remove
        entire publications merely because their feed exposed older material.
        """),
        md("## Holdout design and unavailable true subheadlines"),
        code("""
        holdout_summary = (
            data.groupby(["weak_label", "source_id", "holdout_outlet"], observed=True)
            .size().rename("records").reset_index()
        )
        holdout_summary[holdout_summary["holdout_outlet"]].sort_values("weak_label")
        """),
        code("""
        text_availability = pd.Series({
            "true_subheadlines": data["subheadline"].fillna("").ne("").sum(),
            "rss_summaries": data["feed_summary"].fillna("").ne("").sum(),
            "missing_publisher_timestamps": pd.to_datetime(data["published_at"], errors="coerce", utc=True).isna().sum(),
        })
        text_availability
        """),
        md("""
        ## Audit conclusion

        The snapshot is large enough for a **pilot**, not a final performance claim. The first
        experiment will use headline-only word TF-IDF, balanced Logistic Regression, a temporal
        validation tail, and five untouched publication holdouts. The most important limitation is
        construct validity: success means predicting outlet-associated language/content patterns.
        """),
    ],
)


write_notebook(
    "02_split_design.ipynb",
    [
        md("""
        # 02 — Leakage-aware split design

        This notebook freezes three roles:

        - `train`: older headlines from non-holdout publications;
        - `validation`: newest 20% within each non-holdout publication;
        - `test_ood`: all records from one unseen publication per class.

        The OOD set is not used for model selection. Exact normalized headlines cannot cross active
        splits; a development record duplicated in the frozen OOD set is marked `excluded`.
        """),
        code(COMMON_SETUP),
        code("""
        import hashlib
        import json
        import pandas as pd

        from bias_dataset.modeling import (
            LABEL_ORDER, assert_split_integrity, load_modeling_data, make_pilot_split
        )

        SPLIT_DIR = ROOT / "data/splits"
        SPLIT_PATH = SPLIT_DIR / "pilot_v1.csv"
        """),
        md("## Load data and create the deterministic manifest"),
        code("""
        data = load_modeling_data(DATA_PATH)
        manifest = make_pilot_split(data, validation_fraction=0.20)
        manifest.head()
        """),
        md("## Count each split before training"),
        code("""
        split_counts = manifest["split"].value_counts().rename_axis("split").to_frame("records")
        split_counts
        """),
        code("""
        split_by_class = pd.crosstab(manifest["weak_label"], manifest["split"]).reindex(LABEL_ORDER)
        split_by_class
        """),
        md("""
        Macro-F1 later gives every class equal importance even though the frozen OOD publications
        contribute different record counts. Training also uses balanced class weights.
        """),
        md("## Confirm publication isolation"),
        code("""
        source_allocation = (
            manifest.groupby(["weak_label", "source_id", "split"], observed=True)
            .size().rename("records").reset_index()
        )
        source_allocation.sort_values(["weak_label", "source_id", "split"])
        """),
        code("""
        development_sources = set(manifest.loc[manifest["split"].isin(["train", "validation"]), "source_id"])
        ood_sources = set(manifest.loc[manifest["split"].eq("test_ood"), "source_id"])
        print("Development outlets:", sorted(development_sources))
        print("OOD outlets:", sorted(ood_sources))
        print("Overlap:", development_sources & ood_sources)
        """),
        md("## Confirm exact-headline isolation"),
        code("""
        active = manifest[manifest["split"].isin(["train", "validation", "test_ood"])]
        hashes_crossing_splits = active.groupby("headline_hash")["split"].nunique().gt(1).sum()
        print("Exact headline hashes crossing active splits:", hashes_crossing_splits)
        """),
        code("""
        excluded = manifest[manifest["split"].eq("excluded")]
        print(f"Excluded cross-boundary duplicate records: {len(excluded)}")
        excluded[["record_id", "source_id", "weak_label", "split_reason"]]
        """),
        md("## Run executable leakage assertions"),
        code("""
        assert_split_integrity(manifest)
        print("PASS: unique IDs, all five labels, outlet isolation, and exact-hash isolation.")
        """),
        md("## Inspect the within-outlet time boundary"),
        code("""
        boundaries = (
            manifest[manifest["split"].isin(["train", "validation"])]
            .groupby(["source_id", "split"], observed=True)["model_timestamp"]
            .agg(["min", "max", "count"])
        )
        boundaries
        """),
        md("""
        An exact duplicate can promote an older training record into validation. This is intentional:
        duplicate isolation takes precedence over a perfectly sharp date boundary.
        """),
        md("## Save the versioned manifest and checksum"),
        code("""
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(SPLIT_PATH, index=False)

        dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
        print("Manifest:", SPLIT_PATH.relative_to(ROOT))
        print("Dataset SHA-256:", dataset_sha256)
        """),
        code("""
        metadata = {
            "split_version": "pilot_v1",
            "dataset_sha256": dataset_sha256,
            "validation_fraction": 0.20,
            "counts": manifest["split"].value_counts().sort_index().to_dict(),
            "ood_sources": sorted(ood_sources),
        }
        metadata_path = SPLIT_DIR / "pilot_v1_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\\n")
        metadata
        """),
    ],
)


write_notebook(
    "03_interpretable_baselines.ipynb",
    [
        md("""
        # 03 — Interpretable development baselines

        We compare a trivial class-prior predictor with two transparent models:

        1. word-unigram TF-IDF + balanced Logistic Regression;
        2. word unigram/bigram TF-IDF + balanced Logistic Regression.

        TF-IDF is fitted inside each pipeline on `train` only. The frozen unseen-outlet test is not
        read during model selection. Logistic Regression is chosen first because every feature has a
        class-specific coefficient and the model exposes probabilities (not yet calibrated).
        """),
        code(COMMON_SETUP),
        code("""
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier

        from bias_dataset.modeling import (
            attach_split, build_word_logistic, load_modeling_data, prediction_metrics
        )

        SPLIT_PATH = ROOT / "data/splits/pilot_v1.csv"
        MODEL_DIR = ROOT / "models"
        REPORT_DIR = ROOT / "reports/modeling"
        """),
        md("## Load only the development partitions"),
        code("""
        data = load_modeling_data(DATA_PATH)
        manifest = pd.read_csv(SPLIT_PATH)
        modeled = attach_split(data, manifest)

        train = modeled[modeled["split"].eq("train")].copy()
        validation = modeled[modeled["split"].eq("validation")].copy()
        print({"train": len(train), "validation": len(validation)})
        """),
        code("""
        assert not train["holdout_outlet"].any()
        assert not validation["holdout_outlet"].any()
        print("PASS: no holdout-outlet rows entered development.")
        """),
        md("## Establish the class-prior baseline"),
        code("""
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(train[["model_text"]], train["weak_label"])
        dummy_prediction = dummy.predict(validation[["model_text"]])
        dummy_metrics = prediction_metrics(validation["weak_label"], dummy_prediction)
        pd.Series(dummy_metrics, name="class_prior")
        """),
        md("""
        The dummy model ignores words and always predicts the most common training class. Any useful
        text model should improve substantially on its macro-F1.
        """),
        md("## Train the unigram model"),
        code("""
        unigram_model = build_word_logistic(ngram_range=(1, 1), min_df=2)
        unigram_model.fit(train["model_text"], train["weak_label"])
        unigram_prediction = unigram_model.predict(validation["model_text"])
        unigram_metrics = prediction_metrics(validation["weak_label"], unigram_prediction)
        pd.Series(unigram_metrics, name="word_unigrams")
        """),
        md("""
        Feature notes: lowercasing and Unicode accent stripping reduce accidental sparsity; stopwords
        remain because negation and function words may carry framing; `sublinear_tf` prevents repeated
        terms from growing linearly; L2 normalization keeps long headlines from dominating by length.
        """),
        md("## Train the unigram + bigram model"),
        code("""
        bigram_model = build_word_logistic(ngram_range=(1, 2), min_df=2)
        bigram_model.fit(train["model_text"], train["weak_label"])
        bigram_prediction = bigram_model.predict(validation["model_text"])
        bigram_metrics = prediction_metrics(validation["weak_label"], bigram_prediction)
        pd.Series(bigram_metrics, name="word_1_2grams")
        """),
        md("""
        Bigrams allow features such as `white house` or `border policy` to differ from their individual
        words. `min_df=2` removes one-off phrases that cannot demonstrate a repeatable association in
        this small pilot.
        """),
        md("## Compare models on the same temporal validation rows"),
        code("""
        comparison = pd.DataFrame({
            "class_prior": dummy_metrics,
            "word_unigrams": unigram_metrics,
            "word_1_2grams": bigram_metrics,
        }).T.sort_values("macro_f1", ascending=False)
        comparison.round(3)
        """),
        code("""
        model_candidates = {
            "word_unigrams": unigram_model,
            "word_1_2grams": bigram_model,
        }
        selected_name = comparison.loc[list(model_candidates), "macro_f1"].idxmax()
        selected_model = model_candidates[selected_name]
        print("Selected by validation macro-F1:", selected_name)
        """),
        md("## Save the development artifact and validation predictions"),
        code("""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(selected_model, MODEL_DIR / "selected_development_lr.joblib")
        comparison.to_csv(REPORT_DIR / "development_model_comparison.csv")
        """),
        code("""
        selected_prediction = selected_model.predict(validation["model_text"])
        validation_output = validation[["record_id", "source_id", "headline", "weak_label"]].copy()
        validation_output["prediction"] = selected_prediction
        validation_output.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
        validation_output.head()
        """),
        md("""
        ## Development conclusion

        The selected model is only a choice between two word-feature configurations. Character
        n-grams, LinearSVC and ComplementNB belong in the next model-comparison stage. Notebook 04
        asks what this first model learned before Notebook 05 opens the OOD test.
        """),
    ],
)


write_notebook(
    "04_feature_interpretation.ipynb",
    [
        md("""
        # 04 — What did the first model learn?

        Logistic Regression is additive in TF-IDF feature space. A positive coefficient raises a
        class score; a negative coefficient lowers it. Coefficients reveal association with weak
        publication labels, **not causal evidence or a universal bias dictionary**.
        """),
        code(COMMON_SETUP),
        code("""
        import joblib
        import pandas as pd

        from bias_dataset.modeling import (
            LABEL_ORDER, attach_split, coefficient_table, explain_document, load_modeling_data
        )

        SPLIT_PATH = ROOT / "data/splits/pilot_v1.csv"
        MODEL_PATH = ROOT / "models/selected_development_lr.joblib"
        REPORT_DIR = ROOT / "reports/modeling"
        """),
        md("## Load the selected development model"),
        code("""
        model = joblib.load(MODEL_PATH)
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        print("N-gram range:", vectorizer.ngram_range)
        print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
        print("Classes learned:", list(classifier.classes_))
        """),
        md("## Build the complete coefficient table"),
        code("""
        coefficients = coefficient_table(model)
        coefficients.head()
        """),
        md("## Top positive associations for each class"),
        code("""
        top_positive = (
            coefficients.sort_values(["class", "coefficient"], ascending=[True, False])
            .groupby("class", sort=False).head(15)
        )
        top_positive.pivot(index="feature", columns="class", values="coefficient").fillna("")
        """),
        md("""
        Read these as: “holding other observed features fixed, this phrase pushes the linear score
        toward this outlet class.” Topic, named-entity, and publication fingerprints can all appear.
        """),
        md("## Most negative associations for each class"),
        code("""
        top_negative = (
            coefficients.sort_values(["class", "coefficient"], ascending=[True, True])
            .groupby("class", sort=False).head(10)
        )
        top_negative.pivot(index="feature", columns="class", values="coefficient").fillna("")
        """),
        md("## Audit obvious publisher-fingerprint terms"),
        code("""
        fingerprint_terms = {
            "cnn", "fox", "newsmax", "breitbart", "mediaite", "vox", "reason",
            "mother jones", "daily wire", "the hill", "newsnation", "dispatch",
            "examiner", "atlantic", "abc", "nbc",
        }
        fingerprint_hits = coefficients[coefficients["feature"].isin(fingerprint_terms)]
        fingerprint_hits.reindex(fingerprint_hits["coefficient"].abs().sort_values(ascending=False).index).head(30)
        """),
        md("""
        A fingerprint hit is a warning, not automatic proof of leakage: an outlet name can be the
        subject of a legitimate political headline. Large branded coefficients should nevertheless
        motivate later masking and source-specific error audits.
        """),
        md("## Choose a correctly predicted validation example"),
        code("""
        data = load_modeling_data(DATA_PATH)
        manifest = pd.read_csv(SPLIT_PATH)
        validation = attach_split(data, manifest).query("split == 'validation'").copy()

        probabilities = model.predict_proba(validation["model_text"])
        validation["prediction"] = model.classes_[probabilities.argmax(axis=1)]
        validation["confidence"] = probabilities.max(axis=1)
        example = validation[validation["prediction"].eq(validation["weak_label"])].nlargest(1, "confidence").iloc[0]
        example[["headline", "weak_label", "prediction", "confidence"]]
        """),
        md("## Decompose that prediction feature by feature"),
        code("""
        explanation = explain_document(model, example["model_text"], example["prediction"])
        explanation.head(15)
        """),
        md("""
        `contribution = TF-IDF value × class coefficient`. The intercept and the competing classes'
        scores also affect the final softmax probability, so the table explains the chosen class
        score rather than claiming each token independently caused the prediction.
        """),
        md("## Save compact interpretation artifacts"),
        code("""
        top_features = pd.concat([
            top_positive.assign(direction="positive"),
            top_negative.assign(direction="negative"),
        ])
        top_features.to_csv(REPORT_DIR / "top_model_features.csv", index=False)
        explanation.to_csv(REPORT_DIR / "example_feature_contributions.csv", index=False)
        print("Saved top features and the example decomposition.")
        """),
    ],
)


write_notebook(
    "05_error_analysis.ipynb",
    [
        md("""
        # 05 — Frozen unseen-outlet evaluation and error analysis

        Only now do we open `test_ood`. We first recover the selected n-gram configuration, refit it
        on all non-holdout development data, and evaluate once on the five held-out publications.
        The resulting scores measure **publication-proxy generalization**, not independently judged
        headline bias.
        """),
        code(COMMON_SETUP),
        code("""
        import json
        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        from bias_dataset.modeling import (
            LABEL_ORDER, ORDINAL_VALUE, attach_split, build_word_logistic,
            load_modeling_data, per_class_report, prediction_metrics,
        )

        sns.set_theme(style="whitegrid")
        """),
        code("""
        SPLIT_PATH = ROOT / "data/splits/pilot_v1.csv"
        DEVELOPMENT_MODEL_PATH = ROOT / "models/selected_development_lr.joblib"
        FINAL_MODEL_PATH = ROOT / "models/pilot_final_lr.joblib"
        REPORT_DIR = ROOT / "reports/modeling"
        """),
        md("## Reconfirm the frozen data roles"),
        code("""
        data = load_modeling_data(DATA_PATH)
        manifest = pd.read_csv(SPLIT_PATH)
        modeled = attach_split(data, manifest)

        development = modeled[modeled["split"].isin(["train", "validation"])].copy()
        test_ood = modeled[modeled["split"].eq("test_ood")].copy()
        print({"development": len(development), "test_ood": len(test_ood)})
        """),
        code("""
        development_sources = set(development["source_id"])
        test_sources = set(test_ood["source_id"])
        assert development_sources.isdisjoint(test_sources)
        assert set(development["headline_hash"]).isdisjoint(test_ood["headline_hash"])
        print("PASS: no outlet or exact-headline overlap with OOD test.")
        """),
        md("## Refit the selected architecture on all development rows"),
        code("""
        development_model = joblib.load(DEVELOPMENT_MODEL_PATH)
        selected_ngram_range = development_model.named_steps["tfidf"].ngram_range
        print("Selected n-gram range:", selected_ngram_range)
        """),
        code("""
        final_model = build_word_logistic(ngram_range=selected_ngram_range, min_df=2)
        final_model.fit(development["model_text"], development["weak_label"])
        joblib.dump(final_model, FINAL_MODEL_PATH)
        print("Final vocabulary size:", len(final_model.named_steps["tfidf"].get_feature_names_out()))
        """),
        md("## Make the one-time OOD predictions"),
        code("""
        prediction = final_model.predict(test_ood["model_text"])
        probability = final_model.predict_proba(test_ood["model_text"])
        metrics = prediction_metrics(test_ood["weak_label"], prediction)
        pd.Series(metrics, name="unseen_outlet_test").round(3)
        """),
        md("""
        Macro-F1 is primary because every ideological class counts equally. Ordinal MAE measures how
        many class steps an error travels, while within-one-class accuracy credits adjacent mistakes.
        Quadratic kappa penalizes distant ideological errors more strongly.
        """),
        md("## Inspect per-class precision and recall"),
        code("""
        class_report = per_class_report(test_ood["weak_label"], prediction)
        class_report.round(3)
        """),
        md("## Raw and normalized confusion matrices"),
        code("""
        raw_matrix = confusion_matrix(test_ood["weak_label"], prediction, labels=LABEL_ORDER)
        normalized_matrix = confusion_matrix(
            test_ood["weak_label"], prediction, labels=LABEL_ORDER, normalize="true"
        )
        """),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(raw_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
        sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, vmin=0, vmax=1)
        axes[0].set_title("OOD confusion matrix — counts")
        axes[1].set_title("OOD confusion matrix — true-class normalized")
        for axis in axes:
            axis.set(xlabel="Predicted", ylabel="Weak outlet label")
            axis.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.show()
        """),
        md("## Build a row-level error table"),
        code("""
        results = test_ood[["record_id", "source_id", "headline", "weak_label"]].copy()
        results["prediction"] = prediction
        results["confidence"] = probability.max(axis=1)
        results["correct"] = results["weak_label"].eq(results["prediction"])
        results["ordinal_distance"] = [
            abs(ORDINAL_VALUE[truth] - ORDINAL_VALUE[pred])
            for truth, pred in zip(results["weak_label"], results["prediction"])
        ]
        results.head()
        """),
        md("## Review the most confident errors"),
        code("""
        columns = ["source_id", "headline", "weak_label", "prediction", "confidence", "ordinal_distance"]
        confident_errors = results[~results["correct"]].nlargest(20, "confidence")
        confident_errors[columns]
        """),
        md("""
        High-confidence errors are especially useful for finding outlet fingerprints, neutral
        headlines carrying noisy weak labels, named-entity shortcuts, and story-selection effects.
        These probabilities are **not calibrated**, so confidence is only a ranking signal here.
        """),
        md("## Compare error severity by held-out publication"),
        code("""
        outlet_errors = results.groupby("source_id").agg(
            records=("record_id", "size"),
            accuracy=("correct", "mean"),
            mean_ordinal_distance=("ordinal_distance", "mean"),
            mean_confidence=("confidence", "mean"),
        ).sort_values("accuracy")
        outlet_errors.round(3)
        """),
        md("## Save reproducible evaluation artifacts"),
        code("""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(REPORT_DIR / "ood_predictions.csv", index=False)
        class_report.to_csv(REPORT_DIR / "ood_per_class_report.csv")
        outlet_errors.to_csv(REPORT_DIR / "ood_outlet_errors.csv")

        metrics_path = REPORT_DIR / "ood_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\\n")
        print("Saved OOD metrics, predictions, and error summaries.")
        """),
        md("""
        ## Interpretation boundary and next step

        This is the first honest weak-label baseline. Performance can be limited by only two or three
        development publications per class, uneven archive depth, topic differences, and label noise.
        The next iteration should compare character TF-IDF, LinearSVC and ComplementNB using the same
        development split—without reopening or optimizing against this OOD result.
        """),
    ],
)


print(f"Wrote five notebooks to {NOTEBOOK_DIR}")
