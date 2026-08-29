from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


LABEL_ORDER = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
ORDINAL_VALUE = {label: index - 2 for index, label in enumerate(LABEL_ORDER)}
REQUIRED_COLUMNS = {
    "record_id",
    "source_id",
    "headline",
    "published_at",
    "observed_at",
    "headline_hash",
    "weak_label",
    "holdout_outlet",
}


def load_modeling_data(path: str | Path) -> pd.DataFrame:
    """Load the processed snapshot and normalize modeling-specific dtypes."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["holdout_outlet"] = (
        frame["holdout_outlet"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )
    if frame["holdout_outlet"].isna().any():
        raise ValueError("holdout_outlet contains a value other than true/false")

    unknown_labels = set(frame["weak_label"]) - set(LABEL_ORDER)
    if unknown_labels:
        raise ValueError(f"Unknown weak labels: {', '.join(sorted(unknown_labels))}")

    published = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    observed = pd.to_datetime(frame["observed_at"], errors="coerce", utc=True)
    frame["model_timestamp"] = published.fillna(observed)
    if frame["model_timestamp"].isna().any():
        raise ValueError("Every row needs either a publication or observation timestamp")

    frame["model_text"] = frame["headline"].fillna("").astype(str).str.strip()
    if frame["model_text"].eq("").any():
        raise ValueError("Blank headlines cannot be modeled")
    return frame


def make_pilot_split(frame: pd.DataFrame, validation_fraction: float = 0.20) -> pd.DataFrame:
    """Create an outlet-OOD test and within-outlet chronological validation split.

    Holdout outlets are never used during development. For every remaining outlet,
    its newest records form a temporal validation tail. Exact normalized headlines
    are kept together by promoting an entire development-side hash group to
    validation if any member initially lands there.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    manifest = frame[["record_id", "source_id", "weak_label", "headline_hash", "model_timestamp"]].copy()
    manifest["split"] = "train"
    manifest["split_reason"] = "older_non_holdout_record"

    holdout_mask = frame["holdout_outlet"].to_numpy(dtype=bool)
    manifest.loc[holdout_mask, "split"] = "test_ood"
    manifest.loc[holdout_mask, "split_reason"] = "frozen_holdout_outlet"

    development = manifest.loc[~holdout_mask]
    for _, group in development.groupby("source_id", sort=True):
        ordered = group.sort_values(["model_timestamp", "record_id"])
        validation_size = max(1, int(np.ceil(len(ordered) * validation_fraction)))
        validation_ids = ordered.tail(validation_size)["record_id"]
        selected = manifest["record_id"].isin(validation_ids)
        manifest.loc[selected, "split"] = "validation"
        manifest.loc[selected, "split_reason"] = "newest_within_outlet"

    validation_hashes = set(manifest.loc[manifest["split"].eq("validation"), "headline_hash"])
    duplicate_group_mask = manifest["headline_hash"].isin(validation_hashes) & manifest["split"].eq("train")
    manifest.loc[duplicate_group_mask, "split"] = "validation"
    manifest.loc[duplicate_group_mask, "split_reason"] = "exact_hash_grouped_with_validation"

    ood_hashes = set(manifest.loc[manifest["split"].eq("test_ood"), "headline_hash"])
    cross_boundary = manifest["headline_hash"].isin(ood_hashes) & ~manifest["split"].eq("test_ood")
    manifest.loc[cross_boundary, "split"] = "excluded"
    manifest.loc[cross_boundary, "split_reason"] = "exact_hash_also_in_frozen_ood"

    assert_split_integrity(manifest)
    return manifest.sort_values("record_id").reset_index(drop=True)


def assert_split_integrity(manifest: pd.DataFrame) -> None:
    """Raise when source or exact-headline leakage invalidates the pilot split."""
    required = {"record_id", "source_id", "weak_label", "headline_hash", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise AssertionError(f"Manifest lacks: {', '.join(sorted(missing))}")
    if manifest["record_id"].duplicated().any():
        raise AssertionError("A record appears more than once in the split manifest")

    development_sources = set(manifest.loc[manifest["split"].isin(["train", "validation"]), "source_id"])
    ood_sources = set(manifest.loc[manifest["split"].eq("test_ood"), "source_id"])
    if development_sources & ood_sources:
        raise AssertionError("An outlet occurs in both development and OOD test data")

    active = manifest.loc[manifest["split"].isin(["train", "validation", "test_ood"])]
    split_counts_per_hash = active.groupby("headline_hash")["split"].nunique()
    if split_counts_per_hash.gt(1).any():
        raise AssertionError("An exact headline hash crosses active splits")

    for split in ("train", "validation", "test_ood"):
        labels = set(manifest.loc[manifest["split"].eq(split), "weak_label"])
        if labels != set(LABEL_ORDER):
            raise AssertionError(f"{split} does not contain all five labels")


def attach_split(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach split metadata by immutable record ID."""
    columns = ["record_id", "split", "split_reason"]
    merged = frame.merge(manifest[columns], on="record_id", how="left", validate="one_to_one")
    if merged["split"].isna().any():
        raise ValueError("Some dataset records are missing from the split manifest")
    return merged


def build_word_logistic(ngram_range: tuple[int, int] = (1, 1), min_df: int = 2) -> Pipeline:
    """Return the first interpretable headline model: word TF-IDF plus LR."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=0.98,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def prediction_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    """Compute class-balanced and ordinal metrics for ordered five-way labels."""
    truth = np.asarray(list(y_true))
    prediction = np.asarray(list(y_pred))
    true_ordinal = np.asarray([ORDINAL_VALUE[label] for label in truth])
    pred_ordinal = np.asarray([ORDINAL_VALUE[label] for label in prediction])
    distance = np.abs(true_ordinal - pred_ordinal)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision_score(truth, prediction, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, prediction, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=LABEL_ORDER, average="weighted", zero_division=0)),
        "ordinal_mae": float(distance.mean()),
        "within_one_class_accuracy": float((distance <= 1).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(true_ordinal, pred_ordinal, weights="quadratic")),
    }


def per_class_report(y_true: Iterable[str], y_pred: Iterable[str]) -> pd.DataFrame:
    """Return precision, recall and F1 in ideological order."""
    report = classification_report(
        list(y_true), list(y_pred), labels=LABEL_ORDER, output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).T.loc[LABEL_ORDER, ["precision", "recall", "f1-score", "support"]]


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Return every learned coefficient with its class and TF-IDF feature."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    features = vectorizer.get_feature_names_out()
    tables = []
    for class_index, class_name in enumerate(classifier.classes_):
        tables.append(
            pd.DataFrame(
                {
                    "class": class_name,
                    "feature": features,
                    "coefficient": classifier.coef_[class_index],
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def explain_document(model: Pipeline, text: str, predicted_class: str | None = None) -> pd.DataFrame:
    """Decompose one class score into non-zero TF-IDF × coefficient contributions."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    vector = vectorizer.transform([text])
    chosen_class = predicted_class or str(model.predict([text])[0])
    class_index = list(classifier.classes_).index(chosen_class)
    feature_names = vectorizer.get_feature_names_out()
    indices = vector.indices
    contributions = vector.data * classifier.coef_[class_index, indices]
    explanation = pd.DataFrame(
        {
            "feature": feature_names[indices],
            "tfidf": vector.data,
            "coefficient": classifier.coef_[class_index, indices],
            "contribution": contributions,
        }
    )
    return explanation.sort_values("contribution", ascending=False).reset_index(drop=True)
