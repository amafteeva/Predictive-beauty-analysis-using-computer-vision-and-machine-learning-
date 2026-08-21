"""Optional review-text satisfaction-classification experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from foundation_matcher.config import RANDOM_STATE


def prepare_review_data(
    reviews: pd.DataFrame,
    *,
    text_column: str | None = None,
    rating_column: str | None = None,
    group_column: str | None = None,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Create a clean text dataset and a binary satisfaction target."""

    text_column = _resolve_column(
        reviews,
        text_column,
        ("review_text", "review", "text", "review_body"),
        "review text",
    )
    rating_column = _resolve_column(
        reviews,
        rating_column,
        ("rating", "review_rating", "stars", "star_rating", "review_stars"),
        "rating",
    )
    group_column = _resolve_column(
        reviews,
        group_column,
        ("product_link_id", "item_id", "product_id"),
        "product identifier",
    )

    required = {text_column, rating_column, group_column}
    missing = sorted(required.difference(reviews.columns))
    if missing:
        raise ValueError(f"Review data is missing required columns: {missing}")

    prepared = reviews[[text_column, rating_column, group_column]].copy().rename(
        columns={
            text_column: "review_text",
            rating_column: "rating",
            group_column: "product_link_id",
        }
    )
    prepared["rating"] = pd.to_numeric(prepared["rating"], errors="coerce")
    prepared["review_text"] = prepared["review_text"].astype("string").str.strip()
    prepared = prepared.dropna(
        subset=["review_text", "rating", "product_link_id"]
    )
    prepared = prepared.loc[prepared["review_text"].str.len() > 0]
    prepared = prepared.drop_duplicates(subset=["review_text", "product_link_id"])
    prepared["positive_review"] = (
        prepared["rating"] >= positive_threshold
    ).astype(int)
    return prepared.reset_index(drop=True)


def _resolve_column(
    data: pd.DataFrame,
    requested: str | None,
    candidates: tuple[str, ...],
    description: str,
) -> str:
    """Resolve an optional schema override against a short candidate list."""

    if requested is not None:
        if requested not in data:
            raise ValueError(f"Requested {description} column not found: {requested}")
        return requested
    for candidate in candidates:
        if candidate in data:
            return candidate
    raise ValueError(
        f"Could not identify a {description} column. Available columns: "
        f"{data.columns.tolist()}"
    )


def grouped_train_test_split(
    reviews: pd.DataFrame,
    *,
    text_column: str = "review_text",
    target_column: str = "positive_review",
    group_column: str = "product_link_id",
    random_state: int = RANDOM_STATE,
):
    """Make an approximately 80/20 split with no product leakage."""

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )
    features = reviews[text_column]
    target = reviews[target_column]
    groups = reviews[group_column]
    train_indices, test_indices = next(splitter.split(features, target, groups))
    return (
        features.iloc[train_indices],
        features.iloc[test_indices],
        target.iloc[train_indices],
        target.iloc[test_indices],
        groups.iloc[train_indices],
        groups.iloc[test_indices],
    )


def build_review_classifier(
    *,
    max_features: int = 40_000,
    minimum_document_frequency: int = 5,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create a TF-IDF and class-balanced logistic-regression baseline."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=minimum_document_frequency,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=300,
                    solver="saga",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_review_classifier(
    model: Pipeline,
    test_text: pd.Series,
    test_target: pd.Series,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Return imbalance-aware metrics, a report, and a confusion matrix."""

    predicted = model.predict(test_text)
    probabilities = model.predict_proba(test_text)[:, 1]
    metrics = pd.Series(
        {
            "accuracy": accuracy_score(test_target, predicted),
            "balanced_accuracy": balanced_accuracy_score(test_target, predicted),
            "precision": precision_score(test_target, predicted, zero_division=0),
            "recall": recall_score(test_target, predicted, zero_division=0),
            "f1": f1_score(test_target, predicted, zero_division=0),
            "roc_auc": roc_auc_score(test_target, probabilities),
            "average_precision": average_precision_score(test_target, probabilities),
        },
        name="score",
    )
    report = pd.DataFrame(
        classification_report(
            test_target,
            predicted,
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    matrix = confusion_matrix(test_target, predicted)
    return metrics, report, matrix
