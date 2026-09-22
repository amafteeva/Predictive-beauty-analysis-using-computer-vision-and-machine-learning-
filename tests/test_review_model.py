import numpy as np
import pandas as pd
import pytest

from foundation_matcher.review_model import (
    attach_product_metadata,
    build_review_classifier,
    grouped_train_test_split,
    prepare_review_data,
    summarize_luxxify_products,
    summarize_luxxify_reviews,
)


def _reviews():
    return pd.DataFrame(
        {
            "comments": [f"review {i}" for i in range(15)],
            "rating": [5, 5, 4, 2, 1, 5, 4, 3, 5, 2, 4, 1, 5, 3, 4],
            "product_link_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        }
    )


def _products():
    return pd.DataFrame(
        {
            "product_link_id": [1, 2, 3, 4, 5],
            "category": ["Foundation", "Foundation", None, "Concealer", "Foundation"],
            "price": [20.0, 45.0, 30.0, 15.0, 25.0],
            "brand": ["Alpha", "Beta", "Gamma", None, "Alpha"],
        }
    )


def test_attach_product_metadata_joins_expected_columns():
    reviews = prepare_review_data(
        _reviews(),
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )
    merged = attach_product_metadata(reviews, _products())

    assert {"category", "price", "brand"}.issubset(merged.columns)
    assert len(merged) == len(reviews)
    assert merged.loc[merged["product_link_id"] == 1, "price"].iloc[0] == 20.0


def test_attach_product_metadata_does_not_duplicate_rows_on_duplicate_product_ids():
    reviews = prepare_review_data(
        _reviews(),
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )
    duplicated_products = pd.concat([_products(), _products()], ignore_index=True)

    merged = attach_product_metadata(reviews, duplicated_products)

    assert len(merged) == len(reviews)


def test_attach_product_metadata_raises_on_missing_columns():
    reviews = prepare_review_data(
        _reviews(),
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )
    incomplete_products = _products().drop(columns=["price"])

    with pytest.raises(ValueError, match="price"):
        attach_product_metadata(reviews, incomplete_products)


def test_grouped_train_test_split_with_feature_columns_has_no_leakage():
    reviews = prepare_review_data(
        _reviews(),
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )
    merged = attach_product_metadata(reviews, _products())

    X_train, X_test, y_train, y_test, train_groups, test_groups = (
        grouped_train_test_split(
            merged,
            feature_columns=("review_text", "category", "price", "brand"),
        )
    )

    assert list(X_train.columns) == ["review_text", "category", "price", "brand"]
    assert not set(train_groups).intersection(test_groups)
    assert len(X_train) + len(X_test) == len(merged)


def test_build_review_classifier_fits_and_predicts_with_product_metadata():
    reviews = prepare_review_data(
        _reviews(),
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )
    merged = attach_product_metadata(reviews, _products())
    features = merged[["review_text", "category", "price", "brand"]]
    target = merged["positive_review"]

    model = build_review_classifier(
        categorical_columns=("category", "brand"),
        numeric_columns=("price",),
        minimum_document_frequency=1,
    )
    model.fit(features, target)
    predictions = model.predict(features)

    assert len(predictions) == len(features)


def test_summarize_luxxify_products_reports_missingness_and_price():
    raw_products = pd.DataFrame(
        {
            "category": ["Foundation", "Foundation", None, "Blush"],
            "brand": ["Alpha", None, "Gamma", "Delta"],
            "price": [20.0, 45.0, 30.0, 15.0],
            "num_shades": [np.nan, np.nan, np.nan, 12.0],
        }
    )

    summary = summarize_luxxify_products(raw_products)

    assert summary["total_products"] == 4
    assert summary["total_categories"] == 2
    assert summary["category_missing_rate"] == 0.25
    assert summary["brand_missing_rate"] == 0.25
    assert summary["price_missing_rate"] == 0.0
    assert summary["num_shades_missing_rate"] == 0.75
    assert summary["price_median"] == 25.0


def test_summarize_luxxify_reviews_reports_text_coverage_and_spread():
    raw_reviews = pd.DataFrame(
        {
            "comments": ["one two three", None, "four five six seven", None, "eight"],
            "product_link_id": [1, 1, 2, 2, 2],
            "is_verified_buyer": ["t", "f", "f", "f", "t"],
        }
    )

    summary = summarize_luxxify_reviews(raw_reviews)

    assert summary["total_reviews"] == 5
    assert summary["reviews_with_text"] == 3
    assert summary["text_missing_rate"] == 0.4
    assert summary["verified_buyer_rate"] == 0.4
    assert summary["median_reviews_per_product"] == 2.5
    assert summary["max_reviews_per_product"] == 3
