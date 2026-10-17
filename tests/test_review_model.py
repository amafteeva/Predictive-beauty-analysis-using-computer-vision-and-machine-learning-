import numpy as np
import pandas as pd
import pytest

from foundation_matcher.review_model import (
    attach_product_metadata,
    build_leakage_safe_groups,
    build_majority_baseline,
    build_review_classifier,
    detect_review_languages,
    evaluate_review_classifier,
    find_duplicate_review_clusters,
    grouped_train_test_split,
    prepare_review_data,
    product_disjoint_three_way_split,
    select_logistic_regression_config,
    summarize_language_check,
    summarize_luxxify_products,
    summarize_luxxify_reviews,
    summarize_review_data_quality,
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


def _synthetic_raw_reviews(n_groups: int = 40, per_group: int = 3) -> pd.DataFrame:
    """Reviews whose rating is correlated with (but not identical to) its text.

    Large enough (>=120 rows across 40 products) for StratifiedGroupKFold to
    have room to stratify, with class balance skewed like real review data.
    """

    rng = np.random.default_rng(0)
    rows = []
    review_id = 0
    for group in range(n_groups):
        positive_group = group % 3 != 0  # ~2/3 of products skew positive
        for _ in range(per_group):
            is_positive = bool(rng.random() < (0.85 if positive_group else 0.25))
            rating = rng.choice([4, 5]) if is_positive else rng.choice([1, 2, 3])
            text = (
                f"great product works well number {review_id}"
                if is_positive
                else f"disappointing product broke fast number {review_id}"
            )
            rows.append(
                {
                    "unique_review_id": review_id,
                    "comments": text,
                    "rating": rating,
                    "product_link_id": group,
                }
            )
            review_id += 1
    return pd.DataFrame(rows)


def _prepared_synthetic_reviews() -> pd.DataFrame:
    return prepare_review_data(
        _synthetic_raw_reviews(),
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )


def test_prepare_review_data_carries_review_id_from_unique_review_id_column():
    prepared = _prepared_synthetic_reviews()

    assert "review_id" in prepared.columns
    assert prepared["review_id"].is_unique
    assert set(prepared["review_id"]) <= {str(i) for i in range(1000)}


def test_prepare_review_data_generates_review_id_when_id_column_absent():
    raw = _reviews().rename(columns={})  # no unique_review_id column present
    prepared = prepare_review_data(
        raw,
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )

    assert "review_id" in prepared.columns
    assert prepared["review_id"].is_unique
    assert prepared["review_id"].notna().all()


def test_summarize_review_data_quality_reports_missing_invalid_and_duplicates():
    raw = pd.DataFrame(
        {
            "comments": ["Great!", "Great!", None, "Bad product", "  "],
            "rating": [5, 5, 4, "not-a-number", 3],
            "product_link_id": [1, 1, 2, 2, 3],
        }
    )

    summary = summarize_review_data_quality(
        raw,
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )

    assert summary["total_rows"] == 5
    # Missing text: index 2 (None) and index 4 ("  ", blank after stripping)
    assert summary["missing_text_rate"] == pytest.approx(2 / 5)
    assert summary["invalid_rating_rate"] == pytest.approx(1 / 5)
    # Usable: text present and rating valid -> indices 0, 1
    assert summary["usable_rows"] == 2
    assert summary["exact_duplicate_text_rate"] == pytest.approx(1 / 2)


def test_find_duplicate_review_clusters_groups_normalized_duplicates():
    reviews = pd.DataFrame(
        {
            "review_text": [
                "Great Product!!",
                "great product",
                "totally different review",
            ],
        }
    )

    clusters = find_duplicate_review_clusters(reviews)

    assert clusters.iloc[0] == clusters.iloc[1]
    assert clusters.iloc[2] != clusters.iloc[0]


def test_build_leakage_safe_groups_unions_products_sharing_duplicate_text():
    reviews = pd.DataFrame(
        {
            "review_text": [
                "this exact templated review text was copy pasted everywhere",
                "this exact templated review text was copy pasted everywhere",
                "a completely different and unrelated review here",
            ],
            "product_link_id": [1, 2, 3],
        }
    )

    groups = build_leakage_safe_groups(reviews)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[2] != groups.iloc[0]


def test_build_leakage_safe_groups_ignores_short_generic_duplicate_text():
    # Short generic phrases like "Love it!!" are independently written by
    # many reviewers across unrelated products; unioning on them would
    # collapse most of the catalogue into one giant component instead of
    # protecting against real leakage (see module docstring / regression
    # found while running the full experiment on the real Luxxify export).
    reviews = pd.DataFrame(
        {
            "review_text": ["Love it!!"] * 5,
            "product_link_id": [1, 2, 3, 4, 5],
        }
    )

    groups = build_leakage_safe_groups(reviews)

    assert groups.nunique() == 5


def test_product_disjoint_three_way_split_has_no_group_leakage_and_saves_manifest():
    prepared = _prepared_synthetic_reviews()

    result = product_disjoint_three_way_split(
        prepared,
        split_group_column="product_link_id",
        test_fraction=0.2,
        validation_fraction=0.2,
        random_state=0,
    )

    splits = result["splits"]
    train_groups = set(splits["train"]["groups"])
    val_groups = set(splits["validation"]["groups"])
    test_groups = set(splits["test"]["groups"])

    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    total_rows = sum(len(data["y"]) for data in splits.values())
    assert total_rows == len(prepared)

    manifest = result["manifest"]
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert len(manifest) == len(prepared)
    assert manifest["review_id"].is_unique

    for name in ("train", "validation", "test"):
        assert 0.0 <= result["proportions"][name]["positive_rate"] <= 1.0


def test_product_disjoint_three_way_split_rejects_invalid_fractions():
    prepared = _prepared_synthetic_reviews()

    with pytest.raises(ValueError, match="test_fraction"):
        product_disjoint_three_way_split(
            prepared, test_fraction=0.7, validation_fraction=0.5
        )


def test_build_majority_baseline_predicts_constant_majority_class():
    prepared = _prepared_synthetic_reviews()
    baseline = build_majority_baseline()

    baseline.fit(prepared[["review_text"]], prepared["positive_review"])
    predictions = baseline.predict(prepared[["review_text"]])

    majority_class = prepared["positive_review"].mode().iloc[0]
    assert set(predictions) == {majority_class}


def test_evaluate_review_classifier_handles_constant_probabilities_from_dummy():
    prepared = _prepared_synthetic_reviews()
    baseline = build_majority_baseline()
    baseline.fit(prepared[["review_text"]], prepared["positive_review"])

    metrics, report, matrix = evaluate_review_classifier(
        baseline, prepared[["review_text"]], prepared["positive_review"]
    )

    assert "macro_f1" in metrics.index
    # ROC AUC/average precision are undefined for a constant-probability
    # model; this must not raise and should report NaN instead.
    assert np.isnan(metrics["roc_auc"]) or 0.0 <= metrics["roc_auc"] <= 1.0
    assert matrix.shape == (2, 2)


def test_select_logistic_regression_config_picks_best_val_macro_f1():
    prepared = _prepared_synthetic_reviews()
    split = product_disjoint_three_way_split(prepared, random_state=0)
    train, val = split["splits"]["train"], split["splits"]["validation"]

    result = select_logistic_regression_config(
        train["X"],
        train["y"],
        val["X"],
        val["y"],
        C_values=(0.1, 1.0),
        class_weight_options=(None, "balanced"),
        minimum_document_frequency=1,
        max_iter=200,
    )

    assert len(result["results"]) == 4
    assert "converged" in result["results"].columns
    best_row = result["results"].iloc[0]
    assert best_row["val_macro_f1"] == result["results"]["val_macro_f1"].max()
    assert hasattr(result["best_pipeline"], "predict")


def test_detect_review_languages_and_summarize_language_check_on_english_text():
    pytest.importorskip("langdetect")
    texts = pd.Series(
        [
            "This foundation matches my skin tone perfectly and lasts all day.",
            "I really love how lightweight this feels on my face every morning.",
            "The coverage is great but it broke me out after a few days of use.",
        ]
        * 5
    )

    table = detect_review_languages(texts, sample_size=None, random_state=0)
    check = summarize_language_check(table)

    assert "language" in table.columns
    assert check["predominantly_english"] is True
    assert check["recommended_checkpoint"] == "distilbert-base-uncased"


def test_rebalance_training_set_undersample_and_oversample():
    from foundation_matcher.review_model import rebalance_training_set

    train = pd.DataFrame({"positive_review": [1] * 8 + [0] * 2, "x": range(10)})
    under = rebalance_training_set(train, strategy="undersample", random_state=0)
    assert under["positive_review"].value_counts().to_dict() == {1: 2, 0: 2}
    assert set(under.loc[under["positive_review"] == 0, "x"]) == {8, 9}
    over = rebalance_training_set(train, strategy="oversample", random_state=0)
    assert over["positive_review"].value_counts().to_dict() == {1: 8, 0: 8}
    assert set(over.loc[over["positive_review"] == 1, "x"]) == set(range(8))
    for strategy in ("none", "class_weight"):
        assert rebalance_training_set(train, strategy=strategy).equals(train)


def test_rebalance_training_set_rejects_bad_input():
    from foundation_matcher.review_model import rebalance_training_set

    train = pd.DataFrame({"positive_review": [1, 1]})
    with pytest.raises(ValueError):
        rebalance_training_set(train, strategy="smote")
    with pytest.raises(ValueError):
        rebalance_training_set(train, strategy="undersample")


def test_placeholder_brands_become_missing_in_metadata_and_summary():
    from foundation_matcher.review_model import (
        attach_product_metadata,
        summarize_luxxify_products,
    )

    products = pd.DataFrame(
        {
            "product_link_id": ["a", "b", "c", "d"],
            "category": ["Foundation"] * 4,
            "price": [10.0, 20.0, 30.0, 40.0],
            "brand": ["Ask A Question", "Find your shade", "Tarte", None],
            "num_shades": [None] * 4,
        }
    )
    reviews = pd.DataFrame({"product_link_id": ["a", "b", "c", "d"], "x": range(4)})
    merged = attach_product_metadata(reviews, products)
    assert merged["brand"].isna().tolist() == [True, True, False, True]
    summary = summarize_luxxify_products(products)
    assert summary["brand_missing_rate"] == 0.25
    assert summary["brand_placeholder_rate"] == 0.5
    assert summary["brand_missing_or_placeholder_rate"] == 0.75
