"""Optional review-text satisfaction-classification experiment."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from foundation_matcher.config import RANDOM_STATE
from foundation_matcher.data import clean_brand_placeholders

_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9\s]")


def summarize_luxxify_products(products: pd.DataFrame) -> dict[str, float]:
    """Summarize the raw Luxxify product export before any cleaning.

    Surfaces column-level coverage that attach_product_metadata doesn't
    need to know about, but that matters for judging how reliable this
    export is as a source of product context: e.g. num_shades is missing
    for the vast majority of rows, and brand for roughly a third (over half
    once scraped page-button labels such as "Ask A Question" are counted).
    """

    return {
        "total_products": float(len(products)),
        "total_categories": float(products["category"].nunique()),
        "category_missing_rate": round(float(products["category"].isna().mean()), 3),
        "brand_missing_rate": round(float(products["brand"].isna().mean()), 3),
        # Scraped page-button labels ("Ask A Question", ...) are not brands.
        "brand_placeholder_rate": round(
            float(
                (
                    clean_brand_placeholders(products["brand"]).isna()
                    & products["brand"].notna()
                ).mean()
            ),
            3,
        ),
        "brand_missing_or_placeholder_rate": round(
            float(clean_brand_placeholders(products["brand"]).isna().mean()), 3
        ),
        "price_missing_rate": round(float(products["price"].isna().mean()), 3),
        "num_shades_missing_rate": round(float(products["num_shades"].isna().mean()), 3),
        "price_median": round(float(products["price"].median()), 2),
    }


def summarize_luxxify_reviews(reviews: pd.DataFrame) -> dict[str, float]:
    """Summarize the raw Luxxify review export before any cleaning.

    prepare_review_data silently drops every row with no review text; this
    shows how much of the raw export that actually is, and how unevenly
    review volume is spread across products.
    """

    text_missing_rate = float(reviews["comments"].isna().mean())
    lengths = reviews["comments"].dropna().str.split().str.len()
    reviews_per_product = reviews.groupby("product_link_id").size()
    verified_rate = float((reviews["is_verified_buyer"] == "t").mean())

    return {
        "total_reviews": float(len(reviews)),
        "reviews_with_text": float(reviews["comments"].notna().sum()),
        "text_missing_rate": round(text_missing_rate, 3),
        "median_review_word_count": float(lengths.median()),
        "verified_buyer_rate": round(verified_rate, 3),
        "median_reviews_per_product": float(reviews_per_product.median()),
        "max_reviews_per_product": float(reviews_per_product.max()),
    }


def prepare_review_data(
    reviews: pd.DataFrame,
    *,
    text_column: str | None = None,
    rating_column: str | None = None,
    group_column: str | None = None,
    id_column: str | None = None,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Create a clean text dataset and a binary satisfaction target.

    ``positive_review`` is a rating-derived label (rating >= threshold),
    not a measurement of personalized customer satisfaction -- a review's
    text and its star rating both reflect the same underlying opinion, so
    this is best read as review-text classification, not a satisfaction
    predictor.

    A stable ``review_id`` is carried through so downstream splitting can
    save exactly which rows landed in which split. When ``id_column`` is
    not given, the export's own ``unique_review_id`` primary key is used if
    present; otherwise an id is assigned from the row's position in
    ``reviews`` before any filtering, which is reproducible for a fixed
    input but not a stable identity across different raw exports.
    """

    text_column = _resolve_column(
        reviews,
        text_column,
        ("review_text", "review", "text", "review_body", "comments", "comment", "body"),
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

    has_id_column = id_column is not None or "unique_review_id" in reviews.columns
    if id_column is None and "unique_review_id" in reviews.columns:
        id_column = "unique_review_id"
    if id_column is not None and id_column not in reviews.columns:
        raise ValueError(f"Requested review id column not found: {id_column}")

    columns = [text_column, rating_column, group_column]
    renames = {
        text_column: "review_text",
        rating_column: "rating",
        group_column: "product_link_id",
    }
    if has_id_column:
        columns.append(id_column)
        renames[id_column] = "review_id"

    prepared = reviews[columns].copy().rename(columns=renames)
    if not has_id_column:
        prepared["review_id"] = np.arange(len(reviews))
    prepared["review_id"] = prepared["review_id"].astype("string")

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


def attach_product_metadata(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    *,
    group_column: str = "product_link_id",
    product_id_column: str | None = None,
    metadata_columns: tuple[str, ...] = ("category", "price", "brand"),
) -> pd.DataFrame:
    """Attach per-product metadata (price, category, brand) to reviews.

    A review's text is all `prepare_review_data` keeps, but the Luxxify
    export also has a separate per-product table with this context. `price`
    and `category` are populated for effectively every product; `num_shades`
    is missing for over 95% of rows in this export and is deliberately not
    included here since it carries almost no signal.
    """

    product_id_column = product_id_column or group_column
    required = {product_id_column, *metadata_columns}
    missing = sorted(required.difference(products.columns))
    if missing:
        raise ValueError(f"Product data is missing required columns: {missing}")

    metadata = (
        products[[product_id_column, *metadata_columns]]
        .drop_duplicates(subset=[product_id_column])
        .rename(columns={product_id_column: group_column})
    )
    if "brand" in metadata.columns:
        # Scraped page-button labels must not become a fake "brand" category.
        metadata["brand"] = clean_brand_placeholders(metadata["brand"])
    return reviews.merge(metadata, on=group_column, how="left")


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
    feature_columns: tuple[str, ...] | None = None,
    text_column: str = "review_text",
    target_column: str = "positive_review",
    group_column: str = "product_link_id",
    random_state: int = RANDOM_STATE,
):
    """Make an approximately 80/20 split with no product leakage.

    ``feature_columns`` selects which columns travel through the split as a
    DataFrame (defaults to just ``text_column``). Pass the review text column
    plus any attached product-metadata columns to train on both.
    """

    if feature_columns is None:
        feature_columns = (text_column,)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )
    features = reviews[list(feature_columns)]
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
    text_column: str = "review_text",
    categorical_columns: tuple[str, ...] = (),
    numeric_columns: tuple[str, ...] = (),
    C: float = 1.0,
    class_weight: str | None = "balanced",
    max_iter: int = 300,
) -> Pipeline:
    """Create a TF-IDF (+ optional product-metadata) logistic-regression baseline.

    With no ``categorical_columns``/``numeric_columns``, this trains on review
    text alone, as before. Passing e.g. ``categorical_columns=("category",
    "brand")`` and ``numeric_columns=("price",)`` combines that product
    context with the text features via a ColumnTransformer.

    ``C``, ``class_weight``, and ``max_iter`` are exposed so a caller (e.g.
    ``select_logistic_regression_config``) can sweep regularization strength
    and imbalance handling; the defaults reproduce the original fixed
    configuration.
    """

    transformers = [
        (
            "text",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=minimum_document_frequency,
                max_features=max_features,
                sublinear_tf=True,
            ),
            text_column,
        )
    ]
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_columns),
            )
        )
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric_columns),
            )
        )

    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    solver="saga",
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_review_classifier(
    model,
    test_text: pd.Series,
    test_target: pd.Series,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Return imbalance-aware metrics, a report, and a confusion matrix.

    Works for any fitted scikit-learn-compatible classifier with ``predict``
    (and, when available, ``predict_proba``) -- the TF-IDF/logistic-regression
    pipeline, ``build_majority_baseline``'s ``DummyClassifier``, or a
    ``DistilBertSklearnAdapter`` (see ``review_transformer``) -- so the same
    function benchmarks every model in the comparison. ROC AUC and average
    precision are reported as ``NaN`` when a model (e.g. the majority-class
    baseline) produces constant scores for which they are undefined.
    """

    predicted = model.predict(test_text)
    roc_auc = np.nan
    average_precision = np.nan
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_text)[:, 1]
        try:
            roc_auc = roc_auc_score(test_target, probabilities)
            average_precision = average_precision_score(test_target, probabilities)
        except ValueError:
            pass
    metrics = pd.Series(
        {
            "accuracy": accuracy_score(test_target, predicted),
            "balanced_accuracy": balanced_accuracy_score(test_target, predicted),
            "macro_f1": f1_score(
                test_target, predicted, average="macro", zero_division=0
            ),
            "precision": precision_score(test_target, predicted, zero_division=0),
            "recall": recall_score(test_target, predicted, zero_division=0),
            "f1": f1_score(test_target, predicted, zero_division=0),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
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


def summarize_review_data_quality(
    raw_reviews: pd.DataFrame,
    *,
    text_column: str | None = None,
    rating_column: str | None = None,
    group_column: str | None = None,
    positive_threshold: float = 4.0,
    valid_rating_range: tuple[float, float] = (1.0, 5.0),
) -> dict[str, float]:
    """Report the data-quality issues that matter before splitting or training.

    Runs on the raw export, before `prepare_review_data` silently drops
    rows, so it shows what is actually being discarded and why: missing
    text, ratings that don't parse or fall outside the valid range, class
    balance among the rows that would survive, and how much of the
    text-bearing data is an exact duplicate once case/punctuation/whitespace
    are normalized away. `find_duplicate_review_clusters` reuses the same
    normalization to group duplicates for splitting.
    """

    text_column = _resolve_column(
        raw_reviews,
        text_column,
        ("review_text", "review", "text", "review_body", "comments", "comment", "body"),
        "review text",
    )
    rating_column = _resolve_column(
        raw_reviews,
        rating_column,
        ("rating", "review_rating", "stars", "star_rating", "review_stars"),
        "rating",
    )
    group_column = _resolve_column(
        raw_reviews,
        group_column,
        ("product_link_id", "item_id", "product_id"),
        "product identifier",
    )

    text = raw_reviews[text_column].astype("string").str.strip()
    text_missing = text.isna() | (text.str.len() == 0)

    numeric_rating = pd.to_numeric(raw_reviews[rating_column], errors="coerce")
    invalid_rating = (
        numeric_rating.isna()
        | (numeric_rating < valid_rating_range[0])
        | (numeric_rating > valid_rating_range[1])
    )

    usable = ~text_missing & ~invalid_rating
    usable_text = text[usable]
    usable_rating = numeric_rating[usable]

    normalized = usable_text.map(_normalize_review_text)
    duplicate_rate = float(normalized.duplicated().mean()) if len(normalized) else 0.0
    unique_normalized_clusters = int(normalized.nunique()) if len(normalized) else 0

    positive_rate = (
        float((usable_rating >= positive_threshold).mean())
        if len(usable_rating)
        else float("nan")
    )

    return {
        "total_rows": float(len(raw_reviews)),
        "missing_text_rate": round(float(text_missing.mean()), 4),
        "invalid_rating_rate": round(float(invalid_rating.mean()), 4),
        "usable_rows": float(usable.sum()),
        "positive_class_rate": (
            round(positive_rate, 4) if not np.isnan(positive_rate) else float("nan")
        ),
        "negative_class_rate": (
            round(1 - positive_rate, 4) if not np.isnan(positive_rate) else float("nan")
        ),
        "exact_duplicate_text_rate": round(duplicate_rate, 4),
        "unique_normalized_texts": float(unique_normalized_clusters),
        "unique_products": float(raw_reviews.loc[usable, group_column].nunique()),
    }


def _normalize_review_text(text: object) -> str:
    """Lowercase and strip punctuation/whitespace for duplicate detection."""

    lowered = str(text).lower()
    stripped = _NORMALIZE_PATTERN.sub(" ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def find_duplicate_review_clusters(
    reviews: pd.DataFrame, *, text_column: str = "review_text"
) -> pd.Series:
    """Assign a cluster id to rows with identical normalized text.

    Two reviews can carry different ids -- and even sit under different
    products -- yet be the same boilerplate or copy-pasted text. If one
    lands in the training split and its near-duplicate in the test split,
    a model can look like it generalizes when it has really just
    memorized. Normalization only strips case, punctuation, and whitespace,
    so this catches exact and near-exact duplicates, not paraphrases;
    `build_leakage_safe_groups` uses the result to keep such rows together.
    ``sort=True`` makes cluster ids depend only on the text values, not row
    order, so they are reproducible across runs and datasets.
    """

    normalized = reviews[text_column].astype(str).map(_normalize_review_text)
    codes, _ = pd.factorize(normalized, sort=True)
    return pd.Series(codes, index=reviews.index, name="duplicate_cluster")


def build_leakage_safe_groups(
    reviews: pd.DataFrame,
    *,
    group_column: str = "product_link_id",
    text_column: str = "review_text",
    minimum_shared_words: int = 6,
) -> pd.Series:
    """Union product groups that share duplicate/near-duplicate review text.

    A product-disjoint split alone still leaks if the identical text of a
    templated or copy-pasted review shows up under two different products --
    one copy in training, one in test. This unions any products connected by
    a shared duplicate-text cluster (`find_duplicate_review_clusters`) into
    one combined split group via union-find, so a split on the result stays
    leakage-safe even across products. Unaffected products keep their own
    `group_column` value as their split group.

    Only duplicate text with at least `minimum_shared_words` normalized
    words is used to union products. Short generic phrases ("Love it!!",
    "Great product") are written independently by many unrelated reviewers
    and are extremely common across completely unrelated products -- in the
    Luxxify export, "Love it!!" alone appears under 70 different products.
    Unioning on text that short collapses most of the catalogue into one
    giant component, which defeats product-disjoint splitting rather than
    protecting it. Longer shared text is far more likely to be a genuine
    copy-pasted or templated duplicate, which is what this guards against.
    """

    clusters = find_duplicate_review_clusters(reviews, text_column=text_column)
    groups = reviews[group_column].astype(str).to_numpy()
    def _word_count(text: str) -> int:
        return len(_normalize_review_text(text).split())

    word_counts = reviews[text_column].astype(str).map(_word_count)
    distinctive = (word_counts >= minimum_shared_words).to_numpy()

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            # Always attach the lexicographically larger root under the
            # smaller one, so the result doesn't depend on row order.
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    cluster_frame = pd.DataFrame(
        {"cluster": clusters.to_numpy(), "group": groups, "distinctive": distinctive}
    )
    distinctive_rows = cluster_frame[cluster_frame["distinctive"]]
    for _, cluster_rows in distinctive_rows.groupby("cluster"):
        unique_groups = pd.unique(cluster_rows["group"])
        if len(unique_groups) > 1:
            first = str(unique_groups[0])
            for other in unique_groups[1:]:
                union(first, str(other))

    resolved = pd.Series(
        [find(group) for group in groups], index=reviews.index, name="split_group"
    )
    return resolved


def detect_review_languages(
    texts: pd.Series,
    *,
    sample_size: int | None = 5_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Detect each review's language and summarize the distribution.

    A pretrained English DistilBERT checkpoint assumes English input;
    fine-tuning it on predominantly non-English text would silently
    degrade it below even the TF-IDF baseline instead of failing loudly.
    This is checked before any model is trained. Detection is randomized
    internally by `langdetect`, so `DetectorFactory.seed` is fixed for
    reproducible results; up to `sample_size` rows are sampled for speed on
    large datasets (pass `sample_size=None` to check every row).

    Requires the optional `langdetect` package.
    """

    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = random_state

    sample = texts.dropna()
    if sample_size is not None and len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=random_state)

    def safe_detect(text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

    languages = sample.map(safe_detect)
    counts = languages.value_counts()
    table = counts.rename_axis("language").reset_index(name="count")
    table["proportion"] = (table["count"] / table["count"].sum()).round(4)
    return table.sort_values("count", ascending=False).reset_index(drop=True)


def summarize_language_check(
    language_table: pd.DataFrame, *, english_threshold: float = 0.9
) -> dict[str, object]:
    """Decide whether an English DistilBERT checkpoint is appropriate.

    Reads the table from `detect_review_languages` and returns a
    recommendation plus the evidence behind it, so the reasoning can be
    logged and reported rather than assumed.
    """

    english_proportion = float(
        language_table.loc[language_table["language"] == "en", "proportion"].sum()
    )
    predominantly_english = english_proportion >= english_threshold

    if predominantly_english:
        explanation = (
            f"{english_proportion:.1%} of sampled reviews were detected as English, "
            f"at or above the {english_threshold:.0%} threshold, so an English "
            "DistilBERT checkpoint (distilbert-base-uncased) is appropriate."
        )
    else:
        explanation = (
            f"Only {english_proportion:.1%} of sampled reviews were detected as "
            f"English, below the {english_threshold:.0%} threshold. Fine-tuning an "
            "English-only checkpoint on this mix would systematically "
            "under-tokenize and misclassify the non-English share. A multilingual "
            "checkpoint (e.g. distilbert-base-multilingual-cased) or per-language "
            "routing would be needed instead; this project does not select one "
            "automatically."
        )

    return {
        "english_proportion": round(english_proportion, 4),
        "english_threshold": english_threshold,
        "predominantly_english": predominantly_english,
        "recommended_checkpoint": (
            "distilbert-base-uncased" if predominantly_english else None
        ),
        "explanation": explanation,
    }


def product_disjoint_three_way_split(
    reviews: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    text_column: str = "review_text",
    target_column: str = "positive_review",
    split_group_column: str = "product_link_id",
    id_column: str = "review_id",
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Make a reproducible, leakage-safe train/validation/test split.

    Splitting is by `split_group_column` -- pass `product_link_id` for a
    plain product-disjoint split, or the output of
    `build_leakage_safe_groups` to also keep duplicate-text clusters intact
    across products -- using two rounds of `StratifiedGroupKFold` so no
    group crosses a split boundary: a test fold is held out first, then a
    validation fold is carved from what remains. Because whole groups move
    together and group sizes vary, the realized split sizes and class
    balance will not exactly match `test_fraction`/`validation_fraction`;
    that is expected, which is why the return value reports the realized
    `proportions` rather than assuming the requested ones were hit.

    Returns a dict with per-split ``X``/``y``/``groups``/``ids`` under
    ``splits``, a long-format ``manifest`` DataFrame (``review_id``,
    ``split``) suitable for saving to disk so every model reuses the exact
    same rows, and the realized ``proportions``.
    """

    if feature_columns is None:
        feature_columns = (text_column,)
    if not (0 < test_fraction < 1) or not (0 < validation_fraction < 1):
        raise ValueError("test_fraction and validation_fraction must be in (0, 1).")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be below 1.")

    features = reviews[list(feature_columns)]
    target = reviews[target_column]
    groups = reviews[split_group_column]

    test_splits = max(2, round(1 / test_fraction))
    test_splitter = StratifiedGroupKFold(
        n_splits=test_splits, shuffle=True, random_state=random_state
    )
    remainder_idx, test_idx = next(test_splitter.split(features, target, groups))

    remainder_fraction = 1 - test_fraction
    val_of_remainder = validation_fraction / remainder_fraction
    val_splits = max(2, round(1 / val_of_remainder))
    val_splitter = StratifiedGroupKFold(
        n_splits=val_splits, shuffle=True, random_state=random_state
    )
    remainder_features = features.iloc[remainder_idx]
    remainder_target = target.iloc[remainder_idx]
    remainder_groups = groups.iloc[remainder_idx]
    train_sub_idx, val_sub_idx = next(
        val_splitter.split(remainder_features, remainder_target, remainder_groups)
    )
    train_idx = remainder_idx[train_sub_idx]
    val_idx = remainder_idx[val_sub_idx]

    has_ids = id_column in reviews.columns

    def subset(indices: np.ndarray) -> dict[str, object]:
        return {
            "X": features.iloc[indices],
            "y": target.iloc[indices],
            "groups": groups.iloc[indices],
            "ids": (
                reviews[id_column].iloc[indices]
                if has_ids
                else pd.Series(indices, name="review_id").astype("string")
            ),
        }

    splits = {
        "train": subset(train_idx),
        "validation": subset(val_idx),
        "test": subset(test_idx),
    }

    train_groups = set(splits["train"]["groups"])
    val_groups = set(splits["validation"]["groups"])
    test_groups = set(splits["test"]["groups"])
    any_overlap = (
        (train_groups & val_groups)
        or (train_groups & test_groups)
        or (val_groups & test_groups)
    )
    if any_overlap:
        raise AssertionError("Split groups leaked across train/validation/test.")

    manifest = pd.concat(
        [
            pd.DataFrame({"review_id": data["ids"].to_numpy(), "split": name})
            for name, data in splits.items()
        ],
        ignore_index=True,
    )

    proportions = {
        name: {
            "rows": int(len(data["y"])),
            "positive_rate": round(float(data["y"].mean()), 4),
            "split_groups": int(data["groups"].nunique()),
        }
        for name, data in splits.items()
    }

    return {
        "splits": splits,
        "manifest": manifest,
        "proportions": proportions,
        "random_state": random_state,
    }


BALANCING_STRATEGIES = ("none", "class_weight", "undersample", "oversample")


def rebalance_training_set(
    train: pd.DataFrame,
    *,
    label_column: str = "positive_review",
    strategy: str = "undersample",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a class-balanced copy of a TRAINING frame by random resampling.

    ``"undersample"`` keeps a random subset of every larger class equal to the
    smallest class's size; ``"oversample"`` duplicates random rows of every
    smaller class up to the largest class's size. ``"none"`` and
    ``"class_weight"`` return the frame unchanged (class weighting is applied
    by the classifier instead). Only ever apply this to training rows:
    validation and test rows must keep the real class mix, or every reported
    metric describes a population that does not exist.
    """

    if strategy not in BALANCING_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {BALANCING_STRATEGIES}, got {strategy!r}"
        )
    if strategy in ("none", "class_weight"):
        return train.copy()
    counts = train[label_column].value_counts()
    if len(counts) < 2:
        raise ValueError("Training data must contain at least two classes.")
    target = counts.min() if strategy == "undersample" else counts.max()
    parts = []
    for label, size in counts.items():
        rows = train[train[label_column] == label]
        replace = strategy == "oversample" and size < target
        parts.append(
            rows.sample(n=int(target), replace=replace, random_state=random_state)
        )
    balanced = pd.concat(parts)
    return balanced.sample(frac=1.0, random_state=random_state)


def build_majority_baseline(*, random_state: int = RANDOM_STATE) -> DummyClassifier:
    """The simplest possible baseline: always predict the majority class.

    A learned model's accuracy/macro-F1 is only meaningful relative to this;
    if TF-IDF+logistic-regression or DistilBERT can't beat it by a clear
    margin, the review text isn't adding real predictive signal.
    """

    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def select_logistic_regression_config(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    text_column: str = "review_text",
    categorical_columns: tuple[str, ...] = (),
    numeric_columns: tuple[str, ...] = (),
    C_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    class_weight_options: tuple[str | None, ...] = (None, "balanced"),
    max_features: int = 40_000,
    minimum_document_frequency: int = 5,
    max_iter: int = 300,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Grid-search regularization strength and class weighting on validation macro-F1.

    Each `(C, class_weight)` combination is fit on `X_train`/`y_train` and
    scored on `X_val`/`y_val` by macro-F1, which weighs both classes
    equally regardless of the training-set imbalance. Convergence is
    checked via each fit's `n_iter_` against `max_iter`: a run that
    saturates `max_iter` is flagged rather than silently accepted, since its
    coefficients may not reflect a converged optimum.
    """

    if not C_values:
        raise ValueError("C_values must not be empty.")
    if not class_weight_options:
        raise ValueError("class_weight_options must not be empty.")

    rows = []
    best: dict[str, object] | None = None
    for C in C_values:
        for class_weight in class_weight_options:
            pipeline = build_review_classifier(
                max_features=max_features,
                minimum_document_frequency=minimum_document_frequency,
                random_state=random_state,
                text_column=text_column,
                categorical_columns=categorical_columns,
                numeric_columns=numeric_columns,
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
            )
            pipeline.fit(X_train, y_train)

            val_predicted = pipeline.predict(X_val)
            val_macro_f1 = f1_score(
                y_val, val_predicted, average="macro", zero_division=0
            )
            n_iter = int(np.max(pipeline.named_steps["classifier"].n_iter_))
            converged = n_iter < max_iter

            row = {
                "C": C,
                "class_weight": class_weight if class_weight is not None else "none",
                "val_macro_f1": round(float(val_macro_f1), 4),
                "n_iter": n_iter,
                "converged": converged,
            }
            rows.append(row)

            if best is None or val_macro_f1 > best["val_macro_f1"]:
                best = {
                    **row,
                    "val_macro_f1": float(val_macro_f1),
                    "pipeline": pipeline,
                }

    results = (
        pd.DataFrame(rows)
        .sort_values("val_macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    assert best is not None
    best_config = {key: value for key, value in best.items() if key != "pipeline"}
    return {
        "best_pipeline": best["pipeline"],
        "best_config": best_config,
        "results": results,
    }
