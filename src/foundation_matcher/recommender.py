"""Perceptual matching and optional K-Means shade clustering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from skimage.color import deltaE_ciede2000

from foundation_matcher.config import LAB_COLUMNS, RANDOM_STATE

RECOMMENDATION_COLUMNS = [
    "brand",
    "product",
    "hex",
    *LAB_COLUMNS,
    "color_distance",
]
# Included only when present on the input catalogue (e.g. after
# attach_price_estimates), so callers without it are unaffected.
OPTIONAL_RECOMMENDATION_COLUMNS = ["price"]


def recommend_foundations(
    skin_lab: Iterable[float] | np.ndarray,
    products: pd.DataFrame,
    *,
    top_n: int = 5,
    unique_products: bool = True,
) -> pd.DataFrame:
    """Rank catalogue products by CIEDE2000 distance from a skin estimate."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    results = products.copy()
    product_lab = results[LAB_COLUMNS].to_numpy(dtype=float)
    skin_lab_array = np.asarray(skin_lab, dtype=float).reshape(1, 3)
    results["color_distance"] = deltaE_ciede2000(product_lab, skin_lab_array)
    results = results.sort_values("color_distance")

    if unique_products:
        results = results.drop_duplicates(subset=["brand", "product"])

    output_columns = [column for column in RECOMMENDATION_COLUMNS if column != "color_distance"]
    output_columns += [
        column for column in OPTIONAL_RECOMMENDATION_COLUMNS if column in results.columns
    ]
    output_columns.append("color_distance")

    return results.head(top_n)[output_columns].reset_index(drop=True)


def evaluate_cluster_counts(
    products: pd.DataFrame,
    cluster_counts: Iterable[int] = range(2, 13),
    *,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare K-Means solutions with silhouette and Davies-Bouldin scores."""

    features = products[LAB_COLUMNS].to_numpy(dtype=float)
    rows = []
    for cluster_count in cluster_counts:
        model = KMeans(
            n_clusters=int(cluster_count),
            random_state=random_state,
            n_init=20,
        )
        labels = model.fit_predict(features)
        rows.append(
            {
                "clusters": int(cluster_count),
                "silhouette_score": silhouette_score(features, labels),
                "davies_bouldin_score": davies_bouldin_score(features, labels),
            }
        )
    return pd.DataFrame(rows)


def fit_shade_clusters(
    products: pd.DataFrame,
    number_of_clusters: int,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[KMeans, pd.DataFrame]:
    """Fit K-Means and return both the model and a labelled catalogue copy."""

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=random_state,
        n_init=20,
    )
    labelled_products = products.copy()
    labelled_products["shade_cluster"] = model.fit_predict(
        labelled_products[LAB_COLUMNS].to_numpy(dtype=float)
    )
    return model, labelled_products


def recommend_foundations_clustered(
    skin_lab: Iterable[float] | np.ndarray,
    products: pd.DataFrame,
    cluster_model: KMeans,
    *,
    top_n: int = 5,
) -> tuple[int, pd.DataFrame]:
    """Restrict candidates to the predicted K-Means cluster, then rank by Delta E."""

    if "shade_cluster" not in products:
        raise ValueError("products must include a 'shade_cluster' column.")

    skin_lab_array = np.asarray(skin_lab, dtype=float).reshape(1, 3)
    predicted_cluster = int(cluster_model.predict(skin_lab_array)[0])
    candidates = products.loc[products["shade_cluster"] == predicted_cluster]
    recommendations = recommend_foundations(
        skin_lab_array,
        candidates,
        top_n=top_n,
    )
    recommendations.insert(3, "shade_cluster", predicted_cluster)
    return predicted_cluster, recommendations


def evaluate_cluster_recommendation_impact(
    products: pd.DataFrame,
    cluster_counts: Iterable[int] = range(2, 13),
    *,
    number_of_queries: int = 300,
    noise_scale: tuple[float, float, float] = (3.0, 2.0, 2.0),
    perceptible_delta_e: float = 2.0,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Measure how much cluster-restricted ranking costs vs. unrestricted ranking.

    evaluate_cluster_counts only scores cluster geometry (silhouette,
    Davies-Bouldin); it says nothing about whether recommend_foundations_
    clustered's candidate restriction actually degrades the recommendation a
    user would see. This simulates realistic skin-tone queries (real
    catalogue shades plus lighting/camera noise, the same noise model as
    simulate_colour_robustness) and compares the clustered top-1 match
    against the unrestricted top-1 match for each candidate K. Restricting
    to a cluster can only ever match or lose to the unrestricted search, so
    every loss value is >= 0.
    """

    if number_of_queries < 1:
        raise ValueError("number_of_queries must be at least 1.")

    product_lab = products[LAB_COLUMNS].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    query_indices = rng.integers(0, len(product_lab), size=number_of_queries)
    queries = product_lab[query_indices] + rng.normal(
        0, noise_scale, size=(number_of_queries, 3)
    )

    rows = []
    for cluster_count in cluster_counts:
        cluster_model, clustered_products = fit_shade_clusters(
            products, int(cluster_count), random_state=random_state
        )
        losses = np.empty(number_of_queries)
        mismatches = 0
        for row_index, query_lab in enumerate(queries):
            global_row = recommend_foundations(query_lab, products, top_n=1).iloc[0]
            _, clustered_match = recommend_foundations_clustered(
                query_lab, clustered_products, cluster_model, top_n=1
            )
            clustered_row = clustered_match.iloc[0]
            losses[row_index] = (
                clustered_row["color_distance"] - global_row["color_distance"]
            )
            if (global_row["brand"], global_row["product"], global_row["hex"]) != (
                clustered_row["brand"],
                clustered_row["product"],
                clustered_row["hex"],
            ):
                mismatches += 1

        rows.append(
            {
                "clusters": int(cluster_count),
                "top1_mismatch_rate": mismatches / number_of_queries,
                "mean_delta_e_loss": float(losses.mean()),
                "median_delta_e_loss": float(np.median(losses)),
                "max_delta_e_loss": float(losses.max()),
                "perceptible_loss_rate": float(
                    np.mean(losses > perceptible_delta_e)
                ),
            }
        )

    return pd.DataFrame(rows)
