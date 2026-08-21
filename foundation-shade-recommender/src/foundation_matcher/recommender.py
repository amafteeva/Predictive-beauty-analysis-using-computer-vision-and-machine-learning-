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

    return results.head(top_n)[RECOMMENDATION_COLUMNS].reset_index(drop=True)


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
