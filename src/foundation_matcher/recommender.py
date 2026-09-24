"""Brute-force perceptual shade matching."""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
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
    """Rank catalogue products by CIEDE2000 distance from a skin estimate.

    This compares the skin estimate against every product in the catalogue
    (a brute-force nearest-neighbour search), so the ranking is always exact.
    CIEDE2000 is not a proper metric (it doesn't satisfy the triangle
    inequality), so it isn't compatible with spatial index structures like
    KD-trees/ball-trees that approximate nearest-neighbour search relies on;
    benchmark_brute_force_search shows this full scan stays fast well past
    the size of the real catalogue, so no such approximation is needed here.
    """

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


def benchmark_brute_force_search(
    products: pd.DataFrame,
    *,
    catalog_sizes: Iterable[int] = (100, 300, 625, 1_250, 2_500, 5_000, 10_000),
    number_of_queries: int = 50,
    top_n: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Time brute-force CIEDE2000 search at several synthetic catalogue sizes.

    The real catalogue is only a few hundred rows, far too small to need an
    approximate or candidate-restricted search. To see how per-query latency
    actually scales, this resamples the real catalogue's LAB colours (with
    small jitter, so colours stay realistic) up to each requested size and
    times recommend_foundations over random queries against it.
    """

    if number_of_queries < 1:
        raise ValueError("number_of_queries must be at least 1.")

    rng = np.random.default_rng(random_state)
    base_lab = products[LAB_COLUMNS].to_numpy(dtype=float)
    base_columns = products[["brand", "product", "hex"]].reset_index(drop=True)

    rows = []
    for catalog_size in catalog_sizes:
        if catalog_size < 1:
            raise ValueError("catalog_sizes must all be at least 1.")

        repeats = int(np.ceil(catalog_size / len(base_lab)))
        synthetic_lab = np.tile(base_lab, (repeats, 1))[:catalog_size]
        synthetic_lab = synthetic_lab + rng.normal(0, 1.0, size=synthetic_lab.shape)
        synthetic = pd.concat([base_columns] * repeats, ignore_index=True)
        synthetic = synthetic.iloc[:catalog_size].copy()
        synthetic[LAB_COLUMNS] = synthetic_lab

        query_indices = rng.integers(0, catalog_size, size=number_of_queries)
        start = time.perf_counter()
        for index in query_indices:
            recommend_foundations(synthetic_lab[index], synthetic, top_n=top_n)
        elapsed = time.perf_counter() - start

        rows.append(
            {
                "catalog_size": int(catalog_size),
                "queries": number_of_queries,
                "seconds_per_query": elapsed / number_of_queries,
                "queries_per_second": number_of_queries / elapsed,
            }
        )

    return pd.DataFrame(rows)
