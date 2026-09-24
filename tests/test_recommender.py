import numpy as np
import pandas as pd
import pytest

from foundation_matcher.recommender import (
    benchmark_brute_force_search,
    recommend_foundations,
)


def test_exact_lab_match_is_ranked_first():
    products = pd.DataFrame(
        {
            "brand": ["A", "B"],
            "product": ["Exact", "Far"],
            "hex": ["AA8866", "FFFFFF"],
            "lab_L": [50.0, 90.0],
            "lab_a": [10.0, 0.0],
            "lab_b": [15.0, 0.0],
        }
    )
    result = recommend_foundations(np.array([50.0, 10.0, 15.0]), products, top_n=1)
    assert result.iloc[0]["product"] == "Exact"
    assert result.iloc[0]["color_distance"] == 0.0
    assert "price" not in result.columns


def test_price_is_included_when_present_on_the_catalogue():
    products = pd.DataFrame(
        {
            "brand": ["A", "B"],
            "product": ["Exact", "Far"],
            "hex": ["AA8866", "FFFFFF"],
            "lab_L": [50.0, 90.0],
            "lab_a": [10.0, 0.0],
            "lab_b": [15.0, 0.0],
            "price": [25.0, np.nan],
        }
    )
    result = recommend_foundations(np.array([50.0, 10.0, 15.0]), products, top_n=2)
    assert list(result.columns).index("price") < list(result.columns).index(
        "color_distance"
    )
    assert result.loc[result["product"] == "Exact", "price"].iloc[0] == 25.0


def _spread_out_catalog(n: int = 15) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "brand": [f"Brand{i}" for i in range(n)],
            "product": [f"Product{i}" for i in range(n)],
            "hex": ["AA8866"] * n,
            "lab_L": np.linspace(20, 90, n),
            "lab_a": np.linspace(-10, 25, n),
            "lab_b": np.linspace(-5, 35, n),
        }
    )


def test_benchmark_brute_force_search_reports_one_row_per_catalog_size():
    result = benchmark_brute_force_search(
        _spread_out_catalog(),
        catalog_sizes=(20, 50),
        number_of_queries=5,
        random_state=1,
    )
    assert list(result["catalog_size"]) == [20, 50]
    assert (result["seconds_per_query"] > 0).all()
    assert (result["queries_per_second"] > 0).all()


def test_benchmark_brute_force_search_rejects_invalid_query_count():
    with pytest.raises(ValueError, match="number_of_queries"):
        benchmark_brute_force_search(
            _spread_out_catalog(), catalog_sizes=(20,), number_of_queries=0
        )
