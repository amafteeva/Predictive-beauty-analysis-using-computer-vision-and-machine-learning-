import numpy as np
import pandas as pd
import pytest

from foundation_matcher.recommender import (
    evaluate_cluster_recommendation_impact,
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


def test_cluster_recommendation_impact_is_zero_with_a_single_cluster():
    result = evaluate_cluster_recommendation_impact(
        _spread_out_catalog(),
        cluster_counts=(1,),
        number_of_queries=20,
        random_state=1,
    )
    row = result.iloc[0]
    assert row["clusters"] == 1
    assert row["top1_mismatch_rate"] == 0.0
    assert row["mean_delta_e_loss"] == 0.0
    assert row["max_delta_e_loss"] == 0.0


def test_cluster_recommendation_impact_losses_are_never_negative():
    result = evaluate_cluster_recommendation_impact(
        _spread_out_catalog(),
        cluster_counts=(3, 5),
        number_of_queries=40,
        random_state=1,
    )
    assert set(result["clusters"]) == {3, 5}
    assert (result["mean_delta_e_loss"] >= 0).all()
    assert (result["max_delta_e_loss"] >= result["mean_delta_e_loss"]).all()
    assert (result["top1_mismatch_rate"] >= 0).all()
    assert (result["perceptible_loss_rate"] >= 0).all()


def test_cluster_recommendation_impact_rejects_invalid_query_count():
    with pytest.raises(ValueError, match="number_of_queries"):
        evaluate_cluster_recommendation_impact(
            _spread_out_catalog(), cluster_counts=(2,), number_of_queries=0
        )
