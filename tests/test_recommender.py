import numpy as np
import pandas as pd

from foundation_matcher.recommender import recommend_foundations


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
