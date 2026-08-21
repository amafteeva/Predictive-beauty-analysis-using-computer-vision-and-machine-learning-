import pandas as pd

from foundation_matcher.data import prepare_foundation_catalog


def test_catalogue_preparation_removes_invalid_rows_and_adds_features():
    raw = pd.DataFrame(
        {
            "brand": ["Brand A", "Brand B", "Brand C"],
            "product": ["Base", "Tint", "Invalid"],
            "hex": ["#AA8866", "F0D0B0", "bad"],
        }
    )
    prepared = prepare_foundation_catalog(raw)
    assert len(prepared) == 2
    assert set(["R", "G", "B", "lab_L", "lab_a", "lab_b"]).issubset(
        prepared.columns
    )
    assert prepared[["lab_L", "lab_a", "lab_b"]].notna().all().all()
