import numpy as np
import pandas as pd

from foundation_matcher.data import (
    apply_manual_prices,
    attach_price_estimates,
    normalize_brand_name,
    prepare_foundation_catalog,
    summarize_catalog_quality,
)


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


def test_normalize_brand_name_strips_case_and_punctuation():
    assert normalize_brand_name("CoverGirl + Olay") == "covergirl olay"
    assert normalize_brand_name("  Fenty  ") == "fenty"


def test_attach_price_estimates_matches_exact_and_prefix_brands():
    products = pd.DataFrame({"brand": ["Fenty", "Maybelline", "Obscure Brand"]})
    reference = pd.DataFrame(
        {
            "brand": [
                "FENTY BEAUTY by Rihanna",
                "FENTY BEAUTY by Rihanna",
                "Maybelline",
                "Maybelline",
            ],
            "price": [40.0, 44.0, 10.0, 12.0],
        }
    )

    result = attach_price_estimates(products, reference)

    assert result.loc[result["brand"] == "Fenty", "price"].iloc[0] == 42.0
    assert result.loc[result["brand"] == "Maybelline", "price"].iloc[0] == 11.0
    assert np.isnan(result.loc[result["brand"] == "Obscure Brand", "price"].iloc[0])


def test_attach_price_estimates_prefers_foundation_category_price():
    products = pd.DataFrame({"brand": ["Alpha"]})
    reference = pd.DataFrame(
        {
            "brand": ["Alpha", "Alpha", "Alpha"],
            "category": ["Blush", "Blush", "Foundation"],
            "price": [5.0, 7.0, 40.0],
        }
    )

    result = attach_price_estimates(products, reference)

    # The brand's all-category median (6.0, from the two Blush rows) would
    # be a poor stand-in for its one real Foundation price (40.0).
    assert result.loc[0, "price"] == 40.0


def test_attach_price_estimates_falls_back_to_all_categories_when_brand_lacks_target():
    products = pd.DataFrame({"brand": ["Beta"]})
    reference = pd.DataFrame(
        {
            "brand": ["Beta", "Beta"],
            "category": ["Blush", "Bronzer"],
            "price": [10.0, 20.0],
        }
    )

    result = attach_price_estimates(products, reference)

    # No Foundation rows for this brand at all -> fall back rather than NaN.
    assert result.loc[0, "price"] == 15.0


def test_attach_price_estimates_ignores_reference_rows_missing_price():
    products = pd.DataFrame({"brand": ["Maybelline"]})
    reference = pd.DataFrame(
        {"brand": ["Maybelline", "Maybelline"], "price": [10.0, None]}
    )

    result = attach_price_estimates(products, reference)

    assert result.loc[0, "price"] == 10.0


def test_summarize_catalog_quality_flags_thin_brands_and_duplicates():
    products = pd.DataFrame(
        {
            "brand": ["Big"] * 6 + ["Thin"] * 2,
            "hex": ["AA8866", "BB8866", "CC8866", "DD8866", "EE8866", "EE8866"]
            + ["112233", "112233"],
            "lab_L": [50.0, 55.0, 60.0, 65.0, 70.0, 70.0, 20.0, 20.0],
            "lab_a": [0.0] * 8,
            "lab_b": [0.0] * 8,
        }
    )

    summary = summarize_catalog_quality(products, thin_brand_threshold=5)

    assert summary["total_rows"] == 8
    assert summary["total_brands"] == 2
    assert summary["brands_with_fewer_than_5_shades"] == 1
    # "EE8866" repeats within Big, and "112233" repeats within Thin.
    assert summary["duplicate_hex_rows"] == 2
    assert summary["lab_L_min"] == 20.0
    assert summary["lab_L_max"] == 70.0
    assert "price_coverage" not in summary


def test_summarize_catalog_quality_reports_price_coverage_when_present():
    products = pd.DataFrame(
        {
            "brand": ["A", "A", "B"],
            "hex": ["AA8866", "BB8866", "CC8866"],
            "lab_L": [50.0, 55.0, 60.0],
            "lab_a": [0.0, 0.0, 0.0],
            "lab_b": [0.0, 0.0, 0.0],
            "price": [10.0, None, 20.0],
        }
    )

    summary = summarize_catalog_quality(products)

    assert summary["price_coverage"] == round(2 / 3, 3)


def _priced_catalog():
    return pd.DataFrame(
        {
            "brand": ["Known", "Gap", "Gap", "Other"],
            "product": ["P1", "P2", "P3", "P4"],
            "price": [15.0, np.nan, np.nan, np.nan],
        }
    )


def test_apply_manual_prices_fills_gaps_without_overwriting_known_prices():
    result = apply_manual_prices(
        _priced_catalog(), {"Gap": 30.0, "Known": 99.0, "Missing": None}
    )

    assert result.loc[result["brand"] == "Known", "price"].iloc[0] == 15.0
    assert (result.loc[result["brand"] == "Gap", "price"] == 30.0).all()
    assert np.isnan(result.loc[result["brand"] == "Other", "price"].iloc[0])


def test_apply_manual_prices_prefers_product_level_over_brand_level():
    result = apply_manual_prices(
        _priced_catalog(),
        {"Gap": 30.0},
        product_prices={("Gap", "P2"): 12.0},
    )

    assert result.loc[result["product"] == "P2", "price"].iloc[0] == 12.0
    assert result.loc[result["product"] == "P3", "price"].iloc[0] == 30.0
