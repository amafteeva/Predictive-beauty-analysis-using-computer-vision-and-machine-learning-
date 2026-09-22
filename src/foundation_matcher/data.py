"""Dataset loading and catalogue preparation."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from skimage.color import rgb2lab

from foundation_matcher.color import hex_to_rgb, normalize_hex
from foundation_matcher.config import FOUNDATION_DATA_URL

REQUIRED_FOUNDATION_COLUMNS = {"brand", "product", "hex"}


def validate_columns(data: pd.DataFrame, required: set[str]) -> None:
    """Raise a helpful error when required columns are missing."""

    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_foundation_catalog(raw_products: pd.DataFrame) -> pd.DataFrame:
    """Clean catalogue rows and add RGB and CIELAB features."""

    validate_columns(raw_products, REQUIRED_FOUNDATION_COLUMNS)
    products = raw_products.dropna(subset=sorted(REQUIRED_FOUNDATION_COLUMNS)).copy()

    normalized_hex: list[str | None] = []
    for value in products["hex"]:
        try:
            normalized_hex.append(normalize_hex(value))
        except ValueError:
            normalized_hex.append(None)

    products["hex"] = normalized_hex
    products = products.dropna(subset=["hex"]).reset_index(drop=True)

    rgb_normalized = np.vstack(products["hex"].map(hex_to_rgb).to_numpy())
    lab_values = rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)

    products[["R", "G", "B"]] = rgb_normalized * 255.0
    products[["lab_L", "lab_a", "lab_b"]] = lab_values
    return products


def load_foundation_catalog(url: str = FOUNDATION_DATA_URL) -> pd.DataFrame:
    """Download and prepare the public foundation-shade catalogue."""

    return prepare_foundation_catalog(pd.read_csv(url))


def normalize_brand_name(value: object) -> str:
    """Lowercase and strip punctuation for approximate brand-name matching."""

    normalized = re.sub(r"[^a-z0-9\s]", " ", str(value).lower())
    return re.sub(r"\s+", " ", normalized).strip()


def attach_price_estimates(
    products: pd.DataFrame,
    price_reference: pd.DataFrame,
    *,
    brand_column: str = "brand",
    reference_brand_column: str = "brand",
    reference_price_column: str = "price",
    reference_category_column: str | None = "category",
    target_category: str | None = "Foundation",
) -> pd.DataFrame:
    """Attach a brand-level median price estimate from an external product export.

    The foundation catalogue and a price reference like the Luxxify product
    export share no product-level ID, only brand names recorded by two
    independently-scraped sources, so matching is by normalized brand name:
    an exact match after lowercasing/stripping punctuation, or one name being
    a whole-word prefix of the other (e.g. "Fenty" / "FENTY BEAUTY by
    Rihanna"). A price reference typically covers only mainstream brands, so
    niche or international brands in the catalogue get price = NaN rather
    than a guessed value — check for that explicitly rather than assuming
    every row has a price.

    When ``reference_category_column``/``target_category`` are given and
    present, the median is computed from that brand's matching-category rows
    first (e.g. only its Foundation products) rather than every product it
    sells: a brand's primers, blushes, etc. are often priced quite
    differently (on Luxxify, the median absolute difference between a
    brand's all-category and Foundation-only price is ~16%, and >100% for
    some brands). A brand with no rows in the target category falls back to
    its all-category median instead of losing price coverage entirely.
    """

    def brand_median_price(reference_subset: pd.DataFrame) -> pd.Series:
        normalized_brand = reference_subset[reference_brand_column].map(normalize_brand_name)
        return reference_subset[reference_price_column].groupby(normalized_brand).median()

    reference = price_reference.dropna(
        subset=[reference_brand_column, reference_price_column]
    )

    use_category = (
        reference_category_column is not None
        and target_category is not None
        and reference_category_column in reference.columns
    )

    fallback_price = brand_median_price(reference)
    target_price = (
        brand_median_price(reference[reference[reference_category_column] == target_category])
        if use_category
        else fallback_price
    )

    def match_price(normalized_brand: str, brand_price: pd.Series) -> float | None:
        if normalized_brand in brand_price.index:
            return brand_price.loc[normalized_brand]
        for candidate in brand_price.index:
            if candidate.startswith(normalized_brand + " ") or normalized_brand.startswith(
                candidate + " "
            ):
                return brand_price.loc[candidate]
        return None

    def resolve_price(normalized_brand: str) -> float:
        price = match_price(normalized_brand, target_price)
        if price is None:
            price = match_price(normalized_brand, fallback_price)
        return np.nan if price is None else price

    catalog_brands = products[brand_column].map(normalize_brand_name)
    price_lookup = {brand: resolve_price(brand) for brand in catalog_brands.unique()}

    result = products.copy()
    result["price"] = catalog_brands.map(price_lookup)
    return result


def apply_manual_prices(
    products: pd.DataFrame,
    brand_prices: dict[str, float | None] | None = None,
    *,
    product_prices: dict[tuple[str, str], float | None] | None = None,
) -> pd.DataFrame:
    """Fill still-missing prices from hand-entered values, keeping known prices.

    ``attach_price_estimates`` only covers brands present in the external
    price reference. This fills the remaining ``NaN`` price rows from
    hand-entered values: a finer ``(brand, product) -> price`` mapping is
    applied first, then a ``brand -> price`` mapping for whatever is still
    missing. Entries whose value is ``None`` are skipped, so a template dict
    can list every unpriced brand and be filled in gradually. Prices that
    are already set are never overwritten.
    """

    result = products.copy()
    if "price" not in result.columns:
        result["price"] = np.nan
    result["price"] = result["price"].astype(float)

    def fill_from(mapping: pd.Series) -> None:
        still_missing = result["price"].isna()
        if still_missing.any():
            result.loc[still_missing, "price"] = mapping.loc[still_missing]

    if product_prices:
        clean = {key: value for key, value in product_prices.items() if value is not None}
        keys = list(zip(result["brand"], result["product"]))
        fill_from(pd.Series([clean.get(key) for key in keys], index=result.index))

    if brand_prices:
        clean = {key: value for key, value in brand_prices.items() if value is not None}
        fill_from(result["brand"].map(clean))

    return result


def summarize_catalog_quality(
    products: pd.DataFrame,
    *,
    thin_brand_threshold: int = 5,
) -> dict[str, float]:
    """Summarize data-quality signals a plain row/brand count hides.

    ``load_foundation_catalog`` reports the catalogue size, but not whether
    that size is actually usable: how many brands have too few shades to
    support a real recommendation, whether two catalogue rows are exact
    colour duplicates, or how much of the catalogue has price data (when
    ``attach_price_estimates`` has been applied).
    """

    shades_per_brand = products.groupby("brand").size()
    thin_brands = shades_per_brand[shades_per_brand < thin_brand_threshold]

    summary: dict[str, float] = {
        "total_rows": float(len(products)),
        "total_brands": float(products["brand"].nunique()),
        f"brands_with_fewer_than_{thin_brand_threshold}_shades": float(len(thin_brands)),
        "duplicate_hex_rows": float(products.duplicated(subset=["hex"]).sum()),
        "lab_L_min": round(float(products["lab_L"].min()), 2),
        "lab_L_max": round(float(products["lab_L"].max()), 2),
    }
    if "price" in products.columns:
        summary["price_coverage"] = round(float(products["price"].notna().mean()), 3)
    return summary
