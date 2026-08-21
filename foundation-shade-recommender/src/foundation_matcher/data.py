"""Dataset loading and catalogue preparation."""

from __future__ import annotations

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
