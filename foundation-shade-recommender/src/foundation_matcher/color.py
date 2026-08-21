"""Colour conversion helpers used throughout the project."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
from skimage.color import rgb2lab

HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{6}$")


def normalize_hex(value: object) -> str:
    """Return a six-character uppercase HEX value without the leading hash."""

    normalized = str(value).strip().lstrip("#")
    if not HEX_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid six-character HEX colour: {value!r}")
    return normalized.upper()


def hex_to_rgb(value: object) -> np.ndarray:
    """Convert HEX to an RGB float array in the interval [0, 1]."""

    normalized = normalize_hex(value)
    return np.array(
        [int(normalized[index : index + 2], 16) for index in (0, 2, 4)],
        dtype=float,
    ) / 255.0


def rgb_to_lab(rgb: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert one RGB colour to CIELAB.

    The input may use either the [0, 1] or [0, 255] RGB scale.
    """

    rgb_array = np.asarray(rgb, dtype=float).reshape(1, 1, 3)
    if rgb_array.max() > 1.0:
        rgb_array = rgb_array / 255.0
    if rgb_array.min() < 0 or rgb_array.max() > 1:
        raise ValueError("RGB values must be within [0, 1] or [0, 255].")
    return rgb2lab(rgb_array)[0, 0]


def rgb_to_hex(rgb: Iterable[float] | np.ndarray) -> str:
    """Convert one [0, 255] RGB colour to a display-ready HEX string."""

    rgb_array = np.asarray(rgb, dtype=float).reshape(3)
    rgb_array = np.clip(np.rint(rgb_array), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb_array)
