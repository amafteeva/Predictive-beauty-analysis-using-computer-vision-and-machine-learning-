import numpy as np
import pytest

from foundation_matcher.color import hex_to_rgb, normalize_hex, rgb_to_hex, rgb_to_lab


def test_hex_normalization_and_conversion():
    assert normalize_hex("#ff0080") == "FF0080"
    np.testing.assert_allclose(hex_to_rgb("FF0000"), [1.0, 0.0, 0.0])
    assert rgb_to_hex([255, 0, 128]) == "#FF0080"


def test_invalid_hex_raises_helpful_error():
    with pytest.raises(ValueError, match="Invalid"):
        normalize_hex("not-a-colour")


def test_rgb_to_lab_has_three_components():
    lab = rgb_to_lab([128, 100, 75])
    assert lab.shape == (3,)
    assert np.isfinite(lab).all()
