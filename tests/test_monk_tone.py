import numpy as np
import pandas as pd
import pytest

from foundation_matcher.monk_tone import (
    MST_HEX,
    OrdinalRidge,
    build_models,
    metrics_by_band,
    mst_reference_lab,
    nearest_mst,
    tone_band,
    tone_metrics,
)


def test_reference_swatches_run_light_to_deep():
    lab = mst_reference_lab()
    assert lab.shape == (10, 3) and len(MST_HEX) == 10
    assert np.all(np.diff(lab[:, 0]) < 1.5)  # L* (lightness) falls from MST 1 to 10
    assert lab[0, 0] > 90 and lab[-1, 0] < 20


def test_nearest_mst_recovers_each_swatch():
    assert nearest_mst(mst_reference_lab()).tolist() == list(range(1, 11))


def test_tone_band_boundaries():
    assert [tone_band(m) for m in (1, 3, 4, 6, 7, 10)] == [
        "light (MST 1-3)",
        "light (MST 1-3)",
        "medium (MST 4-6)",
        "medium (MST 4-6)",
        "deep (MST 7-10)",
        "deep (MST 7-10)",
    ]
    with pytest.raises(ValueError):
        tone_band(11)


def test_macro_mae_penalises_predicting_majority_for_everyone():
    y_true = np.array([2] * 90 + [9] * 10)
    majority = np.full(100, 2)
    m = tone_metrics(y_true, majority)
    assert m["off_by_one_accuracy"] == pytest.approx(0.9)
    assert m["macro_mae_mst_steps"] == pytest.approx(3.5)  # (0 + 7) / 2


def test_models_learn_a_monotone_colour_to_tone_mapping():
    rng = np.random.default_rng(0)
    mst = rng.integers(1, 11, 600)
    lightness = 95 - 8 * mst + rng.normal(0, 2, 600)
    X = np.column_stack([lightness, rng.normal(12, 2, 600), rng.normal(18, 3, 600)])
    for name, model in {"ordinal_ridge": OrdinalRidge(), **build_models()}.items():
        pred = model.fit(X[:400], mst[:400]).predict(X[400:])
        assert np.mean(np.abs(pred - mst[400:]) <= 1) > 0.8, name


def test_metrics_by_band_counts_people():
    frame = pd.DataFrame(
        {"mst": [1, 2, 5, 8], "person": ["a", "a", "b", "c"], "pred": [1, 3, 5, 6]}
    )
    bands = metrics_by_band(frame, "pred").set_index("tone_band")
    assert bands.loc["light (MST 1-3)", "people"] == 1
    assert bands.loc["deep (MST 7-10)", "mean_abs_error_mst_steps"] == 2
