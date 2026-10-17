import numpy as np
import pandas as pd
import pytest

from foundation_matcher.skin_calibration import (
    FEATURE_COLUMNS,
    build_calibration_model,
    calibrate_lab,
    cross_validate_calibration,
    summarize_calibration,
)


def _synthetic_table(participants=30, photos_each=4, seed=0):
    """Measured colour = known biased transform of the estimate + small noise."""

    rng = np.random.default_rng(seed)
    rows = []
    for p in range(participants):
        true = np.array([rng.uniform(25, 80), rng.uniform(8, 20), rng.uniform(10, 30)])
        for _ in range(photos_each):
            est = true * np.array([0.9, 1.15, 0.85]) + np.array([4.0, -1.0, 3.0])
            est = est + rng.normal(0, 0.5, 3)
            rows.append(
                {
                    "image_path": f"p{p}.jpg",
                    "participant_id": f"p{p}",
                    "environment": "indoor" if p % 2 else "outdoor",
                    "measured_L": true[0],
                    "measured_a": true[1],
                    "measured_b": true[2],
                    "est_L": est[0],
                    "est_a": est[1],
                    "est_b": est[2],
                    "forehead_L": est[0],
                    "forehead_a": est[1],
                    "forehead_b": est[2],
                    "wb_gain_r": 1.0,
                    "wb_gain_g": 1.0,
                    "wb_gain_b": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_calibration_beats_biased_baseline_on_held_out_people():
    results = cross_validate_calibration(_synthetic_table())
    assert (
        results["calibrated_delta_e"].median()
        < results["baseline_delta_e"].median() / 2
    )
    # every participant's photos are predicted by exactly one held-out fold
    assert (results.groupby("participant_id")["fold"].nunique() == 1).all()


def test_summary_has_overall_tone_and_extra_groups():
    results = cross_validate_calibration(_synthetic_table())
    summary = summarize_calibration(results, group_columns=("environment",))
    assert set(summary["group"]) == {"all", "measured_tone_tertile", "environment"}
    assert set(summary.loc[summary["group"] == "measured_tone_tertile", "value"]) == {
        "deep",
        "medium",
        "light",
    }
    assert summary.loc[summary["group"] == "all", "images"].item() == len(results)


def test_too_few_participants_is_rejected():
    with pytest.raises(ValueError):
        cross_validate_calibration(_synthetic_table(participants=3))


def test_calibrate_lab_applies_trained_model():
    table = _synthetic_table()
    model = build_calibration_model().fit(
        table[list(FEATURE_COLUMNS)].to_numpy(float),
        table[["measured_L", "measured_a", "measured_b"]].to_numpy(float),
    )
    row = table.iloc[0]
    predicted = calibrate_lab({c: row[c] for c in FEATURE_COLUMNS}, model)[0]
    assert np.allclose(
        predicted,
        row[["measured_L", "measured_a", "measured_b"]].astype(float),
        atol=1.5,
    )
