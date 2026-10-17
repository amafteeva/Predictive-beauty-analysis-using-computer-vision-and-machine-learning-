"""Learned skin-colour calibration from colorimeter ground truth (opt-in).

The recommender's skin estimate (face.extract_skin_tone) is a fixed rule with
no way to check its accuracy. Datasets such as CHROMA-FIT pair face photos
with colorimeter-measured skin colour (CIELAB, forehead), which makes that
step trainable and measurable:

    photo -> image features (this module) -> ridge regression -> measured L*a*b*

Evaluation is always grouped by participant (no person in both train and
test), and the baseline is the uncalibrated estimate itself, so the reported
gain is "how much closer to the measured colour", in CIEDE2000.

Nothing here changes recommend_foundations; a trained model is applied only
by callers that opt in via ``calibrate_lab``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foundation_matcher.color import rgb_to_lab
from foundation_matcher.face import (
    _estimate_white_balance_gains,
    _reject_color_outliers,
    extract_skin_tone,
)

TARGET_COLUMNS = ("measured_L", "measured_a", "measured_b")
REGION_LANDMARKS = {"left_cheek": 117, "right_cheek": 346, "forehead": 151}
FEATURE_COLUMNS = (
    "est_L", "est_a", "est_b",
    "forehead_L", "forehead_a", "forehead_b",
    "wb_gain_r", "wb_gain_g", "wb_gain_b",
)  # fmt: skip
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def extract_calibration_features(
    image_path: str | Path, landmarker
) -> dict[str, float]:
    """Features for one photo: the pipeline's own estimate plus forehead colour.

    ``est_*`` is exactly what the recommender uses today (extract_skin_tone).
    ``forehead_*`` is the median of the forehead sample after the same
    white balance and outlier filter, because colorimeter ground truth is
    usually taken on the forehead. ``wb_gain_*`` lets the model learn how
    much the lighting correction moved the colour.
    """

    tone = extract_skin_tone(image_path, landmarker)
    rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    detection = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    )
    landmarks = detection.face_landmarks[0]
    gains = _estimate_white_balance_gains(rgb, landmarks, width, height)
    corrected = np.clip(rgb.astype(float) * gains, 0, 255).astype(np.uint8)
    xs = np.array([p.x * width for p in landmarks])
    radius = max(6, int((xs.max() - xs.min()) * 0.04))
    point = landmarks[REGION_LANDMARKS["forehead"]]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(point.x * width), int(point.y * height)), radius, 255, -1)
    pixels = _reject_color_outliers(corrected[mask == 255])
    forehead = rgb_to_lab(np.median(pixels, axis=0)) if len(pixels) else tone.lab
    return {
        "est_L": float(tone.lab[0]),
        "est_a": float(tone.lab[1]),
        "est_b": float(tone.lab[2]),
        "forehead_L": float(forehead[0]),
        "forehead_a": float(forehead[1]),
        "forehead_b": float(forehead[2]),
        "wb_gain_r": float(tone.white_balance_gains[0]),
        "wb_gain_g": float(tone.white_balance_gains[1]),
        "wb_gain_b": float(tone.white_balance_gains[2]),
    }


def build_feature_table(
    manifest: pd.DataFrame, landmarker, *, root: Path | None = None
) -> pd.DataFrame:
    """Run feature extraction for every manifest row; failures are recorded."""

    rows = []
    for record in manifest.to_dict("records"):
        path = Path(record["image_path"])
        if root is not None and not path.is_absolute():
            path = root / path
        try:
            features = extract_calibration_features(path, landmarker)
            rows.append({**record, **features, "extraction_error": None})
        except Exception as error:  # keep going; failures are reported
            rows.append({**record, "extraction_error": str(error)})
    return pd.DataFrame(rows)


def build_calibration_model() -> Pipeline:
    """Standardised ridge regression, regularisation chosen by internal CV."""

    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])


def delta_e(predicted: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return deltaE_ciede2000(np.asarray(predicted, float), np.asarray(measured, float))


def cross_validate_calibration(
    table: pd.DataFrame,
    *,
    n_splits: int = 5,
    features: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Participant-grouped out-of-fold predictions plus the uncalibrated baseline.

    Returns one row per image with ``baseline_delta_e`` (pipeline estimate vs
    measured) and ``calibrated_delta_e`` (out-of-fold model vs measured).
    """

    data = table.dropna(subset=[*features, *TARGET_COLUMNS]).reset_index(drop=True)
    participants = data["participant_id"].nunique()
    if participants < n_splits:
        raise ValueError(f"Need at least {n_splits} participants, got {participants}.")
    X = data[list(features)].to_numpy(float)
    y = data[list(TARGET_COLUMNS)].to_numpy(float)
    predicted = np.full_like(y, np.nan)
    fold = np.full(len(data), -1)
    for k, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(X, y, data["participant_id"])
    ):
        model = build_calibration_model().fit(X[train], y[train])
        predicted[test] = model.predict(X[test])
        fold[test] = k
    out = data.copy()
    out[["pred_L", "pred_a", "pred_b"]] = predicted
    out["fold"] = fold
    out["baseline_delta_e"] = delta_e(
        data[["est_L", "est_a", "est_b"]].to_numpy(float), y
    )
    out["calibrated_delta_e"] = delta_e(predicted, y)
    return out


def summarize_calibration(
    results: pd.DataFrame, *, group_columns: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Median/mean ΔE before and after calibration, overall and per group.

    Adds a ``measured_tone_tertile`` group (light / medium / deep by measured
    L*) so error can be compared across skin tones.
    """

    data = results.copy()
    data["measured_tone_tertile"] = pd.qcut(
        data["measured_L"], 3, labels=["deep", "medium", "light"]
    )
    groups = [("all", None), ("measured_tone_tertile", "measured_tone_tertile")]
    groups += [(g, g) for g in group_columns if g in data.columns]
    rows = []
    for name, column in groups:
        parts = (
            [("all", data)]
            if column is None
            else list(data.groupby(column, observed=True))
        )
        for value, part in parts:
            rows.append(
                {
                    "group": name,
                    "value": str(value),
                    "images": len(part),
                    "participants": part["participant_id"].nunique(),
                    "baseline_median_delta_e": part["baseline_delta_e"].median(),
                    "calibrated_median_delta_e": part["calibrated_delta_e"].median(),
                    "baseline_mean_delta_e": part["baseline_delta_e"].mean(),
                    "calibrated_mean_delta_e": part["calibrated_delta_e"].mean(),
                    "share_improved": float(
                        (part["calibrated_delta_e"] < part["baseline_delta_e"]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def calibrate_lab(
    features: dict[str, float] | pd.DataFrame,
    model: Pipeline,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Apply a trained calibration model (opt-in; not used by the recommender)."""

    frame = pd.DataFrame([features]) if isinstance(features, dict) else features
    return model.predict(frame[list(feature_columns)].to_numpy(float))
