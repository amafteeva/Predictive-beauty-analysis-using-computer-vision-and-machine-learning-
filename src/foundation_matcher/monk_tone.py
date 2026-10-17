"""Monk Skin Tone (MST) prediction from the pipeline's skin-colour features.

Trains small models that map the recommender's own skin estimate (plus the
forehead colour and white-balance gains from skin_calibration) to the
10-point Monk Skin Tone scale, using annotated face datasets such as STW
(Skin Tone in the Wild). The untrained baseline assigns the nearest official
MST swatch by CIEDE2000 -- the same nearest-colour logic the shade
recommender uses -- so the comparison shows what learning adds.

MST labels are human annotations of perceived tone, not colorimeter
measurements, so this evaluates whether the extracted colour orders and
groups skin tones the way people do. It is not a shade-match accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foundation_matcher.color import hex_to_rgb, rgb_to_lab

# Official Monk Skin Tone scale swatches, MST 1 (lightest) to MST 10 (deepest).
MST_HEX = (
    "f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96",
    "a07e56", "825c43", "604134", "3a312a", "292420",
)  # fmt: skip
MST_CLASSES = tuple(range(1, 11))
TONE_BANDS = {
    "light (MST 1-3)": (1, 3),
    "medium (MST 4-6)": (4, 6),
    "deep (MST 7-10)": (7, 10),
}


def mst_reference_lab() -> np.ndarray:
    """CIELAB of the 10 MST swatches, shape (10, 3)."""

    return np.vstack([rgb_to_lab(hex_to_rgb(h)) for h in MST_HEX])


def nearest_mst(lab: np.ndarray) -> np.ndarray:
    """Untrained baseline: index (1-10) of the closest MST swatch by CIEDE2000."""

    lab = np.atleast_2d(np.asarray(lab, dtype=float))
    refs = mst_reference_lab()
    distances = np.stack(
        [deltaE_ciede2000(lab, np.tile(r, (len(lab), 1))) for r in refs], axis=1
    )
    return distances.argmin(axis=1) + 1


def tone_band(mst: int) -> str:
    for name, (low, high) in TONE_BANDS.items():
        if low <= mst <= high:
            return name
    raise ValueError(f"MST must be 1-10, got {mst}")


class OrdinalRidge(BaseEstimator, ClassifierMixin):
    """Treat MST as a number: ridge regression, then round and clip to 1-10."""

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas

    def fit(self, X, y):
        self.model_ = Pipeline(
            [("scale", StandardScaler()), ("ridge", RidgeCV(alphas=self.alphas))]
        ).fit(X, y)
        self.classes_ = np.array(MST_CLASSES)
        return self

    def predict(self, X):
        return np.clip(np.rint(self.model_.predict(X)), 1, 10).astype(int)


def build_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Small, regularised models suited to 9 features and a skewed label set."""

    return {
        "ordinal_ridge": OrdinalRidge(),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def tone_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "off_by_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mean_abs_error_mst_steps": float(np.mean(np.abs(y_true - y_pred))),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", labels=list(MST_CLASSES), zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        # Mean over annotated classes of each class's MAE: every tone counts
        # equally, so a model that predicts "light" for everyone scores badly.
        "macro_mae_mst_steps": float(
            np.mean(
                [np.mean(np.abs(y_pred[y_true == c] - c)) for c in np.unique(y_true)]
            )
        ),
    }


def metrics_by_band(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """Accuracy, off-by-one and MAE within light / medium / deep tone bands."""

    rows = []
    for band, part in frame.groupby(frame["mst"].map(tone_band), sort=False):
        m = tone_metrics(part["mst"], part[prediction_column])
        rows.append(
            {
                "tone_band": band,
                "images": len(part),
                "people": part["person"].nunique(),
                "accuracy": m["accuracy"],
                "off_by_one_accuracy": m["off_by_one_accuracy"],
                "mean_abs_error_mst_steps": m["mean_abs_error_mst_steps"],
            }
        )
    order = list(TONE_BANDS)
    return pd.DataFrame(rows).sort_values("tone_band", key=lambda s: s.map(order.index))
