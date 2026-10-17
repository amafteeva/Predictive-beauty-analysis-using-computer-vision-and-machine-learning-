"""MediaPipe face-landmark setup and skin-colour extraction."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from skimage.color import deltaE_ciede2000

from foundation_matcher.color import rgb_to_lab
from foundation_matcher.config import (
    FACE_LANDMARKER_URL,
    LEFT_EYE_CONTOUR_IDS,
    LEFT_IRIS_IDS,
    RIGHT_EYE_CONTOUR_IDS,
    RIGHT_IRIS_IDS,
    SKIN_LANDMARK_IDS,
)


@dataclass(frozen=True)
class SkinTone:
    """Extracted skin-colour estimate and a marked preview image."""

    rgb: np.ndarray
    lab: np.ndarray
    preview_rgb: np.ndarray
    face_count: int
    white_balance_gains: np.ndarray


def download_face_landmarker(
    destination: str | Path = "models/face_landmarker.task",
    url: str = FACE_LANDMARKER_URL,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the MediaPipe task file if it is not already available."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not destination_path.exists():
        urllib.request.urlretrieve(url, destination_path)
    return destination_path


def create_face_landmarker(
    model_path: str | Path,
    *,
    min_detection_confidence: float = 0.3,
    min_presence_confidence: float = 0.3,
    max_faces: int = 5,
):
    """Create an image-mode MediaPipe Face Landmarker.

    ``max_faces`` caps how many faces MediaPipe will report. extract_skin_tone
    always estimates skin tone from the first detected face, but a cap above
    1 lets it detect and warn about additional faces in the photo instead of
    silently discarding them at the detection stage.
    """

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_presence_confidence,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _reject_color_outliers(pixels: np.ndarray, *, threshold: float = 3.5) -> np.ndarray:
    """Drop pixels whose colour deviates strongly from the sample's own median.

    Uses a modified z-score (median absolute deviation) on each pixel's
    distance from the sample median, rather than an absolute skin-colour
    range. A fixed RGB/YCbCr skin-colour range is a known source of bias
    against darker skin tones, so outliers are judged relative to the
    sample itself: this catches a strand of hair or a shadow crossing a
    sampling circle the same way regardless of the underlying skin tone.
    """

    if len(pixels) < 4:
        return pixels

    median = np.median(pixels, axis=0)
    distances = np.linalg.norm(pixels.astype(float) - median, axis=1)
    deviations = np.abs(distances - np.median(distances))
    mad = np.median(deviations)

    if mad > 0:
        modified_z_scores = 0.6745 * distances / mad
    else:
        # A small, uncontaminated batch of pixels can have >50% at distance
        # zero from the median, which collapses the MAD to zero even though
        # a minority of far outliers remain. Fall back to the mean absolute
        # deviation (Iglewicz & Hoaglin's recommended fallback) instead of
        # skipping rejection entirely.
        mean_abs_deviation = np.mean(deviations)
        if mean_abs_deviation == 0:
            return pixels
        modified_z_scores = distances / (1.253314 * mean_abs_deviation)

    return pixels[modified_z_scores <= threshold]


def _eye_region_pixels(
    photo_rgb: np.ndarray,
    landmarks,
    width: int,
    height: int,
    contour_ids: tuple[int, ...],
    iris_ids: tuple[int, ...],
) -> np.ndarray:
    """Return pixels inside an eye's contour, excluding the iris and pupil."""

    contour_points = np.array(
        [(landmarks[i].x * width, landmarks[i].y * height) for i in contour_ids],
        dtype=np.int32,
    )
    eye_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(eye_mask, cv2.convexHull(contour_points), 255)

    iris_points = np.array(
        [(landmarks[i].x * width, landmarks[i].y * height) for i in iris_ids]
    )
    iris_centre = iris_points.mean(axis=0)
    iris_radius = np.linalg.norm(iris_points - iris_centre, axis=1).max()
    # Pad past the iris boundary so its edge and the pupil don't leak into
    # the sclera sample.
    cv2.circle(eye_mask, tuple(iris_centre.astype(int)), int(iris_radius * 1.3), 0, -1)

    return photo_rgb[eye_mask == 255]


def _estimate_white_balance_gains(
    photo_rgb: np.ndarray,
    landmarks,
    width: int,
    height: int,
    *,
    min_candidate_pixels: int = 30,
    max_gain: float = 1.8,
) -> np.ndarray:
    """Estimate per-channel correction gains from the sclera (eye white).

    The sclera is a near-white reference visible in nearly every face photo,
    so its measured colour under the shot's actual lighting reveals that
    lighting's colour cast (warm indoor bulbs, cool daylight, mixed
    sources). Scaling it back to neutral gray corrects the whole photo's
    white balance. This is more reliable than a plain gray-world assumption
    (averaging the whole frame to gray), which breaks on close-up selfies
    where skin -- not a neutral tone -- fills most of the image.

    This corrects a colour *cast* (an unusual light source). It cannot
    restore colour that a filter or edit has deliberately desaturated out
    of the skin itself -- there, the sclera will show a normal, mild cast
    while the skin sits at implausibly low saturation for its lightness;
    that failure mode needs a human to notice and pick a different photo,
    not an automatic correction.
    """

    region_pixels = [
        _eye_region_pixels(photo_rgb, landmarks, width, height, contour_ids, iris_ids)
        for contour_ids, iris_ids in (
            (LEFT_EYE_CONTOUR_IDS, LEFT_IRIS_IDS),
            (RIGHT_EYE_CONTOUR_IDS, RIGHT_IRIS_IDS),
        )
    ]
    region_pixels = [pixels for pixels in region_pixels if len(pixels) > 0]
    if not region_pixels:
        return np.ones(3)
    pixels = np.vstack(region_pixels)

    # Keep only bright, low-saturation pixels (true sclera, not eyelid skin,
    # eyelashes, or blood vessels) that aren't clipped to pure white
    # (a blown-out highlight carries no colour-cast information).
    hsv = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    saturation, value = hsv[:, 1], hsv[:, 2]
    not_clipped = pixels.max(axis=1) < 250
    white_candidates = pixels[(saturation < 40) & (value > 120) & not_clipped]

    if len(white_candidates) < min_candidate_pixels:
        return np.ones(3)

    white_point = np.clip(np.median(white_candidates, axis=0).astype(float), 1.0, None)
    gains = white_point.mean() / white_point
    return np.clip(gains, 1.0 / max_gain, max_gain)


def extract_skin_tone(
    image_path: str | Path,
    landmarker,
    *,
    landmark_ids: tuple[int, ...] = SKIN_LANDMARK_IDS,
    radius_ratio: float = 0.04,
    trim_percent: float = 5.0,
    outlier_threshold: float = 3.5,
    region_consistency_delta_e: float = 20.0,
    correct_white_balance: bool = True,
) -> SkinTone:
    """Estimate skin colour from both cheeks and the centre forehead.

    When ``correct_white_balance`` is true (the default), the photo's
    colour cast is corrected against its own sclera before skin pixels are
    sampled, so the estimate is less sensitive to the light the photo was
    taken under. If no reliable sclera reading is available (eyes closed,
    extreme angle), correction is skipped and the raw photo colours are
    used, matching the previous behaviour.
    """

    image_path = Path(image_path)
    photo_bgr = cv2.imread(str(image_path))
    if photo_bgr is None:
        raise ValueError(f"Image could not be opened: {image_path}")

    # Decode once and hand MediaPipe the same in-memory array used for pixel
    # sampling. cv2.imread and MediaPipe's own file loader can disagree on
    # EXIF orientation, which would otherwise let landmark coordinates land
    # on a differently-rotated pixel array than the one they were meant for.
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    height, width = photo_rgb.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(photo_rgb)
    )
    detection = landmarker.detect(mp_image)
    if not detection.face_landmarks:
        raise ValueError("No face was detected.")

    face_count = len(detection.face_landmarks)
    landmarks = detection.face_landmarks[0]

    if correct_white_balance:
        white_balance_gains = _estimate_white_balance_gains(
            photo_rgb, landmarks, width, height
        )
        photo_rgb = np.clip(
            photo_rgb.astype(float) * white_balance_gains, 0, 255
        ).astype(np.uint8)
    else:
        white_balance_gains = np.ones(3)

    landmark_x = np.array([point.x * width for point in landmarks])
    face_width = landmark_x.max() - landmark_x.min()
    radius = max(6, int(face_width * radius_ratio))

    preview_rgb = photo_rgb.copy()
    region_pixel_sets: list[np.ndarray] = []

    for landmark_id in landmark_ids:
        point = landmarks[landmark_id]
        centre = (int(point.x * width), int(point.y * height))
        cv2.circle(preview_rgb, centre, radius, (0, 255, 0), 2)

        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(region_mask, centre, radius, 255, -1)
        region_pixels = photo_rgb[region_mask == 255]
        if len(region_pixels) == 0:
            continue
        region_pixel_sets.append(
            _reject_color_outliers(region_pixels, threshold=outlier_threshold)
        )

    region_pixel_sets = [pixels for pixels in region_pixel_sets if len(pixels) > 0]
    if not region_pixel_sets:
        raise ValueError("No pixels were extracted from the selected regions.")

    # A sampling circle that lands mostly on hair, glasses, or shadow will
    # produce a region median far from the others. Compare each region
    # against the cross-region median (robust to a single bad region out of
    # three) and drop any region that disagrees too strongly, instead of
    # silently blending occluded pixels into the final estimate.
    region_medians = np.array([np.median(pixels, axis=0) for pixels in region_pixel_sets])
    region_labs = np.array([rgb_to_lab(median) for median in region_medians])
    overall_lab_median = np.median(region_labs, axis=0).reshape(1, 3)
    region_deltas = deltaE_ciede2000(region_labs, overall_lab_median)

    consistent_pixel_sets = [
        pixels
        for pixels, delta in zip(region_pixel_sets, region_deltas, strict=True)
        if delta <= region_consistency_delta_e
    ]
    if not consistent_pixel_sets:
        raise ValueError(
            "Sampled skin regions disagree too strongly (possible occlusion or "
            "misdetected landmarks); could not produce a reliable skin tone."
        )

    skin_pixels = np.vstack(consistent_pixel_sets)

    brightness = (
        0.2126 * skin_pixels[:, 0]
        + 0.7152 * skin_pixels[:, 1]
        + 0.0722 * skin_pixels[:, 2]
    )
    lower, upper = np.percentile(brightness, [trim_percent, 100 - trim_percent])
    valid_pixels = skin_pixels[(brightness >= lower) & (brightness <= upper)]
    if len(valid_pixels) == 0:
        valid_pixels = skin_pixels

    skin_rgb = np.median(valid_pixels, axis=0).astype(np.uint8)
    return SkinTone(
        rgb=skin_rgb,
        lab=rgb_to_lab(skin_rgb),
        preview_rgb=preview_rgb,
        face_count=face_count,
        white_balance_gains=white_balance_gains,
    )
