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
from foundation_matcher.config import FACE_LANDMARKER_URL, SKIN_LANDMARK_IDS


@dataclass(frozen=True)
class SkinTone:
    """Extracted skin-colour estimate and a marked preview image."""

    rgb: np.ndarray
    lab: np.ndarray
    preview_rgb: np.ndarray
    face_count: int


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


def extract_skin_tone(
    image_path: str | Path,
    landmarker,
    *,
    landmark_ids: tuple[int, ...] = SKIN_LANDMARK_IDS,
    radius_ratio: float = 0.04,
    trim_percent: float = 5.0,
    outlier_threshold: float = 3.5,
    region_consistency_delta_e: float = 20.0,
) -> SkinTone:
    """Estimate skin colour from both cheeks and the centre forehead."""

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
    )
