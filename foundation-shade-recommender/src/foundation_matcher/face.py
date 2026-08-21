"""MediaPipe face-landmark setup and skin-colour extraction."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from foundation_matcher.color import rgb_to_lab
from foundation_matcher.config import FACE_LANDMARKER_URL, SKIN_LANDMARK_IDS


@dataclass(frozen=True)
class SkinTone:
    """Extracted skin-colour estimate and a marked preview image."""

    rgb: np.ndarray
    lab: np.ndarray
    preview_rgb: np.ndarray


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
):
    """Create an image-mode MediaPipe Face Landmarker."""

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_presence_confidence,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def extract_skin_tone(
    image_path: str | Path,
    landmarker,
    *,
    landmark_ids: tuple[int, ...] = SKIN_LANDMARK_IDS,
    radius_ratio: float = 0.04,
    trim_percent: float = 5.0,
) -> SkinTone:
    """Estimate skin colour from both cheeks and the centre forehead."""

    image_path = Path(image_path)
    photo_bgr = cv2.imread(str(image_path))
    if photo_bgr is None:
        raise ValueError(f"Image could not be opened: {image_path}")

    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    height, width = photo_rgb.shape[:2]
    detection = landmarker.detect(mp.Image.create_from_file(str(image_path)))
    if not detection.face_landmarks:
        raise ValueError("No face was detected.")

    landmarks = detection.face_landmarks[0]
    landmark_x = np.array([point.x * width for point in landmarks])
    face_width = landmark_x.max() - landmark_x.min()
    radius = max(6, int(face_width * radius_ratio))

    skin_mask = np.zeros((height, width), dtype=np.uint8)
    preview_rgb = photo_rgb.copy()

    for landmark_id in landmark_ids:
        point = landmarks[landmark_id]
        centre = (int(point.x * width), int(point.y * height))
        cv2.circle(skin_mask, centre, radius, 255, -1)
        cv2.circle(preview_rgb, centre, radius, (0, 255, 0), 2)

    skin_pixels = photo_rgb[skin_mask == 255]
    if len(skin_pixels) == 0:
        raise ValueError("No pixels were extracted from the selected regions.")

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
    )
