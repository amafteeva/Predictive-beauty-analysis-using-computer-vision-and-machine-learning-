import cv2
import numpy as np
import pytest

from foundation_matcher.face import _reject_color_outliers, extract_skin_tone

IMAGE_SIZE = 200
SKIN_RGB = (200, 150, 120)
HAIR_RGB = (20, 20, 20)


class FakeLandmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class FakeDetectionResult:
    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks


class FakeLandmarker:
    """Stands in for mp.tasks.vision.FaceLandmarker in tests.

    Records the image it was asked to detect on so tests can confirm
    extract_skin_tone hands it the same decoded array used for pixel
    sampling, instead of re-reading the file through a second decoder.
    """

    def __init__(self, detection_result):
        self._detection_result = detection_result
        self.received_image = None

    def detect(self, image):
        self.received_image = image
        return self._detection_result


def _make_face_landmarks(cheek_left_xy, cheek_right_xy, forehead_xy):
    # Index 117/346/151 match SKIN_LANDMARK_IDS; the rest are filler points
    # spanning the full image so face-width detection stays deterministic.
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(350)]
    landmarks[0] = FakeLandmark(0.0, 0.5)
    landmarks[1] = FakeLandmark(1.0, 0.5)
    landmarks[117] = FakeLandmark(*cheek_left_xy)
    landmarks[346] = FakeLandmark(*cheek_right_xy)
    landmarks[151] = FakeLandmark(*forehead_xy)
    return landmarks


def _uniform_image(color=SKIN_RGB):
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[:] = color
    return image


def _save_bgr(path, rgb_image):
    cv2.imwrite(str(path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


def test_reject_color_outliers_removes_far_pixels_only():
    base = np.tile(np.array(SKIN_RGB, dtype=np.uint8), (100, 1))
    outliers = np.tile(np.array(HAIR_RGB, dtype=np.uint8), (5, 1))
    pixels = np.vstack([base, outliers])

    filtered = _reject_color_outliers(pixels)

    assert len(filtered) == 100
    assert np.all(filtered == SKIN_RGB)


def test_reject_color_outliers_keeps_small_samples_unfiltered():
    pixels = np.array([[10, 10, 10], [200, 200, 200]], dtype=np.uint8)
    assert len(_reject_color_outliers(pixels)) == 2


def test_extract_skin_tone_matches_uniform_skin_colour(tmp_path):
    image_path = tmp_path / "uniform.jpg"
    _save_bgr(image_path, _uniform_image())

    landmarks = _make_face_landmarks((0.3, 0.6), (0.7, 0.6), (0.5, 0.2))
    landmarker = FakeLandmarker(FakeDetectionResult([landmarks]))

    skin_tone = extract_skin_tone(image_path, landmarker)

    assert np.allclose(skin_tone.rgb, SKIN_RGB, atol=2)
    # Detection ran against the same decoded array used for sampling, not a
    # second, potentially differently-oriented decode of the file.
    assert landmarker.received_image.width == IMAGE_SIZE
    assert landmarker.received_image.height == IMAGE_SIZE


def test_extract_skin_tone_ignores_an_occluded_region(tmp_path):
    image = _uniform_image()
    forehead_pixel = (int(0.5 * IMAGE_SIZE), int(0.2 * IMAGE_SIZE))
    # Simulate hair covering the entire forehead sampling circle.
    cv2.circle(image, forehead_pixel, 20, HAIR_RGB, -1)
    image_path = tmp_path / "occluded.jpg"
    _save_bgr(image_path, image)

    landmarks = _make_face_landmarks((0.3, 0.6), (0.7, 0.6), (0.5, 0.2))
    landmarker = FakeLandmarker(FakeDetectionResult([landmarks]))

    skin_tone = extract_skin_tone(image_path, landmarker)

    assert np.allclose(skin_tone.rgb, SKIN_RGB, atol=2)


def test_extract_skin_tone_reports_face_count(tmp_path):
    image_path = tmp_path / "two_faces.jpg"
    _save_bgr(image_path, _uniform_image())

    landmarks = _make_face_landmarks((0.3, 0.6), (0.7, 0.6), (0.5, 0.2))
    second_face_landmarks = _make_face_landmarks((0.3, 0.6), (0.7, 0.6), (0.5, 0.2))
    landmarker = FakeLandmarker(
        FakeDetectionResult([landmarks, second_face_landmarks])
    )

    skin_tone = extract_skin_tone(image_path, landmarker)

    assert skin_tone.face_count == 2


def test_extract_skin_tone_raises_when_all_regions_disagree(tmp_path):
    # Three mutually saturated, unrelated colours chosen so that even the
    # per-channel median across the three regions lands far (CIEDE2000 > 20)
    # from every one of them, so none should survive the consistency check.
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    cheek_left_pixel = (int(0.3 * IMAGE_SIZE), int(0.6 * IMAGE_SIZE))
    cheek_right_pixel = (int(0.7 * IMAGE_SIZE), int(0.6 * IMAGE_SIZE))
    forehead_pixel = (int(0.5 * IMAGE_SIZE), int(0.2 * IMAGE_SIZE))
    cv2.circle(image, cheek_left_pixel, 20, (255, 165, 0), -1)
    cv2.circle(image, cheek_right_pixel, 20, (0, 128, 128), -1)
    cv2.circle(image, forehead_pixel, 20, (128, 0, 128), -1)
    image_path = tmp_path / "inconsistent_regions.jpg"
    _save_bgr(image_path, image)

    landmarks = _make_face_landmarks((0.3, 0.6), (0.7, 0.6), (0.5, 0.2))
    landmarker = FakeLandmarker(FakeDetectionResult([landmarks]))

    with pytest.raises(ValueError, match="disagree too strongly"):
        extract_skin_tone(image_path, landmarker)


def test_extract_skin_tone_raises_when_no_face_detected(tmp_path):
    image_path = tmp_path / "no_face.jpg"
    _save_bgr(image_path, _uniform_image())
    landmarker = FakeLandmarker(FakeDetectionResult([]))

    with pytest.raises(ValueError, match="No face was detected"):
        extract_skin_tone(image_path, landmarker)
