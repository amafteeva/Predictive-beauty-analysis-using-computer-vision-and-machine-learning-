"""Shared constants for notebooks and source modules."""

FOUNDATION_DATA_URL = (
    "https://raw.githubusercontent.com/the-pudding/data/"
    "master/makeup-shades/shades.csv"
)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

LUXXIFY_PRODUCT_URL = (
    "https://raw.githubusercontent.com/zara-sarkar/"
    "Luxxify-Makeup-Recommender/main/cleaned_makeup_products.csv"
)

LUXXIFY_REVIEW_URL = (
    "https://raw.githubusercontent.com/zara-sarkar/"
    "Luxxify-Makeup-Recommender/main/cleaned_makeup_reviews.csv"
)

LAB_COLUMNS = ["lab_L", "lab_a", "lab_b"]
SKIN_LANDMARK_IDS = (117, 346, 151)
RANDOM_STATE = 42

FAIRFACE_RACE_NAMES = [
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino/Hispanic",
    "Southeast Asian",
]
