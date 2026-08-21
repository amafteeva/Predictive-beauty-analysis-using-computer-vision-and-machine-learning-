"""Pipeline-completion and simulated colour-robustness evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000

from foundation_matcher.config import LAB_COLUMNS, RANDOM_STATE
from foundation_matcher.face import extract_skin_tone
from foundation_matcher.recommender import recommend_foundations


def select_balanced_adult_samples(
    dataset: Iterable[dict],
    *,
    number_of_groups: int = 7,
    samples_per_group: int = 20,
    adult_age_minimum: int = 3,
) -> list[dict]:
    """Select an equal number of adult samples from each integer-coded group."""

    counts = {group: 0 for group in range(number_of_groups)}
    selected: list[dict] = []

    for sample in dataset:
        group = int(sample["race"])
        age_group = int(sample["age"])
        if age_group >= adult_age_minimum and counts[group] < samples_per_group:
            selected.append(sample)
            counts[group] += 1
        if all(count == samples_per_group for count in counts.values()):
            break

    missing = {group: count for group, count in counts.items() if count < samples_per_group}
    if missing:
        raise ValueError(f"Could not build the requested balanced sample: {missing}")
    return selected


def evaluate_face_pipeline(
    samples: Sequence[dict],
    group_names: Sequence[str],
    landmarker,
    products: pd.DataFrame,
    *,
    image_directory: str | Path,
    top_n: int = 5,
    preview_limit: int = 14,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run the complete pipeline and record completion or failure per image."""

    image_directory = Path(image_directory)
    image_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    previews: list[dict] = []

    for image_number, sample in enumerate(samples):
        group_id = int(sample["race"])
        group_name = group_names[group_id]
        image_path = image_directory / f"image_{image_number:03d}.jpg"
        sample["image"].convert("RGB").save(image_path)

        try:
            skin_tone = extract_skin_tone(image_path, landmarker)
            matches = recommend_foundations(skin_tone.lab, products, top_n=top_n)
            top_match = matches.iloc[0]
            rows.append(
                {
                    "image_id": image_number,
                    "group": group_name,
                    "pipeline_success": True,
                    "skin_R": int(skin_tone.rgb[0]),
                    "skin_G": int(skin_tone.rgb[1]),
                    "skin_B": int(skin_tone.rgb[2]),
                    "skin_L": float(skin_tone.lab[0]),
                    "skin_a": float(skin_tone.lab[1]),
                    "skin_b": float(skin_tone.lab[2]),
                    "recommendation_count": len(matches),
                    "top_brand": top_match["brand"],
                    "top_product": top_match["product"],
                    "top_distance": float(top_match["color_distance"]),
                    "error": None,
                }
            )
            if len(previews) < preview_limit:
                previews.append(
                    {"group": group_name, "image": skin_tone.preview_rgb}
                )
        except Exception as error:  # Keep evaluation running after one bad image.
            rows.append(
                {
                    "image_id": image_number,
                    "group": group_name,
                    "pipeline_success": False,
                    "skin_R": np.nan,
                    "skin_G": np.nan,
                    "skin_B": np.nan,
                    "skin_L": np.nan,
                    "skin_a": np.nan,
                    "skin_b": np.nan,
                    "recommendation_count": 0,
                    "top_brand": None,
                    "top_product": None,
                    "top_distance": np.nan,
                    "error": str(error),
                }
            )

    return pd.DataFrame(rows), previews


def summarize_group_completion(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Summarize descriptive pipeline-completion rates by FairFace group."""

    summary = (
        evaluation.groupby("group")["pipeline_success"]
        .agg(tested="count", successful="sum", success_rate="mean")
        .reset_index()
    )
    summary["success_rate"] *= 100
    return summary.sort_values("group").reset_index(drop=True)


def simulate_colour_robustness(
    products: pd.DataFrame,
    *,
    number_of_tests: int = 500,
    noise_scale: tuple[float, float, float] = (2.0, 1.5, 1.5),
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Measure ranking stability under small synthetic perturbations in LAB."""

    rng = np.random.default_rng(random_state)
    product_lab = products[LAB_COLUMNS].to_numpy(dtype=float)
    exact_top1 = 0
    exact_top5 = 0
    top1_errors: list[float] = []
    top5_errors: list[float] = []

    for _ in range(number_of_tests):
        original_index = int(rng.integers(len(product_lab)))
        original_colour = product_lab[original_index]
        test_colour = original_colour + rng.normal(0, noise_scale)
        distances = deltaE_ciede2000(product_lab, test_colour.reshape(1, 3))
        ranking = np.argsort(distances)

        exact_top1 += int(original_index == ranking[0])
        exact_top5 += int(original_index in ranking[:5])

        first_colour = product_lab[ranking[0]]
        top1_errors.append(
            float(
                deltaE_ciede2000(
                    original_colour.reshape(1, 3), first_colour.reshape(1, 3)
                )[0]
            )
        )
        five_colours = product_lab[ranking[:5]]
        top5_errors.append(
            float(
                deltaE_ciede2000(
                    np.repeat(original_colour.reshape(1, 3), 5, axis=0),
                    five_colours,
                ).min()
            )
        )

    top1_array = np.asarray(top1_errors)
    top5_array = np.asarray(top5_errors)
    return {
        "number_of_tests": float(number_of_tests),
        "exact_top1_recovery": exact_top1 / number_of_tests,
        "exact_top5_recovery": exact_top5 / number_of_tests,
        "median_top1_delta_e": float(np.median(top1_array)),
        "mean_top1_delta_e": float(np.mean(top1_array)),
        "top1_within_delta_e_2": float(np.mean(top1_array <= 2)),
        "top1_within_delta_e_5": float(np.mean(top1_array <= 5)),
        "top5_contains_delta_e_2": float(np.mean(top5_array <= 2)),
    }
