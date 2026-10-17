"""Golden-output regression check for the shade-matching pipeline.

Runs before and after the review-classification experiment is added, to
prove that MediaPipe skin extraction, CIELAB conversion, and brute-force
CIEDE2000 shade ranking are byte-for-byte unaffected by it. Run with
``--mode before`` prior to making changes and ``--mode after`` once done;
the second run diffs itself against the first snapshot.

Usage:
    python scripts/verify_shade_pipeline.py --mode before
    python scripts/verify_shade_pipeline.py --mode after
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_matcher.data import load_foundation_catalog  # noqa: E402
from foundation_matcher.face import (  # noqa: E402
    create_face_landmarker,
    extract_skin_tone,
)
from foundation_matcher.recommender import recommend_foundations  # noqa: E402

SNAPSHOT_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "shade_pipeline_verification_before.json"
)

FIXED_LAB_QUERIES = [
    (45.0, 12.0, 18.0),
    (60.0, 8.0, 22.0),
    (30.0, 15.0, 10.0),
    (70.0, 5.0, 15.0),
    (50.0, 20.0, 25.0),
]

TEST_PHOTOS = sorted((PROJECT_ROOT / "data").glob("example_selfie*.jpg"))

LATENCY_REPEATS = 30


def _time_repeated(fn, repeats: int = LATENCY_REPEATS) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "p95_seconds": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "mean_seconds": statistics.mean(samples),
    }


def capture_snapshot() -> dict:
    products = load_foundation_catalog()

    shade_results = {}
    for query in FIXED_LAB_QUERIES:
        matches = recommend_foundations(np.array(query), products, top_n=5)
        shade_results[str(query)] = {
            "brand_product": [
                f"{row.brand}||{row.product}" for row in matches.itertuples()
            ],
            "color_distance": [round(float(d), 10) for d in matches["color_distance"]],
        }

    latency = _time_repeated(
        lambda: recommend_foundations(np.array(FIXED_LAB_QUERIES[0]), products, top_n=5)
    )

    face_results = {}
    landmarker = None
    if TEST_PHOTOS:
        model_path = PROJECT_ROOT / "models" / "face_landmarker.task"
        if model_path.exists():
            landmarker = create_face_landmarker(model_path)
            for photo in TEST_PHOTOS:
                try:
                    skin_tone = extract_skin_tone(photo, landmarker)
                    face_results[photo.name] = {
                        "rgb": [int(v) for v in skin_tone.rgb],
                        "lab": [round(float(v), 6) for v in skin_tone.lab],
                        "face_count": skin_tone.face_count,
                    }
                except ValueError as error:
                    face_results[photo.name] = {"error": str(error)}

    catalog_hash = int(
        pd.util.hash_pandas_object(products[["brand", "product", "hex"]]).sum()
    )
    return {
        "catalog_rows": int(len(products)),
        "catalog_hash": catalog_hash,
        "shade_results": shade_results,
        "recommend_latency": latency,
        "face_results": face_results,
    }


def _import_check() -> dict[str, bool]:
    """Confirm core modules import without any transformer dependency."""

    import subprocess

    script = (
        "import sys; sys.path.insert(0, 'src'); "
        "import foundation_matcher.face, foundation_matcher.recommender, "
        "foundation_matcher.color, foundation_matcher.data; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    succeeded = result.returncode == 0 and "OK" in result.stdout
    return {
        "core_imports_without_transformers": succeeded,
        "stderr": result.stderr[-2000:] if result.returncode != 0 else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after"], required=True)
    args = parser.parse_args()

    snapshot = capture_snapshot()
    snapshot["import_check"] = _import_check()

    if args.mode == "before":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
        print(f"Saved baseline snapshot to {SNAPSHOT_PATH}")
        return

    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"No baseline snapshot at {SNAPSHOT_PATH}; run with --mode before first."
        )
    before = json.loads(SNAPSHOT_PATH.read_text())

    problems = []
    if before["catalog_rows"] != snapshot["catalog_rows"]:
        problems.append("catalog row count changed")
    if before["catalog_hash"] != snapshot["catalog_hash"]:
        problems.append("catalog content changed")
    if before["shade_results"] != snapshot["shade_results"]:
        problems.append("shade ranking or CIEDE2000 distances changed")
    if before["face_results"] != snapshot["face_results"]:
        problems.append("face-extraction results changed")
    if not snapshot["import_check"]["core_imports_without_transformers"]:
        problems.append(
            "core modules failed to import cleanly without transformers: "
            + snapshot["import_check"]["stderr"]
        )

    before_latency = before["recommend_latency"]["median_seconds"]
    after_latency = snapshot["recommend_latency"]["median_seconds"]
    latency_ratio = after_latency / before_latency if before_latency else float("inf")
    if latency_ratio > 1.5:
        problems.append(
            f"recommend_foundations median latency regressed {latency_ratio:.2f}x "
            f"({before_latency * 1000:.3f}ms -> {after_latency * 1000:.3f}ms)"
        )

    print(f"Before median latency: {before_latency * 1000:.4f} ms")
    print(
        f"After median latency:  {after_latency * 1000:.4f} ms ({latency_ratio:.2f}x)"
    )

    if problems:
        print("\nVERIFICATION FAILED:")
        for problem in problems:
            print(f" - {problem}")
        raise SystemExit(1)

    print("\nVERIFICATION PASSED: shade pipeline output and performance are unchanged.")


if __name__ == "__main__":
    main()
