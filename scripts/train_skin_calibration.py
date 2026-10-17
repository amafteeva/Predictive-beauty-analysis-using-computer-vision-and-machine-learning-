"""Train and evaluate a learned skin-colour calibration (e.g. on CHROMA-FIT).

Input is a manifest CSV with one row per photo:

    image_path,participant_id,measured_L,measured_a,measured_b[,environment]

``measured_*`` is the colorimeter CIELAB reading (CHROMA-FIT provides
forehead and forearm readings; use the forehead). Optional columns such as
``environment`` (indoor/outdoor) are reported as extra groups. Run
``--write-template`` to get an empty manifest to fill in.

Outputs (default outputs/skin_calibration/):
  features.csv            pipeline features per photo (+ extraction errors)
  cv_predictions.csv      participant-grouped out-of-fold predictions
  summary.csv             ΔE before vs after calibration, overall / by tone / by group
  calibration_model.joblib  model refit on all participants (opt-in use only)
  calibration_delta_e.png   before/after chart
  REPORT.md

Usage:
    python scripts/train_skin_calibration.py \
        --write-template data/chroma_fit_manifest.csv
    python scripts/train_skin_calibration.py --manifest data/chroma_fit_manifest.csv \\
        --image-root /path/to/CHROMA-FIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_matcher.face import create_face_landmarker  # noqa: E402
from foundation_matcher.skin_calibration import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_calibration_model,
    build_feature_table,
    cross_validate_calibration,
    summarize_calibration,
)

REQUIRED = ("image_path", "participant_id", *TARGET_COLUMNS)
BLUE, ORANGE, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--manifest", help="CSV with image_path, participant_id, measured_L/a/b"
    )
    parser.add_argument(
        "--image-root", default=None, help="Base folder for relative image paths"
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs/skin_calibration")
    )
    parser.add_argument(
        "--model-path", default=str(PROJECT_ROOT / "models/face_landmarker.task")
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--group-columns", nargs="*", default=["environment"])
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse features.csv from a previous run instead of re-extracting",
    )
    parser.add_argument(
        "--write-template",
        metavar="PATH",
        help="Write an empty manifest template and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        Path(args.write_template).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[*REQUIRED, "environment"]).to_csv(
            args.write_template, index=False
        )
        print(f"Wrote template {args.write_template}")
        return
    if not args.manifest:
        raise SystemExit("--manifest is required (or use --write-template).")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    missing = [c for c in REQUIRED if c not in manifest.columns]
    if missing:
        raise SystemExit(f"Manifest is missing columns: {missing}")

    features_path = out / "features.csv"
    if args.reuse_features and features_path.exists():
        table = pd.read_csv(features_path)
    else:
        landmarker = create_face_landmarker(args.model_path)
        try:
            table = build_feature_table(
                manifest,
                landmarker,
                root=Path(args.image_root) if args.image_root else None,
            )
        finally:
            landmarker.close()
        table.to_csv(features_path, index=False)
    failed = int(table["extraction_error"].notna().sum())
    print(f"Photos: {len(table)}; feature extraction failed for {failed}")

    results = cross_validate_calibration(table, n_splits=args.n_splits)
    results.to_csv(out / "cv_predictions.csv", index=False)
    summary = summarize_calibration(results, group_columns=tuple(args.group_columns))
    summary.to_csv(out / "summary.csv", index=False)

    final = build_calibration_model().fit(
        results[list(FEATURE_COLUMNS)].to_numpy(float),
        results[list(TARGET_COLUMNS)].to_numpy(float),
    )
    joblib.dump(
        {"model": final, "features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS},
        out / "calibration_model.joblib",
    )

    # Chart: median ΔE before vs after, overall and by measured tone tertile
    plot = summary[summary["group"].isin(["all", "measured_tone_tertile"])]
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for j, (col, lab, color) in enumerate(
        [
            ("baseline_median_delta_e", "Current pipeline", BLUE),
            ("calibrated_median_delta_e", "Learned calibration", ORANGE),
        ]
    ):
        bars = ax.bar(
            x + (j - 0.5) * 0.38, plot[col], width=0.36, color=color, label=lab
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(
        x,
        [
            v if g == "measured_tone_tertile" else "All photos"
            for g, v in zip(plot["group"], plot["value"], strict=True)
        ],
    )
    ax.set_ylabel("Median CIEDE2000 ΔE to measured skin colour")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title(
        "Skin-colour error before and after learned calibration",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=24,
    )
    ax.annotate(
        f"{results['participant_id'].nunique()} participants · {args.n_splits}-fold, "
        "grouped by participant (held-out people only)",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(out / "calibration_delta_e.png", dpi=300, facecolor="white")
    plt.close(fig)

    overall = summary[summary["group"] == "all"].iloc[0]
    table_md = summary.round(3)
    lines = [
        "# Learned skin-colour calibration",
        "",
        f"Manifest: `{args.manifest}` · photos {len(table)} · "
        f"extraction failures {failed} · photos used {len(results)} · "
        f"participants {results['participant_id'].nunique()}.",
        "",
        f"Evaluation: {args.n_splits}-fold cross-validation grouped by participant;",
        "the baseline is the current pipeline's estimate (`extract_skin_tone`)",
        "compared directly with the",
        "colorimeter reading.",
        "",
        "| " + " | ".join(table_md.columns) + " |",
        "|" + "---|" * len(table_md.columns),
        *["| " + " | ".join(str(v) for v in row) + " |" for row in table_md.values],
        "",
        f"Overall median ΔE: {overall['baseline_median_delta_e']:.2f} (current) → "
        f"{overall['calibrated_median_delta_e']:.2f} (calibrated).",
        "",
        "This measures skin-colour accuracy, not foundation-shade accuracy. The",
        "model is not used by `recommend_foundations` unless a caller applies",
        "`calibrate_lab` explicitly.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
