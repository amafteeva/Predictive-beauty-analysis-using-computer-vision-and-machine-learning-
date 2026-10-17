"""Train Monk Skin Tone (MST) predictors on STW (Skin Tone in the Wild).

Uses only the publicly downloadable part of STW (Google Drive `images.zip`
+ `splits/`), which needs no account or request. The CASIA and CelebA parts
of STW are not used, so deep tones (MST 7-10) are rare -- every result is
also reported per tone band.

Pipeline: face crop -> the recommender's own skin features
(skin_calibration.extract_calibration_features) -> model -> MST 1-10.
Baseline (untrained): nearest official MST swatch by CIEDE2000.
Evaluation: STW's person-level train/test split (no person in both).

Outputs (default outputs/monk_tone/; no images are copied):
  features.csv, completion_by_mst.csv, metrics.csv, metrics_by_tone_band.csv,
  test_predictions.csv, confusion_matrix.png, mst_model.joblib, REPORT.md

Usage:
    python scripts/train_monk_tone.py --stw-root ../external_data/STW
    python scripts/train_monk_tone.py --stw-root ../external_data/STW --reuse-features
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_matcher.face import create_face_landmarker  # noqa: E402
from foundation_matcher.monk_tone import (  # noqa: E402
    MST_CLASSES,
    build_models,
    metrics_by_band,
    nearest_mst,
    tone_band,
    tone_metrics,
)
from foundation_matcher.skin_calibration import (  # noqa: E402
    FEATURE_COLUMNS,
    extract_calibration_features,
)

INK, INK_2 = "#0b0b0b", "#52514e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stw-root", default=str(PROJECT_ROOT.parent / "external_data/STW")
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/monk_tone"))
    parser.add_argument(
        "--model-path", default=str(PROJECT_ROOT / "models/face_landmarker.task")
    )
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only N images (smoke test)"
    )
    return parser.parse_args()


def build_image_table(root: Path) -> pd.DataFrame:
    labels = pd.read_csv(root / "splits/all_annotated_data.csv")
    labels["key"] = labels["new_tokens"] + labels["paths"].map(lambda p: Path(p).stem)
    by_key = labels.drop_duplicates("key").set_index("key")
    split_dir = root / "splits/individuals_train_test_splits"
    train_people = set(pd.read_csv(split_dir / "train.csv")["new_tokens"])
    test_people = set(pd.read_csv(split_dir / "test.csv")["new_tokens"])
    overlap = train_people & test_people
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} people are in both STW train and test splits"
        )
    rows = []
    for path in sorted((root / "images/all_cropped_images_by_class").glob("*/*.jpg")):
        key = path.name.rsplit("_face_", 1)[0]
        if key not in by_key.index:
            continue
        label = by_key.loc[key]
        person = label["new_tokens"]
        split = (
            "train"
            if person in train_people
            else "test"
            if person in test_people
            else "unassigned"
        )
        rows.append(
            {
                "image": str(path.relative_to(root)),
                "person": person,
                "dataset": label["dataset"],
                "mst": int(label["class"]),
                "split": split,
            }
        )
    table = pd.DataFrame(rows)
    mismatched = (
        table["image"].map(lambda p: int(Path(p).parent.name)) != table["mst"]
    ).sum()
    if mismatched:
        raise RuntimeError(
            f"{mismatched} images sit in a folder that disagrees with their label"
        )
    return table


def extract_all(table: pd.DataFrame, root: Path, model_path: str) -> pd.DataFrame:
    landmarker = create_face_landmarker(model_path)
    rows, start = [], time.time()
    try:
        for i, record in enumerate(table.to_dict("records"), 1):
            try:
                rows.append(
                    {
                        **record,
                        **extract_calibration_features(
                            root / record["image"], landmarker
                        ),
                        "extraction_error": None,
                    }
                )
            except Exception as error:
                rows.append({**record, "extraction_error": str(error)})
            if i % 1000 == 0:
                print(
                    f"  {i:,}/{len(table):,} images ({time.time() - start:.0f}s)",
                    flush=True,
                )
    finally:
        landmarker.close()
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root, out = Path(args.stw_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features_path = out / "features.csv"

    if args.reuse_features and features_path.exists():
        data = pd.read_csv(features_path)
    else:
        table = build_image_table(root)
        if args.limit:
            table = pd.concat(
                g.sample(min(len(g), args.limit // 2), random_state=0)
                for _, g in table.groupby("split")
            )
        print(
            f"Images: {len(table):,} ({table['split'].value_counts().to_dict()}); "
            "extracting features ..."
        )
        data = extract_all(table, root, args.model_path)
        data.to_csv(features_path, index=False)

    data["tone_band"] = data["mst"].map(tone_band)
    ok = data["extraction_error"].isna()
    completion = data.groupby("mst").agg(
        images=("image", "size"),
        people=("person", "nunique"),
        completed=("extraction_error", lambda e: e.isna().mean()),
    )
    completion.to_csv(out / "completion_by_mst.csv")

    usable = data[ok & data["split"].isin(["train", "test"])]
    train, test = (
        usable[usable["split"] == "train"],
        usable[usable["split"] == "test"].copy(),
    )
    X_train, X_test = (
        train[list(FEATURE_COLUMNS)].to_numpy(float),
        test[list(FEATURE_COLUMNS)].to_numpy(float),
    )

    rows = []
    test["pred_baseline_nearest_swatch"] = nearest_mst(
        test[["est_L", "est_a", "est_b"]].to_numpy(float)
    )
    rows.append(
        {
            "model": "baseline_nearest_swatch (untrained)",
            **tone_metrics(test["mst"], test["pred_baseline_nearest_swatch"]),
        }
    )
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, train["mst"].to_numpy(int))
        test[f"pred_{name}"] = model.predict(X_test).astype(int)
        rows.append({"model": name, **tone_metrics(test["mst"], test[f"pred_{name}"])})
        fitted[name] = model
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)

    # Select on per-class-averaged error so rare deep tones count as much as light ones.
    best = metrics.iloc[1:].sort_values("macro_mae_mst_steps").iloc[0]["model"]
    band_tables = []
    for column, label in [
        ("pred_baseline_nearest_swatch", "baseline_nearest_swatch"),
        (f"pred_{best}", best),
    ]:
        band_tables.append(metrics_by_band(test, column).assign(model=label))
    bands = pd.concat(band_tables)
    bands.to_csv(out / "metrics_by_tone_band.csv", index=False)
    keep = ["image", "person", "dataset", "mst", "tone_band"] + [
        c for c in test.columns if c.startswith("pred_")
    ]
    test[keep].to_csv(out / "test_predictions.csv", index=False)
    joblib.dump(
        {"model": fitted[best], "features": FEATURE_COLUMNS, "name": best},
        out / "mst_model.joblib",
    )

    cm = confusion_matrix(test["mst"], test[f"pred_{best}"], labels=list(MST_CLASSES))
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if share[i, j] > 0.5 else INK,
                )
    ax.set_xticks(range(10), MST_CLASSES)
    ax.set_yticks(range(10), MST_CLASSES)
    ax.set_xlabel("Predicted MST")
    ax.set_ylabel("Annotated MST")
    ax.set_title(
        f"Monk tone prediction ({best}), held-out people",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=22,
    )
    ax.annotate(
        f"{len(test):,} test images · {test['person'].nunique():,} people · "
        "colour = share of row",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=300, facecolor="white")
    plt.close(fig)

    def md(frame: pd.DataFrame) -> list[str]:
        frame = frame.round(3)
        return [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "|" + "---|" * len(frame.columns),
            *["| " + " | ".join(map(str, r)) + " |" for r in frame.values],
        ]

    lines = [
        "# Monk Skin Tone prediction (STW public subset)",
        "",
        f"Images: {len(data):,} · feature extraction failed for {int((~ok).sum()):,} "
        f"({(~ok).mean():.1%}) · train {len(train):,} images / "
        f"{train['person'].nunique():,} people · test {len(test):,} images / "
        f"{test['person'].nunique():,} people (person-disjoint).",
        "",
        "Only the openly downloadable STW images are used (no CASIA, no CelebA), so",
        "deep",
        "tones are rare; see the per-band table before drawing conclusions about them.",
        "",
        "## Pipeline completion by annotated tone",
        "",
        *md(completion.reset_index()),
        "",
        "## Held-out test metrics",
        "",
        *md(metrics),
        "",
        "Best trained model (lowest per-class-averaged error, "
        f"`macro_mae_mst_steps`): **{best}**.",
        "",
        "## By tone band",
        "",
        *md(
            bands[
                [
                    "model",
                    "tone_band",
                    "images",
                    "people",
                    "accuracy",
                    "off_by_one_accuracy",
                    "mean_abs_error_mst_steps",
                ]
            ]
        ),
        "",
        "MST labels are annotators' judgements of perceived tone, not colorimeter",
        "readings; this measures how well the extracted skin colour reproduces those",
        "judgements, not",
        "foundation-shade accuracy. The model is not used by the recommender.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(metrics.round(3).to_string(index=False))
    print(bands.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
