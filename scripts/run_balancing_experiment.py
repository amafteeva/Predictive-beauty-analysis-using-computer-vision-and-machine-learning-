"""Compare ways of handling the review-label imbalance (~74% positive).

Strategies (TF-IDF + logistic regression, text only):
  none          no correction
  class_weight  class_weight="balanced" (the current default)
  undersample   random undersampling of the majority class, training rows only
  oversample    random oversampling of the minority class, training rows only

Every strategy uses the saved leakage-safe split from
outputs/review_experiment/splits/manifest.csv. Only training rows are ever
resampled; validation (for choosing C) and test keep the real class mix, so
test metrics describe real reviews. Resampling strategies are repeated over
several resampling seeds. Results go to outputs/review_balancing/.

Usage:
    python scripts/run_balancing_experiment.py
    python scripts/run_balancing_experiment.py \\
        --reviews-path data/cleaned_makeup_reviews.csv \\
        --products-path data/cleaned_makeup_products.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_matcher.config import (  # noqa: E402
    LUXXIFY_PRODUCT_URL,
    LUXXIFY_REVIEW_URL,
)
from foundation_matcher.review_model import (  # noqa: E402
    BALANCING_STRATEGIES,
    attach_product_metadata,
    build_review_classifier,
    prepare_review_data,
    rebalance_training_set,
)

EXPERIMENT = PROJECT_ROOT / "outputs" / "review_experiment"
BLUE, ORANGE, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
LABELS = {
    "none": "No correction",
    "class_weight": "Class weights\n(current default)",
    "undersample": "Undersample\nmajority",
    "oversample": "Oversample\nminority",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reviews-path", default=LUXXIFY_REVIEW_URL)
    parser.add_argument("--products-path", default=LUXXIFY_PRODUCT_URL)
    parser.add_argument("--manifest", default=str(EXPERIMENT / "splits/manifest.csv"))
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs/review_balancing")
    )
    parser.add_argument(
        "--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0]
    )
    parser.add_argument("--resample-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def test_metrics(y_true, predicted, probability) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "macro_f1": f1_score(y_true, predicted, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "roc_auc": roc_auc_score(y_true, probability),
        "negative_precision": precision_score(y_true, predicted, pos_label=0),
        "negative_recall": recall_score(y_true, predicted, pos_label=0),
        "negative_f1": f1_score(y_true, predicted, pos_label=0),
        "positive_recall": recall_score(y_true, predicted, pos_label=1),
        "predicted_positive_rate": float(np.mean(predicted)),
    }


def fit(train: pd.DataFrame, C: float, class_weight, max_iter: int):
    model = build_review_classifier(C=C, class_weight=class_weight, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train[["review_text"]], train["positive_review"])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing reviews (same steps as run_review_experiment) ...")
    prepared = attach_product_metadata(
        prepare_review_data(
            pd.read_csv(args.reviews_path, low_memory=False),
            text_column="comments",
            rating_column="rating",
            group_column="product_link_id",
        ),
        pd.read_csv(args.products_path, low_memory=False),
    )
    prepared["review_id"] = prepared["review_id"].astype(str)
    manifest = pd.read_csv(args.manifest, dtype={"review_id": str})
    data = prepared.merge(manifest, on="review_id", how="inner", validate="one_to_one")
    if len(data) != len(manifest):
        raise RuntimeError(
            f"Only {len(data)} of {len(manifest)} manifest rows were found; the "
            "review export differs from the one the split was built on."
        )
    splits = {
        name: data[data["split"] == name] for name in ("train", "validation", "test")
    }
    mix = {
        name: {
            "rows": len(frame),
            "positive_rate": round(frame["positive_review"].mean(), 4),
        }
        for name, frame in splits.items()
    }
    print(json.dumps(mix, indent=2))

    search_rows, test_rows = [], []
    for strategy in BALANCING_STRATEGIES:
        seeds = (
            args.resample_seeds if strategy in ("undersample", "oversample") else [0]
        )
        class_weight = "balanced" if strategy == "class_weight" else None
        for seed in seeds:
            train = rebalance_training_set(
                splits["train"], strategy=strategy, random_state=seed
            )
            best = None
            for C in args.c_values:
                model, converged = fit(train, C, class_weight, args.max_iter)
                val_pred = model.predict(splits["validation"][["review_text"]])
                val_f1 = f1_score(
                    splits["validation"]["positive_review"], val_pred, average="macro"
                )
                search_rows.append(
                    {
                        "strategy": strategy,
                        "resample_seed": seed,
                        "C": C,
                        "val_macro_f1": val_f1,
                        "converged": converged,
                        "train_rows": len(train),
                        "train_positive_rate": train["positive_review"].mean(),
                    }
                )
                if best is None or val_f1 > best[0]:
                    best = (val_f1, C, model, converged)
            val_f1, C, model, converged = best
            test = splits["test"]
            probability = model.predict_proba(test[["review_text"]])[:, 1]
            predicted = (probability >= 0.5).astype(int)
            test_rows.append(
                {
                    "strategy": strategy,
                    "resample_seed": seed,
                    "selected_C": C,
                    "val_macro_f1": val_f1,
                    "converged": converged,
                    "train_rows": len(train),
                    "train_positive_rate": train["positive_review"].mean(),
                    **test_metrics(test["positive_review"], predicted, probability),
                }
            )
            print(
                f"{strategy:13s} seed {seed}: C={C:g} test macro-F1 "
                f"{test_rows[-1]['macro_f1']:.4f}"
            )

    search = pd.DataFrame(search_rows)
    per_run = pd.DataFrame(test_rows)
    search.to_csv(out / "validation_search.csv", index=False)
    per_run.to_csv(out / "test_results_per_run.csv", index=False)
    metric_cols = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "roc_auc",
        "negative_recall",
        "negative_precision",
        "positive_recall",
        "predicted_positive_rate",
    ]
    summary = per_run.groupby("strategy", sort=False).agg(
        runs=("resample_seed", "size"),
        train_rows=("train_rows", "first"),
        train_positive_rate=("train_positive_rate", "first"),
        selected_C=("selected_C", lambda c: ", ".join(sorted({f"{v:g}" for v in c}))),
        all_converged=("converged", "all"),
        **{m: (m, "mean") for m in metric_cols},
        macro_f1_sd=("macro_f1", "std"),
    )
    summary = summary.reset_index()
    summary.to_csv(out / "comparison.csv", index=False)

    # Consistency check against the saved experiment (same split, same model).
    saved = pd.read_csv(EXPERIMENT / "metrics/comparison.csv").set_index("model")
    reproduced = float(summary.set_index("strategy").loc["class_weight", "macro_f1"])
    saved_f1 = float(saved.loc["logistic_regression_text", "macro_f1"])
    consistency = {
        "class_weight_macro_f1_now": reproduced,
        "saved_logistic_regression_text_macro_f1": saved_f1,
        "abs_difference": abs(reproduced - saved_f1),
        "split_class_mix": mix,
    }
    (out / "consistency_check.json").write_text(json.dumps(consistency, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(summary))
    for j, (col, lab, color) in enumerate(
        [
            ("macro_f1", "Macro-F1", BLUE),
            ("negative_recall", "Negative-class recall", ORANGE),
        ]
    ):
        bars = ax.bar(
            x + (j - 0.5) * 0.38, summary[col], width=0.36, color=color, label=lab
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{b.get_height():.3f}",
                ha="center",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x, [LABELS[s] for s in summary["strategy"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out test score (5,648 reviews, real class mix)")
    ax.legend(frameon=False, loc="upper right", ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "Handling the 74% / 26% label imbalance",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=24,
    )
    ax.annotate(
        "TF-IDF + logistic regression · only training rows resampled · "
        "resampling averaged over 3 seeds",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(out / "balancing_comparison.png", dpi=300, facecolor="white")
    plt.close(fig)

    table = summary[
        ["strategy", "train_rows", "train_positive_rate", "selected_C", *metric_cols]
    ].round(4)
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    lines = [
        "# Handling the review-label imbalance",
        "",
        "Labels: positive = 4-5 stars. Real class mix per split:",
        "",
        *[
            f"- {k}: {v['rows']:,} rows, {v['positive_rate']:.1%} positive"
            for k, v in mix.items()
        ],
        "",
        "Only training rows are resampled; validation (to choose C) and test keep",
        "the real mix, so test metrics describe real reviews. Resampling",
        "strategies are averaged over seeds "
        f"{', '.join(map(str, args.resample_seeds))}.",
        "",
        header,
        rule,
        *body,
        "",
        f"Best test macro-F1: **{best['strategy']}** ({best['macro_f1']:.4f}).",
        "",
        "Consistency check: the class-weight strategy gives test macro-F1 "
        f"{reproduced:.4f}; the saved experiment reported {saved_f1:.4f} "
        f"(difference {abs(reproduced - saved_f1):.4f}).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(table.to_string(index=False))
    print(json.dumps(consistency, indent=2))


if __name__ == "__main__":
    main()
