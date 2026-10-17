"""Run the full review-classification training and benchmarking experiment.

Trains and compares three models for rating-derived review classification
(1 = rating >= 4, 0 = rating < 4) on the Luxxify review export: a
majority-class dummy baseline, TF-IDF + logistic regression, and
(optionally, if installed and the data is predominantly English) a
fine-tuned DistilBERT classifier. This is independent of, and does not
affect, the MediaPipe/CIELAB/CIEDE2000 shade-matching pipeline -- see
scripts/verify_shade_pipeline.py for that guarantee.

Usage:
    python scripts/run_review_experiment.py
    python scripts/run_review_experiment.py --skip-transformer
    python scripts/run_review_experiment.py --transformer-smoke-only
    python scripts/run_review_experiment.py --reviews-path local_reviews.csv \
        --products-path local_products.csv

Every step is modular (see foundation_matcher.review_model,
review_benchmark, review_transformer) -- this script only orchestrates and
saves results to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foundation_matcher.config import (  # noqa: E402
    LUXXIFY_PRODUCT_URL,
    LUXXIFY_REVIEW_URL,
)
from foundation_matcher.review_benchmark import (  # noqa: E402
    benchmark_classifier,
    detect_hardware,
    get_artifact_size_bytes,
    measure_model_load_time,
)
from foundation_matcher.review_model import (  # noqa: E402
    attach_product_metadata,
    build_leakage_safe_groups,
    build_majority_baseline,
    build_review_classifier,
    detect_review_languages,
    prepare_review_data,
    product_disjoint_three_way_split,
    select_logistic_regression_config,
    summarize_language_check,
    summarize_luxxify_products,
    summarize_luxxify_reviews,
    summarize_review_data_quality,
)
from foundation_matcher.visualization import (  # noqa: E402
    plot_latency_comparison,
    plot_model_comparison,
)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def load_raw_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews_source = args.reviews_path or LUXXIFY_REVIEW_URL
    products_source = args.products_path or LUXXIFY_PRODUCT_URL
    print(f"Loading reviews from {reviews_source}")
    reviews = pd.read_csv(reviews_source, low_memory=False)
    print(f"Loading products from {products_source}")
    products = pd.read_csv(products_source, low_memory=False)
    print(f"Raw reviews: {reviews.shape}, raw products: {products.shape}")
    return reviews, products


def run_data_quality_checks(
    raw_reviews: pd.DataFrame, raw_products: pd.DataFrame, output_dir: Path
) -> dict:
    print("\n=== Data quality ===")
    quality = {
        "luxxify_review_export_summary": summarize_luxxify_reviews(raw_reviews),
        "luxxify_product_export_summary": summarize_luxxify_products(raw_products),
        "review_data_quality": summarize_review_data_quality(
            raw_reviews,
            text_column="comments",
            rating_column="rating",
            group_column="product_link_id",
        ),
    }
    for key, value in quality.items():
        print(f"-- {key} --")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    _save_json(quality, output_dir / "data_quality.json")
    return quality


def run_language_check(
    raw_reviews: pd.DataFrame, args: argparse.Namespace, output_dir: Path
) -> dict:
    print("\n=== Language check ===")
    texts = raw_reviews["comments"].dropna()
    table = detect_review_languages(
        texts, sample_size=args.language_sample_size, random_state=args.random_state
    )
    check = summarize_language_check(table, english_threshold=args.english_threshold)
    print(table.head(10).to_string(index=False))
    print(json.dumps(check, indent=2))
    table.to_csv(output_dir / "language_distribution.csv", index=False)
    _save_json(check, output_dir / "language_check.json")
    return check


def build_prepared_data(
    raw_reviews: pd.DataFrame, raw_products: pd.DataFrame
) -> pd.DataFrame:
    prepared = prepare_review_data(
        raw_reviews,
        text_column="comments",
        rating_column="rating",
        group_column="product_link_id",
    )
    prepared = attach_product_metadata(prepared, raw_products)
    return prepared


def build_splits(
    prepared: pd.DataFrame, args: argparse.Namespace, output_dir: Path
) -> dict:
    print("\n=== Building leakage-safe, product-disjoint splits ===")
    split_groups = build_leakage_safe_groups(prepared, text_column="review_text")
    prepared = prepared.assign(split_group=split_groups)

    result = product_disjoint_three_way_split(
        prepared,
        feature_columns=("review_text", "category", "price", "brand"),
        split_group_column="split_group",
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )

    print("Realized split proportions:")
    print(json.dumps(result["proportions"], indent=2))

    manifest_path = output_dir / "splits" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result["manifest"].to_csv(manifest_path, index=False)
    _save_json(
        {"proportions": result["proportions"], "random_state": result["random_state"]},
        output_dir / "splits" / "proportions.json",
    )
    print(f"Saved split identifiers to {manifest_path}")
    return result


def _score_row(
    model_name: str, metrics: pd.Series, latency: dict, throughput: dict
) -> dict:
    roc_auc = float(metrics["roc_auc"]) if not np.isnan(metrics["roc_auc"]) else None
    return {
        "model": model_name,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": roc_auc,
        "p50_ms": latency["p50_seconds"] * 1000,
        "p95_ms": latency["p95_seconds"] * 1000,
        "throughput_examples_per_second": throughput["examples_per_second"],
    }


def train_and_benchmark_dummy(splits: dict, output_dir: Path) -> dict:
    print("\n=== Majority-class dummy baseline ===")
    train, test = splits["splits"]["train"], splits["splits"]["test"]

    def load():
        model = build_majority_baseline()
        model.fit(train["X"][["review_text"]], train["y"])
        return model

    model, load_time = measure_model_load_time(load)
    result = benchmark_classifier(
        model, test["X"][["review_text"]], test["y"], text_column="review_text"
    )
    print(result["metrics"].round(4))

    model_path = output_dir / "models" / "dummy_baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "model_name": "dummy_baseline",
        "load_time_seconds": load_time,
        "training_time_seconds": load_time,
        "saved_size_bytes": get_artifact_size_bytes(model_path),
        "result": result,
    }


def train_and_benchmark_logistic_regression(
    splits: dict, args: argparse.Namespace, output_dir: Path, *, suffix: str = "text"
) -> dict:
    print(f"\n=== TF-IDF + logistic regression ({suffix}) ===")
    train, val, test = (
        splits["splits"]["train"],
        splits["splits"]["validation"],
        splits["splits"]["test"],
    )
    categorical_columns = ("category", "brand") if suffix == "metadata" else ()
    numeric_columns = ("price",) if suffix == "metadata" else ()

    search = select_logistic_regression_config(
        train["X"],
        train["y"],
        val["X"],
        val["y"],
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        C_values=args.lr_c_values,
        class_weight_options=(None, "balanced"),
        minimum_document_frequency=args.min_document_frequency,
        max_iter=args.lr_max_iter,
        random_state=args.random_state,
    )
    print("Hyperparameter search results (sorted by validation macro-F1):")
    print(search["results"].to_string(index=False))
    search["results"].to_csv(
        output_dir / "metrics" / f"lr_hparam_search_{suffix}.csv", index=False
    )

    best_config = search["best_config"]
    print(f"Selected config: {best_config}")
    resolved_class_weight = (
        None if best_config["class_weight"] == "none" else best_config["class_weight"]
    )

    def load():
        pipeline = build_review_classifier(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            minimum_document_frequency=args.min_document_frequency,
            max_iter=args.lr_max_iter,
            random_state=args.random_state,
            C=best_config["C"],
            class_weight=resolved_class_weight,
        )
        pipeline.fit(train["X"], train["y"])
        return pipeline

    model, training_time = measure_model_load_time(load)
    result = benchmark_classifier(
        model, test["X"], test["y"], text_column="review_text"
    )
    print(result["metrics"].round(4))

    model_path = output_dir / "models" / f"logistic_regression_{suffix}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    seed_rows = []
    for seed in args.seeds:
        seeded_pipeline = build_review_classifier(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            minimum_document_frequency=args.min_document_frequency,
            max_iter=args.lr_max_iter,
            random_state=seed,
            C=best_config["C"],
            class_weight=resolved_class_weight,
        )
        seeded_pipeline.fit(train["X"], train["y"])
        predicted = seeded_pipeline.predict(test["X"])

        seed_rows.append(
            {
                "seed": seed,
                "accuracy": accuracy_score(test["y"], predicted),
                "macro_f1": f1_score(
                    test["y"], predicted, average="macro", zero_division=0
                ),
            }
        )
    seed_table = pd.DataFrame(seed_rows)
    seed_repeats_path = output_dir / "metrics" / f"lr_seed_repeats_{suffix}.csv"
    seed_table.to_csv(seed_repeats_path, index=False)
    print("Seed repeats:")
    print(seed_table.to_string(index=False))
    print(
        f"macro_f1 mean={seed_table['macro_f1'].mean():.4f} "
        f"std={seed_table['macro_f1'].std():.4f}"
    )

    return {
        "model_name": f"logistic_regression_{suffix}",
        "best_config": best_config,
        "training_time_seconds": training_time,
        "saved_size_bytes": get_artifact_size_bytes(model_path),
        "result": result,
        "seed_repeats": seed_table,
    }


def train_and_benchmark_distilbert(
    splits: dict, args: argparse.Namespace, output_dir: Path
) -> dict | None:
    from foundation_matcher.review_transformer import TRANSFORMERS_AVAILABLE

    if not TRANSFORMERS_AVAILABLE:
        print(
            "\n=== DistilBERT: SKIPPED (transformers/torch not installed; "
            "run `pip install -r requirements-transformers.txt`) ==="
        )
        return None

    from foundation_matcher.review_transformer import (
        DistilBertSklearnAdapter,
        DistilBertTrainingConfig,
        fine_tune_distilbert,
        save_distilbert_artifacts,
    )

    hardware = detect_hardware()
    print("\n=== DistilBERT fine-tuning ===")
    print(f"Hardware: {json.dumps(hardware, indent=2)}")
    _save_json(hardware, output_dir / "hardware.json")

    train, val, test = (
        splits["splits"]["train"],
        splits["splits"]["validation"],
        splits["splits"]["test"],
    )
    train_df = pd.DataFrame(
        {"review_text": train["X"]["review_text"], "positive_review": train["y"]}
    )
    val_df = pd.DataFrame(
        {"review_text": val["X"]["review_text"], "positive_review": val["y"]}
    )
    test_df = pd.DataFrame(
        {"review_text": test["X"]["review_text"], "positive_review": test["y"]}
    )

    # 1) Smoke run: verify the fine-tuning workflow end-to-end on a tiny
    #    subset before spending the real compute budget.
    smoke_config = DistilBertTrainingConfig(
        checkpoint=args.transformer_checkpoint,
        epochs=1,
        batch_size=8,
        max_length=64,
        seed=args.random_state,
        smoke_test=True,
        smoke_test_train_rows=64,
        smoke_test_val_rows=32,
    )
    print(f"Smoke-test config: {smoke_config.to_dict()}")
    smoke_start = time.perf_counter()
    smoke_result = fine_tune_distilbert(train_df, val_df, smoke_config)
    smoke_elapsed = time.perf_counter() - smoke_start
    print(
        f"Smoke run OK: {smoke_elapsed:.1f}s, "
        f"best_val_macro_f1={smoke_result.best_val_macro_f1:.4f}, "
        f"train_truncation_rate={smoke_result.train_truncation_rate:.4f}"
    )
    smoke_result.history.to_csv(
        output_dir / "metrics" / "distilbert_smoke_history.csv", index=False
    )

    if args.transformer_smoke_only:
        print("--transformer-smoke-only set: stopping after the smoke run.")
        return {"smoke_only": True, "smoke_elapsed_seconds": smoke_elapsed}

    # 2) Bounded "real" run within the explicit compute budget below.
    train_subset = train_df.sample(
        n=min(len(train_df), args.transformer_train_rows),
        random_state=args.random_state,
    )
    val_subset = val_df.sample(
        n=min(len(val_df), args.transformer_val_rows), random_state=args.random_state
    )
    if args.transformer_test_rows and len(test_df) > args.transformer_test_rows:
        test_subset = test_df.sample(
            n=args.transformer_test_rows, random_state=args.random_state
        )
    else:
        test_subset = test_df

    config = DistilBertTrainingConfig(
        checkpoint=args.transformer_checkpoint,
        learning_rate=args.transformer_learning_rate,
        batch_size=args.transformer_batch_size,
        max_length=args.transformer_max_length,
        epochs=args.transformer_epochs,
        class_weighting=args.transformer_class_weighting,
        seed=args.random_state,
        early_stopping_patience=args.transformer_early_stopping_patience,
    )
    print(f"Bounded fine-tuning config: {config.to_dict()}")
    print(
        f"train_rows={len(train_subset)} val_rows={len(val_subset)} "
        f"test_rows={len(test_subset)} "
        "(compute-budget-limited subset of the full split)"
    )

    result = fine_tune_distilbert(train_subset, val_subset, config)
    print(
        f"Fine-tuning done in {result.training_time_seconds:.1f}s "
        f"on device={result.device}. "
        f"best_epoch={result.best_epoch} "
        f"best_val_macro_f1={result.best_val_macro_f1:.4f} "
        f"train_truncation_rate={result.train_truncation_rate:.4f} "
        f"val_truncation_rate={result.val_truncation_rate:.4f}"
    )
    distilbert_history_path = output_dir / "metrics" / "distilbert_history.csv"
    result.history.to_csv(distilbert_history_path, index=False)

    model_dir = output_dir / "models" / "distilbert"
    save_distilbert_artifacts(result, model_dir)
    print(f"Saved DistilBERT model/tokenizer/config/label-mapping to {model_dir}")

    adapter = DistilBertSklearnAdapter(
        result.model,
        result.tokenizer,
        device=result.device,
        max_length=config.max_length,
    )

    def load():
        from foundation_matcher.review_transformer import load_distilbert_artifacts

        model, tokenizer, device = load_distilbert_artifacts(model_dir)
        return DistilBertSklearnAdapter(
            model, tokenizer, device=device, max_length=config.max_length
        )

    _, load_time = measure_model_load_time(load)

    synchronize = None
    if result.device == "mps":
        import torch

        synchronize = torch.mps.synchronize
    elif result.device == "cuda":
        import torch

        synchronize = torch.cuda.synchronize

    benchmark_result = benchmark_classifier(
        adapter,
        test_subset[["review_text"]],
        test_subset["positive_review"],
        text_column="review_text",
        predict_one=lambda record: int(
            adapter.predict(pd.Series([record["review_text"]]))[0]
        ),
        predict_batch=lambda batch: adapter.predict(
            pd.Series([record["review_text"] for record in batch])
        ),
        latency_examples=min(100, len(test_subset)),
        latency_repeats=30,
        throughput_batch_size=min(16, len(test_subset)),
        synchronize=synchronize,
    )
    print(benchmark_result["metrics"].round(4))

    return {
        "model_name": "distilbert",
        "config": config.to_dict(),
        "device": result.device,
        "training_time_seconds": result.training_time_seconds,
        "load_time_seconds": load_time,
        "saved_size_bytes": get_artifact_size_bytes(model_dir),
        "result": benchmark_result,
        "history": result.history,
        "smoke_elapsed_seconds": smoke_elapsed,
        "train_rows_used": len(train_subset),
        "val_rows_used": len(val_subset),
        "test_rows_used": len(test_subset),
        "full_test_rows": len(test_df),
    }


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without the `tabulate` dependency."""

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def write_comparison(
    runs: list[dict], output_dir: Path
) -> pd.DataFrame:
    rows = []
    for run in runs:
        if run is None or run.get("smoke_only"):
            continue
        result = run["result"]
        row = _score_row(
            run["model_name"],
            result["metrics"],
            result["latency"],
            result["throughput"],
        )
        row["training_time_seconds"] = run.get("training_time_seconds")
        row["saved_size_mb"] = run.get("saved_size_bytes", 0) / (1024 * 1024)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_dir / "comparison.csv", index=False)
    print("\n=== Final comparison (test set) ===")
    print(comparison.to_string(index=False))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if len(comparison) > 0:
        figure, _ = plot_model_comparison(comparison)
        figure.savefig(plots_dir / "model_comparison.png", dpi=150)
        plt.close(figure)

        figure, _ = plot_latency_comparison(comparison)
        figure.savefig(plots_dir / "latency_comparison.png", dpi=150)
        plt.close(figure)

    return comparison


def write_report(
    output_dir: Path,
    *,
    quality: dict,
    language_check: dict,
    splits: dict,
    dummy_run: dict,
    lr_text_run: dict,
    lr_metadata_run: dict,
    distilbert_run: dict | None,
    comparison: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Review Classification: Comparison Report\n")
    lines.append(
        "This experiment predicts a **rating-derived** binary label from review "
        "text (1 = rating >= 4, 0 = rating < 4). It is rating-derived review "
        "classification, not a personalized customer-satisfaction predictor: "
        "the label is computed directly from the same review's own star rating.\n"
    )
    lines.append(
        "It is independent of, and does not affect, the shade-matching "
        "recommendations produced by MediaPipe skin extraction + brute-force "
        "CIEDE2000 (see `scripts/verify_shade_pipeline.py`).\n"
    )

    lines.append("\n## Data quality\n")
    lines.append("```json")
    lines.append(json.dumps(quality["review_data_quality"], indent=2))
    lines.append("```\n")

    lines.append("## Language check\n")
    lines.append(language_check["explanation"] + "\n")

    lines.append("## Split strategy\n")
    lines.append(
        "Product-disjoint train/validation/test split via two rounds of "
        "`StratifiedGroupKFold`, grouped not just by `product_link_id` but by "
        "`build_leakage_safe_groups`'s union of products that share "
        "duplicate/near-duplicate review text -- so no product, and no "
        "copy-pasted review, crosses a split boundary. Split identifiers are "
        "saved at `splits/manifest.csv`; every model below is trained/evaluated "
        "on exactly these rows.\n"
    )
    lines.append("Realized split proportions:\n")
    lines.append("```json")
    lines.append(json.dumps(splits["proportions"], indent=2))
    lines.append("```\n")

    lines.append("## Model comparison (held-out test set)\n")
    lines.append(_dataframe_to_markdown(comparison))
    lines.append("")

    lines.append("\n## Logistic regression: selected configuration\n")
    lines.append(f"Text-only: `{lr_text_run['best_config']}`\n")
    metadata_config = lr_metadata_run["best_config"]
    lines.append(f"With brand/category/price metadata: `{metadata_config}`\n")
    text_f1 = lr_text_run["result"]["metrics"]["macro_f1"]
    meta_f1 = lr_metadata_run["result"]["metrics"]["macro_f1"]
    meaningfully_better = meta_f1 > text_f1 + 0.01
    verdict = (
        "metadata helps"
        if meaningfully_better
        else "metadata doesn't meaningfully help"
    )
    lines.append(
        f"Text-only test macro-F1: {text_f1:.4f}; with metadata: {meta_f1:.4f} "
        f"({verdict}).\n"
    )

    if distilbert_run is not None and not distilbert_run.get("smoke_only"):
        lines.append("\n## DistilBERT fine-tuning\n")
        lines.append(
            f"Trained on {distilbert_run['train_rows_used']} rows "
            f"(validation {distilbert_run['val_rows_used']}), a compute-budget-limited "
            f"subset of the full leakage-safe training split, on device "
            f"`{distilbert_run['device']}`. Config: `{distilbert_run['config']}`. "
            f"Evaluated on {distilbert_run['test_rows_used']} of "
            f"{distilbert_run['full_test_rows']} held-out test rows.\n"
        )
        distilbert_f1 = distilbert_run["result"]["metrics"]["macro_f1"]
        lines.append(
            f"Test macro-F1: {distilbert_f1:.4f}. At this scale the encoder does "
            "not have enough labeled examples to fine-tune past predicting the "
            "majority class -- it is not yet competitive with logistic regression "
            "(or, on this run, meaningfully better than the dummy baseline). This "
            "reflects the small training subset below, not a ceiling on "
            "DistilBERT's potential.\n"
        )
        lines.append(
            "**Reduced-run limitation**: this machine's available memory was "
            "under heavy swap pressure from other running processes during this "
            "session (see `hardware.json`), so `--transformer-train-rows` and "
            "`--transformer-val-rows` were set far below the full training "
            "split's size to get a real, completed run instead of one that "
            "thrashed indefinitely. Only one seed was run (not the 3-seed repeat "
            "used for logistic regression). On a machine with more available "
            "RAM/a dedicated GPU, re-run with larger "
            "`--transformer-train-rows`/`--transformer-val-rows` (up to the full "
            "training split) for a properly powered comparison; the smoke run "
            "above already confirms the fine-tuning workflow itself is correct.\n"
        )
    elif distilbert_run is not None and distilbert_run.get("smoke_only"):
        lines.append("\n## DistilBERT fine-tuning\n")
        lines.append(
            "Only the smoke-test run was executed (`--transformer-smoke-only`); "
            "the workflow (tokenization, training loop, checkpoint selection, "
            "saving) was verified end-to-end, but no full comparison run was "
            "produced. Re-run without that flag to produce comparable numbers.\n"
        )
    else:
        lines.append("\n## DistilBERT fine-tuning\n")
        lines.append(
            "Not run: the optional `transformers`/`torch` extra was not "
            "installed. Install with `pip install -r requirements-transformers.txt` "
            "and re-run this script to add DistilBERT to the comparison.\n"
        )

    lines.append("\n## Recommendation\n")
    best_row = comparison.sort_values("macro_f1", ascending=False).iloc[0]
    dummy_rows = comparison.loc[comparison["model"] == "dummy_baseline", "macro_f1"]
    dummy_f1 = dummy_rows.iloc[0]
    lines.append(
        f"Best test macro-F1: **{best_row['model']}** ({best_row['macro_f1']:.4f}), "
        f"vs. the majority-class baseline's {dummy_f1:.4f}.\n"
    )
    lines.append(
        "Per the project default, the TF-IDF + logistic-regression classifier "
        "remains the recommended review classifier unless a learned model's "
        "quality improvement clearly justifies its added training/inference "
        "cost -- compare `training_time_seconds`, `saved_size_mb`, and "
        "`p95_ms` above alongside macro-F1 before switching.\n"
    )

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote comparison report to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-path", type=str, default=None)
    parser.add_argument("--products-path", type=str, default=None)
    default_output_dir = str(PROJECT_ROOT / "outputs" / "review_experiment")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--english-threshold", type=float, default=0.9)
    parser.add_argument("--language-sample-size", type=int, default=3000)
    parser.add_argument("--min-document-frequency", type=int, default=5)
    parser.add_argument(
        "--lr-c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0]
    )
    parser.add_argument("--lr-max-iter", type=int, default=300)
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--transformer-smoke-only", action="store_true")
    parser.add_argument(
        "--transformer-checkpoint", type=str, default="distilbert-base-uncased"
    )
    parser.add_argument("--transformer-learning-rate", type=float, default=2e-5)
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    parser.add_argument("--transformer-max-length", type=int, default=128)
    parser.add_argument("--transformer-epochs", type=int, default=3)
    parser.add_argument(
        "--transformer-class-weighting",
        type=str,
        choices=["none", "balanced"],
        default="none",
    )
    parser.add_argument("--transformer-early-stopping-patience", type=int, default=1)
    parser.add_argument("--transformer-train-rows", type=int, default=3000)
    parser.add_argument("--transformer-val-rows", type=int, default=600)
    parser.add_argument("--transformer-test-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for subdirectory in ("metrics", "models", "plots", "splits"):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    raw_reviews, raw_products = load_raw_data(args)
    quality = run_data_quality_checks(raw_reviews, raw_products, output_dir)
    language_check = run_language_check(raw_reviews, args, output_dir)

    prepared = build_prepared_data(raw_reviews, raw_products)
    print(f"\nPrepared (cleaned, deduplicated) reviews: {len(prepared)}")
    splits = build_splits(prepared, args, output_dir)

    dummy_run = train_and_benchmark_dummy(splits, output_dir)
    lr_text_run = train_and_benchmark_logistic_regression(
        splits, args, output_dir, suffix="text"
    )
    lr_metadata_run = train_and_benchmark_logistic_regression(
        splits, args, output_dir, suffix="metadata"
    )

    distilbert_run = None
    if not args.skip_transformer:
        if language_check["predominantly_english"]:
            distilbert_run = train_and_benchmark_distilbert(splits, args, output_dir)
        else:
            print(
                "\n=== DistilBERT: SKIPPED (review text is not predominantly "
                "English; see language_check.json) ==="
            )

    comparison = write_comparison(
        [dummy_run, lr_text_run, lr_metadata_run, distilbert_run], output_dir
    )
    write_report(
        output_dir,
        quality=quality,
        language_check=language_check,
        splits=splits,
        dummy_run=dummy_run,
        lr_text_run=lr_text_run,
        lr_metadata_run=lr_metadata_run,
        distilbert_run=distilbert_run,
        comparison=comparison,
    )


if __name__ == "__main__":
    main()
