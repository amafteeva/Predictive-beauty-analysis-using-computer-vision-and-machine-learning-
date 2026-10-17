import time

import pandas as pd
import pytest

from foundation_matcher.review_benchmark import (
    benchmark_classifier,
    detect_hardware,
    get_artifact_size_bytes,
    measure_batch_throughput,
    measure_model_load_time,
    measure_peak_memory_bytes,
    measure_prediction_latency,
    representative_errors,
)
from foundation_matcher.review_model import build_majority_baseline


def test_measure_model_load_time_returns_object_and_positive_elapsed():
    def load():
        time.sleep(0.01)
        return object()

    model, elapsed = measure_model_load_time(load)

    assert model is not None
    assert elapsed >= 0.01


def test_measure_prediction_latency_reports_percentiles():
    result = measure_prediction_latency(
        lambda text: len(text), ["a", "bb", "ccc"], warmup=2, repeats=10
    )

    assert result["p50_seconds"] >= 0
    assert result["p95_seconds"] >= result["p50_seconds"]
    assert result["repeats"] == 10


def test_measure_prediction_latency_rejects_empty_examples():
    with pytest.raises(ValueError, match="examples"):
        measure_prediction_latency(lambda text: text, [], repeats=5)


def test_measure_batch_throughput_reports_positive_rate():
    result = measure_batch_throughput(
        lambda batch: [len(t) for t in batch],
        ["a", "b", "c", "d"],
        batch_size=2,
        warmup_batches=1,
        repeats=3,
    )

    assert result["examples_per_second"] > 0
    assert result["total_examples"] == 6


def test_measure_peak_memory_bytes_is_positive():
    _, peak_bytes = measure_peak_memory_bytes(lambda: [0] * 1000)
    assert peak_bytes > 0


def test_get_artifact_size_bytes_sums_directory_contents(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world!!")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("!")

    size = get_artifact_size_bytes(tmp_path)

    assert size == len("hello") + len("world!!") + len("!")


def test_get_artifact_size_bytes_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_artifact_size_bytes(tmp_path / "does_not_exist")


def test_representative_errors_only_contains_mismatches():
    texts = ["t1", "t2", "t3", "t4"]
    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 0, 0]  # index 0: false positive, index 3: false negative

    errors = representative_errors(texts, y_true, y_pred, n=5)

    assert len(errors["false_positives"]) == 1
    assert errors["false_positives"]["text"].iloc[0] == "t1"
    assert len(errors["false_negatives"]) == 1
    assert errors["false_negatives"]["text"].iloc[0] == "t4"


def test_detect_hardware_reports_a_device_without_requiring_torch():
    hardware = detect_hardware()

    assert "device" in hardware
    assert hardware["cpu_count"] >= 1


def test_benchmark_classifier_runs_end_to_end_on_dummy_baseline():
    reviews = pd.DataFrame(
        {
            "review_text": [f"review number {i}" for i in range(20)],
            "positive_review": [1, 0] * 10,
        }
    )
    model = build_majority_baseline()
    model.fit(reviews[["review_text"]], reviews["positive_review"])

    result = benchmark_classifier(
        model,
        reviews[["review_text"]],
        reviews["positive_review"],
        latency_examples=10,
        latency_repeats=5,
        throughput_batch_size=4,
    )

    assert "macro_f1" in result["metrics"].index
    assert result["confusion_matrix"].shape == (2, 2)
    assert result["latency"]["p50_seconds"] >= 0
    assert result["throughput"]["examples_per_second"] > 0
    assert "false_positives" in result["errors"]


def test_benchmark_classifier_default_predict_preserves_extra_feature_columns():
    """A model needing more than the text column (e.g. brand/category/price)
    must still work through the default predict_one/predict_batch -- these
    must reconstruct full rows, not just the text column."""

    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    reviews = pd.DataFrame(
        {
            "review_text": [f"review number {i}" for i in range(20)],
            "brand": ["A", "B"] * 10,
            "positive_review": [1, 0] * 10,
        }
    )
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [("brand", OneHotEncoder(handle_unknown="ignore"), ["brand"])]
                ),
            ),
            ("classifier", LogisticRegression()),
        ]
    )
    model.fit(reviews[["review_text", "brand"]], reviews["positive_review"])

    result = benchmark_classifier(
        model,
        reviews[["review_text", "brand"]],
        reviews["positive_review"],
        latency_examples=10,
        latency_repeats=5,
        throughput_batch_size=4,
    )

    assert result["latency"]["p50_seconds"] >= 0
