"""Quality and efficiency benchmarking shared by every review classifier.

Used identically for the majority-class dummy baseline, the TF-IDF +
logistic-regression pipeline, and (optionally) the fine-tuned DistilBERT
model from `review_transformer`, so the three are compared on the same
measurements. Nothing here imports torch/transformers -- a caller
benchmarking DistilBERT passes in `predict_one`/`predict_batch` callables
instead (see `benchmark_classifier`).
"""

from __future__ import annotations

import os
import platform
import resource
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from foundation_matcher.review_model import evaluate_review_classifier


def measure_model_load_time(load_fn: Callable[[], object]) -> tuple[object, float]:
    """Time how long it takes to load a model, separately from inference."""

    start = time.perf_counter()
    model = load_fn()
    elapsed = time.perf_counter() - start
    return model, elapsed


def measure_prediction_latency(
    predict_one: Callable[[str], object],
    examples: Sequence[str],
    *,
    warmup: int = 10,
    repeats: int = 100,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Measure per-example prediction latency (median/p95) after warm-up.

    `predict_one` should run the full path a caller cares about -- including
    text preprocessing/tokenization -- for one example, since that is part
    of what gets measured; model *loading* is deliberately not included
    (time that separately via `measure_model_load_time`). Pass
    `synchronize` (e.g. `torch.mps.synchronize`/`torch.cuda.synchronize`)
    when timing a GPU-backed model, so queued device work is captured
    rather than only the time to enqueue it.
    """

    if not examples:
        raise ValueError("examples must not be empty.")

    pool = list(examples)
    for index in range(min(warmup, repeats)):
        predict_one(pool[index % len(pool)])
        if synchronize is not None:
            synchronize()

    samples = []
    for index in range(repeats):
        example = pool[index % len(pool)]
        start = time.perf_counter()
        predict_one(example)
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - start)

    samples.sort()
    return {
        "p50_seconds": statistics.median(samples),
        "p95_seconds": samples[int(0.95 * (len(samples) - 1))],
        "mean_seconds": statistics.mean(samples),
        "repeats": repeats,
    }


def measure_batch_throughput(
    predict_batch: Callable[[Sequence[str]], object],
    examples: Sequence[str],
    *,
    batch_size: int = 32,
    warmup_batches: int = 2,
    repeats: int = 5,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Measure batched throughput (examples/second) after warm-up."""

    if not examples:
        raise ValueError("examples must not be empty.")

    pool = list(examples)
    total_batches = warmup_batches + repeats
    batches = [
        [pool[(start + offset) % len(pool)] for offset in range(batch_size)]
        for start in range(0, batch_size * total_batches, batch_size)
    ]

    for batch in batches[:warmup_batches]:
        predict_batch(batch)
        if synchronize is not None:
            synchronize()

    start_time = time.perf_counter()
    total_examples = 0
    for batch in batches[warmup_batches:]:
        predict_batch(batch)
        if synchronize is not None:
            synchronize()
        total_examples += len(batch)
    elapsed = time.perf_counter() - start_time

    rate = total_examples / elapsed if elapsed > 0 else float("inf")
    return {
        "examples_per_second": rate,
        "batch_size": batch_size,
        "total_examples": total_examples,
        "elapsed_seconds": elapsed,
    }


def measure_peak_memory_bytes(fn: Callable[[], object]) -> tuple[object, int]:
    """Run `fn`, returning its result and the process's peak RSS afterward.

    Uses `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a whole-process
    high-water mark that -- unlike `tracemalloc` -- also captures native
    allocations from scikit-learn/PyTorch, not just Python-level ones. It
    cannot isolate `fn`'s own contribution from memory the process already
    held before `fn` ran; take a baseline reading the same way beforehand
    if an isolated delta is needed. `ru_maxrss` is reported in bytes on
    macOS and kibibytes on Linux -- this normalizes both to bytes.
    """

    result = fn()
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = raw if platform.system() == "Darwin" else raw * 1024
    return result, int(peak_bytes)


def get_artifact_size_bytes(path: str | Path) -> int:
    """Total size on disk of a saved model file or directory."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def representative_errors(
    texts: pd.Series | Sequence[str],
    y_true: pd.Series | Sequence[int],
    y_pred: Sequence[int],
    *,
    n: int = 5,
    random_state: int = 0,
) -> dict[str, pd.DataFrame]:
    """Sample representative false positives and false negatives for review."""

    frame = pd.DataFrame(
        {
            "text": np.asarray(texts, dtype=object),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )
    false_positive = frame[(frame["y_true"] == 0) & (frame["y_pred"] == 1)]
    false_negative = frame[(frame["y_true"] == 1) & (frame["y_pred"] == 0)]
    rng = np.random.default_rng(random_state)

    def sample(subset: pd.DataFrame) -> pd.DataFrame:
        if len(subset) <= n:
            return subset.reset_index(drop=True)
        indices = rng.choice(len(subset), size=n, replace=False)
        return subset.iloc[indices].reset_index(drop=True)

    return {
        "false_positives": sample(false_positive),
        "false_negatives": sample(false_negative),
    }


def detect_hardware() -> dict[str, object]:
    """Report the compute available for benchmarking/fine-tuning.

    Does not require torch/transformers to be installed: if unavailable,
    only CPU information is reported and `device` is set to a CPU fallback,
    so callers can make an explicit compute-budget decision either way.
    """

    info: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(torch.backends.mps.is_available())
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            info["device"] = "mps"
        else:
            info["device"] = "cpu"
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["mps_available"] = False
        info["device"] = "cpu (torch not installed)"
    return info


def benchmark_classifier(
    model,
    X_test: pd.DataFrame | pd.Series,
    y_test: pd.Series,
    *,
    text_column: str = "review_text",
    predict_one: Callable[[str], int] | None = None,
    predict_batch: Callable[[Sequence[str]], Sequence[int]] | None = None,
    latency_examples: int = 200,
    latency_repeats: int = 100,
    throughput_batch_size: int = 32,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Run the full quality + efficiency benchmark for one fitted model.

    Quality metrics reuse `evaluate_review_classifier`, so every model in
    the comparison is scored identically. `predict_one`/`predict_batch`
    default to calling `model.predict` on a one-row DataFrame built from
    the *same columns as `X_test`* (not just `text_column`), so this works
    unchanged whether `model` is a text-only pipeline or one that also
    expects `category`/`brand`/`price` columns; pass explicit callables to
    benchmark something with a different calling convention entirely, such
    as a DistilBERT adapter that expects raw text.

    Latency and throughput are measured on the same held-out examples used
    for quality metrics, after a warm-up phase, so cold-start effects
    (model JIT/graph compilation, first-call cache population) don't bias
    the numbers.
    """

    texts = X_test[text_column] if isinstance(X_test, pd.DataFrame) else X_test
    metrics, report, matrix = evaluate_review_classifier(model, X_test, y_test)

    predicted = model.predict(X_test)
    errors = representative_errors(texts, y_test, predicted)

    if isinstance(X_test, pd.DataFrame):
        example_records: list[dict] = X_test.to_dict("records")
    else:
        example_records = [{text_column: text} for text in X_test]

    if predict_one is None:

        def predict_one(record: dict) -> int:
            return int(model.predict(pd.DataFrame([record]))[0])

    if predict_batch is None:

        def predict_batch(batch: Sequence[dict]) -> Sequence[int]:
            return model.predict(pd.DataFrame(list(batch)))

    example_pool = example_records[:latency_examples] or example_records

    latency = measure_prediction_latency(
        predict_one, example_pool, repeats=latency_repeats, synchronize=synchronize
    )
    throughput = measure_batch_throughput(
        predict_batch,
        example_pool,
        batch_size=throughput_batch_size,
        synchronize=synchronize,
    )

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "errors": errors,
        "latency": latency,
        "throughput": throughput,
    }
