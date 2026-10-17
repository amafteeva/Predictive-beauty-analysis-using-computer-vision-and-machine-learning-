"""Optional DistilBERT fine-tuning for the review-classification experiment.

This module is never imported by `face.py`, `recommender.py`, `color.py`,
`data.py`, or by `review_model.py`/`review_benchmark.py` -- the
shade-matching pipeline and the TF-IDF/logistic-regression review baseline
work with no transformer dependency installed at all. torch and
transformers are imported here, at this module's top level, inside a
`try/except ImportError`; every public function calls `require_transformers`
first, so importing this module without the optional extra installed
succeeds, and only calling one of its functions raises a clear,
actionable error.

Install the optional extra with:
    pip install -r requirements-transformers.txt
    # or: pip install "foundation-shade-matcher[transformers]"
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from foundation_matcher.config import RANDOM_STATE

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when the extra isn't installed
    TRANSFORMERS_AVAILABLE = False
    Dataset = object  # placeholder base class so _EncodedDataset still defines cleanly


def require_transformers() -> None:
    """Raise a clear, actionable error if torch/transformers aren't installed."""

    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "foundation_matcher.review_transformer requires the optional "
            "'transformers' extra (torch + transformers), which is not installed. "
            "Install it with: pip install -r requirements-transformers.txt "
            "(or `pip install \"foundation-shade-matcher[transformers]\"`). "
            "The shade-matching pipeline and the TF-IDF/logistic-regression "
            "review classifier do not need this."
        )


@dataclass
class DistilBertTrainingConfig:
    """Every fine-tuning knob the experiment spec asks to be configurable."""

    checkpoint: str = "distilbert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_length: int = 128
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    class_weighting: str = "none"  # "none" or "balanced"
    seed: int = RANDOM_STATE
    early_stopping_patience: int = 2
    device: str | None = None  # None -> auto-detect (cuda > mps > cpu)
    smoke_test: bool = False
    smoke_test_train_rows: int = 64
    smoke_test_val_rows: int = 32
    max_train_steps_per_epoch: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DistilBertTrainingResult:
    model: object
    tokenizer: object
    device: str
    history: pd.DataFrame
    best_epoch: int
    best_val_macro_f1: float
    training_time_seconds: float
    train_truncation_rate: float
    val_truncation_rate: float
    config: DistilBertTrainingConfig


class _EncodedDataset(Dataset):
    def __init__(self, encodings: dict, labels) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch for a reproducible fine-tuning run."""

    require_transformers()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tokenize_texts(
    tokenizer, texts, *, max_length: int
) -> tuple[dict, float]:
    """Tokenize a batch of texts and report the truncation rate.

    The truncation rate is computed from an untruncated pass (cheap -- it's
    only tokenization, not a model forward pass), so it reflects how many
    reviews actually exceed `max_length` tokens rather than an estimate.
    """

    require_transformers()
    text_list = [str(text) for text in texts]
    unrestricted = tokenizer(text_list, truncation=False)
    lengths = [len(ids) for ids in unrestricted["input_ids"]]
    truncated = sum(1 for length in lengths if length > max_length)
    truncation_rate = truncated / len(lengths) if lengths else 0.0

    encodings = tokenizer(
        text_list,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings, truncation_rate


def _balanced_class_weights(labels: np.ndarray, *, num_classes: int = 2):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def fine_tune_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: DistilBertTrainingConfig,
    *,
    text_column: str = "review_text",
    label_column: str = "positive_review",
) -> DistilBertTrainingResult:
    """Fine-tune a pretrained DistilBERT encoder + classification head.

    Both the encoder and the head are updated (no layer freezing), with
    validation-based checkpoint selection and early stopping on macro-F1
    (matching how `select_logistic_regression_config` selects the logistic
    regression). Set `config.smoke_test = True` to sample down to
    `smoke_test_train_rows`/`smoke_test_val_rows` and (optionally)
    `max_train_steps_per_epoch`, for a fast end-to-end correctness check
    before spending the compute budget on a full run.
    """

    require_transformers()
    set_seed(config.seed)
    device = config.device or _select_device()

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint, num_labels=2
    )
    model.to(device)

    if config.smoke_test:
        train_df = train_df.sample(
            n=min(len(train_df), config.smoke_test_train_rows), random_state=config.seed
        )
        val_df = val_df.sample(
            n=min(len(val_df), config.smoke_test_val_rows), random_state=config.seed
        )

    train_encodings, train_truncation_rate = tokenize_texts(
        tokenizer, train_df[text_column], max_length=config.max_length
    )
    val_encodings, val_truncation_rate = tokenize_texts(
        tokenizer, val_df[text_column], max_length=config.max_length
    )

    train_labels = torch.tensor(train_df[label_column].to_numpy(), dtype=torch.long)
    val_labels = torch.tensor(val_df[label_column].to_numpy(), dtype=torch.long)

    train_dataset = _EncodedDataset(train_encodings, train_labels)
    val_dataset = _EncodedDataset(val_encodings, val_labels)

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    weights = None
    if config.class_weighting == "balanced":
        weights = _balanced_class_weights(train_df[label_column].to_numpy()).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = (
        min(len(train_loader), config.max_train_steps_per_epoch)
        if config.max_train_steps_per_epoch
        else len(train_loader)
    )
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        step_cap = config.max_train_steps_per_epoch
        for step, batch in enumerate(train_loader):
            if step_cap and step >= step_cap:
                break
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss = running_loss / seen if seen else float("nan")

        model.eval()
        val_loss = 0.0
        all_predictions: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item() * labels.size(0)
                predictions = outputs.logits.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_loss = val_loss / len(val_dataset) if len(val_dataset) else float("nan")
        val_macro_f1 = f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
            }
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    training_time_seconds = time.perf_counter() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    return DistilBertTrainingResult(
        model=model,
        tokenizer=tokenizer,
        device=str(device),
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_val_macro_f1=float(best_val_macro_f1),
        training_time_seconds=training_time_seconds,
        train_truncation_rate=train_truncation_rate,
        val_truncation_rate=val_truncation_rate,
        config=config,
    )


def save_distilbert_artifacts(
    result: DistilBertTrainingResult,
    output_dir: str | Path,
    *,
    label_mapping: dict[int, str] | None = None,
) -> Path:
    """Save the model, tokenizer, config, and label mapping locally.

    Everything is written to `output_dir` on the local filesystem only --
    no `push_to_hub` or other upload is ever called.
    """

    require_transformers()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_pretrained(output_dir)
    result.tokenizer.save_pretrained(output_dir)

    label_mapping = label_mapping or {0: "negative", 1: "positive"}
    manifest = {
        "label_mapping": {str(key): value for key, value in label_mapping.items()},
        "config": result.config.to_dict(),
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "training_time_seconds": result.training_time_seconds,
        "device": result.device,
        "train_truncation_rate": result.train_truncation_rate,
        "val_truncation_rate": result.val_truncation_rate,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_dir


def load_distilbert_artifacts(
    model_dir: str | Path, *, device: str | None = None
) -> tuple[object, object, str]:
    """Load a locally saved DistilBERT model + tokenizer for inference."""

    require_transformers()
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    resolved_device = device or _select_device()
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


class DistilBertSklearnAdapter:
    """Minimal scikit-learn-style wrapper around a fine-tuned DistilBERT model.

    Exposes `predict`/`predict_proba` with the same calling convention as
    the scikit-learn pipelines in `review_model.py`, so
    `evaluate_review_classifier` and `review_benchmark.benchmark_classifier`
    work identically across the dummy baseline, TF-IDF+logistic-regression,
    and DistilBERT -- the three are benchmarked with exactly the same code.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: str,
        max_length: int,
        text_column: str = "review_text",
        batch_size: int = 32,
    ) -> None:
        require_transformers()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.text_column = text_column
        self.batch_size = batch_size

    def _texts(self, X) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return X[self.text_column].astype(str).tolist()
        if isinstance(X, pd.Series):
            return X.astype(str).tolist()
        return [str(text) for text in X]

    def predict_proba(self, X) -> np.ndarray:
        texts = self._texts(X)
        self.model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encodings = {
                    key: value.to(self.device) for key, value in encodings.items()
                }
                logits = self.model(**encodings).logits
                batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if not batches:
            return np.zeros((0, 2))
        return np.concatenate(batches, axis=0)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
