"""Tests for the optional DistilBERT fine-tuning module.

Every test here requires the optional `transformers` extra and network
access to Hugging Face Hub (for a tiny, randomly-initialized test
checkpoint -- not a real pretrained model), and is skipped when either is
unavailable. `test_review_transformer_module_imports_without_torch`
(guaranteeing the module degrades gracefully when the extra is absent) runs
unconditionally and does not require network access.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from foundation_matcher import review_transformer as rt  # noqa: E402

TINY_CHECKPOINT = "hf-internal-testing/tiny-random-distilbert"


def _network_available() -> bool:
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(TINY_CHECKPOINT)
        return True
    except Exception:
        return False


requires_network = pytest.mark.skipif(
    not _network_available(), reason="Hugging Face Hub is not reachable"
)


def _synthetic_split():
    rng_positive = [f"amazing quality product number {i}" for i in range(12)]
    rng_negative = [f"terrible broke immediately number {i}" for i in range(12)]
    train = pd.DataFrame(
        {
            "review_text": rng_positive[:8] + rng_negative[:8],
            "positive_review": [1] * 8 + [0] * 8,
        }
    )
    val = pd.DataFrame(
        {
            "review_text": rng_positive[8:] + rng_negative[8:],
            "positive_review": [1] * 4 + [0] * 4,
        }
    )
    return train, val


def test_require_transformers_does_not_raise_when_available():
    rt.require_transformers()  # should not raise; torch/transformers are installed here


@requires_network
def test_tokenize_texts_reports_truncation_rate():
    tokenizer = transformers.AutoTokenizer.from_pretrained(TINY_CHECKPOINT)
    texts = ["short text", "word " * 500]

    encodings, truncation_rate = rt.tokenize_texts(tokenizer, texts, max_length=16)

    assert encodings["input_ids"].shape[0] == 2
    assert truncation_rate == pytest.approx(0.5)


@requires_network
def test_fine_tune_distilbert_smoke_run_improves_or_matches_random_baseline():
    train_df, val_df = _synthetic_split()
    config = rt.DistilBertTrainingConfig(
        checkpoint=TINY_CHECKPOINT,
        epochs=2,
        batch_size=4,
        max_length=32,
        seed=0,
        early_stopping_patience=2,
        smoke_test=True,
        smoke_test_train_rows=16,
        smoke_test_val_rows=8,
    )

    result = rt.fine_tune_distilbert(train_df, val_df, config)

    assert result.best_epoch >= 0
    assert len(result.history) >= 1
    assert result.training_time_seconds > 0
    assert 0.0 <= result.train_truncation_rate <= 1.0
    assert 0.0 <= result.val_truncation_rate <= 1.0
    assert result.device in {"cpu", "mps", "cuda"}


@requires_network
def test_save_and_load_distilbert_artifacts_round_trip(tmp_path):
    train_df, val_df = _synthetic_split()
    config = rt.DistilBertTrainingConfig(
        checkpoint=TINY_CHECKPOINT,
        epochs=1,
        batch_size=4,
        max_length=32,
        seed=0,
        smoke_test=True,
        smoke_test_train_rows=8,
        smoke_test_val_rows=8,
    )
    result = rt.fine_tune_distilbert(train_df, val_df, config)

    output_dir = rt.save_distilbert_artifacts(result, tmp_path / "model")

    assert (output_dir / "training_manifest.json").exists()
    assert (output_dir / "tokenizer_config.json").exists() or (
        output_dir / "vocab.txt"
    ).exists()

    model, tokenizer, device = rt.load_distilbert_artifacts(output_dir, device="cpu")
    adapter = rt.DistilBertSklearnAdapter(
        model, tokenizer, device=device, max_length=32
    )

    predictions = adapter.predict(pd.Series(["a new review to classify"]))
    probabilities = adapter.predict_proba(pd.Series(["a new review to classify"]))

    assert predictions.shape == (1,)
    assert probabilities.shape == (1, 2)


def test_review_transformer_module_imports_without_torch():
    """Confirm the module degrades gracefully when torch isn't installed.

    Runs a subprocess with torch/transformers hidden from import machinery
    (rather than actually uninstalling them), and checks that importing
    `review_transformer` still succeeds while calling one of its functions
    raises a clear ImportError -- the same guarantee the shade-matching
    pipeline relies on (see scripts/verify_shade_pipeline.py).
    """

    script = (
        "import sys, builtins\n"
        "real_import = builtins.__import__\n"
        "def blocked(name, *args, **kwargs):\n"
        "    blocked_names = name == 'torch' or name.startswith('torch.')\n"
        "    blocked_names = blocked_names or name == 'transformers'\n"
        "    if blocked_names:\n"
        "        raise ImportError(f'blocked: {name}')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = blocked\n"
        "sys.path.insert(0, 'src')\n"
        "from foundation_matcher import review_transformer as rt\n"
        "assert rt.TRANSFORMERS_AVAILABLE is False\n"
        "try:\n"
        "    rt.require_transformers()\n"
        "    raise SystemExit('expected ImportError')\n"
        "except ImportError as error:\n"
        "    assert 'transformers' in str(error)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
