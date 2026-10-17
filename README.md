# Predictive Beauty Analytics: Foundation Shade Recommendation
An end-to-end data science, computer vision, and machine learning project that estimates skin tone from facial images and recommends foundation shades using perceptual color similarity.
The project combines MediaPipe facial landmark detection, CIELAB color analysis, CIEDE2000 color distance, brute-force nearest-neighbor shade matching, and model evaluation to build and assess a foundation shade recommendation pipeline.
## Project Overview
Choosing a foundation shade online is difficult because product shade names are inconsistent across brands and photographs can be affected by lighting, cameras, and skin-tone variation.
This project explores a data-driven approach that:
1. Detects a face from an input image.
2. Extracts representative skin-color regions.
3. Converts skin and foundation colors into the CIELAB color space.
4. Measures perceptual similarity using CIEDE2000.
5. Ranks the closest available foundation shades.
6. Evaluates pipeline robustness across diverse facial images.
7. Uses clustering and review-based machine learning experiments to explore additional recommendation signals.

## Pipeline
```text
Input Selfie
    ↓
Face Detection & Facial Landmarks
    ↓
Skin Region Extraction
    ↓
RGB → CIELAB Conversion
    ↓
Skin Tone Estimation
    ↓
CIEDE2000 Color Distance
    ↓
Foundation Shade Ranking
    ↓
Top-N Recommendations

```

## Review Classification Experiment (Optional, Independent)

`src/foundation_matcher/review_model.py`, `review_benchmark.py`, and
`review_transformer.py` implement a separate experiment: predicting a
**rating-derived** binary label from Luxxify review text (1 = rating >= 4,
0 = rating < 4). This is review-text classification, not a personalized
customer-satisfaction predictor -- the label comes directly from the same
review's own star rating. It never influences shade recommendations; see
`scripts/verify_shade_pipeline.py` for the before/after guarantee that the
shade-matching pipeline is unaffected.

Three models are compared on identical, leakage-safe, product-disjoint
splits: a majority-class dummy baseline, TF-IDF + logistic regression
(the default), and, optionally, a fine-tuned DistilBERT classifier. Run
the full comparison with:

```bash
pip install -r requirements.txt          # core (includes langdetect)
pip install -r requirements-transformers.txt   # optional, for DistilBERT
python scripts/run_review_experiment.py
```

Add `--skip-transformer` to run only the dummy baseline and logistic
regression (no torch/transformers required), or
`--transformer-smoke-only` to verify the DistilBERT fine-tuning workflow
without the full training run. Results (split identifiers, hyperparameter
search, metrics, plots, saved models, and a `REPORT.md` comparison) are
written to `outputs/review_experiment/`.
