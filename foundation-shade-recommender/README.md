# Foundation Shade Recommender

An end-to-end prototype that estimates a skin-colour sample from a selfie and
ranks foundation products by perceptual colour similarity.

The repository separates the user-facing analysis from reusable Python code.
The main recommendation method is intentionally transparent: MediaPipe locates
facial landmarks, sampled pixels are converted to CIELAB, and products are
ranked with CIEDE2000 colour distance. K-Means clustering and review-text
classification are included as clearly labelled experiments.

## Project questions

1. Can a face-landmark pipeline obtain a usable skin-colour estimate from a
   selfie?
2. Which catalogue shades are closest to that estimate in perceptual colour
   space?
3. How consistently does the pipeline complete across a small, balanced
   FairFace sample?
4. Do foundation colours form useful unsupervised clusters?

## How the system works

1. Load The Pudding's foundation-shade catalogue.
2. Clean HEX values and convert product colours from RGB to CIELAB.
3. Detect 478 face landmarks with MediaPipe Face Landmarker.
4. Sample small regions on both cheeks and the forehead.
5. Remove the darkest and brightest 5% of sampled pixels.
6. Use the median colour as the skin-colour estimate.
7. Rank products with CIEDE2000; a smaller Delta E means a closer colour match.

## Repository structure

```text
foundation-shade-recommender/
├── README.md
├── GITHUB_UPLOAD_GUIDE.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   ├── 01_foundation_matcher_demo.ipynb
│   ├── 02_fairface_pipeline_evaluation.ipynb
│   └── 03_shade_clustering.ipynb
├── experiments/
│   └── 04_review_satisfaction_baseline.ipynb
├── src/foundation_matcher/
│   ├── color.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── face.py
│   ├── recommender.py
│   ├── review_model.py
│   └── visualization.py
├── data/README.md
├── models/README.md
├── outputs/README.md
└── tests/
```

## Quick start

### Google Colab

After pushing this folder to GitHub, open one of the notebooks and run the
setup cell. Replace the placeholder repository URL in that cell with your own:

```python
REPOSITORY_URL = "https://github.com/YOUR_USERNAME/foundation-shade-recommender.git"
```

The demo notebook prompts you to upload a selfie. Images are processed only in
the current notebook runtime unless you explicitly save them.

### Local installation

```bash
git clone https://github.com/YOUR_USERNAME/foundation-shade-recommender.git
cd foundation-shade-recommender
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
jupyter lab
```

Run tests with:

```bash
python -m pytest
```

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_foundation_matcher_demo.ipynb` | Complete selfie-to-recommendation demonstration |
| `02_fairface_pipeline_evaluation.ipynb` | Balanced 140-image pipeline-completion evaluation and robustness simulation |
| `03_shade_clustering.ipynb` | K-Means cluster selection and cluster-restricted recommendations |
| `04_review_satisfaction_baseline.ipynb` | Independent TF-IDF and logistic-regression experiment on review text |

## Evaluation interpretation

The original exploratory notebook reported:

- 128 successful pipeline runs out of 140 images (91.4%).
- Median Top-1 perceptual error of 1.78 Delta E under simulated perturbations.
- 97.6% of first recommendations within Delta E 5.
- 86.2% of Top-5 lists containing a shade within Delta E 2.

These numbers should be regenerated from `02_fairface_pipeline_evaluation.ipynb`
before being cited. The FairFace experiment measures **pipeline completion**,
not foundation-match accuracy. FairFace does not provide professionally matched
foundation labels, so accuracy, precision, recall, and F1 are not valid metrics
for shade correctness in this experiment.

The simulation measures stability under artificial LAB perturbations. Exact-row
recovery is intentionally reported separately because the catalogue contains
duplicate and near-duplicate colours from different brands.

## What is and is not machine learning

| Component | Type |
| --- | --- |
| MediaPipe Face Landmarker | Pretrained machine-learning model; not fine-tuned here |
| K-Means shade grouping | Unsupervised model trained on catalogue LAB values |
| Review satisfaction baseline | Supervised text classifier trained on review labels |
| HEX/RGB/CIELAB conversion | Colour mathematics |
| CIEDE2000 ranking | Deterministic perceptual-distance calculation |
| FairFace sample | Evaluation data, not training data for the matcher |

## Data sources

- [The Pudding: The Diversity of Makeup Shades](https://github.com/the-pudding/data/tree/master/makeup-shades)
- [FairFace dataset card](https://huggingface.co/datasets/HuggingFaceM4/FairFace)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Luxxify Makeup Recommender data](https://github.com/zara-sarkar/Luxxify-Makeup-Recommender)

The datasets and model file are downloaded at runtime and are not committed to
this repository.

## Limitations and responsible use

- Camera white balance, lighting, shadows, makeup, filters, and display colour
  calibration can change the estimated colour.
- Three small facial regions cannot represent every variation across a face.
- Catalogue HEX values are digital approximations, not calibrated measurements
  of physical products.
- The source catalogue does not consistently provide shade names, prices,
  formulas, stock status, undertone labels, or regional availability.
- Race labels are not skin-tone measurements. Group completion rates must not be
  interpreted as proof of fairness or shade accuracy.
- A production system needs consent, privacy protections, calibrated images,
  professionally labelled matches, broader demographic testing, and human
  evaluation.

This repository is an educational prototype, not a purchasing guarantee or a
biometric-identification system.
