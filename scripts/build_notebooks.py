"""Generate the cleaned project notebooks as deterministic JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def _lines(text: str) -> list[str]:
    normalized = dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CELL = r'''
from pathlib import Path
import os
import subprocess
import sys

REPOSITORY_URL = "https://github.com/YOUR_USERNAME/foundation-shade-recommender.git"


def find_project_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


project_root = find_project_root(Path.cwd())

if project_root is None and "google.colab" in sys.modules:
    if "YOUR_USERNAME" in REPOSITORY_URL:
        raise ValueError(
            "Replace YOUR_USERNAME in REPOSITORY_URL after publishing the project to GitHub."
        )
    project_root = Path("/content/foundation-shade-recommender")
    if not project_root.exists():
        subprocess.run(["git", "clone", REPOSITORY_URL, str(project_root)], check=True)

if project_root is None:
    raise FileNotFoundError("Run this notebook from inside the project folder.")

os.chdir(project_root)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
    check=True,
)
print("Project root:", project_root)
'''


DEMO_CELLS = [
    markdown(
        r'''
        # Foundation Shade Recommender: Interactive Demo

        This notebook follows one clear path from a selfie to a ranked list of
        foundation colours. It is the best starting point for readers.

        **Method:** MediaPipe landmarks → cheek/forehead pixels → median RGB →
        CIELAB → CIEDE2000 product ranking.

        The result is a colour-similarity prototype, not a purchasing guarantee.
        Lighting, camera processing, makeup, and catalogue colour quality all
        affect the result.
        '''
    ),
    markdown(
        r'''
        ## 1. Setup

        When running in Colab after publishing this project, replace
        `YOUR_USERNAME` below with your GitHub username. Local users should run
        the notebook from inside the cloned repository.
        '''
    ),
    code(SETUP_CELL),
    code(
        r'''
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        from IPython.display import display

        from foundation_matcher.data import load_foundation_catalog
        from foundation_matcher.face import (
            create_face_landmarker,
            download_face_landmarker,
            extract_skin_tone,
        )
        from foundation_matcher.recommender import recommend_foundations
        from foundation_matcher.visualization import (
            plot_catalog_lab,
            plot_match_swatches,
            plot_skin_preview,
        )
        '''
    ),
    markdown(
        r'''
        ## 2. Prepare the foundation catalogue

        The source catalogue stores digital HEX colours. The project cleans those
        values and converts them to CIELAB, where numerical distance is more
        closely related to perceived colour difference than ordinary RGB distance.
        '''
    ),
    code(
        r'''
        products = load_foundation_catalog()

        print(f"Catalogue rows: {len(products):,}")
        print(f"Brands: {products['brand'].nunique():,}")
        print(f"Unique HEX colours: {products['hex'].nunique():,}")

        display(
            products[
                ["brand", "product", "hex", "lab_L", "lab_a", "lab_b"]
            ].head()
        )
        '''
    ),
    code(
        r'''
        display(products[["lab_L", "lab_a", "lab_b"]].describe().round(2))
        plot_catalog_lab(products)
        plt.show()
        '''
    ),
    markdown(
        r'''
        ## 3. Select a selfie

        For a better estimate, use an unfiltered, front-facing image taken in
        even natural light. Avoid dramatic shadows and coloured lighting.

        The image remains in the current runtime unless you explicitly save it.
        Do not commit personal images to GitHub.
        '''
    ),
    code(
        r'''
        model_path = download_face_landmarker("models/face_landmarker.task")

        if "google.colab" in sys.modules:
            from google.colab import files

            uploaded = files.upload()
            if not uploaded:
                raise ValueError("No image was uploaded.")
            image_path = Path(next(iter(uploaded)))
        else:
            # Change this path when running locally.
            image_path = Path("data/example_selfie.jpg")
            if not image_path.exists():
                raise FileNotFoundError(
                    "Set image_path to a local selfie before running this cell."
                )

        print("Selected image:", image_path)
        '''
    ),
    markdown(
        r'''
        ## 4. Detect facial regions and estimate skin colour

        The green circles show the sampled regions: both cheeks and the centre
        forehead. The estimator trims the darkest and brightest 5% of sampled
        pixels and uses the median of the remainder.
        '''
    ),
    code(
        r'''
        landmarker = create_face_landmarker(model_path)
        try:
            skin_tone = extract_skin_tone(image_path, landmarker)
        finally:
            landmarker.close()

        print("Detected RGB:", skin_tone.rgb.tolist())
        print("Detected LAB:", skin_tone.lab.round(2).tolist())
        plot_skin_preview(skin_tone)
        plt.show()
        '''
    ),
    markdown(
        r'''
        ## 5. Rank foundation colours

        CIEDE2000 compares the extracted LAB colour with every catalogue colour.
        Smaller Delta E values indicate closer perceptual similarity.
        '''
    ),
    code(
        r'''
        TOP_N = 5

        recommendations = recommend_foundations(
            skin_tone.lab,
            products,
            top_n=TOP_N,
        )

        display(
            recommendations[
                ["brand", "product", "hex", "color_distance"]
            ].style.format({"color_distance": "{:.2f}"})
        )
        plot_match_swatches(skin_tone.rgb, recommendations)
        plt.show()
        '''
    ),
    markdown(
        r'''
        ## 6. Interpretation

        The first row is the closest digital colour in the catalogue, not
        necessarily the best real-world product. Formula, oxidation, coverage,
        undertone preferences, lighting, price, and availability are outside the
        current dataset.

        A production version should use calibrated images and professionally
        labelled foundation matches rather than relying only on digital HEX values.
        '''
    ),
]


EVALUATION_CELLS = [
    markdown(
        r'''
        # FairFace Pipeline Evaluation

        This notebook tests whether the complete face-detection, skin-sampling,
        and recommendation pipeline finishes successfully on a balanced sample
        of 140 adult FairFace images (20 from each of seven labelled groups).

        **Important:** FairFace does not provide correct foundation labels. This
        notebook measures pipeline completion, not shade-match accuracy. Group
        rates are descriptive diagnostics and are not proof of fairness.
        '''
    ),
    markdown("## 1. Setup"),
    code(SETUP_CELL),
    code(
        r'''
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd
        from datasets import load_dataset
        from IPython.display import display

        from foundation_matcher.config import FAIRFACE_RACE_NAMES
        from foundation_matcher.data import load_foundation_catalog
        from foundation_matcher.evaluation import (
            evaluate_face_pipeline,
            select_balanced_adult_samples,
            simulate_colour_robustness,
            summarize_group_completion,
        )
        from foundation_matcher.face import (
            create_face_landmarker,
            download_face_landmarker,
        )
        from foundation_matcher.visualization import (
            plot_group_completion,
            plot_previews,
        )
        '''
    ),
    markdown(
        r'''
        ## 2. Build one reproducible 140-image sample

        The original notebook selected 35 images and later asserted that the same
        object contained 140. This version selects 20 images per group once and
        verifies the final total before evaluation.
        '''
    ),
    code(
        r'''
        products = load_foundation_catalog()
        model_path = download_face_landmarker("models/face_landmarker.task")

        fairface_stream = load_dataset(
            "HuggingFaceM4/FairFace",
            "0.25",
            split="validation",
            streaming=True,
        ).shuffle(seed=42, buffer_size=2_000)

        balanced_samples = select_balanced_adult_samples(
            fairface_stream,
            samples_per_group=20,
        )

        assert len(balanced_samples) == 140
        print("Balanced evaluation images:", len(balanced_samples))
        '''
    ),
    markdown(
        r'''
        ## 3. Run the complete pipeline

        A run is successful only when a face is detected, skin colour is
        extracted, and five recommendations are returned. Individual failures
        are recorded instead of stopping the whole experiment.
        '''
    ),
    code(
        r'''
        landmarker = create_face_landmarker(model_path)
        try:
            evaluation, previews = evaluate_face_pipeline(
                balanced_samples,
                FAIRFACE_RACE_NAMES,
                landmarker,
                products,
                image_directory="outputs/fairface_test_140",
                top_n=5,
                preview_limit=14,
            )
        finally:
            landmarker.close()

        overall_completion = evaluation["pipeline_success"].mean()
        print(f"Successful runs: {evaluation['pipeline_success'].sum()}/{len(evaluation)}")
        print(f"Overall pipeline completion: {overall_completion:.1%}")

        output_path = Path("outputs/tables/fairface_evaluation_140.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation.to_csv(output_path, index=False)
        print("Saved:", output_path)
        '''
    ),
    markdown("## 4. Inspect completion by group and failure reason"),
    code(
        r'''
        group_summary = summarize_group_completion(evaluation)
        display(group_summary.style.format({"success_rate": "{:.1f}%"}))

        plot_group_completion(group_summary)
        plt.show()
        '''
    ),
    code(
        r'''
        failures = evaluation.loc[
            ~evaluation["pipeline_success"],
            ["image_id", "group", "error"],
        ]

        print("Failed runs:", len(failures))
        display(failures if not failures.empty else pd.DataFrame({"status": ["No failures"]}))

        if previews:
            plot_previews(previews)
            plt.show()
        '''
    ),
    markdown(
        r'''
        ## 5. Simulated colour robustness

        This second experiment starts with a catalogue LAB colour, adds a small
        synthetic perturbation, and checks what the recommender retrieves.

        Exact-row recovery is not the same as perceptual quality because different
        brands can share identical or nearly identical digital colours. Delta E
        thresholds are therefore reported separately.
        '''
    ),
    code(
        r'''
        robustness = simulate_colour_robustness(
            products,
            number_of_tests=500,
            random_state=42,
        )

        robustness_table = pd.Series(robustness, name="value").to_frame()
        display(robustness_table.round(3))
        '''
    ),
    markdown(
        r'''
        ## 6. What this evaluation can support

        **Supported conclusion:** report the overall and group-specific pipeline
        completion rates for this sample, with failure reasons and sample sizes.

        **Unsupported conclusion:** do not call these results match accuracy,
        precision, recall, or F1. Those metrics require ground-truth foundation
        labels. A stronger study would collect professionally labelled matches,
        calibrated images, repeated lighting conditions, and enough participants
        for uncertainty estimates and subgroup analysis.
        '''
    ),
]


CLUSTERING_CELLS = [
    markdown(
        r'''
        # Unsupervised Foundation Shade Clustering

        This notebook trains K-Means on product LAB values to explore whether the
        catalogue contains naturally separated colour groups. Clustering is an
        optional candidate-filtering experiment; CIEDE2000 remains the final
        ranking method.
        '''
    ),
    markdown("## 1. Setup"),
    code(SETUP_CELL),
    code(
        r'''
        import matplotlib.pyplot as plt
        import numpy as np
        from IPython.display import display

        from foundation_matcher.data import load_foundation_catalog
        from foundation_matcher.recommender import (
            evaluate_cluster_counts,
            fit_shade_clusters,
            recommend_foundations,
            recommend_foundations_clustered,
        )
        from foundation_matcher.visualization import (
            plot_cluster_metrics,
            plot_shade_clusters,
        )
        '''
    ),
    markdown("## 2. Compare candidate values of K"),
    code(
        r'''
        products = load_foundation_catalog()
        cluster_evaluation = evaluate_cluster_counts(products, range(2, 13))
        display(cluster_evaluation.round(3))

        plot_cluster_metrics(cluster_evaluation)
        plt.show()
        '''
    ),
    markdown(
        r'''
        Silhouette score is better when higher; Davies-Bouldin score is better
        when lower. A mathematically strong K is not automatically the most useful
        business segmentation, especially if a small K only separates light and
        dark shades. Interpret cluster centres and sizes before assigning meaning.
        '''
    ),
    code(
        r'''
        best_k = int(
            cluster_evaluation.loc[
                cluster_evaluation["silhouette_score"].idxmax(),
                "clusters",
            ]
        )
        print("Best K by silhouette score:", best_k)

        cluster_model, clustered_products = fit_shade_clusters(products, best_k)
        display(
            clustered_products["shade_cluster"]
            .value_counts()
            .sort_index()
            .rename("products")
            .to_frame()
        )
        '''
    ),
    code(
        r'''
        plot_shade_clusters(clustered_products)
        plt.show()
        '''
    ),
    markdown(
        r'''
        ## 3. Compare global and cluster-restricted ranking

        The example below uses a synthetic LAB input so the notebook is
        reproducible without a personal selfie. The clustered method predicts one
        group first, then ranks products within it. Compare it with the global
        baseline because cluster boundaries can exclude a genuinely close colour.
        '''
    ),
    code(
        r'''
        example_skin_lab = np.array([65.0, 12.0, 18.0])

        global_matches = recommend_foundations(
            example_skin_lab,
            products,
            top_n=5,
        )
        predicted_cluster, clustered_matches = recommend_foundations_clustered(
            example_skin_lab,
            clustered_products,
            cluster_model,
            top_n=5,
        )

        print("Predicted shade cluster:", predicted_cluster)
        print("\nGlobal CIEDE2000 ranking")
        display(global_matches[["brand", "product", "hex", "color_distance"]])
        print("\nCluster-restricted ranking")
        display(
            clustered_matches[
                ["brand", "product", "hex", "shade_cluster", "color_distance"]
            ]
        )
        '''
    ),
    markdown(
        r'''
        ## 4. Conclusion

        Treat K-Means as an interpretable exploratory model, not evidence that it
        improves recommendations. To claim improvement, compare both methods
        against independent, professionally labelled foundation matches.
        '''
    ),
]


REVIEW_CELLS = [
    markdown(
        r'''
        # Product Satisfaction from Review Text

        This is an **independent experimental extension**. It does not improve the
        selfie colour matcher directly. The goal is to predict whether a review is
        positive from its text using a TF-IDF and logistic-regression baseline.

        The split is grouped by product so that reviews for the same product do
        not appear in both training and test sets.
        '''
    ),
    markdown("## 1. Setup"),
    code(SETUP_CELL),
    code(
        r'''
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import display
        from sklearn.metrics import ConfusionMatrixDisplay

        from foundation_matcher.config import LUXXIFY_REVIEW_URL
        from foundation_matcher.review_model import (
            build_review_classifier,
            evaluate_review_classifier,
            grouped_train_test_split,
            prepare_review_data,
        )
        '''
    ),
    markdown(
        r'''
        ## 2. Load and prepare reviews

        Ratings of 4 or 5 are labelled positive. Because the dataset is strongly
        imbalanced, later evaluation includes balanced accuracy, class-specific
        precision/recall/F1, ROC AUC, and average precision—not accuracy alone.
        '''
    ),
    code(
        r'''
        makeup_reviews = pd.read_csv(LUXXIFY_REVIEW_URL, low_memory=False)

        print("Raw review shape:", makeup_reviews.shape)
        print("Available columns:", makeup_reviews.columns.tolist())

        review_ml = prepare_review_data(
            makeup_reviews,
            positive_threshold=4,
        )

        print("Prepared examples:", len(review_ml))
        display(
            review_ml["positive_review"]
            .value_counts()
            .rename_axis("class")
            .to_frame("count")
            .assign(proportion=lambda table: table["count"] / table["count"].sum())
        )
        '''
    ),
    markdown(
        r'''
        ### Optional Colab-sized sample

        The full dataset can be used by setting `MAX_ROWS = None`. The default
        keeps the baseline practical on a standard Colab runtime while retaining
        both classes. This is a development sample, not a final benchmark.
        '''
    ),
    code(
        r'''
        MAX_ROWS = 120_000

        if MAX_ROWS is not None and len(review_ml) > MAX_ROWS:
            fraction = MAX_ROWS / len(review_ml)
            review_ml = (
                review_ml.groupby("positive_review", group_keys=False)
                .sample(frac=fraction, random_state=42)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

        print("Examples used:", len(review_ml))
        '''
    ),
    markdown("## 3. Make a product-grouped train/test split"),
    code(
        r'''
        (
            X_train,
            X_test,
            y_train,
            y_test,
            train_groups,
            test_groups,
        ) = grouped_train_test_split(review_ml)

        product_overlap = set(train_groups).intersection(test_groups)
        assert not product_overlap, "Product leakage detected."

        print(f"Training reviews: {len(X_train):,}")
        print(f"Testing reviews: {len(X_test):,}")
        print("Products appearing in both sets:", len(product_overlap))
        print("Training positive rate:", f"{y_train.mean():.1%}")
        print("Testing positive rate:", f"{y_test.mean():.1%}")
        '''
    ),
    markdown("## 4. Train the baseline"),
    code(
        r'''
        review_classifier = build_review_classifier(
            max_features=40_000,
            minimum_document_frequency=5,
        )

        review_classifier.fit(X_train, y_train)
        print("Training complete.")
        '''
    ),
    markdown("## 5. Evaluate with imbalance-aware metrics"),
    code(
        r'''
        metrics, classification_table, confusion = evaluate_review_classifier(
            review_classifier,
            X_test,
            y_test,
        )

        always_positive_accuracy = y_test.mean()
        print("Always-positive accuracy:", f"{always_positive_accuracy:.3f}")
        display(metrics.to_frame().round(3))
        display(classification_table.round(3))

        ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=["Negative", "Positive"],
        ).plot(cmap="Blues", colorbar=False)
        plt.title("Review Satisfaction Confusion Matrix")
        plt.grid(False)
        plt.show()
        '''
    ),
    markdown("## 6. Inspect influential words and phrases"),
    code(
        r'''
        feature_names = review_classifier.named_steps["tfidf"].get_feature_names_out()
        coefficients = review_classifier.named_steps["classifier"].coef_[0]

        coefficient_table = pd.DataFrame(
            {"term": feature_names, "coefficient": coefficients}
        )

        print("Terms most associated with negative reviews")
        display(coefficient_table.nsmallest(15, "coefficient"))

        print("Terms most associated with positive reviews")
        display(coefficient_table.nlargest(15, "coefficient"))
        '''
    ),
    markdown(
        r'''
        ## 7. Interpretation and next steps

        Compare the trained model with the always-positive baseline. High raw
        accuracy is not impressive when most reviews are positive. Balanced
        accuracy and negative-class recall reveal whether the classifier learns
        useful minority-class signals.

        Ratings are used to create the label, so this experiment predicts
        rating-derived sentiment rather than long-term product satisfaction.
        Stronger work could add calibration, repeated grouped cross-validation,
        error analysis, and product metadata while preserving product-level
        separation between training and evaluation.
        '''
    ),
]


OUTPUTS = {
    ROOT / "notebooks" / "01_foundation_matcher_demo.ipynb": DEMO_CELLS,
    ROOT / "notebooks" / "02_fairface_pipeline_evaluation.ipynb": EVALUATION_CELLS,
    ROOT / "notebooks" / "03_shade_clustering.ipynb": CLUSTERING_CELLS,
    ROOT / "experiments" / "04_review_satisfaction_baseline.ipynb": REVIEW_CELLS,
}


def main() -> None:
    for output_path, cells in OUTPUTS.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path.relative_to(ROOT)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
