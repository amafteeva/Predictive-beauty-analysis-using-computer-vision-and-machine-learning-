"""Consistent, notebook-friendly visualizations."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from foundation_matcher.color import rgb_to_hex


def plot_catalog_lab(products: pd.DataFrame):
    """Show catalogue coverage in the LAB a/b plane using product HEX colours."""

    colours = [f"#{value}" for value in products["hex"]]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        products["lab_a"],
        products["lab_b"],
        c=colours,
        s=35,
        edgecolors="black",
        linewidths=0.2,
    )
    axis.set(
        title="Foundation Shade Colour Distribution",
        xlabel="LAB a: Green to Red",
        ylabel="LAB b: Blue to Yellow",
    )
    axis.grid(False)
    figure.tight_layout()
    return figure, axis


def plot_brand_coverage(products: pd.DataFrame, *, thin_brand_threshold: int = 5):
    """Show shade count per brand, flagging brands too thin for a real match.

    A brand with only a handful of shades can dominate a top-N recommendation
    list without actually offering good coverage across skin tones.
    """

    counts = products.groupby("brand").size().sort_values()
    colours = [
        "#D62728" if count < thin_brand_threshold else "#1F77B4" for count in counts
    ]

    figure, axis = plt.subplots(figsize=(8, max(4, 0.28 * len(counts))))
    axis.barh(counts.index, counts.to_numpy(), color=colours)
    axis.set(
        title=f"Shades per Brand (red = fewer than {thin_brand_threshold})",
        xlabel="Number of shades",
    )
    axis.grid(False)
    figure.tight_layout()
    return figure, axis


def plot_lab_distributions(products: pd.DataFrame):
    """Show the marginal distribution of each CIELAB channel in the catalogue."""

    channels = [("lab_L", "L: Lightness"), ("lab_a", "a: Green-Red"), ("lab_b", "b: Blue-Yellow")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (column, title) in zip(axes, channels, strict=True):
        axis.hist(products[column], bins=30, color="#1F77B4", edgecolor="white")
        axis.set(title=title, xlabel=column)
        axis.grid(False)
    figure.tight_layout()
    return figure, axes


def plot_price_distribution(
    products: pd.DataFrame,
    *,
    row_label: str = "Shades",
    title: str = "Estimated Price Distribution",
):
    """Show the distribution of prices, noting how much of the data has one.

    Reusable for both the foundation catalogue (after attach_price_estimates,
    where every price genuinely is an estimate) and a raw product export
    with its own real listed prices — pass row_label/title to match
    whichever it is.
    """

    if "price" not in products.columns:
        raise ValueError("products has no 'price' column; run attach_price_estimates first.")

    priced = products["price"].dropna()
    coverage = len(priced) / len(products)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(priced, bins=30, color="#2CA02C", edgecolor="white")
    axis.set(
        title=f"{title} ({coverage:.0%} of rows have a price)",
        xlabel="Price (USD)",
        ylabel=row_label,
    )
    axis.grid(False)
    figure.tight_layout()
    return figure, axis


def plot_rating_distribution(reviews: pd.DataFrame, *, rating_column: str = "rating"):
    """Show the raw star-rating distribution before any positive/negative split."""

    counts = reviews[rating_column].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(counts.index.astype(str), counts.to_numpy(), color="#1F77B4")
    axis.set(title="Review Rating Distribution", xlabel="Stars", ylabel="Reviews")
    axis.grid(False)
    figure.tight_layout()
    return figure, axis


def plot_review_length_distribution(reviews: pd.DataFrame, *, text_column: str = "comments"):
    """Show how long (in words) the reviews that have text actually are."""

    lengths = reviews[text_column].dropna().str.split().str.len()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(
        lengths,
        bins=40,
        range=(0, lengths.quantile(0.99)),
        color="#9467BD",
        edgecolor="white",
    )
    axis.set(
        title="Review Length Distribution (word count)",
        xlabel="Words",
        ylabel="Reviews",
    )
    axis.grid(False)
    figure.tight_layout()
    return figure, axis


def plot_category_distribution(
    products: pd.DataFrame, *, category_column: str = "category", top_n: int = 12
):
    """Show the most common product categories in a product export."""

    counts = products[category_column].value_counts().head(top_n).sort_values()
    figure, axis = plt.subplots(figsize=(8, max(4, 0.35 * len(counts))))
    axis.barh(counts.index.astype(str), counts.to_numpy(), color="#FF7F0E")
    axis.set(title=f"Top {top_n} Product Categories", xlabel="Products")
    axis.grid(False)
    figure.tight_layout()
    return figure, axis


def plot_skin_preview(skin_tone):
    """Show selected facial regions beside the extracted median colour."""

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(skin_tone.preview_rgb)
    axes[0].set_title("Selected cheek and forehead regions")
    axes[0].axis("off")

    swatch = np.full((200, 200, 3), skin_tone.rgb, dtype=np.uint8)
    axes[1].imshow(swatch)
    axes[1].set_title(f"Detected RGB: {skin_tone.rgb.tolist()}")
    axes[1].axis("off")
    figure.tight_layout()
    return figure, axes


def plot_skin_previews(skin_tones, labels: list[str] | None = None):
    """Show selected regions and detected colour for several images, one row each."""

    if not skin_tones:
        raise ValueError("skin_tones must contain at least one SkinTone.")
    if labels is None:
        labels = [f"Image {index + 1}" for index in range(len(skin_tones))]
    elif len(labels) != len(skin_tones):
        raise ValueError("labels must have the same length as skin_tones.")

    figure, axes = plt.subplots(len(skin_tones), 2, figsize=(10, 5 * len(skin_tones)))
    axes = np.atleast_2d(axes)
    for row, (skin_tone, label) in enumerate(zip(skin_tones, labels, strict=True)):
        axes[row, 0].imshow(skin_tone.preview_rgb)
        axes[row, 0].set_title(f"{label}: selected regions")
        axes[row, 0].axis("off")

        swatch = np.full((200, 200, 3), skin_tone.rgb, dtype=np.uint8)
        axes[row, 1].imshow(swatch)
        axes[row, 1].set_title(f"{label}: RGB {skin_tone.rgb.tolist()}")
        axes[row, 1].axis("off")
    figure.tight_layout()
    return figure, axes


def plot_match_swatches(
    skin_rgb: np.ndarray,
    recommendations: pd.DataFrame,
):
    """Display the detected skin colour beside recommended product colours."""

    colours = [rgb_to_hex(skin_rgb)] + [f"#{value}" for value in recommendations["hex"]]
    has_price = "price" in recommendations.columns

    def _label(row) -> str:
        price_line = ""
        if has_price:
            price_line = (
                "\nUnknown price"
                if pd.isna(row.price)
                else f"\n${row.price:.0f}"
            )
        return f"{row.brand}\n{row.product}{price_line}\nDelta E {row.color_distance:.2f}"

    labels = ["Detected skin"] + [_label(row) for row in recommendations.itertuples()]
    figure, axes = plt.subplots(1, len(colours), figsize=(3 * len(colours), 3))
    axes = np.atleast_1d(axes)
    for axis, colour, label in zip(axes, colours, labels, strict=True):
        swatch = np.ones((100, 100, 3)) * np.asarray(to_rgb(colour))
        axis.imshow(swatch)
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    figure.suptitle("Closest Foundation Colour Matches", fontsize=14)
    figure.tight_layout()
    return figure, axes


def plot_brute_force_latency(latency: pd.DataFrame):
    """Plot brute-force CIEDE2000 search latency against catalogue size.

    Takes the output of recommender.benchmark_brute_force_search and shows
    whether a full scan over every product stays fast as the catalogue grows,
    which is what justifies skipping any candidate-restriction step.
    """

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        latency["catalog_size"],
        latency["seconds_per_query"] * 1000,
        marker="o",
        color="#1F77B4",
    )
    axis.set(
        title="Brute-Force CIEDE2000 Search Latency",
        xlabel="Catalogue size (products)",
        ylabel="Milliseconds per query",
    )
    axis.grid(False)
    figure.tight_layout()
    return figure, axis


def plot_fairface_label_distribution(distributions: dict[str, pd.DataFrame]):
    """Show race/age/gender counts in a raw (unbalanced) FairFace sample.

    Expects the dict returned by summarize_fairface_labels, so categories
    are already in their natural order (age bins are not alphabetical).
    """

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, key in zip(axes, ["race", "age", "gender"], strict=True):
        table = distributions[key]
        axis.bar(table[key].astype(str), table["count"], color="#1F77B4")
        axis.set(title=f"{key.title()} Distribution", ylabel="Samples")
        axis.tick_params(axis="x", rotation=60)
        axis.grid(False)
    figure.tight_layout()
    return figure, axes


def plot_group_completion(group_summary: pd.DataFrame):
    """Plot descriptive pipeline-completion percentages by group."""

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(
        group_summary["group"],
        group_summary["success_rate"],
        color="#1F77B4",
    )
    axis.set(
        title="Pipeline Completion by FairFace Group",
        xlabel="Group",
        ylabel="Completion rate (%)",
        ylim=(0, 105),
    )
    axis.tick_params(axis="x", rotation=45)
    axis.grid(False)
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    figure.tight_layout()
    return figure, axis


def plot_previews(previews: list[dict], columns: int = 7):
    """Display marked facial sampling regions from successful evaluations."""

    rows = max(1, math.ceil(len(previews) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    for axis in axes:
        axis.axis("off")
    for axis, preview in zip(axes, previews):
        axis.imshow(preview["image"])
        axis.set_title(preview["group"], fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    return figure, axes
