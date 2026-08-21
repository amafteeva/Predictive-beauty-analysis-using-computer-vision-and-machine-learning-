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


def plot_match_swatches(
    skin_rgb: np.ndarray,
    recommendations: pd.DataFrame,
):
    """Display the detected skin colour beside recommended product colours."""

    colours = [rgb_to_hex(skin_rgb)] + [f"#{value}" for value in recommendations["hex"]]
    labels = ["Detected skin"] + [
        f"{row.brand}\n{row.product}\nDelta E {row.color_distance:.2f}"
        for row in recommendations.itertuples()
    ]
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


def plot_cluster_metrics(cluster_evaluation: pd.DataFrame):
    """Plot both K-Means selection metrics without background grids."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(
        cluster_evaluation["clusters"],
        cluster_evaluation["silhouette_score"],
        marker="o",
        color="#1F77B4",
    )
    axes[0].set(title="K-Means Silhouette Score", xlabel="Clusters", ylabel="Score")
    axes[1].plot(
        cluster_evaluation["clusters"],
        cluster_evaluation["davies_bouldin_score"],
        marker="o",
        color="orange",
    )
    axes[1].set(
        title="K-Means Davies-Bouldin Score",
        xlabel="Clusters",
        ylabel="Score",
    )
    for axis in axes:
        axis.grid(False)
    figure.tight_layout()
    return figure, axes


def plot_shade_clusters(products: pd.DataFrame):
    """Visualize learned shade groups in the LAB a/b plane."""

    if "shade_cluster" not in products:
        raise ValueError("products must include a 'shade_cluster' column.")
    figure, axis = plt.subplots(figsize=(10, 6))
    scatter = axis.scatter(
        products["lab_a"],
        products["lab_b"],
        c=products["shade_cluster"],
        cmap="tab10",
        s=35,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.2,
    )
    axis.set(
        title="Foundation Shade Clusters Learned by K-Means",
        xlabel="LAB a: Green to Red",
        ylabel="LAB b: Blue to Yellow",
    )
    axis.grid(False)
    figure.colorbar(scatter, ax=axis, label="Shade cluster")
    figure.tight_layout()
    return figure, axis


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
