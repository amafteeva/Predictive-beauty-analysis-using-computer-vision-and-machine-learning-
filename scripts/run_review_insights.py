"""Turn Luxxify reviews into display-only product insights for the recommender.

Implements the five review uses in foundation_matcher.review_insights and
writes everything to outputs/review_insights/:

  product_links.csv              catalogue product line -> Luxxify foundation products
  foundation_aspect_summary.csv  aspect mention rates + rating difference (EDA)
  shade_sentence_scores.csv      out-of-fold classifier scores for shade sentences
  shade_sentence_validation.json agreement of those scores with the review's own rating
  product_insights.csv           one insight card per catalogue product line
  insight_cards.md               the same cards as readable text
  tiebreak_evaluation.csv        how often the experimental tie-break changes results
  plots/*.png                    aspect prevalence, rating effect, tie-break impact
  REPORT.md                      summary with all caveats

Shade recommendations themselves are never changed by this script.

Usage:
    python scripts/run_review_insights.py
    python scripts/run_review_insights.py \
        --reviews-path data/cleaned_makeup_reviews.csv \
        --products-path data/cleaned_makeup_products.csv \
        --catalogue-path data/shades.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foundation_matcher.config import (  # noqa: E402
    FOUNDATION_DATA_URL,
    LUXXIFY_PRODUCT_URL,
    LUXXIFY_REVIEW_URL,
)
from foundation_matcher.data import load_foundation_catalog  # noqa: E402
from foundation_matcher.recommender import recommend_foundations  # noqa: E402
from foundation_matcher.review_insights import (  # noqa: E402
    build_product_insights,
    insight_card_text,
    link_catalogue_to_luxxify,
    product_tags,
    rerank_with_review_tiebreak,
    score_shade_sentences,
    summarize_aspects,
)
from foundation_matcher.review_model import (  # noqa: E402
    attach_product_metadata,
    prepare_review_data,
)

BLUE, ORANGE, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
FIXED_LAB_QUERIES = [
    (45.0, 12.0, 18.0),
    (60.0, 8.0, 22.0),
    (30.0, 15.0, 10.0),
    (70.0, 5.0, 15.0),
    (50.0, 20.0, 25.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reviews-path", default=LUXXIFY_REVIEW_URL)
    parser.add_argument("--products-path", default=LUXXIFY_PRODUCT_URL)
    parser.add_argument("--catalogue-path", default=FOUNDATION_DATA_URL)
    parser.add_argument(
        "--fairface-table",
        default=str(PROJECT_ROOT / "outputs/tables/fairface_evaluation_140.csv"),
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs/review_insights")
    )
    parser.add_argument(
        "--lr-config",
        default=str(
            PROJECT_ROOT / "outputs/review_experiment/metrics/lr_hparam_search_text.csv"
        ),
        help="Selected LR settings (best row used); falls back to C=10, balanced.",
    )
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--min-shade-sentences", type=int, default=10)
    return parser.parse_args()


def style_axis(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=24)
    ax.annotate(
        subtitle,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def md_table(frame: pd.DataFrame) -> str:
    """Minimal Markdown table (avoids an optional `tabulate` dependency)."""
    cells = [[str(v) for v in frame.columns]] + [
        [str(v) for v in row] for row in frame.itertuples(index=False)
    ]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines)


def label(aspect: str) -> str:
    return aspect.replace("_", " ")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    reviews_raw = pd.read_csv(args.reviews_path, low_memory=False)
    products = pd.read_csv(args.products_path, low_memory=False)
    catalogue = load_foundation_catalog(args.catalogue_path)

    prepared = attach_product_metadata(
        prepare_review_data(
            reviews_raw,
            text_column="comments",
            rating_column="rating",
            group_column="product_link_id",
            positive_threshold=4,
        ),
        products,
    )
    foundation = prepared[prepared["category"] == "Foundation"]
    print(
        f"Prepared reviews: {len(prepared):,}; "
        f"foundation reviews with text: {len(foundation):,}"
    )

    # 1. Linkage
    links = link_catalogue_to_luxxify(catalogue, products)
    links_out = links.assign(
        luxxify_product_link_ids=links["luxxify_product_link_ids"].map(
            lambda v: ";".join(map(str, v))
        ),
        luxxify_product_names=links["luxxify_product_names"].map("; ".join),
    )
    links_out.to_csv(out / "product_links.csv", index=False)
    print(f"Linked product lines: {links['linked'].sum()} / {len(links)}")

    # 2. Aspect EDA over foundation reviews
    aspects = summarize_aspects(foundation)
    aspects.to_csv(out / "foundation_aspect_summary.csv", index=False)

    # 3. Classifier on shade sentences, cross-fitted
    C, class_weight = 10.0, "balanced"
    if Path(args.lr_config).exists():
        best = (
            pd.read_csv(args.lr_config)
            .sort_values("val_macro_f1", ascending=False)
            .iloc[0]
        )
        C, class_weight = (
            float(best["C"]),
            (None if best["class_weight"] == "none" else best["class_weight"]),
        )
    print(f"Scoring shade sentences (C={C}, class_weight={class_weight}) ...")
    sentences = score_shade_sentences(
        prepared, foundation, C=C, class_weight=class_weight
    )
    sentences[
        [
            "review_id",
            "product_link_id",
            "rating",
            "positive_review",
            "sentence",
            "p_positive",
            "review_level_p_positive",
            "fold",
        ]
    ].to_csv(out / "shade_sentence_scores.csv", index=False)
    scored = sentences.dropna(subset=["p_positive"])
    validation = {
        "shade_sentences": int(len(scored)),
        "reviews_with_shade_sentences": int(scored["review_id"].nunique()),
        "foundation_text_reviews": int(len(foundation)),
        "auc_sentence_score_vs_review_label": float(
            roc_auc_score(scored["positive_review"], scored["p_positive"])
        ),
        "auc_full_review_score_vs_review_label": float(
            roc_auc_score(scored["positive_review"], scored["review_level_p_positive"])
        ),
        "classifier_settings": {
            "C": C,
            "class_weight": class_weight,
            "folds": 5,
            "grouping": "build_leakage_safe_groups (product + shared-text unions)",
        },
        "note": (
            "AUC against the review's own star-rating label: agreement with "
            "overall sentiment, not shade-match accuracy."
        ),
    }
    (out / "shade_sentence_validation.json").write_text(
        json.dumps(validation, indent=2)
    )

    # 4. Insight cards
    insights = build_product_insights(
        links, reviews_raw, foundation, sentences, product_tags(products)
    )
    insights.to_csv(out / "product_insights.csv", index=False)
    linked = insights[insights["linked"]].sort_values("text_reviews", ascending=False)
    cards = [
        "# Review insight cards (display only)",
        "",
        "Descriptive signals from Luxxify reviews for catalogue product lines "
        "that could be linked. Mentions are keyword matches (negation ignored); "
        "ratings are product-level, not shade-level.",
        "",
    ]
    cards += [f"- {insight_card_text(r)}" for r in linked.to_dict("records")]
    cards += [
        "",
        f"Not linked ({(~insights['linked']).sum()} product lines): "
        + ", ".join(
            f"{b} {p}"
            for b, p in insights.loc[~insights["linked"], ["brand", "product"]].values
        ),
    ]
    (out / "insight_cards.md").write_text("\n".join(cards))

    # 5. Experimental tie-break evaluation: fixed queries + saved FairFace skin colours
    queries = [("fixed", q) for q in FIXED_LAB_QUERIES]
    if Path(args.fairface_table).exists():
        ff = pd.read_csv(args.fairface_table)
        ff = ff[ff["pipeline_success"]]
        queries += [
            ("fairface", tuple(v))
            for v in ff[["skin_L", "skin_a", "skin_b"]].to_numpy()
        ]
    neutral = float(scored["p_positive"].mean())
    rows = []
    for tol in args.tolerances:
        for source, q in queries:
            pool = recommend_foundations(np.array(q), catalogue, top_n=15)
            base = pool.head(5)
            new = rerank_with_review_tiebreak(
                pool,
                insights,
                tolerance=tol,
                top_n=5,
                min_shade_sentences=args.min_shade_sentences,
                neutral_score=neutral,
            )
            rows.append(
                {
                    "tolerance": tol,
                    "source": source,
                    "order_changed": list(zip(base.brand, base["product"], strict=True))
                    != list(zip(new.brand, new["product"], strict=True)),
                    "set_changed": set(zip(base.brand, base["product"], strict=True))
                    != set(zip(new.brand, new["product"], strict=True)),
                    "top1_changed": (base.brand[0], base["product"][0])
                    != (new.brand[0], new["product"][0]),
                    "top1_delta_e_cost": float(
                        new["color_distance"][0] - base["color_distance"][0]
                    ),
                    "max_rank_delta_e_cost": float(
                        (
                            new["color_distance"].to_numpy()
                            - base["color_distance"].to_numpy()
                        ).max()
                    ),
                }
            )
    tb = pd.DataFrame(rows)
    tb.to_csv(out / "tiebreak_evaluation_per_query.csv", index=False)
    summary = (
        tb.groupby("tolerance")
        .agg(
            queries=("source", "size"),
            order_changed=("order_changed", "mean"),
            set_changed=("set_changed", "mean"),
            top1_changed=("top1_changed", "mean"),
            max_top1_delta_e_cost=("top1_delta_e_cost", "max"),
            mean_top1_delta_e_cost=("top1_delta_e_cost", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "tiebreak_evaluation.csv", index=False)
    assert (tb["top1_delta_e_cost"] <= tb["tolerance"] + 1e-9).all(), (
        "tie-break exceeded its ΔE tolerance"
    )

    # Plots
    a = aspects.sort_values("share_of_reviews")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(
        [label(x) for x in a["aspect"]],
        a["share_of_reviews"] * 100,
        color=BLUE,
        height=0.65,
    )
    for b in bars:
        ax.text(
            b.get_width() + 0.5,
            b.get_y() + b.get_height() / 2,
            f"{b.get_width():.1f}%",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, a["share_of_reviews"].max() * 118)
    ax.set_xlabel("% of foundation reviews with text that mention the aspect")
    style_axis(
        ax,
        "What foundation reviewers talk about",
        f"{len(foundation):,} de-duplicated Luxxify foundation reviews · "
        "keyword mentions, negation ignored",
    )
    fig.tight_layout()
    fig.savefig(
        out / "plots/foundation_aspect_prevalence.png", dpi=300, facecolor="white"
    )
    plt.close(fig)

    a = aspects.sort_values("positive_rate_difference")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    vals = a["positive_rate_difference"] * 100
    bars = ax.barh(
        [label(x) for x in a["aspect"]],
        vals,
        color=[ORANGE if v < 0 else BLUE for v in vals],
        height=0.65,
    )
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            v + (0.8 if v >= 0 else -0.8),
            b.get_y() + b.get_height() / 2,
            f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color=INK,
        )
    ax.axvline(0, color=INK_2, linewidth=0.8)
    lim = max(abs(vals.min()), abs(vals.max())) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(
        "Positive-review rate when mentioned minus otherwise (percentage points)"
    )
    style_axis(
        ax,
        "Which mentions go with lower ratings?",
        "Positive = 4–5 stars · association only, not cause",
    )
    fig.tight_layout()
    fig.savefig(
        out / "plots/foundation_aspect_rating_effect.png", dpi=300, facecolor="white"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(summary))
    for j, (col, lab, color) in enumerate(
        [
            ("order_changed", "Top-5 order changed", BLUE),
            ("top1_changed", "#1 shade changed", ORANGE),
        ]
    ):
        bars = ax.bar(
            x + (j - 0.5) * 0.38, summary[col] * 100, width=0.36, color=color, label=lab
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.8,
                f"{b.get_height():.0f}%",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(x, [f"ΔE tolerance {t:g}" for t in summary["tolerance"]])
    ax.set_ylabel(f"% of {int(summary['queries'].iloc[0])} skin-colour queries")
    ax.set_ylim(0, max(summary["order_changed"].max() * 100 * 1.25, 10))
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    style_axis(
        ax,
        "Experimental review tie-break: how often it changes results",
        "Only shades within the ΔE tolerance of each other can swap · "
        "benefit is unmeasured",
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "plots/tiebreak_impact.png", dpi=300, facecolor="white")
    plt.close(fig)

    # Report
    n_linked = int(links["linked"].sum())
    ok = insights[insights["evidence"] == "ok"]
    with_sentences = insights.loc[insights["shade_sentences"] > 0, "shade_sentences"]
    aspect_cols = [
        "aspect",
        "reviews_mentioning",
        "share_of_reviews",
        "positive_rate_when_mentioned",
        "positive_rate_otherwise",
        "positive_rate_difference",
    ]
    linked_names = links_out.loc[
        links_out["linked"], ["brand", "product", "luxxify_product_names"]
    ]
    most_mentioned = ", ".join(
        f"{label(r.aspect)} ({r.share_of_reviews:.1%})"
        for r in aspects.head(4).itertuples()
    )
    auc_sentence = validation["auc_sentence_score_vs_review_label"]
    auc_review = validation["auc_full_review_score_vs_review_label"]
    lines = [
        "# Review insights for the shade recommender",
        "",
        "Generated by `scripts/run_review_insights.py`. **Shade rankings are not",
        "changed by anything here**: insight cards are joined onto recommendations",
        "for display only, and the tie-break is an opt-in experiment.",
        "",
        "## What the reviews can and cannot do",
        "",
        "- Reviews never record which shade was bought or the reviewer's skin tone,",
        "  so they **cannot validate a shade match**.",
        f"- Only **{n_linked} of {len(links)}** catalogue product lines link to a",
        "  Luxxify foundation product (brand + product-line name; see",
        f"  `product_links.csv`); {len(ok)} of them have >= 30 text reviews.",
        "- Aspect mentions are keyword matches that ignore negation",
        '  ("doesn\'t oxidise" counts as an oxidising mention).',
        "",
        "## 1. Product linkage",
        "",
        md_table(linked_names),
        "",
        f"## 2. What foundation reviewers talk about ({len(foundation):,} reviews)",
        "",
        "De-duplicated Luxxify foundation reviews with text.",
        "",
        md_table(aspects[aspect_cols].round(3)),
        "",
        f"Most mentioned: {most_mentioned}.",
        "Rating differences are associations, not causes.",
        "",
        "## 3. Shade-sentence sentiment (classifier repurposed)",
        "",
        f"The TF-IDF + logistic-regression classifier (C = {C:g}, class_weight =",
        f"{class_weight}) was cross-fitted over 5 leakage-safe product-group folds,",
        "so every sentence is scored by a model that never saw its product.",
        f"{validation['shade_sentences']:,} shade-related sentences from",
        f"{validation['reviews_with_shade_sentences']:,} reviews were scored.",
        "Agreement with the review's own star-rating label: ROC AUC",
        f"{auc_sentence:.3f} for the shade sentence alone vs {auc_review:.3f} for",
        "the whole review. This measures agreement with overall sentiment,",
        "**not** whether the shade matched.",
        "",
        "## 4. Insight cards",
        "",
        "See `insight_cards.md` and `product_insights.csv`; join them onto",
        "recommendations with",
        "`review_insights.attach_review_insights(recommendations, insights)`.",
        "The `pooled_tag_*` columns come from the product export's pooled tag",
        "lists (no counts). Each tag appears on 59-72% of foundation products,",
        "so they are not shown on cards.",
        "",
        "## 5. Experimental tie-break (not used by default)",
        "",
        md_table(summary.round(3)),
        "",
        "Queries: 5 fixed CIELAB queries + the saved skin colours of the",
        "successful FairFace runs. By construction the #1 shade's ΔE never rises",
        "by more than the tolerance (max observed cost in the table above).",
        f"Products with fewer than {args.min_shade_sentences} scored shade sentences",
        "get a neutral score equal to the mean of all scored shade sentences",
        f"({neutral:.3f}), so the tie-break favours linked products scoring above",
        "average and demotes those below it. Product scores rest on only",
        f"{int(with_sentences.min())}-{int(with_sentences.max())} sentences each",
        "(see `shade_sentence_score_se`), and only",
        f"{n_linked} of {len(links)} product lines can be scored at all, so many of",
        "the changes in the table above come from noisy scores for a few big",
        "brands. **Whether it improves real-world matches is unmeasured**: that",
        "would need ground-truth shade labels.",
        "",
    ]
    report = "\n".join(lines)
    (out / "REPORT.md").write_text(report)
    print(f"Done. Outputs in {out}")
    print(json.dumps(validation, indent=2))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
