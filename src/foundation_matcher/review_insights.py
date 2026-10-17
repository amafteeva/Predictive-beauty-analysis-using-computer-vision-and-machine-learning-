"""Review-derived product insights for the shade recommender (display-first).

The Luxxify reviews cannot validate a shade match: they never record which
shade a reviewer bought or the reviewer's skin tone, and only a handful of
the colour catalogue's product lines exist in the Luxxify foundation export.
This module therefore uses reviews to *describe* products rather than to
choose shades:

1. ``link_catalogue_to_luxxify`` -- conservative brand + product-line name
   linkage between the colour catalogue and Luxxify foundation products.
2. ``detect_aspects`` / ``summarize_aspects`` -- keyword-lexicon *mentions*
   of foundation aspects (shade problems, oxidising, coverage, finish, skin
   type, ...) and how star ratings differ when an aspect is mentioned.
3. ``score_shade_sentences`` -- applies the project's TF-IDF + logistic
   regression review classifier, cross-fitted by leakage-safe product
   groups, to shade-related *sentences* to give each product an
   out-of-fold "shade-sentence sentiment" score.
4. ``build_product_insights`` / ``attach_review_insights`` -- one insight
   card per catalogue product line, joined onto recommendations for display
   only (ranking and distances are untouched).
5. ``rerank_with_review_tiebreak`` -- an opt-in, experimental tie-break that
   may reorder only shades whose CIEDE2000 distances are within a small
   tolerance of each other. It is never called by ``recommend_foundations``.

Keyword mentions ignore negation and sarcasm, and the classifier's label is
the review's own star rating, so every score here is a descriptive signal,
not a measure of shade-match accuracy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from foundation_matcher.data import normalize_brand_name
from foundation_matcher.review_model import (
    build_leakage_safe_groups,
    build_review_classifier,
)

# --------------------------------------------------------------------------- aspects
ASPECT_PATTERNS: dict[str, str] = {
    "shade_mismatch": (
        r"\btoo (?:dark|light|pale|yellow|pink|orange|ashy|grey|gray)\b"
        r"|\bwrong (?:shade|colou?r)\b"
        r"|\b(?:couldn'?t|could not|can'?t|cannot) find "
        r"(?:my|a|the right) (?:shade|colou?r|match)\b"
        r"|\b(?:doesn'?t|does not|didn'?t|did not) match\b"
    ),
    "shade_match_positive": (
        r"\bperfect (?:shade |colou?r )?match\b"
        r"|\bmatch(?:es|ed)? my (?:skin|skin ?tone|complexion|colou?r)\b"
        r"|\bfound my (?:perfect )?(?:shade|match|colou?r)\b"
        r"|\bshade (?:is|was) perfect\b"
    ),
    "oxidizes_or_orange": (
        r"\boxidi[sz]|\b(?:turn(?:s|ed)?|goes|went|looks?|looked) orange\b"
    ),
    "undertone": (
        r"\bundertones?\b|\bolive\b"
        r"|\b(?:cool|warm|neutral|pink|yellow) (?:tones?|undertones?)\b"
    ),
    "coverage": r"\bcoverage\b",
    "finish_matte": r"\bmatte\b",
    "finish_dewy": r"\bdewy\b|\bglow(?:y|ing)?\b|\bradiant\b",
    "longevity": (
        r"\blast(?:s|ed)? all day\b|\blong[- ]?(?:wear|wearing|lasting)\b"
        r"|\bwears? off\b|\bfad(?:e|es|ed)\b"
    ),
    "cakey_or_creasing": (
        r"\bcak(?:ey|ed|ing)\b|\bcreas(?:e|es|ed|ing)\b"
        r"|\bsettles? in(?:to)? (?:my )?(?:fine )?lines\b|\bpatchy\b"
    ),
    "oily_skin": r"\boily\b",
    "dry_skin": r"\bdry skin\b|\bdry patches\b|\bflak(?:y|ing|es)\b",
    "breakouts_or_sensitive": (
        r"\bbr(?:oke|eak(?:s|ing)?) (?:me )?out\b|\bbreakouts?\b|\bsensitive\b|\bacne\b"
    ),
}

SHADE_SENTENCE_PATTERN = (
    r"\bshades?\b|\bcolou?r match|\bmatch(?:es|ed)? my\b"
    r"|\btoo (?:dark|light|pale|yellow|pink|orange|ashy)\b"
    r"|\boxidi[sz]|\borange\b|\bundertones?\b|\bashy\b"
)

PRODUCT_TAG_PATTERNS: dict[str, tuple[str, str]] = {
    # tag name: (product-export column, pattern). These export fields are pooled
    # tag lists without counts, so a product with many reviews tends to carry every
    # tag (each appears on 59-72% of Luxxify foundation products). Kept as raw
    # columns only; not shown on cards.
    "best_for_oily_or_acne_prone": ("best_uses", r"oily|acne"),
    "best_for_dry_skin": ("best_uses", r"dry skin"),
    "best_for_mature_skin": ("best_uses", r"mature"),
    "best_for_sensitive_skin": ("best_uses", r"sensitive"),
    "con_oxidizes_or_orange": ("cons", r"oxidi[sz]|orange"),
    "con_shade_match": ("cons", r"shade|colou?r match|too dark|too light"),
    "con_cakey_or_creasing": ("cons", r"cak|creas|settles"),
}

# Known spelling differences between the two sources (colour catalogue -> Luxxify).
PRODUCT_NAME_ALIASES: dict[str, str] = {"infalliable": "infallible"}

MIN_TEXT_REVIEWS_FOR_EVIDENCE = 30
NEUTRAL_SCORE = 0.5


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(value).lower())).strip()


def detect_aspects(texts: pd.Series) -> pd.DataFrame:
    """Return a boolean frame: does each text *mention* each aspect?"""

    lowered = texts.fillna("").astype(str).str.lower()
    return pd.DataFrame(
        {
            aspect: lowered.str.contains(pattern, regex=True)
            for aspect, pattern in ASPECT_PATTERNS.items()
        },
        index=texts.index,
    )


def summarize_aspects(
    reviews: pd.DataFrame,
    *,
    text_column: str = "review_text",
    label_column: str = "positive_review",
    rating_column: str = "rating",
) -> pd.DataFrame:
    """Prevalence of each aspect and how ratings differ when it is mentioned."""

    flags = detect_aspects(reviews[text_column])
    rows = []
    for aspect in flags.columns:
        mentioned = flags[aspect]
        rows.append(
            {
                "aspect": aspect,
                "reviews_mentioning": int(mentioned.sum()),
                "share_of_reviews": float(mentioned.mean()),
                "positive_rate_when_mentioned": float(
                    reviews.loc[mentioned, label_column].mean()
                )
                if mentioned.any()
                else np.nan,
                "positive_rate_otherwise": float(
                    reviews.loc[~mentioned, label_column].mean()
                )
                if (~mentioned).any()
                else np.nan,
                "mean_rating_when_mentioned": float(
                    reviews.loc[mentioned, rating_column].mean()
                )
                if mentioned.any()
                else np.nan,
                "mean_rating_otherwise": float(
                    reviews.loc[~mentioned, rating_column].mean()
                )
                if (~mentioned).any()
                else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["positive_rate_difference"] = (
        table["positive_rate_when_mentioned"] - table["positive_rate_otherwise"]
    )
    return table.sort_values("share_of_reviews", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- linkage
def _brands_match(catalogue_brand: str, luxxify_brand: str) -> bool:
    a, b = normalize_brand_name(catalogue_brand), normalize_brand_name(luxxify_brand)
    if not a or not b:
        return False
    return a == b or a.startswith(b + " ") or b.startswith(a + " ")


def link_catalogue_to_luxxify(
    catalogue: pd.DataFrame,
    luxxify_products: pd.DataFrame,
    *,
    category: str = "Foundation",
    aliases: Mapping[str, str] = PRODUCT_NAME_ALIASES,
    manual_links: Mapping[tuple[str, str], list[str]] | None = None,
) -> pd.DataFrame:
    """Link each catalogue (brand, product line) to Luxxify product ids.

    A link requires (a) matching brand names (exact after normalisation, or
    one a whole-word prefix of the other) and (b) the catalogue's product-line
    name appearing as a whole phrase in the Luxxify product name. This is
    deliberately conservative: Luxxify rows whose brand field is a scraping
    artefact (e.g. "Find your shade") are not guessed at. ``manual_links``
    can add reviewed links as {(brand, product): [product_link_id, ...]}.
    """

    lux = luxxify_products
    if category is not None and "category" in lux.columns:
        lux = lux[lux["category"] == category]
    lux = lux.dropna(subset=["brand", "product_name"])
    lines = catalogue[["brand", "product"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for line in lines.itertuples(index=False):
        phrase = _normalize_text(line.product)
        for wrong, right in aliases.items():
            phrase = re.sub(rf"\b{re.escape(wrong)}\b", right, phrase)
        pattern = rf"(?:^|\s){re.escape(phrase)}(?:\s|$)"
        matches = lux[
            lux["brand"].map(lambda b, cb=line.brand: _brands_match(cb, b))
            & lux["product_name"].map(
                lambda n, p=pattern: re.search(p, _normalize_text(n)) is not None
            )
        ]
        ids = matches["product_link_id"].tolist()
        method = "brand + product-line name"
        if manual_links and (line.brand, line.product) in manual_links:
            ids = list(
                dict.fromkeys(ids + list(manual_links[(line.brand, line.product)]))
            )
            method = "manual + automatic" if len(matches) else "manual"
        rows.append(
            {
                "brand": line.brand,
                "product": line.product,
                "linked": bool(ids),
                "luxxify_product_link_ids": ids,
                "luxxify_product_names": lux.loc[
                    lux["product_link_id"].isin(ids), "product_name"
                ].tolist(),
                "link_method": method if ids else "not linked",
            }
        )
    return pd.DataFrame(rows)


def product_tags(luxxify_products: pd.DataFrame) -> pd.DataFrame:
    """Presence of reviewer tags in the product export's aggregated tag fields."""

    out = pd.DataFrame({"product_link_id": luxxify_products["product_link_id"]})
    for tag, (column, pattern) in PRODUCT_TAG_PATTERNS.items():
        if column in luxxify_products.columns:
            out[tag] = (
                luxxify_products[column]
                .fillna("")
                .str.lower()
                .str.contains(pattern, regex=True)
            )
        else:
            out[tag] = False
    return out


# ------------------------------------------------- classifier on shade sentences
def split_sentences(text: object) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s*\\n\s*", str(text))
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def extract_shade_sentences(
    reviews: pd.DataFrame, *, text_column: str = "review_text"
) -> pd.DataFrame:
    """One row per shade-related sentence, keeping the parent review's columns."""

    rows = []
    pattern = re.compile(SHADE_SENTENCE_PATTERN, flags=re.IGNORECASE)
    for idx, text in reviews[text_column].items():
        for sentence in split_sentences(text):
            if pattern.search(sentence):
                rows.append({"review_index": idx, "sentence": sentence})
    sentences = pd.DataFrame(rows, columns=["review_index", "sentence"])
    return sentences.join(reviews, on="review_index")


def score_shade_sentences(
    all_reviews: pd.DataFrame,
    target_reviews: pd.DataFrame,
    *,
    n_splits: int = 5,
    C: float = 10.0,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> pd.DataFrame:
    """Out-of-fold P(positive) for shade-related sentences in ``target_reviews``.

    ``all_reviews`` (output of prepare_review_data) trains the TF-IDF +
    logistic-regression classifier with the experiment's selected settings.
    Folds are formed from ``build_leakage_safe_groups`` so a sentence is
    always scored by a model that never saw its product (or a product sharing
    copied review text). ``target_reviews`` must be a subset of
    ``all_reviews`` (same index).
    """

    groups = build_leakage_safe_groups(all_reviews)
    sentences = extract_shade_sentences(target_reviews)
    sentences["p_positive"] = np.nan
    sentences["review_level_p_positive"] = np.nan
    sentences["fold"] = -1
    folds = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(
        folds.split(all_reviews, groups=groups)
    ):
        held_out = set(all_reviews.index[test_idx])
        mask = sentences["review_index"].isin(held_out)
        if not mask.any():
            continue
        train = all_reviews.iloc[train_idx]
        model = build_review_classifier(
            C=C, class_weight=class_weight, random_state=random_state
        )
        model.fit(train[["review_text"]], train["positive_review"])
        sentences.loc[mask, "p_positive"] = model.predict_proba(
            sentences.loc[mask, ["sentence"]].rename(
                columns={"sentence": "review_text"}
            )
        )[:, 1]
        sentences.loc[mask, "review_level_p_positive"] = model.predict_proba(
            sentences.loc[mask, ["review_text"]]
        )[:, 1]
        sentences.loc[mask, "fold"] = fold
    return sentences


# ---------------------------------------------------------------- insight cards
def build_product_insights(
    links: pd.DataFrame,
    reviews_all_rows: pd.DataFrame,
    prepared_reviews: pd.DataFrame,
    sentence_scores: pd.DataFrame,
    tags: pd.DataFrame,
    *,
    min_text_reviews: int = MIN_TEXT_REVIEWS_FOR_EVIDENCE,
) -> pd.DataFrame:
    """One row per catalogue product line with review-derived descriptors.

    ``reviews_all_rows`` is the raw export (for star ratings, with or without
    text); ``prepared_reviews`` is the de-duplicated text set from
    prepare_review_data (for aspect shares).
    """

    aspects = detect_aspects(prepared_reviews["review_text"])
    rows = []
    for link in links.itertuples(index=False):
        ids = list(link.luxxify_product_link_ids)
        row = {
            "brand": link.brand,
            "product": link.product,
            "linked": link.linked,
            "luxxify_products": "; ".join(link.luxxify_product_names),
        }
        ratings = pd.to_numeric(
            reviews_all_rows.loc[
                reviews_all_rows["product_link_id"].isin(ids), "rating"
            ],
            errors="coerce",
        ).dropna()
        text_mask = prepared_reviews["product_link_id"].isin(ids)
        sent = sentence_scores[sentence_scores["product_link_id"].isin(ids)]
        row.update(
            {
                "ratings_count": int(len(ratings)),
                "mean_rating": float(ratings.mean()) if len(ratings) else np.nan,
                "text_reviews": int(text_mask.sum()),
                "shade_sentences": int(len(sent)),
                "shade_sentence_score": float(sent["p_positive"].mean())
                if len(sent)
                else np.nan,
            }
        )
        for aspect in aspects.columns:
            row[f"mentions_{aspect}"] = (
                float(aspects.loc[text_mask, aspect].mean())
                if text_mask.any()
                else np.nan
            )
        tag_rows = tags[tags["product_link_id"].isin(ids)]
        row["shade_sentence_score_se"] = (
            float(sent["p_positive"].std(ddof=1) / np.sqrt(len(sent)))
            if len(sent) > 1
            else np.nan
        )
        for tag in PRODUCT_TAG_PATTERNS:
            row[f"pooled_tag_{tag}"] = (
                bool(tag_rows[tag].any()) if len(tag_rows) else False
            )
        if not link.linked:
            row["evidence"] = "no linked reviews"
        elif row["text_reviews"] < min_text_reviews:
            row["evidence"] = f"low (< {min_text_reviews} text reviews)"
        else:
            row["evidence"] = "ok"
        rows.append(row)
    return pd.DataFrame(rows)


def insight_card_text(insight: Mapping) -> str:
    """Short human-readable card for one product line (display only)."""

    title = f"{insight['brand']} {insight['product']}"
    if not insight["linked"]:
        return f"{title}: no Luxxify reviews linked."
    parts = [
        f"{title} — {insight['ratings_count']:,} ratings, "
        f"mean {insight['mean_rating']:.2f}★, "
        f"{insight['text_reviews']:,} text reviews (evidence: {insight['evidence']})."
    ]
    notable = [
        (k.replace("mentions_", "").replace("_", " "), v)
        for k, v in insight.items()
        if k.startswith("mentions_") and pd.notna(v) and v >= 0.05
    ]
    if notable:
        parts.append(
            "Often mentioned: "
            + ", ".join(
                f"{name} {share:.0%}"
                for name, share in sorted(notable, key=lambda x: -x[1])
            )
            + "."
        )
    if pd.notna(insight.get("shade_sentence_score")) and insight["shade_sentences"]:
        se = insight.get("shade_sentence_score_se")
        spread = f" ± {se:.2f}" if se is not None and pd.notna(se) else ""
        parts.append(
            f"Shade-sentence sentiment {insight['shade_sentence_score']:.2f}{spread} "
            f"from {insight['shade_sentences']} sentences."
        )
    return " ".join(parts)


def attach_review_insights(
    recommendations: pd.DataFrame,
    insights: pd.DataFrame,
    columns: tuple[str, ...] = (
        "mean_rating",
        "ratings_count",
        "text_reviews",
        "mentions_oxidizes_or_orange",
        "mentions_shade_mismatch",
        "shade_sentence_score",
        "evidence",
    ),
) -> pd.DataFrame:
    """Add review descriptors to recommendations without changing their order."""

    keep = ["brand", "product", *[c for c in columns if c in insights.columns]]
    merged = recommendations.merge(
        insights[keep], on=["brand", "product"], how="left", validate="many_to_one"
    )
    merged["evidence"] = merged.get(
        "evidence", pd.Series(index=merged.index, dtype=object)
    ).fillna("no linked reviews")
    assert merged[["brand", "product"]].equals(
        recommendations[["brand", "product"]].reset_index(drop=True)
    )
    return merged


# ------------------------------------------------------- experimental tie-break
def rerank_with_review_tiebreak(
    candidates: pd.DataFrame,
    insights: pd.DataFrame,
    *,
    tolerance: float = 1.0,
    top_n: int = 5,
    min_shade_sentences: int = 10,
    neutral_score: float = NEUTRAL_SCORE,
) -> pd.DataFrame:
    """EXPERIMENTAL: reorder near-tied shades by review signal.

    ``candidates`` must be recommend_foundations output (sorted by
    color_distance), ideally with a larger pool than ``top_n``. Shades are
    chained into tie groups: a group starts at a shade and absorbs following
    shades whose distance is within ``tolerance`` ΔE of the group's first
    shade. Only inside a group are shades re-sorted, by shade-sentence score
    (products with fewer than ``min_shade_sentences`` scored sentences, or
    no linked reviews, get ``neutral_score`` -- pass the mean score of all
    scored shade sentences so "neutral" means "average"), then by distance.
    So no shade can move ahead of one more than ``tolerance`` ΔE closer to the
    skin colour.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    pool = candidates.reset_index(drop=True).copy()
    pool["colour_rank"] = np.arange(1, len(pool) + 1)
    scores = insights.set_index(["brand", "product"])

    def score(row) -> float:
        key = (row["brand"], row["product"])
        if key not in scores.index:
            return neutral_score
        s = scores.loc[key]
        if s["shade_sentences"] < min_shade_sentences or pd.isna(
            s["shade_sentence_score"]
        ):
            return neutral_score
        return float(s["shade_sentence_score"])

    pool["review_score"] = pool.apply(score, axis=1)
    group, start = [], None
    current = -1
    for d in pool["color_distance"]:
        if start is None or d - start > tolerance:
            current += 1
            start = d
        group.append(current)
    pool["tie_group"] = group
    reranked = pool.sort_values(
        ["tie_group", "review_score", "color_distance"],
        ascending=[True, False, True],
        kind="mergesort",
    ).head(top_n)
    reranked = reranked.reset_index(drop=True)
    reranked["moved"] = reranked["colour_rank"] != np.arange(1, len(reranked) + 1)
    return reranked
