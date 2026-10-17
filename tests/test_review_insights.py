import numpy as np
import pandas as pd
import pytest

from foundation_matcher.review_insights import (
    NEUTRAL_SCORE,
    attach_review_insights,
    build_product_insights,
    detect_aspects,
    extract_shade_sentences,
    insight_card_text,
    link_catalogue_to_luxxify,
    product_tags,
    rerank_with_review_tiebreak,
    score_shade_sentences,
    split_sentences,
    summarize_aspects,
)


def test_detect_aspects_flags_expected_mentions():
    texts = pd.Series(
        [
            "It oxidizes and turns orange by noon.",
            "Perfect match for my skin, great coverage.",
            "Way too dark for me.",
            "Love it.",
        ]
    )
    flags = detect_aspects(texts)
    assert flags.loc[0, "oxidizes_or_orange"]
    assert flags.loc[1, "shade_match_positive"] and flags.loc[1, "coverage"]
    assert flags.loc[2, "shade_mismatch"]
    assert not flags.loc[3].any()


def test_summarize_aspects_reports_rate_difference():
    reviews = pd.DataFrame(
        {
            "review_text": ["it oxidizes", "oxidizes badly", "lovely", "great"],
            "positive_review": [0, 0, 1, 1],
            "rating": [2, 1, 5, 5],
        }
    )
    table = summarize_aspects(reviews).set_index("aspect")
    row = table.loc["oxidizes_or_orange"]
    assert row["reviews_mentioning"] == 2
    assert row["positive_rate_difference"] == pytest.approx(-1.0)


def test_link_requires_brand_and_whole_phrase_and_applies_alias():
    catalogue = pd.DataFrame(
        {
            "brand": ["Maybelline", "Maybelline", "L'Oréal", "MAC"],
            "product": ["Fit Me", "Fit Me", "Infalliable", "Studio Fix"],
        }
    )
    luxxify = pd.DataFrame(
        {
            "product_link_id": ["a", "b", "c", "d"],
            "category": ["Foundation"] * 4,
            "brand": ["MAYBELLINE", "Find your shade", "L'Oréal Paris", "Maybelline"],
            "product_name": [
                "Fit Me Matte + Poreless",
                "Fit Me Dewy",
                "Infallible 24H Fresh Wear",
                "Studio Fix Fluid",
            ],
        }
    )
    links = link_catalogue_to_luxxify(catalogue, luxxify).set_index("product")
    assert links.loc["Fit Me", "luxxify_product_link_ids"] == [
        "a"
    ]  # artefact brand "Find your shade" not guessed
    assert links.loc["Infalliable", "luxxify_product_link_ids"] == [
        "c"
    ]  # alias + brand prefix
    assert not links.loc["Studio Fix", "linked"]  # name matches but brand does not
    assert len(links) == 3  # duplicate catalogue lines collapsed


def test_split_and_extract_shade_sentences():
    assert split_sentences("Great stuff here. Too dark for me sadly! ok") == [
        "Great stuff here.",
        "Too dark for me sadly!",
    ]
    reviews = pd.DataFrame(
        {
            "review_text": ["Lovely texture overall. The shade is way too dark."],
            "x": [1],
        }
    )
    sentences = extract_shade_sentences(reviews)
    assert sentences["sentence"].tolist() == ["The shade is way too dark."]
    assert sentences["x"].tolist() == [1]


def _toy_reviews(n_products=10, per_product=12):
    rows = []
    for p in range(n_products):
        for i in range(per_product):
            good = (i + p) % 3 != 0
            text = (
                f"I love this foundation shade, perfect match number {i} product {p}"
                if good
                else f"Terrible, shade too dark and orange, returned {i} item {p}"
            )
            rows.append(
                {
                    "review_text": text,
                    "positive_review": int(good),
                    "rating": 5 if good else 1,
                    "product_link_id": f"p{p}",
                    "review_id": f"r{p}_{i}",
                }
            )
    return pd.DataFrame(rows)


def test_score_shade_sentences_is_out_of_fold_and_informative():
    reviews = _toy_reviews()
    scores = score_shade_sentences(reviews, reviews, n_splits=5)
    assert scores["p_positive"].notna().all()
    assert set(scores["fold"]) == set(range(5))
    folds_per_product = scores.groupby("product_link_id")["fold"].nunique()
    assert (folds_per_product == 1).all()  # each product is scored by one held-out fold
    assert scores.groupby("positive_review")["p_positive"].mean().diff().iloc[-1] > 0


def _insights():
    return pd.DataFrame(
        {
            "brand": ["A", "B", "C"],
            "product": ["x", "y", "z"],
            "linked": [True, True, False],
            "shade_sentences": [50, 50, 0],
            "shade_sentence_score": [0.2, 0.9, np.nan],
            "mean_rating": [3.0, 4.5, np.nan],
            "ratings_count": [100, 100, 0],
            "text_reviews": [40, 40, 0],
            "mentions_oxidizes_or_orange": [0.1, 0.0, np.nan],
            "mentions_shade_mismatch": [0.05, 0.0, np.nan],
            "evidence": ["ok", "ok", "no linked reviews"],
        }
    )


def _candidates(distances):
    names = [("A", "x"), ("B", "y"), ("C", "z")]
    return pd.DataFrame(
        {
            "brand": [n[0] for n in names[: len(distances)]],
            "product": [n[1] for n in names[: len(distances)]],
            "color_distance": distances,
        }
    )


def test_tiebreak_swaps_only_within_tolerance():
    insights = _insights()
    near = rerank_with_review_tiebreak(
        _candidates([1.0, 1.5, 3.0]), insights, tolerance=1.0, top_n=3
    )
    assert near["brand"].tolist() == [
        "B",
        "A",
        "C",
    ]  # B (0.9) jumps A (0.2), within 0.5 ΔE
    far = rerank_with_review_tiebreak(
        _candidates([1.0, 2.5, 3.0]), insights, tolerance=1.0, top_n=3
    )
    assert far["brand"].tolist() == [
        "A",
        "B",
        "C",
    ]  # gap 1.5 > tolerance: colour order kept
    zero = rerank_with_review_tiebreak(
        _candidates([1.0, 1.5, 3.0]), insights, tolerance=0.0, top_n=3
    )
    assert zero["brand"].tolist() == ["A", "B", "C"]


def test_tiebreak_gives_unscored_products_neutral_score():
    out = rerank_with_review_tiebreak(
        _candidates([1.0, 1.2, 1.4]), _insights(), tolerance=1.0, top_n=3
    )
    assert out.set_index("brand").loc["C", "review_score"] == NEUTRAL_SCORE
    assert out["brand"].tolist() == ["B", "C", "A"]


def test_tiebreak_uses_supplied_neutral_score():
    out = rerank_with_review_tiebreak(
        _candidates([1.0, 1.2, 1.4]),
        _insights(),
        tolerance=1.0,
        top_n=3,
        neutral_score=0.1,
    )
    assert out["brand"].tolist() == ["B", "A", "C"]


def test_tiebreak_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        rerank_with_review_tiebreak(_candidates([1.0]), _insights(), tolerance=-1)


def test_attach_review_insights_preserves_order_and_distances():
    recs = _candidates([2.0, 1.0, 3.0])
    merged = attach_review_insights(recs, _insights())
    assert merged[["brand", "product", "color_distance"]].equals(recs)
    assert merged.loc[2, "evidence"] == "no linked reviews"


def test_build_product_insights_and_card():
    links = pd.DataFrame(
        {
            "brand": ["A", "B"],
            "product": ["x", "y"],
            "linked": [True, False],
            "luxxify_product_link_ids": [["p1"], []],
            "luxxify_product_names": [["A X Foundation"], []],
        }
    )
    raw = pd.DataFrame({"product_link_id": ["p1"] * 3, "rating": [5, 4, 1]})
    prepared = pd.DataFrame(
        {
            "product_link_id": ["p1", "p1"],
            "review_text": ["it oxidizes", "great coverage"],
        }
    )
    sentences = pd.DataFrame({"product_link_id": ["p1"], "p_positive": [0.3]})
    tags = product_tags(
        pd.DataFrame(
            {
                "product_link_id": ["p1"],
                "cons": ["Turns me orange"],
                "best_uses": [None],
            }
        )
    )
    insights = build_product_insights(
        links, raw, prepared, sentences, tags, min_text_reviews=30
    )
    a = insights.set_index("brand").loc["A"]
    assert a["ratings_count"] == 3 and a["mean_rating"] == pytest.approx(10 / 3)
    assert (
        a["mentions_oxidizes_or_orange"] == 0.5
        and a["pooled_tag_con_oxidizes_or_orange"]
    )
    assert a["evidence"].startswith("low")
    assert "no Luxxify reviews" in insight_card_text(insights.iloc[1].to_dict())
    assert "oxidizes or orange 50%" in insight_card_text(insights.iloc[0].to_dict())
