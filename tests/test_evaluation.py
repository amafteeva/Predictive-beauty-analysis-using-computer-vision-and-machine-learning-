import numpy as np
import pandas as pd
import pytest

from foundation_matcher.evaluation import evaluate_group_completion_disparity


def _evaluation_frame(group_a_successes: int, group_b_successes: int, per_group: int = 20):
    groups = ["A"] * per_group + ["B"] * per_group
    success = (
        [True] * group_a_successes
        + [False] * (per_group - group_a_successes)
        + [True] * group_b_successes
        + [False] * (per_group - group_b_successes)
    )
    return pd.DataFrame({"group": groups, "pipeline_success": success})


def test_identical_group_rates_have_zero_spread_and_high_p_value():
    evaluation = _evaluation_frame(group_a_successes=18, group_b_successes=18)

    result = evaluate_group_completion_disparity(
        evaluation, number_of_permutations=500, random_state=1
    )

    assert result["observed_completion_rate_spread"] == 0.0
    assert result["permutation_p_value"] == 1.0


def test_extreme_group_gap_is_detected_as_significant():
    # One group always succeeds, the other always fails: as extreme a gap
    # as this sample size can produce, so it should never occur by chance.
    evaluation = _evaluation_frame(group_a_successes=20, group_b_successes=0)

    result = evaluate_group_completion_disparity(
        evaluation, number_of_permutations=2000, random_state=1
    )

    assert result["observed_completion_rate_spread"] == 1.0
    assert result["permutation_p_value"] < 0.01


def test_rejects_invalid_permutation_count():
    evaluation = _evaluation_frame(group_a_successes=18, group_b_successes=16)
    with pytest.raises(ValueError, match="number_of_permutations"):
        evaluate_group_completion_disparity(evaluation, number_of_permutations=0)
