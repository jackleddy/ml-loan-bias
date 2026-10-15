import numpy as np
import pandas as pd

from loanfair.analysis import adjust_pvalues, group_summary, residual_regression


def shifted_groups():
    rows = []
    for lender in range(25):
        for group, offset in [("White", 0), ("Asian", 30)]:
            for i in range(10):
                rows.append(dict(row_id=len(rows), lei=f"L{lender}", derived_race=group, actual_spread=offset / 100, predicted_spread=0.0, residual_bp=offset + (lender % 3 - 1), activity_year=2021 + i % 5))
    return pd.DataFrame(rows)


def test_known_gap_bootstrap_and_regression(config):
    data = shifted_groups()
    result = group_summary(data, "derived_race", config)
    asian = result.set_index("group").loc["Asian"]
    assert asian.inference_eligible
    assert np.isclose(asian.difference_from_reference_bp, 30)
    assert np.isclose(asian.ci_low_bp, 30)
    assert np.isclose(asian.ci_high_bp, 30)
    rows = residual_regression(data, "derived_race", config)
    assert np.isclose(rows[0]["adjusted_difference_bp"], 30)


def test_small_cells_and_lender_threshold_are_descriptive(config):
    data = shifted_groups()
    config = dict(config, min_group_size=1000)
    result = group_summary(data, "derived_race", config)
    assert not result.inference_eligible.any()
    assert result.welch_p.isna().all()
    config = dict(config, min_group_size=10, min_lenders=30)
    result = group_summary(data, "derived_race", config)
    assert not result.inference_eligible.any()
    assert result.ci_low_bp.isna().all()


def test_multiple_comparison_adjustment():
    result = adjust_pvalues(pd.DataFrame({"welch_p": [0.01, 0.04, np.nan]}))
    np.testing.assert_allclose(result.welch_q.iloc[:2], [0.02, 0.04])
    assert np.isnan(result.welch_q.iloc[2])
