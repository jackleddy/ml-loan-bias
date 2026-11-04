"""Tests for the machine-learning extension studies (loanfair/ml_extensions.py and the five study files)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from loanfair.fairness_tradeoff import ParityPenalizedLinear, calibrated_shift, gap_weights
from loanfair.high_priced import high_priced_label
from loanfair.interpretability import original_feature
from loanfair.ml_extensions import PARTS, run
from loanfair.ml_shared import read_part_summaries
from loanfair.prepare import prepare


def test_high_priced_label_uses_conforming_and_jumbo_cutoffs():
    data = pd.DataFrame({"conforming_loan_limit": ["C", "C", "NC", "NC"], "rate_spread": [1.49, 1.5, 2.49, 2.5]})
    assert high_priced_label(data).tolist() == [0, 1, 0, 1]


def test_gap_weights_give_difference_in_group_means():
    ethnicity = pd.Series(["Hispanic or Latino", "Not Hispanic or Latino", "Not Hispanic or Latino", "Ethnicity Not Available"])
    predictions = np.array([4.0, 1.0, 3.0, 100.0])
    # Hispanic mean 4, non-Hispanic mean 2, unreported borrower ignored.
    assert np.isclose(gap_weights(ethnicity) @ predictions, 2.0)


def test_penalized_linear_matches_ols_at_zero_and_closes_gap_when_strong():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    group = np.where(X[:, 0] > 0, "Hispanic or Latino", "Not Hispanic or Latino")  # group correlates with an input
    y = X @ [1.0, 0.5, -0.2] + rng.normal(0, 0.1, 300)
    weights = gap_weights(group)
    plain = ParityPenalizedLinear(0).fit(X, y, gap_weights=weights)
    np.testing.assert_allclose(plain.predict(X), LinearRegression().fit(X, y).predict(X), atol=1e-8)
    strict = ParityPenalizedLinear(1e6).fit(X, y, gap_weights=weights)
    assert abs(weights @ plain.predict(X)) > 1
    assert abs(weights @ strict.predict(X)) < 1e-3


def test_shifted_targets_reproduce_exact_penalized_solution_for_linear_models():
    # The XGBoost method (ordinary fit on calibrated shifted targets) must land on the
    # same answer as the exact penalized solver when the model is linear.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    group = np.where(X[:, 0] + rng.normal(0, 1, 400) > 0, "Hispanic or Latino", "Not Hispanic or Latino")
    y = X @ [1.0, 0.5, -0.2] + rng.normal(0, 0.1, 400)
    weights = gap_weights(group)
    fit_gap = lambda target: (lambda m: (m, weights @ m.predict(X)))(LinearRegression().fit(X, target))
    _, base_gap = fit_gap(y)
    probe_shift = base_gap / (weights @ weights)
    _, probe_gap = fit_gap(y - probe_shift * weights)
    for strength in [0.1, 1.0, 100.0]:
        shifted, _ = fit_gap(y - calibrated_shift(base_gap, probe_gap, probe_shift, len(y), strength) * weights)
        exact = ParityPenalizedLinear(strength).fit(X, y, gap_weights=weights)
        np.testing.assert_allclose(shifted.predict(X), exact.predict(X), atol=1e-8)


def test_encoded_names_map_back_to_inputs():
    assert original_feature("numeric__log_loan_amount") == "log_loan_amount"
    assert original_feature("categorical__activity_year_2023") == "activity_year"
    assert original_feature("categorical__debt_to_income_ratio_20%-<30%") == "debt_to_income_ratio"
    assert original_feature("categorical__conforming_loan_limit_NC") == "conforming_loan_limit"


def test_all_extension_studies_run_and_index_every_chart(config, raw_loans, references, monkeypatch):
    # Chart every group, however few high-priced loans it has, so small synthetic groups are exercised.
    monkeypatch.setattr("loanfair.high_priced.MIN_POSITIVES", 1)
    # Make some synthetic loans high-priced so the yes/no study has both answers.
    raw_loans.loc[raw_loans.index % 8 == 0, "rate_spread"] = "2.8"
    # "Joint" is a group name under both race and ethnicity; charts must keep them apart.
    raw_loans.loc[raw_loans.index % 7 == 0, ["derived_race", "derived_ethnicity"]] = "Joint"
    raw_loans.to_csv(config["input"], index=False, compression="gzip")
    prepare(config)
    config["ml_extensions"] = {
        "bias_variance_folds": 2, "depths": [1, 2], "tree_counts": [2, 4], "learning_curve_sizes": [50, 100],
        "ridge_alphas": [0.1, 10.0], "ridge_small_sample": 50, "interpret_sample": 200, "pdp_sample": 50,
        "permutation_repeats": 2, "fairness_lambdas": [0, 10],
    }
    run(config)
    output = Path(config["report_dir"]) / "ml_extensions"
    summaries = read_part_summaries(config)
    assert {s["part"] for s in summaries} == set(PARTS)
    index = (output / "README.md").read_text(encoding="utf-8")
    for summary in summaries:
        for figure in summary["figures"]:
            assert (output / figure).stat().st_size > 0, figure
            assert figure in index
