import numpy as np
import pandas as pd

from loanfair.models import build_features, make_model, race_weights
from loanfair.prepare import classify_rows
from loanfair.train import out_of_fold


def test_allowlist_excludes_demographics_pricing_and_lender(raw_loans, config, tracts):
    raw_loans["row_id"] = range(len(raw_loans))
    data = classify_rows(raw_loans, config, tracts)
    original = build_features(data)
    changed = data.copy()
    for col in ["derived_race", "derived_sex", "derived_ethnicity", "lei", "interest_rate", "rate_spread", "tract_minority_population_percent", "tract_to_msa_income_percentage"]:
        changed[col] = "POISON"
    pd.testing.assert_frame_equal(original, build_features(changed))
    assert "latitude" not in build_features(data, geography=False)


def test_weights_depend_on_training_subset_only():
    race = pd.Series(["White"] * 90 + ["Asian"] * 9 + ["Joint"] + ["Race Not Available"])
    weights = race_weights(race, cap=10)
    assert weights.max() <= 10
    assert weights[-1] == 1
    assert weights[-2] == 10
    assert np.isclose(weights[0], 100 / (3 * 90))


def test_oof_isolation_and_training_only_preprocessing(raw_loans, config, tracts, monkeypatch):
    raw_loans["row_id"] = range(len(raw_loans))
    data = classify_rows(raw_loans, config, tracts)
    seen = []
    real_factory = make_model
    def factory(*args, **kwargs):
        model = real_factory(*args, **kwargs)
        fit = model.fit
        predict = model.predict
        def checked_fit(frame, target, **fit_args):
            model._fit_ids = set(frame.row_id)
            result = fit(frame, target, **fit_args)
            expected = build_features(frame)[["log_loan_amount", "signed_log_income", "combined_loan_to_value_ratio", "latitude", "longitude"]].mean().to_numpy()
            np.testing.assert_allclose(model.named_steps["preprocessing"].named_transformers_["numeric"].mean_, expected)
            return result
        def checked_predict(frame, **predict_args):
            assert not model._fit_ids.intersection(frame.row_id)
            seen.extend(frame.row_id.tolist())
            return predict(frame, **predict_args)
        model.fit, model.predict = checked_fit, checked_predict
        return model
    monkeypatch.setattr("loanfair.train.make_model", factory)
    predictions = out_of_fold(data, "linear", config)
    assert sorted(seen) == sorted(data.row_id)
    assert predictions.row_id.is_unique
    np.testing.assert_allclose(predictions.residual_bp, (predictions.actual_spread - predictions.predicted_spread) * 100)


def test_balanced_weights_receive_only_training_records(raw_loans, config, tracts, monkeypatch):
    from loanfair.train import fold_assignments
    raw_loans["row_id"] = range(len(raw_loans))
    data = classify_rows(raw_loans, config, tracts)
    folds = fold_assignments(data, config)
    expected = [data.loc[folds != fold, "derived_race"] for fold in range(config["folds"])]
    calls = []
    def checked_weights(race, cap):
        pd.testing.assert_series_equal(race, expected[len(calls)])
        calls.append(len(race))
        return race_weights(race, cap)
    monkeypatch.setattr("loanfair.train.race_weights", checked_weights)
    out_of_fold(data, "linear", config, balanced=True)
    assert len(calls) == config["folds"]
