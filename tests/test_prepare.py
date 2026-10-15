import numpy as np
import pytest

from loanfair.prepare import classify_rows, normalize


def test_cohort_filters_and_reconciliation(raw_loans, config, tracts):
    raw = raw_loans.iloc[:9].copy()
    raw["row_id"] = range(len(raw))
    raw.loc[1, "action_taken"] = "2"
    raw.loc[2, "derived_loan_product_type"] = "FHA:First Lien"
    raw.loc[3, "loan_term"] = "180"
    raw.loc[4, "derived_dwelling_category"] = "Multifamily:Site-Built"
    raw.loc[5, "rate_spread"] = "100"
    raw.loc[6, "rate_spread"] = "NA"
    raw.loc[7, "census_tract"] = "51001999999"
    raw.loc[8, "county_code"] = "06001"
    data = classify_rows(raw, config, tracts)
    assert data.exclusion_reason.tolist() == ["", "not_originated", "not_conventional_first_lien", "not_360_month_term", "not_site_built_1_to_4_units", "target_outlier", "invalid_target", "unmatched_tract", "outside_virginia"]
    assert data.eligible_including_outliers.sum() == 2
    assert data.cohort_eligible.sum() == 4


def test_aliases_missing_codes_and_negative_income(raw_loans):
    raw = raw_loans.iloc[:3].copy()
    raw.loc[0, "income"] = "-5996"
    raw.loc[1, "income"] = "1111"
    raw.loc[2, "income"] = "9999"
    raw.loc[0, "loan_to_value_ratio"] = "1111"
    raw.loc[0, "debt_to_income_ratio"] = "1111"
    raw.loc[1, "loan_amount"] = "8888"
    raw.loc[2, "derived_race"] = "Race Not Available"
    data = normalize(raw)
    assert data.income.tolist() == [-5996, 1111, 9999]
    assert np.isnan(data.loc[0, "combined_loan_to_value_ratio"])
    assert data.loc[0, "debt_to_income_ratio"] == "Missing"
    assert data.loc[1, "loan_amount"] == 8888
    assert data.loc[2, "derived_race"] == "Race Not Available"
    assert "derived_msa_md" in data
    assert data.census_tract.iloc[0] == "51001000100"


def test_vintage_is_selected_by_loan_year(raw_loans, config, tracts):
    raw = raw_loans.iloc[[0, 80]].copy()
    raw["row_id"] = [0, 80]
    data = classify_rows(raw, config, tracts)
    assert data.tract_vintage.tolist() == [2017, 2020]
    assert data.latitude.tolist() == [37.1, 37.2]


def test_invalid_dti_and_ambiguous_alias_rejected(raw_loans):
    raw = raw_loans.iloc[:1].copy()
    raw.loc[0, "debt_to_income_ratio"] = "garbage"
    with pytest.raises(ValueError, match="DTI"):
        normalize(raw)
    raw = raw_loans.iloc[:1].copy()
    raw["combined_loan_to_value_ratio"] = "80"
    with pytest.raises(ValueError, match="duplicate"):
        normalize(raw)
