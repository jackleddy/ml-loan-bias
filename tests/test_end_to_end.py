import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from loanfair.analysis import analyze
from loanfair.compare import actual_spread, borrower_frame, compare
from loanfair.prepare import prepare
from loanfair.train import train


def test_full_fixture_pipeline_and_borrower_comparison(config, raw_loans, references):
    raw_loans.loc[0, "rate_spread"] = "100"
    raw_loans.to_csv(config["input"], index=False, compression="gzip")
    primary = prepare(config)
    assert len(primary) == 399
    audit = json.loads((Path(config["processed_dir"]) / "audit.json").read_text())
    assert sum(audit["exclusive_row_counts"].values()) == len(raw_loans)
    assert audit["exclusive_row_counts"]["target_outlier"] == 1
    metadata = train(config)
    assert len(metadata["completed_scenarios"]) == 5
    summary = analyze(config)
    assert summary["primary_rows"] == 399
    assert (Path(config["report_dir"]) / "report.md").exists()
    assert (Path(config["report_dir"]) / "apor_fixed30.png").stat().st_size > 0
    record = dict(year=2024, loan_amount=205000, annual_income=70000, cltv_percent=80, dti=35, conforming_status="C", census_tract="51001000100", actual_rate_spread=0.5)
    result = compare(record, config)
    assert np.isclose(result["residual_bp"], (0.5 - result["predicted_rate_spread"]) * 100)
    assert 0 <= result["historical_residual_percentile"] <= 100
    frame, _ = borrower_frame(record, config, metadata)
    assert frame.income.iloc[0] == 70
    assert frame.debt_to_income_ratio.iloc[0] == "30%-<36%"
    model = joblib.load(Path(config["artifact_dir"]) / "selected_model.joblib")
    assert np.isclose(model.predict(frame)[0], result["predicted_rate_spread"])
    with pytest.raises(ValueError, match="historical years"):
        compare(dict(record, year=2026), config)
    with pytest.raises(ValueError, match="finite"):
        compare(dict(record, annual_income=float("inf")), config)
    with pytest.raises(ValueError, match="finite"):
        borrower_frame(dict(record, annual_income=True), config, metadata)
    result = compare(dict(record, annual_income=100000000), config)
    assert any("income" in warning for warning in result["warnings"])


def test_apr_lookup_requires_explicit_fixed_30_and_units(config, references):
    base = dict(year=2024, apr=7.25, rate_set_date="2024-06-12")
    with pytest.raises(ValueError, match="fixed_rate"):
        actual_spread(base, config)
    spread, details = actual_spread(dict(base, fixed_rate=True, loan_term_months=360), config)
    assert spread == 2.25
    assert details["apor_effective_date"] == "2024-06-10"
    spread, _ = actual_spread(dict(apr=7.25, apor=6.25), config)
    assert spread == 1
    with pytest.raises(ValueError, match="not both"):
        actual_spread(dict(actual_rate_spread=0.5, apr=7.25), config)
