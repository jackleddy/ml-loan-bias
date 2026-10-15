import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loanfair.common import DEFAULT_CONFIG, write_json


@pytest.fixture
def config(tmp_path):
    value = json.loads(DEFAULT_CONFIG.read_text())
    for key, name in [("reference_dir", "reference"), ("processed_dir", "processed"), ("artifact_dir", "artifacts"), ("report_dir", "reports")]:
        value[key] = str(tmp_path / name)
        Path(value[key]).mkdir()
    value.update(input=str(tmp_path / "loans.csv.gz"), chunksize=37, folds=2, min_group_size=10, min_lenders=3, bootstrap_replicates=100, n_jobs=1)
    value["xgboost"] = dict(value["xgboost"], n_estimators=8, max_depth=2)
    return value


@pytest.fixture
def tracts():
    return pd.DataFrame([
        {"census_tract": "51001000100", "tract_vintage": 2017, "latitude": 37.1, "longitude": -77.1},
        {"census_tract": "51001000100", "tract_vintage": 2020, "latitude": 37.2, "longitude": -77.2},
    ])


@pytest.fixture
def raw_loans():
    rng = np.random.default_rng(4)
    records = []
    for year in range(2021, 2026):
        for i in range(80):
            race = "White" if i % 2 == 0 else "Black or African American"
            records.append({
                "activity_year": str(year), "lei": f"lender-{i % 10}",
                "derived_msa.md": "12345", "county_code": "51001", "census_tract": "51001000100",
                "conforming_loan_limit": "C" if i % 3 else "NC", "derived_loan_product_type": "Conventional:First Lien",
                "derived_dwelling_category": "Single Family (1-4 Units):Site-Built", "action_taken": "1",
                "loan_amount": str(205000 + i * 1000), "loan_to_value_ratio": str(75 + i % 20),
                "interest_rate": "6.5", "loan_term": "360", "income": str(70 + i),
                "debt_to_income_ratio": "30%-<36%", "rate_spread": str(0.2 + (race != "White") * 0.3 + rng.normal(0, 0.05)),
                "derived_ethnicity": "Not Hispanic or Latino" if i % 3 else "Hispanic or Latino",
                "derived_race": race, "derived_sex": "Male" if i % 4 < 2 else "Female",
                "tract_minority_population_percent": str(i % 100), "tract_to_msa_income_percentage": str(70 + i),
                "preapproval": "2", "submission_of_application": "1", "initially_payable_to_institution": "1",
            })
    return pd.DataFrame(records)


@pytest.fixture
def references(config, tracts):
    directory = Path(config["reference_dir"])
    tracts.to_parquet(directory / "tracts.parquet", index=False)
    dates = pd.date_range("2020-01-06", "2025-12-29", freq="W-MON")
    pd.DataFrame({"effective_date": dates, "amortization": "fixed", "term_years": 30, "apor": 5.0}).to_parquet(directory / "apor.parquet", index=False)
    write_json(directory / "manifest.json", {"fixture": {"url": "synthetic", "sha256": "synthetic"}})
    return directory
