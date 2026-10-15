"""Compare one historical borrower record against expected rate spread."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import fingerprint
from .prepare import DTI_CATEGORIES
from .references import load_tracts, lookup_apor


def finite_number(record, name):
    value = record.get(name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def borrower_frame(record, config, metadata):
    year_value = finite_number(record, "year")
    if not year_value.is_integer() or int(year_value) not in metadata["supported_years"]:
        raise ValueError("Supported historical years are 2021-2025, subject to training coverage.")
    year = int(year_value)
    if "loan_term_months" in record and record["loan_term_months"] != 360:
        raise ValueError("This comparison supports only 360-month loans.")
    loan = finite_number(record, "loan_amount")
    cltv = finite_number(record, "cltv_percent")
    income = finite_number(record, "annual_income") / 1000  # Public HMDA income is in thousands.
    if loan <= 0 or cltv <= 0:
        raise ValueError("Loan amount and CLTV percentage must be positive.")
    conforming = record.get("conforming_status")
    if conforming not in {"C", "NC", "U", "Missing"}:
        raise ValueError("conforming_status must be C, NC, U, or Missing.")
    dti_value = record.get("dti")
    if isinstance(dti_value, bool):
        raise ValueError("dti must be a percentage or public HMDA category.")
    # Convert a borrower's exact DTI to the same public-data bucket used in training.
    try:
        dti_number = float(dti_value)
    except (ValueError, TypeError):
        dti = str(dti_value)
    else:
        if not np.isfinite(dti_number) or dti_number < 0:
            raise ValueError("Numeric DTI must be finite and nonnegative.")
        if dti_number < 20:
            dti = "<20%"
        elif dti_number < 30:
            dti = "20%-<30%"
        elif dti_number < 36:
            dti = "30%-<36%"
        elif dti_number < 50:
            dti = str(int(dti_number))
        elif dti_number <= 60:
            dti = "50%-60%"
        else:
            dti = ">60%"
    if dti not in DTI_CATEGORIES:
        raise ValueError("dti must be an exact percentage or a documented public HMDA DTI category.")
    has_coords = "latitude" in record or "longitude" in record
    if has_coords and "census_tract" in record:
        raise ValueError("Provide coordinates or census_tract, not both.")
    if has_coords:
        lat, lon = finite_number(record, "latitude"), finite_number(record, "longitude")
    elif "census_tract" in record:
        tract = str(record["census_tract"])
        tracts = load_tracts(config)
        match = tracts[tracts.census_tract.eq(tract) & tracts.tract_vintage.eq(2017 if year == 2021 else 2020)]
        if len(match) != 1:
            raise ValueError("Unknown Virginia census tract for the requested year's geography.")
        lat, lon = float(match.iloc[0].latitude), float(match.iloc[0].longitude)
    else:
        raise ValueError("Provide latitude/longitude or a census_tract identifier.")
    if not (36 <= lat <= 40.1 and -84 <= lon <= -75):
        raise ValueError("Coordinates must fall within Virginia's bounding box; only Virginia loans are supported.")
    data = pd.DataFrame([{"activity_year": year, "loan_amount": loan, "income": income, "combined_loan_to_value_ratio": cltv, "debt_to_income_ratio": dti, "conforming_loan_limit": conforming, "latitude": lat, "longitude": lon}])
    warnings = []
    for column, limits in metadata["ranges"].items():
        if not limits["min"] <= float(data.iloc[0][column]) <= limits["max"]:
            warnings.append(f"{column} is outside the training range.")
    for column, categories in metadata["categories"].items():
        if str(data.iloc[0][column]) not in categories:
            warnings.append(f"{column} category was not observed during training.")
    return data, warnings


def actual_spread(record, config):
    if "actual_rate_spread" in record:
        if any(k in record for k in ["apr", "apor", "rate_set_date"]):
            raise ValueError("Provide actual_rate_spread or APR benchmark inputs, not both.")
        return finite_number(record, "actual_rate_spread"), {"spread_source": "reported"}
    if "apr" not in record:
        if any(k in record for k in ["apor", "rate_set_date"]):
            raise ValueError("APR is required with APOR or rate_set_date.")
        return None, {}
    apr = finite_number(record, "apr")
    if apr < 0:
        raise ValueError("APR cannot be negative.")
    if "apor" in record:
        if "rate_set_date" in record:
            raise ValueError("Provide explicit APOR or rate_set_date for lookup, not both.")
        benchmark = finite_number(record, "apor")
        if benchmark < 0:
            raise ValueError("APOR cannot be negative.")
        return apr - benchmark, {"spread_source": "apr_minus_supplied_apor", "apor": benchmark}
    if record.get("fixed_rate") is not True or record.get("loan_term_months") != 360 or "rate_set_date" not in record:
        raise ValueError("Automatic APOR lookup requires fixed_rate: true, loan_term_months: 360, and rate_set_date.")
    try:
        date = pd.Timestamp(record["rate_set_date"])
    except (ValueError, TypeError) as exc:
        raise ValueError("rate_set_date must be a valid historical date.") from exc
    if pd.isna(date) or date.tzinfo is not None or not pd.Timestamp("2020-01-01") <= date <= pd.Timestamp("2025-12-31"):
        raise ValueError("Automatic APOR lookup supports rate-set dates from 2020 through 2025.")
    if date.year > int(record["year"]):
        raise ValueError("rate_set_date cannot follow the loan's reported year.")
    table = pd.read_parquet(Path(config["reference_dir"]) / "apor.parquet")
    benchmark, effective = lookup_apor(table, date)
    return apr - benchmark, {"spread_source": "apr_minus_fixed30_apor", "apor": benchmark, "apor_effective_date": effective}


def compare(record, config):
    artifact = Path(config["artifact_dir"])
    metadata = json.loads((artifact / "metadata.json").read_text())
    model_path = artifact / "selected_model.joblib"
    if fingerprint(model_path) != metadata["model_sha256"]:
        raise ValueError("Saved model fingerprint differs from metadata; rerun training.")
    data, warnings = borrower_frame(record, config, metadata)
    actual, detail = actual_spread(record, config)
    # Only load locally generated model artifacts, never untrusted joblib files.
    model = joblib.load(model_path)
    predicted = float(model.predict(data)[0])
    result = {
        "model": metadata["selected_model"], "year": int(data.iloc[0].activity_year),
        "predicted_rate_spread": predicted, "spread_unit": "percentage_points",
        "warnings": warnings, **detail,
        "interpretation": "Historical pricing comparison, not proof of discrimination or a guaranteed fair price. Residual percentiles are descriptive, not calibrated probabilities.",
    }
    if actual is not None:
        residual = (actual - predicted) * 100
        residuals = pd.read_parquet(artifact / "comparison_residuals.parquet")
        reference = residuals[residuals.activity_year.eq(result["year"])].residual_bp.to_numpy()
        if not len(reference):
            raise ValueError("No out-of-fold residual distribution for this year.")
        percentile = float(100 * (np.sum(reference < residual) + 0.5 * np.sum(reference == residual)) / len(reference))
        result.update(actual_rate_spread=actual, residual_bp=residual, historical_residual_percentile=percentile, comparison_loans=len(reference))
        if not config["spread_min"] <= actual <= config["spread_max"]:
            warnings.append("Actual spread is outside the primary study's target bounds.")
    return result
