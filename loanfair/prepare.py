"""Filter the supplied loan file and reconcile every input row."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import fingerprint, log, write_json
from .references import load_tracts

ALIASES = {"derived_msa.md": "derived_msa_md", "derived_msa-md": "derived_msa_md", "loan_to_value_ratio": "combined_loan_to_value_ratio"}
NUMERIC = ["activity_year", "action_taken", "loan_term", "loan_amount", "income", "combined_loan_to_value_ratio", "rate_spread", "interest_rate", "tract_minority_population_percent", "tract_to_msa_income_percentage"]
REQUIRED = ["activity_year", "lei", "county_code", "census_tract", "derived_msa_md", "derived_loan_product_type", "derived_dwelling_category", "action_taken", "loan_term", "loan_amount", "income", "combined_loan_to_value_ratio", "debt_to_income_ratio", "conforming_loan_limit", "rate_spread", "derived_race", "derived_sex", "derived_ethnicity"]
DTI_CATEGORIES = ["<20%", "20%-<30%", "30%-<36%"] + [str(n) for n in range(36, 50)] + ["50%-60%", ">60%", "Missing"]
MISSING_TEXT = {"", "NA", "N/A", "Exempt", "exempt", "nan", "None"}


def normalize(data):
    data = data.rename(columns=ALIASES).copy()
    if data.columns.duplicated().any():
        raise ValueError("Input has ambiguous duplicate aliases.")
    missing = sorted(set(REQUIRED) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(missing)}")
    for col in data.select_dtypes(include=["object", "str"]).columns:
        data[col] = data[col].astype(str).str.strip()
    for col in NUMERIC:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col].replace(list(MISSING_TEXT), np.nan), errors="coerce")
        # HMDA exemption is field-specific. Income/loan amounts can legitimately be 1111/8888/9999.
        if col in {"combined_loan_to_value_ratio", "loan_term"}:
            data.loc[data[col].eq(1111), col] = np.nan
        data.loc[~np.isfinite(data[col]), col] = np.nan
    data["debt_to_income_ratio"] = data.debt_to_income_ratio.replace(list(MISSING_TEXT) + ["1111"], "Missing")
    unknown = set(data.debt_to_income_ratio) - set(DTI_CATEGORIES)
    if unknown:
        raise ValueError(f"Unrecognized public HMDA DTI categories: {sorted(unknown)}")
    data["conforming_loan_limit"] = data.conforming_loan_limit.replace(list(MISSING_TEXT), "Missing")
    if not data.conforming_loan_limit.isin(["C", "NC", "U", "Missing"]).all():
        raise ValueError("Unrecognized conforming-loan-limit category.")
    for col, label in [("derived_race", "Race Not Available"), ("derived_sex", "Sex Not Available"), ("derived_ethnicity", "Ethnicity Not Available")]:
        data[col] = data[col].replace(list(MISSING_TEXT), label)
    data["lei"] = data.lei.replace(list(MISSING_TEXT), "Lender Not Available")
    return data


def classify_rows(data, config, tracts):
    """Reasons are sequential and mutually exclusive, so all rows reconcile."""
    data = normalize(data)
    reason = pd.Series("", index=data.index, dtype="str")
    checks = [
        ("unsupported_year", data.activity_year.isin(config["years"])),
        ("outside_virginia", data.county_code.str.match(r"^51\d{3}$") & data.census_tract.str.match(r"^51\d{9}$")),
        ("not_originated", data.action_taken.eq(1)),
        ("not_site_built_1_to_4_units", data.derived_dwelling_category.eq("Single Family (1-4 Units):Site-Built")),
        ("not_conventional_first_lien", data.derived_loan_product_type.eq("Conventional:First Lien")),
        ("not_360_month_term", data.loan_term.eq(360)),
    ]
    for name, valid in checks:
        reason.loc[reason.eq("") & ~valid] = name
    data["cohort_eligible"] = reason.eq("")
    data["tract_vintage"] = np.where(data.activity_year.eq(2021), 2017, 2020)
    data = data.merge(tracts, on=["census_tract", "tract_vintage"], how="left", validate="many_to_one", sort=False)
    # merge produces a fresh RangeIndex; reasons follow original row order.
    reason = pd.Series(reason.to_numpy(), index=data.index, dtype="str")
    valid_target = data.rate_spread.notna()
    bad_features = data.loan_amount.isna() | data.loan_amount.le(0) | data.income.isna() | data.combined_loan_to_value_ratio.isna() | data.combined_loan_to_value_ratio.le(0)
    data["target_outlier"] = valid_target & ~data.rate_spread.between(config["spread_min"], config["spread_max"])
    for name, invalid in [("invalid_target", ~valid_target), ("invalid_financial_features", bad_features), ("unmatched_tract", data.latitude.isna() | data.longitude.isna())]:
        reason.loc[reason.eq("") & invalid] = name
    data["eligible_including_outliers"] = reason.eq("")
    reason.loc[reason.eq("") & data.target_outlier] = "target_outlier"
    data["exclusion_reason"] = reason
    return data


def prepare(config):
    tracts = load_tracts(config)
    output = Path(config["processed_dir"])
    output.mkdir(parents=True, exist_ok=True)
    counts, cohort_parts, exclusions, coverage_parts = Counter(), [], [], []
    offset = 0
    for raw in pd.read_csv(config["input"], compression="infer", dtype=str, keep_default_na=False, chunksize=config["chunksize"]):
        raw["row_id"] = np.arange(offset, offset + len(raw), dtype=np.int64)
        offset += len(raw)
        data = classify_rows(raw, config, tracts)
        counts.update(data.exclusion_reason.replace("", "included"))
        coverage_parts.append(data.groupby("activity_year", dropna=False).agg(input_rows=("row_id", "size"), input_lenders=("lei", lambda s: set(s)), cohort_rows=("cohort_eligible", "sum"), primary_rows=("exclusion_reason", lambda s: s.eq("").sum())).reset_index())
        cohort_parts.append(data[data.eligible_including_outliers])
        exclusions.append(data.loc[data.exclusion_reason.ne(""), ["row_id", "activity_year", "lei", "census_tract", "rate_spread", "derived_race", "derived_sex", "derived_ethnicity", "exclusion_reason"]])
        log(f"Prepared {offset:,} input rows")
    if not cohort_parts:
        raise ValueError("Input file has no records.")
    cohort = pd.concat(cohort_parts, ignore_index=True)
    if cohort.empty or not cohort.exclusion_reason.eq("").any():
        raise ValueError("No eligible primary loans remain; inspect filters and reference coverage.")
    primary = cohort[cohort.exclusion_reason.eq("")].copy()
    cohort.to_parquet(output / "eligible_including_outliers.parquet", index=False)
    primary.to_parquet(output / "loans.parquet", index=False)
    pd.concat(exclusions, ignore_index=True).to_csv(output / "exclusions.csv", index=False)
    coverage = pd.concat(coverage_parts).groupby("activity_year", dropna=False).agg(input_rows=("input_rows", "sum"), input_lenders=("input_lenders", lambda sets: len(set().union(*sets))), cohort_rows=("cohort_rows", "sum"), primary_rows=("primary_rows", "sum")).reset_index()
    coverage.to_csv(output / "coverage.csv", index=False)
    lender_counts = primary.groupby("activity_year").lei.nunique().to_dict()
    manifest = Path(config["reference_dir"]) / "manifest.json"
    audit = {
        "input": config["input"], "input_sha256": fingerprint(config["input"]),
        "input_rows": offset, "cohort_rows_before_cleaning": int(coverage.cohort_rows.sum()),
        "primary_rows": len(primary), "eligible_including_outliers": len(cohort),
        "exclusive_row_counts": dict(counts), "primary_lenders_by_year": {str(int(k)): int(v) for k, v in lender_counts.items()},
        "spread_bounds": [config["spread_min"], config["spread_max"]],
        "processed_sha256": fingerprint(output / "loans.parquet"),
        "references": json.loads(manifest.read_text()) if manifest.exists() else {},
        "limitations": ["Prior cleaning exclusions cannot be reconstructed.", "Occupancy, fixed-rate status, purpose, exactly one unit, credit score, and precise timing are unavailable.", "Input coverage is not verified against the full HMDA population; 2023 is unusually sparse."],
    }
    assert sum(counts.values()) == offset
    write_json(output / "audit.json", audit)
    log(f"Primary cohort: {len(primary):,}; reconciled {offset:,} input rows.")
    return primary
