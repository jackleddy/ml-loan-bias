"""Allowlisted feature pipeline shared by training and borrower comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

RAW_FEATURES = ["activity_year", "loan_amount", "income", "combined_loan_to_value_ratio", "debt_to_income_ratio", "conforming_loan_limit", "latitude", "longitude"]
NUMERIC_FEATURES = ["log_loan_amount", "signed_log_income", "combined_loan_to_value_ratio"]
CATEGORICAL_FEATURES = ["activity_year", "debt_to_income_ratio", "conforming_loan_limit"]


def build_features(data, geography=True):
    required = RAW_FEATURES if geography else RAW_FEATURES[:-2]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing model inputs: {sorted(missing)}")
    # Explicit construction prevents incidental columns (including targets) entering training.
    result = pd.DataFrame(index=data.index)
    result["log_loan_amount"] = np.log(data.loan_amount.astype(float))
    income = data.income.astype(float)
    result["signed_log_income"] = np.sign(income) * np.log1p(np.abs(income))
    result["combined_loan_to_value_ratio"] = data.combined_loan_to_value_ratio.astype(float)
    result["activity_year"] = data.activity_year.astype(int).astype(str)
    for col in CATEGORICAL_FEATURES[1:]:
        result[col] = data[col].astype(str)
    if geography:
        result["latitude"] = data.latitude.astype(float)
        result["longitude"] = data.longitude.astype(float)
    numeric = result.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Model inputs must be finite; loan amount must be positive.")
    return result


def make_model(name, config, geography=True):
    numeric = NUMERIC_FEATURES + (["latitude", "longitude"] if geography else [])
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler() if name == "linear" else "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ], remainder="drop")
    if name == "linear":
        estimator = LinearRegression(n_jobs=config["n_jobs"])
    elif name == "xgboost":
        estimator = XGBRegressor(**config["xgboost"], objective="reg:squarederror", tree_method="hist", device="cpu", random_state=config["seed"], n_jobs=config["n_jobs"])
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("features", FunctionTransformer(build_features, kw_args={"geography": geography}, validate=False)),
        ("preprocessing", preprocessing), ("model", estimator),
    ])


def race_weights(race, cap=10.0):
    """Equal total weight per reported known-race category before capping; training only."""
    known = ~race.isin(["Race Not Available", "Free Form Text Only"])
    counts = race[known].value_counts()
    weights = np.ones(len(race), dtype=float)
    if len(counts):
        mapping = len(race[known]) / (len(counts) * counts)
        weights[known.to_numpy()] = race[known].map(mapping).to_numpy(dtype=float)
    return np.minimum(weights, cap)
