"""Out-of-fold research predictions, temporal check, sensitivities, final model."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .common import fingerprint, log, write_json
from .models import RAW_FEATURES, make_model, race_weights

SCENARIOS = ["primary", "without_2023", "without_geography", "including_outliers", "race_balanced"]
MODELS = ["linear", "xgboost"]


def fold_assignments(data, config):
    if data.activity_year.value_counts().min() < config["folds"]:
        raise ValueError("Each year needs at least one record per fold.")
    splitter = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    folds = np.full(len(data), -1, dtype=int)
    for fold, (_, validation) in enumerate(splitter.split(data, data.activity_year)):
        folds[validation] = fold
    return folds


def out_of_fold(data, model_name, config, geography=True, balanced=False):
    folds = fold_assignments(data, config)
    predictions = np.full(len(data), np.nan)
    for fold in range(config["folds"]):
        train_mask, validation = folds != fold, folds == fold
        training = data.iloc[np.flatnonzero(train_mask)]
        model = make_model(model_name, config, geography)
        fit_args = {"model__sample_weight": race_weights(training.derived_race, config["weight_cap"])} if balanced else {}
        with threadpool_limits(limits=config["n_jobs"]):
            model.fit(training, training.rate_spread, **fit_args)
            predictions[validation] = model.predict(data.iloc[np.flatnonzero(validation)])
        log(f"  {model_name}: fold {fold + 1}/{config['folds']} ({len(training):,} training loans)")
    if not np.isfinite(predictions).all():
        raise ValueError("Out-of-fold prediction coverage is incomplete.")
    return pd.DataFrame({
        "row_id": data.row_id.to_numpy(), "activity_year": data.activity_year.astype(int).to_numpy(),
        "fold": folds, "actual_spread": data.rate_spread.to_numpy(), "predicted_spread": predictions,
        "residual_bp": (data.rate_spread.to_numpy() - predictions) * 100,
    })


def error_metrics(actual, predicted):
    return {"n": len(actual), "mae_bp": float(mean_absolute_error(actual, predicted) * 100), "rmse_bp": float(np.sqrt(mean_squared_error(actual, predicted)) * 100)}


def metric_rows(data, predictions, scenario, model):
    merged = predictions.merge(data[["row_id", "derived_race", "derived_sex", "derived_ethnicity"]], on="row_id", validate="one_to_one")
    rows = [{"scenario": scenario, "model": model, "dimension": "overall", "group": "All", **error_metrics(merged.actual_spread, merged.predicted_spread)}]
    for dim in ["activity_year", "derived_race", "derived_sex", "derived_ethnicity"]:
        for group, subset in merged.groupby(dim, observed=True):
            rows.append({"scenario": scenario, "model": model, "dimension": dim, "group": str(group), **error_metrics(subset.actual_spread, subset.predicted_spread)})
    return rows


def train(config, scenarios=None):
    scenarios = SCENARIOS if scenarios is None else scenarios
    if "primary" not in scenarios or set(scenarios) - set(SCENARIOS):
        raise ValueError("Training must include primary; scenarios must be recognized.")
    output = Path(config["artifact_dir"])
    output.mkdir(parents=True, exist_ok=True)
    processed = Path(config["processed_dir"])
    primary = pd.read_parquet(processed / "loans.parquet")
    audit = json.loads((processed / "audit.json").read_text())
    if audit["processed_sha256"] != fingerprint(processed / "loans.parquet"):
        raise ValueError("Processed data fingerprint differs from preparation audit; rerun preparation.")
    metrics, primary_predictions = [], {}
    temporal_metrics = {}
    for scenario in scenarios:
        data = pd.read_parquet(processed / "eligible_including_outliers.parquet") if scenario == "including_outliers" else primary
        if scenario == "without_2023":
            data = data[data.activity_year.ne(2023)]
        log(f"Training scenario {scenario}: {len(data):,} loans")
        for name in MODELS:
            predictions = out_of_fold(data, name, config, geography=scenario != "without_geography", balanced=scenario == "race_balanced")
            predictions.to_parquet(output / f"predictions_{scenario}_{name}.parquet", index=False)
            metrics.extend(metric_rows(data, predictions, scenario, name))
            pd.DataFrame(metrics).to_csv(output / "model_metrics.csv", index=False)
            if scenario == "primary":
                primary_predictions[name] = predictions
    primary_metrics = pd.DataFrame(metrics).query("scenario == 'primary' and dimension == 'overall'").sort_values(["mae_bp", "rmse_bp", "model"])
    selected = primary_metrics.iloc[0].model
    history, future = primary[primary.activity_year.lt(2025)], primary[primary.activity_year.eq(2025)]
    if history.empty or future.empty:
        raise ValueError("Temporal check requires pre-2025 and 2025 data.")
    for name in MODELS:
        model = make_model(name, config)
        with threadpool_limits(limits=config["n_jobs"]):
            model.fit(history, history.rate_spread)
            prediction = model.predict(future)
        temporal_metrics[name] = error_metrics(future.rate_spread, prediction)
        pd.DataFrame({"row_id": future.row_id, "actual_spread": future.rate_spread, "predicted_spread": prediction, "residual_bp": (future.rate_spread - prediction) * 100}).to_parquet(output / f"temporal_{name}.parquet", index=False)
    final = make_model(selected, config)
    with threadpool_limits(limits=config["n_jobs"]):
        final.fit(primary, primary.rate_spread)
    joblib.dump(final, output / "selected_model.joblib")
    primary_predictions[selected][["row_id", "activity_year", "residual_bp"]].to_parquet(output / "comparison_residuals.parquet", index=False)
    ranges = {col: {"min": float(primary[col].min()), "max": float(primary[col].max())} for col in ["loan_amount", "income", "combined_loan_to_value_ratio", "latitude", "longitude"]}
    categories = {col: sorted(primary[col].astype(str).unique().tolist()) for col in ["debt_to_income_ratio", "conforming_loan_limit"]}
    metadata = {
        "selected_model": selected, "selection": primary_metrics.to_dict("records"), "temporal_2025": temporal_metrics,
        "raw_features": RAW_FEATURES, "target": "rate_spread", "target_unit": "percentage_points", "residual_unit": "basis_points",
        "income_input_unit": "thousands_of_dollars", "supported_years": sorted(primary.activity_year.astype(int).unique().tolist()),
        "ranges": ranges, "categories": categories, "config": config, "completed_scenarios": scenarios,
        "source_audit": audit, "processed_sha256": fingerprint(processed / "loans.parquet"),
        "model_sha256": fingerprint(output / "selected_model.joblib"),
        "versions": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost", "pyarrow", "statsmodels"]},
        "temporal_note": "2025 is unseen in training; its one-hot year indicators are all zero. This is a separate extrapolation diagnostic.",
        "prediction_note": "Comparison uses a final all-data fit and an out-of-fold residual distribution; percentiles are descriptive, not calibrated probabilities or intervals.",
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "metrics.json", {"selected_model": selected, "cross_validation": metrics, "temporal_2025": temporal_metrics})
    log(f"Selected {selected}; saved model and out-of-fold residuals.")
    return metadata
