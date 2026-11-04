"""
Extension study 5: Interpretability -- what does the model pay attention to?
============================================================================

WHAT THIS STUDY DOES (in plain language)
    Models can feel like black boxes. This study opens them up with three standard
    tools, so the report can show the model's reasoning and not just its score.

    1. SHAP values (XGBoost only)
       For every single loan, SHAP splits the model's prediction into pieces: "the
       average loan is predicted at X; this loan's high loan-to-value added +12 bp,
       its income subtracted -3 bp, its location added +5 bp, ..." The pieces always
       add up exactly to the prediction. Averaging the SIZE of each piece over many
       loans shows which inputs matter most overall. XGBoost computes these exactly
       for tree models, so no extra library is needed.

    2. Partial dependence (both models)
       "If we took every loan and changed only its loan-to-value (or income, or loan
       amount), how would the average prediction move?" This draws the shape of each
       relationship, e.g. whether pricing rises smoothly or jumps at a threshold.

    3. Permutation importance (both models)
       Shuffle one input across loans (scrambling its link to the price) and measure
       how much worse the predictions get. Big damage = the model relied on that input.

    We also compare average SHAP pieces for Hispanic or Latino vs. non-Hispanic
    borrowers. This shows WHICH inputs push the two groups' predicted prices apart,
    which connects to the proxy test (proxy_test.py).

    All explanations use held-out loans the model did not train on (one of the main
    study's five folds), with models trained on the other four folds.

VISUALS PRODUCED (saved in reports/ml_extensions/)
    interpret_shap_importance.png        Average impact of each input (XGBoost, SHAP)
    interpret_shap_beeswarm.png          Every loan's SHAP value per input, colored by input value
    interpret_shap_by_ethnicity.png      Average SHAP piece per input, Hispanic vs. non-Hispanic
    interpret_partial_dependence.png     Prediction vs. loan-to-value, income, and loan amount
    interpret_permutation_importance.png How much error rises when each input is scrambled

TABLES PRODUCED
    interpret_shap_importance.csv        Average SHAP impact per input
    interpret_shap_by_ethnicity.csv      Average SHAP piece per input for each ethnicity group
    interpret_partial_dependence.csv     The partial dependence curves' data
    interpret_permutation_importance.csv Permutation importance for both models
    interpret_summary.json               Key numbers, used to build the folder's README.md
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from .common import log
from .ml_shared import FRIENDLY_NAMES, MODEL_COLORS, MODEL_LABELS, legend_below, load_primary, output_dir, save_figure, settings, write_part_summary
from .models import CATEGORICAL_FEATURES, RAW_FEATURES, make_model
from .train import fold_assignments

PART = "interpret"

# Inputs drawn in the partial dependence chart, with axis labels and units.
PDP_FEATURES = [
    ("combined_loan_to_value_ratio", "Combined loan-to-value (%)"),
    ("income", "Annual income ($ thousands)"),
    ("loan_amount", "Loan amount ($)"),
]

# Numeric raw columns used to color the beeswarm dots (high value = yellow, low = purple).
RAW_VALUE_FOR = {
    "log_loan_amount": "loan_amount", "signed_log_income": "income",
    "combined_loan_to_value_ratio": "combined_loan_to_value_ratio",
    "latitude": "latitude", "longitude": "longitude",
}


def original_feature(encoded_name):
    """
    Map an encoded column name back to the input it came from. The preprocessing step
    renames columns, e.g. "numeric__log_loan_amount" or "categorical__activity_year_2023"
    (one yes/no column per year). Both map back to one input: "log_loan_amount", "activity_year".
    """
    kind, _, rest = encoded_name.partition("__")
    if kind == "categorical":
        for column in CATEGORICAL_FEATURES:
            if rest.startswith(column + "_"):
                return column
    return rest


def shap_values(model, loans):
    """
    Return a table of SHAP values in basis points: one row per loan, one column per
    original input. Yes/no columns from the same input (e.g. every year) are added
    together so each input gets a single number per loan.
    """
    encoded = np.asarray(model[:-1].transform(loans), dtype=float)  # every pipeline step except the model
    names = model.named_steps["preprocessing"].get_feature_names_out()
    booster = model.named_steps["model"].get_booster()
    # pred_contribs=True asks XGBoost for exact SHAP values. The last column is the
    # "bias" (the average prediction), which we drop.
    contributions = booster.predict(xgb.DMatrix(encoded), pred_contribs=True)[:, :-1]
    table = pd.DataFrame(contributions * 100, columns=[original_feature(n) for n in names], index=loans.index)
    return table.T.groupby(level=0).sum().T


def partial_dependence(model, loans, column, points=20):
    """
    Manual partial dependence: set `column` to the same value for every loan, predict,
    and average. Repeat across a grid of values from the 2nd to the 98th percentile
    (skipping extreme values where there are too few loans to trust the curve).
    """
    grid = np.unique(np.quantile(loans[column], np.linspace(0.02, 0.98, points)))
    averages = []
    for value in grid:
        changed = loans.copy()
        changed[column] = value
        averages.append(float(model.predict(changed).mean() * 100))  # percentage points -> bp
    return grid, np.array(averages)


def run(config):
    """Fit both models, explain them, save the charts and tables, and return the key numbers."""
    s = settings(config)
    output = output_dir(config)
    data = load_primary(config)

    # Train on four folds and explain loans from the fifth (held-out) fold.
    folds = fold_assignments(data, config)
    training, holdout = data[folds != 0], data[folds == 0]
    sample = holdout.sample(min(s["interpret_sample"], len(holdout)), random_state=config["seed"])
    log(f"Interpretability: training on {len(training):,} loans, explaining {len(sample):,} held-out loans")
    models = {}
    for name in ["linear", "xgboost"]:
        models[name] = make_model(name, config)
        with threadpool_limits(limits=config["n_jobs"]):
            models[name].fit(training, training.rate_spread)

    # ---- SHAP values ----------------------------------------------------------------
    shap = shap_values(models["xgboost"], sample)
    importance = shap.abs().mean().sort_values()
    importance.rename("mean_abs_shap_bp").rename_axis("feature").to_csv(output / f"{PART}_shap_importance.csv")

    # Chart 1: average impact of each input.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([FRIENDLY_NAMES.get(f, f) for f in importance.index], importance.values, color=MODEL_COLORS["xgboost"])
    ax.set(title="Which inputs move XGBoost's price predictions most? (SHAP)", xlabel="Average size of effect on prediction (basis points)")
    save_figure(fig, output / "interpret_shap_importance.png")

    # Chart 2: beeswarm-style strip plot. Each dot is one loan; its horizontal position
    # is how much that input pushed the loan's predicted price up (right) or down (left).
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rng = np.random.default_rng(config["seed"])
    for row, feature in enumerate(importance.index):
        jitter = rng.uniform(-0.3, 0.3, len(shap))
        raw = RAW_VALUE_FOR.get(feature)
        if raw:
            # Color by where the loan's value ranks among all loans (0 = lowest, 1 = highest).
            color = sample[raw].rank(pct=True).to_numpy()
            points = ax.scatter(shap[feature], row + jitter, c=color, cmap="viridis", s=2, alpha=0.5, rasterized=True)
        else:
            ax.scatter(shap[feature], row + jitter, color="gray", s=2, alpha=0.3, rasterized=True)
    ax.set_yticks(range(len(importance)), [FRIENDLY_NAMES.get(f, f) for f in importance.index])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="SHAP values for each loan (XGBoost)", xlabel="Effect on predicted price (basis points)")
    fig.colorbar(points, ax=ax, label="Input value (low = purple, high = yellow; gray = category)")
    save_figure(fig, output / "interpret_shap_beeswarm.png")

    # Chart 3: average SHAP piece per input, Hispanic vs. non-Hispanic borrowers.
    groups = ["Hispanic or Latino", "Not Hispanic or Latino"]
    by_group = shap.groupby(sample.derived_ethnicity).mean().reindex(groups).T.loc[importance.index]
    by_group.rename_axis("feature").to_csv(output / f"{PART}_shap_by_ethnicity.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(by_group))
    for offset, group, color in zip([-0.2, 0.2], groups, ["#e67e22", "#7f8c8d"]):
        ax.barh(positions + offset, by_group[group], height=0.4, color=color, label=group)
    ax.set_yticks(positions, [FRIENDLY_NAMES.get(f, f) for f in by_group.index])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Which inputs push predicted prices apart for each group? (SHAP)", xlabel="Average effect on predicted price (basis points)")
    legend_below(fig, ax, columns=2)
    save_figure(fig, output / "interpret_shap_by_ethnicity.png")

    # ---- Partial dependence ---------------------------------------------------------
    pdp_sample = sample.sample(min(s["pdp_sample"], len(sample)), random_state=config["seed"])
    pdp_rows = []
    fig, axes = plt.subplots(1, len(PDP_FEATURES), figsize=(14, 4.2), sharey=True)
    for ax, (column, axis_label) in zip(axes, PDP_FEATURES):
        for name, model in models.items():
            grid, averages = partial_dependence(model, pdp_sample, column)
            ax.plot(grid, averages, color=MODEL_COLORS[name], marker=".", label=MODEL_LABELS[name].replace(" / logistic", ""))
            pdp_rows += [{"model": name, "feature": column, "value": v, "average_prediction_bp": p} for v, p in zip(grid, averages)]
        ax.set(xlabel=axis_label, title=FRIENDLY_NAMES[column])
        if column == "loan_amount":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value / 1000:,.0f}k"))
    axes[0].set_ylabel("Average predicted rate spread (basis points)")
    legend_below(fig, axes, columns=2)
    fig.suptitle("Partial dependence: change one input for every loan, holding the rest fixed")
    save_figure(fig, output / "interpret_partial_dependence.png")
    pd.DataFrame(pdp_rows).to_csv(output / f"{PART}_partial_dependence.csv", index=False)

    # ---- Permutation importance -----------------------------------------------------
    # Only the raw model inputs are passed in, so only those columns get shuffled.
    inputs = sample[RAW_FEATURES]
    permutation_rows = []
    for name, model in models.items():
        with threadpool_limits(limits=config["n_jobs"]):
            result = permutation_importance(model, inputs, sample.rate_spread, scoring="neg_mean_absolute_error", n_repeats=s["permutation_repeats"], random_state=config["seed"])
        for feature, mean, spread in zip(RAW_FEATURES, result.importances_mean, result.importances_std):
            # Score is negative MAE, so the drop in score = rise in error. Convert to bp.
            permutation_rows.append({"model": name, "feature": feature, "mae_increase_bp": mean * 100, "std_bp": spread * 100})
        log(f"  permutation importance done for {name}")
    permutation = pd.DataFrame(permutation_rows)
    permutation.to_csv(output / f"{PART}_permutation_importance.csv", index=False)

    # Chart 5: side-by-side bars for both models, sorted by XGBoost importance.
    order = permutation[permutation.model.eq("xgboost")].sort_values("mae_increase_bp").feature.tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(order))
    for offset, name in zip([-0.2, 0.2], ["linear", "xgboost"]):
        values = permutation[permutation.model.eq(name)].set_index("feature").reindex(order)
        ax.barh(positions + offset, values.mae_increase_bp, xerr=values.std_bp, height=0.4, color=MODEL_COLORS[name], label=MODEL_LABELS[name].replace(" / logistic", ""))
    ax.set_yticks(positions, [FRIENDLY_NAMES.get(f, f) for f in order])
    ax.set(title="Permutation importance: error increase when one input is scrambled", xlabel="Increase in average error (MAE, basis points)")
    legend_below(fig, ax, columns=2)
    save_figure(fig, output / "interpret_permutation_importance.png")

    # ---- Key numbers for the summary ------------------------------------------------
    findings = {
        "explained_loans": len(sample),
        "shap_mean_abs_bp": importance.sort_values(ascending=False).astype(float).to_dict(),
        "shap_mean_by_ethnicity_bp": {g: by_group[g].astype(float).to_dict() for g in groups if g in by_group},
        "permutation_mae_increase_bp": {name: permutation[permutation.model.eq(name)].set_index("feature").mae_increase_bp.astype(float).to_dict() for name in models},
    }
    figures = {
        "interpret_shap_importance.png": "Average impact of each input on XGBoost predictions (SHAP)",
        "interpret_shap_beeswarm.png": "Each loan's SHAP value per input, colored by the input's value",
        "interpret_shap_by_ethnicity.png": "Average SHAP effect per input, Hispanic or Latino vs. non-Hispanic borrowers",
        "interpret_partial_dependence.png": "Predicted price vs. loan-to-value, income, and loan amount (both models)",
        "interpret_permutation_importance.png": "Error increase when each input is scrambled (both models)",
    }
    write_part_summary(config, PART, "5. Interpretability (what the model pays attention to)", figures, findings)
    return findings
