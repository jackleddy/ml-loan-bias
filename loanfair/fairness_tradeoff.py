"""
Extension study 7: The fairness-accuracy tradeoff
=================================================

WHAT THIS STUDY DOES (in plain language)
    Imagine a lender used our model as a PRICING ENGINE: it would quote each borrower
    the model's predicted rate spread. Even though the model never sees ethnicity, it
    can still quote Hispanic or Latino borrowers a different price on average, because
    its inputs (location, income, loan size, ...) differ between groups. The proxy test
    (proxy_test.py) shows those inputs carry demographic information.

    A common fairness rule for a pricing engine is "demographic parity": on average,
    both groups should be quoted the same price. We add that rule to the model's
    training as a PENALTY:

        training goal = (prediction error) + strength x (average price gap)^2

    where "average price gap" = average predicted spread for Hispanic or Latino
    borrowers minus the average for Not Hispanic or Latino borrowers. With strength 0
    this is the ordinary model; as the strength grows, the model is pushed harder to
    close the gap, even if that makes its predictions less accurate.

    Ethnicity is used ONLY while training (to measure the gap being penalized). The
    finished model still predicts from financial and location inputs alone.

    We train both model types at several strengths, score every loan with a model that
    never saw it (5-fold cross-validation), and plot accuracy against the price gap.
    The result is a tradeoff curve: there is no free lunch, and choosing a point on it
    is a values decision for people and policymakers, not something the math settles.

    IMPORTANT DISTINCTION (also shown in the second chart):
      * "Quoted price gap" = the difference in what the pricing engine would QUOTE each
        group. This is what the fairness rule shrinks.
      * "Unexplained gap" (residual gap) = what borrowers ACTUALLY paid minus what the
        model predicted. This is the main study's disparity measure. It always equals
        (actual gap) - (quoted gap), and forcing equal quotes cannot change what borrowers
        actually paid, so the unexplained gap gets LARGER as the quoted gap shrinks. We
        deliberately do NOT penalize the unexplained gap: closing it would teach the
        model to quote Hispanic borrowers higher prices through proxies, building the
        disparity into the model.

HOW THE PENALTY IS TRAINED
    * Linear regression: the penalty is itself a squared error, so it can be added as
      one extra "row" of data and solved exactly with least squares
      (ParityPenalizedLinear below).
    * XGBoost: we train on SHIFTED TARGETS. Calculus shows the penalized model's best
      predictions satisfy the same equation as an ordinary model fitted to slightly
      adjusted prices: Hispanic loans' prices nudged down and non-Hispanic loans' prices
      nudged up, by an amount set by the strength. Two quick calibration fits per fold
      tell us how big that nudge must be for each strength (fit_shifted_xgboost below).
      For linear regression this shortcut gives exactly the same answer as the exact
      solver, which the tests check.

VISUALS PRODUCED (saved in reports/ml_extensions/)
    fairness_tradeoff_curve.png      Accuracy vs. quoted price gap, one curve per model
    fairness_gaps_by_strength.png    Quoted gap, unexplained gap, and error vs. penalty strength

TABLES PRODUCED
    fairness_tradeoff.csv            Every model/strength result
    fairness_summary.json            Key numbers, used to build the folder's README.md
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from threadpoolctl import threadpool_limits

from .common import log
from .ml_shared import MODEL_COLORS, MODEL_LABELS, legend_below, load_primary, mae_bp, output_dir, save_figure, settings, write_part_summary
from .models import make_model
from .train import fold_assignments

PART = "fairness"
PROTECTED, REFERENCE = "Hispanic or Latino", "Not Hispanic or Latino"


def gap_weights(ethnicity):
    """
    Weights that turn a set of predictions into the average price gap with one dot product:
        gap = sum(weights * predictions)
            = mean(prediction for Hispanic or Latino) - mean(prediction for Not Hispanic or Latino)
    Hispanic borrowers get +1/(number of Hispanic borrowers), non-Hispanic borrowers get
    -1/(number of non-Hispanic borrowers), and everyone else (e.g. not reported) gets 0.
    """
    ethnicity = pd.Series(ethnicity).reset_index(drop=True)
    protected, reference = ethnicity.eq(PROTECTED), ethnicity.eq(REFERENCE)
    if not protected.any() or not reference.any():
        raise ValueError("Both ethnicity groups are needed to measure the price gap.")
    return (protected / protected.sum() - reference / reference.sum()).to_numpy(dtype=float)


class ParityPenalizedLinear(BaseEstimator, RegressorMixin):
    """
    Linear regression with a fairness penalty, solved exactly.

    Ordinary least squares minimizes   sum of (actual - predicted)^2.
    We minimize                         sum of (actual - predicted)^2 + n * strength * gap^2,
    where gap = gap_weights . predictions. Because predictions = X . coefficients, the
    penalty equals one extra squared-error term: an extra data row whose inputs are
    sqrt(n * strength) * (gap_weights . X) and whose target is 0. Appending that row and
    running ordinary least squares solves the penalized problem exactly.
    """

    def __init__(self, strength=0.0):
        self.strength = strength

    def fit(self, X, y, gap_weights=None):
        design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])  # leading 1s = intercept
        target = np.asarray(y, dtype=float)
        if gap_weights is not None and self.strength > 0:
            penalty_row = np.sqrt(len(design) * self.strength) * (gap_weights @ design)
            design = np.vstack([design, penalty_row])
            target = np.append(target, 0.0)
        # lstsq copes with the redundant one-hot columns by picking the smallest solution.
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.intercept_, self.coef_ = solution[0], solution[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


def calibrated_shift(base_gap, probe_gap, probe_shift, n, strength):
    """
    How far to shift the targets so an ordinary fit lands on the penalized solution.

    Background: the penalized goal is smallest where, for every loan,
        (predicted - actual) + n * strength * gap * weight = 0,
    i.e. where the model fits the shifted targets  actual - shift * weight  with
        shift = n * strength * gap.                                          (1)
    The gap itself depends on the shift: an ordinary fit to shifted targets has a gap of
    about  base_gap - sensitivity * shift.  We measure the sensitivity from two fits
    (shift 0 -> base_gap, shift probe_shift -> probe_gap), then solve (1):
        shift = n * strength * base_gap / (1 + n * strength * sensitivity).
    For a linear model the gap changes exactly linearly, so this is exact.
    """
    sensitivity = max((base_gap - probe_gap) / probe_shift, 1e-12) if probe_shift else 1e-12
    return n * strength * base_gap / (1 + n * strength * sensitivity)


def fit_shifted_xgboost(training, weights, strengths, config):
    """
    Fit one XGBoost pricing model per strength using shifted targets (see calibrated_shift).
    Returns {strength: fitted model}. Only the TRAINING loans are used.
    """
    actual = training.rate_spread.to_numpy()

    def fit(shift):
        model = make_model("xgboost", config)
        with threadpool_limits(limits=config["n_jobs"]):
            # Hispanic loans have positive weights, so their targets are nudged down;
            # non-Hispanic loans have negative weights, so theirs are nudged up.
            model.fit(training, actual - shift * weights)
            gap = float(weights @ model.predict(training))
        return model, gap

    base_model, base_gap = fit(0.0)
    # Probe: the shift that would erase the whole gap if the model could see ethnicity.
    probe_shift = base_gap / float(weights @ weights)
    _, probe_gap = fit(probe_shift)
    models = {}
    for strength in strengths:
        models[strength] = base_model if strength == 0 else fit(calibrated_shift(base_gap, probe_gap, probe_shift, len(actual), strength))[0]
    return models


def strength_label(strength):
    """Short chart label for a penalty strength: 0, 1, 10, 100, 1k, 10k, 100k, ..."""
    return f"{strength / 1000:g}k" if strength >= 1000 else f"{strength:g}"


def evaluate(data, predictions):
    """Accuracy and both kinds of gap (in basis points) for one set of out-of-fold predictions."""
    protected = data.derived_ethnicity.eq(PROTECTED).to_numpy()
    reference = data.derived_ethnicity.eq(REFERENCE).to_numpy()
    actual = data.rate_spread.to_numpy()
    residual = actual - predictions
    return {
        "mae_bp": mae_bp(actual, predictions),
        "mae_hispanic_bp": mae_bp(actual[protected], predictions[protected]),
        "mae_not_hispanic_bp": mae_bp(actual[reference], predictions[reference]),
        # What the pricing engine would quote each group, on average.
        "predicted_gap_bp": float((predictions[protected].mean() - predictions[reference].mean()) * 100),
        # What borrowers actually paid beyond the quote, on average (main study's measure).
        "residual_gap_bp": float((residual[protected].mean() - residual[reference].mean()) * 100),
        # What borrowers actually paid, on average (does not depend on the model).
        "actual_gap_bp": float((actual[protected].mean() - actual[reference].mean()) * 100),
    }


def run(config):
    """Train both models at every penalty strength, save the charts and tables, and return the key numbers."""
    s = settings(config)
    strengths = s["fairness_lambdas"]
    output = output_dir(config)
    data = load_primary(config)
    folds = fold_assignments(data, config)
    # predictions[(model, strength)] = out-of-fold predictions for every loan
    predictions = {(name, strength): np.full(len(data), np.nan) for name in ["linear", "xgboost"] for strength in strengths}
    for fold in range(config["folds"]):
        train_rows, test_rows = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        training, testing = data.iloc[train_rows], data.iloc[test_rows]
        # The gap weights are computed from training loans only.
        weights = gap_weights(training.derived_ethnicity)
        for strength in strengths:
            linear = make_model("linear", config).set_params(model=ParityPenalizedLinear(strength))
            linear.fit(training, training.rate_spread, model__gap_weights=weights)
            predictions[("linear", strength)][test_rows] = linear.predict(testing)
        for strength, model in fit_shifted_xgboost(training, weights, strengths, config).items():
            predictions[("xgboost", strength)][test_rows] = model.predict(testing)
        log(f"  fairness fold {fold + 1}/{config['folds']} done")
    rows = []
    for (name, strength), values in predictions.items():
        rows.append({"model": name, "strength": strength, **evaluate(data, values)})
        log(f"  {name}, strength {strength:g}: MAE {rows[-1]['mae_bp']:.2f} bp, quoted gap {rows[-1]['predicted_gap_bp']:.2f} bp, unexplained gap {rows[-1]['residual_gap_bp']:.2f} bp")
    results = pd.DataFrame(rows)
    results.to_csv(output / f"{PART}_tradeoff.csv", index=False)

    # ---- Chart 1: the tradeoff curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, subset in results.groupby("model"):
        ax.plot(subset.predicted_gap_bp, subset.mae_bp, marker="o", color=MODEL_COLORS[name], label=MODEL_LABELS[name].replace(" / logistic", ""))
        # Label each point with its strength, skipping points that sit almost on top of the
        # last labeled one (strong penalties all land at parity), so labels never overlap.
        last_gap = None
        for _, row in subset.iterrows():
            if last_gap is None or abs(row.predicted_gap_bp - last_gap) > 0.1:
                ax.annotate(strength_label(row.strength), (row.predicted_gap_bp, row.mae_bp), textcoords="offset points", xytext=(5, 4), fontsize=8, color=MODEL_COLORS[name])
                last_gap = row.predicted_gap_bp
    ax.axvline(0, color="gray", linestyle=":", label="Equal average quotes (parity)")
    ax.set(title="Fairness vs. accuracy for a pricing engine\n(numbers = penalty strength; 0 = no fairness rule; stronger penalties overlap at parity)",
           xlabel="Average quoted price gap, Hispanic minus non-Hispanic (basis points)", ylabel="Average error (MAE, basis points)")
    legend_below(fig, ax)
    save_figure(fig, output / "fairness_tradeoff_curve.png")

    # ---- Chart 2: both gaps and the error vs. penalty strength ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("predicted_gap_bp", "Quoted price gap\n(what the fairness rule shrinks)", "Basis points"),
        ("residual_gap_bp", "Unexplained gap\n(= actual gap minus quoted gap)", "Basis points"),
        ("mae_bp", "Overall error", "MAE (basis points)"),
    ]
    # Strengths jump by 10x (0, 1, 10, 100, ...), so draw them at evenly spaced positions
    # and label each position with its strength.
    positions = {strength: i for i, strength in enumerate(strengths)}
    for ax, (column, title, ylabel) in zip(axes, panels):
        for name, subset in results.groupby("model"):
            ax.plot(subset.strength.map(positions), subset[column], marker="o", color=MODEL_COLORS[name], label=MODEL_LABELS[name].replace(" / logistic", ""))
        if column != "mae_bp":
            ax.axhline(0, color="gray", linestyle=":")
        ax.set_xticks(list(positions.values()), [strength_label(s) for s in strengths])
        ax.set(title=title, xlabel="Penalty strength (0 = no fairness rule)", ylabel=ylabel)
    fig.suptitle(f"Forcing equal quotes cannot change what borrowers paid (actual gap: {results.actual_gap_bp.iloc[0]:.1f} bp)")
    legend_below(fig, axes)
    save_figure(fig, output / "fairness_gaps_by_strength.png")

    # ---- Key numbers for the summary ------------------------------------------------
    findings = {}
    for name, subset in results.groupby("model"):
        base, strictest = subset.iloc[0], subset.iloc[-1]
        findings[name] = {
            "no_rule": {"mae_bp": float(base.mae_bp), "predicted_gap_bp": float(base.predicted_gap_bp), "residual_gap_bp": float(base.residual_gap_bp)},
            "strictest_rule": {"strength": float(strictest.strength), "mae_bp": float(strictest.mae_bp), "predicted_gap_bp": float(strictest.predicted_gap_bp), "residual_gap_bp": float(strictest.residual_gap_bp)},
            "mae_cost_bp": float(strictest.mae_bp - base.mae_bp),
        }
    findings["actual_gap_bp"] = float(results.actual_gap_bp.iloc[0])
    figures = {
        "fairness_tradeoff_curve.png": "Accuracy vs. average quoted price gap at each penalty strength (both models)",
        "fairness_gaps_by_strength.png": "Quoted gap, unexplained gap, and overall error vs. penalty strength",
    }
    write_part_summary(config, PART, "7. Fairness-accuracy tradeoff", figures, findings)
    return findings
