"""
Extension study 1: Bias-variance analysis and model tuning
==========================================================

WHAT THIS STUDY DOES (in plain language)
    A model can fail in two opposite ways:
      * TOO SIMPLE ("high bias"): it misses real patterns, so it is wrong on the
        loans it trained on AND on new loans.
      * TOO COMPLEX ("high variance"): it memorizes its training loans, including
        random noise, so it looks great on training loans but worse on new loans.

    The main study used one fixed set of XGBoost settings (trees 4 levels deep,
    400 trees). This study tries many settings and measures error on loans the
    model did NOT train on ("validation" loans), to show where "too simple" ends,
    where "too complex" begins, and whether the chosen settings are a good choice.

    It runs four experiments:
      1. Tree depth: how many yes/no questions each XGBoost tree may ask in a row.
      2. Number of trees: how many small trees XGBoost adds together.
      3. Learning curve: how error changes as we give each model more training loans.
      4. Ridge regression: a version of linear regression with a "penalty" that
         keeps its coefficients small (regularization), tried at several strengths,
         once with a tiny training set and once with the full training set.

    Error is measured as MAE (mean absolute error) in basis points (bp):
    "on average, how far off was the guess?" 100 bp = 1 percentage point.

VISUALS PRODUCED (saved in reports/ml_extensions/)
    bias_variance_depth.png           Training vs validation error as trees get deeper
    bias_variance_trees.png           Training vs validation error as trees are added
    bias_variance_learning_curve.png  Error vs number of training loans, both models
    bias_variance_ridge.png           Validation error vs Ridge penalty strength

TABLES PRODUCED
    bias_variance_results.csv         Every measured error (one row per fit, fold, and split)
    bias_variance_summary.json        Key numbers, used to build the folder's README.md
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .common import log
from .ml_shared import MODEL_COLORS, MODEL_LABELS, legend_below, load_primary, mae_bp, output_dir, save_figure, settings, write_part_summary
from .models import make_model
from .train import fold_assignments

PART = "bias_variance"


def xgboost_config(config, **changes):
    """Return a copy of the config with some XGBoost settings changed (e.g. max_depth=6)."""
    return {**config, "xgboost": {**config["xgboost"], **changes}}


def split_folds(data, config, folds):
    """
    Split the loans into `folds` groups (each year spread evenly across groups).
    Each group takes one turn as the validation set while the rest are training loans.
    Yields (fold number, training loans, validation loans).
    """
    assignments = fold_assignments(data, {**config, "folds": folds})
    for fold in range(folds):
        yield fold, data[assignments != fold], data[assignments == fold]


def depth_experiment(data, config, s):
    """Experiment 1: fit XGBoost at each tree depth and record training and validation error."""
    rows = []
    for fold, training, validation in split_folds(data, config, s["bias_variance_folds"]):
        for depth in s["depths"]:
            model = make_model("xgboost", xgboost_config(config, max_depth=depth))
            with threadpool_limits(limits=config["n_jobs"]):
                model.fit(training, training.rate_spread)
                # Error on the loans it learned from vs. error on loans it never saw.
                train_error = mae_bp(training.rate_spread, model.predict(training))
                validation_error = mae_bp(validation.rate_spread, model.predict(validation))
            rows.append({"experiment": "depth", "model": "xgboost", "setting": depth, "fold": fold, "split": "training", "mae_bp": train_error})
            rows.append({"experiment": "depth", "model": "xgboost", "setting": depth, "fold": fold, "split": "validation", "mae_bp": validation_error})
            log(f"  depth {depth}, fold {fold + 1}: train {train_error:.2f} bp, validation {validation_error:.2f} bp")
    return rows


def tree_count_experiment(data, config, s):
    """
    Experiment 2: fit XGBoost once with the largest tree count, then score it using
    only the first 25, 50, 100, ... trees. XGBoost builds trees one after another, so
    "the first k trees" is exactly what a k-tree model would have been.
    """
    rows = []
    most = max(s["tree_counts"])
    for fold, training, validation in split_folds(data, config, s["bias_variance_folds"]):
        model = make_model("xgboost", xgboost_config(config, n_estimators=most))
        with threadpool_limits(limits=config["n_jobs"]):
            model.fit(training, training.rate_spread)
            for count in s["tree_counts"]:
                # iteration_range=(0, count) tells XGBoost to use only the first `count` trees.
                train_error = mae_bp(training.rate_spread, model.predict(training, iteration_range=(0, count)))
                validation_error = mae_bp(validation.rate_spread, model.predict(validation, iteration_range=(0, count)))
                rows.append({"experiment": "trees", "model": "xgboost", "setting": count, "fold": fold, "split": "training", "mae_bp": train_error})
                rows.append({"experiment": "trees", "model": "xgboost", "setting": count, "fold": fold, "split": "validation", "mae_bp": validation_error})
        log(f"  tree counts, fold {fold + 1} done")
    return rows


def learning_curve_experiment(data, config, s):
    """
    Experiment 3: train each model on a random subset of the training loans (small to
    large) and always test on the same full validation set. A big, persistent gap
    between training and validation error signals variance; two high, close lines
    signal bias.
    """
    rows = []
    rng = np.random.default_rng(config["seed"])
    for fold, training, validation in split_folds(data, config, s["bias_variance_folds"]):
        # Use each requested size that fits, plus the entire training set as the last point.
        sizes = sorted({size for size in s["learning_curve_sizes"] if size < len(training)} | {len(training)})
        for size in sizes:
            subset = training.iloc[rng.choice(len(training), size=size, replace=False)]
            for name in ["linear", "xgboost"]:
                model = make_model(name, config)
                with threadpool_limits(limits=config["n_jobs"]):
                    model.fit(subset, subset.rate_spread)
                    train_error = mae_bp(subset.rate_spread, model.predict(subset))
                    validation_error = mae_bp(validation.rate_spread, model.predict(validation))
                rows.append({"experiment": "learning_curve", "model": name, "setting": size, "fold": fold, "split": "training", "mae_bp": train_error})
                rows.append({"experiment": "learning_curve", "model": name, "setting": size, "fold": fold, "split": "validation", "mae_bp": validation_error})
        log(f"  learning curve, fold {fold + 1} done")
    return rows


def ridge_experiment(data, config, s):
    """
    Experiment 4: Ridge regression = linear regression plus a penalty on large
    coefficients. A bigger `alpha` means a stronger penalty (a simpler model).
    We try it with a tiny training set (where overfitting is possible) and with the
    full training set (where it usually is not), to show when regularization helps.
    """
    rows = []
    rng = np.random.default_rng(config["seed"])
    for fold, training, validation in split_folds(data, config, s["bias_variance_folds"]):
        small = training.iloc[rng.choice(len(training), size=min(s["ridge_small_sample"], len(training)), replace=False)]
        for label, subset in [("small", small), ("full", training)]:
            for alpha in s["ridge_alphas"]:
                # Reuse the project's linear pipeline (same inputs and scaling) but swap in Ridge.
                model = make_model("linear", config).set_params(model=Ridge(alpha=alpha))
                model.fit(subset, subset.rate_spread)
                rows.append({"experiment": f"ridge_{label}", "model": "linear", "setting": alpha, "fold": fold, "split": "training", "mae_bp": mae_bp(subset.rate_spread, model.predict(subset))})
                rows.append({"experiment": f"ridge_{label}", "model": "linear", "setting": alpha, "fold": fold, "split": "validation", "mae_bp": mae_bp(validation.rate_spread, model.predict(validation))})
        log(f"  ridge, fold {fold + 1} done")
    return rows


def average(results, experiment, model=None):
    """Average the error across folds for one experiment -> one row per setting and split."""
    subset = results[results.experiment.eq(experiment)]
    if model:
        subset = subset[subset.model.eq(model)]
    return subset.groupby(["setting", "split"]).mae_bp.mean().unstack("split")


def plot_train_vs_validation(table, ax, color, label_prefix=""):
    """Draw the standard bias-variance picture: a dashed training line and a solid validation line."""
    ax.plot(table.index, table["training"], linestyle="--", marker="o", color=color, alpha=0.7, label=f"{label_prefix}training loans (dashed)")
    ax.plot(table.index, table["validation"], linestyle="-", marker="o", color=color, label=f"{label_prefix}validation loans, unseen (solid)")


def run(config):
    """Run all four experiments, save the charts and tables, and return the key numbers."""
    s = settings(config)
    output = output_dir(config)
    data = load_primary(config)
    log(f"Bias-variance study on {len(data):,} loans ({s['bias_variance_folds']} folds)")

    rows = depth_experiment(data, config, s) + tree_count_experiment(data, config, s)
    rows += learning_curve_experiment(data, config, s) + ridge_experiment(data, config, s)
    results = pd.DataFrame(rows)
    results.to_csv(output / f"{PART}_results.csv", index=False)

    # ---- Chart 1: tree depth --------------------------------------------------------
    depth = average(results, "depth")
    best_depth = int(depth["validation"].idxmin())
    current_depth = config["xgboost"]["max_depth"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_train_vs_validation(depth, ax, MODEL_COLORS["xgboost"])
    ax.axvline(best_depth, color="green", linestyle=":", label=f"Lowest validation error (depth {best_depth})")
    if current_depth != best_depth:
        ax.axvline(current_depth, color="gray", linestyle=":", label=f"Setting used in main study (depth {current_depth})")
    ax.set(title="XGBoost tree depth: from too simple (left) to too complex (right)", xlabel="Maximum tree depth", ylabel="Average error (MAE, basis points)")
    legend_below(fig, ax)
    save_figure(fig, output / "bias_variance_depth.png")

    # ---- Chart 2: number of trees ---------------------------------------------------
    trees = average(results, "trees")
    best_trees = int(trees["validation"].idxmin())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_train_vs_validation(trees, ax, MODEL_COLORS["xgboost"])
    ax.axvline(config["xgboost"]["n_estimators"], color="gray", linestyle=":", label=f"Setting used in main study ({config['xgboost']['n_estimators']} trees)")
    ax.set_xscale("log")
    # Label the x-axis with the actual tree counts (25, 50, ...) instead of powers of ten.
    ax.set_xticks(s["tree_counts"], [str(count) for count in s["tree_counts"]])
    ax.minorticks_off()
    ax.set(title=f"XGBoost: adding more trees (depth {current_depth})", xlabel="Number of trees (log scale)", ylabel="Average error (MAE, basis points)")
    legend_below(fig, ax)
    save_figure(fig, output / "bias_variance_trees.png")

    # ---- Chart 3: learning curves ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    curves = {}
    for ax, name in zip(axes, ["linear", "xgboost"]):
        curves[name] = average(results, "learning_curve", name)
        model_label = MODEL_LABELS[name].replace(" / logistic", "")
        # Prefix labels with the model name so the shared legend can tell the panels apart.
        plot_train_vs_validation(curves[name], ax, MODEL_COLORS[name], label_prefix=f"{model_label}, ")
        ax.set_xscale("log")
        ax.set(title=model_label, xlabel="Number of training loans (log scale)")
    axes[0].set_ylabel("Average error (MAE, basis points)")
    fig.suptitle("Learning curves: does more data help?")
    legend_below(fig, axes, columns=2)
    save_figure(fig, output / "bias_variance_learning_curve.png")

    # ---- Chart 4: Ridge regularization ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ridge = {}
    for ax, label in zip(axes, ["small", "full"]):
        ridge[label] = average(results, f"ridge_{label}")
        plot_train_vs_validation(ridge[label], ax, MODEL_COLORS["linear"])
        ax.set_xscale("log")
        count = s["ridge_small_sample"] if label == "small" else "all"
        # Matplotlib zooms each panel to fit its data, so state the true range in the title
        # to avoid tiny differences looking dramatic.
        span = ridge[label]["validation"].max() - ridge[label]["validation"].min()
        ax.set(title=f"Ridge regression, {count} training loans\n(validation error varies by only {span:.2f} bp)",xlabel="Penalty strength alpha (log scale; right = simpler model)", ylabel="Average error (MAE, basis points)")
    legend_below(fig, axes)
    save_figure(fig, output / "bias_variance_ridge.png")

    # ---- Key numbers for the summary ------------------------------------------------
    final_gap = {name: float(table["validation"].iloc[-1] - table["training"].iloc[-1]) for name, table in curves.items()}
    findings = {
        "best_depth": best_depth,
        "best_depth_validation_mae_bp": float(depth.loc[best_depth, "validation"]),
        "main_study_depth": current_depth,
        "main_study_depth_validation_mae_bp": float(depth.loc[current_depth, "validation"]) if current_depth in depth.index else None,
        "deepest_tree_train_minus_validation_gap_bp": float(depth["validation"].iloc[-1] - depth["training"].iloc[-1]),
        "best_tree_count": best_trees,
        "best_tree_count_validation_mae_bp": float(trees.loc[best_trees, "validation"]),
        "learning_curve_final_validation_mae_bp": {name: float(table["validation"].iloc[-1]) for name, table in curves.items()},
        "learning_curve_final_validation_minus_training_bp": final_gap,
        "ridge_best_alpha": {label: float(table["validation"].idxmin()) for label, table in ridge.items()},
        "ridge_improvement_over_weakest_penalty_bp": {label: float(table["validation"].iloc[0] - table["validation"].min()) for label, table in ridge.items()},
    }
    figures = {
        "bias_variance_depth.png": "Training vs validation error as XGBoost trees get deeper",
        "bias_variance_trees.png": "Training vs validation error as more XGBoost trees are added",
        "bias_variance_learning_curve.png": "Error vs number of training loans for both models",
        "bias_variance_ridge.png": "Ridge regression error vs penalty strength, small vs full training set",
    }
    write_part_summary(config, PART, "1. Bias-variance analysis and tuning", figures, findings)
    return findings
