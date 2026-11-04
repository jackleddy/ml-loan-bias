"""
Extension study 4: Proxy test -- can the model's inputs secretly reveal demographics?
=====================================================================================

WHAT THIS STUDY DOES (in plain language)
    The pricing models never see race, sex, or ethnicity. That sounds fair, but other
    facts can give demographics away. For example, neighborhoods differ in who lives
    there, so knowing WHERE a house is may reveal a lot about WHO is buying it.
    Inputs that stand in for demographics like this are called "proxies".

    This study tests that directly. We train yes/no models that try to guess a
    borrower's ethnicity or race using ONLY the pricing models' inputs (loan amount,
    income, loan-to-value, debt-to-income, conforming status, year, and location):
      * Hispanic or Latino           vs. Not Hispanic or Latino
      * Black or African American    vs. White
      * Asian                        vs. White
    Borrowers whose demographics were not reported are left out of each comparison.

    We run every test twice: once WITH location (latitude/longitude) and once
    WITHOUT it, to see how much of the demographic signal comes from location.
    Every borrower is scored by a model that never saw them (5-fold cross-validation).

    How to read the score (ROC AUC):
      * 0.5 means the inputs reveal nothing (no better than a coin flip).
      * 1.0 means the inputs reveal demographics perfectly.
    If the score is well above 0.5, then "we left demographics out" does not mean the
    model is blind to them. That is why the main study also runs a version without
    location (the "without_geography" sensitivity).

VISUALS PRODUCED (saved in reports/ml_extensions/)
    proxy_auc.png           How well each group can be guessed, with vs. without location
    proxy_roc_hispanic.png  ROC curves for guessing Hispanic or Latino borrowers
    proxy_map_hispanic.png  Map of counties: actual Hispanic share vs. the model's guesses

TABLES PRODUCED
    proxy_results.csv       Scores for every group, model, and feature set
    proxy_summary.json      Key numbers, used to build the folder's README.md
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .common import log
from .high_priced import oof_probabilities
from .ml_shared import MODEL_COLORS, MODEL_LABELS, legend_below, load_primary, output_dir, save_figure, write_part_summary

PART = "proxy"

# Each test: (short name, demographic column, group we try to detect, comparison group).
PROXY_TARGETS = [
    ("Hispanic or Latino", "derived_ethnicity", "Hispanic or Latino", "Not Hispanic or Latino"),
    ("Black or African American", "derived_race", "Black or African American", "White"),
    ("Asian", "derived_race", "Asian", "White"),
]

# The two feature sets compared: all pricing inputs, or all of them except location.
FEATURE_SETS = {True: "with location", False: "without location"}


def top_decile_lift(actual, probability):
    """
    "Lift" in the top 10%: among the 10% of borrowers the model is most confident
    about, how much more common is the group than it is overall? 1.0 = no better
    than random; 3.0 = three times as common as average.
    """
    order = np.argsort(-np.asarray(probability))
    top = np.asarray(actual)[order[: max(1, len(order) // 10)]]
    return float(top.mean() / np.mean(actual))


def run(config):
    """Run every proxy test, save the charts and tables, and return the key numbers."""
    output = output_dir(config)
    data = load_primary(config)
    rows, curves, hispanic_scores = [], {}, None

    for target, column, positive, negative in PROXY_TARGETS:
        # Keep only the two groups being compared, and label the target group as 1.
        subset = data[data[column].isin([positive, negative])].reset_index(drop=True)
        label = subset[column].eq(positive).astype(int)
        # Each fold needs examples of both groups; skip a test when there are too few.
        if label.sum() < 2 * config["folds"] or (1 - label).sum() < 2 * config["folds"]:
            log(f"Proxy test skipped for {target}: not enough borrowers in both groups")
            continue
        log(f"Proxy test: {target} ({label.sum():,}) vs {negative} ({(1 - label).sum():,})")
        for geography, feature_set in FEATURE_SETS.items():
            for name in ["linear", "xgboost"]:
                scored = oof_probabilities(subset, label, name, config, geography=geography)
                rows.append({
                    "target": target, "comparison_group": negative, "feature_set": feature_set, "model": name,
                    "target_loans": int(label.sum()), "comparison_loans": int((1 - label).sum()),
                    "roc_auc": roc_auc_score(scored.actual, scored.probability),
                    "top_decile_lift": top_decile_lift(scored.actual, scored.probability),
                })
                if target == "Hispanic or Latino":
                    curves[(name, feature_set)] = roc_curve(scored.actual, scored.probability)[:2]
                    if name == "xgboost" and geography:
                        # Keep these scores for the county map below.
                        hispanic_scores = scored.merge(subset[["row_id", "county_code", "latitude", "longitude"]], on="row_id")
                log(f"    {name}, {feature_set}: AUC {rows[-1]['roc_auc']:.3f}")
    results = pd.DataFrame(rows)
    results.to_csv(output / f"{PART}_results.csv", index=False)
    if results.empty:
        raise ValueError("No proxy test had enough borrowers in both groups.")

    # ---- Chart 1: AUC bar chart -----------------------------------------------------
    targets = results.target.unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    bars = [("linear", "with location"), ("linear", "without location"), ("xgboost", "with location"), ("xgboost", "without location")]
    for i, (name, feature_set) in enumerate(bars):
        values = results[(results.model == name) & (results.feature_set == feature_set)].set_index("target").reindex(targets).roc_auc
        # Solid bars = with location; hatched, lighter bars = without location.
        hatch = "" if feature_set == "with location" else "//"
        alpha = 1.0 if feature_set == "with location" else 0.55
        positions = np.arange(len(targets)) + (i - 1.5) * width
        ax.bar(positions, values, width=width, color=MODEL_COLORS[name], alpha=alpha, hatch=hatch, edgecolor="white", label=f"{MODEL_LABELS[name]}, {feature_set}")
        for x, value in zip(positions, values):
            if np.isfinite(value):
                ax.text(x, value + 0.005, f"{value:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", label="Coin flip (0.5)")
    ax.set_xticks(np.arange(len(targets)), [f"{t}\nvs. {results[results.target == t].comparison_group.iloc[0]}" for t in targets])
    # Leave room above the tallest bar for its value label.
    ax.set_ylim(0.45, max(0.75, results.roc_auc.max() + 0.05))
    ax.set(title="Can the pricing inputs guess a borrower's demographics?", ylabel="ROC AUC (0.5 = coin flip, 1.0 = perfect)")
    legend_below(fig, ax, columns=2)
    save_figure(fig, output / "proxy_auc.png")

    # ---- Chart 2: ROC curves for the Hispanic or Latino test ------------------------
    if curves:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for (name, feature_set), (false_positive_rate, true_positive_rate) in curves.items():
            style = "-" if feature_set == "with location" else "--"
            auc = results[(results.target == "Hispanic or Latino") & (results.model == name) & (results.feature_set == feature_set)].roc_auc.iloc[0]
            ax.plot(false_positive_rate, true_positive_rate, linestyle=style, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]}, {feature_set} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Coin flip")
        ax.set(title="Guessing Hispanic or Latino borrowers\nfrom pricing inputs", xlabel="Non-Hispanic borrowers wrongly picked", ylabel="Hispanic borrowers correctly picked")
        legend_below(fig, ax, columns=1)
        save_figure(fig, output / "proxy_roc_hispanic.png")

    # ---- Chart 3: county map, actual vs. guessed Hispanic share ---------------------
    if hispanic_scores is not None:
        county = hispanic_scores.groupby("county_code").agg(
            latitude=("latitude", "mean"), longitude=("longitude", "mean"),
            actual_share=("actual", "mean"), guessed_share=("probability", "mean"), loans=("row_id", "size"))
        # Skip tiny counties, whose shares are too noisy to show.
        county = county[county.loans >= config["min_group_size"]]
        if len(county):
            # "constrained" layout leaves room for one color bar shared by both maps.
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
            top = max(county.actual_share.max(), county.guessed_share.max())
            for ax, column, title in zip(axes, ["actual_share", "guessed_share"], ["Actual share of Hispanic or Latino borrowers", "Model's average guess (XGBoost, with location)"]):
                points = ax.scatter(county.longitude, county.latitude, c=county[column], cmap="viridis", vmin=0, vmax=top, s=np.clip(county.loans / 50, 10, 200))
                ax.set(title=title, xlabel="Longitude")
            axes[0].set_ylabel("Latitude")
            fig.colorbar(points, ax=axes, label="Share Hispanic or Latino", format=plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
            fig.suptitle("Virginia counties (dot size = number of loans)")
            fig.savefig(output / "proxy_map_hispanic.png", dpi=160)
            plt.close(fig)

    # ---- Key numbers for the summary ------------------------------------------------
    table = results.set_index(["target", "model", "feature_set"])
    findings = {
        "roc_auc": {f"{t} | {m} | {f}": float(v) for (t, m, f), v in table.roc_auc.items()},
        "top_decile_lift": {f"{t} | {m} | {f}": float(v) for (t, m, f), v in table.top_decile_lift.items()},
    }
    figures = {
        "proxy_auc.png": "How well each group can be guessed from pricing inputs, with vs. without location",
        "proxy_roc_hispanic.png": "ROC curves for guessing Hispanic or Latino borrowers",
        "proxy_map_hispanic.png": "County map: actual Hispanic or Latino share vs. the model's guesses",
    }
    write_part_summary(config, PART, "4. Proxy test (can inputs reveal demographics?)", figures, findings)
    return findings
