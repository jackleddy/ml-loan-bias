"""
Extension study 3: Classification -- "Was this a high-priced loan?"
==================================================================

WHAT THIS STUDY DOES (in plain language)
    The main study predicts a NUMBER (how much extra a borrower paid above the
    benchmark rate). This study asks a simpler YES/NO question instead:
    "Was this loan officially high-priced?"

    Federal rules (Regulation Z, "higher-priced mortgage loans") already define the
    cutoff for first-lien loans: a loan is high-priced when its rate spread is
      * 1.5 percentage points or more for conforming loans ("C"), or
      * 2.5 percentage points or more for larger, non-conforming "jumbo" loans ("NC").
    About 2% of the loans in this study cross that line, so it is a rare event.

    We train two yes/no models using the same financial and location inputs as the
    main study (never race, sex, or ethnicity):
      * Logistic regression: the yes/no cousin of linear regression.
      * XGBoost classifier: the yes/no version of the main study's XGBoost model.
    Each loan is scored by a model that did NOT see it during training (5-fold
    cross-validation, the same folds as the main study).

    Each model outputs a probability. To turn probabilities into yes/no flags, we
    flag the riskiest loans: exactly as many as were high-priced in the training
    loans. That cutoff is chosen from the training loans only, so it never peeks
    at the loans being scored.

    Finally, we check whether the model's MISTAKES fall evenly across groups:
      * Missed rate (false negative rate): of the loans that really were
        high-priced, what share did the model fail to flag?
      * False alarm rate (false positive rate): of the loans that were NOT
        high-priced, what share did the model wrongly flag?
    If one group's expensive loans are missed much more often, those borrowers are
    paying high prices that their financial facts do not explain.

VISUALS PRODUCED (saved in reports/ml_extensions/)
    high_priced_roc.png               ROC curves: how well each model separates yes from no
    high_priced_precision_recall.png  Precision-recall curves (better suited to rare events)
    high_priced_confusion.png         Confusion matrices: counts of right and wrong answers
    high_priced_group_errors.png      Missed rate and false alarm rate by race and ethnicity

TABLES PRODUCED
    high_priced_metrics.csv           Overall scores for each model
    high_priced_group_errors.csv      Error rates for each demographic group
    high_priced_summary.json          Key numbers, used to build the folder's README.md
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .common import log
from .ml_shared import MODEL_COLORS, MODEL_LABELS, legend_below, load_primary, output_dir, save_figure, write_part_summary
from .models import make_model
from .train import fold_assignments

PART = "high_priced"

# Regulation Z higher-priced mortgage thresholds for first liens, in percentage points.
HIGH_PRICED_THRESHOLD_PP = {"C": 1.5, "NC": 2.5}

# A group needs at least this many truly high-priced loans before we chart its
# missed rate; with fewer, the percentage would bounce around too much to trust.
MIN_POSITIVES = 30

# Demographic labels that mean "not reported" rather than a real group.
UNAVAILABLE = ("Not Available", "Free Form Text Only")


def high_priced_label(data):
    """Return 1 for high-priced loans and 0 otherwise, using the Regulation Z cutoffs above."""
    cutoff = data.conforming_loan_limit.map(HIGH_PRICED_THRESHOLD_PP).fillna(HIGH_PRICED_THRESHOLD_PP["C"])
    return (data.rate_spread >= cutoff).astype(int)


def make_classifier(name, config, geography=True):
    """
    Build a yes/no model that uses exactly the same inputs and preprocessing as the
    main study's pricing models. We build the project's regression pipeline and swap
    its final step for a classifier, so the inputs can never drift apart.
      name="linear"  -> logistic regression
      name="xgboost" -> XGBoost classifier with the main study's tree settings
    """
    pipeline = make_model(name, config, geography)
    if name == "linear":
        classifier = LogisticRegression(max_iter=2000)
    else:
        classifier = XGBClassifier(**config["xgboost"], objective="binary:logistic", tree_method="hist", device="cpu", random_state=config["seed"], n_jobs=config["n_jobs"])
    return pipeline.set_params(model=classifier)


def oof_probabilities(data, label, name, config, geography=True, flag_cutoff=False):
    """
    Score every loan with a model that never saw it ("out-of-fold"), using the same
    year-balanced folds as the main study.

    Returns a table with each loan's row_id, true label, and predicted probability.
    If flag_cutoff=True it also adds a yes/no `flagged` column: within each fold we
    flag the same share of loans as were truly positive in that fold's TRAINING loans.
    """
    folds = fold_assignments(data, config)
    probability = np.full(len(data), np.nan)
    flagged = np.zeros(len(data), dtype=int)
    for fold in range(config["folds"]):
        train_rows, test_rows = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        training, testing = data.iloc[train_rows], data.iloc[test_rows]
        model = make_classifier(name, config, geography)
        with threadpool_limits(limits=config["n_jobs"]):
            model.fit(training, label.iloc[train_rows])
            probability[test_rows] = model.predict_proba(testing)[:, 1]
            if flag_cutoff:
                # Pick the probability cutoff on TRAINING loans only: flag the top X%,
                # where X% is the share of training loans that were truly high-priced.
                base_rate = label.iloc[train_rows].mean()
                cutoff = np.quantile(model.predict_proba(training)[:, 1], 1 - base_rate)
                flagged[test_rows] = (probability[test_rows] >= cutoff).astype(int)
        log(f"  {name}: fold {fold + 1}/{config['folds']}")
    result = pd.DataFrame({"row_id": data.row_id.to_numpy(), "actual": label.to_numpy(), "probability": probability})
    if flag_cutoff:
        result["flagged"] = flagged
    return result


def group_error_rates(data, scored, model):
    """
    For each race and ethnicity group, compute:
      missed rate      = share of truly high-priced loans the model did not flag
      false alarm rate = share of not-high-priced loans the model flagged
    Groups with too few high-priced loans are kept in the table but marked unreliable.
    """
    merged = scored.merge(data[["row_id", "derived_race", "derived_ethnicity"]], on="row_id", validate="one_to_one")
    rows = []
    for dimension in ["derived_ethnicity", "derived_race"]:
        for group, subset in merged.groupby(dimension):
            if any(label in group for label in UNAVAILABLE):
                continue
            # "Joint" exists for both race and ethnicity, so its chart label names which one.
            display = f"{group} ({'ethnicity' if dimension == 'derived_ethnicity' else 'race'})" if group == "Joint" else group
            positives, negatives = subset[subset.actual.eq(1)], subset[subset.actual.eq(0)]
            rows.append({
                "model": model, "dimension": dimension, "group": display, "loans": len(subset),
                "high_priced_loans": len(positives), "actual_high_priced_rate": subset.actual.mean(),
                "flagged_rate": subset.flagged.mean(),
                "missed_rate": 1 - positives.flagged.mean() if len(positives) else np.nan,
                "false_alarm_rate": negatives.flagged.mean() if len(negatives) else np.nan,
                "reliable": len(positives) >= MIN_POSITIVES,
            })
    return rows


def run(config):
    """Train both yes/no models, measure them, save the charts and tables, and return the key numbers."""
    output = output_dir(config)
    data = load_primary(config)
    label = high_priced_label(data)
    log(f"High-priced classification: {label.sum():,} of {len(data):,} loans are high-priced ({label.mean():.1%})")

    scored, metrics, group_rows = {}, [], []
    for name in ["linear", "xgboost"]:
        scored[name] = oof_probabilities(data, label, name, config, flag_cutoff=True)
        result = scored[name]
        tn, fp, fn, tp = confusion_matrix(result.actual, result.flagged, labels=[0, 1]).ravel()
        metrics.append({
            "model": name,
            # ROC AUC: chance a random high-priced loan gets a higher score than a random
            # normal loan. 0.5 = coin flip, 1.0 = perfect.
            "roc_auc": roc_auc_score(result.actual, result.probability),
            # Average precision (PR AUC): area under the precision-recall curve. A coin
            # flip would score the base rate (about 0.02 here), so compare to that.
            "pr_auc": average_precision_score(result.actual, result.probability),
            "base_rate": result.actual.mean(),
            "precision": tp / (tp + fp) if tp + fp else np.nan,  # of flagged loans, share truly high-priced
            "recall": tp / (tp + fn) if tp + fn else np.nan,     # of high-priced loans, share flagged
            "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
        })
        group_rows += group_error_rates(data, result, name)
    metrics = pd.DataFrame(metrics)
    groups = pd.DataFrame(group_rows)
    metrics.to_csv(output / f"{PART}_metrics.csv", index=False)
    groups.to_csv(output / f"{PART}_group_errors.csv", index=False)

    # ---- Chart 1: ROC curves --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, result in scored.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(result.actual, result.probability)
        auc = metrics.set_index("model").loc[name, "roc_auc"]
        ax.plot(false_positive_rate, true_positive_rate, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Coin flip (AUC 0.5)")
    ax.set(title="ROC curve: spotting high-priced loans", xlabel="False alarm rate (normal loans flagged)", ylabel="Catch rate (high-priced loans flagged)")
    legend_below(fig, ax, columns=1)
    save_figure(fig, output / "high_priced_roc.png")

    # ---- Chart 2: precision-recall curves -------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, result in scored.items():
        precision, recall, _ = precision_recall_curve(result.actual, result.probability)
        ap = metrics.set_index("model").loc[name, "pr_auc"]
        ax.plot(recall, precision, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]} (AP {ap:.3f})")
    ax.axhline(label.mean(), color="gray", linestyle="--", label=f"Random guessing ({label.mean():.1%})")
    ax.set(title="Precision-recall: spotting high-priced loans", xlabel="Recall (share of high-priced loans caught)", ylabel="Precision (share of flags that were right)")
    legend_below(fig, ax, columns=1)
    save_figure(fig, output / "high_priced_precision_recall.png")

    # ---- Chart 3: confusion matrices ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, result) in zip(axes, scored.items()):
        matrix = confusion_matrix(result.actual, result.flagged, labels=[0, 1])
        # Color each cell by its share of its row, so rows (true answers) are comparable.
        shares = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}\n({shares[i, j]:.1%} of row)", ha="center", va="center", color="white" if shares[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
        ax.set_yticks([0, 1], ["Not high-priced", "High-priced"])
        ax.set(title=MODEL_LABELS[name], xlabel="Model's answer", ylabel="True answer")
    fig.suptitle("Confusion matrices (out-of-fold)")
    save_figure(fig, output / "high_priced_confusion.png")

    # ---- Chart 4: error rates by group ----------------------------------------------
    reliable = groups[groups.reliable]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.5 * reliable.group.nunique() + 1.5)), sharey=True)
    group_order = reliable.drop_duplicates("group").sort_values(["dimension", "group"]).group.tolist()
    positions = np.arange(len(group_order))
    for ax, column, title in zip(axes, ["missed_rate", "false_alarm_rate"], ["Missed rate: high-priced loans not flagged", "False alarm rate: normal loans flagged"]):
        for offset, name in zip([-0.2, 0.2], ["linear", "xgboost"]):
            values = reliable[reliable.model.eq(name)].set_index("group").reindex(group_order)[column]
            ax.barh(positions + offset, values, height=0.4, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
        ax.set(title=title, xlabel="Rate")
        # False alarm rates are small (a few percent), so they get one decimal place.
        decimals = 1 if column == "false_alarm_rate" else 0
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _, d=decimals: f"{value:.{d}%}"))
    axes[0].set_yticks(positions, group_order)
    axes[0].invert_yaxis()
    legend_below(fig, axes)
    fig.suptitle(f"Do the models' mistakes fall evenly across groups? (groups with {MIN_POSITIVES}+ high-priced loans)")
    save_figure(fig, output / "high_priced_group_errors.png")

    # ---- Key numbers for the summary ------------------------------------------------
    by_model = metrics.set_index("model")
    xgb_groups = groups[groups.model.eq("xgboost")].set_index("group")
    findings = {
        "high_priced_loans": int(label.sum()), "high_priced_share": float(label.mean()),
        "roc_auc": by_model.roc_auc.astype(float).to_dict(), "pr_auc": by_model.pr_auc.astype(float).to_dict(),
        "precision": by_model.precision.astype(float).to_dict(), "recall": by_model.recall.astype(float).to_dict(),
        "xgboost_missed_rate_by_group": xgb_groups[xgb_groups.reliable].missed_rate.astype(float).to_dict(),
        "xgboost_actual_high_priced_rate_by_group": xgb_groups.actual_high_priced_rate.astype(float).to_dict(),
    }
    figures = {
        "high_priced_roc.png": "ROC curves for spotting high-priced loans, both models",
        "high_priced_precision_recall.png": "Precision-recall curves, both models",
        "high_priced_confusion.png": "Confusion matrices (right and wrong answers), both models",
        "high_priced_group_errors.png": "Missed rate and false alarm rate by race and ethnicity",
    }
    write_part_summary(config, PART, "3. High-priced loan classification", figures, findings)
    return findings
