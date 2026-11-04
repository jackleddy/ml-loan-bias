"""
Shared helpers for the machine-learning extension studies
=========================================================

WHAT THIS FILE DOES
    This file holds small helper functions that every extension study uses, so
    each study file can focus on its own idea instead of repeating setup code.
    It does NOT produce any charts by itself.

    The extension studies (each in its own file) are:
        bias_variance.py      -> Is the model too simple or too complex? (tuning)
        high_priced.py        -> Yes/no question: was this an expensive loan?
        proxy_test.py         -> Can the model secretly guess demographics?
        interpretability.py   -> What does the model pay attention to?
        fairness_tradeoff.py  -> How much accuracy does fairness cost?

    All of their charts and tables are saved in one folder:
        reports/ml_extensions/

HELPERS PROVIDED
    settings()        -> study settings from config.json, with sensible defaults
    load_primary()    -> the 438,863 cleaned loans used by the main study
    output_dir()      -> the folder where charts and tables are saved
    save_figure()     -> saves a Matplotlib chart as a PNG and closes it
    mae_bp()          -> average prediction error, in basis points
    write_part_summary() -> saves a small JSON summary of a study's key numbers
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Matplotlib (the charting library) needs a writable cache folder. We point it at
# the system temp folder, matching what analysis.py does for the main report.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "loanfair-matplotlib"))
import matplotlib

# "Agg" draws charts straight to image files without opening any windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import write_json

# Every extension chart and table goes into reports/ml_extensions/.
OUTPUT_SUBDIR = "ml_extensions"

# Default settings for the studies. Any of these can be overridden by adding an
# "ml_extensions" section to config.json (the tests use that to run tiny versions).
DEFAULT_SETTINGS = {
    # bias_variance.py: tree depths and tree counts to try, plus how many folds.
    "bias_variance_folds": 3,
    "depths": [1, 2, 3, 4, 6, 8, 10],
    "tree_counts": [25, 50, 100, 200, 400, 800],
    # bias_variance.py: how many training loans to use when drawing the learning curve.
    "learning_curve_sizes": [1000, 3000, 10000, 30000, 100000, 300000],
    # bias_variance.py: Ridge regression penalty strengths, and the size of the
    # deliberately small training set used to show when regularization helps.
    "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "ridge_small_sample": 300,
    # interpretability.py: how many held-out loans to explain, and how many to use
    # for the (slower) partial dependence curves.
    "interpret_sample": 20000,
    "pdp_sample": 3000,
    "permutation_repeats": 5,
    # fairness_tradeoff.py: fairness rule strengths to try (0 = no fairness rule).
    "fairness_lambdas": [0, 1, 10, 100, 1000, 10000, 100000],
}

# Plain-English names for the model inputs, used as chart labels.
FRIENDLY_NAMES = {
    "log_loan_amount": "Loan amount",
    "loan_amount": "Loan amount",
    "signed_log_income": "Income",
    "income": "Income",
    "combined_loan_to_value_ratio": "Loan-to-value (CLTV)",
    "activity_year": "Year",
    "debt_to_income_ratio": "Debt-to-income (DTI)",
    "conforming_loan_limit": "Conforming status",
    "latitude": "Latitude",
    "longitude": "Longitude",
}

# The two model types used throughout the project, with display colors.
MODEL_COLORS = {"linear": "#174a7e", "xgboost": "#c0392b"}
MODEL_LABELS = {"linear": "Linear / logistic regression", "xgboost": "XGBoost"}


def settings(config):
    """Return the study settings: the defaults, updated with anything in config.json."""
    return {**DEFAULT_SETTINGS, **config.get("ml_extensions", {})}


def load_primary(config):
    """Load the cleaned loans (the same 438,863 records the main study uses)."""
    path = Path(config["processed_dir"]) / "loans.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/prepare_data.py first.")
    return pd.read_parquet(path).reset_index(drop=True)


def output_dir(config):
    """Return reports/ml_extensions/, creating it if needed."""
    path = Path(config["report_dir"]) / OUTPUT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def legend_below(fig, axes, columns=3):
    """
    Draw one legend for the whole chart BELOW the plotting area, so it can never cover
    data. Entries from every panel are combined, and repeated labels are shown once.
    """
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    # Anchor the legend's top edge at the bottom of the figure; save_figure() then
    # enlarges the saved image to fit it.
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=min(columns, len(labels)), frameon=False)


def save_figure(fig, path):
    """Save a chart as a 160-dpi PNG (same quality as the main report) and free its memory."""
    fig.tight_layout()
    # bbox_inches="tight" grows the image to include anything outside the plot area,
    # such as legends placed underneath by legend_below().
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def mae_bp(actual, predicted):
    """
    Mean absolute error in basis points.

    "On average, how far off was the guess?" The target (rate spread) is measured in
    percentage points, and 1 percentage point = 100 basis points, so we multiply by 100.
    """
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))) * 100)


def write_part_summary(config, part, title, figures, findings):
    """
    Save a small JSON file describing one study: its title, the charts it made,
    and its key numbers. The runner (ml_extensions.py) combines these files into
    reports/ml_extensions/README.md so it is easy to see which chart came from where.
    """
    write_json(output_dir(config) / f"{part}_summary.json", {"part": part, "title": title, "figures": figures, "findings": findings})


def read_part_summaries(config):
    """Load every study summary written so far, in a stable order."""
    summaries = []
    for path in sorted(output_dir(config).glob("*_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries
