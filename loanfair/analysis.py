"""Descriptive residual reports and explicitly exploratory group inference."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Keep Matplotlib's cache in a writable temporary directory, including in sandboxes.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "loanfair-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests

from .common import fingerprint, log, write_json

REFERENCES = {"derived_race": "White", "derived_sex": "Male", "derived_ethnicity": "Not Hispanic or Latino", "race_by_sex": "White | Male"}
UNAVAILABLE = ("Not Available", "Free Form Text Only")


def add_intersection(data):
    data = data.copy()
    data["race_by_sex"] = data.derived_race + " | " + data.derived_sex
    return data


def adjust_pvalues(table, source="welch_p", target="welch_q"):
    table = table.copy()
    table[target] = np.nan
    if source in table:
        valid = table[source].notna() & np.isfinite(table[source])
        if valid.any():
            table.loc[valid, target] = multipletests(table.loc[valid, source], method="fdr_bh")[1]
    return table


def clustered_draws(data, dimension, config):
    """Resample whole lenders, retaining unequal group counts within each cluster."""
    grouped = data.groupby(["lei", dimension], observed=True).residual_bp.agg(["sum", "size"])
    sums = grouped["sum"].unstack(fill_value=0)
    counts = grouped["size"].unstack(fill_value=0).reindex(index=sums.index, columns=sums.columns, fill_value=0)
    rng = np.random.default_rng(config["seed"])
    draws = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=config["bootstrap_replicates"])
    totals = draws @ sums.to_numpy()
    sizes = draws @ counts.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        means = totals / sizes
    return pd.DataFrame(means, columns=sums.columns)


def group_summary(data, dimension, config, inference=True):
    reference = REFERENCES[dimension]
    grouped = data.groupby(dimension, observed=True)
    summary = grouped.agg(n=("row_id", "size"), n_lenders=("lei", "nunique"), actual_mean_spread=("actual_spread", "mean"), predicted_mean_spread=("predicted_spread", "mean"), mean_residual_bp=("residual_bp", "mean"), median_residual_bp=("residual_bp", "median")).reset_index().rename(columns={dimension: "group"})
    summary["dimension"], summary["reference_group"] = dimension, reference
    ref_data = data[data[dimension].eq(reference)]
    reference_mean = ref_data.residual_bp.mean()
    summary["difference_from_reference_bp"] = summary.mean_residual_bp - reference_mean
    summary["inference_eligible"] = False
    summary["ci_low_bp"], summary["ci_high_bp"], summary["welch_p"] = np.nan, np.nan, np.nan
    summary["inference_status"] = "descriptive_only"
    if not inference or ref_data.empty:
        return summary
    ref_sufficient = len(ref_data) >= config["min_group_size"] and ref_data.lei.nunique() >= config["min_lenders"] and not ref_data.lei.eq("Lender Not Available").any()
    boot = clustered_draws(data, dimension, config) if ref_sufficient else None
    for index, row in summary.iterrows():
        group = row["group"]
        if group == reference:
            summary.loc[index, "inference_status"] = "reference"
            continue
        if any(label in group for label in UNAVAILABLE):
            summary.loc[index, "inference_status"] = "unavailable_demographics"
            continue
        subset = data[data[dimension].eq(group)]
        sufficient = ref_sufficient and row.n >= config["min_group_size"] and row.n_lenders >= config["min_lenders"] and not subset.lei.eq("Lender Not Available").any()
        if not sufficient:
            summary.loc[index, "inference_status"] = "insufficient_group_or_lender_count"
            continue
        differences = (boot[group] - boot[reference]).dropna()
        if len(differences) < 0.95 * config["bootstrap_replicates"]:
            summary.loc[index, "inference_status"] = "unstable_bootstrap"
            continue
        summary.loc[index, ["ci_low_bp", "ci_high_bp"]] = differences.quantile([0.025, 0.975]).to_numpy()
        p = float(ttest_ind(subset.residual_bp, ref_data.residual_bp, equal_var=False).pvalue)
        summary.loc[index, "welch_p"] = p if np.isfinite(p) else np.nan
        summary.loc[index, "inference_eligible"] = True
        summary.loc[index, "inference_status"] = "exploratory"
    return summary


def residual_regression(data, dimension, config):
    """Separate year-adjusted group regression, with lender-clustered covariance."""
    reference = REFERENCES[dimension]
    counts = data.groupby(dimension, observed=True).agg(n=("row_id", "size"), lenders=("lei", "nunique"))
    valid_groups = counts.index[(counts.n >= config["min_group_size"]) & (counts.lenders >= config["min_lenders"])].tolist()
    valid_groups = [g for g in valid_groups if not any(label in g for label in UNAVAILABLE)]
    if reference not in valid_groups or len(valid_groups) < 2:
        return []
    subset = data[data[dimension].isin(valid_groups) & data.lei.ne("Lender Not Available")].copy()
    group_dummies = pd.get_dummies(subset[dimension], dtype=float).drop(columns=reference)
    group_dummies.columns = [f"group::{g}" for g in group_dummies.columns]
    year_dummies = pd.get_dummies(subset.activity_year.astype(int).astype(str), prefix="year", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([group_dummies, year_dummies], axis=1), has_constant="add")
    if len(design) <= design.shape[1] or np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        return []
    result = sm.OLS(subset.residual_bp.astype(float), design).fit(cov_type="cluster", cov_kwds={"groups": subset.lei.to_numpy()}, use_t=True)
    intervals = result.conf_int()
    rows = []
    for term in group_dummies.columns:
        group = term.removeprefix("group::")
        rows.append({"dimension": dimension, "group": group, "reference_group": reference, "adjusted_difference_bp": float(result.params[term]), "ci_low_bp": float(intervals.loc[term, 0]), "ci_high_bp": float(intervals.loc[term, 1]), "cluster_p": float(result.pvalues[term]), "regression_n": len(subset), "regression_lenders": subset.lei.nunique()})
    return rows


def descriptive_table(data, keys):
    return data.groupby(keys, observed=True, dropna=False).agg(n=("row_id", "size"), actual_mean_spread=("actual_spread", "mean"), predicted_mean_spread=("predicted_spread", "mean"), mean_residual_bp=("residual_bp", "mean"), median_residual_bp=("residual_bp", "median")).reset_index()


def joined_predictions(base, path):
    predictions = pd.read_parquet(path)
    joined = predictions.merge(base.drop(columns=["activity_year"]), on="row_id", how="left", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError(f"Predictions in {path} do not match processed rows.")
    return add_intersection(joined.drop(columns="_merge"))


def benchmark_report(config, output):
    table = pd.read_parquet(Path(config["reference_dir"]) / "apor.parquet")
    rates = table[(table.amortization == "fixed") & (table.term_years == 30) & table.effective_date.dt.year.isin(config["years"])].copy()
    if rates.empty:
        raise ValueError("Reference cache has no APOR values in the study period.")
    rates["year"] = rates.effective_date.dt.year
    # Weekly effective observations, not loan-weighted or daily-weighted annual averages.
    summary = rates.groupby("year").apor.agg(["count", "mean", "min", "max"]).reset_index()
    summary.to_csv(output / "apor_annual.csv", index=False)
    rates.to_csv(output / "apor_weekly_fixed30.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rates.effective_date, rates.apor)
    ax.set(title="30-year fixed Average Prime Offer Rate", ylabel="APOR (%)", xlabel="Effective date")
    fig.tight_layout(); fig.savefig(output / "apor_fixed30.png", dpi=160); plt.close(fig)


def plot_groups(summary, selected, output):
    for dimension, filename in [("derived_race", "race_gaps.png"), ("derived_sex", "sex_gaps.png")]:
        data = summary[(summary.model == selected) & (summary.dimension == dimension)].copy().sort_values("difference_from_reference_bp")
        data = data[data.difference_from_reference_bp.notna()]
        fig, ax = plt.subplots(figsize=(10, max(3, len(data) * 0.45)))
        positions = np.arange(len(data))
        ax.scatter(data.difference_from_reference_bp, positions, color="#174a7e")
        for position, (_, row) in zip(positions, data.iterrows()):
            if np.isfinite(row.ci_low_bp) and np.isfinite(row.ci_high_bp):
                ax.plot([row.ci_low_bp, row.ci_high_bp], [position, position], color="#174a7e")
        ax.set_yticks(positions, data["group"])
        ax.axvline(0, color="gray", linewidth=1)
        ax.set(title=f"{selected}: out-of-fold residual differences", xlabel="Difference from reference (basis points)")
        fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig)


def analyze(config):
    artifact, output = Path(config["artifact_dir"]), Path(config["report_dir"])
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((artifact / "metadata.json").read_text())
    processed = Path(config["processed_dir"])
    if metadata["processed_sha256"] != fingerprint(processed / "loans.parquet"):
        raise ValueError("Training and processed datasets differ; rerun training.")
    primary = pd.read_parquet(processed / "loans.parquet")
    extended = pd.read_parquet(processed / "eligible_including_outliers.parquet")
    statewide_parts, regressions, sensitivity_parts, yearly_parts, geo_parts, lender_parts, lender_group_parts, area_parts, correlations = [], [], [], [], [], [], [], [], []
    selected_data = None
    for scenario in metadata["completed_scenarios"]:
        for model in ["linear", "xgboost"]:
            log(f"Analyzing {scenario}/{model}")
            path = artifact / f"predictions_{scenario}_{model}.parquet"
            data = joined_predictions(extended if scenario == "including_outliers" else primary, path)
            summaries = []
            for dimension in REFERENCES:
                summaries.append(group_summary(data, dimension, config, inference=scenario == "primary"))
                if scenario == "primary":
                    regressions.extend(dict(row, model=model) for row in residual_regression(data, dimension, config))
            summary = pd.concat(summaries, ignore_index=True)
            summary["model"], summary["scenario"] = model, scenario
            if scenario != "primary":
                sensitivity_parts.append(summary)
                continue
            summary = adjust_pvalues(summary)
            statewide_parts.append(summary)
            for year, subset in data.groupby("activity_year"):
                for dimension in REFERENCES:
                    table = group_summary(subset, dimension, config, inference=False)
                    table["activity_year"], table["model"] = year, model
                    yearly_parts.append(table)
            for dim in ["county_code", "derived_msa_md"]:
                table = descriptive_table(data, ["activity_year", dim])
                table["dimension"], table["model"] = dim, model
                table = table.rename(columns={dim: "geography"})
                geo_parts.append(table)
            table = descriptive_table(data, ["lei"])
            table["model"] = model
            table["adequate_size"] = table.n >= config["min_group_size"]
            lender_parts.append(table)
            for dimension, reference in REFERENCES.items():
                table = descriptive_table(data, ["lei", dimension]).rename(columns={dimension: "group"})
                ref = table[table["group"].eq(reference)][["lei", "n", "mean_residual_bp"]].rename(columns={"n": "reference_n", "mean_residual_bp": "reference_mean_residual_bp"})
                table = table.merge(ref, on="lei", how="left", validate="many_to_one")
                table["difference_from_reference_bp"] = table.mean_residual_bp - table.reference_mean_residual_bp
                table["adequate_size"] = table.n.ge(config["min_group_size"]) & table.reference_n.ge(config["min_group_size"])
                table["dimension"], table["model"] = dimension, model
                lender_group_parts.append(table)
            for column, bins in [("tract_minority_population_percent", [0, 20, 40, 60, 80, 100]), ("tract_to_msa_income_percentage", [0, 50, 80, 100, 120, np.inf])]:
                finite = data[column].notna() & np.isfinite(data[column])
                subset = data[finite].copy()
                subset["area_bin"] = pd.cut(subset[column], bins=bins, include_lowest=True).astype(str)
                table = descriptive_table(subset, ["area_bin"])
                table["dimension"], table["model"] = column, model
                area_parts.append(table)
                if len(subset) > 2 and subset[column].nunique() > 1:
                    rho = float(spearmanr(subset[column], subset.residual_bp).statistic)
                    correlations.append({"model": model, "field": column, "n": len(subset), "spearman_rho": rho if np.isfinite(rho) else None})
            if model == metadata["selected_model"]:
                selected_data = data
    statewide = pd.concat(statewide_parts, ignore_index=True)
    statewide.to_csv(output / "group_comparisons.csv", index=False)
    pd.concat(yearly_parts, ignore_index=True).to_csv(output / "yearly_groups.csv", index=False)
    pd.concat(geo_parts, ignore_index=True).to_csv(output / "geography_summary.csv", index=False)
    pd.concat(lender_parts, ignore_index=True).to_csv(output / "lender_summary.csv", index=False)
    pd.concat(lender_group_parts, ignore_index=True).to_csv(output / "lender_group_comparisons.csv", index=False)
    pd.concat(area_parts, ignore_index=True).to_csv(output / "area_demographics.csv", index=False)
    if sensitivity_parts:
        pd.concat(sensitivity_parts, ignore_index=True).to_csv(output / "sensitivity_groups.csv", index=False)
    else:
        # Prevent a primary-only rerun from leaving an older sensitivity table in place.
        statewide.iloc[:0].to_csv(output / "sensitivity_groups.csv", index=False)
    regression_table = pd.DataFrame(regressions)
    if len(regression_table):
        regression_table = pd.concat([adjust_pvalues(group, "cluster_p", "cluster_q") for _, group in regression_table.groupby("model")], ignore_index=True)
    regression_table.to_csv(output / "residual_regressions.csv", index=False)
    benchmark_report(config, output)
    plot_groups(statewide, metadata["selected_model"], output)
    yearly_error = pd.read_csv(artifact / "model_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, subset in yearly_error.query("scenario == 'primary' and dimension == 'activity_year'").groupby("model"):
        ax.plot(subset["group"].astype(int), subset.mae_bp, marker="o", label=model)
    ax.set(title="Out-of-fold prediction error by year", xlabel="Year", ylabel="MAE (basis points)"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "error_by_year.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    county = selected_data.groupby("county_code").agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), residual=("residual_bp", "mean"), n=("row_id", "size"))
    adequate = county[county.n >= config["min_group_size"]]
    if len(adequate):
        points = ax.scatter(adequate.longitude, adequate.latitude, c=adequate.residual, cmap="coolwarm", s=25)
        fig.colorbar(points, ax=ax, label="Mean residual (basis points)")
    ax.set(title="County pricing residuals (representative locations)", xlabel="Longitude", ylabel="Latitude")
    fig.tight_layout(); fig.savefig(output / "geographic_residuals.png", dpi=160); plt.close(fig)
    disagreement = model_disagreement(statewide)
    disagreement.to_csv(output / "model_disagreement.csv", index=False)
    summary = {"selected_model": metadata["selected_model"], "primary_rows": len(primary), "exploratory_eligible_comparisons": int(statewide.inference_eligible.sum()), "model_direction_disagreements": int(disagreement.direction_disagrees.sum()), "area_correlations": correlations, "inference_note": "Bootstrap conditions on fitted cross-validation models; Welch tests ignore lender clustering. Neither inference method establishes causality. Cross-validation model selection adds uncertainty not captured here."}
    write_json(output / "summary.json", summary)
    write_report(config, metadata, statewide, disagreement, output)
    log(f"Report written to {output / 'report.md'}")
    return summary


def model_disagreement(summary):
    result = summary.pivot(index=["dimension", "group"], columns="model", values="difference_from_reference_bp").reset_index()
    result["direction_disagrees"] = result.linear.notna() & result.xgboost.notna() & ((result.linear * result.xgboost) < 0)
    result["gap_difference_bp"] = result.xgboost - result.linear
    return result


def write_report(config, metadata, groups, disagreement, output):
    audit = metadata["source_audit"]
    metrics = pd.read_csv(Path(config["artifact_dir"]) / "model_metrics.csv")
    overall = metrics.query("scenario == 'primary' and dimension == 'overall'")
    lines = ["# Virginia mortgage pricing analysis", "", f"Analyzed **{audit['primary_rows']:,} loans** from the supplied 2021–2025 file. Selected **{metadata['selected_model']}** by cross-validated MAE, with RMSE as tie-breaker.", "", "The model predicts historical rate spread (APR minus APOR). Positive residuals indicate higher pricing than predicted for the observed financial and geographic inputs. These are disparities relative to historical pricing, not proof of discrimination or a guaranteed fair price.", "", "## Prediction performance", "", "| Model | MAE (bp) | RMSE (bp) |", "|---|---:|---:|"]
    for row in overall.itertuples():
        lines.append(f"| {row.model} | {row.mae_bp:.2f} | {row.rmse_bp:.2f} |")
    lines += ["", "![Error by year](error_by_year.png)", "", "## Demographic pricing differences", "", "Differences below use out-of-fold residuals and the selected model. Reference groups are White, Male, Not Hispanic or Latino, and White | Male. Intervals resample whole lenders and condition on the fitted models; small cells and unavailable demographics are descriptive only.", "", "| Dimension | Group | Loans | Difference (bp) | 95% lender-cluster interval (bp) |", "|---|---|---:|---:|---|"]
    subset = groups[(groups.model == metadata["selected_model"]) & groups.dimension.ne("race_by_sex")]
    for row in subset.itertuples():
        interval = f"[{row.ci_low_bp:.2f}, {row.ci_high_bp:.2f}]" if np.isfinite(row.ci_low_bp) else "Descriptive / reference"
        difference = f"{row.difference_from_reference_bp:.2f}" if np.isfinite(row.difference_from_reference_bp) else "Unavailable"
        lines.append(f"| {row.dimension} | {row.group} | {row.n:,} | {difference} | {interval} |")
    lines += ["", "![Race differences](race_gaps.png)", "", "![Sex differences](sex_gaps.png)", "", f"The two models disagree on the direction of **{int(disagreement.direction_disagrees.sum())}** demographic contrasts, including descriptive contrasts. See `model_disagreement.csv`; compare effect sizes and model error before interpreting small differences.", "", "## Coverage and sensitivity", "", f"Reconciled {audit['input_rows']:,} input rows; {audit['cohort_rows_before_cleaning']:,} satisfy the cohort filters before further cleaning. Exclusive disposition counts:", ""]
    lines += [f"- {reason}: {count:,}" for reason, count in audit["exclusive_row_counts"].items()]
    lines += ["", "Primary lenders by year: " + ", ".join(f"{year}: {count}" for year, count in audit["primary_lenders_by_year"].items()) + ".", "", "2023 has markedly fewer loans and lenders; coverage is not verified against statewide HMDA totals. The source's earlier removal of incomplete rows cannot be reconstructed.", "", "Completed scenarios: " + ", ".join(metadata["completed_scenarios"]) + ". Sensitivity tables preserve natural evaluation frequencies; race weights are calculated using training folds only. The no-geography fit checks how much geographic adjustment absorbs group differences.", "", "Separate temporal check (train 2021–2024, evaluate 2025):"]
    for model, metric in metadata["temporal_2025"].items():
        lines.append(f"- {model}: MAE {metric['mae_bp']:.2f} bp; RMSE {metric['rmse_bp']:.2f} bp.")
    lines += ["", metadata["temporal_note"], "", "## Market benchmark", "", "![Historical APOR](apor_fixed30.png)", "", "Annual APOR summaries average weekly effective observations; they are not loan-weighted. APOR is a mortgage APR benchmark, not a Federal Reserve recommended customer interest rate. Historical loans retain their reported spread because their exact rate-set dates and APRs are unavailable. APOR methodology changed in April 2023.", "", "## Interpretation and limitations", "", "- The available cohort is originated, conventional first-lien, 360-month loans on site-built homes with 1–4 units. Owner occupancy, fixed-rate status, purchase/refinance purpose, and exactly one unit cannot be checked.", "- Credit scores, points/fee detail, exact timing, and other relevant factors are unavailable. Omitting demographics does not make historical predictions unbiased; geography can act as a demographic proxy.", "- This study describes pricing among originated loans and cannot measure approval or denial disparities.", "- Lender tables are descriptive and use LEIs, not verified names. Within-lender differences can reflect borrower/product composition; rankings do not establish lender misconduct.", "- Welch tests are exploratory and ignore lender clustering; their p-values use BH correction within each model across statewide contrasts. Regression p-values use a separate BH family per model. Cluster-bootstrap intervals are pointwise and do not include model-refit or selection uncertainty.", "- Borrower residual percentiles are historical descriptive comparisons, not individual discrimination probabilities or calibrated prediction intervals.", "", "## Outputs and reproducibility", "", "CSV tables cover statewide and yearly groups, intersections, lender and geographic diagnostics, model disagreement, area demographics, and completed sensitivities. Model metrics, configuration, dependency versions, row identifiers/fold assignments, and SHA-256 source fingerprints accompany the artifacts. `data/processed/exclusions.csv` records excluded source rows.", "", "References: [HMDA fields](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/), [APOR tables and methodology](https://ffiec.cfpb.gov/tools/rate-spread), [FFIEC Census geography](https://www.ffiec.gov/data/census/overview).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
