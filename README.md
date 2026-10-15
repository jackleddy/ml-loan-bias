# Virginia Mortgage Pricing Analysis

CS 4828, Fall 2026 — Jack Leddy, Thomas Henning, Yuzhuo Zhang.

Predict historical **rate spread**, then compare actual-minus-predicted pricing across race, sex, ethnicity, intersections, lenders, years, and geographies. Ordinary linear regression is the baseline; XGBoost provides the nonlinear comparison. Demographics, lender identity, area demographics, and actual interest rate never enter the pricing models.

Rate spread is **APR minus Average Prime Offer Rate (APOR)**, in percentage points. A residual of `0.25` percentage points is **25 basis points** above predicted pricing. APR includes costs that nominal interest rate does not. These results describe historical pricing disparities; they do not prove discrimination or establish a guaranteed fair price.

## Setup and run

Use Python 3.11 or newer. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

The lock file records the tested Python 3.11 Windows environment. To resolve the declared dependency ranges on another supported Python/platform combination:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The first run downloads official Census and CFPB APOR references; it **never downloads additional loan records**. After the references have been cached, require offline operation with:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --offline
```

Each stage also runs independently:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_references.py
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/train_models.py
.\.venv\Scripts\python.exe scripts/analyze_fairness.py
.\.venv\Scripts\python.exe scripts/compare_loan.py examples/borrower.json
```

Every script accepts `--config path/to/config.json`; configuration paths resolve relative to that file. `fetch_references.py --refresh` explicitly refreshes official references. Offline operation fails clearly if a raw cache file is missing.

Default training includes the primary fit and all four sensitivity scenarios, with both models and five folds. A quicker primary-only run is available:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --offline --scenarios primary
```

The report lists the scenarios actually completed. Configure threads, chunk size, bootstrap repetitions, outlier bounds, and XGBoost parameters in `config.json`. Full sensitivity training takes longer than preparation; no GPU or external database is needed.

## Input and cohort

`totalVA_cleaned.csv.gz` is the only loan-record source: **918,502 rows, 24 columns, 24.3 MB compressed / 190.8 MB uncompressed**. It is read directly in chunks and preserved.

The cohort requires Virginia geography, activity years 2021–2025, originated loans (`action_taken == 1`), site-built homes with **1–4 units**, `Conventional:First Lien`, and a 360-month term. Both conforming and nonconforming conventional loans remain included; conforming status does not identify conventional loan type.

The supplied file has **438,865** cohort records before additional cleaning. Two spreads outside the configurable **−10 to +20 percentage-point** bounds are excluded, leaving **438,863** primary records. These bounds are cleaning choices, not regulatory thresholds; the outlier sensitivity includes both records. All eligible tract coordinates matched. Invalid targets, essential financial inputs, and unmatched coordinates are separately reported exclusions.

Public HMDA income is annual income in **thousands of dollars**. Negative/zero incomes are retained using signed-log income; loan amount uses a log transformation. DTI remains a public bucket/category, including a Missing category. Exemption handling is field-specific: `1111` is treated as exempt for CLTV, term, and DTI; 8888 and 9999 are not globally removed because they may be legitimate financial values.

The legacy `loan_to_value_ratio` column is treated as renamed public **combined loan-to-value ratio**. MSA aliases normalize to `derived_msa_md`; ambiguous duplicate aliases fail. Identifiers remain strings. Census TIGER 2017 coordinates are used for 2021 and TIGER 2020 for 2022–2025, matching FFIEC tract vintages. Coordinates are tract internal points, not property locations; county/MSA identifiers are used for reporting.

## Models and analysis

| Role | Fields |
|---|---|
| Predictors | Year, loan amount, income, CLTV, DTI category, conforming status, latitude, longitude |
| Target | Reported rate spread |
| Analysis only | Race, sex, ethnicity, lender LEI, county/tract/MSA, area demographics, actual interest rate |
| Filters / deferred | Product, dwelling category, action, term, preapproval, submission channel, initially payable status |

Preprocessing is fitted within each training fold. The linear model standardizes numeric inputs; both models one-hot encode year, DTI, and conforming status. XGBoost uses 400 CPU histogram trees, depth 4, learning rate 0.05, minimum child weight 10, and 0.8 row/column sampling by default.

Five-fold splits are stratified by year with a fixed seed. Every research residual comes from a model that excluded that record. Selection uses overall cross-validated MAE, then RMSE; both models' demographic results remain available. The selected model is refitted on all primary loans for borrower comparisons.

A separate temporal diagnostic trains on 2021–2024 and evaluates on 2025. The unseen 2025 category has zero one-hot year indicators in that diagnostic; it evaluates extrapolation separately from the retrospective study.

Sensitivity fits exclude 2023, exclude geography, include target outliers, or apply race-balanced training weights. Weights are calculated inside each training fold: equal total weight per reported known-race category before a cap of 10. Joint remains a separate reported category; unavailable/free-form race keeps weight 1. Evaluation populations keep their natural frequencies. Balancing does not establish fairness.

Reports include group counts, actual/predicted means, residual means/medians, reference differences, lender-cluster bootstrap confidence intervals, year-adjusted residual regressions with lender-clustered standard errors, and exploratory Welch tests. Reference groups are White, Male, Not Hispanic or Latino, and White | Male for intersections.

Statewide inference requires **100 loans and 20 lenders in each compared group**, including its reference. Small cells, unavailable/free-form demographics, and lender rankings are descriptive. Lender tables expose `adequate_size` flags. BH correction is applied across statewide Welch comparisons per model, with a separate family for regression p-values. Welch tests ignore clustering. Bootstrap intervals are pointwise, condition on fitted models, and exclude model-refit/selection uncertainty.

Annual, lender, county/MSA, intersectional, area-demographic, sensitivity, and model-disagreement tables accompany the main report. APOR charts show actual historical weekly effective benchmarks; annual summaries average weekly observations, not loans. Historical loans retain reported spreads because exact rate-set dates and APRs are unavailable.

## Borrower comparison

```powershell
.\.venv\Scripts\python.exe scripts/compare_loan.py examples/borrower.json --output reports/borrower_result.json
.\.venv\Scripts\python.exe scripts/compare_loan.py examples/borrower_apr.json
```

| JSON field | Meaning |
|---|---|
| `year` | Historical reporting year, 2021–2025 |
| `loan_amount` | Dollars |
| `annual_income` | Annual dollars, converted to HMDA thousands |
| `cltv_percent` | Combined loan-to-value percentage, e.g. `80` |
| `dti` | Exact percentage or public HMDA category; exact values map to public buckets |
| `conforming_status` | `C`, `NC`, `U` (not applicable), or `Missing` |
| `latitude`, `longitude` | Virginia location, preferably tract representative coordinates |
| `census_tract` | Alternative to coordinates; string identifier resolved with the year's geography |

Demographics and lender identity are unnecessary. The caller must establish that the loan satisfies the study cohort. Coordinates receive a Virginia bounding-box check, not a point-in-state boundary check.

Optionally provide exactly one actual-pricing input mode:

- `actual_rate_spread`, in percentage points.
- `apr` plus the **applicable** `apor`, both percentages.
- `apr`, `rate_set_date`, `fixed_rate: true`, and `loan_term_months: 360` for automatic fixed 30-year APOR lookup. Effective-week selection supports 2020–2025 historical rate-set dates, including prior-year locks.

Without actual pricing, the CLI returns expected spread only. Otherwise it adds actual spread, residual basis points, and the residual percentile among that year's out-of-fold residuals. Percentiles compare an all-data fitted prediction with an out-of-fold historical reference: they are descriptive, not calibrated probabilities, intervals, or discrimination scores. Unsupported years fail; inputs outside training ranges generate warnings. Nominal interest rate is never substituted for APR or reconstructed from predicted spread.

## Files and validation

| Location | Contents |
|---|---|
| `loanfair/`, `scripts/` | Shared implementation and six executable commands |
| `data/reference/` | Raw official downloads, normalized tables, source manifest |
| `data/processed/` | Primary/outlier-inclusive Parquet, exclusions CSV, coverage CSV, audit JSON |
| `artifacts/` | Fold predictions, metrics, temporal predictions, saved model, metadata |
| `reports/` | `report.md`, tables, figures, machine-readable summary |
| `examples/`, `tests/` | Borrower JSON examples and synthetic tests |

Generated datasets, caches, models, and reports are ignored by Git. Stable `row_id` is the zero-based position in the source file. Every row receives one exclusive disposition. SHA-256 source/processed/model fingerprints, configuration, and dependency versions accompany outputs. Load joblib artifacts only from trusted sources. `cleaningData.R` is preserved as legacy work and is not invoked.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests run without network and cover filters/reconciliation, aliases, units, missing codes, geography vintages, APOR weeks, offline caching, feature leakage, fold isolation, known residual gaps, small cells, p-value correction, all sensitivity fits, serialization, and CLI comparisons.

## Limits and sources

2023 has unusually sparse coverage: **19,404 input records, 7,467 cohort loans, and 91 cohort lenders**. Statewide completeness is unverified. The legacy R script removed incomplete records; their counts and demographic composition cannot be reconstructed.

Owner occupancy, fixed-rate status, purchase versus refinancing, exactly one unit, credit scores, precise timing, and other relevant factors cannot be verified from these columns. Public financial values are rounded/bucketed. Geography can absorb demographic differences, motivating the no-geography sensitivity. This study concerns pricing among originated loans and cannot measure approval/denial disparities, current quotes, or other states.

APOR is a comparable mortgage APR benchmark, **not a Federal Reserve recommended customer interest rate**. No loan quarter or weekly benchmark is invented. The official APOR methodology changed in April 2023.

- [HMDA field definitions](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/)
- [HMDA Data Browser API](https://ffiec.cfpb.gov/documentation/api/data-browser/)
- [CFPB rate-spread definition, downloads, and methodology](https://ffiec.cfpb.gov/tools/rate-spread)
- [FFIEC Census geography vintages](https://www.ffiec.gov/data/census/overview)
- Bartlett et al., [Consumer-Lending Discrimination in the FinTech Era](https://doi.org/10.1016/j.jfineco.2021.05.047)
- Fuster et al., [Predictably Unequal?](https://doi.org/10.1111/jofi.13090)
