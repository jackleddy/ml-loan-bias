Measuring Inequality and Bias in Mortgage Pricing

CS 4828 Project | Fall 2026

Team: Jack Leddy, Thomas Henning, Yuzhuo Zhang

Overview

This project uses machine learning to measure potential inequality in mortgage pricing. We train a model to predict the expected interest rate on a mortgage using only financial, loan, geographic, and time-related features. Race, ethnicity, sex, and lender identity are not used to train the main model.

For each loan, we compute a pricing residual:

e_i = r_i - r_hat_i

where r_i is the actual interest rate and r_hat_i is the predicted rate. A positive residual means the borrower paid more than expected for a similar loan. We then compare residuals across demographic groups and lenders to look for systematic differences.

These results would not prove discrimination. Historical market data may already contain bias, so the goal is to flag patterns that deserve further investigation.

Data
Source: Home Mortgage Disclosure Act (HMDA) Loan/Application Register data, from the FFIEC HMDA Data Browser (https://ffiec.cfpb.gov/data-browser/)
Scope: Virginia, 2021-2025 (one file per year)
Raw data is not stored in this repo. See "Data Access" below.
Cleaning Filters
Loans originated only (action_taken == 1)
Single-family, site-built homes (derived_dwelling_category == "Single Family (1-4 Units):Site-Built")
Planned (per the proposal): first-lien, owner-occupied, conventional, 30-year fixed loans
Notes on the Raw Data
NA appears as a literal string in many columns
Some fields (e.g. debt-to-income ratio, applicant age) are bucketed categories, not numbers
Sentinel codes such as 8888 and 9999 mean "exempt" or "not applicable," not real values
Repository Structure
data/raw/ - untouched HMDA downloads (not tracked in git)
data/processed/ - cleaned, analysis-ready dataset
scripts/ - R cleaning and combining scripts
models/ - model training code
analysis/ - residual and fairness analysis
README.md - this file
Data Access

TODO: Team to decide and document one of the following:

Git LFS for the cleaned dataset only, with raw files downloaded locally by each member
Shared cloud storage (e.g. Google Drive) with the link posted here

Keep folder names identical across machines so file paths in scripts match.

Methods
Models: linear regression, random forest, neural network
Model selection: cross-validated MAE and RMSE
Macro controls: rate-lock quarter and the 10-year Treasury yield
Fairness analysis: t-tests and residual regressions across demographic groups and lenders, checked for robustness across all three models
Team Roles
Member	Role
Jack Leddy	Model development, programming
Thomas Henning	Data preprocessing, model development
Yuzhuo Zhang	Feature, residual, and fairness analysis
Timeline
Weeks	Phase
1-2	Data collection and cleaning
2-3	Feature engineering
4-6	Model development
7-8	Residual and fairness analysis
9-10	Evaluation and write-up (one buffer week before Nov 25)
References
Bartlett, R. P., et al. "Consumer-Lending Discrimination in the FinTech Era." Journal of Financial Economics, 143(1), 2022, 30-56. https://doi.org/10.1016/j.jfineco.2021.05.047
Fuster, A., et al. "Predictably Unequal? The Effects of Machine Learning on Credit Markets." The Journal of Finance, 77(1), 2022, 5-47. https://doi.org/10.1111/jofi.13090
Das, S., Stanton, R., and Wallace, N. "Algorithmic Fairness." Annual Review of Financial Economics, 15, 2023, 565-593. https://doi.org/10.1146/annurev-financial-110921-125930
