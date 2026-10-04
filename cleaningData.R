# Cleaning and combining data for 
# machine learning project fall 26

va21 <- read.csv("state_VA21.csv")
va22 <- read.csv("state_VA22.csv")
va23 <- read.csv("state_VA23.csv")
va24 <- read.csv("state_VA24.csv")
va25 <- read.csv("state_VA25.csv")

# the following loop created this file, do we don't need to run it again
vaTotal <- read.csv("totalVA_cleaned.csv")

totalVA <- data.frame()

listOfYears <- list(va21, va22, va23, va24, va25)

# DUCKDB IS POTENTIAL FREE DATABASE FOR BIG STUFF (maybe not, look into it)

# this combines the datasets and only leaves the columns we wanna look at
# for (year in listOfYears) {
#   Sys.sleep(3)
#   cleanedYear <- year %>%
#     select(activity_year, lei, derived_msa.md, county_code,
#            census_tract, conforming_loan_limit, derived_loan_product_type,
#            derived_dwelling_category, action_taken, loan_amount,
#            loan_to_value_ratio, interest_rate, loan_term, income, debt_to_income_ratio,
#            rate_spread, derived_ethnicity, derived_race, derived_sex,
#            tract_minority_population_percent, tract_to_msa_income_percentage,
#            preapproval, derived_dwelling_category, submission_of_application,
#            initially_payable_to_institution)
#   
#   totalVA <- rbind(totalVA, cleanedYear)
# }
# 
# # Now I'm going to clean the rows. 
# # we want single family homes and loans that have been closed
# 
# totalVA <- totalVA %>%
#   filter(derived_dwelling_category == "Single Family (1-4 Units):Site-Built",
#          action_taken == 1)


# to export the new, cleaned csv, use this command in the terminal
# readr::write_csv(totalVA, "totalVA_cleaned.csv")

# this makes the numbers be read as numbers and not characters
vaTotal <- read.csv("totalVA_cleaned.csv", na.strings = c("NA", "", "Exempt"))
# this gets rid of the NA's
vaTotal <- na.omit(vaTotal)
