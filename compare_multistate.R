
# R script for multi-state Aalen-Johansen comparison
library(survival)
library(mstate)

# Load lung data
data(lung)

# Simple 2-state model (Alive -> Dead)
# status: 1=censored, 2=dead
lung_clean <- lung[!is.na(lung$time) & !is.na(lung$status), ]

# Fit survival (which uses Aalen-Johansen for multi-state)
fit_simple <- survfit(Surv(time, status) ~ 1, data=lung_clean)

# Print survival probabilities
print("Kaplan-Meier / 2-state Aalen-Johansen:")
print(summary(fit_simple, times=c(50, 100, 200, 300, 400, 500)))

# For multi-state, we need to create explicit states
# Create a simple competing risks example with lung data
# Let's use sex as competing risks: male death vs female death (illustrative)

# Actually, for true multi-state, let's use the mstate package example
# Load myeloid data from survival package
data(mgus, package="survival")

# Create illness-death model from mgus data
# States: 0=Alive without PCM, 1=PCM (plasma cell malignancy), 2=Death

# This is a simple example - in practice you'd format the data properly
print("\n=== For true illness-death model, see mstate vignettes ===")
    