
# R validation script for Aalen-Johansen vs Kaplan-Meier
library(survival)

# Load lung data
data(lung)
lung_clean <- lung[complete.cases(lung[, c("time", "status")]), ]

# Kaplan-Meier estimate
km_fit <- survfit(Surv(time, status) ~ 1, data=lung_clean)

# Print results
cat("\n=== Kaplan-Meier Results ===\n")
print(summary(km_fit))

# Save to CSV for comparison
km_df <- data.frame(
  time = km_fit$time,
  survival = km_fit$surv,
  n.risk = km_fit$n.risk,
  n.event = km_fit$n.event
)

write.csv(km_df, "r_kaplan_meier_lung.csv", row.names=FALSE)
cat("\nSaved KM results to: r_kaplan_meier_lung.csv\n")

# Also print first 20 time points
cat("\n=== First 20 Time Points ===\n")
print(head(km_df, 20))

cat("\n=== Last 10 Time Points ===\n")
print(tail(km_df, 10))
