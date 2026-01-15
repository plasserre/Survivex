
# Minimal Multi-State Cox Validation - FIXED
# Uses pre-formatted long data (bypasses msprep)
library(survival)

# Load pre-formatted long data
data <- read.csv("minimal_test_long.csv")

cat("Data loaded:", nrow(data), "rows\n")
cat("Unique subjects:", length(unique(data$id)), "\n")

# Check event counts by transition
cat("\nEvent counts by transition:\n")
for (trans in 1:2) {
  trans_data <- data[data$trans == trans, ]
  n_events <- sum(trans_data$status)
  cat(sprintf("  Transition %d: %d events\n", trans, n_events))
}

cat("\n")
cat("================================================================================\n")
cat("R RESULTS\n")
cat("================================================================================\n")

# Fit Cox models for each transition
for (trans in 1:2) {
  cat(sprintf("\n--- Transition %d ---\n", trans))
  
  trans_data <- data[data$trans == trans, ]
  cat(sprintf("Events: %d\n", sum(trans_data$status)))
  
  # Fit Cox model
  fit <- coxph(
    Surv(Tstart, Tstop, status) ~ X1 + X2,
    data = trans_data,
    method = "efron"
  )
  
  cat(sprintf("Coefficients:  X1=%.10f  X2=%.10f\n", 
              coef(fit)[1], coef(fit)[2]))
  cat(sprintf("Std Errors:    X1=%.10f  X2=%.10f\n", 
              sqrt(diag(vcov(fit)))[1], sqrt(diag(vcov(fit)))[2]))
  cat(sprintf("Log-Likelihood: %.10f\n", fit$loglik[2]))
}

cat("\n================================================================================\n")
cat("COMPARISON: Check that Python and R values match to 8+ decimal places\n")
cat("================================================================================\n")
