# File: test_msm_cox_minimal_fixed.py

"""
FIXED Minimal Multi-State Cox Validation
Creates data in the EXACT format R expects (long format, not wide format)
"""

import numpy as np
import pandas as pd
from survivex.models.multi_state import *
from survivex.models.multi_state import MultiStateCoxPH


def test_minimal_competing_risks_fixed():
    """
    Minimal competing risks test - FIXED for R compatibility.
    """
    print("\n" + "="*80)
    print("MINIMAL COMPETING RISKS TEST (FIXED FOR R)")
    print("="*80)
    
    # Use fixed seed
    np.random.seed(123)
    n = 100
    
    print(f"\nSetup: n = {n}, 2 competing events, 2 covariates")
    
    # Simple covariates
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X = np.column_stack([X1, X2])
    
    # True parameters
    beta1_X1 = 0.5
    beta1_X2 = -0.3
    beta2_X1 = -0.4
    beta2_X2 = 0.6
    
    print(f"\nTrue parameters:")
    print(f"  Event 1: β_X1={beta1_X1}, β_X2={beta1_X2}")
    print(f"  Event 2: β_X1={beta2_X1}, β_X2={beta2_X2}")
    
    # Simulate competing risks
    lambda1 = 0.05
    lambda2 = 0.03
    
    durations = np.zeros(n)
    events = np.zeros(n, dtype=int)
    
    for i in range(n):
        h1 = lambda1 * np.exp(beta1_X1 * X1[i] + beta1_X2 * X2[i])
        h2 = lambda2 * np.exp(beta2_X1 * X1[i] + beta2_X2 * X2[i])
        
        t1 = np.random.exponential(1 / h1)
        t2 = np.random.exponential(1 / h2)
        
        if t1 < t2:
            durations[i] = t1
            events[i] = 1
        else:
            durations[i] = t2
            events[i] = 2
    
    # Censoring
    censoring_time = np.random.uniform(0, np.percentile(durations, 70), n)
    censored = durations > censoring_time
    durations[censored] = censoring_time[censored]
    events[censored] = 0
    
    print(f"\nEvent distribution:")
    print(f"  Event 1: {np.sum(events == 1)}")
    print(f"  Event 2: {np.sum(events == 2)}")
    print(f"  Censored: {np.sum(events == 0)}")
    
    # Create transition matrix
    trans_matrix = create_competing_risks_matrix(
        n_competing_events=2,
        state_names=['Alive', 'Event1', 'Event2']
    )
    
    # Prepare multi-state data (Python format)
    ms_data = prepare_multistate_data_simple(
        durations=durations,
        events=events,
        trans_matrix=trans_matrix,
        covariates=X
    )
    
    # Fit Python model
    print("\n" + "="*80)
    print("PYTHON RESULTS")
    print("="*80)
    
    mscox = MultiStateCoxPH(trans_matrix, tie_method='efron')
    result = mscox.fit(ms_data, covariate_names=['X1', 'X2'])
    
    print("\n" + result.summary())
    
    # Print formatted results
    print("\n" + "="*80)
    print("PYTHON RESULTS (formatted)")
    print("="*80)
    
    for trans_num in sorted(result.transition_results.keys()):
        tres = result.transition_results[trans_num]
        print(f"\n--- Transition {tres.transition_number}: {tres.transition_name} ---")
        print(f"Events: {tres.n_events}")
        print(f"Coefficients:  X1={tres.coefficients[0]:.10f}  X2={tres.coefficients[1]:.10f}")
        print(f"Std Errors:    X1={tres.standard_errors[0]:.10f}  X2={tres.standard_errors[1]:.10f}")
        print(f"Log-Likelihood: {tres.log_likelihood:.10f}")
    
    # Export data in R LONG FORMAT (not wide format for msprep)
    # Create the exact format R expects after msprep
    
    # For each subject, create rows for each possible transition
    long_data = []
    
    for i in range(n):
        # Transition 1: Alive -> Event1
        long_data.append({
            'id': i + 1,
            'from': 1,  # R uses 1-indexing
            'to': 2,
            'trans': 1,
            'Tstart': 0.0,
            'Tstop': durations[i],
            'status': 1 if events[i] == 1 else 0,
            'X1': X1[i],
            'X2': X2[i]
        })
        
        # Transition 2: Alive -> Event2
        long_data.append({
            'id': i + 1,
            'from': 1,
            'to': 3,
            'trans': 2,
            'Tstart': 0.0,
            'Tstop': durations[i],
            'status': 1 if events[i] == 2 else 0,
            'X1': X1[i],
            'X2': X2[i]
        })
    
    long_df = pd.DataFrame(long_data)
    
    csv_file = 'minimal_test_long.csv'
    long_df.to_csv(csv_file, index=False)
    
    print(f"\n✓ Data saved to: {csv_file} (R long format)")
    
    # Generate R script that uses the long format directly
    r_script = """
# Minimal Multi-State Cox Validation - FIXED
# Uses pre-formatted long data (bypasses msprep)
library(survival)

# Load pre-formatted long data
data <- read.csv("minimal_test_long.csv")

cat("Data loaded:", nrow(data), "rows\\n")
cat("Unique subjects:", length(unique(data$id)), "\\n")

# Check event counts by transition
cat("\\nEvent counts by transition:\\n")
for (trans in 1:2) {
  trans_data <- data[data$trans == trans, ]
  n_events <- sum(trans_data$status)
  cat(sprintf("  Transition %d: %d events\\n", trans, n_events))
}

cat("\\n")
cat("================================================================================\\n")
cat("R RESULTS\\n")
cat("================================================================================\\n")

# Fit Cox models for each transition
for (trans in 1:2) {
  cat(sprintf("\\n--- Transition %d ---\\n", trans))
  
  trans_data <- data[data$trans == trans, ]
  cat(sprintf("Events: %d\\n", sum(trans_data$status)))
  
  # Fit Cox model
  fit <- coxph(
    Surv(Tstart, Tstop, status) ~ X1 + X2,
    data = trans_data,
    method = "efron"
  )
  
  cat(sprintf("Coefficients:  X1=%.10f  X2=%.10f\\n", 
              coef(fit)[1], coef(fit)[2]))
  cat(sprintf("Std Errors:    X1=%.10f  X2=%.10f\\n", 
              sqrt(diag(vcov(fit)))[1], sqrt(diag(vcov(fit)))[2]))
  cat(sprintf("Log-Likelihood: %.10f\\n", fit$loglik[2]))
}

cat("\\n================================================================================\\n")
cat("COMPARISON: Check that Python and R values match to 8+ decimal places\\n")
cat("================================================================================\\n")
"""
    
    r_file = 'minimal_test_long.R'
    with open(r_file, 'w') as f:
        f.write(r_script)
    
    print(f"✓ R script saved to: {r_file}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\nRun R script: source('{r_file}')")
    print("\nThis bypasses msprep and uses the exact long format R expects.")
    print("The event counts should now match perfectly!")
    
    return result


if __name__ == "__main__":
    print("="*80)
    print("FIXED MINIMAL MULTI-STATE COX VALIDATION")
    print("="*80)
    
    result = test_minimal_competing_risks_fixed()
    
    print("\n" + "="*80)
    print("READY FOR R COMPARISON")
    print("="*80)