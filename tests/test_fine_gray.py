"""
Test ONLY the censoring distribution - match R exactly
"""

import numpy as np
import torch
import subprocess


def test_censoring_only():
    """Test only censoring distribution calculation."""
    
    print("\n" + "="*70)
    print("TESTING: CENSORING DISTRIBUTION G(t) ONLY")
    print("="*70)
    
    # Simple data
    np.random.seed(42)
    n = 50
    
    time = np.random.exponential(scale=200, size=n)
    status = np.random.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
    
    print(f"\nData: n={n}")
    print(f"Censored: {np.sum(status==0)}")
    print(f"Event 1:  {np.sum(status==1)}")
    print(f"Event 2:  {np.sum(status==2)}")
    
    # ====================================================================
    # PYTHON: Compute G(t)
    # ====================================================================
    from survivex.models.kaplan_meier import KaplanMeierEstimatorWith100Points
    
    censoring_indicator = (status == 0).astype(float)
    km = KaplanMeierEstimatorWith100Points(device='cpu')
    km.fit(torch.from_numpy(time).double(), torch.from_numpy(censoring_indicator).double())
    
    python_times = km.timeline_.numpy()
    python_surv = km.survival_function_.numpy()
    
    print(f"\n" + "-"*70)
    print("PYTHON Results:")
    print(f"-"*70)
    print(f"Number of times: {len(python_times)}")
    print(f"Dtype: {km.timeline_.dtype}")
    print(f"First 10 times:\n  {python_times[:10]}")
    print(f"First 10 G(t):\n  {python_surv[:10]}")
    
    # ====================================================================
    # R: Compute G(t)
    # ====================================================================
    
    # Save data with FULL PRECISION
    with open('censoring_test.csv', 'w') as f:
        f.write("time,status\n")
        for i in range(n):
            f.write(f"{time[i]:.17g},{int(status[i])}\n")  # Full float64 precision
    
    r_code = """
suppressMessages(library(survival))

data <- read.csv("censoring_test.csv")

# Censoring indicator: 1 if censored, 0 otherwise
cens_indicator <- ifelse(data$status == 0, 1, 0)

# Fit KM to censoring
km <- survfit(Surv(data$time, cens_indicator) ~ 1)

cat("===TIMES===\\n")
cat(paste(km$time, collapse=","), "\\n")

cat("===SURVIVAL===\\n")
cat(paste(km$surv, collapse=","), "\\n")

cat("===N_RISK===\\n")
cat(paste(km$n.risk, collapse=","), "\\n")

cat("===N_EVENT===\\n")
cat(paste(km$n.event, collapse=","), "\\n")
"""
    
    result = subprocess.run(['R', '--vanilla', '--slave'],
                          input=r_code,
                          capture_output=True,
                          text=True,
                          timeout=30)
    
    if result.returncode != 0:
        print(f"R Error: {result.stderr}")
        return
    
    # Parse R output
    results = {}
    current_section = None
    
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line.startswith('===') and line.endswith('==='):
            current_section = line.strip('=')
            results[current_section] = []
        elif current_section and line:
            results[current_section].append(line)
    
    r_times = np.array([float(x) for x in results['TIMES'][0].split(',')])
    r_surv = np.array([float(x) for x in results['SURVIVAL'][0].split(',')])
    r_n_risk = np.array([int(x) for x in results['N_RISK'][0].split(',')])
    r_n_event = np.array([int(x) for x in results['N_EVENT'][0].split(',')])
    
    print(f"\n" + "-"*70)
    print("R Results:")
    print(f"-"*70)
    print(f"Number of times: {len(r_times)}")
    print(f"First 10 times:\n  {r_times[:10]}")
    print(f"First 10 G(t):\n  {r_surv[:10]}")
    print(f"First 10 n.risk:\n  {r_n_risk[:10]}")
    print(f"First 10 n.event:\n  {r_n_event[:10]}")
    
    # ====================================================================
    # COMPARE
    # ====================================================================
    
    print(f"\n" + "="*70)
    print("COMPARISON:")
    print(f"="*70)
    
    print(f"\nNumber of time points:")
    print(f"  Python: {len(python_times)}")
    print(f"  R:      {len(r_times)}")
    
    if len(python_times) != len(r_times):
        print(f"  ✗ DIFFERENT NUMBER OF TIMES!")
        return False
    
    # Compare times
    time_diff = np.max(np.abs(python_times - r_times))
    print(f"\nTime values:")
    print(f"  Max difference: {time_diff:.2e}")
    
    if time_diff > 1e-10:  # Tighter tolerance now
        print(f"  ✗ Times differ!")
        print(f"\n  Showing first 5 differences:")
        for i in range(min(5, len(python_times))):
            diff = abs(python_times[i] - r_times[i])
            print(f"    [{i}] Python: {python_times[i]:.15f}, R: {r_times[i]:.15f}, diff: {diff:.2e}")
        return False
    else:
        print(f"  ✓ Times match perfectly!")
    
    # Compare survival
    surv_diff = np.max(np.abs(python_surv - r_surv))
    print(f"\nSurvival values:")
    print(f"  Max difference: {surv_diff:.2e}")
    
    if surv_diff > 1e-10:
        print(f"  ✗ Survival values differ!")
        print(f"\n  Showing first 5 differences:")
        for i in range(min(5, len(python_surv))):
            diff = abs(python_surv[i] - r_surv[i])
            print(f"    [{i}] Python: {python_surv[i]:.15f}, R: {r_surv[i]:.15f}, diff: {diff:.2e}")
        return False
    else:
        print(f"  ✓ Survival values match perfectly!")
    
    print(f"\n" + "="*70)
    print("✓✓✓ CENSORING DISTRIBUTION MATCHES R EXACTLY! ✓✓✓")
    print("="*70)
    return True


if __name__ == "__main__":
    success = test_censoring_only()
    if success:
        print("\n🎉 STEP 1 COMPLETE - Ready for Step 2!")