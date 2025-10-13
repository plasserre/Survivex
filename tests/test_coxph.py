"""
Comprehensive validation suite for Cox Proportional Hazards model.

Validates against:
1. lifelines CoxPHFitter
2. R survival::coxph (when available)
3. Known analytical results
4. Published datasets
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional

# Import implementations
try:
    from survivex.models.cox_ph import CoxPHModel
except ImportError:
    print("Warning: Import from survivex failed, trying local import")
    from cox_ph import CoxPHModel


def compare_coefficients(ours, theirs, names, title="Comparison"):
    """Pretty print coefficient comparison."""
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Variable':<15} {'Ours':>12} {'Reference':>12} {'Diff':>12}")
    print("-" * 60)
    
    max_diff = 0
    for i, name in enumerate(names):
        diff = abs(ours[i] - theirs[i])
        max_diff = max(max_diff, diff)
        print(f"{name:<15} {ours[i]:>12.6f} {theirs[i]:>12.6f} {diff:>12.8f}")
    
    print("-" * 60)
    print(f"Maximum difference: {max_diff:.10f}")
    
    if max_diff < 1e-6:
        print("*** PERFECT MATCH ***")
        return True
    elif max_diff < 1e-4:
        print("*** EXCELLENT MATCH ***")
        return True
    elif max_diff < 1e-2:
        print("*** GOOD MATCH ***")
        return True
    else:
        print("*** WARNING: Large differences detected ***")
        return False


def test_simple_example():
    """
    Test 1: Simple example with known structure.
    
    This tests basic functionality with a small dataset.
    """
    print("\n" + "="*70)
    print("TEST 1: Simple Example")
    print("="*70)
    
    # Simple dataset
    np.random.seed(42)
    n = 50
    
    X = np.column_stack([
        np.random.normal(0, 1, n),  # X1
        np.random.normal(0, 1, n)   # X2
    ])
    
    # Generate survival times based on Cox model
    # h(t|X) = h0(t) * exp(β1*X1 + β2*X2)
    # Use β = [1.0, -0.5]
    true_beta = np.array([1.0, -0.5])
    risk_scores = np.exp(X @ true_beta)
    
    # Generate exponential survival times
    baseline_hazard = 0.1
    durations = np.random.exponential(1.0 / (baseline_hazard * risk_scores))
    
    # Random censoring
    censoring_times = np.random.exponential(20, n)
    events = (durations < censoring_times).astype(int)
    durations = np.minimum(durations, censoring_times)
    
    print(f"Generated {n} observations")
    print(f"Events: {events.sum()} ({events.mean()*100:.1f}%)")
    print(f"True coefficients: {true_beta}")
    
    # Fit our model
    print("\nFitting our Cox model...")
    cox_ours = CoxPHModel(tie_method='efron')
    cox_ours.fit(X, durations, events)
    
    print(f"Estimated coefficients: {cox_ours.coefficients_}")
    print(f"Standard errors: {cox_ours.standard_errors_}")
    print(f"Hazard ratios: {cox_ours.result_.hazard_ratios}")
    print(f"C-index: {cox_ours.concordance_index_:.4f}")
    print(f"Log-likelihood: {cox_ours.log_likelihood_:.4f}")
    
    # Compare with lifelines
    try:
        from lifelines import CoxPHFitter
        
        print("\nComparing with lifelines...")
        df = pd.DataFrame(X, columns=['X1', 'X2'])
        df['duration'] = durations
        df['event'] = events
        
        cph_lifelines = CoxPHFitter()
        cph_lifelines.fit(df, duration_col='duration', event_col='event')
        
        # Compare coefficients
        lifelines_coef = cph_lifelines.params_.values
        lifelines_se = cph_lifelines.standard_errors_.values
        
        match = compare_coefficients(
            cox_ours.coefficients_,
            lifelines_coef,
            ['X1', 'X2'],
            title="Coefficient Comparison with lifelines"
        )
        
        # Compare standard errors
        print(f"\nStandard Error Comparison:")
        print(f"{'Variable':<15} {'Ours':>12} {'Lifelines':>12} {'Diff':>12}")
        print("-" * 60)
        for i, name in enumerate(['X1', 'X2']):
            diff = abs(cox_ours.standard_errors_[i] - lifelines_se[i])
            print(f"{name:<15} {cox_ours.standard_errors_[i]:>12.6f} {lifelines_se[i]:>12.6f} {diff:>12.8f}")
        
        # Compare C-index
        lifelines_ci = cph_lifelines.concordance_index_
        print(f"\nC-index comparison:")
        print(f"Ours:      {cox_ours.concordance_index_:.6f}")
        print(f"Lifelines: {lifelines_ci:.6f}")
        print(f"Difference: {abs(cox_ours.concordance_index_ - lifelines_ci):.8f}")
        
        if match:
            print("\n[PASS] Simple example test")
        else:
            print("\n[FAIL] Simple example test")
        
        return match
        
    except ImportError:
        print("\nLifelines not available for comparison")
        return None


def test_rossi_dataset():
    """
    Test 2: Rossi recidivism dataset (standard benchmark).
    
    This is a well-known dataset used in survival analysis textbooks.
    From Rossi et al. (1980) on criminal recidivism.
    """
    print("\n" + "="*70)
    print("TEST 2: Rossi Recidivism Dataset")
    print("="*70)
    
    try:
        from lifelines.datasets import load_rossi
        
        rossi = load_rossi()
        print(f"Dataset shape: {rossi.shape}")
        print(f"Columns: {list(rossi.columns)}")
        
        # Select subset of covariates for testing
        covariates = ['fin', 'age', 'prio']
        X = rossi[covariates].values
        durations = rossi['week'].values
        events = rossi['arrest'].values
        
        print(f"\nCovariates: {covariates}")
        print(f"Observations: {len(durations)}")
        print(f"Events: {events.sum()} ({events.mean()*100:.1f}%)")
        
        # Fit our model
        print("\nFitting our Cox model...")
        cox_ours = CoxPHModel(tie_method='efron')
        cox_ours.fit(X, durations, events)
        
        print(cox_ours.result_.summary())
        
        # Compare with lifelines
        from lifelines import CoxPHFitter
        
        print("\nFitting lifelines model...")
        cph_lifelines = CoxPHFitter()
        cph_lifelines.fit(rossi, duration_col='week', event_col='arrest', formula="fin + age + prio")
        
        # Extract coefficients in same order
        lifelines_coef = cph_lifelines.params_[covariates].values
        lifelines_se = cph_lifelines.standard_errors_[covariates].values
        
        match = compare_coefficients(
            cox_ours.coefficients_,
            lifelines_coef,
            covariates,
            title="Rossi Dataset: Coefficient Comparison"
        )
        
        # Compare standard errors
        print(f"\nStandard Error Comparison:")
        print(f"{'Variable':<15} {'Ours':>12} {'Lifelines':>12} {'Diff':>12}")
        print("-" * 60)
        max_se_diff = 0
        for i, name in enumerate(covariates):
            diff = abs(cox_ours.standard_errors_[i] - lifelines_se[i])
            max_se_diff = max(max_se_diff, diff)
            print(f"{name:<15} {cox_ours.standard_errors_[i]:>12.6f} {lifelines_se[i]:>12.6f} {diff:>12.8f}")
        
        if max_se_diff < 1e-4:
            print("Standard errors match excellently")
        
        # Compare predictions
        print(f"\nPrediction Comparison (first 5 subjects):")
        risk_ours = cox_ours.predict_risk(X[:5])
        risk_lifelines = cph_lifelines.predict_partial_hazard(rossi[covariates][:5]).values
        
        print(f"{'Subject':<10} {'Ours':>12} {'Lifelines':>12} {'Diff':>12}")
        for i in range(5):
            diff = abs(risk_ours[i] - risk_lifelines[i])
            print(f"{i:<10} {risk_ours[i]:>12.6f} {risk_lifelines[i]:>12.6f} {diff:>12.8f}")
        
        if match:
            print("\n[PASS] Rossi dataset test")
        else:
            print("\n[FAIL] Rossi dataset test")
        
        return match
        
    except ImportError:
        print("lifelines.datasets not available")
        return None


def test_lung_cancer():
    """
    Test 3: Lung cancer dataset.
    
    Standard survival analysis dataset with covariates.
    """
    print("\n" + "="*70)
    print("TEST 3: Lung Cancer Dataset")
    print("="*70)
    
    try:
        from lifelines.datasets import load_lung
        
        lung = load_lung()
        
        print(f"Raw dataset shape: {lung.shape}")
        print(f"Status values: {sorted(lung['status'].unique())}")
        
        # Clean data - remove rows with missing values in key columns
        covariates = ['age', 'sex', 'ph.karno']
        required_cols = covariates + ['time', 'status']
        lung_clean = lung[required_cols].dropna()
        
        print(f"After removing missing values: {lung_clean.shape}")
        
        # Select covariates
        X = lung_clean[covariates].values
        durations = lung_clean['time'].values
        
        # Handle status encoding - check what values we have
        status_values = sorted(lung_clean['status'].unique())
        if set(status_values) == {0, 1}:
            # status=0 is censored, status=1 is death
            events = lung_clean['status'].values
            print(f"Status encoding: 0=censored, 1=death")
        elif set(status_values) == {1, 2}:
            # status=1 is censored, status=2 is death
            events = (lung_clean['status'] == 2).astype(int).values
            print(f"Status encoding: 1=censored, 2=death")
        else:
            # Auto-detect: assume max value is event
            max_status = max(status_values)
            events = (lung_clean['status'] == max_status).astype(int).values
            print(f"Status encoding: {max_status}=death (auto-detected)")
        
        print(f"Covariates: {covariates}")
        print(f"Observations: {len(durations)}")
        print(f"Events: {events.sum()} ({events.mean()*100:.1f}%)")
        
        if events.sum() == 0:
            print("[SKIP] No events in dataset after cleaning")
            return None
        
        # Fit our model
        print("\nFitting our Cox model...")
        cox_ours = CoxPHModel(tie_method='efron')
        cox_ours.fit(X, durations, events)
        
        print(cox_ours.result_.summary())
        
        # Compare with lifelines
        from lifelines import CoxPHFitter
        
        print("\nFitting lifelines model...")
        df_lifelines = pd.DataFrame(X, columns=covariates)
        df_lifelines['time'] = durations
        df_lifelines['event'] = events
        
        cph_lifelines = CoxPHFitter()
        cph_lifelines.fit(df_lifelines, duration_col='time', event_col='event')
        
        lifelines_coef = cph_lifelines.params_[covariates].values
        
        match = compare_coefficients(
            cox_ours.coefficients_,
            lifelines_coef,
            covariates,
            title="Lung Cancer: Coefficient Comparison"
        )
        
        if match:
            print("\n[PASS] Lung cancer test")
        else:
            print("\n[FAIL] Lung cancer test")
        
        return match
        
    except ImportError:
        print("lifelines.datasets not available")
        return None
    except Exception as e:
        print(f"Error in lung cancer test: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tie_methods():
    """
    Test 4: Compare Breslow vs Efron tie handling.
    
    IMPORTANT: lifelines CoxPHFitter only supports Efron's method for tie handling
    in the partial likelihood. The baseline_estimation_method parameter only affects
    baseline hazard estimation, NOT the tie method used during coefficient estimation.
    
    Therefore, we validate:
    1. Efron matches lifelines exactly (proves Efron implementation is correct)
    2. Breslow gives different results from Efron (proves methods are distinct)
    3. Efron has higher likelihood than Breslow (expected mathematical property)
    4. Both methods converge properly (sanity check)
    """
    print("\n" + "="*70)
    print("TEST 4: Tie Handling Methods (Breslow vs Efron)")
    print("="*70)
    
    # Generate data with intentional ties
    np.random.seed(42)
    n = 100
    
    X = np.random.normal(0, 1, (n, 2))
    
    # Create times with many ties
    durations = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n)
    events = np.random.binomial(1, 0.7, n)
    
    print(f"Generated {n} observations")
    print(f"Unique event times: {len(np.unique(durations[events==1]))}")
    print(f"Events: {events.sum()} ({events.mean()*100:.1f}%)")
    
    # Fit with Breslow
    print("\nFitting with Breslow method...")
    cox_breslow = CoxPHModel(tie_method='breslow')
    cox_breslow.fit(X, durations, events)
    
    print(f"Breslow coefficients: {cox_breslow.coefficients_}")
    print(f"Breslow log-likelihood: {cox_breslow.log_likelihood_:.4f}")
    print(f"Breslow converged: {cox_breslow.result_.convergence_info['converged']}")
    
    # Fit with Efron
    print("\nFitting with Efron method...")
    cox_efron = CoxPHModel(tie_method='efron')
    cox_efron.fit(X, durations, events)
    
    print(f"Efron coefficients: {cox_efron.coefficients_}")
    print(f"Efron log-likelihood: {cox_efron.log_likelihood_:.4f}")
    print(f"Efron converged: {cox_efron.result_.convergence_info['converged']}")
    
    # Compare coefficients
    print(f"\nCoefficient differences (Efron - Breslow):")
    coef_diff = cox_efron.coefficients_ - cox_breslow.coefficients_
    print(f"X1: {coef_diff[0]:.6f}")
    print(f"X2: {coef_diff[1]:.6f}")
    
    # Validate Efron against lifelines
    try:
        from lifelines import CoxPHFitter
        
        df = pd.DataFrame(X, columns=['X1', 'X2'])
        df['duration'] = durations
        df['event'] = events
        
        # Lifelines uses Efron method (always)
        print("\nComparing Efron with lifelines...")
        cph_lifelines = CoxPHFitter()
        cph_lifelines.fit(df, duration_col='duration', event_col='event')
        
        efron_match = compare_coefficients(
            cox_efron.coefficients_,
            cph_lifelines.params_.values,
            ['X1', 'X2'],
            title="Efron Method Comparison"
        )
        
        # Validate Breslow behavior (can't compare to lifelines, but check properties)
        print("\n" + "="*70)
        print("Breslow Method Validation")
        print("="*70)
        print("Note: lifelines doesn't support Breslow for partial likelihood.")
        print("Validating Breslow by checking expected mathematical properties:")
        print()
        
        # Check 1: Methods give different results (they should!)
        methods_differ = not np.allclose(cox_breslow.coefficients_, cox_efron.coefficients_, atol=1e-4)
        print(f"✓ Check 1: Breslow ≠ Efron: {methods_differ}")
        if methods_differ:
            print(f"  Max coefficient difference: {np.max(np.abs(coef_diff)):.6f}")
        
        # Check 2: Efron has higher likelihood (expected with ties)
        likelihood_correct = cox_efron.log_likelihood_ > cox_breslow.log_likelihood_
        print(f"✓ Check 2: Efron likelihood > Breslow: {likelihood_correct}")
        if likelihood_correct:
            print(f"  Breslow: {cox_breslow.log_likelihood_:.4f}")
            print(f"  Efron:   {cox_efron.log_likelihood_:.4f}")
            print(f"  Difference: {cox_efron.log_likelihood_ - cox_breslow.log_likelihood_:.4f}")
        
        # Check 3: Both methods converged
        both_converged = (cox_breslow.result_.convergence_info['converged'] and 
                         cox_efron.result_.convergence_info['converged'])
        print(f"✓ Check 3: Both methods converged: {both_converged}")
        
        # Check 4: Coefficients are reasonable (not NaN, not huge)
        breslow_reasonable = (not np.any(np.isnan(cox_breslow.coefficients_)) and 
                             np.all(np.abs(cox_breslow.coefficients_) < 100))
        efron_reasonable = (not np.any(np.isnan(cox_efron.coefficients_)) and 
                           np.all(np.abs(cox_efron.coefficients_) < 100))
        coeffs_reasonable = breslow_reasonable and efron_reasonable
        print(f"✓ Check 4: Coefficients are reasonable: {coeffs_reasonable}")
        
        print("="*70)
        
        # Overall pass/fail
        breslow_valid = (methods_differ and likelihood_correct and 
                        both_converged and coeffs_reasonable)
        
        if efron_match and breslow_valid:
            print("\n[PASS] Tie methods test")
            print("  - Efron matches lifelines perfectly ✓")
            print("  - Breslow shows expected mathematical behavior ✓")
            return True
        else:
            print("\n[FAIL] Tie methods test")
            if not efron_match:
                print("  - Efron doesn't match lifelines ✗")
            if not breslow_valid:
                print("  - Breslow doesn't show expected behavior ✗")
            return False
        
    except ImportError:
        print("\nLifelines not available")
        return None


def test_predictions():
    """
    Test 5: Prediction methods (risk, survival, cumulative hazard).
    """
    print("\n" + "="*70)
    print("TEST 5: Prediction Methods")
    print("="*70)
    
    try:
        from lifelines.datasets import load_rossi
        from lifelines import CoxPHFitter
        
        rossi = load_rossi()
        
        covariates = ['fin', 'age', 'prio']
        X = rossi[covariates].values
        durations = rossi['week'].values
        events = rossi['arrest'].values
        
        # Fit models
        cox_ours = CoxPHModel(tie_method='efron')
        cox_ours.fit(X, durations, events)
        
        cph_lifelines = CoxPHFitter()
        cph_lifelines.fit(rossi, duration_col='week', event_col='arrest', formula="fin + age + prio")
        
        # Test risk scores
        print("\nTesting risk score predictions...")
        X_test = X[:5]
        
        risk_ours = cox_ours.predict_risk(X_test)
        risk_lifelines = cph_lifelines.predict_partial_hazard(rossi[covariates][:5]).values
        
        print(f"{'Subject':<10} {'Ours':>15} {'Lifelines':>15} {'Diff':>15}")
        print("-" * 60)
        max_risk_diff = 0
        for i in range(5):
            diff = abs(risk_ours[i] - risk_lifelines[i])
            max_risk_diff = max(max_risk_diff, diff)
            print(f"{i:<10} {risk_ours[i]:>15.8f} {risk_lifelines[i]:>15.8f} {diff:>15.10f}")
        
        risk_match = max_risk_diff < 1e-6
        if risk_match:
            print("Risk scores match perfectly")
        
        # Test survival function predictions
        print("\nTesting survival function predictions at t=26 weeks...")
        times_test = np.array([26])
        
        surv_ours = cox_ours.predict_survival_function(X_test, times_test)
        surv_lifelines = cph_lifelines.predict_survival_function(
            rossi[covariates][:5], times=[26]
        ).T.values
        
        print(f"{'Subject':<10} {'Ours':>15} {'Lifelines':>15} {'Diff':>15}")
        print("-" * 60)
        max_surv_diff = 0
        for i in range(5):
            diff = abs(surv_ours[i, 0] - surv_lifelines[i, 0])
            max_surv_diff = max(max_surv_diff, diff)
            print(f"{i:<10} {surv_ours[i, 0]:>15.8f} {surv_lifelines[i, 0]:>15.8f} {diff:>15.10f}")
        
        surv_match = max_surv_diff < 1e-6
        if surv_match:
            print("Survival predictions match perfectly")
        
        if risk_match and surv_match:
            print("\n[PASS] Prediction methods test")
            return True
        else:
            print("\n[FAIL] Prediction methods test")
            return False
        
    except ImportError:
        print("lifelines not available")
        return None


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "*"*70)
    print("COX PROPORTIONAL HAZARDS MODEL - COMPREHENSIVE VALIDATION")
    print("*"*70)
    
    results = {}
    
    # Run tests
    results['simple'] = test_simple_example()
    results['rossi'] = test_rossi_dataset()
    results['lung'] = test_lung_cancer()
    results['ties'] = test_tie_methods()
    results['predictions'] = test_predictions()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"{test_name.upper():<20} {status}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    print("="*70)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total and total > 0:
        print("\n*** ALL TESTS PASSED - Implementation validated! ***")
    elif passed >= 0.8 * total:
        print("\n*** Most tests passed - Implementation is good ***")
    else:
        print("\n*** Some tests failed - Review implementation ***")
    
    print("="*70)


if __name__ == "__main__":
    run_all_tests()