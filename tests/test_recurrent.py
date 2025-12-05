"""
Comprehensive Validation Tests for Recurrent Event Models

This file provides test cases that validate Python implementations against 
R's survival package. Tests cover:

1. Andersen-Gill (AG) model
2. PWP-TT (Total Time) model  
3. PWP-GT (Gap Time) model
4. Frailty models (Gamma and Gaussian)

FIXED VERSION - Works with local project structure
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os

# Add project to path - adjust this to your actual project location
# The test assumes it's in survivex/tests/ directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import from the actual project files
try:
    from survivex.models.andersen_gill import AndersenGillModel
except ImportError:
    print("WARNING: andersen_gill.py not found - using uploaded version")
    AndersenGillModel = None

try:
    from survivex.models.recurrent_event import PWPTTModel
except ImportError:
    print("WARNING: PWPTTModel not found in recurrent_event.py")
    PWPTTModel = None

try:
    from survivex.models.pwp import PWPGTModel
except ImportError:
    print("WARNING: pwp_gt.py not found - need to add to project")
    PWPGTModel = None

try:
    from survivex.models.frailty import FrailtyModel
except ImportError:
    print("WARNING: frailty_models.py not found - need to add to project")
    FrailtyModel = None

try:
    from survivex.models.cox_ph import CoxPHModel
except ImportError:
    print("WARNING: frailty_models.py not found - need to add to project")
    FrailtyModel = None


def test_andersen_gill_cgd():
    """
    Test Andersen-Gill model using CGD (Chronic Granulomatous Disease) dataset.
    
    R Code to generate reference values:
    ------------------------------------
    library(survival)
    
    # Load CGD data
    data(cgd)
    
    # Fit Andersen-Gill model
    ag_fit <- coxph(Surv(tstart, tstop, status) ~ treat + age, 
                     data=cgd, 
                     id=id,
                     method="efron")
    
    print(summary(ag_fit))
    """
    print("\n" + "="*80)
    print("TEST: Andersen-Gill Model - CGD Dataset")
    print("="*80)
    
    if AndersenGillModel is None:
        print("SKIPPED: AndersenGillModel not available")
        return None
    
    # Simulated CGD-like data
    np.random.seed(42)
    n_subjects = 50
    
    subject_ids = []
    event_times = []
    event_status = []
    treat_vals = []
    age_vals = []
    
    for i in range(n_subjects):
        n_events = np.random.randint(0, 5)
        treat = np.random.binomial(1, 0.5)
        age = np.random.normal(10, 5)
        
        times = []
        status = []
        t = 0
        for j in range(n_events):
            gap = np.random.exponential(100)
            t += gap
            if t > 400:
                status.append(0)
                times.append(400)
                break
            else:
                status.append(1)
                times.append(t)
        
        if len(times) == 0:
            times = [400]
            status = [0]
        
        subject_ids.append(i)
        event_times.append(np.array(times))
        event_status.append(np.array(status))
        treat_vals.append(treat)
        age_vals.append(age)
    
    subject_ids = np.array(subject_ids)
    covariates = np.column_stack([treat_vals, age_vals])
    
    # Fit Andersen-Gill model
    ag_model = AndersenGillModel(tie_method='efron')
    ag_model.fit_simple(
        subject_ids=subject_ids,
        event_times=event_times,
        event_status=event_status,
        covariates=covariates
    )
    
    print(ag_model.result_.summary())
    
    print("\n" + "-"*80)
    print("Python Implementation Results:")
    print("-"*80)
    print(f"Coefficients: {ag_model.coefficients_}")
    print(f"Robust SE: {ag_model.standard_errors_}")
    print(f"Naive SE: {ag_model.naive_standard_errors_}")
    print(f"Log-likelihood: {ag_model.log_likelihood_}")
    
    print("\n✓ Test completed successfully")
    return ag_model


def test_pwp_tt_bladder():
    """Test PWP-TT model using bladder cancer dataset."""
    print("\n" + "="*80)
    print("TEST: PWP-TT Model - Bladder Cancer Dataset")
    print("="*80)
    
    if PWPTTModel is None:
        print("SKIPPED: PWPTTModel not available")
        return None
    
    # Simulated bladder-like data
    np.random.seed(123)
    n_subjects = 40
    
    subject_ids = []
    event_times = []
    event_status = []
    treat_vals = []
    size_vals = []
    
    for i in range(n_subjects):
        n_events = np.random.randint(0, 4)
        treat = np.random.choice([0, 1, 2])
        size = np.random.normal(2, 1)
        
        times = []
        status = []
        for j in range(n_events):
            t = np.random.uniform(0, 50) + j * 10
            if t > 50:
                times.append(50)
                status.append(0)
                break
            times.append(t)
            status.append(1)
        
        if len(times) == 0:
            times = [50]
            status = [0]
        
        subject_ids.append(i)
        event_times.append(np.array(sorted(times)))
        event_status.append(np.array(status))
        treat_vals.append(treat)
        size_vals.append(size)
    
    subject_ids = np.array(subject_ids)
    covariates = np.column_stack([treat_vals, size_vals])
    
    # Fit PWP-TT model
    pwptt_model = PWPTTModel(tie_method='efron')
    pwptt_model.fit_simple(
        subject_ids=subject_ids,
        event_times=event_times,
        event_status=event_status,
        covariates=covariates
    )
    
    print(pwptt_model.result_.summary())
    print("\n✓ Test completed successfully")
    
    return pwptt_model


def test_pwp_gt():
    """Test PWP-GT (Gap Time) model."""
    print("\n" + "="*80)
    print("TEST: PWP-GT Model - Gap Time")
    print("="*80)
    
    if PWPGTModel is None:
        print("SKIPPED: PWPGTModel not available")
        print("NOTE: Copy pwp_gt.py to your project directory")
        return None
    
    # Simulated data
    np.random.seed(456)
    n_subjects = 30
    
    subject_ids = []
    event_times = []
    event_status = []
    treat_vals = []
    
    for i in range(n_subjects):
        n_events = np.random.randint(1, 5)
        treat = np.random.binomial(1, 0.5)
        
        times = []
        status = []
        t = 0
        for j in range(n_events):
            gap = np.random.exponential(20)
            t += gap
            if t > 100:
                times.append(t)
                status.append(0)
                break
            times.append(t)
            status.append(1)
        
        subject_ids.append(i)
        event_times.append(np.array(times))
        event_status.append(np.array(status))
        treat_vals.append(treat)
    
    subject_ids = np.array(subject_ids)
    covariates = np.column_stack([treat_vals])
    
    # Fit PWP-GT model
    pwpgt_model = PWPGTModel(tie_method='efron')
    pwpgt_model.fit_simple(
        subject_ids=subject_ids,
        event_times=event_times,
        event_status=event_status,
        covariates=covariates
    )
    
    print(pwpgt_model.result_.summary())
    print("\n✓ Test completed successfully")
    
    return pwpgt_model


def test_frailty_gamma():
    """Test Gamma frailty model."""
    print("\n" + "="*80)
    print("TEST: Gamma Frailty Model")
    print("="*80)
    
    if FrailtyModel is None:
        print("SKIPPED: FrailtyModel not available")
        print("NOTE: Copy frailty_models.py to your project directory")
        return None
    
    # Simulated data with true frailty structure
    np.random.seed(789)
    n_subjects = 40
    true_theta = 0.5  # True frailty variance
    
    # Generate frailties
    true_frailties = np.random.gamma(1/true_theta, true_theta, n_subjects)
    
    X_list = []
    durations_list = []
    events_list = []
    cluster_list = []
    
    for i in range(n_subjects):
        n_obs = np.random.randint(2, 6)
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        
        for j in range(n_obs):
            # Time depends on covariates and frailty
            base_hazard = 0.05
            hazard = true_frailties[i] * base_hazard * np.exp(0.5*x1 - 0.3*x2)
            time = np.random.exponential(1/hazard)
            
            censoring_time = np.random.uniform(10, 50)
            observed_time = min(time, censoring_time)
            event = 1 if time < censoring_time else 0
            
            X_list.append([x1, x2])
            durations_list.append(observed_time)
            events_list.append(event)
            cluster_list.append(i)
    
    X = np.array(X_list)
    durations = np.array(durations_list)
    events = np.array(events_list)
    cluster_id = np.array(cluster_list)
    
    # Fit Gamma frailty model
    frailty_model = FrailtyModel(distribution='gamma', tie_method='efron')
    frailty_model.fit(X, durations, events, cluster_id)
    
    print(frailty_model.result_.summary())
    
    print(f"\nTrue frailty variance: {true_theta:.4f}")
    print(f"Estimated frailty variance: {frailty_model.frailty_variance_:.4f}")
    
    # Compare true vs estimated frailties
    print("\nFrailty comparison (first 10 subjects):")
    print("Subject | True | Estimated")
    print("-" * 35)
    for i in range(min(10, n_subjects)):
        print(f"   {i:2d}   | {true_frailties[i]:.3f} |   {frailty_model.frailty_values_[i]:.3f}")
    
    print("\n✓ Test completed successfully")
    return frailty_model


def test_frailty_gaussian():
    """Test Gaussian (log-normal) frailty model."""
    print("\n" + "="*80)
    print("TEST: Gaussian Frailty Model")
    print("="*80)
    
    if FrailtyModel is None:
        print("SKIPPED: FrailtyModel not available")
        return None
    
    # Similar to Gamma frailty but with log-normal distribution
    np.random.seed(101112)
    n_subjects = 35
    true_sigma_sq = 0.4
    
    # Generate log-normal frailties: Z ~ LogNormal(-σ²/2, σ²) so E[Z] = 1
    true_frailties = np.random.lognormal(-true_sigma_sq/2, np.sqrt(true_sigma_sq), n_subjects)
    
    X_list = []
    durations_list = []
    events_list = []
    cluster_list = []
    
    for i in range(n_subjects):
        n_obs = np.random.randint(2, 5)
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        
        for j in range(n_obs):
            base_hazard = 0.04
            hazard = true_frailties[i] * base_hazard * np.exp(0.4*x1 - 0.5*x2)
            time = np.random.exponential(1/hazard)
            
            censoring_time = np.random.uniform(15, 60)
            observed_time = min(time, censoring_time)
            event = 1 if time < censoring_time else 0
            
            X_list.append([x1, x2])
            durations_list.append(observed_time)
            events_list.append(event)
            cluster_list.append(i)
    
    X = np.array(X_list)
    durations = np.array(durations_list)
    events = np.array(events_list)
    cluster_id = np.array(cluster_list)
    
    # Fit Gaussian frailty model
    frailty_model = FrailtyModel(distribution='gaussian', tie_method='efron')
    frailty_model.fit(X, durations, events, cluster_id)
    
    print(frailty_model.result_.summary())
    
    print(f"\nTrue log-frailty variance: {true_sigma_sq:.4f}")
    print(f"Estimated log-frailty variance: {frailty_model.frailty_variance_:.4f}")
    
    print("\n✓ Test completed successfully")
    return frailty_model


def generate_r_validation_script():
    """
    Generate R script to create reference values for validation.
    """
    r_script = '''# R Validation Script for Recurrent Event Models
# Run this in R to generate reference values for Python validation

library(survival)

# ============================================================================
# 1. Andersen-Gill Model with CGD Data
# ============================================================================
cat("\\n", rep("=", 80), "\\n", sep="")
cat("ANDERSEN-GILL MODEL\\n")
cat(rep("=", 80), "\\n", sep="")

data(cgd)

# Fit AG model
ag_fit <- coxph(Surv(tstart, tstop, status) ~ treat + age, 
                data=cgd, 
                id=id,
                method="efron")

cat("\\nCoefficients:\\n")
print(ag_fit$coefficients)

cat("\\nRobust SE:\\n")
print(sqrt(diag(ag_fit$var)))

cat("\\nNaive SE:\\n")
print(sqrt(diag(ag_fit$naive.var)))

cat("\\nLog-likelihood:\\n")
print(ag_fit$loglik)

cat("\\nRobust Variance Matrix:\\n")
print(ag_fit$var)

cat("\\nNaive Variance Matrix:\\n")
print(ag_fit$naive.var)

# ============================================================================
# 2. PWP-TT Model with Bladder Data
# ============================================================================
cat("\\n", rep("=", 80), "\\n", sep="")
cat("PWP-TT MODEL\\n")
cat(rep("=", 80), "\\n", sep="")

data(bladder)

# Fit PWP-TT
pwptt_fit <- coxph(Surv(start, stop, event) ~ treatment + strata(enum), 
                   data=bladder,
                   id=id,
                   method="efron")

cat("\\nCoefficients:\\n")
print(pwptt_fit$coefficients)

cat("\\nRobust SE:\\n")
print(sqrt(diag(pwptt_fit$var)))

cat("\\nLog-likelihood:\\n")
print(pwptt_fit$loglik)

# ============================================================================
# 3. Frailty Model (Gamma)
# ============================================================================
cat("\\n", rep("=", 80), "\\n", sep="")
cat("GAMMA FRAILTY MODEL\\n")
cat(rep("=", 80), "\\n", sep="")

# Use kidney data (has recurrent events)
data(kidney)

frailty_gamma <- coxph(Surv(time, status) ~ age + sex + disease + 
                        frailty(id, distribution="gamma"),
                       data=kidney)

cat("\\nCoefficients:\\n")
print(frailty_gamma$coefficients)

cat("\\nSE:\\n")
print(sqrt(diag(frailty_gamma$var)))

cat("\\nFrailty variance (theta):\\n")
print(frailty_gamma$history$frailty[[1]]$theta)

cat("\\nLog-likelihood:\\n")
print(frailty_gamma$loglik)

# ============================================================================
# 4. Frailty Model (Gaussian)
# ============================================================================
cat("\\n", rep("=", 80), "\\n", sep="")
cat("GAUSSIAN FRAILTY MODEL\\n")
cat(rep("=", 80), "\\n", sep="")

frailty_gaussian <- coxph(Surv(time, status) ~ age + sex + disease + 
                          frailty(id, distribution="gaussian"),
                         data=kidney)

cat("\\nCoefficients:\\n")
print(frailty_gaussian$coefficients)

cat("\\nSE:\\n")
print(sqrt(diag(frailty_gaussian$var)))

cat("\\nFrailty variance (sigma^2):\\n")
print(frailty_gaussian$history$frailty[[1]]$theta)

cat("\\nLog-likelihood:\\n")
print(frailty_gaussian$loglik)

cat("\\n", rep("=", 80), "\\n", sep="")
cat("VALIDATION COMPLETE\\n")
cat(rep("=", 80), "\\n", sep="")
'''
    
    # Save to current directory (where test is run)
    output_path = os.path.join(os.getcwd(), 'r_validation_script.R')
    try:
        with open(output_path, 'w') as f:
            f.write(r_script)
        print(f"\n✓ R validation script saved to: {output_path}")
        print("  Run this script in R to generate reference values")
    except Exception as e:
        print(f"\n✗ Could not save R script: {e}")
        print("  Script content printed below - save manually:")
        print("\n" + "="*80)
        print(r_script)
        print("="*80)


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION TESTS FOR RECURRENT EVENT MODELS")
    print("="*80)
    
    print("\nThese tests validate Python implementations against R's survival package.")
    print("For exact validation, run the generated R script and compare outputs.")
    
    results = {}
    
    # Run tests
    print("\n" + "="*80)
    try:
        results['ag'] = test_andersen_gill_cgd()
        print("✓ Andersen-Gill test completed")
    except Exception as e:
        print(f"✗ Andersen-Gill test failed: {e}")
        results['ag'] = None
    
    print("\n" + "="*80)
    try:
        results['pwptt'] = test_pwp_tt_bladder()
        print("✓ PWP-TT test completed")
    except Exception as e:
        print(f"✗ PWP-TT test failed: {e}")
        results['pwptt'] = None
    
    print("\n" + "="*80)
    try:
        results['pwpgt'] = test_pwp_gt()
        print("✓ PWP-GT test completed")
    except Exception as e:
        print(f"✗ PWP-GT test failed: {e}")
        results['pwpgt'] = None
    
    print("\n" + "="*80)
    try:
        results['gamma'] = test_frailty_gamma()
        print("✓ Gamma frailty test completed")
    except Exception as e:
        print(f"✗ Gamma frailty test failed: {e}")
        results['gamma'] = None
    
    print("\n" + "="*80)
    try:
        results['gaussian'] = test_frailty_gaussian()
        print("✓ Gaussian frailty test completed")
    except Exception as e:
        print(f"✗ Gaussian frailty test failed: {e}")
        results['gaussian'] = None
    
    # Generate R script
    print("\n" + "="*80)
    generate_r_validation_script()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    completed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"Completed: {completed}/{total} tests")
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Add missing model files to your project:")
    for model, result in results.items():
        if result is None:
            if model == 'pwpgt':
                print("   - Copy pwp_gt.py to project root")
            elif model in ['gamma', 'gaussian']:
                print("   - Copy frailty_models.py to project root")
    print("\n2. Run r_validation_script.R in R to get reference values")
    print("3. Compare Python outputs with R outputs")
    print("4. Verify coefficients, SEs, and log-likelihoods match")
    
    return results


if __name__ == "__main__":
    run_all_tests()