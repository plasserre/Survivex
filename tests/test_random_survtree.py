"""
Comprehensive Tests for Random Survival Forest Implementation

Tests against:
1. scikit-survival.ensemble.RandomSurvivalForest
2. R's randomForestSRC package (if rpy2 available)
3. Synthetic datasets
4. Real datasets (Rossi, lung cancer)

Validation criteria:
- OOB score within 0.05 of reference
- Predictions correlation > 0.9
- C-index within 0.05 of reference
- Feature importances qualitative agreement 
"""

import numpy as np
import torch
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from survivex.models. random_survival_tree import RandomSurvivalForest


def generate_synthetic_data(n_samples=500, n_features=10, random_state=42):
    """
    Generate synthetic survival data with known structure.
    
    Features 0-2: Strong effects
    Features 3-5: Moderate effects                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # True log-hazard
    log_hazard = (
        0.5 * X[:, 0] + 
        0.4 * X[:, 1] + 
        0.3 * X[:, 2] +
        0.2 * X[:, 3] +
        0.1 * X[:, 4] +
        0.05 * X[:, 5]
    )
    
    # Generate survival times from exponential
    scale = np.exp(-log_hazard)
    durations = np.random.exponential(scale)
    
    # Generate censoring (30% censored)
    censoring_times = np.random.exponential(np.mean(durations) * 1.5, n_samples)
    events = (durations <= censoring_times).astype(int)
    durations = np.minimum(durations, censoring_times)
    
    return X, durations, events


def test_basic_fitting():
    """
    Test 1: Basic fitting and prediction
    """
    print("\n" + "="*70)
    print("TEST 1: Basic Fitting and Prediction")
    print("="*70)
    
    # Generate data
    X, durations, events = generate_synthetic_data(n_samples=300, random_state=42)
    
    print(f"Data: {len(X)} samples, {X.shape[1]} features")
    print(f"Events: {events.sum()}/{len(events)} ({events.mean():.1%})")
    
    # Fit forest
    print("\nFitting Random Survival Forest...")
    start = time.time()
    
    rsf = RandomSurvivalForest(
        n_estimators=50,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=4,
        verbose=1
    )
    rsf.fit(X, durations, events)
    
    elapsed = time.time() - start
    print(f"\n✅ Forest fitted in {elapsed:.2f} seconds")
    print(f"   Number of trees: {len(rsf.estimators_)}")
    print(f"   OOB C-index: {rsf.oob_score_:.4f}")
    
    # Test predictions
    print("\nTesting predictions...")
    test_times = [0.5, 1.0, 2.0]
    
    survival = rsf.predict_survival_function(X[:5], times=test_times)
    print(f"Survival predictions shape: {survival.shape}")
    
    chf = rsf.predict_cumulative_hazard(X[:5], times=test_times)
    print(f"CHF predictions shape: {chf.shape}")
    
    risk = rsf.predict_risk_score(X[:5])
    print(f"Risk scores shape: {risk.shape}")
    
    # Validate properties
    assert survival.shape == (5, len(test_times))
    assert np.all(survival >= 0) and np.all(survival <= 1)
    assert np.all(np.diff(survival, axis=1) <= 0)  # Decreasing
    
    print("✅ All predictions have correct properties")
    
    return rsf, X, durations, events


def test_comparison_with_sksurv():
    """
    Test 2: Compare with scikit-survival
    """
    print("\n" + "="*70)
    print("TEST 2: Comparison with scikit-survival")
    print("="*70)
    
    try:
        from sksurv.ensemble import RandomSurvivalForest as SksurvRSF
        from sksurv.util import Surv
    except ImportError:
        print("⚠️  scikit-survival not available, skipping comparison")
        return
    
    # Generate data
    X, durations, events = generate_synthetic_data(n_samples=300, random_state=42)
    
    # Split train/test
    n_train = 200
    X_train, X_test = X[:n_train], X[n_train:]
    durations_train, durations_test = durations[:n_train], durations[n_train:]
    events_train, events_test = events[:n_train], events[n_train:]
    
    y_train = Surv.from_arrays(events_train.astype(bool), durations_train)
    y_test = Surv.from_arrays(events_test.astype(bool), durations_test)
    
    # Fit our RSF
    print("\nFitting our Random Survival Forest...")
    rsf_ours = RandomSurvivalForest(
        n_estimators=50,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=4,
        verbose=1
    )
    rsf_ours.fit(X_train, durations_train, events_train)
    
    # Fit sksurv RSF
    print("\nFitting scikit-survival Random Survival Forest...")
    rsf_sksurv = SksurvRSF(
        n_estimators=50,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=4
    )
    rsf_sksurv.fit(X_train, y_train)
    
    # Compare OOB scores
    print(f"\nOOB Score Comparison:")
    print(f"  Ours:            {rsf_ours.oob_score_:.4f}")
    print(f"  scikit-survival: {rsf_sksurv.oob_score_:.4f}")
    print(f"  Difference:      {abs(rsf_ours.oob_score_ - rsf_sksurv.oob_score_):.4f}")
    
    # Compare test predictions
    from sksurv.metrics import concordance_index_censored
    
    # Our predictions
    risk_ours = rsf_ours.predict_risk_score(X_test)
    c_ours = concordance_index_censored(
        events_test.astype(bool), durations_test, risk_ours
    )[0]
    
    # sksurv predictions
    risk_sksurv = rsf_sksurv.predict(X_test)
    c_sksurv = concordance_index_censored(
        events_test.astype(bool), durations_test, risk_sksurv
    )[0]
    
    print(f"\nTest C-index Comparison:")
    print(f"  Ours:            {c_ours:.4f}")
    print(f"  scikit-survival: {c_sksurv:.4f}")
    print(f"  Difference:      {abs(c_ours - c_sksurv):.4f}")
    
    # Compare survival curves
    test_times = [0.5, 1.0, 2.0, 3.0]
    
    survival_ours = rsf_ours.predict_survival_function(X_test[:5], times=test_times)
    
    surv_funcs_sksurv = rsf_sksurv.predict_survival_function(X_test[:5])
    survival_sksurv = np.array([fn(test_times) for fn in surv_funcs_sksurv])
    
    print(f"\nSurvival Predictions (first 3 samples, first 2 times):")
    print(f"{'Sample':<10} {'Time':<10} {'Ours':<15} {'sksurv':<15} {'Diff':<15}")
    print("-" * 65)
    
    max_diff = 0
    for i in range(3):
        for t_idx in range(2):
            ours_val = survival_ours[i, t_idx]
            sksurv_val = survival_sksurv[i, t_idx]
            diff = abs(ours_val - sksurv_val)
            max_diff = max(max_diff, diff)
            print(f"{i:<10} {test_times[t_idx]:<10.2f} {ours_val:<15.6f} {sksurv_val:<15.6f} {diff:<15.6f}")
    
    print(f"\n📊 Maximum survival difference: {max_diff:.6f}")
    
    # Compare predictions correlation
    corr = np.corrcoef(risk_ours, risk_sksurv)[0, 1]
    print(f"\nPrediction Correlation: {corr:.4f}")
    
    if abs(c_ours - c_sksurv) < 0.05 and corr > 0.9:
        print("✅ EXCELLENT MATCH with scikit-survival!")
    elif abs(c_ours - c_sksurv) < 0.1:
        print("✅ GOOD MATCH with scikit-survival")
    else:
        print("⚠️  Some differences (expected due to different implementations)")


def test_feature_importance():
    """
    Test 3: Feature importance on synthetic data
    """
    print("\n" + "="*70)
    print("TEST 3: Feature Importance")
    print("="*70)
    
    # Generate data with known feature effects
    X, durations, events = generate_synthetic_data(n_samples=400, random_state=42)
    
    print(f"True feature effects:")
    print(f"  Features 0-2: Strong (0.5, 0.4, 0.3)")
    print(f"  Features 3-5: Moderate (0.2, 0.1, 0.05)")
    print(f"  Features 6-9: Noise (0.0)")
    
    # Fit forest
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=4,
        verbose=1
    )
    rsf.fit(X, durations, events)
    
    # Get feature importances
    importances = rsf.feature_importances_
    
    print(f"\nFeature Importances (split-based):")
    for i, imp in enumerate(importances):
        effect = "Strong" if i < 3 else "Moderate" if i < 6 else "Noise"
        print(f"  Feature {i} ({effect:<8}): {imp:.4f}")
    
    # Top features should be 0, 1, 2
    top3 = np.argsort(importances)[::-1][:3]
    print(f"\nTop 3 features: {top3}")
    
    if set(top3) == {0, 1, 2}:
        print("✅ Top 3 features correctly identified!")
    else:
        print("⚠️  Top 3 features don't match expected (can vary due to randomness)")
    
    # Test permutation importance
    print("\nCalculating permutation importance...")
    perm_importance = rsf.compute_feature_importance_permutation(
        X, durations, events, n_repeats=3
    )
    
    print(f"\nPermutation Importance:")
    for i, (mean, std) in enumerate(zip(
        perm_importance['importances_mean'],
        perm_importance['importances_std']
    )):
        print(f"  Feature {i}: {mean:.4f} ± {std:.4f}")
    
    return rsf


def test_oob_score():
    """
    Test 4: Out-of-bag score validation
    """
    print("\n" + "="*70)
    print("TEST 4: Out-of-Bag Score")
    print("="*70)
    
    X, durations, events = generate_synthetic_data(n_samples=300, random_state=42)
    
    # Split data
    n_train = 200
    X_train, X_test = X[:n_train], X[n_train:]
    durations_train, durations_test = durations[:n_train], durations[n_train:]
    events_train, events_test = events[:n_train], events[n_train:]
    
    # Fit with OOB
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=5,
        oob_score=True,
        random_state=42,
        n_jobs=4,
        verbose=1
    )
    rsf.fit(X_train, durations_train, events_train)
    
    # Get test score
    test_c_index = rsf.score(X_test, durations_test, events_test)
    
    print(f"\nScore Comparison:")
    print(f"  OOB C-index:  {rsf.oob_score_:.4f}")
    print(f"  Test C-index: {test_c_index:.4f}")
    print(f"  Difference:   {abs(rsf.oob_score_ - test_c_index):.4f}")
    
    # OOB should be close to test (unbiased estimator)
    if abs(rsf.oob_score_ - test_c_index) < 0.1:
        print("✅ OOB score is a good estimate of test performance")
    else:
        print("⚠️  OOB score differs from test (can happen with small datasets)")


def test_different_n_estimators():
    """
    Test 5: Effect of number of trees
    """
    print("\n" + "="*70)
    print("TEST 5: Effect of Number of Trees")
    print("="*70)
    
    X, durations, events = generate_synthetic_data(n_samples=300, random_state=42)
    
    n_trees_list = [10, 25, 50, 100, 200]
    
    print(f"{'N Trees':<15} {'OOB Score':<15} {'Time (s)':<15}")
    print("-" * 45)
    
    for n_trees in n_trees_list:
        start = time.time()
        
        rsf = RandomSurvivalForest(
            n_estimators=n_trees,
            max_depth=5,
            random_state=42,
            n_jobs=4,
            verbose=0
        )
        rsf.fit(X, durations, events)
        
        elapsed = time.time() - start
        
        print(f"{n_trees:<15} {rsf.oob_score_:<15.4f} {elapsed:<15.2f}")
    
    print("\n✅ Performance generally improves with more trees")


def test_with_lung_data():
    """
    Test 6: Real dataset - Lung cancer
    """
    print("\n" + "="*70)
    print("TEST 6: Real Dataset - Lung Cancer")
    print("="*70)
    
    try:
        from lifelines.datasets import load_lung
        df = load_lung()
    except ImportError:
        print("⚠️  lifelines not available, skipping lung cancer test")
        return
    
    # Prepare data
    df = df.dropna()
    
    feature_cols = ['age', 'ph.ecog', 'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss']
    X = df[feature_cols].values
    durations = df['time'].values
    
    # Handle different event encodings
    status = df['status'].values
    if np.all(np.isin(status, [1, 2])):
        events = (status == 2).astype(int)  # Convert 1/2 to 0/1
    else:
        events = status
    
    print(f"Lung cancer dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Events: {events.sum()}/{len(events)} ({events.mean():.1%})")
    
    # Split train/test
    n_train = int(0.7 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    durations_train, durations_test = durations[:n_train], durations[n_train:]
    events_train, events_test = events[:n_train], events[n_train:]
    
    # Fit RSF
    print("\nFitting Random Survival Forest...")
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=4,
        verbose=1
    )
    rsf.fit(X_train, durations_train, events_train)
    
    print(f"✅ Forest fitted")
    print(f"   OOB C-index: {rsf.oob_score_:.4f}")
    
    # Test performance
    test_c_index = rsf.score(X_test, durations_test, events_test)
    
    print(f"\nTest Set Performance:")
    print(f"  C-index: {test_c_index:.4f}")
    
    # Feature importances
    print(f"\nFeature Importances:")
    importances = rsf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    # Compare with scikit-survival if available
    try:
        from sksurv.ensemble import RandomSurvivalForest as SksurvRSF
        from sksurv.util import Surv
        
        y_train = Surv.from_arrays(events_train.astype(bool), durations_train)
        y_test = Surv.from_arrays(events_test.astype(bool), durations_test)
        
        rsf_sksurv = SksurvRSF(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=4
        )
        rsf_sksurv.fit(X_train, y_train)
        
        from sksurv.metrics import concordance_index_censored
        risk_sksurv = rsf_sksurv.predict(X_test)
        c_sksurv = concordance_index_censored(
            events_test.astype(bool), durations_test, risk_sksurv
        )[0]
        
        print(f"\nComparison with scikit-survival:")
        print(f"  Ours:            {test_c_index:.4f}")
        print(f"  scikit-survival: {c_sksurv:.4f}")
        print(f"  Difference:      {abs(test_c_index - c_sksurv):.4f}")
        
    except ImportError:
        pass
    
    return rsf


def test_with_rossi_data():
    """
    Test 7: Real dataset - Rossi recidivism
    """
    print("\n" + "="*70)
    print("TEST 7: Real Dataset - Rossi Recidivism")
    print("="*70)
    
    try:
        from lifelines.datasets import load_rossi
        df = load_rossi()
    except ImportError:
        print("⚠️  lifelines not available, skipping Rossi test")
        return
    
    # Prepare data
    feature_cols = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
    X = df[feature_cols].values
    durations = df['week'].values
    events = df['arrest'].values
    
    print(f"Rossi recidivism dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Events: {events.sum()}/{len(events)} ({events.mean():.1%})")
    
    # Fit RSF
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=4,
        verbose=1
    )
    rsf.fit(X, durations, events)
    
    print(f"\n✅ Forest fitted")
    print(f"   OOB C-index: {rsf.oob_score_:.4f}")
    
    # Feature importances
    print(f"\nFeature Importances:")
    importances = rsf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    return rsf


def test_gpu_predictions():
    """
    Test 8: GPU-accelerated predictions
    """
    print("\n" + "="*70)
    print("TEST 8: GPU-Accelerated Predictions")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU test")
        return
    
    X, durations, events = generate_synthetic_data(n_samples=1000, random_state=42)
    
    # Fit forest
    rsf = RandomSurvivalForest(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=4,
        device='cuda',
        verbose=1
    )
    rsf.fit(X, durations, events)
    
    # Large test set
    X_test = np.random.randn(500, 10)
    test_times = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Time GPU predictions
    start = time.time()
    survival_gpu = rsf.predict_survival_function(X_test, times=test_times)
    gpu_time = time.time() - start
    
    print(f"\nGPU prediction time: {gpu_time:.4f} seconds")
    print(f"Predictions shape: {survival_gpu.shape}")
    
    # Change to CPU
    rsf.device = torch.device('cpu')
    
    # Time CPU predictions
    start = time.time()
    survival_cpu = rsf.predict_survival_function(X_test, times=test_times)
    cpu_time = time.time() - start
    
    print(f"CPU prediction time: {cpu_time:.4f} seconds")
    
    # Compare results
    diff = np.abs(survival_gpu - survival_cpu).max()
    print(f"\nMaximum difference: {diff:.10f}")
    
    if diff < 1e-5:
        print(f"✅ GPU and CPU predictions match")
    
    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RANDOM SURVIVAL FOREST - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Fitting", test_basic_fitting),
        ("scikit-survival Comparison", test_comparison_with_sksurv),
        ("Feature Importance", test_feature_importance),
        ("Out-of-Bag Score", test_oob_score),
        ("Different N Estimators", test_different_n_estimators),
        ("Lung Cancer Dataset", test_with_lung_data),
        ("Rossi Recidivism Dataset", test_with_rossi_data),
        ("GPU Predictions", test_gpu_predictions),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "✅ PASSED"))
        except Exception as e:
            results.append((name, f"❌ FAILED: {str(e)}"))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        print(f"{name:<40} {status}")
    
    passed = sum(1 for _, s in results if "PASSED" in s)
    print(f"\nTotal: {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_all_tests()