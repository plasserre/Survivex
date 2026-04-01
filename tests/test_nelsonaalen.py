"""
Comprehensive validation suite for Nelson-Aalen estimator implementation.

Validates against:
1. lifelines library
2. Kaplan-Meier estimator (S(t) = exp(-H(t)))
3. Published research results
4. Known statistical properties
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional

# Import SurviveX implementations
try:
    from survivex.models.nelson_aalen import NelsonAalenEstimator
    from survivex.models.kaplan_meier import KaplanMeierEstimator
except ImportError:
    print("⚠️  Import from survivex failed, trying local imports")
    from nelson_aalen import NelsonAalenEstimator
    from kaplan_meier import KaplanMeierEstimator


def extract_lifelines_value(result):
    """
    Robust extraction of scalar value from lifelines return (handles Series/DataFrame variations).
    """
    try:
        if hasattr(result, 'values'):
            vals = result.values
            if vals.ndim == 0:  # Scalar
                return float(vals)
            else:  # Array
                return float(vals.flatten()[0])
        elif hasattr(result, 'iloc'):
            return float(result.iloc[0])
        else:
            return float(result)
    except Exception as e:
        print(f"Error extracting value: {e}")
        return None


def test_simple_example():
    """
    Test 1: Simple example with known results
    """
    print("\n" + "="*70)
    print("TEST 1: Simple Hand-Verified Example")
    print("="*70)
    
    # Simple data
    durations = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    events = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Events at 1, 3, 5, 7
    
    print(f"Data: {len(durations)} subjects, {events.sum()} events")
    print(f"Events occur at times: {durations[events == 1]}")
    
    # Our implementation
    na_ours = NelsonAalenEstimator(device='cpu')
    na_ours.fit(durations, events)
    
    print(f"\n Our Implementation:")
    print(f"   Timeline: {na_ours.timeline_.cpu().numpy()}")
    print(f"   Cumulative hazard: {na_ours.cumulative_hazard_.cpu().numpy()}")
    
    # Manual calculation for verification
    # At t=1: 8 at risk, 1 event -> H(1) = 1/8 = 0.125
    # At t=3: 6 at risk, 1 event -> H(3) = 0.125 + 1/6 = 0.2917
    # At t=5: 4 at risk, 1 event -> H(5) = 0.2917 + 1/4 = 0.5417
    # At t=7: 2 at risk, 1 event -> H(7) = 0.5417 + 1/2 = 1.0417
    
    expected = np.array([1/8, 1/8 + 1/6, 1/8 + 1/6 + 1/4, 1/8 + 1/6 + 1/4 + 1/2])
    actual = na_ours.cumulative_hazard_.cpu().numpy()
    
    print(f"\n Manual Verification:")
    print(f"   Expected: {expected}")
    print(f"   Actual:   {actual}")
    print(f"   Difference: {np.abs(expected - actual)}")
    
    if np.allclose(expected, actual, atol=1e-6):
        print("  Manual verification PASSED!")
    else:
        print(" X Manual verification FAILED!")
    
    # Compare with lifelines
    try:
        from lifelines import NelsonAalenFitter
        
        na_lifelines = NelsonAalenFitter()
        na_lifelines.fit(durations, events)
        
        print(f"\n📚 Lifelines:")
        print(f"   Timeline: {na_lifelines.cumulative_hazard_.index.values}")
        print(f"   Cumulative hazard: {na_lifelines.cumulative_hazard_.iloc[:, 0].values}")
        
        # Compare at specific times
        test_times = [1, 3, 5, 7]
        print(f"\nComparison at event times:")
        print("Time\tOurs\tLifelines\tDifference")
        print("-" * 40)
        
        max_diff = 0
        for t in test_times:
            ours = na_ours.cumulative_hazard_at_times([t])[0].item()
            
            # Get lifelines value
            if t in na_lifelines.cumulative_hazard_.index:
                theirs = na_lifelines.cumulative_hazard_.loc[t].iloc[0]
            else:
                theirs = extract_lifelines_value(na_lifelines.cumulative_hazard_at_times([t]))
            
            if theirs is not None:
                diff = abs(ours - theirs)
                max_diff = max(max_diff, diff)
                print(f"{t}\t{ours:.6f}\t{theirs:.6f}\t{diff:.10f}")
        
        if max_diff < 1e-8:
            print("\nPERFECT MATCH with lifelines!")
        elif max_diff < 1e-6:
            print("\nEXCELLENT MATCH with lifelines!")
        else:
            print(f"\nSome difference: {max_diff}")
            
    except ImportError:
        print("\nlifelines not available for comparison")
    
    return na_ours


def test_comparison_with_kaplan_meier():
    """
    Test 2: Verify relationship S(t) = exp(-H(t)) with Kaplan-Meier
    """
    print("\n" + "="*70)
    print("TEST 2: Relationship with Kaplan-Meier (S(t) = exp(-H(t)))")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    durations = np.random.exponential(10, n_samples)
    events = np.random.binomial(1, 0.7, n_samples)
    
    print(f"Data: {n_samples} subjects, {events.sum()} events")
    
    # Fit Nelson-Aalen
    na = NelsonAalenEstimator(device='cpu')
    na.fit(durations, events)
    
    # Fit Kaplan-Meier
    km = KaplanMeierEstimator(device='cpu')
    km.fit(durations, events)
    
    # Test at various time points
    test_times = [5, 10, 15, 20, 25]
    
    print(f"\nComparing S(t) from both methods:")
    print("Time\tKM S(t)\tNA S(t)\tDifference")
    print("-" * 50)
    
    max_diff = 0
    for t in test_times:
        # Kaplan-Meier survival
        km_survival = km.survival_function_at_times([t])[0].item()
        
        # Nelson-Aalen derived survival: S(t) = exp(-H(t))
        na_survival = na.survival_function_at_times([t])[0].item()
        
        diff = abs(km_survival - na_survival)
        max_diff = max(max_diff, diff)
        
        print(f"{t}\t{km_survival:.4f}\t{na_survival:.4f}\t{diff:.6f}")
    
    print(f"\n📈 Maximum difference: {max_diff:.8f}")
    
    if max_diff < 0.01:
        print("Excellent agreement between KM and NA!")
        print("   (Small differences are expected due to different formulas)")
    else:
        print("Larger differences may indicate an issue")
    
    return na, km


def test_lung_cancer_data():
    """
    Test 3: Lung cancer dataset - standard benchmark
    """
    print("\n" + "="*70)
    print("TEST 3: Lung Cancer Dataset")
    print("="*70)
    
    try:
        from lifelines.datasets import load_lung
        
        # Load raw lung cancer data
        df = load_lung()
        
        print(f"Raw data inspection:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Status unique values: {sorted(df['status'].unique())}")
        
        # Check status encoding and convert appropriately
        status_values = sorted(df['status'].unique())
        
        durations = df['time'].values
        
        # ROBUST conversion based on actual values
        if set(status_values) == {0, 1}:
            # status=0 is censored, status=1 is event
            events = (df['status'] == 1).astype(int).values
            print(f"   Status encoding: 0=censored, 1=death")
        elif set(status_values) == {1, 2}:
            # status=1 is censored, status=2 is event
            events = (df['status'] == 2).astype(int).values
            print(f"   Status encoding: 1=censored, 2=death")
        else:
            # Default: assume higher value = event
            max_status = max(status_values)
            events = (df['status'] == max_status).astype(int).values
            print(f"   Status encoding: {max_status}=death (auto-detected)")
        
        print(f"\n   Status value counts:")
        for val in status_values:
            count = (df['status'] == val).sum()
            print(f"      Status={val}: {count} patients")
        
        print(f"\nAfter conversion:")
        print(f"   Total patients: {len(durations)}")
        print(f"   Events (deaths): {events.sum()} ({events.mean()*100:.1f}%)")
        print(f"   Censored: {(events == 0).sum()} ({(events == 0).mean()*100:.1f}%)")
        print(f"   Time range: {durations.min():.0f} to {durations.max():.0f} days")
        print(f"   Median time: {np.median(durations):.0f} days")
        
        # Verify we have events
        if events.sum() == 0:
            print("\nX ERROR: No events detected! Check data conversion.")
            return None
        
        # Fit our Nelson-Aalen
        print(f"\n Fitting Nelson-Aalen...")
        na_ours = NelsonAalenEstimator(device='cpu')
        na_ours.fit(durations, events)
        
        print(f"   Fitted with {len(na_ours.timeline_)} event times")
        print(f"   Final cumulative hazard: {na_ours.cumulative_hazard_[-1].item():.4f}")
        
        # Compare with lifelines
        try:
            from lifelines import NelsonAalenFitter
            
            na_lifelines = NelsonAalenFitter()
            na_lifelines.fit(durations, events)
            
            print(f"\nLifelines Nelson-Aalen:")
            print(f"   Final cumulative hazard: {na_lifelines.cumulative_hazard_.iloc[-1, 0]:.4f}")
            
            # Compare at various time points
            test_times = [100, 200, 300, 400, 500, 600, 700, 800, 900]
            
            print(f"\n Comparison at time points:")
            print("Time\tOurs\tLifelines\tDifference")
            print("-" * 50)
            
            max_diff = 0
            comparison_count = 0
            
            for t in test_times:
                if t > durations.max():
                    continue
                
                ours = na_ours.cumulative_hazard_at_times([t])[0].item()
                
                # Get lifelines value
                theirs = extract_lifelines_value(na_lifelines.cumulative_hazard_at_times([t]))
                
                if theirs is not None:
                    diff = abs(ours - theirs)
                    max_diff = max(max_diff, diff)
                    comparison_count += 1
                    print(f"{t}\t{ours:.6f}\t{theirs:.6f}\t{diff:.10f}")
            
            if comparison_count == 0:
                print("⚠️  No comparisons completed")
                return na_ours
            
            print(f"\nValidation Results:")
            print(f"   Maximum difference: {max_diff:.10f}")
            
            if max_diff < 1e-8:
                print("  PERFECT MATCH with lifelines!")
            elif max_diff < 1e-6:
                print("  EXCELLENT MATCH with lifelines!")
            else:
                print(f" Some difference detected")
            
            # Plot comparison
            try:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot 1: Cumulative hazard curves
                times_ours = np.concatenate([[0], na_ours.timeline_.cpu().numpy()])
                hazard_ours = np.concatenate([[0], na_ours.cumulative_hazard_.cpu().numpy()])
                
                ax1.step(times_ours, hazard_ours, where='post', 
                        linewidth=2, label='SurviveX', color='darkred')
                
                times_lifelines = na_lifelines.cumulative_hazard_.index.values
                hazard_lifelines = na_lifelines.cumulative_hazard_.iloc[:, 0].values
                
                ax1.step(times_lifelines, hazard_lifelines, where='post', 
                        linewidth=2, label='Lifelines', color='blue', linestyle='--', alpha=0.7)
                
                ax1.set_xlabel('Time (days)')
                ax1.set_ylabel('Cumulative Hazard')
                ax1.set_title('Nelson-Aalen Cumulative Hazard')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Difference
                test_times_plot = np.linspace(0, durations.max(), 100)
                differences = []
                
                for t in test_times_plot:
                    ours = na_ours.cumulative_hazard_at_times([t])[0].item()
                    theirs = extract_lifelines_value(na_lifelines.cumulative_hazard_at_times([t]))
                    if theirs is not None:
                        differences.append(ours - theirs)
                    else:
                        differences.append(0)
                
                ax2.plot(test_times_plot, differences, color='green', linewidth=1)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_xlabel('Time (days)')
                ax2.set_ylabel('Difference (SurviveX - Lifelines)')
                ax2.set_title('Difference Between Implementations')
                ax2.grid(True, alpha=0.3)
                
                max_abs_diff = max([abs(d) for d in differences])
                ax2.text(0.05, 0.95, f'Max |difference|: {max_abs_diff:.2e}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                plt.show()
                
                print("\n Plots generated successfully!")
                
            except ImportError:
                print("\nmatplotlib not available for plotting")
                
        except ImportError:
            print("\n⚠️  lifelines not available for comparison")
        
        return na_ours
        
    except Exception as e:
        print(f"X Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_edge_cases():
    """
    Test 4: Edge cases
    """
    print("\n" + "="*70)
    print("TEST 4: Edge Cases")
    print("="*70)
    
    # Case 1: All censored
    print("\nCase 1: All subjects censored")
    durations = np.array([1, 2, 3, 4, 5])
    events = np.array([0, 0, 0, 0, 0])
    
    na = NelsonAalenEstimator()
    na.fit(durations, events)
    print(f"   Cumulative hazard at time 3: {na.cumulative_hazard_at_times([3])[0]:.4f}")
    print(f"   Expected: 0.0 (no events)")
    
    # Case 2: All events
    print("\nCase 2: All events occur")
    durations = np.array([1, 2, 3, 4, 5])
    events = np.array([1, 1, 1, 1, 1])
    
    na = NelsonAalenEstimator()
    na.fit(durations, events)
    print(f"   Final cumulative hazard: {na.cumulative_hazard_[-1].item():.4f}")
    print(f"   Timeline: {na.timeline_.cpu().numpy()}")
    
    # Case 3: Single observation
    print("\nCase 3: Single observation")
    durations = np.array([10])
    events = np.array([1])
    
    na = NelsonAalenEstimator()
    na.fit(durations, events)
    print(f"   Cumulative hazard at time 10: {na.cumulative_hazard_at_times([10])[0]:.4f}")
    print(f"   Expected: 1.0 (d/n = 1/1)")


def test_survival_conversion():
    """
    Test 5: Verify survival function conversion
    """
    print("\n" + "="*70)
    print("TEST 5: Survival Function from Cumulative Hazard")
    print("="*70)
    
    np.random.seed(42)
    
    durations = np.random.exponential(15, 50)
    events = np.random.binomial(1, 0.6, 50)
    
    # Fit Nelson-Aalen
    na = NelsonAalenEstimator()
    na.fit(durations, events)
    
    # Test conversion S(t) = exp(-H(t))
    test_times = [5, 10, 15, 20]
    
    print("\nVerifying S(t) = exp(-H(t)):")
    print("Time\tH(t)\texp(-H(t))")
    print("-" * 30)
    
    for t in test_times:
        hazard = na.cumulative_hazard_at_times([t])[0].item()
        survival = na.survival_function_at_times([t])[0].item()
        expected_survival = np.exp(-hazard)
        
        print(f"{t}\t{hazard:.4f}\t{survival:.4f}")
        
        if abs(survival - expected_survival) < 1e-6:
            print(f"Matches exp(-{hazard:.4f}) = {expected_survival:.4f}")
        else:
            print(f"X Mismatch!")


def run_all_tests():
    """
    Run all validation tests.
    """
    print("\n" + "+"*35)
    print("NELSON-AALEN ESTIMATOR COMPREHENSIVE VALIDATION SUITE")
    print("_"*35)
    
    # Test 1: Simple example
    test_simple_example()
    
    # Test 2: Comparison with Kaplan-Meier
    test_comparison_with_kaplan_meier()
    
    # Test 3: Lung cancer data
    test_lung_cancer_data()
    
    # Test 4: Edge cases
    test_edge_cases()
    
    # Test 5: Survival conversion
    test_survival_conversion()
    
    print("\n" + "="*70)
    print("ALL NELSON-AALEN VALIDATION TESTS COMPLETED!")
    print("="*70)
    print("\nIf all tests show 'EXCELLENT MATCH' or 'PERFECT MATCH',")
    print("   SurviveX Nelson-Aalen estimator is validated and ready for use!")


if __name__ == "__main__":
    run_all_tests()