"""
Comprehensive validation suite for Log-Rank test implementation.

Validates against:
1. lifelines library
2. Published research results
3. R survival package results (when available)
4. Known statistical examples
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional

# Import your implementations
try:
    from survivex.models.log_rank_test import LogRankTest, logrank_test
    from survivex.models.kaplan_meier import KaplanMeierEstimator
    from survivex.datasets.converters import load_from_lifelines
    from survivex.datasets.loaders import load_survival_dataset
except ImportError:
    print("Import from survivex failed, trying local imports")
    from log_rank_test import LogRankTest, logrank_test
    from kaplan_meier import KaplanMeierEstimator


def test_simple_example():
    """
    Test 1: Simple hand-calculated example
    
    This example is simple enough to verify by hand calculation.
    """
    print("\n" + "="*70)
    print("TEST 1: Simple Hand-Calculated Example")
    print("="*70)
    
    # Group A: 5 subjects
    durations_A = np.array([4, 5, 6, 7, 8])
    events_A = np.array([1, 0, 1, 0, 1])  # 3 events
    
    # Group B: 5 subjects  
    durations_B = np.array([2, 3, 4, 5, 9])
    events_B = np.array([1, 1, 0, 1, 0])  # 3 events
    
    print(f"Group A: {len(durations_A)} subjects, {events_A.sum()} events")
    print(f"Group B: {len(durations_B)} subjects, {events_B.sum()} events")
    
    # Our implementation
    result_ours = logrank_test(durations_A, events_A, durations_B, events_B)
    
    print(f"\nOur Implementation:")
    print(f"   Test statistic: {result_ours.test_statistic:.6f}")
    print(f"   p-value: {result_ours.p_value:.6f}")
    print(f"   Group A - Observed: {result_ours.summary['group_A']['observed']:.2f}, "
          f"Expected: {result_ours.summary['group_A']['expected']:.2f}")
    print(f"   Group B - Observed: {result_ours.summary['group_B']['observed']:.2f}, "
          f"Expected: {result_ours.summary['group_B']['expected']:.2f}")
    
    # Compare with lifelines
    try:
        from lifelines.statistics import logrank_test as lifelines_logrank
        
        result_lifelines = lifelines_logrank(
            durations_A, durations_B,
            events_A, events_B
        )
        
        print(f"\nLifelines:")
        print(f"   Test statistic: {result_lifelines.test_statistic:.6f}")
        print(f"   p-value: {result_lifelines.p_value:.6f}")
        
        # Compare
        stat_diff = abs(result_ours.test_statistic - result_lifelines.test_statistic)
        p_diff = abs(result_ours.p_value - result_lifelines.p_value)
        
        print(f"\nComparison:")
        print(f"   Test statistic difference: {stat_diff:.10f}")
        print(f"   p-value difference: {p_diff:.10f}")
        
        if stat_diff < 1e-6 and p_diff < 1e-6:
            print("EXCELLENT MATCH!")
        elif stat_diff < 1e-4 and p_diff < 1e-4:
            print("VERY GOOD MATCH!")
        else:
            print("Some differences detected")
            
    except ImportError:
        print("\nlifelines not available for comparison")
    
    return result_ours


def test_published_example():
    """
    Test 2: Example from Collett (2003) - Modelling Survival Data in Medical Research
    
    This is a well-known example from survival analysis literature.
    """
    print("\n" + "="*70)
    print("TEST 2: Published Example (Collett, 2003)")
    print("="*70)
    
    # Treatment group
    durations_treatment = np.array([6, 7, 10, 15, 19, 25])
    events_treatment = np.array([1, 1, 1, 1, 0, 0])
    
    # Control group
    durations_control = np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 11, 11, 12, 15, 17, 22, 23])
    events_control = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    print(f"Treatment: {len(durations_treatment)} subjects, {events_treatment.sum()} events")
    print(f"Control: {len(durations_control)} subjects, {events_control.sum()} events")
    
    # Our implementation
    result_ours = logrank_test(durations_treatment, events_treatment,
                               durations_control, events_control)
    
    print(f"\n Our Implementation:")
    print(result_ours)
    
    # Compare with lifelines
    try:
        from lifelines.statistics import logrank_test as lifelines_logrank
        
        result_lifelines = lifelines_logrank(
            durations_treatment, durations_control,
            events_treatment, events_control
        )
        
        print(f"Lifelines:")
        print(f"   Test statistic: {result_lifelines.test_statistic:.6f}")
        print(f"   p-value: {result_lifelines.p_value:.6f}")
        
        stat_diff = abs(result_ours.test_statistic - result_lifelines.test_statistic)
        p_diff = abs(result_ours.p_value - result_lifelines.p_value)
        
        print(f"\nComparison:")
        print(f"   Difference in test statistic: {stat_diff:.10f}")
        print(f"   Difference in p-value: {p_diff:.10f}")
        
        if stat_diff < 1e-6:
            print("PERFECT MATCH!")
        elif stat_diff < 1e-4:
            print("EXCELLENT MATCH!")
        else:
            print("Investigating differences...")
            
    except ImportError:
        print("\nlifelines not available")
    
    return result_ours


def test_lung_cancer_sex_comparison():
    """
    Test 3: Lung cancer dataset - comparing male vs female survival
    
    This is a standard benchmark used in survival analysis literature.
    Expected: Males have significantly worse survival (p < 0.001)
    """
    print("\n" + "="*70)
    print("TEST 3: Lung Cancer Dataset - Male vs Female")
    print("="*70)
    
    try:
        # Load lung cancer data
        print("Loading lung cancer dataset...")
        df_raw = load_from_lifelines('lung', standardize=True)
        
        survival_data = load_survival_dataset(
            df_raw,
            time_col='time',
            event_col='event',
            auto_fix=True,
            verbose=False
        )
        
        df = survival_data.to_pandas()
        
        # Check if sex column exists
        if 'sex' not in df.columns:
            print("Sex column not found in dataset")
            return None
        
        # Split by sex (typically 1=male, 2=female)
        male_mask = df['sex'] == 1
        female_mask = df['sex'] == 2
        
        durations_male = df[male_mask]['time'].values
        events_male = df[male_mask]['event'].values
        
        durations_female = df[female_mask]['time'].values
        events_female = df[female_mask]['event'].values
        
        print(f"\nDataset Summary:")
        print(f"   Males: {len(durations_male)} subjects, {events_male.sum()} events ({events_male.mean()*100:.1f}%)")
        print(f"   Females: {len(durations_female)} subjects, {events_female.sum()} events ({events_female.mean()*100:.1f}%)")
        
        # Our implementation
        print(f"\n🔬 Running our Log-Rank test...")
        result_ours = logrank_test(durations_male, events_male,
                                   durations_female, events_female)
        
        print(f"\nOur Implementation:")
        print(result_ours)
        
        # Fit Kaplan-Meier curves for visualization
        print(f"\nKaplan-Meier curves:")
        km_male = KaplanMeierEstimator()
        km_male.fit(durations_male, events_male)
        
        km_female = KaplanMeierEstimator()
        km_female.fit(durations_female, events_female)
        
        print(f"   Male median survival: {km_male.median_survival_time()}")
        print(f"   Female median survival: {km_female.median_survival_time()}")
        
        # Compare with lifelines
        try:
            from lifelines.statistics import logrank_test as lifelines_logrank
            
            result_lifelines = lifelines_logrank(
                durations_male, durations_female,
                events_male, events_female
            )
            
            print(f"\nLifelines Results:")
            print(f"   Test statistic: {result_lifelines.test_statistic:.6f}")
            print(f"   p-value: {result_lifelines.p_value:.8f}")
            
            stat_diff = abs(result_ours.test_statistic - result_lifelines.test_statistic)
            p_diff = abs(result_ours.p_value - result_lifelines.p_value) if result_ours.p_value else None
            
            print(f"\nValidation:")
            print(f"   Test statistic difference: {stat_diff:.10f}")
            if p_diff is not None:
                print(f"   p-value difference: {p_diff:.10f}")
            
            if stat_diff < 1e-8:
                print("PERFECT MATCH with lifelines!")
            elif stat_diff < 1e-6:
                print("EXCELLENT MATCH with lifelines!")
            else:
                print(f"Difference detected: {stat_diff}")
            
            # Check against known result
            print(f"\nLiterature Reference:")
            print(f"   Expected: Significant difference (p < 0.001)")
            if result_ours.p_value < 0.001:
                print(f"Matches expected result!")
            else:
                print(f"Result differs from expectation")
                
        except ImportError:
            print("\nlifelines not available")
        
        # Visualize if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Survival curves
            times_male = np.concatenate([[0], km_male.timeline_.cpu().numpy()])
            surv_male = np.concatenate([[1], km_male.survival_function_.cpu().numpy()])
            
            times_female = np.concatenate([[0], km_female.timeline_.cpu().numpy()])
            surv_female = np.concatenate([[1], km_female.survival_function_.cpu().numpy()])
            
            ax1.step(times_male, surv_male, where='post', label='Male', linewidth=2, color='blue')
            ax1.step(times_female, surv_female, where='post', label='Female', linewidth=2, color='red')
            ax1.set_xlabel('Time (days)')
            ax1.set_ylabel('Survival Probability')
            ax1.set_title('Lung Cancer: Male vs Female Survival')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Add p-value to plot
            ax1.text(0.05, 0.05, f'Log-rank p-value: {result_ours.p_value:.6f}',
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Number at risk
            unique_times = np.sort(np.unique(np.concatenate([durations_male, durations_female])))
            n_at_risk_male = [np.sum(durations_male >= t) for t in unique_times]
            n_at_risk_female = [np.sum(durations_female >= t) for t in unique_times]
            
            ax2.plot(unique_times, n_at_risk_male, label='Male', linewidth=2, color='blue')
            ax2.plot(unique_times, n_at_risk_female, label='Female', linewidth=2, color='red')
            ax2.set_xlabel('Time (days)')
            ax2.set_ylabel('Number at Risk')
            ax2.set_title('Number at Risk Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print("\nPlots generated successfully!")
            
        except ImportError:
            print("\nmatplotlib not available for plotting")
        
        return result_ours
        
    except Exception as e:
        print(f"Error loading/processing lung cancer data: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_groups():
    """
    Test 4: Multiple group comparison (k > 2)
    
    Test with 3 groups to verify the generalized log-rank test.
    """
    print("\n" + "="*70)
    print("TEST 4: Multiple Group Comparison (3 groups)")
    print("="*70)
    
    np.random.seed(42)
    
    # Group A: Best survival
    durations_A = np.random.exponential(20, 30)
    events_A = np.random.binomial(1, 0.6, 30)
    
    # Group B: Intermediate survival
    durations_B = np.random.exponential(15, 30)
    events_B = np.random.binomial(1, 0.7, 30)
    
    # Group C: Worst survival
    durations_C = np.random.exponential(10, 30)
    events_C = np.random.binomial(1, 0.8, 30)
    
    print(f"Group A: {len(durations_A)} subjects, {events_A.sum()} events")
    print(f"Group B: {len(durations_B)} subjects, {events_B.sum()} events")
    print(f"Group C: {len(durations_C)} subjects, {events_C.sum()} events")
    
    # Our implementation
    lr = LogRankTest()
    result_ours = lr.compare_multiple(
        [durations_A, durations_B, durations_C],
        [events_A, events_B, events_C],
        group_names=['Group A', 'Group B', 'Group C']
    )
    
    print(f"\nOur Implementation:")
    print(result_ours)
    
    # Compare with lifelines
    try:
        from lifelines.statistics import multivariate_logrank_test
        
        # Prepare data for lifelines
        all_durations = np.concatenate([durations_A, durations_B, durations_C])
        all_events = np.concatenate([events_A, events_B, events_C])
        all_groups = np.concatenate([
            np.full(len(durations_A), 0),
            np.full(len(durations_B), 1),
            np.full(len(durations_C), 2)
        ])
        
        result_lifelines = multivariate_logrank_test(
            all_durations, all_groups, all_events
        )
        
        print(f"\nLifelines Results:")
        print(f"   Test statistic: {result_lifelines.test_statistic:.6f}")
        print(f"   p-value: {result_lifelines.p_value:.6f}")
        
        stat_diff = abs(result_ours.test_statistic - result_lifelines.test_statistic)
        p_diff = abs(result_ours.p_value - result_lifelines.p_value)
        
        print(f"\nComparison:")
        print(f"   Test statistic difference: {stat_diff:.10f}")
        print(f"   p-value difference: {p_diff:.10f}")
        
        if stat_diff < 1e-6:
            print("EXCELLENT MATCH!")
        else:
            print(f"Difference: {stat_diff}")
            
    except ImportError:
        print("\nlifelines not available")
    
    return result_ours


def test_edge_cases():
    """
    Test 5: Edge cases and special scenarios
    """
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70)
    
    # Case 1: No events in one group
    print("\nCase 1: No events in group B")
    durations_A = np.array([1, 2, 3, 4, 5])
    events_A = np.array([1, 1, 1, 0, 0])
    durations_B = np.array([1, 2, 3, 4, 5])
    events_B = np.array([0, 0, 0, 0, 0])  # All censored
    
    result = logrank_test(durations_A, events_A, durations_B, events_B)
    print(f"   Test statistic: {result.test_statistic:.6f}")
    print(f"   p-value: {result.p_value:.6f}")
    
    # Case 2: Identical groups
    print("\nCase 2: Identical groups")
    durations = np.array([1, 2, 3, 4, 5])
    events = np.array([1, 0, 1, 0, 1])
    
    result = logrank_test(durations, events, durations, events)
    print(f"   Test statistic: {result.test_statistic:.6f}")
    print(f"   p-value: {result.p_value:.6f}")
    print(f"   Expected: p-value ≈ 1.0 (no difference)")
    
    # Case 3: Very different groups
    print("\nCase 3: Very different groups")
    durations_A = np.array([10, 12, 15, 18, 20])
    events_A = np.array([0, 0, 0, 0, 0])  # All survive
    durations_B = np.array([1, 2, 3, 4, 5])
    events_B = np.array([1, 1, 1, 1, 1])  # All die early
    
    result = logrank_test(durations_A, events_A, durations_B, events_B)
    print(f"   Test statistic: {result.test_statistic:.6f}")
    print(f"   p-value: {result.p_value:.6f}")
    print(f"   Expected: p-value ≈ 0.0 (very different)")


def run_all_tests():
    """
    Run all validation tests.
    """
    print("\n" + "*"*35)
    print("LOG-RANK TEST COMPREHENSIVE VALIDATION SUITE")
    print("*"*35)
    
    # Test 1: Simple example
    test_simple_example()
    
    # Test 2: Published example
    test_published_example()
    
    # Test 3: Lung cancer (real data benchmark)
    test_lung_cancer_sex_comparison()
    
    # Test 4: Multiple groups
    test_multiple_groups()
    
    # Test 5: Edge cases
    test_edge_cases()
    
    print("\n" + "="*70)
    print("ALL VALIDATION TESTS COMPLETED!")
    print("="*70)
    print("\nIf all tests show 'EXCELLENT MATCH' or 'PERFECT MATCH',")
    print("   your Log-Rank test implementation is validated and ready for use!")


if __name__ == "__main__":
    run_all_tests()