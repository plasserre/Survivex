import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from survivex.datasets.loaders import load_survival_dataset
from survivex.datasets.validators import validate_survival_data
from survivex.datasets.converters import auto_detect_format, load_from_lifelines

import pandas as pd
import numpy as np
from survivex.datasets.loaders import load_survival_dataset
from survivex.datasets.validators import validate_survival_data
from survivex.datasets.converters import (
    load_from_lifelines, 
    standardize_column_names,
    auto_detect_format
)
from survivex.models.kaplan_meier import KaplanMeierEstimator




def extract_survival_prob_from_lifelines(km_lifelines, time_point):
    """
    Robust function to extract survival probability from lifelines at a specific time.
    """
    try:
        # Method 1: Use survival_function_at_times
        result = km_lifelines.survival_function_at_times([time_point])
        
        # Try different ways to access the value
        if hasattr(result, 'iloc'):
            try:
                return float(result.iloc[0])
            except:
                try:
                    return float(result.iloc[0, 0])
                except:
                    pass
        
        if hasattr(result, 'values'):
            if result.values.size == 1:
                return float(result.values.item())
            else:
                return float(result.values[0])
        
        # Try direct conversion
        return float(result)
        
    except:
        # Method 2: Use the survival_function_ attribute directly
        try:
            # Get the survival function DataFrame
            sf = km_lifelines.survival_function_
            # Find the latest time <= our time_point
            valid_times = sf.index[sf.index <= time_point]
            if len(valid_times) > 0:
                latest_time = valid_times[-1]
                return float(sf.loc[latest_time].iloc[0])
            else:
                # Time is before any events, survival = 1.0
                return 1.0
        except:
            pass
    
    # If all methods fail, return None
    return None

def simple_test():
    """Simple test of Kaplan-Meier implementation."""
    print("Simple Kaplan-Meier Test")
    print("=" * 30)
    
    # Generate sample data
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 50  # Smaller for easier debugging
    durations = np.random.exponential(10, n_samples)
    events = np.random.binomial(1, 0.7, n_samples)
    
    print(f"Generated {n_samples} samples")
    print(f"Event rate: {events.mean():.2%}")
    print(f"Duration range: {durations.min():.2f} to {durations.max():.2f}")
    
    # Fit our implementation
    print("\nFitting our implementation...")
    km_ours = KaplanMeierEstimator()
    km_ours.fit(durations, events)
    
    print(f"Number of time points: {len(km_ours.timeline_)}")
    median_time = km_ours.median_survival_time()
    print(f"Median survival time: {median_time}")
    
    # Test a few time points
    test_times = [5, 10, 15]
    print(f"\nSurvival probabilities:")
    for t in test_times:
        prob = km_ours.survival_function_at_times([t])[0].item()
        print(f"S({t}) = {prob:.4f}")
    
    # Compare with lifelines if available
    try:
        from lifelines import KaplanMeierFitter
        
        print("\nComparing with lifelines...")
        km_lifelines = KaplanMeierFitter()
        km_lifelines.fit(durations, events)
        
        print("Time\tOurs\tLifelines\tDifference")
        print("-" * 40)
        
        all_match = True
        for t in test_times:
            ours = km_ours.survival_function_at_times([t])[0].item()
            theirs = extract_survival_prob_from_lifelines(km_lifelines, t)
            
            if theirs is not None:
                diff = abs(ours - theirs)
                print(f"{t}\t{ours:.4f}\t{theirs:.4f}\t{diff:.6f}")
                if diff > 1e-6:
                    all_match = False
            else:
                print(f"{t}\t{ours:.4f}\tERROR\t-")
                all_match = False
        
        # Compare median survival times
        our_median = km_ours.median_survival_time()
        their_median = km_lifelines.median_survival_time_
        
        print(f"\nMedian survival:")
        print(f"Ours: {our_median}")
        print(f"Lifelines: {their_median}")
        
        if our_median is not None and their_median is not None:
            median_diff = abs(our_median - their_median)
            print(f"Difference: {median_diff:.6f}")
            if median_diff > 1e-6:
                all_match = False
        
        if all_match:
            print("\nAll comparisons match!")
        else:
            print("\nSome differences found")
            
    except ImportError:
        print("Lifelines not available")
    except Exception as e:
        print(f"Error comparing with lifelines: {e}")
    
    # Simple plot test
    try:
        print("\nTesting plot functionality...")
        km_ours.plot(title="Simple Test Survival Curve")
        print("Plotting works!")
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Plotting error: {e}")
    
    return km_ours

def test_with_known_data():
    """Test with manually created data where we know the expected results."""
    print("\n\nTesting with Known Data")
    print("=" * 30)
    
    # Create simple data where we can calculate results by hand
    durations = np.array([1, 2, 3, 4, 5])
    events = np.array([1, 0, 1, 0, 1])  # Events at times 1, 3, 5
    
    print("Data:")
    for i, (d, e) in enumerate(zip(durations, events)):
        status = "Event" if e == 1 else "Censored"
        print(f"Subject {i+1}: Time={d}, Status={status}")
    
    # Fit our estimator
    km = KaplanMeierEstimator()
    km.fit(durations, events)
    
    print(f"\nTimeline: {km.timeline_.cpu().numpy()}")
    print(f"Survival function: {km.survival_function_.cpu().numpy()}")
    
    # Manual calculation for verification:
    # At t=1: 5 at risk, 1 event -> S(1) = (5-1)/5 = 0.8
    # At t=3: 3 at risk, 1 event -> S(3) = 0.8 * (3-1)/3 = 0.8 * 2/3 = 0.533...
    # At t=5: 1 at risk, 1 event -> S(5) = 0.533... * (1-1)/1 = 0
    
    expected_at_1 = 4/5  # 0.8
    expected_at_3 = expected_at_1 * 2/3  # 0.533...
    expected_at_5 = 0.0
    
    actual_at_1 = km.survival_function_at_times([1])[0].item()
    actual_at_3 = km.survival_function_at_times([3])[0].item()
    actual_at_5 = km.survival_function_at_times([5])[0].item()
    
    print(f"\nManual verification:")
    print(f"S(1): Expected={expected_at_1:.4f}, Actual={actual_at_1:.4f}")
    print(f"S(3): Expected={expected_at_3:.4f}, Actual={actual_at_3:.4f}")
    print(f"S(5): Expected={expected_at_5:.4f}, Actual={actual_at_5:.4f}")
    
    if (abs(actual_at_1 - expected_at_1) < 1e-6 and 
        abs(actual_at_3 - expected_at_3) < 1e-6 and 
        abs(actual_at_5 - expected_at_5) < 1e-6):
        print("Manual verification passed!")
    else:
        print("Manual verification failed!")




def load_lung_cancer_data():
    """Load lung cancer data using your loader system."""
    print("Loading lung cancer data...")
    
    # Load from lifelines using your converter
    try:
        df_raw = load_from_lifelines('lung', standardize=True)
        print(f"Loaded lung cancer data: {df_raw.shape}")
        print(f"Columns: {list(df_raw.columns)}")
        
        # Use your loader to prepare survival data
        survival_data = load_survival_dataset(
            df_raw,
            time_col='time',
            event_col='event',
            auto_fix=True,
            verbose=True
        )
        
        # Convert to pandas for easier handling
        df = survival_data.to_pandas()
        print(f"Prepared survival data: {df.shape}")
        print(f"Final columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error with your loader, trying direct lifelines approach: {e}")
        
        # Fallback to direct lifelines loading
        from lifelines.datasets import load_lung
        df_raw = load_lung()
        
        # Manual preparation to match your expected format
        df = df_raw.copy()
        
        # Map status to event (assuming status: 1=censored, 2=event)
        if 'status' in df.columns and 'event' not in df.columns:
            df['event'] = (df['status'] == 2).astype(int)
        
        # Rename time column if needed
        if 'T' in df.columns and 'time' not in df.columns:
            df = df.rename(columns={'T': 'time'})
        
        print(f"Loaded with fallback method: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df

def extract_survival_prob_lifelines(km_lifelines, time_point):
    """
    Robust extraction of survival probability from lifelines at a specific time.
    """
    try:
        result = km_lifelines.survival_function_at_times([time_point])
        
        # Handle different return types
        if hasattr(result, 'values'):
            if result.values.size == 1:
                return float(result.values.item())
            else:
                return float(result.values[0])
        elif hasattr(result, 'iloc'):
            return float(result.iloc[0])
        else:
            return float(result)
            
    except:
        # Fallback: use survival function directly
        sf = km_lifelines.survival_function_
        valid_times = sf.index[sf.index <= time_point]
        if len(valid_times) > 0:
            latest_time = valid_times[-1]
            return float(sf.loc[latest_time].iloc[0])
        else:
            return 1.0

def test_lung_cancer_kaplan_meier():
    """
    Comprehensive test of Kaplan-Meier implementation on lung cancer data.
    """
    print("LUNG CANCER KAPLAN-MEIER ANALYSIS")
    print("=" * 60)
    
    # Load the data
    df = load_lung_cancer_data()
    
    # Extract time and event columns
    time_col = 'time'
    event_col = 'event'
    
    if time_col not in df.columns or event_col not in df.columns:
        print(f"Required columns not found. Available: {list(df.columns)}")
        return None
    
    durations = df[time_col].values
    events = df[event_col].values
    
    # Data summary
    print(f"\nDataset Summary:")
    print(f"   Number of patients: {len(durations)}")
    print(f"   Follow-up time range: {durations.min():.1f} to {durations.max():.1f} days")
    print(f"   Number of events: {events.sum()} ({events.mean():.1%})")
    print(f"   Number of censored: {len(events) - events.sum()} ({(1-events.mean()):.1%})")
    print(f"   Median follow-up: {np.median(durations):.1f} days")
    
    # Fit your Kaplan-Meier estimator
    print(f"\nFitting your Kaplan-Meier estimator...")
    km_ours = KaplanMeierEstimator(device='cpu')
    km_ours.fit(durations, events)
    
    print(f"   Fitted with {len(km_ours.timeline_)} unique event times")
    
    our_median = km_ours.median_survival_time()
    if our_median:
        print(f"   Median survival time: {our_median:.1f} days")
    else:
        print(f"   Median survival time: Not reached")
    
    # Fit lifelines Kaplan-Meier estimator
    print(f"\nFitting lifelines Kaplan-Meier estimator...")
    try:
        from lifelines import KaplanMeierFitter
        
        km_lifelines = KaplanMeierFitter()
        km_lifelines.fit(durations, events)
        
        lifelines_median = km_lifelines.median_survival_time_
        print(f"  Lifelines fitted")
        print(f"  Median survival time: {lifelines_median}")
        
        # Compare survival curves at multiple time points
        print(f"\nComparing survival probabilities:")
        print("Time (days)\tOurs\tLifelines\tDifference")
        print("-" * 50)
        
        # Test at various percentiles of follow-up time
        test_times = [
            50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000
        ]
        
        max_diff = 0
        comparison_results = []
        
        for t in test_times:
            if t <= durations.max():  # Only test times within data range
                ours = km_ours.survival_function_at_times([t])[0].item()
                theirs = extract_survival_prob_lifelines(km_lifelines, t)
                
                if theirs is not None:
                    diff = abs(ours - theirs)
                    max_diff = max(max_diff, diff)
                    comparison_results.append((t, ours, theirs, diff))
                    print(f"{t:>11}\t{ours:.4f}\t{theirs:.4f}\t{diff:.8f}")
        
        # Compare median survival times
        print(f"\nMedian survival comparison:")
        print(f"   Your implementation: {our_median}")
        print(f"   Lifelines:          {lifelines_median}")
        
        if our_median is not None and lifelines_median is not None:
            median_diff = abs(our_median - lifelines_median)
            print(f"   Difference:         {median_diff:.6f} days")
        
        # Overall validation
        print(f"\n🎯 Validation Results:")
        print(f"   Maximum difference at any time point: {max_diff:.10f}")
        
        if max_diff < 1e-10:
            print("   PERFECT MATCH! Your implementation is identical to lifelines.")
        elif max_diff < 1e-8:
            print(" EXCELLENT! Differences are within numerical precision.")
        elif max_diff < 1e-6:
            print("VERY GOOD! Minor differences likely due to floating point precision.")
        else:
            print(f"Some differences found. May need investigation.")
        
        # Create comparison plot
        try:
            create_comparison_plot(km_ours, km_lifelines, durations)
        except Exception as e:
            print(f"Could not create plot: {e}")
        
        return km_ours, km_lifelines, comparison_results
        
    except ImportError:
        print("Lifelines not available for comparison")
        return km_ours, None, None

def create_comparison_plot(km_ours, km_lifelines, durations):
    """
    Create a comparison plot of both survival curves.
    """
    print(f"\nCreating comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Overlayed survival curves
    times_ours = km_ours.timeline_.cpu().numpy()
    survival_ours = km_ours.survival_function_.cpu().numpy()
    
    # Add starting point (0, 1)
    plot_times_ours = np.concatenate([[0], times_ours])
    plot_survival_ours = np.concatenate([[1.0], survival_ours])
    
    ax1.step(plot_times_ours, plot_survival_ours, where='post', 
             linewidth=2, label='Your Implementation', color='red', alpha=0.8)
    
    # Get lifelines data
    lifelines_times = km_lifelines.survival_function_.index.values
    lifelines_survival = km_lifelines.survival_function_.iloc[:, 0].values
    
    ax1.step(lifelines_times, lifelines_survival, where='post', 
             linewidth=2, label='Lifelines', color='blue', alpha=0.6, linestyle='--')
    
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('Kaplan-Meier Survival Curves Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Add median survival lines
    our_median = km_ours.median_survival_time()
    their_median = km_lifelines.median_survival_time_
    
    if our_median:
        ax1.axvline(x=our_median, color='red', linestyle=':', alpha=0.7, 
                   label=f'Your Median: {our_median:.1f}')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    if their_median:
        ax1.axvline(x=their_median, color='blue', linestyle=':', alpha=0.7,
                   label=f'Lifelines Median: {their_median:.1f}')
    
    ax1.legend()
    
    # Plot 2: Difference plot
    test_times = np.linspace(0, durations.max(), 100)
    differences = []
    
    for t in test_times:
        ours = km_ours.survival_function_at_times([t])[0].item()
        theirs = extract_survival_prob_lifelines(km_lifelines, t)
        if theirs is not None:
            differences.append(ours - theirs)
        else:
            differences.append(0)
    
    ax2.plot(test_times, differences, color='green', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Difference (Yours - Lifelines)')
    ax2.set_title('Difference Between Implementations')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    max_abs_diff = max([abs(d) for d in differences])
    ax2.text(0.05, 0.95, f'Max |difference|: {max_abs_diff:.2e}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"   ✅ Comparison plot created successfully")

def test_stratified_analysis(df):
    """
    Test stratified analysis if categorical variables are available.
    """
    print(f"\nSTRATIFIED ANALYSIS")
    print("=" * 40)
    
    # Look for categorical variables to stratify by
    categorical_cols = []
    for col in df.columns:
        if col not in ['time', 'event'] and df[col].dtype in ['object', 'category'] or df[col].nunique() <= 5:
            categorical_cols.append(col)
    
    if not categorical_cols:
        print("   No suitable categorical variables found for stratification")
        return
    
    # Use the first categorical variable (or 'sex' if available)
    strat_col = 'sex' if 'sex' in categorical_cols else categorical_cols[0]
    print(f"   Stratifying by: {strat_col}")
    
    unique_groups = df[strat_col].unique()
    print(f"   Groups: {unique_groups}")
    
    results = {}
    
    for group in unique_groups:
        if pd.isna(group):
            continue
            
        group_data = df[df[strat_col] == group]
        durations = group_data['time'].values
        events = group_data['event'].values
        
        print(f"\n   Group '{group}': {len(group_data)} patients, {events.sum()} events")
        
        # Fit your KM estimator
        km_ours = KaplanMeierEstimator(device='cpu')
        km_ours.fit(durations, events)
        
        median_ours = km_ours.median_survival_time()
        
        # Fit lifelines KM estimator
        try:
            from lifelines import KaplanMeierFitter
            km_lifelines = KaplanMeierFitter()
            km_lifelines.fit(durations, events)
            median_lifelines = km_lifelines.median_survival_time_
            
            print(f"     Your median: {median_ours}")
            print(f"     Lifelines median: {median_lifelines}")
            
            if median_ours and median_lifelines:
                diff = abs(median_ours - median_lifelines)
                print(f"     Difference: {diff:.6f}")
            
            results[group] = {
                'ours': km_ours,
                'lifelines': km_lifelines,
                'median_ours': median_ours,
                'median_lifelines': median_lifelines
            }
            
        except ImportError:
            print(f"     Lifelines not available")
            results[group] = {'ours': km_ours}
    
    return results


if __name__ == "__main__":
    # Run simple test
    km = simple_test()
    
    # # Run known data test
    test_with_known_data()

    print("STARTING LUNG CANCER KAPLAN-MEIER VALIDATION")
    print("=" * 70)
    
    # Main comparison test
    km_ours, km_lifelines, comparison_results = test_lung_cancer_kaplan_meier()
    
    if km_ours is not None:
        print("\nMain analysis completed successfully!")
        
        # Try stratified analysis if we have the data
        try:
            df = load_lung_cancer_data()
            if len(df.columns) > 2:  # More than just time and event
                test_stratified_analysis(df)
        except Exception as e:
            print(f"Stratified analysis skipped: {e}")
    
    print("\n" + "=" * 70)
    print("LUNG CANCER VALIDATION COMPLETED!")
    print("=" * 70)
    
    print("\n" + "="*50)
    print("Tests completed!")