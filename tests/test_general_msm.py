

import numpy as np
import sys
sys.path.insert(0, '/mnt/project')

from survivex.models.multi_state import (
    TransitionMatrix,
    MultiStateData,
    create_illness_death_matrix,
    prepare_multistate_data_simple,
    MultiStateAalenJohansen,
    print_data_summary
)


def simulate_illness_death_data(n: int = 200, seed: int = 42):
    """
    Simulate data from an illness-death model.
    
    Model: Healthy -> Illness -> Death
                  \-> Death (direct)
    """
    np.random.seed(seed)
    
    # Simulate transition times
    time_healthy_to_illness = np.random.exponential(50, n)
    time_healthy_to_death = np.random.exponential(100, n)
    time_illness_to_death = np.random.exponential(30, n)
    
    # For each subject, determine their path
    subjects_data = []
    
    for i in range(n):
        # Check if they die directly or get ill first
        if time_healthy_to_death[i] < time_healthy_to_illness[i]:
            # Direct path: Healthy -> Death
            subjects_data.append({
                'id': i,
                'from': 0,  # Healthy
                'to': 2,    # Death
                'time': time_healthy_to_death[i],
                'status': 1
            })
        else:
            # Path through illness: Healthy -> Illness -> Death
            time_to_illness = time_healthy_to_illness[i]
            time_to_death = time_to_illness + time_illness_to_death[i]
            
            # Apply censoring at time 200
            if time_to_death > 200:
                # Censored while ill
                subjects_data.append({
                    'id': i,
                    'from': 0,  # Healthy
                    'to': 1,    # Illness
                    'time': time_to_illness,
                    'status': 1
                })
            else:
                # Complete path: both transitions
                subjects_data.append({
                    'id': i,
                    'from': 0,  # Healthy
                    'to': 1,    # Illness
                    'time': time_to_illness,
                    'status': 1
                })
                subjects_data.append({
                    'id': i,
                    'from': 1,  # Illness
                    'to': 2,    # Death
                    'time': time_to_death,
                    'status': 1
                })
    
    return subjects_data


def prepare_illness_death_data(subjects_data):
    """
    Convert simulated data to proper multi-state long format.
    """
    # Create transition matrix
    trans_matrix = create_illness_death_matrix(with_recovery=False)
    
    # Build long-format data
    all_rows = []
    
    # Group by subject
    subjects_dict = {}
    for s in subjects_data:
        sid = s['id']
        if sid not in subjects_dict:
            subjects_dict[sid] = []
        subjects_dict[sid].append(s)
    
    # Process each subject
    for subj_id, subj_transitions in subjects_dict.items():
        # Sort by time
        subj_transitions = sorted(subj_transitions, key=lambda x: x['time'])
        
        # Track current state and time
        current_state = 0  # Start healthy
        current_time = 0.0
        
        for trans in subj_transitions:
            next_state = trans['to']
            next_time = trans['time']
            
            # Create rows for all possible transitions from current state
            for target_state in range(trans_matrix.n_states):
                trans_num = trans_matrix.get_transition_number(current_state, target_state)
                
                if trans_num > 0:  # This transition is possible
                    # Did this transition happen?
                    status = 1 if target_state == next_state else 0
                    
                    all_rows.append({
                        'subject_id': subj_id,
                        'from_state': current_state,
                        'to_state': target_state,
                        'time_start': current_time,
                        'time_stop': next_time,
                        'status': status,
                        'transition_number': trans_num
                    })
            
            # Update state
            current_state = next_state
            current_time = next_time
        
        # If not in absorbing state, add censoring rows
        if current_state != 2:  # Not dead
            # Censored at time 200
            censor_time = 200.0
            
            for target_state in range(trans_matrix.n_states):
                trans_num = trans_matrix.get_transition_number(current_state, target_state)
                
                if trans_num > 0:
                    all_rows.append({
                        'subject_id': subj_id,
                        'from_state': current_state,
                        'to_state': target_state,
                        'time_start': current_time,
                        'time_stop': censor_time,
                        'status': 0,  # Censored
                        'transition_number': trans_num
                    })
    
    # Convert to arrays
    data = MultiStateData(
        subject_id=np.array([r['subject_id'] for r in all_rows]),
        from_state=np.array([r['from_state'] for r in all_rows]),
        to_state=np.array([r['to_state'] for r in all_rows]),
        time_start=np.array([r['time_start'] for r in all_rows]),
        time_stop=np.array([r['time_stop'] for r in all_rows]),
        status=np.array([r['status'] for r in all_rows]),
        transition_number=np.array([r['transition_number'] for r in all_rows])
    )
    
    return data, trans_matrix


def test_illness_death_model():
    """Test Aalen-Johansen with illness-death model."""
    print("\n" + "="*70)
    print("STEP 3: ILLNESS-DEATH MODEL TEST")
    print("="*70)
    
    # Simulate data
    print("\nSimulating illness-death data...")
    subjects_data = simulate_illness_death_data(n=200, seed=42)
    
    print(f"Simulated {len(set(s['id'] for s in subjects_data))} subjects")
    print(f"Total transition events: {len(subjects_data)}")
    
    # Count transitions
    trans_counts = {}
    for s in subjects_data:
        key = (s['from'], s['to'])
        trans_counts[key] = trans_counts.get(key, 0) + 1
    
    print("\nObserved transitions:")
    state_names = ["Healthy", "Illness", "Death"]
    for (from_s, to_s), count in sorted(trans_counts.items()):
        print(f"  {state_names[from_s]} -> {state_names[to_s]}: {count}")
    
    # Prepare multi-state data
    print("\nPreparing multi-state data...")
    ms_data, trans_matrix = prepare_illness_death_data(subjects_data)
    
    print_data_summary(ms_data, trans_matrix)
    
    # Check if counts match
    print("\n" + "="*70)
    print("VERIFICATION: Event counts should match simulation")
    print("="*70)
    for (from_s, to_s), count in sorted(trans_counts.items()):
        trans_num = trans_matrix.get_transition_number(from_s, to_s)
        mask = (ms_data.transition_number == trans_num) & (ms_data.status == 1)
        observed_count = np.sum(mask)
        match = "✓" if observed_count == count else "✗"
        print(f"{match} {state_names[from_s]} -> {state_names[to_s]}: "
              f"Expected {count}, Got {observed_count}")
    
    # Fit Aalen-Johansen
    print("\n" + "="*70)
    print("Fitting Aalen-Johansen estimator...")
    print("="*70)
    
    ajest = MultiStateAalenJohansen(trans_matrix)
    result = ajest.fit(ms_data, start_state=0)
    
    print(f"Estimated at {len(result.times)} time points")
    print(f"Time range: {result.times.min():.1f} to {result.times.max():.1f}")
    
    # Display results
    print("\n" + "="*70)
    print("STATE OCCUPATION PROBABILITIES")
    print("="*70)
    print(f"{'Time':<10} {'P(Healthy)':<15} {'P(Illness)':<15} {'P(Death)':<15} {'Sum':<10}")
    print("-"*70)
    
    time_points = [0, 20, 40, 60, 80, 100, 150, 200]
    for t in time_points:
        idx = np.searchsorted(result.times, t)
        if idx >= len(result.times):
            idx = len(result.times) - 1
        
        p_healthy = result.state_probs[0][idx]
        p_illness = result.state_probs[1][idx]
        p_death = result.state_probs[2][idx]
        total = p_healthy + p_illness + p_death
        
        print(f"{t:<10.0f} {p_healthy:<15.6f} {p_illness:<15.6f} {p_death:<15.6f} {total:<10.6f}")
    
    # Verify probabilities sum to 1
    print("\n" + "="*70)
    print("VERIFICATION: Probabilities Sum to 1.0")
    print("="*70)
    
    all_sums = []
    for idx in range(len(result.times)):
        total = sum(result.state_probs[j][idx] for j in range(3))
        all_sums.append(total)
    
    max_deviation = max(abs(s - 1.0) for s in all_sums)
    print(f"Maximum deviation from sum=1.0: {max_deviation:.10f}")
    
    if max_deviation < 1e-10:
        print("✓✓✓ Probabilities sum to 1.0 (perfect!)")
    elif max_deviation < 1e-6:
        print("✓✓ Probabilities sum to 1.0 (excellent!)")
    else:
        print("✗ Probabilities don't sum to 1.0 properly")
    
    # Show transition probabilities
    print("\n" + "="*70)
    print("TRANSITION PROBABILITIES FROM HEALTHY STATE")
    print("="*70)
    print(f"{'Time':<10} {'P(H->H)':<15} {'P(H->I)':<15} {'P(H->D)':<15} {'Sum':<10}")
    print("-"*70)
    
    for t in [0, 50, 100, 150, 200]:
        idx = np.searchsorted(result.times, t)
        if idx >= len(result.times):
            idx = len(result.times) - 1
        
        p_hh = result.transition_probs[(0, 0)][idx]
        p_hi = result.transition_probs[(0, 1)][idx]
        p_hd = result.transition_probs[(0, 2)][idx]
        total = p_hh + p_hi + p_hd
        
        print(f"{t:<10.0f} {p_hh:<15.6f} {p_hi:<15.6f} {p_hd:<15.6f} {total:<10.6f}")
    
    return result, trans_matrix, ms_data, subjects_data


def print_r_comparison_script(subjects_data):
    """Print R script for validation (don't save to file)."""
    print("\n" + "="*70)
    print("R COMPARISON SCRIPT")
    print("="*70)
    
    # Create data frame text
    print("\n# 1. Create the data in R:")
    print("```R")
    print("# Illness-death simulated data")
    print("library(survival)")
    print("library(mstate)")
    print("")
    print("# Raw transition data")
    print("trans_data <- data.frame(")
    print("  id = c(" + ", ".join([str(s['id']) for s in subjects_data[:10]]) + ", ...),")
    print("  from = c(" + ", ".join([str(s['from']) for s in subjects_data[:10]]) + ", ...),")
    print("  to = c(" + ", ".join([str(s['to']) for s in subjects_data[:10]]) + ", ...),")
    print("  time = c(" + ", ".join([f"{s['time']:.2f}" for s in subjects_data[:10]]) + ", ...),")
    print("  status = c(" + ", ".join([str(s['status']) for s in subjects_data[:10]]) + ", ...)")
    print(")")
    print("")
    print("# Transition matrix for illness-death (without recovery)")
    print("tmat <- transMat(x = list(c(2, 3), c(3), c()), names = c('Healthy', 'Illness', 'Death'))")
    print("print(tmat)")
    print("")
    print("# This creates:")
    print("#          to")
    print("# from     Healthy Illness Death")
    print("#   Healthy     NA       1     2")
    print("#   Illness     NA      NA     3")
    print("#   Death       NA      NA    NA")
    print("")
    print("# To compare with our results, you would need to:")
    print("# 1. Convert to wide format (one row per subject)")
    print("# 2. Use msprep() to create long format")
    print("# 3. Fit survfit() with the multi-state formula")
    print("# 4. Extract state probabilities")
    print("")
    print("# Example (simplified):")
    print("# ms_fit <- survfit(Surv(Tstart, Tstop, status) ~ 1, data = ms_long, id = id)")
    print("# summary(ms_fit, times = c(0, 50, 100, 150, 200))")
    print("```")
    
    print("\n" + "="*70)
    print("Note: Full R comparison requires converting data to wide format first,")
    print("then using mstate::msprep() for proper long format preparation.")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("MULTI-STATE SURVIVAL ANALYSIS - STEP 3")
    print("Testing True Multi-State Model (Illness-Death)")
    print("="*70)
    
    try:
        # Test illness-death model
        result, trans_matrix, ms_data, subjects_data = test_illness_death_model()
        
        # Print R comparison script
        print_r_comparison_script(subjects_data)
        
        print("\n" + "="*70)
        print("✓ Step 3 Complete!")
        print("Next: Validate against R mstate package")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED:")
        print("="*70)
        import traceback
        traceback.print_exc()