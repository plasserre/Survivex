"""
Debug: Check Python's full B matrix computation
Save as: debug_b_matrix.py
"""

import numpy as np
import pandas as pd
from survivex.models.andersen_gill import AndersenGillModel

# Load data
cgd = pd.read_csv("cgd_data.csv")
treat_map = {'placebo': 0, 'rIFN-g': 1}
cgd['treat'] = cgd['treat'].map(treat_map)

for col in ['id', 'treat', 'age', 'tstart', 'tstop', 'status']:
    cgd[col] = pd.to_numeric(cgd[col])

subject_ids = cgd['id'].values
X = cgd[['treat', 'age']].values
time_start = cgd['tstart'].values
time_stop = cgd['tstop'].values
status = cgd['status'].values

# Fit model
model = AndersenGillModel(tie_method='efron')
model.fit(X=X, time_start=time_start, time_stop=time_stop, 
          events=status, subject_id=subject_ids)

print("=" * 80)
print("PYTHON B MATRIX DIAGNOSTIC")
print("=" * 80)

# Manually compute B matrix using the robust variance code
X_sorted_std = model.cox_model_.X_sorted_.cpu().numpy()
durations_sorted = model.cox_model_.durations_sorted_.cpu().numpy()
events_sorted = model.cox_model_.events_sorted_.cpu().numpy()
sorted_indices = model.cox_model_.sorted_indices_.cpu().numpy()
start_times_sorted = model.cox_model_.start_times_sorted_.cpu().numpy()

cluster_id_sorted = subject_ids[sorted_indices]
beta = model.coefficients_

# Convert to centered
X_std_vals = model.cox_model_.X_std_.cpu().numpy()
X_centered = X_sorted_std * X_std_vals

# Compute risk scores
exp_eta = np.exp(X_centered @ beta)

unique_clusters = np.unique(subject_ids)
n_clusters = len(unique_clusters)
score_matrix = np.zeros((n_clusters, 2))

event_times = np.unique(durations_sorted[events_sorted == 1])

print(f"\nData info:")
print(f"  N clusters: {n_clusters}")
print(f"  N observations: {len(X)}")
print(f"  N events: {np.sum(events_sorted)}")
print(f"  N unique event times: {len(event_times)}")

# Accumulate scores
for t in event_times:
    at_risk = (start_times_sorted < t) & (durations_sorted >= t)
    at_event = (durations_sorted == t) & (events_sorted == 1)
    
    if not np.any(at_event):
        continue
    
    event_indices = np.where(at_event)[0]
    n_events = len(event_indices)
    
    risk_exp = exp_eta[at_risk]
    risk_X = X_centered[at_risk]
    sum_risk_exp = np.sum(risk_exp)
    
    if sum_risk_exp == 0:
        continue
    
    if n_events > 1:  # Efron
        event_exp = exp_eta[at_event]
        event_X = X_centered[at_event]
        sum_event_exp = np.sum(event_exp)
        
        for k in range(n_events):
            event_idx = event_indices[k]
            frac = k / n_events
            denom = sum_risk_exp - frac * sum_event_exp
            
            if denom <= 0:
                continue
            
            num = (np.sum(risk_X * risk_exp[:, np.newaxis], axis=0) - 
                   frac * np.sum(event_X * event_exp[:, np.newaxis], axis=0))
            expected_X = num / denom
            
            score_contrib = X_centered[event_idx] - expected_X
            cluster_idx = np.where(unique_clusters == cluster_id_sorted[event_idx])[0][0]
            score_matrix[cluster_idx] += score_contrib
    else:
        expected_X = np.sum(risk_X * risk_exp[:, np.newaxis], axis=0) / sum_risk_exp
        
        for event_idx in event_indices:
            score_contrib = X_centered[event_idx] - expected_X
            cluster_idx = np.where(unique_clusters == cluster_id_sorted[event_idx])[0][0]
            score_matrix[cluster_idx] += score_contrib

# Compute B
B = score_matrix.T @ score_matrix

print(f"\n=== PYTHON'S B MATRIX ===")
print(B)

print(f"\n=== PYTHON: Score sums by subject (first 10) ===")
for i in range(min(10, n_clusters)):
    subj_id = unique_clusters[i]
    print(f"Subject {subj_id}: treat={score_matrix[i,0]:.6f}, age={score_matrix[i,1]:.6f}")

print(f"\n=== Max absolute scores ===")
print(f"Max treat score: {np.max(np.abs(score_matrix[:,0])):.6f}")
print(f"Max age score: {np.max(np.abs(score_matrix[:,1])):.6f}")

print(f"\n=== COMPARISON WITH R ===")
print("R's B matrix:")
print("  treat: 20.596046")
print("  age: 6982.240623")
print(f"\nPython's B matrix:")
print(f"  treat: {B[0,0]:.6f}")
print(f"  age: {B[1,1]:.6f}")
print(f"\nRatio (Python/R):")
print(f"  treat: {B[0,0]/20.596046:.4f}")
print(f"  age: {B[1,1]/6982.240623:.4f}")

print("\n" + "=" * 80)
print("TESTING: Transform beta for original X scale - CORRECTED")
print("=" * 80)

# Get original X (in sorted order)
X_original_sorted = X[sorted_indices]
X_mean_vals = model.cox_model_.X_mean_.cpu().numpy()

# CORRECTED: beta_original = beta_standardized / std (DIVIDE, not multiply!)
beta_original = beta / X_std_vals  # KEY FIX!

print(f"Beta (standardized): {beta}")
print(f"X std: {X_std_vals}")
print(f"Beta (original scale - CORRECTED): {beta_original}")

# Recompute with transformed beta
exp_eta_correct = np.exp((X_original_sorted - X_mean_vals) @ beta_original)

score_matrix_correct = np.zeros((n_clusters, 2))

for t in event_times:
    at_risk = (start_times_sorted < t) & (durations_sorted >= t)
    at_event = (durations_sorted == t) & (events_sorted == 1)
    
    if not np.any(at_event):
        continue
    
    event_indices = np.where(at_event)[0]
    
    risk_exp = exp_eta_correct[at_risk]
    risk_X_orig = X_original_sorted[at_risk]
    sum_risk_exp = np.sum(risk_exp)
    
    if sum_risk_exp == 0:
        continue
    
    expected_X = np.sum(risk_X_orig * risk_exp[:, np.newaxis], axis=0) / sum_risk_exp
    
    for event_idx in event_indices:
        score_contrib = X_original_sorted[event_idx] - expected_X
        cluster_idx = np.where(unique_clusters == cluster_id_sorted[event_idx])[0][0]
        score_matrix_correct[cluster_idx] += score_contrib

B_correct = score_matrix_correct.T @ score_matrix_correct

print(f"\nB matrix (CORRECTED):")
print(B_correct)

print(f"\n=== First 10 subjects (CORRECTED) ===")
r_scores_treat = [0.968263, -1.318368, 0, 0, -0.038315, 0, -0.026874, 0, -0.087166, 0]
for i in range(min(10, n_clusters)):
    subj_id = unique_clusters[i]
    print(f"Subject {subj_id}: Python={score_matrix_correct[i,0]:.6f}, R={r_scores_treat[i]:.6f}")

print(f"\n=== ULTIMATE COMPARISON ===")
print(f"R B: treat=20.596, age=6982.24")
print(f"Python (CORRECTED) B: treat={B_correct[0,0]:.2f}, age={B_correct[1,1]:.2f}")
print(f"Ratio: treat={B_correct[0,0]/20.596:.4f}, age={B_correct[1,1]/6982.24:.4f}")