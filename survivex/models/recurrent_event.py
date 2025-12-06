

"""
Recurrent Event Data Utilities
"""

import numpy as np
from typing import Optional, List, Tuple
import warnings

from .multi_state import MultiStateData


def prepare_recurrent_data_simple(
    subject_ids: np.ndarray,
    event_times: List[np.ndarray],
    event_status: List[np.ndarray],
    covariates: Optional[np.ndarray] = None
) -> MultiStateData:
    """
    Convert simple recurrent event data to counting process (start, stop] format.
    
    Parameters
    ----------
    subject_ids : array of subject IDs
    event_times : list of time arrays (one per subject)
    event_status : list of status arrays (1=event, 0=censored)
    covariates : optional covariate matrix
    
    Returns
    -------
    MultiStateData in counting process format
    """
    n_subjects = len(subject_ids)
    
    if len(event_times) != n_subjects:
        raise ValueError(f"event_times must have {n_subjects} elements")
    if len(event_status) != n_subjects:
        raise ValueError(f"event_status must have {n_subjects} elements")
    if covariates is not None and covariates.shape[0] != n_subjects:
        raise ValueError(f"covariates must have {n_subjects} rows")
    
    all_subject_ids = []
    all_from_states = []
    all_to_states = []
    all_time_start = []
    all_time_stop = []
    all_status = []
    all_transition_numbers = []
    all_covariates = [] if covariates is not None else None
    
    for subj_idx, subj_id in enumerate(subject_ids):
        times = event_times[subj_idx]
        status = event_status[subj_idx]
        
        if len(times) != len(status):
            raise ValueError(f"Subject {subj_id}: times and status must have same length")
        
        if len(times) == 0:
            continue
        
        if not np.all(np.diff(times) >= 0):
            raise ValueError(f"Subject {subj_id}: times must be non-decreasing")
        
        prev_time = 0.0
        for k in range(len(times)):
            event_number = k + 1
            
            all_subject_ids.append(subj_id)
            all_from_states.append(event_number - 1)
            all_to_states.append(event_number)
            all_time_start.append(prev_time)
            all_time_stop.append(times[k])
            all_status.append(status[k])
            all_transition_numbers.append(event_number)
            
            if covariates is not None:
                all_covariates.append(covariates[subj_idx])
            
            if status[k] == 1:
                prev_time = times[k]
            else:
                break
    
    return MultiStateData(
        subject_id=np.array(all_subject_ids),
        from_state=np.array(all_from_states, dtype=int),
        to_state=np.array(all_to_states, dtype=int),
        time_start=np.array(all_time_start, dtype=float),
        time_stop=np.array(all_time_stop, dtype=float),
        status=np.array(all_status, dtype=int),
        transition_number=np.array(all_transition_numbers, dtype=int),
        covariates=np.array(all_covariates) if all_covariates else None
    )


def prepare_recurrent_data_gap_time(
    subject_ids: np.ndarray,
    event_times: List[np.ndarray],
    event_status: List[np.ndarray],
    covariates: Optional[np.ndarray] = None
) -> MultiStateData:
    """
    Convert to gap time format (time since last event).
    """
    n_subjects = len(subject_ids)
    
    if len(event_times) != n_subjects:
        raise ValueError(f"event_times must have {n_subjects} elements")
    if len(event_status) != n_subjects:
        raise ValueError(f"event_status must have {n_subjects} elements")
    if covariates is not None and covariates.shape[0] != n_subjects:
        raise ValueError(f"covariates must have {n_subjects} rows")
    
    all_subject_ids = []
    all_from_states = []
    all_to_states = []
    all_time_start = []
    all_time_stop = []
    all_status = []
    all_transition_numbers = []
    all_covariates = [] if covariates is not None else None
    
    for subj_idx, subj_id in enumerate(subject_ids):
        times = event_times[subj_idx]
        status = event_status[subj_idx]
        
        if len(times) != len(status):
            raise ValueError(f"Subject {subj_id}: times and status must have same length")
        
        if len(times) == 0:
            continue
        
        prev_time = 0.0
        for k in range(len(times)):
            event_number = k + 1
            gap_duration = times[k] - prev_time
            
            all_subject_ids.append(subj_id)
            all_from_states.append(event_number - 1)
            all_to_states.append(event_number)
            all_time_start.append(0.0)  # Gap time starts at 0
            all_time_stop.append(gap_duration)
            all_status.append(status[k])
            all_transition_numbers.append(event_number)
            
            if covariates is not None:
                all_covariates.append(covariates[subj_idx])
            
            if status[k] == 1:
                prev_time = times[k]
            else:
                break
    
    return MultiStateData(
        subject_id=np.array(all_subject_ids),
        from_state=np.array(all_from_states, dtype=int),
        to_state=np.array(all_to_states, dtype=int),
        time_start=np.array(all_time_start, dtype=float),
        time_stop=np.array(all_time_stop, dtype=float),
        status=np.array(all_status, dtype=int),
        transition_number=np.array(all_transition_numbers, dtype=int),
        covariates=np.array(all_covariates) if all_covariates else None
    )




"""
PWP Total Time (PWP-TT) Model for Recurrent Events

The Prentice, Williams, and Peterson (PWP) Total Time model stratifies by
event number, allowing different baseline hazards for each event, while
using total time since study entry.
"""

import numpy as np
import torch
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
import warnings

from .cox_ph import CoxPHModel


@dataclass
class PWPTTResult:
    """Results from PWP Total Time model."""
    coefficients: np.ndarray
    standard_errors: np.ndarray
    hazard_ratios: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    robust_variance: np.ndarray
    naive_standard_errors: np.ndarray
    log_likelihood: float
    n_subjects: int
    n_observations: int
    n_events: int
    n_strata: int
    events_per_stratum: dict
    
    def summary(self) -> str:
        """Generate summary table."""
        s = "\n" + "="*80 + "\n"
        s += "PWP Total Time Model for Recurrent Events - Results Summary\n"
        s += "="*80 + "\n"
        s += f"Number of subjects: {self.n_subjects}\n"
        s += f"Number of observations: {self.n_observations}\n"
        s += f"Number of events: {self.n_events}\n"
        s += f"Number of strata: {self.n_strata}\n"
        s += f"Log-likelihood: {self.log_likelihood:.4f}\n"
        
        s += "\nEvents per stratum:\n"
        for stratum, n_events in self.events_per_stratum.items():
            s += f"  Event {stratum}: {n_events} events\n"
        
        s += "\n" + "-"*80 + "\n"
        s += f"{'Variable':<12} {'Coef':<10} {'SE(rob)':<10} {'HR':<10} {'z':<8} {'p-value':<10}\n"
        s += "-"*80 + "\n"
        
        for i in range(len(self.coefficients)):
            s += f"{'X' + str(i):<12} "
            s += f"{self.coefficients[i]:>9.5f} "
            s += f"{self.standard_errors[i]:>9.5f} "
            s += f"{self.hazard_ratios[i]:>9.5f} "
            s += f"{self.z_scores[i]:>7.3f} "
            s += f"{self.p_values[i]:>9.6f}\n"
        
        s += "="*80 + "\n"
        s += "Note: Stratified by event number (different baseline hazard per event)\n"
        s += "      Time scale: Total time since study entry\n"
        s += "      SE(rob) = Robust standard errors\n"
        return s


class StratifiedCoxPH:
    """
    Stratified Cox Proportional Hazards model.
    
    Fits separate baseline hazards for each stratum while constraining
    covariate effects to be the same across strata.
    """
    
    def __init__(self, tie_method: str = 'efron', max_iter: int = 50, tol: float = 1e-6):
        self.tie_method = tie_method
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device('cpu')  # Use CPU for float64
        
    def fit(self, X, durations, events, strata, start_times=None):
        """Fit stratified Cox model."""
        # Convert to tensors
        X = self._to_tensor(X)
        durations = self._to_tensor(durations)
        events = self._to_tensor(events)
        strata = self._to_tensor(strata).long()
        
        if start_times is not None:
            start_times = self._to_tensor(start_times)
        else:
            start_times = torch.zeros_like(durations)
        
        n_samples, n_features = X.shape
        
        # Standardize covariates
        self.X_mean_ = torch.mean(X, dim=0)
        self.X_std_ = torch.std(X, dim=0)
        self.X_std_[self.X_std_ == 0] = 1.0
        X_standardized = (X - self.X_mean_) / self.X_std_
        
        # Get unique strata
        unique_strata = torch.unique(strata)
        
        # Initialize coefficients
        beta = torch.zeros(n_features, device=self.device, dtype=torch.float64)
        
        # Newton-Raphson optimization
        converged = False
        
        for iteration in range(self.max_iter):
            # Compute derivatives across all strata
            log_lik = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            gradient = torch.zeros(n_features, device=self.device, dtype=torch.float64)
            hessian = torch.zeros((n_features, n_features), device=self.device, dtype=torch.float64)
            
            # Process each stratum separately
            for stratum_val in unique_strata:
                mask = strata == stratum_val
                
                if not torch.any(mask):
                    continue
                
                # Get data for this stratum
                X_stratum = X_standardized[mask]
                dur_stratum = durations[mask]
                evt_stratum = events[mask]
                start_stratum = start_times[mask]
                
                # Sort by duration within stratum
                sort_idx = torch.argsort(dur_stratum)
                X_sorted = X_stratum[sort_idx]
                dur_sorted = dur_stratum[sort_idx]
                evt_sorted = evt_stratum[sort_idx]
                start_sorted = start_stratum[sort_idx]
                
                # Compute derivatives for this stratum
                ll, grad, hess = self._compute_derivatives_stratum(
                    beta, X_sorted, dur_sorted, evt_sorted, start_sorted
                )
                
                log_lik += ll
                gradient += grad
                hessian += hess
            
            # Check convergence
            gradient_norm = torch.norm(gradient)
            if gradient_norm < self.tol:
                converged = True
                break
            
            # Newton-Raphson step
            try:
                delta = torch.linalg.solve(hessian, gradient)
                beta = beta - delta
            except RuntimeError:
                warnings.warn("Hessian singular, adding regularization")
                reg = torch.eye(n_features, device=self.device) * 1e-6
                delta = torch.linalg.solve(hessian + reg, gradient)
                beta = beta - delta
        
        if not converged:
            warnings.warn(f"Did not converge in {self.max_iter} iterations")
        
        # Transform to original scale
        beta_original = beta / self.X_std_
        
        # Compute final likelihood and variance
        log_lik_final = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        hessian_final = torch.zeros((n_features, n_features), device=self.device, dtype=torch.float64)
        
        for stratum_val in unique_strata:
            mask = strata == stratum_val
            if not torch.any(mask):
                continue
            
            X_stratum = X_standardized[mask]
            dur_stratum = durations[mask]
            evt_stratum = events[mask]
            start_stratum = start_times[mask]
            
            sort_idx = torch.argsort(dur_stratum)
            X_sorted = X_stratum[sort_idx]
            dur_sorted = dur_stratum[sort_idx]
            evt_sorted = evt_stratum[sort_idx]
            start_sorted = start_stratum[sort_idx]
            
            ll, _, hess = self._compute_derivatives_stratum(
                beta, X_sorted, dur_sorted, evt_sorted, start_sorted
            )
            log_lik_final += ll
            hessian_final += hess
        
        # Standard errors
        try:
            hessian_inv = torch.linalg.inv(-hessian_final)
            variance_matrix = hessian_inv / (self.X_std_.unsqueeze(1) * self.X_std_.unsqueeze(0))
            standard_errors = torch.sqrt(torch.diag(variance_matrix))
        except RuntimeError:
            warnings.warn("Could not compute standard errors")
            standard_errors = torch.full_like(beta_original, float('nan'))
            variance_matrix = torch.eye(len(beta_original), device=self.device, dtype=torch.float64) * float('nan')
        
        # Store results
        self.coefficients_ = beta_original.cpu().numpy()
        self.standard_errors_ = standard_errors.cpu().numpy()
        self.variance_covariance_matrix_ = variance_matrix.cpu().numpy()
        self.log_likelihood_ = log_lik_final.item()
        
        # Store original data for robust variance
        self.X_ = X.cpu().numpy()
        self.durations_ = durations.cpu().numpy()
        self.events_ = events.cpu().numpy()
        self.strata_ = strata.cpu().numpy()
        self.start_times_ = start_times.cpu().numpy()
        
        return self
    
    def _compute_derivatives_stratum(self, beta, X, durations, events, start_times):
        """Compute derivatives for a single stratum."""
        n_samples, n_features = X.shape
        
        risk_scores = torch.exp(torch.matmul(X, beta))
        
        log_likelihood = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        gradient = torch.zeros(n_features, device=self.device, dtype=torch.float64)
        hessian = torch.zeros((n_features, n_features), device=self.device, dtype=torch.float64)
        
        unique_times = torch.unique(durations[events == 1])
        
        for t in unique_times:
            at_risk_mask = (start_times < t) & (durations >= t)
            event_mask = (durations == t) & (events == 1)
            
            if not torch.any(event_mask):
                continue
            
            d_t = torch.sum(event_mask).item()
            
            risk_at_risk = risk_scores[at_risk_mask]
            X_at_risk = X[at_risk_mask]
            X_events = X[event_mask]
            risk_events = risk_scores[event_mask]
            
            if self.tie_method == 'efron':
                sum_risk_at_risk = torch.sum(risk_at_risk)
                sum_risk_events = torch.sum(risk_events)
                
                log_lik_numerator = torch.sum(torch.matmul(X_events, beta))
                log_lik_denominator = 0.0
                
                for l in range(d_t):
                    denom_l = sum_risk_at_risk - (l / d_t) * sum_risk_events
                    log_lik_denominator += torch.log(denom_l)
                
                log_likelihood += log_lik_numerator - log_lik_denominator
                
                gradient += torch.sum(X_events, dim=0)
                
                for l in range(d_t):
                    denom_l = sum_risk_at_risk - (l / d_t) * sum_risk_events
                    
                    weighted_X_risk = torch.sum(X_at_risk * risk_at_risk.unsqueeze(1), dim=0)
                    weighted_X_events = (l / d_t) * torch.sum(X_events * risk_events.unsqueeze(1), dim=0)
                    weighted_mean_X = (weighted_X_risk - weighted_X_events) / denom_l
                    
                    gradient -= weighted_mean_X
                    
                    # Hessian
                    weighted_XX_risk = torch.sum(
                        X_at_risk.unsqueeze(2) * X_at_risk.unsqueeze(1) * risk_at_risk.unsqueeze(1).unsqueeze(2),
                        dim=0
                    )
                    weighted_XX_events = (l / d_t) * torch.sum(
                        X_events.unsqueeze(2) * X_events.unsqueeze(1) * risk_events.unsqueeze(1).unsqueeze(2),
                        dim=0
                    )
                    weighted_mean_XX = (weighted_XX_risk - weighted_XX_events) / denom_l
                    
                    variance_matrix = weighted_mean_XX - torch.outer(weighted_mean_X, weighted_mean_X)
                    hessian -= variance_matrix
        
        return log_likelihood, gradient, hessian
    
    def compute_robust_variance(self, cluster_id):
        """Compute robust variance with clustering."""
        X = self.X_
        durations = self.durations_
        events = self.events_
        strata = self.strata_
        start_times = self.start_times_
        beta = self.coefficients_
        
        n, p = X.shape
        I_inv = self.variance_covariance_matrix_
        
        X_centered = X - self.X_mean_.cpu().numpy()
        unique_clusters = np.unique(cluster_id)
        n_clusters = len(unique_clusters)
        unique_strata = np.unique(strata)
        
        score_matrix = np.zeros((n_clusters, p))
        
        # For each stratum
        for stratum_val in unique_strata:
            mask_stratum = strata == stratum_val
            if not np.any(mask_stratum):
                continue
            
            X_stratum = X_centered[mask_stratum]
            dur_stratum = durations[mask_stratum]
            evt_stratum = events[mask_stratum]
            start_stratum = start_times[mask_stratum]
            cluster_stratum = cluster_id[mask_stratum]
            
            exp_eta = np.exp(X_stratum @ beta)
            event_times = np.unique(dur_stratum[evt_stratum == 1])
            
            for t in event_times:
                at_risk = (start_stratum < t) & (dur_stratum >= t)
                at_event = (dur_stratum == t) & (evt_stratum == 1)
                
                if not np.any(at_event):
                    continue
                
                risk_exp_eta = exp_eta[at_risk]
                risk_X = X_stratum[at_risk]
                sum_exp_eta = np.sum(risk_exp_eta)
                
                if sum_exp_eta == 0:
                    continue
                
                weighted_X = np.sum(risk_X * risk_exp_eta[:, np.newaxis], axis=0) / sum_exp_eta
                
                event_indices = np.where(at_event)[0]
                for event_idx in event_indices:
                    # Get global index
                    global_indices = np.where(mask_stratum)[0]
                    global_event_idx = global_indices[event_idx]
                    
                    score_contrib = X_centered[global_event_idx] - weighted_X
                    cluster_idx = np.where(unique_clusters == cluster_id[global_event_idx])[0][0]
                    score_matrix[cluster_idx] += score_contrib
        
        B = score_matrix.T @ score_matrix
        robust_var = I_inv @ B @ I_inv
        
        return robust_var
    
    def _to_tensor(self, data):
        """Convert to tensor."""
        if isinstance(data, torch.Tensor):
            if data.device.type == 'mps':
                data = data.cpu()
            return data.to(self.device).double()
        else:
            return torch.tensor(data, device=self.device, dtype=torch.float64)


class PWPTTModel:
    """PWP Total Time model for recurrent events."""
    
    def __init__(self, tie_method: str = 'efron', alpha: float = 0.05):
        self.tie_method = tie_method
        self.alpha = alpha
        self._is_fitted = False
        self.cox_model_ = None
        
    def fit_simple(self, subject_ids, event_times, event_status, covariates):
        """Fit using simple format."""
        from .recurrent_event import prepare_recurrent_data_simple
        
        data = prepare_recurrent_data_simple(
            subject_ids=subject_ids,
            event_times=event_times,
            event_status=event_status,
            covariates=covariates
        )
        
        return self.fit(
            X=data.covariates,
            time_start=data.time_start,
            time_stop=data.time_stop,
            events=data.status,
            subject_id=data.subject_id,
            stratum=data.transition_number
        )
    
    def fit(self, X, time_start, time_stop, events, subject_id, stratum):
        """Fit stratified model."""
        self.n_subjects_ = len(np.unique(subject_id))
        self.n_observations_ = len(X)
        self.n_events_ = np.sum(events)
        self.n_strata_ = len(np.unique(stratum))
        
        self.events_per_stratum_ = {}
        for s in np.unique(stratum):
            mask = stratum == s
            self.events_per_stratum_[int(s)] = int(np.sum(events[mask]))
        
        # Fit stratified Cox
        self.cox_model_ = StratifiedCoxPH(tie_method=self.tie_method)
        self.cox_model_.fit(X, time_stop, events, stratum, time_start)
        
        # Robust variance
        self.robust_variance_ = self.cox_model_.compute_robust_variance(subject_id)
        
        self.coefficients_ = self.cox_model_.coefficients_
        self.naive_standard_errors_ = self.cox_model_.standard_errors_
        self.standard_errors_ = np.sqrt(np.diag(self.robust_variance_))
        self.log_likelihood_ = self.cox_model_.log_likelihood_
        
        self._is_fitted = True
        self.result_ = self._create_result_object()
        
        return self
    
    def _create_result_object(self):
        """Create result object."""
        hazard_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / self.standard_errors_
        
        try:
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        except ImportError:
            p_values = np.full_like(z_scores, np.nan)
        
        return PWPTTResult(
            coefficients=self.coefficients_,
            standard_errors=self.standard_errors_,
            hazard_ratios=hazard_ratios,
            z_scores=z_scores,
            p_values=p_values,
            robust_variance=self.robust_variance_,
            naive_standard_errors=self.naive_standard_errors_,
            log_likelihood=self.log_likelihood_,
            n_subjects=self.n_subjects_,
            n_observations=self.n_observations_,
            n_events=self.n_events_,
            n_strata=self.n_strata_,
            events_per_stratum=self.events_per_stratum_
        )