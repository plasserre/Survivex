

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
from dataclasses import dataclass


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


class PWPTTModel:
    """PWP Total Time model for recurrent events."""

    def __init__(self, tie_method: str = 'efron', alpha: float = 0.05,
                 device: str = None):
        self.tie_method = tie_method
        self.alpha = alpha
        self.device = device
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
        from .cox_ph import StratifiedCoxPHModel

        self.n_subjects_ = len(np.unique(subject_id))
        self.n_observations_ = len(X)
        self.n_events_ = int(np.sum(events))
        self.n_strata_ = len(np.unique(stratum))

        self.events_per_stratum_ = {}
        for s in np.unique(stratum):
            mask = stratum == s
            self.events_per_stratum_[int(s)] = int(np.sum(events[mask]))

        # Fit stratified Cox with counting process support
        self.cox_model_ = StratifiedCoxPHModel(tie_method=self.tie_method, device=self.device)
        self.cox_model_.fit(X, time_stop, events, stratum, start_times=time_start)

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