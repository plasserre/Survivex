"""
PWP Gap Time Model - WORKING VERSION

This version will work once you add compute_robust_variance to StratifiedCoxPHModel.

Save as: survivex/models/pwp_gt.py
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import warnings


@dataclass 
class PWPGTResult:
    """Results from PWP Gap Time model."""
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
        s += "PWP Gap Time Model for Recurrent Events - Results Summary\n"
        s += "="*80 + "\n"
        s += f"Number of subjects: {self.n_subjects}\n"
        s += f"Number of observations: {self.n_observations}\n"
        s += f"Number of events: {self.n_events}\n"
        s += f"Number of strata (event numbers): {self.n_strata}\n"
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
        s += "Note: SE(rob) = Robust standard errors\n"
        s += "      Gap time: time since previous event\n"
        return s


class PWPGTModel:
    """
    PWP Gap Time model for recurrent events.
    
    Usage:
    ------
    from survivex.models.pwp_gt import PWPGTModel
    
    model = PWPGTModel()
    model.fit_simple(subject_ids, event_times, event_status, covariates)
    print(model.result_.summary())
    """
    
    def __init__(self, 
                 tie_method: str = 'efron',
                 alpha: float = 0.05):
        self.tie_method = tie_method
        self.alpha = alpha
        self._is_fitted = False
        self.cox_model_ = None
        
    def fit_simple(self,
                   subject_ids: np.ndarray,
                   event_times: List[np.ndarray],
                   event_status: List[np.ndarray],
                   covariates: np.ndarray) -> 'PWPGTModel':
        """Fit model using simple recurrent event format."""
        from .recurrent_event import prepare_recurrent_data_gap_time
        
        data = prepare_recurrent_data_gap_time(
            subject_ids=subject_ids,
            event_times=event_times,
            event_status=event_status,
            covariates=covariates
        )
        
        return self.fit(
            X=data.covariates,
            gap_durations=data.time_stop,
            events=data.status,
            subject_id=data.subject_id,
            stratum=data.transition_number
        )
    
    def fit(self,
            X: np.ndarray,
            gap_durations: np.ndarray,
            events: np.ndarray,
            subject_id: np.ndarray,
            stratum: np.ndarray) -> 'PWPGTModel':
        """
        Fit model using gap time format.
        
        Parameters:
        -----------
        X : np.ndarray
            Covariate matrix
        gap_durations : np.ndarray
            Gap times (duration of each gap period)
        events : np.ndarray
            Event indicators
        subject_id : np.ndarray
            Subject identifiers for clustering
        stratum : np.ndarray
            Stratum identifiers (event number)
        """
        # Store original subject IDs before any sorting
        self.original_subject_id_ = subject_id.copy()
        
        # Store data info
        self.n_subjects_ = len(np.unique(subject_id))
        self.n_observations_ = len(X)
        self.n_events_ = np.sum(events)
        self.n_strata_ = len(np.unique(stratum))
        
        # Count events per stratum
        self.events_per_stratum_ = {}
        for s in np.unique(stratum):
            mask = stratum == s
            self.events_per_stratum_[int(s)] = int(np.sum(events[mask]))
        
        # Import stratified Cox model
        from .cox_ph import StratifiedCoxPHModel
        
        # Fit stratified Cox model with gap time
        self.cox_model_ = StratifiedCoxPHModel(tie_method=self.tie_method)
        
        # Fit the model (gap times naturally start at 0)
        self.cox_model_.fit(
            X=X,
            durations=gap_durations,
            events=events,
            strata=stratum
        )
        
        # Compute robust variance using the NEW method
        try:
            self.robust_variance_ = self.cox_model_.compute_robust_variance(
                cluster_id=self.original_subject_id_
            )
            
            # Extract results
            self.coefficients_ = self.cox_model_.coefficients_
            self.naive_standard_errors_ = self.cox_model_.standard_errors_
            self.standard_errors_ = np.sqrt(np.diag(self.robust_variance_))
            self.log_likelihood_ = self.cox_model_.log_likelihood_
            
        except AttributeError as e:
            error_msg = (
                "\n" + "="*80 + "\n"
                "ERROR: StratifiedCoxPHModel missing compute_robust_variance method!\n"
                "="*80 + "\n"
                "You need to add the compute_robust_variance method to StratifiedCoxPHModel.\n\n"
                "Follow these steps:\n"
                "1. Open cox_ph.py\n"
                "2. Find the StratifiedCoxPHModel class\n"
                "3. Add the method from ADD_TO_STRATIFIED_COX.py\n\n"
                "The method should be added after predict methods, before _create_result_object.\n"
                "="*80 + "\n"
            )
            raise AttributeError(error_msg) from e
        
        self._is_fitted = True
        self.result_ = self._create_result_object()
        
        return self
    
    def predict_hazard_ratio(self, X: np.ndarray) -> np.ndarray:
        """Predict hazard ratios."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.cox_model_.predict_risk(X)
    
    def get_confidence_intervals(self, alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals using robust standard errors."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted")
        
        if alpha is None:
            alpha = self.alpha
        
        try:
            from scipy.stats import norm
            z = norm.ppf(1 - alpha/2)
        except ImportError:
            z = 1.96 if alpha == 0.05 else None
            if z is None:
                raise ImportError("scipy required for arbitrary alpha values")
        
        ci_lower = self.coefficients_ - z * self.standard_errors_
        ci_upper = self.coefficients_ + z * self.standard_errors_
        
        return ci_lower, ci_upper
    
    def _create_result_object(self) -> PWPGTResult:
        """Create result object."""
        hazard_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / self.standard_errors_
        
        try:
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        except ImportError:
            p_values = np.full_like(z_scores, np.nan)
        
        return PWPGTResult(
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