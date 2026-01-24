"""
Andersen-Gill Model for Recurrent Events

The Andersen-Gill (AG) model extends the Cox proportional hazards model
to handle recurrent events within subjects. It treats all events as
exchangeable and uses robust variance to account for within-subject
correlation.

Key features:
- All events treated similarly (no stratification by event number)
- Uses counting process (start, stop] format
- Robust (sandwich) variance estimator for clustered data
- Coefficients interpreted as rate ratios

References:
-----------
Andersen, P. K., & Gill, R. D. (1982). Cox's regression model for 
counting processes: a large sample study. The Annals of Statistics, 
10(4), 1100-1120.
"""

import numpy as np
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
import warnings

from .cox_ph import CoxPHModel
from .recurrent_event import prepare_recurrent_data_simple


@dataclass
class AndersenGillResult:
    """
    Results from Andersen-Gill model.
    
    Attributes:
    -----------
    coefficients : np.ndarray
        Estimated coefficients (log rate ratios)
    standard_errors : np.ndarray
        Robust standard errors
    rate_ratios : np.ndarray
        Rate ratios (exp(coefficients))
    z_scores : np.ndarray
        Z-scores for tests
    p_values : np.ndarray
        P-values for tests
    robust_variance : np.ndarray
        Robust variance-covariance matrix
    naive_standard_errors : np.ndarray
        Naive (model-based) standard errors
    log_likelihood : float
        Log partial likelihood
    n_subjects : int
        Number of subjects
    n_observations : int
        Total number of observations (rows)
    n_events : int
        Total number of events
    """
    coefficients: np.ndarray
    standard_errors: np.ndarray
    rate_ratios: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    robust_variance: np.ndarray
    naive_standard_errors: np.ndarray
    log_likelihood: float
    n_subjects: int
    n_observations: int
    n_events: int
    
    def summary(self) -> str:
        """Generate summary table."""
        s = "\n" + "="*80 + "\n"
        s += "Andersen-Gill Model for Recurrent Events - Results Summary\n"
        s += "="*80 + "\n"
        s += f"Number of subjects: {self.n_subjects}\n"
        s += f"Number of observations: {self.n_observations}\n"
        s += f"Number of events: {self.n_events}\n"
        s += f"Log-likelihood: {self.log_likelihood:.4f}\n"
        s += "\n" + "-"*80 + "\n"
        s += f"{'Variable':<12} {'Coef':<10} {'SE(rob)':<10} {'RR':<10} {'z':<8} {'p-value':<10}\n"
        s += "-"*80 + "\n"
        
        for i in range(len(self.coefficients)):
            s += f"{'X' + str(i):<12} "
            s += f"{self.coefficients[i]:>9.5f} "
            s += f"{self.standard_errors[i]:>9.5f} "
            s += f"{self.rate_ratios[i]:>9.5f} "
            s += f"{self.z_scores[i]:>7.3f} "
            s += f"{self.p_values[i]:>9.6f}\n"
        
        s += "="*80 + "\n"
        s += "Note: SE(rob) = Robust standard errors accounting for within-subject correlation\n"
        s += "      RR = Rate ratio (exp(Coef))\n"
        return s


class AndersenGillModel:
    """
    Andersen-Gill model for recurrent event data.
    
    The AG model extends Cox regression to recurrent events by:
    1. Using counting process (start, stop] format
    2. Treating all events as exchangeable
    3. Using robust variance to account for correlation within subjects
    
    The hazard for subject i at time t is:
        λᵢ(t) = λ₀(t) × exp(β'Xᵢ)
    
    where:
        λ₀(t) is the baseline rate function
        β are regression coefficients (log rate ratios)
        Xᵢ are subject-level covariates
    
    Usage:
    ------
    # Simple format (one array per subject)
    model = AndersenGillModel()
    model.fit_simple(
        subject_ids=subject_ids,
        event_times=event_times,
        event_status=event_status,
        covariates=covariates
    )
    
    # Or with pre-formatted counting process data
    model.fit(X, time_start, time_stop, events, subject_id)
    
    References:
    -----------
    Andersen, P. K., & Gill, R. D. (1982). Cox's regression model for 
    counting processes: a large sample study. The Annals of Statistics.
    """
    
    def __init__(self,
                 tie_method: str = 'efron',
                 alpha: float = 0.05,
                 device: str = None):
        """
        Initialize Andersen-Gill model.

        Parameters:
        -----------
        tie_method : str, default='efron'
            Method for handling tied event times ('efron' or 'breslow')
        alpha : float, default=0.05
            Significance level for confidence intervals
        device : str, optional
            Device for computation ('cpu', 'cuda', 'mps')
        """
        self.tie_method = tie_method
        self.alpha = alpha
        self.device = device
        self._is_fitted = False

        # Will store fitted Cox model
        self.cox_model_ = None
        
    def fit_simple(self,
                   subject_ids: np.ndarray,
                   event_times: List[np.ndarray],
                   event_status: List[np.ndarray],
                   covariates: np.ndarray) -> 'AndersenGillModel':
        """
        Fit model using simple recurrent event format.
        
        Parameters:
        -----------
        subject_ids : np.ndarray
            Array of subject identifiers
        event_times : List[np.ndarray]
            List of event time arrays (one per subject)
        event_status : List[np.ndarray]
            List of event status arrays (1=event, 0=censored)
        covariates : np.ndarray, shape (n_subjects, n_features)
            Covariate matrix (one row per subject)
        
        Returns:
        --------
        self : AndersenGillModel
            Fitted model
        
        Example:
        --------
        >>> subject_ids = np.array([1, 2, 3])
        >>> event_times = [np.array([10, 25]), np.array([15]), np.array([8, 20])]
        >>> event_status = [np.array([1, 0]), np.array([0]), np.array([1, 0])]
        >>> covariates = np.array([[1.0, 0.5], [0.5, 1.0], [0.8, 0.8]])
        >>> 
        >>> model = AndersenGillModel()
        >>> model.fit_simple(subject_ids, event_times, event_status, covariates)
        """
        # Convert to counting process format
        from .recurrent_event import prepare_recurrent_data_simple
        
        data = prepare_recurrent_data_simple(
            subject_ids=subject_ids,
            event_times=event_times,
            event_status=event_status,
            covariates=covariates
        )
        
        # Fit using counting process format
        return self.fit(
            X=data.covariates,
            time_start=data.time_start,
            time_stop=data.time_stop,
            events=data.status,
            subject_id=data.subject_id
        )
    
    def fit(self,
            X: np.ndarray,
            time_start: np.ndarray,
            time_stop: np.ndarray,
            events: np.ndarray,
            subject_id: np.ndarray) -> 'AndersenGillModel':
        """
        Fit model using counting process format.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_observations, n_features)
            Covariate matrix
        time_start : np.ndarray, shape (n_observations,)
            Start times for each interval
        time_stop : np.ndarray, shape (n_observations,)
            Stop times for each interval
        events : np.ndarray, shape (n_observations,)
            Event indicators (1=event, 0=censored)
        subject_id : np.ndarray, shape (n_observations,)
            Subject identifiers for clustering
        
        Returns:
        --------
        self : AndersenGillModel
            Fitted model
        """
        # Store data info
        self.n_subjects_ = len(np.unique(subject_id))
        self.n_observations_ = len(X)
        self.n_events_ = np.sum(events)
        
        # Fit Cox model with counting process format
        self.cox_model_ = CoxPHModel(tie_method=self.tie_method, alpha=self.alpha, device=self.device)
        self.cox_model_.fit(
            X=X,
            durations=time_stop,
            events=events,
            start_times=time_start
        )
        
        # Compute robust variance
        self.robust_variance_ = self.cox_model_.compute_robust_variance(
            X=X,
            durations=time_stop,
            events=events,
            cluster_id=subject_id
        )
        
        # Extract results
        self.coefficients_ = self.cox_model_.coefficients_
        self.naive_standard_errors_ = self.cox_model_.standard_errors_
        self.standard_errors_ = np.sqrt(np.diag(self.robust_variance_))
        self.log_likelihood_ = self.cox_model_.log_likelihood_
        
        self._is_fitted = True
        
        # Create result object
        self.result_ = self._create_result_object()
        
        return self
    
    def predict_rate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict rate ratios relative to baseline.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix
        
        Returns:
        --------
        rate_ratios : np.ndarray, shape (n_samples,)
            Predicted rate ratios
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.cox_model_.predict_risk(X)
    
    def predict_cumulative_rate(self, 
                                X: np.ndarray,
                                times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict cumulative rate function.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix
        times : np.ndarray, optional
            Time points for prediction
        
        Returns:
        --------
        cumulative_rates : np.ndarray
            Predicted cumulative rate functions
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.cox_model_.predict_cumulative_hazard(X, times)
    
    def get_confidence_intervals(self, alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals using robust standard errors.
        
        Parameters:
        -----------
        alpha : float, optional
            Significance level (default: self.alpha)
        
        Returns:
        --------
        ci_lower : np.ndarray
            Lower bounds
        ci_upper : np.ndarray
            Upper bounds
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing confidence intervals")
        
        if alpha is None:
            alpha = self.alpha
        
        try:
            from scipy.stats import norm
            z = norm.ppf(1 - alpha/2)
        except ImportError:
            if alpha == 0.05:
                z = 1.96
            else:
                raise ImportError("scipy required for arbitrary alpha values")
        
        ci_lower = self.coefficients_ - z * self.standard_errors_
        ci_upper = self.coefficients_ + z * self.standard_errors_
        
        return ci_lower, ci_upper
    
    def _create_result_object(self) -> AndersenGillResult:
        """Create result object with all fitted values."""
        rate_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / self.standard_errors_
        
        # Compute p-values
        try:
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        except ImportError:
            p_values = np.full_like(z_scores, np.nan)
        
        return AndersenGillResult(
            coefficients=self.coefficients_,
            standard_errors=self.standard_errors_,
            rate_ratios=rate_ratios,
            z_scores=z_scores,
            p_values=p_values,
            robust_variance=self.robust_variance_,
            naive_standard_errors=self.naive_standard_errors_,
            log_likelihood=self.log_likelihood_,
            n_subjects=self.n_subjects_,
            n_observations=self.n_observations_,
            n_events=self.n_events_
        )