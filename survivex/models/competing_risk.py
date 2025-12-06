"""
Competing Risks Survival Analysis Models

Implemented from scratch following:
- lifelines AalenJohansenFitter (exact variance formula)
- R survival package (survfit with competing risks)
- R mstate package
- Mathematical foundations from Aalen & Johansen (1978)

References:
- Aalen, O. O., & Johansen, S. (1978). An empirical transition matrix for 
  non-homogeneous Markov chains based on censored observations.
- Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993). Statistical 
  models based on counting processes. Springer.
- Klein, J. P., & Moeschberger, M. L. (2003). Survival analysis: techniques 
  for censored and truncated data.
- lifelines source code: https://github.com/CamDavidsonPilon/lifelines
"""

import numpy as np
import torch
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CIFResult:
    """
    Results from Cumulative Incidence Function estimation.
    
    Attributes:
    -----------
    event_of_interest : int
        The event type being estimated
    timeline : np.ndarray
        Event times at which CIF is estimated
    cumulative_incidence : np.ndarray
        CIF values at each time point
    variance : Optional[np.ndarray]
        Variance estimates (if calculated)
    confidence_intervals : Optional[Tuple[np.ndarray, np.ndarray]]
        Lower and upper confidence bounds
    """
    event_of_interest: int
    timeline: np.ndarray
    cumulative_incidence: np.ndarray
    variance: Optional[np.ndarray] = None
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None


class AalenJohansenFitter:
    """
    Aalen-Johansen Estimator for Cumulative Incidence Function in Competing Risks.
    
    The Aalen-Johansen estimator is the non-parametric estimator of the cumulative
    incidence function (CIF) in a competing risks framework. It properly accounts
    for competing events by recognizing that subjects who experience a competing
    event are no longer at risk for the event of interest.
    
    Mathematical Foundation:
    -----------------------
    For event type j, the cumulative incidence function is:
    
    CIF_j(t) = P(T <= t, epsilon = j)
    
    The estimator is computed as:
    
    CIF_j(t) = sum[t_i <= t] S(t_i-) * (d_ij / n_i)
    
    Where:
    - t_i are the ordered event times
    - S(t_i-) is the overall survival probability just before t_i
    - d_ij is the number of events of type j at time t_i
    - n_i is the number at risk at time t_i
    
    The overall survival S(t) is estimated using all competing events:
    
    S(t) = product[t_i <= t] (1 - d_i / n_i)
    
    Where d_i is the total number of events (all types) at time t_i.
    
    Variance Estimation:
    -------------------
    The variance is estimated using the exact formula from lifelines:
    
    For CIF at time t:
    Var[F_t] = sum[s <= t] {
        (F_t - F_s)^2 * d_s / [n_s * (n_s - d_s)] +
        S_s^2 * d_js * (n_s - d_js) / n_s^3 +
        -2 * (F_t - F_s) * S_s * d_js / n_s^2
    }
    
    Where:
    - F_t = CIF at time t
    - F_s = CIF at time s (previous times)
    - S_s = Overall survival at s (lagged)
    - d_s = Total events at s
    - d_js = Events of type j at s
    - n_s = At risk at s
    
    This accounts for:
    1. Variance in overall survival (first term)
    2. Direct variance from event of interest (second term)
    3. Covariance between survival and CIF (third term)
    
    Parameters:
    -----------
    alpha : float, default=0.05
        Significance level for confidence intervals (95% CI by default)
    
    calculate_variance : bool, default=True
        Whether to calculate variance and confidence intervals
    
    jitter_level : float, default=1e-8
        Small amount to add to tied event times (Aalen-Johansen requires no ties)
    
    seed : int, optional
        Random seed for reproducibility when jittering
    
    device : str, default='cpu'
        Computing device ('cpu' or 'cuda')
    
    Attributes:
    -----------
    event_of_interest_ : int
        The event type being estimated
    
    timeline_ : torch.Tensor
        Event times at which CIF is estimated (includes time 0)
    
    cumulative_incidence_ : torch.Tensor
        CIF values at each time point
    
    variance_ : torch.Tensor, optional
        Variance estimates at each time point
    
    confidence_interval_lower_ : torch.Tensor, optional
        Lower confidence bounds
    
    confidence_interval_upper_ : torch.Tensor, optional
        Upper confidence bounds
    
    overall_survival_ : torch.Tensor
        Overall survival function S(t) accounting for all competing events
    
    event_counts_ : Dict[int, torch.Tensor]
        Number of each event type at each time point
    
    at_risk_ : torch.Tensor
        Number at risk at each time point
    
    Examples:
    ---------
    >>> # Basic usage with competing risks data
    >>> durations = np.array([1, 2, 3, 4, 5, 6])
    >>> events = np.array([1, 2, 1, 0, 2, 1])  # 0=censored, 1=event of interest, 2=competing
    >>> 
    >>> ajf = AalenJohansenFitter()
    >>> ajf.fit(durations, events, event_of_interest=1)
    >>> print(ajf.cumulative_incidence_)
    
    Notes:
    ------
    - Event codes must be non-negative integers where 0 indicates censoring
    - The estimator handles right-censored data
    - Tied event times are automatically jittered with a warning
    - For a simple two-state model (alive/dead), this reduces to 1 - Kaplan-Meier
    - Variance formula matches lifelines implementation exactly
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        calculate_variance: bool = True,
        jitter_level: float = 1e-8,
        seed: Optional[int] = None,
        device: str = 'cpu'
    ):
        self.alpha = alpha
        self.calculate_variance = calculate_variance
        self.jitter_level = jitter_level
        self.seed = seed
        self.device = device
        
        # Will be set during fit
        self.event_of_interest_ = None
        self.timeline_ = None
        self.cumulative_incidence_ = None
        self.variance_ = None
        self.confidence_interval_lower_ = None
        self.confidence_interval_upper_ = None
        self.overall_survival_ = None
        self.event_counts_ = None
        self.at_risk_ = None
        self._n_samples = None
        self._unique_events = None
        
    def fit(
        self,
        durations: Union[np.ndarray, torch.Tensor, List],
        events: Union[np.ndarray, torch.Tensor, List],
        event_of_interest: int,
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> 'AalenJohansenFitter':
        """
        Fit the Aalen-Johansen estimator to competing risks data.
        
        Parameters:
        -----------
        durations : array-like, shape (n_samples,)
            Observed durations (event times or censoring times)
        
        events : array-like, shape (n_samples,)
            Event indicators:
            - 0: censored
            - positive integers: event types (1, 2, 3, ...)
        
        event_of_interest : int
            The event type for which to estimate the CIF (must be > 0)
        
        weights : array-like, optional
            Case weights (not yet implemented)
        
        Returns:
        --------
        self : AalenJohansenFitter
            Fitted estimator
        """
        # Convert inputs to tensors
        durations = self._to_tensor(durations)
        events = self._to_tensor(events)
        
        # Validation
        assert durations.shape[0] == events.shape[0], "Durations and events must have same length"
        assert event_of_interest > 0, "event_of_interest must be a positive integer"
        assert torch.all(events >= 0), "Event codes must be non-negative"
        assert torch.all(durations > 0), "Durations must be positive"
        
        self._n_samples = len(durations)
        self.event_of_interest_ = event_of_interest
        self._unique_events = torch.unique(events[events > 0]).cpu().numpy()
        
        # Check for tied event times and jitter if necessary
        event_times = durations[events > 0]
        if len(event_times) > len(torch.unique(event_times)):
            print(f"Warning: Tied event times detected. Adding small random jitter (±{self.jitter_level})")
            if self.seed is not None:
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)
            
            # Add uniform random jitter to durations
            jitter = torch.rand_like(durations) * self.jitter_level * 2 - self.jitter_level
            durations = durations + jitter
        
        # Sort by duration
        sorted_indices = torch.argsort(durations)
        durations_sorted = durations[sorted_indices]
        events_sorted = events[sorted_indices]
        
        # Get all unique times (including censoring times for consistency with lifelines/R)
        unique_times = torch.unique(durations_sorted)
        
        # Add time 0 to timeline for consistency with standard implementations
        timeline_with_zero = torch.cat([torch.tensor([0.0], device=self.device), unique_times])
        n_times = len(timeline_with_zero)
        
        # Initialize outputs
        self.timeline_ = timeline_with_zero
        cumulative_incidence = torch.zeros(n_times, device=self.device)
        overall_survival = torch.ones(n_times, device=self.device)
        at_risk = torch.zeros(n_times, device=self.device)
        
        # Store event counts for each type at each time
        event_counts = {}
        for event_type in self._unique_events:
            event_counts[int(event_type)] = torch.zeros(n_times, device=self.device)
        
        # Store lagged survival (S at previous time) for variance calculation
        lagged_survival = torch.ones(n_times, device=self.device)
        
        # Set values at time 0
        at_risk[0] = float(self._n_samples)
        
        # Iterate through each unique event time (skip time 0)
        for i in range(1, n_times):
            t = timeline_with_zero[i]
            
            # Number at risk (everyone with duration >= t)
            n_at_risk = torch.sum(durations_sorted >= t).float()
            at_risk[i] = n_at_risk
            
            # Count events of each type at this time
            at_time_mask = (durations_sorted == t) & (events_sorted > 0)
            events_at_t = events_sorted[at_time_mask]
            
            total_events_at_t = len(events_at_t)
            events_of_interest_at_t = torch.sum(events_at_t == event_of_interest).float()
            
            # Store event counts
            for event_type in self._unique_events:
                count = torch.sum(events_at_t == event_type).float()
                event_counts[int(event_type)][i] = count
            
            # Store survival from previous time (for this iteration, it's the lagged value)
            survival_prev = overall_survival[i-1]
            lagged_survival[i] = survival_prev
            
            # Update overall survival using Kaplan-Meier formula with ALL events
            # S(t) = S(t-) * (1 - d_t / n_t)
            if n_at_risk > 0 and total_events_at_t > 0:
                overall_survival[i] = survival_prev * (1.0 - total_events_at_t / n_at_risk)
            else:
                overall_survival[i] = survival_prev
            
            # Update cumulative incidence using Aalen-Johansen formula
            # CIF(t) = CIF(t-) + S(t-) * (d_j / n_t)
            cif_prev = cumulative_incidence[i-1]
            
            if n_at_risk > 0 and events_of_interest_at_t > 0:
                # Use survival just before this time point
                increment = survival_prev * (events_of_interest_at_t / n_at_risk)
                cumulative_incidence[i] = cif_prev + increment
            else:
                cumulative_incidence[i] = cif_prev
        
        # Store results
        self.cumulative_incidence_ = cumulative_incidence
        self.overall_survival_ = overall_survival
        self.at_risk_ = at_risk
        self.event_counts_ = event_counts
        
        # Calculate variance using exact lifelines formula
        if self.calculate_variance:
            variance = torch.zeros(n_times, device=self.device)
            
            # For each time point t, calculate variance by summing over all s <= t
            for i in range(1, n_times):
                F_t = cumulative_incidence[i]
                var_sum = 0.0
                
                # Sum over all previous time points s <= t (including current)
                for s in range(1, i + 1):
                    F_s = cumulative_incidence[s]
                    S_s = lagged_survival[s]  # Survival just before time s
                    # Sum all events at time s (all types)
                    d_s = torch.sum(torch.stack([event_counts[int(et)][s] for et in self._unique_events]))
                    d_js = event_counts[int(event_of_interest)][s]
                    n_s = at_risk[s]
                    
                    if n_s > 0:
                        # Term 1: Variance from overall survival
                        # (F_t - F_s)^2 * d_s / [n_s * (n_s - d_s)]
                        if d_s > 0 and n_s > d_s:
                            term1 = ((F_t - F_s) ** 2) * d_s / (n_s * (n_s - d_s))
                        else:
                            term1 = 0.0
                        
                        # Term 2: Direct variance from events of interest
                        # S_s^2 * d_js * (n_s - d_js) / n_s^3
                        if d_js > 0:
                            term2 = (S_s ** 2) * d_js * (n_s - d_js) / (n_s ** 3)
                        else:
                            term2 = 0.0
                        
                        # Term 3: Covariance term
                        # -2 * (F_t - F_s) * S_s * d_js / n_s^2
                        if d_js > 0:
                            term3 = -2.0 * (F_t - F_s) * S_s * d_js / (n_s ** 2)
                        else:
                            term3 = 0.0
                        
                        var_sum += term1 + term2 + term3
                
                # Ensure non-negative variance
                variance[i] = max(var_sum, 0.0)
            
            self.variance_ = variance
            
            # Calculate confidence intervals using log-log transformation
            z_score = torch.tensor(1.96, device=self.device)  # 95% CI
            if self.alpha != 0.05:
                from scipy.stats import norm
                z_score = torch.tensor(norm.ppf(1 - self.alpha / 2), device=self.device)
            
            # Standard error
            se = torch.sqrt(variance)
            
            # Avoid log(0) and division by zero issues
            cif_bounded = torch.clamp(cumulative_incidence, min=1e-10, max=1.0 - 1e-10)
            
            # Log-log transformation for CIF > 0
            mask_positive = cumulative_incidence > 1e-10
            
            self.confidence_interval_lower_ = torch.zeros_like(cumulative_incidence)
            self.confidence_interval_upper_ = torch.zeros_like(cumulative_incidence)
            
            if torch.any(mask_positive):
                # For CIF close to 1, use linear CI
                # For other values, use log-log transformation
                mask_near_one = cumulative_incidence > 0.999
                
                # Log-log transformation for most values
                log_log_cif = torch.log(-torch.log(1.0 - cif_bounded))
                
                # Avoid division by zero in standard error calculation
                denominator = cif_bounded * torch.abs(torch.log(1.0 - cif_bounded))
                denominator = torch.clamp(denominator, min=1e-10)
                se_log_log = se / denominator
                
                log_log_lower = log_log_cif - z_score * se_log_log
                log_log_upper = log_log_cif + z_score * se_log_log
                
                # Transform back
                ci_lower = 1.0 - torch.exp(-torch.exp(log_log_lower))
                ci_upper = 1.0 - torch.exp(-torch.exp(log_log_upper))
                
                # For values near 1, use linear approximation
                if torch.any(mask_near_one):
                    linear_lower = cumulative_incidence - z_score * se
                    linear_upper = cumulative_incidence + z_score * se
                    ci_lower[mask_near_one] = linear_lower[mask_near_one]
                    ci_upper[mask_near_one] = linear_upper[mask_near_one]
                
                # Apply where CIF > 0
                self.confidence_interval_lower_[mask_positive] = ci_lower[mask_positive]
                self.confidence_interval_upper_[mask_positive] = ci_upper[mask_positive]
            
            # Ensure bounds are in [0, 1]
            self.confidence_interval_lower_ = torch.clamp(self.confidence_interval_lower_, 0.0, 1.0)
            self.confidence_interval_upper_ = torch.clamp(self.confidence_interval_upper_, 0.0, 1.0)
        
        return self
    
    def predict(self, times: Union[np.ndarray, torch.Tensor, List]) -> torch.Tensor:
        """
        Predict cumulative incidence at specific time points.
        
        Uses step function (left-continuous with right limits).
        
        Parameters:
        -----------
        times : array-like
            Time points at which to evaluate CIF
        
        Returns:
        --------
        predictions : torch.Tensor
            CIF values at requested times
        """
        assert self.cumulative_incidence_ is not None, "Model must be fitted first"
        
        times = self._to_tensor(times)
        predictions = torch.zeros_like(times)
        
        for i, t in enumerate(times):
            # Find the last time point <= t
            mask = self.timeline_ <= t
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]
                predictions[i] = self.cumulative_incidence_[idx]
            else:
                predictions[i] = 0.0  # Before first event
        
        return predictions
    
    def plot(
        self,
        title: Optional[str] = None,
        xlabel: str = "Time",
        ylabel: str = "Cumulative Incidence",
        show_confidence_intervals: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot the cumulative incidence function.
        
        Parameters:
        -----------
        title : str, optional
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        show_confidence_intervals : bool
            Whether to show confidence intervals
        figsize : tuple
            Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        times = self.timeline_.cpu().numpy()
        cif = self.cumulative_incidence_.cpu().numpy()
        
        # Plot CIF as step function
        ax.step(times, cif, where='post', linewidth=2, label=f'Event {self.event_of_interest_}')
        
        # Plot confidence intervals
        if show_confidence_intervals and self.confidence_interval_lower_ is not None:
            ci_lower = self.confidence_interval_lower_.cpu().numpy()
            ci_upper = self.confidence_interval_upper_.cpu().numpy()
            
            ax.fill_between(times, ci_lower, ci_upper, 
                          alpha=0.2, step='post', label=f'{int((1-self.alpha)*100)}% CI')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f'Cumulative Incidence Function (Event {self.event_of_interest_})')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> Dict:
        """
        Get a summary of the fitted model.
        
        Returns:
        --------
        summary : dict
            Dictionary containing model information
        """
        assert self.cumulative_incidence_ is not None, "Model must be fitted first"
        
        summary = {
            'event_of_interest': self.event_of_interest_,
            'n_samples': self._n_samples,
            'n_event_times': len(self.timeline_),
            'unique_events': self._unique_events.tolist(),
            'max_cif': float(self.cumulative_incidence_[-1]),
            'time_range': (float(self.timeline_[0]), float(self.timeline_[-1]))
        }
        
        # Add event counts summary
        for event_type in self._unique_events:
            total_events = int(torch.sum(self.event_counts_[int(event_type)]))
            summary[f'n_events_type_{event_type}'] = total_events
        
        return summary
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor, List]) -> torch.Tensor:
        """Convert input to tensor on the correct device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)


def cumulative_incidence(
    durations: Union[np.ndarray, List],
    events: Union[np.ndarray, List],
    event_of_interest: int,
    alpha: float = 0.05
) -> CIFResult:
    """
    Convenience function to compute cumulative incidence function.
    
    Parameters:
    -----------
    durations : array-like
        Event/censoring times
    events : array-like
        Event indicators (0=censored, positive integers=event types)
    event_of_interest : int
        Event type to estimate CIF for
    alpha : float
        Significance level for confidence intervals
    
    Returns:
    --------
    result : CIFResult
        Named tuple with timeline, cumulative_incidence, variance, and CIs
    """
    ajf = AalenJohansenFitter(alpha=alpha)
    ajf.fit(durations, events, event_of_interest)
    
    result = CIFResult(
        event_of_interest=event_of_interest,
        timeline=ajf.timeline_.cpu().numpy(),
        cumulative_incidence=ajf.cumulative_incidence_.cpu().numpy(),
        variance=ajf.variance_.cpu().numpy() if ajf.variance_ is not None else None,
        confidence_intervals=(
            ajf.confidence_interval_lower_.cpu().numpy(),
            ajf.confidence_interval_upper_.cpu().numpy()
        ) if ajf.confidence_interval_lower_ is not None else None
    )
    
    return result
    



"""
Fine-Gray Subdistribution Hazards Model - CORRECTED

Fixed: Proper weighted partial likelihood matching R cmprsk::crr() exactly
"""

import numpy as np
import torch
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class FineGrayResult:
    """Results from Fine-Gray subdistribution hazard model."""
    event_of_interest: int
    coefficients: np.ndarray
    hazard_ratios: np.ndarray
    standard_errors: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    confidence_intervals: Tuple[np.ndarray, np.ndarray]
    baseline_cumulative_hazard: np.ndarray
    unique_event_times: np.ndarray
    convergence_info: Dict
    variance_matrix: np.ndarray


class FineGrayModel:
    """Fine-Gray Proportional Subdistribution Hazards Model."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        max_iter: int = 100,
        tol: float = 1e-6,
        device: str = 'cpu'
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        
        # Results
        self.coef_ = None
        self.hazard_ratios_ = None
        self.se_ = None
        self.z_scores_ = None
        self.p_values_ = None
        self.variance_matrix_ = None
        self.baseline_cumulative_hazard_ = None
        self.unique_event_times_ = None
        self.censoring_survival_ = None
        self.censoring_times_ = None
        self.convergence_ = None
        self.confidence_intervals_lower_ = None
        self.confidence_intervals_upper_ = None
        self._n_samples = None
        self._n_covariates = None
        self._event_of_interest = None
        
    def fit(
        self,
        durations: Union[np.ndarray, torch.Tensor, List],
        events: Union[np.ndarray, torch.Tensor, List],
        covariates: Union[np.ndarray, torch.Tensor],
        event_of_interest: int
    ) -> 'FineGrayModel':
        """Fit the Fine-Gray model."""
        
        durations = self._to_tensor(durations).double()
        events = self._to_tensor(events).long()
        covariates = self._to_tensor(covariates).double()
        
        if covariates.ndim == 1:
            covariates = covariates.unsqueeze(1)
        
        self._n_samples, self._n_covariates = covariates.shape
        self._event_of_interest = event_of_interest
        
        # Estimate censoring distribution
        self._estimate_censoring_distribution(durations, events)
        
        # Build risk sets
        risk_set_info = self._build_risk_sets(durations, events, covariates, event_of_interest)
        
        # Optimize
        self._newton_raphson(risk_set_info, covariates, durations, events)
        
        # Baseline hazard
        self._estimate_baseline_hazard(risk_set_info, covariates)
        
        # Variance
        self._compute_variance(risk_set_info, covariates, durations, events)
        
        return self
    
    def _estimate_censoring_distribution(
        self,
        durations: torch.Tensor,
        events: torch.Tensor
    ):
        """Estimate G(t) = P(C > t) using KM."""
        from survivex.models.kaplan_meier import KaplanMeierEstimatorWith100Points
        
        # 1 if censored, 0 if event
        censoring_indicator = (events == 0).float()
        
        km = KaplanMeierEstimatorWith100Points(device='cpu')
        km.fit(durations.cpu(), censoring_indicator.cpu())
        
        self.censoring_times_ = km.timeline_.to(self.device).double()
        self.censoring_survival_ = km.survival_function_.to(self.device).double()
    
    def _get_censoring_survival(self, t: torch.Tensor) -> torch.Tensor:
        """Get G(t) with LOCF."""
        if t.ndim == 0:
            t = t.unsqueeze(0)
        
        result = torch.ones_like(t, dtype=torch.double)
        
        for i, time_val in enumerate(t):
            mask = self.censoring_times_ <= time_val
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]
                result[i] = self.censoring_survival_[idx]
        
        return result
    
    def _build_risk_sets(
        self,
        durations: torch.Tensor,
        events: torch.Tensor,
        covariates: torch.Tensor,
        event_of_interest: int
    ) -> Dict:
        """Build risk sets with IPCW weights."""
        
        event_mask = (events == event_of_interest)
        event_times = durations[event_mask]
        
        if len(event_times) == 0:
            raise ValueError(f"No events of type {event_of_interest}")
        
        unique_event_times = event_times.unique().sort()[0]
        
        risk_set_info = {
            'unique_times': unique_event_times,
            'n_times': len(unique_event_times),
            'risk_sets': [],
            'event_indices': [],
            'risk_set_weights': [],
            'event_weights': []  # NEW: store event weights
        }
        
        for t in unique_event_times:
            # Find events at time t
            event_idx = torch.where((durations == t) & (events == event_of_interest))[0]
            
            # Build extended risk set
            # 1. Still at risk (T >= t)
            # 2. Had competing event before t
            still_at_risk = durations >= t
            competing_before = (durations < t) & (events != 0) & (events != event_of_interest)
            risk_set_mask = still_at_risk | competing_before
            risk_set_idx = torch.where(risk_set_mask)[0]
            
            # IPCW weights for risk set
            g_t = self._get_censoring_survival(t)[0]
            rs_weights = torch.zeros(len(risk_set_idx), device=self.device, dtype=torch.double)
            
            for j, idx in enumerate(risk_set_idx):
                if durations[idx] >= t:
                    # Still at risk: weight = 1 / G(t)
                    rs_weights[j] = 1.0 / torch.clamp(g_t, min=1e-10)
                else:
                    # Competing event: weight = 1 / G(T_i)
                    g_ti = self._get_censoring_survival(durations[idx])[0]
                    rs_weights[j] = 1.0 / torch.clamp(g_ti, min=1e-10)
            
            # NEW: Compute event weights (for numerator of Breslow estimator)
            event_weights = torch.zeros(len(event_idx), device=self.device, dtype=torch.double)
            for j, idx in enumerate(event_idx):
                g_ti = self._get_censoring_survival(durations[idx])[0]
                event_weights[j] = 1.0 / torch.clamp(g_ti, min=1e-10)
            
            # Store everything
            risk_set_info['event_indices'].append(event_idx)
            risk_set_info['risk_sets'].append(risk_set_idx)
            risk_set_info['risk_set_weights'].append(rs_weights)
            risk_set_info['event_weights'].append(event_weights)  # NEW
        
        return risk_set_info
    
    def _newton_raphson(
        self,
        risk_set_info: Dict,
        covariates: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor
    ):
        """Newton-Raphson optimization - CORRECTED."""
        
        p = covariates.shape[1]
        beta = torch.zeros(p, device=self.device, dtype=torch.double)
        
        for iteration in range(self.max_iter):
            gradient = torch.zeros(p, device=self.device, dtype=torch.double)
            hessian = torch.zeros((p, p), device=self.device, dtype=torch.double)
            log_lik = 0.0
            
            for idx, t in enumerate(risk_set_info['unique_times']):
                event_idx = risk_set_info['event_indices'][idx]
                risk_set_idx = risk_set_info['risk_sets'][idx]
                rs_weights = risk_set_info['risk_set_weights'][idx]
                
                if len(event_idx) == 0:
                    continue
                
                X_events = covariates[event_idx]
                X_risk = covariates[risk_set_idx]
                
                # Weighted risk
                risk_scores = torch.exp(torch.matmul(X_risk, beta))
                weighted_risk = rs_weights * risk_scores
                sum_weighted_risk = weighted_risk.sum()
                
                # KEY FIX: Numerator is NOT weighted!
                # log L = Σ β'X_i - Σ log(Σ w_j exp(β'X_j))
                log_lik += torch.sum(torch.matmul(X_events, beta))
                log_lik -= len(event_idx) * torch.log(sum_weighted_risk)
                
                # Gradient: Σ X_i - d_t * E_w[X]
                gradient += X_events.sum(dim=0)
                weighted_mean_x = (weighted_risk.unsqueeze(1) * X_risk).sum(dim=0) / sum_weighted_risk
                gradient -= len(event_idx) * weighted_mean_x
                
                # Hessian
                weighted_x_outer = torch.zeros((p, p), device=self.device, dtype=torch.double)
                for j in range(len(risk_set_idx)):
                    x_j = X_risk[j]
                    weighted_x_outer += rs_weights[j] * risk_scores[j] * torch.outer(x_j, x_j)
                weighted_x_outer /= sum_weighted_risk
                
                variance_x = weighted_x_outer - torch.outer(weighted_mean_x, weighted_mean_x)
                hessian -= len(event_idx) * variance_x
            
            # Update
            try:
                delta = torch.linalg.solve(hessian, gradient)
            except:
                delta = 0.01 * gradient
            
            beta_new = beta - delta
            
            if torch.norm(delta) < self.tol:
                self.convergence_ = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_gradient_norm': torch.norm(gradient).item(),
                    'log_likelihood': log_lik.item()
                }
                break
            
            beta = beta_new
        else:
            self.convergence_ = {
                'converged': False,
                'iterations': self.max_iter,
                'final_gradient_norm': torch.norm(gradient).item(),
                'log_likelihood': log_lik.item()
            }
        
        self.coef_ = beta.cpu().numpy()
        self.hazard_ratios_ = np.exp(self.coef_)
    
    def _estimate_baseline_hazard(
        self,
        risk_set_info: Dict,
        covariates: torch.Tensor
    ):
        """Breslow estimator for baseline cumulative hazard."""
        
        beta = torch.from_numpy(self.coef_).to(self.device).double()
        n_times = risk_set_info['n_times']
        
        baseline_hazard = torch.zeros(n_times, device=self.device, dtype=torch.double)
        
        for idx in range(n_times):
            event_idx = risk_set_info['event_indices'][idx]
            risk_set_idx = risk_set_info['risk_sets'][idx]
            rs_weights = risk_set_info['risk_set_weights'][idx]
            event_weights = risk_set_info['event_weights'][idx]  # GET IT
            
            # WEIGHTED number of events
            weighted_events = event_weights.sum()
            
            X_risk = covariates[risk_set_idx]
            risk_scores = torch.exp(torch.matmul(X_risk, beta))
            denominator = torch.sum(rs_weights * risk_scores)
            
            baseline_hazard[idx] = weighted_events / denominator
        
        cumulative_hazard = torch.cumsum(baseline_hazard, dim=0)
        
        self.baseline_cumulative_hazard_ = cumulative_hazard.cpu().numpy()
        self.unique_event_times_ = risk_set_info['unique_times'].cpu().numpy()
        
    
    def _compute_variance(
        self,
        risk_set_info: Dict,
        covariates: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor
    ):
        """
        Compute variance using inverse information matrix.
        
        Note: R's cmprsk uses a more complex robust variance that accounts
        for estimation of the censoring distribution. This implementation
        uses the simpler I^{-1} approach which is still asymptotically valid.
        """
        
        beta = torch.from_numpy(self.coef_).to(self.device).double()
        p = len(beta)
        
        # Compute information matrix
        info_matrix = torch.zeros((p, p), device=self.device, dtype=torch.double)
        
        for idx in range(risk_set_info['n_times']):
            event_idx = risk_set_info['event_indices'][idx]
            risk_set_idx = risk_set_info['risk_sets'][idx]
            rs_weights = risk_set_info['risk_set_weights'][idx]
            
            if len(event_idx) == 0:
                continue
            
            X_risk = covariates[risk_set_idx]
            risk_scores = torch.exp(torch.matmul(X_risk, beta))
            weighted_risk = rs_weights * risk_scores
            sum_weighted_risk = weighted_risk.sum()
            
            # Weighted mean
            weighted_mean_x = (weighted_risk.unsqueeze(1) * X_risk).sum(dim=0) / sum_weighted_risk
            
            # Weighted second moment
            weighted_x_outer = torch.zeros((p, p), device=self.device, dtype=torch.double)
            for j in range(len(risk_set_idx)):
                x_j = X_risk[j]
                weighted_x_outer += rs_weights[j] * risk_scores[j] * torch.outer(x_j, x_j)
            weighted_x_outer /= sum_weighted_risk
            
            # Variance
            variance_x = weighted_x_outer - torch.outer(weighted_mean_x, weighted_mean_x)
            
            # Add contribution
            d_t = len(event_idx)
            info_matrix += d_t * variance_x
        
        # Simple inverse (not robust sandwich)
        try:
            variance_matrix = torch.inverse(info_matrix)
        except:
            variance_matrix = torch.linalg.pinv(info_matrix)
        
        self.variance_matrix_ = variance_matrix.cpu().numpy()
        self.se_ = np.sqrt(np.diag(self.variance_matrix_))
        
        from scipy.stats import norm
        self.z_scores_ = self.coef_ / self.se_
        self.p_values_ = 2 * (1 - norm.cdf(np.abs(self.z_scores_)))
        
        z_crit = norm.ppf(1 - self.alpha / 2)
        self.confidence_intervals_lower_ = self.coef_ - z_crit * self.se_
        self.confidence_intervals_upper_ = self.coef_ + z_crit * self.se_

    
    def predict_cumulative_incidence(
        self,
        covariates: Union[np.ndarray, torch.Tensor],
        times: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """Predict CIF."""
        
        assert self.coef_ is not None, "Model must be fitted first"
        
        covariates = self._to_tensor(covariates).double()
        if covariates.ndim == 1:
            covariates = covariates.unsqueeze(0)
        
        beta = torch.from_numpy(self.coef_).to(self.device).double()
        
        if times is None:
            baseline_chf = torch.from_numpy(self.baseline_cumulative_hazard_).to(self.device)
        else:
            times = self._to_tensor(times).double()
            baseline_times = torch.from_numpy(self.unique_event_times_).to(self.device)
            baseline_chf_full = torch.from_numpy(self.baseline_cumulative_hazard_).to(self.device)
            
            baseline_chf = torch.zeros_like(times)
            for i, t in enumerate(times):
                mask = baseline_times <= t
                if torch.any(mask):
                    idx = torch.where(mask)[0][-1]
                    baseline_chf[i] = baseline_chf_full[idx]
        
        n_samples = covariates.shape[0]
        n_times = len(baseline_chf)
        
        cif = torch.zeros((n_samples, n_times), device=self.device, dtype=torch.double)
        
        for i in range(n_samples):
            linear_pred = torch.dot(covariates[i], beta)
            risk_score = torch.exp(linear_pred)
            cif[i] = 1.0 - torch.exp(-baseline_chf * risk_score)
        
        return cif.cpu().numpy()
    
    def summary(self) -> FineGrayResult:
        """Return summary."""
        assert self.coef_ is not None, "Model must be fitted first"
        
        return FineGrayResult(
            event_of_interest=self._event_of_interest,
            coefficients=self.coef_,
            hazard_ratios=self.hazard_ratios_,
            standard_errors=self.se_,
            z_scores=self.z_scores_,
            p_values=self.p_values_,
            confidence_intervals=(self.confidence_intervals_lower_, self.confidence_intervals_upper_),
            baseline_cumulative_hazard=self.baseline_cumulative_hazard_,
            unique_event_times=self.unique_event_times_,
            convergence_info=self.convergence_,
            variance_matrix=self.variance_matrix_
        )
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor, List]) -> torch.Tensor:
        """Convert to tensor."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        else:
            return torch.tensor(x, device=self.device)