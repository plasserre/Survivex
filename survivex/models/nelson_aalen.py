import torch
import numpy as np
from typing import Optional, Union
import warnings

class NelsonAalenEstimator:
    """
    Nelson-Aalen cumulative hazard estimator implemented from scratch with GPU support.
    
    The Nelson-Aalen estimator estimates the cumulative hazard function:
    H(t) = Σ(i: t_i ≤ t) d_i / n_i
    
    where:
    - d_i = number of events at time t_i
    - n_i = number at risk just before t_i
    
    The cumulative hazard is related to survival by: S(t) = exp(-H(t))
    
    Advantages over Kaplan-Meier:
    - More stable with heavy censoring
    - Better for estimating hazard rates
    - Used in Cox model baseline hazard
    
    References:
    -----------
    - Nelson, W. (1972). Theory and applications of hazard plotting for censored failure data
    - Aalen, O. (1978). Nonparametric estimation of partial transition probabilities in multiple decrement models
    - lifelines.NelsonAalenFitter
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Nelson-Aalen estimator.
        
        Parameters:
        -----------
        device : str, optional
            Device to run computations on ('cpu', 'cuda', 'mps').
            If None, automatically selects best available device.
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        
        # Results storage
        self.timeline_ = None
        self.cumulative_hazard_ = None
        self.confidence_interval_ = None
        self.at_risk_ = None
        self.events_ = None
        
        # Internal parameters
        self._is_fitted = False
        self.alpha_ = 0.05  # Default confidence level (95%)
    
    def fit(self,
            durations: Union[np.ndarray, torch.Tensor],
            events: Union[np.ndarray, torch.Tensor],
            alpha: float = 0.05,
            weights: Optional[Union[np.ndarray, torch.Tensor]] = None) -> 'NelsonAalenEstimator':
        """
        Fit the Nelson-Aalen estimator to survival data.

        Parameters:
        -----------
        durations : array-like
            Time to event or censoring for each subject
        events : array-like
            Event indicator (1 if event occurred, 0 if censored)
        alpha : float, default=0.05
            Significance level for confidence intervals (1-alpha confidence level)
        weights : array-like, optional
            Sample weights for each observation

        Returns:
        --------
        self : NelsonAalenEstimator
            Fitted estimator
        """
        # Convert inputs to tensors and move to device
        durations = self._to_tensor(durations)
        events = self._to_tensor(events)

        if weights is not None:
            weights = self._to_tensor(weights)
        else:
            weights = torch.ones_like(durations, device=self.device)

        # Validation
        self._validate_input(durations, events, weights)

        # Store alpha for confidence intervals
        self.alpha_ = alpha

        # Get unique times and aggregate counts - VECTORIZED for GPU
        unique_times, inverse_indices = torch.unique(durations, sorted=True, return_inverse=True)
        n_times = len(unique_times)

        # Aggregate events and total observations at each unique time using scatter_add
        events_at_time = torch.zeros(n_times, device=self.device, dtype=weights.dtype)
        total_at_time = torch.zeros(n_times, device=self.device, dtype=weights.dtype)

        events_at_time.scatter_add_(0, inverse_indices, weights * events)
        total_at_time.scatter_add_(0, inverse_indices, weights)

        # Calculate number at risk at each time point (cumulative sum from the end)
        at_risk = torch.flip(torch.cumsum(torch.flip(total_at_time, [0]), dim=0), [0])

        # Filter to only times where events occurred
        event_times_mask = events_at_time > 0

        if not torch.any(event_times_mask):
            # No events occurred - everyone was censored
            self.timeline_ = unique_times[-1:].clone()
            self.cumulative_hazard_ = torch.tensor([0.0], device=self.device)
            self.at_risk_ = at_risk[-1:].clone()
            self.events_ = torch.tensor([0.0], device=self.device)
            self.confidence_interval_ = torch.tensor([[0.0, 0.0]], device=self.device)
            self._is_fitted = True
            return self

        # Extract only event times
        self.timeline_ = unique_times[event_times_mask]
        self.events_ = events_at_time[event_times_mask]
        self.at_risk_ = at_risk[event_times_mask]

        # Calculate cumulative hazard using vectorized cumsum
        # H(t) = sum(d_i / n_i) for all event times <= t
        hazard_increments = self.events_ / self.at_risk_
        self.cumulative_hazard_ = torch.cumsum(hazard_increments, dim=0)

        # Calculate confidence intervals using Aalen's variance formula
        self._calculate_confidence_intervals()

        self._is_fitted = True
        return self
    
    def _calculate_confidence_intervals(self):
        """
        Calculate confidence intervals using Aalen's variance formula.

        Var[H(t)] = Σ(i: t_i ≤ t) d_i / n_i²

        Confidence intervals are calculated using log transformation:
        log(H(t)) ± z_α/2 × SE[log(H(t))]
        """
        if len(self.timeline_) == 0:
            self.confidence_interval_ = None
            return

        try:
            from scipy.stats import norm
        except ImportError:
            warnings.warn("scipy not available, skipping confidence intervals")
            self.confidence_interval_ = None
            return

        # Aalen's variance formula - VECTORIZED: Var[H(t)] = Σ d_i / n_i²
        n_i = self.at_risk_
        d_i = self.events_

        # Compute variance terms where valid
        valid_mask = (n_i > 0) & (d_i > 0)
        variance_terms = torch.where(
            valid_mask,
            d_i / (n_i ** 2),
            torch.zeros_like(d_i)
        )

        # Cumulative variance
        cumulative_variance = torch.cumsum(variance_terms, dim=0)

        # Standard error
        standard_error = torch.sqrt(cumulative_variance)

        # Z-score for confidence level
        z_alpha = norm.ppf(1 - self.alpha_ / 2)

        # Calculate confidence intervals on log scale for better properties
        # Avoid log(0)
        epsilon = 1e-15
        safe_hazard = torch.clamp(self.cumulative_hazard_, min=epsilon)

        # Standard error on log scale: SE[log(H)] = SE[H] / H
        log_se = standard_error / safe_hazard

        # Log of cumulative hazard
        log_hazard = torch.log(safe_hazard)

        # Confidence intervals on log scale
        log_upper = log_hazard + z_alpha * log_se
        log_lower = log_hazard - z_alpha * log_se

        # Transform back to hazard scale
        upper_ci = torch.exp(log_upper)
        lower_ci = torch.exp(log_lower)

        # Clamp to reasonable range and handle NaNs
        upper_ci = torch.nan_to_num(upper_ci, nan=safe_hazard[0].item() * 10)
        lower_ci = torch.clamp(torch.nan_to_num(lower_ci, nan=0.0), 0.0)

        self.confidence_interval_ = torch.stack([lower_ci, upper_ci], dim=1)
    
    def cumulative_hazard_at_times(self, times: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the cumulative hazard function at specific time points.
        
        Parameters:
        -----------
        times : array-like
            Time points to evaluate the cumulative hazard at
            
        Returns:
        --------
        torch.Tensor
            Cumulative hazard at the specified times
        """
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before predicting.")
        
        times = self._to_tensor(times)
        hazard_at_times = torch.zeros_like(times, device=self.device, dtype=torch.float32)
        
        for i, t in enumerate(times):
            # Find the largest time in timeline that is <= t
            mask = self.timeline_ <= t
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]  # Last index where timeline <= t
                hazard_at_times[i] = self.cumulative_hazard_[idx]
            # If t < min(timeline), cumulative hazard = 0.0 (default)
        
        return hazard_at_times
    
    def survival_function_at_times(self, times: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Calculate survival function from cumulative hazard: S(t) = exp(-H(t))
        
        Parameters:
        -----------
        times : array-like
            Time points to evaluate the survival function at
            
        Returns:
        --------
        torch.Tensor
            Survival probabilities at the specified times
        """
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before predicting.")
        
        hazard = self.cumulative_hazard_at_times(times)
        return torch.exp(-hazard)
    
    def plot(self, confidence_intervals: bool = True, title: str = "Nelson-Aalen Cumulative Hazard"):
        """
        Plot the Nelson-Aalen cumulative hazard curve.
        
        Parameters:
        -----------
        confidence_intervals : bool, default=True
            Whether to show confidence intervals
        title : str
            Plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before plotting.")
        
        # Convert tensors to numpy for plotting
        times = self.timeline_.cpu().numpy()
        hazard = self.cumulative_hazard_.cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        
        # Create step plot (cumulative hazard increases at events)
        # Add point at t=0 with hazard=0.0
        plot_times = np.concatenate([[0], times])
        plot_hazard = np.concatenate([[0.0], hazard])
        
        plt.step(plot_times, plot_hazard, where='post', linewidth=2, 
                label='Cumulative Hazard', color='darkred')
        
        # Confidence intervals
        if confidence_intervals and self.confidence_interval_ is not None:
            ci = self.confidence_interval_.cpu().numpy()
            plot_ci_lower = np.concatenate([[0.0], ci[:, 0]])
            plot_ci_upper = np.concatenate([[0.0], ci[:, 1]])
            
            plt.fill_between(plot_times, plot_ci_lower, plot_ci_upper, 
                           alpha=0.3, step='post', color='darkred',
                           label=f'{int((1-self.alpha_)*100)}% Confidence Interval')
        
        plt.xlabel('Time')
        plt.ylabel('Cumulative Hazard')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_survival(self, confidence_intervals: bool = True, 
                     title: str = "Survival Curve (from Nelson-Aalen)"):
        """
        Plot the survival function derived from Nelson-Aalen: S(t) = exp(-H(t))
        
        Parameters:
        -----------
        confidence_intervals : bool, default=True
            Whether to show confidence intervals
        title : str
            Plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before plotting.")
        
        # Convert to survival probabilities
        times = self.timeline_.cpu().numpy()
        survival = torch.exp(-self.cumulative_hazard_).cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        
        # Create step plot
        plot_times = np.concatenate([[0], times])
        plot_survival = np.concatenate([[1.0], survival])
        
        plt.step(plot_times, plot_survival, where='post', linewidth=2, 
                label='Survival Function (from NA)', color='darkgreen')
        
        # Confidence intervals
        if confidence_intervals and self.confidence_interval_ is not None:
            ci = self.confidence_interval_.cpu().numpy()
            # Transform hazard CI to survival CI
            survival_ci_upper = np.exp(-ci[:, 0])  # Lower hazard → Higher survival
            survival_ci_lower = np.exp(-ci[:, 1])  # Upper hazard → Lower survival
            
            plot_ci_upper = np.concatenate([[1.0], survival_ci_upper])
            plot_ci_lower = np.concatenate([[1.0], survival_ci_lower])
            
            plt.fill_between(plot_times, plot_ci_lower, plot_ci_upper, 
                           alpha=0.3, step='post', color='darkgreen',
                           label=f'{int((1-self.alpha_)*100)}% Confidence Interval')
        
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input data to tensor and move to device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.tensor(data, device=self.device, dtype=torch.float32)
    
    def _validate_input(self, durations: torch.Tensor, events: torch.Tensor, weights: torch.Tensor):
        """Validate input data."""
        if len(durations) != len(events):
            raise ValueError("durations and events must have the same length")
        
        if len(durations) != len(weights):
            raise ValueError("durations and weights must have the same length")
        
        if torch.any(durations < 0):
            raise ValueError("durations must be non-negative")
        
        if not torch.all((events == 0) | (events == 1)):
            raise ValueError("events must be binary (0 or 1)")
        
        if torch.any(weights < 0):
            raise ValueError("weights must be non-negative")