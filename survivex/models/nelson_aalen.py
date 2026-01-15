import torch
import numpy as np
from typing import Optional, Union
import warnings

class NelsonAalenEstimator:
    """
    Nelson-Aalen cumulative hazard estimator with optimized numpy implementation.

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
            Device for storing results ('cpu', 'cuda', 'mps').
            Defaults to 'cpu' since NA uses optimized numpy internally.
        """
        # NA uses numpy internally for computation, CPU is fastest
        if device is None:
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
        # Convert to numpy for fast CPU computation
        if isinstance(durations, torch.Tensor):
            durations = durations.cpu().numpy()
        else:
            durations = np.asarray(durations, dtype=np.float64)

        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        else:
            events = np.asarray(events, dtype=np.float64)

        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.cpu().numpy()
            else:
                weights = np.asarray(weights, dtype=np.float64)
        else:
            weights = np.ones_like(durations)

        # Validation
        if len(durations) != len(events):
            raise ValueError("durations and events must have the same length")
        if np.any(durations < 0):
            raise ValueError("durations must be non-negative")

        self.alpha_ = alpha

        # Sort by time
        order = np.argsort(durations)
        t_sorted = durations[order]
        e_sorted = events[order]
        w_sorted = weights[order]

        # Get unique times and aggregate using numpy (fast!)
        unique_times, inverse = np.unique(t_sorted, return_inverse=True)
        n_times = len(unique_times)

        # Aggregate events and totals at each time
        events_at_t = np.bincount(inverse, weights=w_sorted * e_sorted, minlength=n_times)
        total_at_t = np.bincount(inverse, weights=w_sorted, minlength=n_times)

        # Number at risk: cumsum from the end
        at_risk = np.cumsum(total_at_t[::-1])[::-1]

        # Filter to event times only
        event_mask = events_at_t > 0

        if not np.any(event_mask):
            # No events - everyone censored
            self.timeline_ = torch.tensor([unique_times[-1]], device=self.device, dtype=torch.float32)
            self.cumulative_hazard_ = torch.tensor([0.0], device=self.device)
            self.at_risk_ = torch.tensor([at_risk[-1]], device=self.device, dtype=torch.float32)
            self.events_ = torch.tensor([0.0], device=self.device)
            self.confidence_interval_ = torch.tensor([[0.0, 0.0]], device=self.device)
            self._is_fitted = True
            return self

        timeline = unique_times[event_mask]
        n_events = events_at_t[event_mask]
        n_at_risk = at_risk[event_mask]

        # Nelson-Aalen: H(t) = cumsum(d / n)
        hazard_increments = n_events / n_at_risk
        cumulative_hazard = np.cumsum(hazard_increments)

        # Store as tensors
        self.timeline_ = torch.tensor(timeline, device=self.device, dtype=torch.float32)
        self.cumulative_hazard_ = torch.tensor(cumulative_hazard, device=self.device, dtype=torch.float32)
        self.at_risk_ = torch.tensor(n_at_risk, device=self.device, dtype=torch.float32)
        self.events_ = torch.tensor(n_events, device=self.device, dtype=torch.float32)

        # Calculate confidence intervals
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

        # Use numpy for computation
        n_i = self.at_risk_.cpu().numpy()
        d_i = self.events_.cpu().numpy()
        hazard = self.cumulative_hazard_.cpu().numpy()

        # Aalen's variance formula
        with np.errstate(divide='ignore', invalid='ignore'):
            variance_terms = np.where(
                (n_i > 0) & (d_i > 0),
                d_i / (n_i ** 2),
                0.0
            )

        cumulative_variance = np.cumsum(variance_terms)
        standard_error = np.sqrt(cumulative_variance)

        z_alpha = norm.ppf(1 - self.alpha_ / 2)

        # Log transformation for CI
        epsilon = 1e-15
        safe_hazard = np.clip(hazard, epsilon, None)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_se = standard_error / safe_hazard

        log_hazard = np.log(safe_hazard)
        log_upper = log_hazard + z_alpha * log_se
        log_lower = log_hazard - z_alpha * log_se

        upper_ci = np.exp(log_upper)
        lower_ci = np.exp(log_lower)

        # Handle NaN and clip
        upper_ci = np.nan_to_num(upper_ci, nan=safe_hazard[0] * 10 if len(safe_hazard) > 0 else 1.0)
        lower_ci = np.clip(np.nan_to_num(lower_ci, nan=0.0), 0.0, None)

        self.confidence_interval_ = torch.tensor(
            np.stack([lower_ci, upper_ci], axis=1),
            device=self.device, dtype=torch.float32
        )
    
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