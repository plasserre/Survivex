import torch
import numpy as np
from typing import Optional, Tuple, Union
import warnings

class KaplanMeierEstimator:
    """
    Kaplan-Meier survival probability estimator with optimized numpy implementation.

    Based on the product-limit estimator:
    S(t) = ∏(i: t_i ≤ t) (n_i - d_i) / n_i

    Reference implementations:
    - lifelines.KaplanMeierFitter
    - Original Kaplan & Meier (1958) paper
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Kaplan-Meier estimator.

        Parameters:
        -----------
        device : str, optional
            Device for storing results ('cpu', 'cuda', 'mps').
            Defaults to 'cpu' since KM uses optimized numpy internally.
        """
        # KM uses numpy internally for computation, CPU is fastest
        if device is None:
            device = 'cpu'

        self.device = torch.device(device)
        
        # Results storage
        self.timeline_ = None
        self.survival_function_ = None
        self.confidence_interval_ = None
        self.at_risk_ = None
        self.events_ = None
        self.censored_ = None
        
        # Internal parameters
        self._is_fitted = False
        self.alpha_ = 0.05  # Default confidence level (95%)
    
    def fit(self,
            durations: Union[np.ndarray, torch.Tensor],
            events: Union[np.ndarray, torch.Tensor],
            alpha: float = 0.05,
            weights: Optional[Union[np.ndarray, torch.Tensor]] = None) -> 'KaplanMeierEstimator':
        """
        Fit the Kaplan-Meier estimator to survival data.

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
        self : KaplanMeierEstimator
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

        # Get unique event times and aggregate using numpy (fast!)
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
            self.survival_function_ = torch.tensor([1.0], device=self.device)
            self.at_risk_ = torch.tensor([at_risk[-1]], device=self.device, dtype=torch.float32)
            self.events_ = torch.tensor([0.0], device=self.device)
            self.confidence_interval_ = torch.tensor([[1.0, 1.0]], device=self.device)
            self._is_fitted = True
            return self

        timeline = unique_times[event_mask]
        n_events = events_at_t[event_mask]
        n_at_risk = at_risk[event_mask]

        # Kaplan-Meier: S(t) = cumprod((n - d) / n)
        survival_factors = (n_at_risk - n_events) / n_at_risk
        survival_probs = np.cumprod(survival_factors)

        # Store as tensors
        self.timeline_ = torch.tensor(timeline, device=self.device, dtype=torch.float32)
        self.survival_function_ = torch.tensor(survival_probs, device=self.device, dtype=torch.float32)
        self.at_risk_ = torch.tensor(n_at_risk, device=self.device, dtype=torch.float32)
        self.events_ = torch.tensor(n_events, device=self.device, dtype=torch.float32)

        # Calculate confidence intervals
        self._calculate_confidence_intervals()

        self._is_fitted = True
        return self
    
    def _calculate_confidence_intervals(self):
        """
        Calculate confidence intervals using Greenwood's formula for variance estimation.

        Var[log(-log(S(t)))] ≈ Σ(i: t_i ≤ t) d_i / (n_i * (n_i - d_i))
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
        survival = self.survival_function_.cpu().numpy()

        # Greenwood's variance formula
        with np.errstate(divide='ignore', invalid='ignore'):
            variance_terms = np.where(
                (n_i > d_i) & (d_i > 0),
                d_i / (n_i * (n_i - d_i)),
                0.0
            )

        cumulative_variance = np.cumsum(variance_terms)

        # Log-log transformation for CI
        epsilon = 1e-15
        safe_survival = np.clip(survival, epsilon, 1.0 - epsilon)

        with np.errstate(divide='ignore', invalid='ignore'):
            greenwood_se = np.sqrt(cumulative_variance) / np.abs(np.log(safe_survival))

        z_alpha = norm.ppf(1 - self.alpha_ / 2)
        log_neg_log_s = np.log(-np.log(safe_survival))

        upper_log = log_neg_log_s + z_alpha * greenwood_se
        lower_log = log_neg_log_s - z_alpha * greenwood_se

        upper_ci = np.exp(-np.exp(upper_log))
        lower_ci = np.exp(-np.exp(lower_log))

        # Handle NaN and clip
        upper_ci = np.clip(np.nan_to_num(upper_ci, nan=1.0), 0.0, 1.0)
        lower_ci = np.clip(np.nan_to_num(lower_ci, nan=0.0), 0.0, 1.0)

        self.confidence_interval_ = torch.tensor(
            np.stack([lower_ci, upper_ci], axis=1),
            device=self.device, dtype=torch.float32
        )
    
    def survival_function_at_times(self, times: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the survival function at specific time points.
        
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
        
        times = self._to_tensor(times)
        survival_at_times = torch.ones_like(times, device=self.device, dtype=torch.float32)
        
        for i, t in enumerate(times):
            # Find the largest time in timeline that is <= t
            mask = self.timeline_ <= t
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]  # Last index where timeline <= t
                survival_at_times[i] = self.survival_function_[idx]
            # If t < min(timeline), survival probability = 1.0 (default)
        
        return survival_at_times
    
    def median_survival_time(self) -> Optional[float]:
        """
        Calculate the median survival time (time when S(t) = 0.5).
        
        Returns:
        --------
        float or None
            Median survival time, or None if not reached
        """
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before calculating median.")
        
        # Find first time where survival probability <= 0.5
        mask = self.survival_function_ <= 0.5
        if torch.any(mask):
            idx = torch.where(mask)[0][0]  # First index where S(t) <= 0.5
            return self.timeline_[idx].item()
        else:
            return None  # Median not reached
    
    def plot(self, confidence_intervals: bool = True, title: str = "Kaplan-Meier Survival Curve"):
        """
        Plot the Kaplan-Meier survival curve.
        
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
        survival = self.survival_function_.cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        
        # Create step plot (survival function is constant between events)
        # Add point at t=0 with survival=1.0
        plot_times = np.concatenate([[0], times])
        plot_survival = np.concatenate([[1.0], survival])
        
        plt.step(plot_times, plot_survival, where='post', linewidth=2, label='Survival Function')
        
        # Confidence intervals
        if confidence_intervals and self.confidence_interval_ is not None:
            ci = self.confidence_interval_.cpu().numpy()
            plot_ci_lower = np.concatenate([[1.0], ci[:, 0]])
            plot_ci_upper = np.concatenate([[1.0], ci[:, 1]])
            
            plt.fill_between(plot_times, plot_ci_lower, plot_ci_upper, alpha=0.3, step='post',
                           label=f'{int((1-self.alpha_)*100)}% Confidence Interval')
        
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add median survival line if it exists
        median_time = self.median_survival_time()
        if median_time is not None:
            plt.axvline(x=median_time, color='red', linestyle='--', alpha=0.7,
                       label=f'Median Survival: {median_time:.2f}')
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            plt.legend()
        
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


class KaplanMeierEstimatorWith100Points:
    """
    Kaplan-Meier survival probability estimator implemented from scratch with GPU support.
    
    Based on the product-limit estimator:
    S(t) = âˆ(i: t_i â‰¤ t) (n_i - d_i) / n_i
    
    Reference implementations:
    - lifelines.KaplanMeierFitter
    - Original Kaplan & Meier (1958) paper
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Kaplan-Meier estimator.
        
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
        self.survival_function_ = None
        self.confidence_interval_ = None
        self.at_risk_ = None
        self.events_ = None
        self.censored_ = None
        
        # Internal parameters
        self._is_fitted = False
        self.alpha_ = 0.05  # Default confidence level (95%)
    
    def fit(self, 
            durations: Union[np.ndarray, torch.Tensor], 
            events: Union[np.ndarray, torch.Tensor],
            alpha: float = 0.05,
            weights: Optional[Union[np.ndarray, torch.Tensor]] = None) -> 'KaplanMeierEstimator':
        """
        Fit the Kaplan-Meier estimator to survival data.
        
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
        self : KaplanMeierEstimator
            Fitted estimator
        """
        # Determine dtype from input (prefer float64 for precision)
        if isinstance(durations, torch.Tensor):
            dtype = durations.dtype
        else:
            dtype = torch.float64
        
        # Convert inputs to tensors with consistent dtype
        durations = self._to_tensor(durations, dtype=dtype)
        events = self._to_tensor(events, dtype=dtype)
        
        if weights is not None:
            weights = self._to_tensor(weights, dtype=dtype)
        else:
            weights = torch.ones_like(durations, device=self.device, dtype=dtype)
        
        # Validation
        self._validate_input(durations, events, weights)
        
        # Store alpha for confidence intervals
        self.alpha_ = alpha
        
        # Sort by durations
        sorted_indices = torch.argsort(durations)
        sorted_durations = durations[sorted_indices]
        sorted_events = events[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Get ALL unique times (both events and censorings) - matches R's survfit
        unique_times = torch.unique(sorted_durations)
        
        # Calculate survival function at each unique time
        timeline = []
        survival_probs = []
        at_risk_counts = []
        event_counts = []
        
        current_survival = 1.0
        
        for t in unique_times:
            # Count subjects at risk just before time t
            at_risk_mask = sorted_durations >= t
            n_at_risk = torch.sum(sorted_weights[at_risk_mask])
            
            # Count events exactly at time t
            event_at_t_mask = (sorted_durations == t) & (sorted_events == 1)
            n_events = torch.sum(sorted_weights[event_at_t_mask])
            
            if n_at_risk > 0:
                # Update survival only if there are events at this time
                if n_events > 0:
                    survival_at_t = (n_at_risk - n_events) / n_at_risk
                    current_survival *= survival_at_t.item()
                
                timeline.append(t.item())
                survival_probs.append(current_survival)
                at_risk_counts.append(n_at_risk.item())
                event_counts.append(n_events.item())
        
        # Convert to tensors
        self.timeline_ = torch.tensor(timeline, device=self.device, dtype=durations.dtype)
        self.survival_function_ = torch.tensor(survival_probs, device=self.device, dtype=durations.dtype)
        self.at_risk_ = torch.tensor(at_risk_counts, device=self.device, dtype=durations.dtype)
        self.events_ = torch.tensor(event_counts, device=self.device, dtype=durations.dtype)
        
        # Calculate confidence intervals using Greenwood's formula
        self._calculate_confidence_intervals()
        
        self._is_fitted = True
        return self
    
    def _calculate_confidence_intervals(self):
        """
        Calculate confidence intervals using Greenwood's formula for variance estimation.
        
        Var[log(-log(S(t)))] â‰ˆ Î£(i: t_i â‰¤ t) d_i / (n_i * (n_i - d_i))
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
        
        # Greenwood's variance formula
        variance_terms = torch.zeros_like(self.timeline_, device=self.device)
        
        for i in range(len(self.timeline_)):
            n_i = self.at_risk_[i]
            d_i = self.events_[i]
            
            if n_i > d_i and d_i > 0:
                variance_terms[i] = d_i / (n_i * (n_i - d_i))
        
        # Cumulative variance for Greenwood's formula
        cumulative_variance = torch.cumsum(variance_terms, dim=0)
        
        # Calculate confidence intervals on log(-log(S(t))) scale
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            # Avoid numerical issues
            epsilon = 1e-15
            safe_survival = torch.clamp(self.survival_function_, min=epsilon, max=1.0 - epsilon)
            
            # Greenwood's standard error
            greenwood_se = torch.sqrt(cumulative_variance) / torch.abs(torch.log(safe_survival))
            
            # Z-score for confidence level
            z_alpha = norm.ppf(1 - self.alpha_ / 2)
            
            # Log-log transformation
            log_neg_log_s = torch.log(-torch.log(safe_survival))
            
            # Confidence intervals on transformed scale
            upper_log_neg_log = log_neg_log_s + z_alpha * greenwood_se
            lower_log_neg_log = log_neg_log_s - z_alpha * greenwood_se
            
            # Transform back to survival probability scale
            upper_ci = torch.exp(-torch.exp(upper_log_neg_log))
            lower_ci = torch.exp(-torch.exp(lower_log_neg_log))
            
            # Clamp to valid range [0, 1] and handle NaNs
            upper_ci = torch.clamp(torch.nan_to_num(upper_ci, nan=1.0), 0.0, 1.0)
            lower_ci = torch.clamp(torch.nan_to_num(lower_ci, nan=0.0), 0.0, 1.0)
            
            self.confidence_interval_ = torch.stack([lower_ci, upper_ci], dim=1)
    
    def survival_function_at_times(self, times: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the survival function at specific time points.
        
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
        
        times = self._to_tensor(times)
        survival_at_times = torch.ones_like(times, device=self.device, dtype=torch.float32)
        
        for i, t in enumerate(times):
            # Find the largest time in timeline that is <= t
            mask = self.timeline_ <= t
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]  # Last index where timeline <= t
                survival_at_times[i] = self.survival_function_[idx]
            # If t < min(timeline), survival probability = 1.0 (default)
        
        return survival_at_times
    
    def median_survival_time(self) -> Optional[float]:
        """
        Calculate the median survival time (time when S(t) = 0.5).
        
        Returns:
        --------
        float or None
            Median survival time, or None if not reached
        """
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before calculating median.")
        
        # Find first time where survival probability <= 0.5
        mask = self.survival_function_ <= 0.5
        if torch.any(mask):
            idx = torch.where(mask)[0][0]  # First index where S(t) <= 0.5
            return self.timeline_[idx].item()
        else:
            return None  # Median not reached
    
    def plot(self, confidence_intervals: bool = True, title: str = "Kaplan-Meier Survival Curve"):
        """
        Plot the Kaplan-Meier survival curve.
        
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
        survival = self.survival_function_.cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        
        # Create step plot (survival function is constant between events)
        # Add point at t=0 with survival=1.0
        plot_times = np.concatenate([[0], times])
        plot_survival = np.concatenate([[1.0], survival])
        
        plt.step(plot_times, plot_survival, where='post', linewidth=2, label='Survival Function')
        
        # Confidence intervals
        if confidence_intervals and self.confidence_interval_ is not None:
            ci = self.confidence_interval_.cpu().numpy()
            plot_ci_lower = np.concatenate([[1.0], ci[:, 0]])
            plot_ci_upper = np.concatenate([[1.0], ci[:, 1]])
            
            plt.fill_between(plot_times, plot_ci_lower, plot_ci_upper, alpha=0.3, step='post',
                           label=f'{int((1-self.alpha_)*100)}% Confidence Interval')
        
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add median survival line if it exists
        median_time = self.median_survival_time()
        if median_time is not None:
            plt.axvline(x=median_time, color='red', linestyle='--', alpha=0.7,
                       label=f'Median Survival: {median_time:.2f}')
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor], dtype=None) -> torch.Tensor:
        """Convert input data to tensor and move to device."""
        if isinstance(data, torch.Tensor):
            tensor = data.to(self.device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        else:
            if dtype is None:
                dtype = torch.float64  # Default to double precision
            return torch.tensor(data, device=self.device, dtype=dtype)
    
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