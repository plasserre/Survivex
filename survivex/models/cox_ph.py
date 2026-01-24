import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict
import warnings
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# Numba-optimized Efron computation (compiled at first call)
@njit(cache=True, fastmath=True)
def _efron_loop_numba_serial(
    beta_np, X_np, durations_np, events_np,
    risk_scores, risk_cumsum_rev, weighted_X_cumsum_rev, weighted_XX_cumsum_rev,
    unique_times, time_to_first_idx, event_indices_flat, event_indices_starts, event_indices_counts
):
    """
    Serial Numba-optimized Efron computation (stable version).
    """
    n_samples, n_features = X_np.shape
    n_unique = len(unique_times)

    log_likelihood = 0.0
    gradient = np.zeros(n_features)
    hessian = np.zeros((n_features, n_features))

    for t_idx in range(n_unique):
        first_idx = time_to_first_idx[t_idx]
        start = event_indices_starts[t_idx]
        d_t = event_indices_counts[t_idx]

        if d_t == 0:
            continue

        sum_risk = risk_cumsum_rev[first_idx]
        sum_weighted_X = weighted_X_cumsum_rev[first_idx].copy()
        sum_weighted_XX = weighted_XX_cumsum_rev[first_idx].copy()

        # Get event indices for this time
        event_idx_list = event_indices_flat[start:start + d_t]

        # Get event data
        X_events = np.empty((d_t, n_features))
        risk_events = np.empty(d_t)
        for i in range(d_t):
            idx = event_idx_list[i]
            risk_events[i] = risk_scores[idx]
            for j in range(n_features):
                X_events[i, j] = X_np[idx, j]

        # Compute sums for events
        sum_risk_events = 0.0
        sum_X_events = np.zeros(n_features)
        sum_weighted_X_events = np.zeros(n_features)
        sum_weighted_XX_events = np.zeros((n_features, n_features))

        for i in range(d_t):
            sum_risk_events += risk_events[i]
            for j in range(n_features):
                sum_X_events[j] += X_events[i, j]
                sum_weighted_X_events[j] += X_events[i, j] * risk_events[i]
                for k in range(n_features):
                    sum_weighted_XX_events[j, k] += X_events[i, j] * X_events[i, k] * risk_events[i]

        # Compute eta contribution
        eta_sum = 0.0
        for i in range(d_t):
            for j in range(n_features):
                eta_sum += X_events[i, j] * beta_np[j]

        if d_t == 1:
            # No ties - simple case
            log_likelihood += eta_sum - np.log(sum_risk)

            for j in range(n_features):
                weighted_mean_j = sum_weighted_X[j] / sum_risk
                gradient[j] += sum_X_events[j] - weighted_mean_j

            for j in range(n_features):
                for k in range(n_features):
                    weighted_mean_jk = sum_weighted_XX[j, k] / sum_risk
                    outer_jk = (sum_weighted_X[j] / sum_risk) * (sum_weighted_X[k] / sum_risk)
                    hessian[j, k] -= weighted_mean_jk - outer_jk
        else:
            # Ties - Efron approximation
            log_likelihood += eta_sum
            for j in range(n_features):
                gradient[j] += sum_X_events[j]

            for l in range(d_t):
                frac = l / d_t
                denom = sum_risk - frac * sum_risk_events

                log_likelihood -= np.log(denom)

                adj_weighted_X = np.empty(n_features)
                for j in range(n_features):
                    adj_weighted_X[j] = (sum_weighted_X[j] - frac * sum_weighted_X_events[j]) / denom
                    gradient[j] -= adj_weighted_X[j]

                for j in range(n_features):
                    for k in range(n_features):
                        adj_weighted_XX_jk = (sum_weighted_XX[j, k] - frac * sum_weighted_XX_events[j, k]) / denom
                        outer_jk = adj_weighted_X[j] * adj_weighted_X[k]
                        hessian[j, k] -= adj_weighted_XX_jk - outer_jk

    return log_likelihood, gradient, hessian


@dataclass
class CoxPHResult:
    """
    Results from fitting a Cox Proportional Hazards model.
    
    Attributes:
    -----------
    coefficients : np.ndarray
        Estimated regression coefficients (beta)
    standard_errors : np.ndarray
        Standard errors of coefficients
    hazard_ratios : np.ndarray
        Hazard ratios (exp(beta))
    z_scores : np.ndarray
        Z-scores for Wald tests
    p_values : np.ndarray
        P-values for Wald tests
    concordance_index : float
        C-index (concordance)
    log_likelihood : float
        Log partial likelihood at convergence
    baseline_hazard : torch.Tensor
        Baseline hazard function
    baseline_cumulative_hazard : torch.Tensor
        Baseline cumulative hazard
    baseline_survival : torch.Tensor
        Baseline survival function
    timeline : torch.Tensor
        Time points for baseline functions
    convergence_info : dict
        Convergence details
    """
    coefficients: np.ndarray
    standard_errors: np.ndarray
    hazard_ratios: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    concordance_index: float
    log_likelihood: float
    baseline_hazard: torch.Tensor
    baseline_cumulative_hazard: torch.Tensor
    baseline_survival: torch.Tensor
    timeline: torch.Tensor
    convergence_info: dict
    
    def summary(self) -> str:
        """Generate a summary table of results."""
        s = "\n" + "="*70 + "\n"
        s += "Cox Proportional Hazards Model - Results Summary\n"
        s += "="*70 + "\n"
        s += f"Number of observations: {len(self.coefficients)}\n"
        s += f"Concordance Index: {self.concordance_index:.4f}\n"
        s += f"Log-likelihood: {self.log_likelihood:.4f}\n"
        s += f"Converged: {self.convergence_info['converged']}\n"
        s += f"Iterations: {self.convergence_info['iterations']}\n"
        s += "\n" + "-"*70 + "\n"
        s += f"{'Covariate':<15} {'Coef':<12} {'SE':<12} {'HR':<12} {'z':<10} {'p-value':<10}\n"
        s += "-"*70 + "\n"
        
        for i in range(len(self.coefficients)):
            s += f"{'X' + str(i):<15} "
            s += f"{self.coefficients[i]:>11.6f} "
            s += f"{self.standard_errors[i]:>11.6f} "
            s += f"{self.hazard_ratios[i]:>11.6f} "
            s += f"{self.z_scores[i]:>9.4f} "
            s += f"{self.p_values[i]:>9.6f}\n"
        
        s += "="*70 + "\n"
        return s


class CoxPHModel:
    """
    Cox Proportional Hazards Model implemented from scratch with GPU support.
    
    The Cox model specifies the hazard as:
        h(t|X) = h₀(t) × exp(β'X)
    
    where:
        h₀(t) is the baseline hazard (unspecified)
        β are regression coefficients
        X are covariates
    
    Parameters are estimated using partial likelihood maximization with
    Newton-Raphson optimization.
    
    References:
    -----------
    - Cox, D. R. (1972). Regression models and life-tables
    - Breslow, N. (1974). Covariance analysis of censored survival data
    - Efron, B. (1977). The efficiency of Cox's likelihood function
    - R survival package: survival::coxph
    - lifelines: CoxPHFitter
    """
    
    def __init__(self, 
                 tie_method: str = 'efron',
                 alpha: float = 0.05,
                 max_iter: int = 50,
                 tol: float = 1e-6,
                 device: Optional[str] = None):
        """
        Initialize Cox Proportional Hazards model.
        
        Parameters:
        -----------
        tie_method : str, default='efron'
            Method for handling tied event times:
            - 'efron': Efron approximation (R default, more accurate)
            - 'breslow': Breslow approximation (simpler, faster)
        alpha : float, default=0.05
            Significance level for confidence intervals
        max_iter : int, default=50
            Maximum iterations for Newton-Raphson
        tol : float, default=1e-6
            Convergence tolerance (gradient norm)
        device : str, optional
            Device for computation ('cpu', 'cuda', 'mps')
            Note: Cox model requires float64 precision. MPS does not support
            float64, so CPU will be used automatically if MPS is selected.
        """
        if tie_method not in ['efron', 'breslow']:
            raise ValueError("tie_method must be 'efron' or 'breslow'")
        
        self.tie_method = tie_method
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        # Device selection
        # Note: Cox PH is FASTER on CPU due to:
        # 1. Sequential operations (cumulative sums, Newton-Raphson iterations)
        # 2. GPU data transfer overhead
        # 3. Highly optimized numpy/numba on CPU
        # Default to CPU unless explicitly requested otherwise
        if device is None:
            device = 'cpu'  # CPU is faster for Cox PH
        elif device == 'mps':
            warnings.warn(
                "MPS device does not support float64 required for Cox model. "
                "Using CPU instead for numerical stability."
            )
            device = 'cpu'
        elif device == 'cuda':
            warnings.warn(
                "Cox PH is typically faster on CPU due to sequential operations. "
                "Consider using device='cpu' for better performance."
            )

        self.device = torch.device(device)
        
        # Model parameters (set after fitting)
        self.coefficients_ = None
        self.standard_errors_ = None
        self.baseline_hazard_ = None
        self.baseline_cumulative_hazard_ = None
        self.baseline_survival_ = None
        self.timeline_ = None
        self.concordance_index_ = None
        self._is_fitted = False
        
    def fit(self, 
        X: Union[np.ndarray, torch.Tensor],
        durations: Union[np.ndarray, torch.Tensor],
        events: Union[np.ndarray, torch.Tensor],
        start_times: Optional[Union[np.ndarray, torch.Tensor]] = None) -> 'CoxPHModel':
        """
        Fit the Cox Proportional Hazards model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariate matrix
        durations : array-like, shape (n_samples,)
            Time to event or censoring (stop time for counting process)
        events : array-like, shape (n_samples,)
            Event indicator (1=event, 0=censored)
        start_times : array-like, shape (n_samples,), optional
            Start times for counting process format (start, stop].
            If None, assumes all observations start at time 0 (standard Cox).
            For recurrent events, provide time_start from MultiStateData.
        
        Returns:
        --------
        self : CoxPHModel
            Fitted model
        """
        # Convert to tensors
        X = self._to_tensor(X)
        durations = self._to_tensor(durations)
        events = self._to_tensor(events)
        
        # NEW: Handle start times
        if start_times is not None:
            start_times = self._to_tensor(start_times)
            # Validate start < stop
            if torch.any(start_times >= durations):
                raise ValueError("start_times must be < durations for all observations")
        
        # Validate inputs
        self._validate_input(X, durations, events)
        
        n_samples, n_features = X.shape
        
        # Standardize covariates for numerical stability
        self.X_mean_ = torch.mean(X, dim=0)
        self.X_std_ = torch.std(X, dim=0)
        self.X_std_[self.X_std_ == 0] = 1.0  # Avoid division by zero
        X_standardized = (X - self.X_mean_) / self.X_std_
        
        # Sort by event times (ascending)
        sorted_indices = torch.argsort(durations)
        durations_sorted = durations[sorted_indices]
        events_sorted = events[sorted_indices]
        X_sorted = X_standardized[sorted_indices]
        
        # Handle start times for counting process
        if start_times is not None:
            start_times = self._to_tensor(start_times)
            start_times_sorted = start_times[sorted_indices]
        else:
            start_times_sorted = None
        
        # Store sorted data for later use
        self.durations_sorted_ = durations_sorted
        self.events_sorted_ = events_sorted
        self.X_sorted_ = X_sorted
        self.sorted_indices_ = sorted_indices  # Store sorting indices
        self.start_times_sorted_ = start_times_sorted 
        
        # Create inverse mapping to unsort residuals later
        self.unsorted_indices_ = torch.argsort(sorted_indices)  # Store unsort mapping
        
        # Initialize coefficients to zero
        beta = torch.zeros(n_features, device=self.device, dtype=torch.float64)

        # Newton-Raphson optimization
        converged = False
        log_likelihood_history = []

        for iteration in range(self.max_iter):
            # Compute log-likelihood, gradient, and Hessian
            log_lik, gradient, hessian = self._compute_derivatives(
                beta, X_sorted, durations_sorted, events_sorted, start_times_sorted
            )

            log_likelihood_history.append(log_lik.item())

            # Newton-Raphson step with line search
            try:
                # Compute Newton direction
                delta = torch.linalg.solve(hessian, gradient)

                # Check convergence based on delta norm (like lifelines)
                # This is more robust than gradient norm for large datasets
                delta_norm = torch.norm(delta)
                if delta_norm < self.tol:
                    converged = True
                    break

                # Line search to ensure likelihood increases
                # Use compute_hessian=False for line search (only need log-likelihood)
                step_size = 1.0
                beta_new = beta - step_size * delta
                log_lik_new, _, _ = self._compute_derivatives(
                    beta_new, X_sorted, durations_sorted, events_sorted, start_times_sorted,
                    compute_hessian=False
                )

                # Backtracking line search with relative tolerance
                # Use relative comparison to handle floating point precision at optimum
                improvement_threshold = max(abs(log_lik.item()) * 1e-12, 1e-12)
                while log_lik_new < log_lik - improvement_threshold and step_size > 1e-8:
                    step_size *= 0.5
                    beta_new = beta - step_size * delta
                    log_lik_new, _, _ = self._compute_derivatives(
                        beta_new, X_sorted, durations_sorted, events_sorted, start_times_sorted,
                        compute_hessian=False
                    )

                beta = beta_new

            except RuntimeError:
                # Hessian is singular, add regularization
                warnings.warn("Hessian is singular, adding regularization")
                regularization = torch.eye(n_features, device=self.device) * 1e-6
                delta = torch.linalg.solve(hessian + regularization, gradient)
                beta = beta - delta

        if not converged:
            warnings.warn(f"Optimization did not converge in {self.max_iter} iterations")
        
        # Transform coefficients back to original scale
        beta_original = beta / self.X_std_
        
        # Compute final derivatives for standard errors
        final_log_lik, final_gradient, final_hessian = self._compute_derivatives(
            beta, X_sorted, durations_sorted, events_sorted, start_times_sorted
        )
        
        # Standard errors from inverse Hessian
        try:
            hessian_inv = torch.linalg.inv(-final_hessian)
            # Transform to original scale
            variance_matrix = hessian_inv / (self.X_std_.unsqueeze(1) * self.X_std_.unsqueeze(0))
            standard_errors = torch.sqrt(torch.diag(variance_matrix))
        except RuntimeError:
            warnings.warn("Could not compute standard errors (singular Hessian)")
            standard_errors = torch.full_like(beta_original, float('nan'))
            variance_matrix = torch.eye(len(beta_original), device=self.device, dtype=torch.float64) * float('nan')

        # Store results
        self.coefficients_ = beta_original.cpu().numpy()
        self.standard_errors_ = standard_errors.cpu().numpy()
        self.variance_covariance_matrix_ = variance_matrix.cpu().numpy()
        self.log_likelihood_ = final_log_lik.item()
        
        
        # Compute baseline hazard using Breslow estimator
        self._estimate_baseline_hazard(beta_original, X, durations, events, start_times)
        
        # Compute concordance index
        self.concordance_index_ = self._compute_concordance_index(
            beta_original, X, durations, events
        )
        
        self._is_fitted = True
        
        # Create result object
        self.result_ = self._create_result_object(converged, iteration + 1)
        
        return self
    
    def _compute_derivatives(self,
                        beta: torch.Tensor,
                        X: torch.Tensor,
                        durations: torch.Tensor,
                        events: torch.Tensor,
                        start_times: Optional[torch.Tensor] = None,
                        compute_hessian: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log partial likelihood, gradient, and Hessian using vectorized operations.

        Uses reverse cumulative sums for efficient O(n) computation instead of O(n²) loops.

        IMPORTANT: Always uses numpy/Numba for Efron method regardless of device.
        The Efron method requires a loop over unique event times, which creates massive
        GPU overhead. CPU with Numba JIT is much faster for this loop-heavy computation.
        GPU is only beneficial for Breslow (fully vectorized).

        Parameters:
        -----------
        compute_hessian : bool
            If False, skip Hessian computation (faster for line search).
        """
        # Always use numpy/Numba for Efron - GPU overhead for loops is prohibitive
        # Only use GPU for Breslow which is fully vectorized
        if self.device.type == 'cpu' or self.tie_method == 'efron':
            return self._compute_derivatives_numpy(beta, X, durations, events, start_times, compute_hessian)
        else:
            return self._compute_derivatives_torch(beta, X, durations, events, start_times, compute_hessian)

    def _compute_derivatives_numpy(self,
                        beta: torch.Tensor,
                        X: torch.Tensor,
                        durations: torch.Tensor,
                        events: torch.Tensor,
                        start_times: Optional[torch.Tensor] = None,
                        compute_hessian: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Numpy-based implementation for fast CPU computation.
        Uses Numba JIT compilation for Efron method when available.

        Parameters:
        -----------
        compute_hessian : bool
            If False, skip expensive Hessian computation (O(n*p²)) - useful for line search.
        """
        # Convert to numpy
        beta_np = beta.cpu().numpy().astype(np.float64)
        X_np = X.cpu().numpy().astype(np.float64)
        durations_np = durations.cpu().numpy().astype(np.float64)
        events_np = events.cpu().numpy().astype(np.float64)

        n_samples, n_features = X_np.shape

        # Compute risk scores: exp(β'X)
        eta = X_np @ beta_np
        risk_scores = np.exp(eta)

        # Reverse cumsum of risk scores
        risk_cumsum_rev = np.cumsum(risk_scores[::-1])[::-1].copy()

        # Reverse cumsum of weighted X (always needed for gradient)
        weighted_X = X_np * risk_scores[:, np.newaxis]
        weighted_X_cumsum_rev = np.cumsum(weighted_X[::-1], axis=0)[::-1].copy()

        # For Hessian: reverse cumsum of X_j X_j^T * exp(β'X_j) - EXPENSIVE
        # Only compute if needed (skip during line search)
        if compute_hessian:
            weighted_XX = X_np[:, :, np.newaxis] * X_np[:, np.newaxis, :] * risk_scores[:, np.newaxis, np.newaxis]
            weighted_XX_cumsum_rev = np.cumsum(weighted_XX[::-1], axis=0)[::-1].copy()
        else:
            weighted_XX_cumsum_rev = None

        # Check for ties - if no ties, use fast vectorized Breslow (equivalent to Efron)
        event_mask = events_np == 1
        event_times = durations_np[event_mask]
        unique_times_check, counts = np.unique(event_times, return_counts=True)
        max_ties = np.max(counts) if len(counts) > 0 else 1
        has_ties = max_ties > 1

        # Use vectorized Breslow if no ties OR if explicitly requested
        if self.tie_method == 'breslow' or not has_ties:
            # For Breslow with ties: all events at the same time must use the same
            # risk set (denominator). Fix cumulative sums so tied durations share
            # the value at the first position of their tie group.
            if has_ties and self.tie_method == 'breslow':
                unique_durs, first_indices = np.unique(durations_np, return_index=True)
                group_idx = np.searchsorted(unique_durs, durations_np)
                first_of_group = first_indices[group_idx]
                risk_cumsum_rev = risk_cumsum_rev[first_of_group]
                weighted_X_cumsum_rev = weighted_X_cumsum_rev[first_of_group]
                if weighted_XX_cumsum_rev is not None:
                    weighted_XX_cumsum_rev = weighted_XX_cumsum_rev[first_of_group]

            # Breslow - fully vectorized (also exact for Efron when no ties)
            log_likelihood = np.sum(eta[event_mask] - np.log(risk_cumsum_rev[event_mask]))
            weighted_mean_X = weighted_X_cumsum_rev[event_mask] / risk_cumsum_rev[event_mask, np.newaxis]
            gradient = np.sum(X_np[event_mask] - weighted_mean_X, axis=0)

            if compute_hessian:
                risk_at_events = risk_cumsum_rev[event_mask, np.newaxis, np.newaxis]
                weighted_mean_XX = weighted_XX_cumsum_rev[event_mask] / risk_at_events
                outer_mean = weighted_mean_X[:, :, np.newaxis] * weighted_mean_X[:, np.newaxis, :]
                hessian = -np.sum(weighted_mean_XX - outer_mean, axis=0)
            else:
                hessian = np.zeros((n_features, n_features))

        else:  # efron with ties
            # For Efron with ties, we need weighted_XX (can't skip for Hessian)
            if weighted_XX_cumsum_rev is None:
                weighted_XX = X_np[:, :, np.newaxis] * X_np[:, np.newaxis, :] * risk_scores[:, np.newaxis, np.newaxis]
                weighted_XX_cumsum_rev = np.cumsum(weighted_XX[::-1], axis=0)[::-1].copy()

            # Efron - use Numba-optimized implementation if available
            event_indices = np.where(event_mask)[0]
            unique_times = unique_times_check
            n_unique = len(unique_times)

            # Pre-compute indices for Numba
            # For each unique time: first index in sorted array and event indices
            time_to_first_idx = np.zeros(n_unique, dtype=np.int64)
            event_indices_list = []

            for t_idx, t in enumerate(unique_times):
                time_mask = durations_np == t
                time_to_first_idx[t_idx] = np.where(time_mask)[0][0]
                event_at_t = time_mask & event_mask
                event_indices_list.append(np.where(event_at_t)[0])

            if NUMBA_AVAILABLE:
                # Use Numba implementation
                # Flatten event indices for Numba compatibility
                event_indices_flat = np.concatenate(event_indices_list) if event_indices_list else np.array([], dtype=np.int64)
                event_indices_starts = np.zeros(n_unique, dtype=np.int64)
                event_indices_counts = np.zeros(n_unique, dtype=np.int64)

                offset = 0
                for t_idx in range(n_unique):
                    event_indices_starts[t_idx] = offset
                    event_indices_counts[t_idx] = len(event_indices_list[t_idx])
                    offset += event_indices_counts[t_idx]

                # Use serial version (more stable, parallel caused crashes on some systems)
                log_likelihood, gradient, hessian = _efron_loop_numba_serial(
                    beta_np, X_np, durations_np, events_np,
                    risk_scores, risk_cumsum_rev, weighted_X_cumsum_rev, weighted_XX_cumsum_rev,
                    unique_times, time_to_first_idx, event_indices_flat, event_indices_starts, event_indices_counts
                )
            else:
                # Fallback: optimized numpy implementation
                log_likelihood = 0.0
                gradient = np.zeros(n_features)
                hessian = np.zeros((n_features, n_features))

                for t_idx, t in enumerate(unique_times):
                    first_idx = time_to_first_idx[t_idx]
                    event_idx = event_indices_list[t_idx]
                    d_t = len(event_idx)

                    if d_t == 0:
                        continue

                    sum_risk = risk_cumsum_rev[first_idx]
                    sum_weighted_X = weighted_X_cumsum_rev[first_idx]
                    sum_weighted_XX = weighted_XX_cumsum_rev[first_idx]

                    X_events = X_np[event_idx]
                    risk_events = risk_scores[event_idx]

                    sum_risk_events = np.sum(risk_events)
                    sum_X_events = np.sum(X_events, axis=0)
                    sum_weighted_X_events = np.sum(X_events * risk_events[:, np.newaxis], axis=0)
                    sum_weighted_XX_events = np.einsum('ij,ik,i->jk', X_events, X_events, risk_events)

                    eta_sum = np.sum(X_events @ beta_np)

                    if d_t == 1:
                        log_likelihood += eta_sum - np.log(sum_risk)
                        weighted_mean = sum_weighted_X / sum_risk
                        gradient += sum_X_events - weighted_mean
                        weighted_mean_XX = sum_weighted_XX / sum_risk
                        hessian -= weighted_mean_XX - np.outer(weighted_mean, weighted_mean)
                    else:
                        l_vals = np.arange(d_t, dtype=np.float64)
                        fracs = l_vals / d_t
                        denoms = sum_risk - fracs * sum_risk_events

                        log_likelihood += eta_sum - np.sum(np.log(denoms))

                        # Vectorized gradient and hessian
                        adj_weighted_X_all = (sum_weighted_X - fracs[:, np.newaxis] * sum_weighted_X_events) / denoms[:, np.newaxis]
                        gradient += sum_X_events - np.sum(adj_weighted_X_all, axis=0)

                        adj_weighted_XX_all = (sum_weighted_XX - fracs[:, np.newaxis, np.newaxis] * sum_weighted_XX_events) / denoms[:, np.newaxis, np.newaxis]
                        outer_all = adj_weighted_X_all[:, :, np.newaxis] * adj_weighted_X_all[:, np.newaxis, :]
                        hessian -= np.sum(adj_weighted_XX_all - outer_all, axis=0)

        # Convert back to torch
        return (
            torch.tensor(log_likelihood, device=self.device, dtype=torch.float64),
            torch.tensor(gradient, device=self.device, dtype=torch.float64),
            torch.tensor(hessian, device=self.device, dtype=torch.float64)
        )

    def _compute_derivatives_torch(self,
                        beta: torch.Tensor,
                        X: torch.Tensor,
                        durations: torch.Tensor,
                        events: torch.Tensor,
                        start_times: Optional[torch.Tensor] = None,
                        compute_hessian: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Torch-based implementation for GPU computation.
        """
        n_samples, n_features = X.shape

        # Compute risk scores: exp(β'X)
        eta = torch.matmul(X, beta)
        risk_scores = torch.exp(eta)

        # Reverse cumsum of risk scores
        risk_cumsum_rev = torch.flip(torch.cumsum(torch.flip(risk_scores, [0]), dim=0), [0])

        # Reverse cumsum of weighted X (always needed for gradient)
        weighted_X = X * risk_scores.unsqueeze(1)
        weighted_X_cumsum_rev = torch.flip(torch.cumsum(torch.flip(weighted_X, [0]), dim=0), [0])

        # For Hessian: reverse cumsum of X_j X_j^T * exp(β'X_j) - EXPENSIVE
        if compute_hessian:
            weighted_XX = X.unsqueeze(2) * X.unsqueeze(1) * risk_scores.unsqueeze(1).unsqueeze(2)
            weighted_XX_cumsum_rev = torch.flip(torch.cumsum(torch.flip(weighted_XX, [0]), dim=0), [0])
        else:
            weighted_XX_cumsum_rev = None

        if self.tie_method == 'breslow':
            # Breslow - fully vectorized
            event_mask = events == 1

            # For Breslow with ties: all events at the same time must use the same
            # risk set. Fix cumulative sums so tied durations share the value at
            # the first position of their tie group (data is sorted ascending).
            unique_durs, inverse = torch.unique(durations, return_inverse=True)
            counts = torch.bincount(inverse)
            first_per_group = torch.zeros(len(unique_durs), dtype=torch.long, device=self.device)
            if len(counts) > 1:
                first_per_group[1:] = torch.cumsum(counts[:-1], dim=0)
            first_of_group = first_per_group[inverse]

            risk_cumsum_rev_adj = risk_cumsum_rev[first_of_group]
            weighted_X_cumsum_rev_adj = weighted_X_cumsum_rev[first_of_group]

            log_likelihood = torch.sum(eta[event_mask] - torch.log(risk_cumsum_rev_adj[event_mask]))
            weighted_mean_X = weighted_X_cumsum_rev_adj[event_mask] / risk_cumsum_rev_adj[event_mask].unsqueeze(1)
            gradient = torch.sum(X[event_mask] - weighted_mean_X, dim=0)

            if compute_hessian:
                if weighted_XX_cumsum_rev is not None:
                    weighted_XX_cumsum_rev_adj = weighted_XX_cumsum_rev[first_of_group]
                else:
                    weighted_XX = X.unsqueeze(2) * X.unsqueeze(1) * risk_scores.unsqueeze(1).unsqueeze(2)
                    weighted_XX_cumsum_rev_adj = torch.flip(torch.cumsum(torch.flip(weighted_XX, [0]), dim=0), [0])[first_of_group]
                risk_at_events = risk_cumsum_rev_adj[event_mask].unsqueeze(1).unsqueeze(2)
                weighted_mean_XX = weighted_XX_cumsum_rev_adj[event_mask] / risk_at_events
                outer_mean = weighted_mean_X.unsqueeze(2) * weighted_mean_X.unsqueeze(1)
                hessian = -torch.sum(weighted_mean_XX - outer_mean, dim=0)
            else:
                hessian = torch.zeros((n_features, n_features), device=self.device, dtype=torch.float64)

        else:  # efron
            # Efron - loop over unique times (hard to fully vectorize)
            event_mask = events == 1
            event_times = durations[event_mask]
            unique_times, inverse_indices = torch.unique(event_times, return_inverse=True)

            # For Efron, we need weighted_XX (can't skip even if only computing likelihood)
            if weighted_XX_cumsum_rev is None:
                weighted_XX = X.unsqueeze(2) * X.unsqueeze(1) * risk_scores.unsqueeze(1).unsqueeze(2)
                weighted_XX_cumsum_rev = torch.flip(torch.cumsum(torch.flip(weighted_XX, [0]), dim=0), [0])

            log_likelihood = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            gradient = torch.zeros(n_features, device=self.device, dtype=torch.float64)
            hessian = torch.zeros((n_features, n_features), device=self.device, dtype=torch.float64)

            for t in unique_times:
                time_mask = durations == t
                event_at_t = time_mask & event_mask

                first_idx = torch.where(time_mask)[0][0]

                sum_risk = risk_cumsum_rev[first_idx]
                sum_weighted_X = weighted_X_cumsum_rev[first_idx]
                sum_weighted_XX = weighted_XX_cumsum_rev[first_idx]

                X_events = X[event_at_t]
                risk_events = risk_scores[event_at_t]
                d_t = X_events.shape[0]

                if d_t == 0:
                    continue

                sum_risk_events = torch.sum(risk_events)
                sum_X_events = torch.sum(X_events, dim=0)
                sum_weighted_X_events = torch.sum(X_events * risk_events.unsqueeze(1), dim=0)
                sum_weighted_XX_events = torch.sum(
                    X_events.unsqueeze(2) * X_events.unsqueeze(1) * risk_events.unsqueeze(1).unsqueeze(2),
                    dim=0
                )

                if d_t == 1:
                    log_likelihood += torch.sum(torch.matmul(X_events, beta)) - torch.log(sum_risk)
                    weighted_mean = sum_weighted_X / sum_risk
                    gradient += sum_X_events - weighted_mean
                    weighted_mean_XX = sum_weighted_XX / sum_risk
                    hessian -= weighted_mean_XX - torch.outer(weighted_mean, weighted_mean)
                else:
                    l_vals = torch.arange(d_t, device=self.device, dtype=torch.float64)
                    fracs = l_vals / d_t

                    denoms = sum_risk - fracs * sum_risk_events

                    log_likelihood += torch.sum(torch.matmul(X_events, beta)) - torch.sum(torch.log(denoms))

                    adj_weighted_X_all = (sum_weighted_X.unsqueeze(0) - fracs.unsqueeze(1) * sum_weighted_X_events.unsqueeze(0)) / denoms.unsqueeze(1)
                    gradient += sum_X_events - torch.sum(adj_weighted_X_all, dim=0)

                    adj_weighted_XX_all = (sum_weighted_XX.unsqueeze(0) - fracs.unsqueeze(1).unsqueeze(2) * sum_weighted_XX_events.unsqueeze(0)) / denoms.unsqueeze(1).unsqueeze(2)
                    outer_all = adj_weighted_X_all.unsqueeze(2) * adj_weighted_X_all.unsqueeze(1)
                    hessian -= torch.sum(adj_weighted_XX_all - outer_all, dim=0)

        return log_likelihood, gradient, hessian
    
    def _estimate_baseline_hazard(self,
                             beta: torch.Tensor,
                             X: torch.Tensor,
                             durations: torch.Tensor,
                             events: torch.Tensor,
                             start_times: Optional[torch.Tensor] = None):
        """
        Estimate baseline hazard using Breslow estimator (fully vectorized).
        """
        # Use numpy for fast computation
        X_np = X.cpu().numpy()
        X_mean_np = self.X_mean_.cpu().numpy()
        beta_np = beta.cpu().numpy()
        durations_np = durations.cpu().numpy()
        events_np = events.cpu().numpy()

        # Compute risk scores using centered X
        X_centered = X_np - X_mean_np
        risk_scores = np.exp(X_centered @ beta_np)

        # Sort by duration
        order = np.argsort(durations_np)
        dur_sorted = durations_np[order]
        evt_sorted = events_np[order]
        risk_sorted = risk_scores[order]

        # Get unique event times
        event_mask = evt_sorted == 1
        event_times = dur_sorted[event_mask]
        unique_times, inverse = np.unique(event_times, return_inverse=True)
        n_unique = len(unique_times)

        if n_unique == 0:
            # No events
            self.timeline_ = torch.tensor([], device=self.device, dtype=torch.float64)
            self.baseline_hazard_ = torch.tensor([], device=self.device, dtype=torch.float64)
            self.baseline_cumulative_hazard_ = torch.tensor([], device=self.device, dtype=torch.float64)
            self.baseline_survival_ = torch.tensor([], device=self.device, dtype=torch.float64)
            return

        # Count events at each unique time
        d_t = np.bincount(inverse, minlength=n_unique).astype(np.float64)

        # Compute risk sums using reverse cumsum
        risk_cumsum_rev = np.cumsum(risk_sorted[::-1])[::-1]

        # For each unique time, get the index of first occurrence using searchsorted
        first_indices = np.searchsorted(dur_sorted, unique_times)

        # Get risk sums at those indices
        sum_risks = risk_cumsum_rev[first_indices]

        # Baseline hazard: d_t / sum_risk
        baseline_hazard = d_t / sum_risks

        # Store as tensors
        self.timeline_ = torch.tensor(unique_times, device=self.device, dtype=torch.float64)
        self.baseline_hazard_ = torch.tensor(baseline_hazard, device=self.device, dtype=torch.float64)

        # Cumulative baseline hazard
        self.baseline_cumulative_hazard_ = torch.cumsum(self.baseline_hazard_, dim=0)

        # Baseline survival function: S₀(t) = exp(-H₀(t))
        self.baseline_survival_ = torch.exp(-self.baseline_cumulative_hazard_)
    
    def predict_risk(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict risk scores: exp(β'X)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariate matrix
            
        Returns:
        --------
        risk_scores : np.ndarray, shape (n_samples,)
            Predicted risk scores
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X)
        
        # Standardize X using training statistics
        X_standardized = (X - self.X_mean_) / self.X_std_
        
        # Use standardized coefficients (stored internally)
        beta_std = torch.tensor(self.coefficients_, device=self.device, dtype=torch.float64) * self.X_std_
        
        risk_scores = torch.exp(torch.matmul(X_standardized, beta_std))
        return risk_scores.cpu().numpy()
    
    def predict_survival_function(self, X: Union[np.ndarray, torch.Tensor], 
                              times: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        """
        Predict survival function: S(t|X) = S₀(t)^exp(β'X)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariate matrix
        times : array-like, optional
            Time points for prediction. If None, uses fitted timeline.
            
        Returns:
        --------
        survival_probabilities : np.ndarray
            Predicted survival probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X)
        
        # Standardize X using training statistics
        X_standardized = (X - self.X_mean_) / self.X_std_
        
        # Use standardized coefficients
        beta_std = torch.tensor(self.coefficients_, device=self.device, dtype=torch.float64) * self.X_std_
        
        # Compute risk scores
        risk_scores = torch.exp(torch.matmul(X_standardized, beta_std))
        
        if times is None:
            times = self.timeline_
        else:
            times = self._to_tensor(times)
        
        # Get baseline survival at requested times
        baseline_survival_at_times = self._baseline_survival_at_times(times)
        
        # S(t|X) = S₀(t)^exp(β'X)
        survival_functions = torch.pow(
            baseline_survival_at_times.unsqueeze(0),
            risk_scores.unsqueeze(1)
        )
        
        return survival_functions.cpu().numpy()
    
    def predict_cumulative_hazard(self, X: Union[np.ndarray, torch.Tensor],
                              times: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        """
        Predict cumulative hazard: H(t|X) = H₀(t) × exp(β'X)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariate matrix
        times : array-like, optional
            Time points for prediction. If None, uses fitted timeline.
            
        Returns:
        --------
        cumulative_hazards : np.ndarray
            Predicted cumulative hazards
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X)
        
        # Standardize X using training statistics
        X_standardized = (X - self.X_mean_) / self.X_std_
        
        # Use standardized coefficients
        beta_std = torch.tensor(self.coefficients_, device=self.device, dtype=torch.float64) * self.X_std_
        
        # Compute risk scores
        risk_scores = torch.exp(torch.matmul(X_standardized, beta_std))
        
        if times is None:
            times = self.timeline_
        else:
            times = self._to_tensor(times)
        
        # Get baseline cumulative hazard at requested times
        baseline_cum_hazard_at_times = self._baseline_cumulative_hazard_at_times(times)
        
        # H(t|X) = H₀(t) × exp(β'X)
        cumulative_hazards = baseline_cum_hazard_at_times.unsqueeze(0) * risk_scores.unsqueeze(1)
        
        return cumulative_hazards.cpu().numpy()
    
    def _baseline_survival_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Get baseline survival function at specified times."""
        survival_at_times = torch.ones_like(times, device=self.device, dtype=torch.float64)
        
        for i, t in enumerate(times):
            mask = self.timeline_ <= t
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]
                survival_at_times[i] = self.baseline_survival_[idx]
        
        return survival_at_times
    
    def _baseline_cumulative_hazard_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Get baseline cumulative hazard at specified times."""
        cum_hazard_at_times = torch.zeros_like(times, device=self.device, dtype=torch.float64)
        
        for i, t in enumerate(times):
            mask = self.timeline_ <= t
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]
                cum_hazard_at_times[i] = self.baseline_cumulative_hazard_[idx]
        
        return cum_hazard_at_times
    
    def _compute_concordance_index(self,
                               beta: torch.Tensor,
                               X: torch.Tensor,
                               durations: torch.Tensor,
                               events: torch.Tensor) -> float:
        """
        Compute concordance index (C-index) using O(n log n) algorithm.

        For all pairs (i,j) where ti < tj and delta_i=1:
        - Concordant if risk(i) > risk(j)
        - Discordant if risk(i) < risk(j)
        - Tie if risk(i) = risk(j)

        C-index = (concordant + 0.5*ties) / total_pairs

        Uses efficient sorting-based algorithm instead of O(n²) pairwise comparison.
        Correctly handles tied durations by processing duration groups together.
        """
        # Compute risk scores and move to numpy for fast computation
        risk_scores = torch.exp(torch.matmul(X, beta)).cpu().numpy()
        durations_np = durations.cpu().numpy()
        events_np = events.cpu().numpy()

        n = len(durations_np)
        if np.sum(events_np) == 0:
            return 0.5

        # Sort by duration (primary) and by -risk (secondary for tie-breaking)
        order = np.lexsort((-risk_scores, durations_np))
        durations_sorted = durations_np[order]
        events_sorted = events_np[order]
        risk_sorted = risk_scores[order]

        # Get unique risk values and their ranks
        unique_risks, risk_ranks = np.unique(risk_sorted, return_inverse=True)
        n_unique_risks = len(unique_risks)

        # Binary indexed tree (Fenwick tree) for efficient prefix sums
        bit = np.zeros(n_unique_risks + 1, dtype=np.int64)

        def bit_update(idx, delta=1):
            idx += 1
            while idx <= n_unique_risks:
                bit[idx] += delta
                idx += idx & (-idx)

        def bit_query(idx):
            idx += 1
            s = 0
            while idx > 0:
                s += bit[idx]
                idx -= idx & (-idx)
            return s

        concordant = 0
        discordant = 0
        tied_risk = 0

        # Get unique durations and process each group
        unique_durations = np.unique(durations_sorted)

        # Process from longest to shortest duration
        for dur in unique_durations[::-1]:
            # Find indices with this duration
            dur_mask = durations_sorted == dur
            dur_indices = np.where(dur_mask)[0]

            # For events at this duration, count pairs with observations at LONGER durations
            # (those already in the BIT)
            for i in dur_indices:
                if events_sorted[i] == 1:
                    rank_i = risk_ranks[i]

                    # Concordant: observations with lower risk already seen
                    n_lower = bit_query(rank_i - 1) if rank_i > 0 else 0

                    # Same risk (ties)
                    n_same = bit_query(rank_i) - n_lower

                    # Discordant: observations with higher risk already seen
                    n_higher = (bit_query(n_unique_risks - 1) - bit_query(rank_i))

                    concordant += n_lower
                    tied_risk += n_same
                    discordant += n_higher

            # Add ALL observations at this duration to the BIT AFTER processing events
            for i in dur_indices:
                bit_update(risk_ranks[i])

        total_pairs = concordant + discordant + tied_risk

        if total_pairs == 0:
            return 0.5

        c_index = (concordant + 0.5 * tied_risk) / total_pairs
        return float(c_index)
    
    def _create_result_object(self, converged: bool, iterations: int) -> CoxPHResult:
        """Create a CoxPHResult object with all fitted values."""
        # Compute statistics
        hazard_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / self.standard_errors_
        
        # Compute p-values
        try:
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        except ImportError:
            p_values = np.full_like(z_scores, np.nan)
        
        convergence_info = {
            'converged': converged,
            'iterations': iterations,
            'tolerance': self.tol,
            'tie_method': self.tie_method
        }
        
        return CoxPHResult(
            coefficients=self.coefficients_,
            standard_errors=self.standard_errors_,
            hazard_ratios=hazard_ratios,
            z_scores=z_scores,
            p_values=p_values,
            concordance_index=self.concordance_index_,
            log_likelihood=self.log_likelihood_,
            baseline_hazard=self.baseline_hazard_,
            baseline_cumulative_hazard=self.baseline_cumulative_hazard_,
            baseline_survival=self.baseline_survival_,
            timeline=self.timeline_,
            convergence_info=convergence_info
        )
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input data to tensor and move to device."""
        if isinstance(data, torch.Tensor):
            # If already a tensor, convert to float64 and move to device
            # Handle MPS tensors by moving to CPU first
            if data.device.type == 'mps':
                data = data.cpu()
            return data.to(self.device).double()
        else:
            return torch.tensor(data, device=self.device, dtype=torch.float64)
    
    def _validate_input(self, X: torch.Tensor, durations: torch.Tensor, events: torch.Tensor):
        """Validate input data."""
        if len(X) != len(durations) or len(X) != len(events):
            raise ValueError("X, durations, and events must have the same length")
        
        if torch.any(durations < 0):
            raise ValueError("durations must be non-negative")
        
        if not torch.all((events == 0) | (events == 1)):
            raise ValueError("events must be binary (0 or 1)")
        
        if torch.sum(events) == 0:
            raise ValueError("Need at least one event in the data")
        

    def get_confidence_intervals(self, alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals for coefficients.
        
        Parameters:
        -----------
        alpha : float, optional
            Significance level. If None, uses self.alpha
            
        Returns:
        --------
        ci_lower : np.ndarray
            Lower bounds of (1-alpha)*100% confidence intervals
        ci_upper : np.ndarray  
            Upper bounds of (1-alpha)*100% confidence intervals
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        if alpha is None:
            alpha = self.alpha
        
        try:
            from scipy.stats import norm
            z = norm.ppf(1 - alpha/2)
        except ImportError:
            # Fallback to approximate values
            if alpha == 0.05:
                z = 1.96
            elif alpha == 0.01:
                z = 2.576
            else:
                raise ImportError("scipy required for arbitrary alpha values")
        
        ci_lower = self.coefficients_ - z * self.standard_errors_
        ci_upper = self.coefficients_ + z * self.standard_errors_
        
        return ci_lower, ci_upper


    def compute_martingale_residuals(self) -> np.ndarray:
        """
        Compute Martingale residuals.
        
        Martingale residuals are useful for:
        - Checking functional form of covariates
        - Identifying influential observations
        
        Definition: M_i = δ_i - H_0(T_i) * exp(β'X_i)
        
        where:
        - δ_i is the event indicator
        - H_0(T_i) is the baseline cumulative hazard at time T_i
        - exp(β'X_i) is the risk score
        
        Returns:
        --------
        residuals : np.ndarray, shape (n_samples,)
            Martingale residuals
        
        References:
        -----------
        Therneau, Grambsch, and Fleming (1990)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing residuals")
        
        # Get risk scores
        risk_scores = self.predict_risk(self.X_sorted_.cpu().numpy())
        
        # Get baseline cumulative hazard at each observation time
        durations = self.durations_sorted_.cpu().numpy()
        events = self.events_sorted_.cpu().numpy()
        
        # Compute H_0(T_i) for each observation
        baseline_cum_hazard_at_times = np.zeros(len(durations))
        for i, t in enumerate(durations):
            mask = self.timeline_.cpu().numpy() <= t
            if np.any(mask):
                idx = np.where(mask)[0][-1]
                baseline_cum_hazard_at_times[i] = self.baseline_cumulative_hazard_[idx].item()
        
        # Martingale residuals: δ_i - H_0(T_i) * exp(β'X_i)
        martingale_residuals = events - baseline_cum_hazard_at_times * risk_scores
        
        return martingale_residuals


    def compute_deviance_residuals(self) -> np.ndarray:
        """
        Compute Deviance residuals.
        
        Deviance residuals are useful for:
        - Identifying outliers
        - Assessing overall model fit
        
        Definition: D_i = sign(M_i) * sqrt(-2 * [M_i + δ_i * log(δ_i - M_i)])
        
        where M_i is the Martingale residual.
        
        These are more symmetric than Martingale residuals and better for
        detecting outliers.
        
        Returns:
        --------
        residuals : np.ndarray, shape (n_samples,)
            Deviance residuals
            
        References:
        -----------
        Therneau, Grambsch, and Fleming (1990)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing residuals")
        
        martingale = self.compute_martingale_residuals()
        events = self.events_sorted_.cpu().numpy()
        
        # For deviance residuals, we need special handling
        deviance = np.zeros_like(martingale)
        
        for i in range(len(martingale)):
            if events[i] == 1:
                # For events: sign(M) * sqrt(-2 * [M + log(1 - M)])
                # Since δ=1, this simplifies
                if martingale[i] < 1:
                    deviance[i] = np.sign(martingale[i]) * np.sqrt(
                        -2 * (martingale[i] + np.log(1 - martingale[i]))
                    )
                else:
                    # Edge case: if martingale >= 1, set to large value
                    deviance[i] = np.sign(martingale[i]) * np.sqrt(-2 * martingale[i])
            else:
                # For censored: sign(M) * sqrt(-2 * M)
                deviance[i] = np.sign(martingale[i]) * np.sqrt(-2 * martingale[i])
        
        return deviance


    def compute_schoenfeld_residuals(self) -> np.ndarray:
        """
        Compute Schoenfeld residuals for testing proportional hazards assumption.
        
        Schoenfeld residuals are computed only at event times and are useful for:
        - Testing proportional hazards assumption
        - Identifying time-varying effects
        
        Definition for covariate j at event time i:
        S_ij = X_ij - E[X_j | R(t_i)]
        
        where E[X_j | R(t_i)] is the expected value of covariate j over the risk set.
        
        Returns:
        --------
        residuals : np.ndarray, shape (n_events, n_features)
            Schoenfeld residuals (one row per event)
            
        References:
        -----------
        Schoenfeld (1982). Partial residuals for the proportional hazards model
        Grambsch and Therneau (1994). Proportional hazards tests
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing residuals")
        
        X = self.X_sorted_.cpu().numpy()
        durations = self.durations_sorted_.cpu().numpy()
        events = self.events_sorted_.cpu().numpy()
        
        # Get event times only
        event_mask = events == 1
        event_times = durations[event_mask]
        X_events = X[event_mask]
        
        n_events = len(event_times)
        n_features = X.shape[1]
        
        # Compute risk scores (centered)
        X_centered = X - self.X_mean_.cpu().numpy()
        beta = torch.tensor(self.coefficients_, device=self.device, dtype=torch.float64)
        risk_scores = torch.exp(torch.matmul(
            torch.tensor(X_centered, device=self.device, dtype=torch.float64), 
            beta
        )).cpu().numpy()
        
        schoenfeld_residuals = np.zeros((n_events, n_features))
        
        # For each event time
        event_idx = 0
        for i, t in enumerate(durations):
            if events[i] == 0:
                continue
            
            # Risk set at this time
            at_risk_mask = durations >= t
            X_at_risk = X[at_risk_mask]
            risk_at_risk = risk_scores[at_risk_mask]
            
            # Expected value of X in risk set (weighted by risk scores)
            sum_risk = np.sum(risk_at_risk)
            expected_X = np.sum(X_at_risk * risk_at_risk[:, np.newaxis], axis=0) / sum_risk
            
            # Schoenfeld residual: observed - expected
            schoenfeld_residuals[event_idx] = X[i] - expected_X
            event_idx += 1
        
        return schoenfeld_residuals


    def compute_score_residuals(self) -> np.ndarray:
        """
        Compute Score (dfbeta) residuals for influence diagnostics.
        
        Score residuals measure the influence of each observation on the
        coefficient estimates. Also known as dfbeta residuals.
        
        Definition: L_i = I^(-1) * U_i
        
        where:
        - I^(-1) is the inverse of the information matrix
        - U_i is the score contribution from observation i
        
        Returns:
        --------
        residuals : np.ndarray, shape (n_samples, n_features)
            Score residuals (influence on each coefficient)
            
        References:
        -----------
        Therneau and Grambsch (2000). Modeling Survival Data
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing residuals")
        
        X = self.X_sorted_
        durations = self.durations_sorted_
        events = self.events_sorted_
        
        n_samples = len(durations)
        n_features = X.shape[1]
        
        # Get standardized coefficients
        beta_std = torch.tensor(self.coefficients_, device=self.device, dtype=torch.float64) * self.X_std_
        
        # Compute information matrix (Hessian)
        _, _, hessian = self._compute_derivatives(beta_std, X, durations, events)
        
        # Inverse information matrix
        try:
            info_inv = torch.linalg.inv(-hessian)
        except RuntimeError:
            warnings.warn("Information matrix is singular, using pseudo-inverse")
            info_inv = torch.linalg.pinv(-hessian)
        
        # Compute score contribution for each observation
        risk_scores = torch.exp(torch.matmul(X, beta_std))
        score_residuals = torch.zeros((n_samples, n_features), device=self.device, dtype=torch.float64)
        
        for i in range(n_samples):
            if events[i] == 0:
                continue
            
            t_i = durations[i]
            
            # Risk set at time t_i
            at_risk_mask = durations >= t_i
            X_at_risk = X[at_risk_mask]
            risk_at_risk = risk_scores[at_risk_mask]
            
            # Expected X in risk set
            sum_risk = torch.sum(risk_at_risk)
            expected_X = torch.sum(X_at_risk * risk_at_risk.unsqueeze(1), dim=0) / sum_risk
            
            # Score contribution: U_i = X_i - E[X | R(t_i)]
            score_contrib = X[i] - expected_X
            
            # Transform back to original scale and apply information matrix
            score_contrib_original = score_contrib / self.X_std_
            score_residuals[i] = torch.matmul(info_inv, score_contrib_original)
        
        return score_residuals.cpu().numpy()


    def check_proportional_hazards(self, plot: bool = False) -> Dict:
        """
        Test proportional hazards assumption using Schoenfeld residuals.
        
        Tests correlation between scaled Schoenfeld residuals and time.
        Significant correlation suggests violation of proportional hazards.
        
        Parameters:
        -----------
        plot : bool, default=False
            If True, create diagnostic plots (requires matplotlib)
            
        Returns:
        --------
        results : dict
            Dictionary with test statistics for each covariate:
            - 'correlation': Correlation with time
            - 'p_value': P-value for test
            - 'global_test': Global test statistic and p-value
            
        References:
        -----------
        Grambsch and Therneau (1994). Proportional hazards tests and diagnostics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before testing assumptions")
        
        # Get Schoenfeld residuals
        schoenfeld = self.compute_schoenfeld_residuals()
        
        # Get event times
        events = self.events_sorted_.cpu().numpy()
        durations = self.durations_sorted_.cpu().numpy()
        event_times = durations[events == 1]
        
        n_features = schoenfeld.shape[1]
        results = {
            'variables': [],
            'correlation': [],
            'chi2': [],
            'p_value': []
        }
        
        try:
            from scipy.stats import pearsonr, chi2
            
            # Test each covariate
            for j in range(n_features):
                # Correlation between residuals and time
                corr, p_value = pearsonr(event_times, schoenfeld[:, j])
                
                # Chi-square statistic
                n_events = len(event_times)
                chi2_stat = (corr ** 2) * n_events
                p_val_chi2 = 1 - chi2.cdf(chi2_stat, df=1)
                
                results['variables'].append(f'X{j}')
                results['correlation'].append(corr)
                results['chi2'].append(chi2_stat)
                results['p_value'].append(p_val_chi2)
            
            # Global test (sum of chi-squares)
            global_chi2 = np.sum(results['chi2'])
            global_p = 1 - chi2.cdf(global_chi2, df=n_features)
            
            results['global_test'] = {
                'chi2': global_chi2,
                'df': n_features,
                'p_value': global_p
            }
            
            # Print summary
            print("\n" + "="*70)
            print("Proportional Hazards Test (Schoenfeld Residuals)")
            print("="*70)
            print(f"{'Variable':<15} {'Corr':>10} {'Chi2':>10} {'p-value':>10}")
            print("-"*70)
            for i in range(len(results['variables'])):
                print(f"{results['variables'][i]:<15} {results['correlation'][i]:>10.4f} "
                    f"{results['chi2'][i]:>10.4f} {results['p_value'][i]:>10.4f}")
            print("-"*70)
            print(f"{'GLOBAL':<15} {'':<10} {global_chi2:>10.4f} {global_p:>10.4f}")
            print("="*70)
            print("Null hypothesis: Proportional hazards assumption holds")
            print("If p-value < 0.05, assumption may be violated")
            
            if plot:
                try:
                    import matplotlib.pyplot as plt
                    
                    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
                    if n_features == 1:
                        axes = [axes]
                    
                    for j in range(n_features):
                        axes[j].scatter(event_times, schoenfeld[:, j], alpha=0.5)
                        axes[j].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                        
                        # Add trend line
                        z = np.polyfit(event_times, schoenfeld[:, j], 1)
                        p = np.poly1d(z)
                        axes[j].plot(event_times, p(event_times), "r-", alpha=0.5)
                        
                        axes[j].set_xlabel('Time')
                        axes[j].set_ylabel('Schoenfeld Residual')
                        axes[j].set_title(f'{results["variables"][j]}\n'
                                        f'ρ={results["correlation"][j]:.3f}, '
                                        f'p={results["p_value"][j]:.3f}')
                        axes[j].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show()
                    
                except ImportError:
                    warnings.warn("matplotlib not available for plotting")
            
            return results
            
        except ImportError:
            warnings.warn("scipy required for proportional hazards test")
            return {}
        
    

    def compute_robust_variance(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        cluster_id: np.ndarray
    ) -> np.ndarray:
        """Compute robust variance using SORTED data from fitting."""
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing robust variance")
        
        # USE SORTED DATA FROM FITTING
        X_sorted_standardized = self.X_sorted_.cpu().numpy()
        durations_sorted = self.durations_sorted_.cpu().numpy()
        events_sorted = self.events_sorted_.cpu().numpy()
        sorted_indices = self.sorted_indices_.cpu().numpy()
        
        # CRITICAL: Get start times if available
        if self.start_times_sorted_ is not None:
            start_times_sorted = self.start_times_sorted_.cpu().numpy()
        else:
            start_times_sorted = np.zeros_like(durations_sorted)
        
        # Sort cluster_id to match
        cluster_id_sorted = cluster_id[sorted_indices]
        
        n, p = X_sorted_standardized.shape
        beta = self.coefficients_
        I_inv = self.variance_covariance_matrix_
        
        # Convert standardized X back to centered
        X_std = self.X_std_.cpu().numpy()
        X_centered = X_sorted_standardized * X_std
        
        # Compute risk scores
        exp_eta = np.exp(X_centered @ beta)
        
        unique_clusters = np.unique(cluster_id)
        n_clusters = len(unique_clusters)
        score_matrix = np.zeros((n_clusters, p))
        
        event_times = np.unique(durations_sorted[events_sorted == 1])
        
        for t in event_times:
            # CRITICAL FIX: Use counting process risk set!
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
            
            if self.tie_method == 'efron' and n_events > 1:
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
        
        B = score_matrix.T @ score_matrix
        robust_var = I_inv @ B @ I_inv
        
        return robust_var
    

    





class StratifiedCoxPHModel:
    """
    Stratified Cox Proportional Hazards Model.
    
    Allows for different baseline hazards in different strata while
    assuming the same covariate effects across strata.
    
    Use when:
    - Proportional hazards assumption violated for a categorical variable
    - Want to adjust for a confounding factor without estimating its effect
    - Different baseline risk in subgroups (e.g., gender, treatment center)
    
    Model: h(t|X,S) = h_{0,s}(t) × exp(β'X)
    
    where S is the stratum and h_{0,s}(t) is stratum-specific baseline hazard.
    
    References:
    -----------
    - Kalbfleisch and Prentice (2002). The Statistical Analysis of Failure Time Data
    - Therneau and Grambsch (2000). Modeling Survival Data
    """
    
    def __init__(self,
                 tie_method: str = 'efron',
                 alpha: float = 0.05,
                 max_iter: int = 50,
                 tol: float = 1e-6,
                 device: Optional[str] = None):
        """
        Initialize Stratified Cox model.
        
        Parameters same as CoxPHModel.
        """
        self.base_model = CoxPHModel(
            tie_method=tie_method,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            device=device
        )
        
        self.tie_method = tie_method
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.device = self.base_model.device
        
        # Stratification-specific attributes
        self.strata_ = None
        self.unique_strata_ = None
        self.baseline_hazards_ = {}  # One per stratum
        self.baseline_cumulative_hazards_ = {}
        self.baseline_survivals_ = {}
        self.timelines_ = {}
        
        self._is_fitted = False
    
    def fit(self,
            X: Union[np.ndarray, torch.Tensor],
            durations: Union[np.ndarray, torch.Tensor],
            events: Union[np.ndarray, torch.Tensor],
            strata: Union[np.ndarray, torch.Tensor]) -> 'StratifiedCoxPHModel':
        """
        Fit stratified Cox model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariate matrix
        durations : array-like, shape (n_samples,)
            Time to event or censoring
        events : array-like, shape (n_samples,)
            Event indicator (1=event, 0=censored)
        strata : array-like, shape (n_samples,)
            Stratum indicator for each observation
            
        Returns:
        --------
        self : StratifiedCoxPHModel
        """
        # Convert to tensors
        X = self.base_model._to_tensor(X)
        durations = self.base_model._to_tensor(durations)
        events = self.base_model._to_tensor(events)
        strata = self.base_model._to_tensor(strata)
        
        n_samples, n_features = X.shape
        
        # Validate
        if len(strata) != n_samples:
            raise ValueError("strata must have same length as X")
        
        # Store strata info
        self.unique_strata_ = torch.unique(strata)
        self.strata_ = strata
        
        print(f"Fitting stratified Cox model with {len(self.unique_strata_)} strata")
        
        # Standardize covariates
        self.X_mean_ = torch.mean(X, dim=0)
        self.X_std_ = torch.std(X, dim=0)
        self.X_std_[self.X_std_ == 0] = 1.0
        X_standardized = (X - self.X_mean_) / self.X_std_
        
        # Store for later use
        self.X_sorted_ = X_standardized
        self.durations_sorted_ = durations
        self.events_sorted_ = events
        
        # Initialize coefficients
        beta = torch.zeros(n_features, device=self.device, dtype=torch.float64)
        
        # Newton-Raphson optimization with stratification
        converged = False
        log_likelihood_history = []
        
        for iteration in range(self.max_iter):
            # Compute derivatives across all strata
            log_lik, gradient, hessian = self._compute_stratified_derivatives(
                beta, X_standardized, durations, events, strata
            )
            
            log_likelihood_history.append(log_lik.item())
            
            # Check convergence
            gradient_norm = torch.norm(gradient)
            if gradient_norm < self.tol:
                converged = True
                break
            
            # Newton-Raphson step
            try:
                delta = torch.linalg.solve(hessian, gradient)
                
                # Line search
                step_size = 1.0
                beta_new = beta - step_size * delta
                log_lik_new, _, _ = self._compute_stratified_derivatives(
                    beta_new, X_standardized, durations, events, strata
                )
                
                while log_lik_new < log_lik and step_size > 1e-8:
                    step_size *= 0.5
                    beta_new = beta - step_size * delta
                    log_lik_new, _, _ = self._compute_stratified_derivatives(
                        beta_new, X_standardized, durations, events, strata
                    )
                
                beta = beta_new
                
            except RuntimeError:
                warnings.warn("Hessian is singular, adding regularization")
                regularization = torch.eye(n_features, device=self.device) * 1e-6
                delta = torch.linalg.solve(hessian + regularization, gradient)
                beta = beta - delta
        
        if not converged:
            warnings.warn(f"Optimization did not converge in {self.max_iter} iterations")
        
        # Transform coefficients back
        beta_original = beta / self.X_std_
        
        # Compute final derivatives for standard errors
        final_log_lik, final_gradient, final_hessian = self._compute_stratified_derivatives(
            beta, X_standardized, durations, events, strata
        )
        
        # Standard errors
        try:
            hessian_inv = torch.linalg.inv(-final_hessian)
            variance_matrix = hessian_inv / (self.X_std_.unsqueeze(1) * self.X_std_.unsqueeze(0))
            standard_errors = torch.sqrt(torch.diag(variance_matrix))
        except RuntimeError:
            warnings.warn("Could not compute standard errors")
            standard_errors = torch.full_like(beta_original, float('nan'))
        
        # Store results
        self.coefficients_ = beta_original.cpu().numpy()
        self.standard_errors_ = standard_errors.cpu().numpy()
        self.log_likelihood_ = final_log_lik.item()
        
        # Estimate baseline hazard for each stratum
        self._estimate_stratified_baseline_hazards(beta_original, X, durations, events, strata)
        
        # Compute concordance index
        self.concordance_index_ = self.base_model._compute_concordance_index(
            beta_original, X, durations, events
        )
        
        self._is_fitted = True
        
        # Create result object
        self.result_ = self._create_result_object(converged, iteration + 1)
        
        return self
    
    def _compute_stratified_derivatives(self,
                                       beta: torch.Tensor,
                                       X: torch.Tensor,
                                       durations: torch.Tensor,
                                       events: torch.Tensor,
                                       strata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log partial likelihood, gradient, and Hessian for stratified model.
        
        The partial likelihood is the product over strata:
        L(β) = ∏_s L_s(β)
        
        So log L(β) = Σ_s log L_s(β)
        
        And derivatives are sums over strata.
        """
        n_features = X.shape[1]
        
        total_log_lik = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        total_gradient = torch.zeros(n_features, device=self.device, dtype=torch.float64)
        total_hessian = torch.zeros((n_features, n_features), device=self.device, dtype=torch.float64)
        
        # Compute derivatives for each stratum
        for s in self.unique_strata_:
            stratum_mask = (strata == s)
            
            X_s = X[stratum_mask]
            durations_s = durations[stratum_mask]
            events_s = events[stratum_mask]
            
            # Sort within stratum
            sorted_indices = torch.argsort(durations_s)
            durations_s_sorted = durations_s[sorted_indices]
            events_s_sorted = events_s[sorted_indices]
            X_s_sorted = X_s[sorted_indices]
            
            # Compute derivatives for this stratum
            log_lik_s, gradient_s, hessian_s = self.base_model._compute_derivatives(
                beta, X_s_sorted, durations_s_sorted, events_s_sorted
            )
            
            # Accumulate
            total_log_lik += log_lik_s
            total_gradient += gradient_s
            total_hessian += hessian_s
        
        return total_log_lik, total_gradient, total_hessian
    
    def _estimate_stratified_baseline_hazards(self,
                                             beta: torch.Tensor,
                                             X: torch.Tensor,
                                             durations: torch.Tensor,
                                             events: torch.Tensor,
                                             strata: torch.Tensor):
        """
        Estimate baseline hazard for each stratum using Breslow estimator.
        """
        # Compute risk scores (centered)
        X_centered = X - self.X_mean_
        risk_scores = torch.exp(torch.matmul(X_centered, beta))
        
        # Estimate for each stratum
        for s in self.unique_strata_:
            stratum_mask = (strata == s)
            
            durations_s = durations[stratum_mask]
            events_s = events[stratum_mask]
            risk_scores_s = risk_scores[stratum_mask]
            
            # Get unique event times in this stratum
            event_times = durations_s[events_s == 1]
            unique_times = torch.unique(event_times)
            
            baseline_hazard = []
            timeline = []
            
            for t in unique_times:
                at_risk_mask = durations_s >= t
                sum_risk = torch.sum(risk_scores_s[at_risk_mask])
                
                event_mask = (durations_s == t) & (events_s == 1)
                d_t = torch.sum(event_mask)
                
                h0_t = d_t / sum_risk
                
                timeline.append(t.item())
                baseline_hazard.append(h0_t.item())
            
            s_key = s.item()
            self.timelines_[s_key] = torch.tensor(timeline, device=self.device)
            self.baseline_hazards_[s_key] = torch.tensor(baseline_hazard, device=self.device)
            self.baseline_cumulative_hazards_[s_key] = torch.cumsum(
                self.baseline_hazards_[s_key], dim=0
            )
            self.baseline_survivals_[s_key] = torch.exp(
                -self.baseline_cumulative_hazards_[s_key]
            )
    
    def predict_survival_function(self,
                                  X: Union[np.ndarray, torch.Tensor],
                                  strata: Union[np.ndarray, torch.Tensor],
                                  times: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        """
        Predict survival function for stratified model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Covariate matrix
        strata : array-like, shape (n_samples,)
            Stratum for each observation
        times : array-like, optional
            Time points for prediction
            
        Returns:
        --------
        survival_probabilities : np.ndarray
            Predicted survival probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.base_model._to_tensor(X)
        strata = self.base_model._to_tensor(strata)
        
        # Standardize
        X_standardized = (X - self.X_mean_) / self.X_std_
        beta_std = torch.tensor(self.coefficients_, device=self.device, dtype=torch.float64) * self.X_std_
        
        # Compute risk scores
        risk_scores = torch.exp(torch.matmul(X_standardized, beta_std))
        
        n_samples = X.shape[0]
        
        # Determine time points
        if times is None:
            # Use union of all stratum timelines
            all_times = []
            for s in self.unique_strata_:
                all_times.extend(self.timelines_[s.item()].cpu().numpy())
            times = torch.tensor(sorted(set(all_times)), device=self.device)
        else:
            times = self.base_model._to_tensor(times)
        
        n_times = len(times)
        survival_functions = torch.zeros((n_samples, n_times), device=self.device, dtype=torch.float64)
        
        # Predict for each observation using their stratum
        for i in range(n_samples):
            s = strata[i].item()
            
            if s not in self.baseline_survivals_:
                warnings.warn(f"Stratum {s} not seen in training, using S(t)=1")
                survival_functions[i, :] = 1.0
                continue
            
            # Get baseline survival for this stratum at requested times
            baseline_surv = self._baseline_survival_at_times(times, s)
            
            # S(t|X) = S_0(t)^exp(β'X)
            survival_functions[i, :] = torch.pow(baseline_surv, risk_scores[i])
        
        return survival_functions.cpu().numpy()
    
    def _baseline_survival_at_times(self, times: torch.Tensor, stratum: int) -> torch.Tensor:
        """Get baseline survival for a specific stratum at specified times."""
        timeline_s = self.timelines_[stratum]
        baseline_surv_s = self.baseline_survivals_[stratum]
        
        survival_at_times = torch.ones_like(times, device=self.device, dtype=torch.float64)
        
        for i, t in enumerate(times):
            mask = timeline_s <= t
            if torch.any(mask):
                idx = torch.where(mask)[0][-1]
                survival_at_times[i] = baseline_surv_s[idx]
        
        return survival_at_times
    
    def _create_result_object(self, converged: bool, iterations: int):
        """Create result object."""
        hazard_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / self.standard_errors_
        
        try:
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        except ImportError:
            p_values = np.full_like(z_scores, np.nan)
        
        convergence_info = {
            'converged': converged,
            'iterations': iterations,
            'tolerance': self.tol,
            'tie_method': self.tie_method,
            'n_strata': len(self.unique_strata_)
        }
        
        # Use dummy baseline functions for result object
        return CoxPHResult(
            coefficients=self.coefficients_,
            standard_errors=self.standard_errors_,
            hazard_ratios=hazard_ratios,
            z_scores=z_scores,
            p_values=p_values,
            concordance_index=self.concordance_index_,
            log_likelihood=self.log_likelihood_,
            baseline_hazard=torch.tensor([]),  # Stratum-specific, stored separately
            baseline_cumulative_hazard=torch.tensor([]),
            baseline_survival=torch.tensor([]),
            timeline=torch.tensor([]),
            convergence_info=convergence_info
        )
    



  

    def compute_robust_variance(self, cluster_id: np.ndarray) -> np.ndarray:
        """
        Compute robust variance with clustering for stratified Cox model.
        
        Parameters:
        -----------
        cluster_id : np.ndarray
            Cluster/subject identifiers (in original order)
        
        Returns:
        --------
        robust_variance : np.ndarray
            Robust variance-covariance matrix
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing robust variance")
        
        import warnings
        
        # Get stored data (in original order)
        X_standardized = self.X_sorted_.cpu().numpy()
        durations = self.durations_sorted_.cpu().numpy()
        events = self.events_sorted_.cpu().numpy()
        strata = self.strata_.cpu().numpy()
        
        n, p = X_standardized.shape
        beta = self.coefficients_
        
        # Convert to centered scale (needed for robust variance)
        X_std = self.X_std_.cpu().numpy()
        X_centered = X_standardized * X_std
        
        # Compute risk scores
        exp_eta = np.exp(X_centered @ beta)
        
        # Get information matrix inverse
        # Try to use stored variance matrix, otherwise compute from standard errors
        if hasattr(self, 'variance_covariance_matrix_'):
            I_inv = self.variance_covariance_matrix_
        else:
            # Use diagonal matrix from standard errors
            # Note: This is an approximation if there are correlations
            I_inv = np.diag(self.standard_errors_ ** 2)
            warnings.warn(
                "Using diagonal variance matrix (ignoring correlations). "
                "For better accuracy, store full variance_covariance_matrix_ during fitting."
            )
        
        # Initialize score matrix for clusters
        unique_clusters = np.unique(cluster_id)
        n_clusters = len(unique_clusters)
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        score_matrix = np.zeros((n_clusters, p))
        
        # Compute score contributions per cluster, per stratum
        unique_strata = np.unique(strata)
        
        for s in unique_strata:
            # Get data for this stratum
            stratum_mask = (strata == s)
            X_s = X_centered[stratum_mask]
            dur_s = durations[stratum_mask]
            evt_s = events[stratum_mask]
            exp_eta_s = exp_eta[stratum_mask]
            cluster_s = cluster_id[stratum_mask]
            
            # Get event times in this stratum
            event_times = np.unique(dur_s[evt_s == 1])
            
            for t in event_times:
                # Risk set and events at time t
                at_risk = dur_s >= t
                at_event = (dur_s == t) & (evt_s == 1)
                
                if not np.any(at_event):
                    continue
                
                event_indices = np.where(at_event)[0]
                n_events = len(event_indices)
                
                # Risk set values
                risk_exp = exp_eta_s[at_risk]
                risk_X = X_s[at_risk]
                sum_risk_exp = np.sum(risk_exp)
                
                if sum_risk_exp == 0:
                    continue
                
                # Handle ties with Efron approximation if requested
                if self.tie_method == 'efron' and n_events > 1:
                    event_exp = exp_eta_s[at_event]
                    event_X = X_s[at_event]
                    sum_event_exp = np.sum(event_exp)
                    
                    for k in range(n_events):
                        event_idx = event_indices[k]
                        frac = k / n_events
                        denom = sum_risk_exp - frac * sum_event_exp
                        
                        if denom <= 0:
                            continue
                        
                        # Expected X in modified risk set (Efron)
                        num = (np.sum(risk_X * risk_exp[:, np.newaxis], axis=0) - 
                            frac * np.sum(event_X * event_exp[:, np.newaxis], axis=0))
                        expected_X = num / denom
                        
                        # Score contribution
                        score_contrib = X_s[event_idx] - expected_X
                        
                        # Add to cluster's score
                        cluster_idx = cluster_to_idx[cluster_s[event_idx]]
                        score_matrix[cluster_idx] += score_contrib
                else:
                    # Breslow or single event
                    expected_X = np.sum(risk_X * risk_exp[:, np.newaxis], axis=0) / sum_risk_exp
                    
                    for event_idx in event_indices:
                        # Score contribution
                        score_contrib = X_s[event_idx] - expected_X
                        
                        # Add to cluster's score
                        cluster_idx = cluster_to_idx[cluster_s[event_idx]]
                        score_matrix[cluster_idx] += score_contrib
        
        # B matrix (outer product of score contributions)
        B = score_matrix.T @ score_matrix
        
        # Sandwich estimator: V_robust = I^{-1} B I^{-1}
        robust_var = I_inv @ B @ I_inv
        
        return robust_var


    


class TimeVaryingCoxPHModel:
    """
    Cox Model with Time-Varying Covariates.
    
    Handles covariates that change value over time.
    
    Data format: Each observation can have multiple rows, one for each
    time interval where covariates remain constant.
    
    Required columns:
    - id: Subject identifier
    - start: Start time of interval
    - stop: Stop time of interval  
    - event: Event indicator (1 only in last interval if event occurs)
    - covariates: Covariate values during this interval
    
    Example:
    --------
    Subject 1 has covariate X that changes at time 5:
    id  start  stop  event  X
    1   0      5     0      0  # X=0 from time 0-5
    1   5      10    1      1  # X=1 from time 5-10, event at 10
    
    References:
    -----------
    - Therneau and Grambsch (2000). Modeling Survival Data, Chapter 8
    - Andersen and Gill (1982). Cox's regression model for counting processes
    """
    
    def __init__(self,
                 tie_method: str = 'efron',
                 alpha: float = 0.05,
                 max_iter: int = 50,
                 tol: float = 1e-6,
                 device: Optional[str] = None):
        """Initialize time-varying Cox model."""
        self.base_model = CoxPHModel(
            tie_method=tie_method,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            device=device
        )
        
        self.tie_method = tie_method
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.device = self.base_model.device
        
        self._is_fitted = False
    
    def fit(self,
            data: pd.DataFrame,
            id_col: str = 'id',
            start_col: str = 'start',
            stop_col: str = 'stop',
            event_col: str = 'event',
            covariate_cols: Optional[list] = None) -> 'TimeVaryingCoxPHModel':
        """
        Fit time-varying Cox model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data in counting process format (start, stop) style
        id_col : str
            Column name for subject ID
        start_col : str
            Column name for interval start time
        stop_col : str
            Column name for interval stop time
        event_col : str
            Column name for event indicator
        covariate_cols : list, optional
            List of covariate column names. If None, uses all numeric columns
            except id, start, stop, event.
            
        Returns:
        --------
        self : TimeVaryingCoxPHModel
        """
        # Validate data
        required_cols = [id_col, start_col, stop_col, event_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Determine covariate columns
        if covariate_cols is None:
            covariate_cols = [col for col in data.select_dtypes(include=[np.number]).columns
                            if col not in required_cols]
        
        if len(covariate_cols) == 0:
            raise ValueError("No covariate columns found")
        
        print(f"Fitting time-varying Cox model with {len(covariate_cols)} covariates")
        print(f"Covariates: {covariate_cols}")
        print(f"Total intervals: {len(data)}")
        print(f"Unique subjects: {data[id_col].nunique()}")
        
        # Store column names
        self.id_col_ = id_col
        self.start_col_ = start_col
        self.stop_col_ = stop_col
        self.event_col_ = event_col
        self.covariate_cols_ = covariate_cols
        
        # Extract data
        X = data[covariate_cols].values
        durations = data[stop_col].values  # Use stop time as duration
        events = data[event_col].values
        ids = data[id_col].values
        starts = data[start_col].values
        
        # Store for risk set computation
        self.data_df_ = data.copy()
        
        # Convert to tensors
        X = self.base_model._to_tensor(X)
        durations = self.base_model._to_tensor(durations)
        events = self.base_model._to_tensor(events)
        ids = self.base_model._to_tensor(ids)
        starts = self.base_model._to_tensor(starts)
        
        n_samples, n_features = X.shape
        
        # Standardize
        self.X_mean_ = torch.mean(X, dim=0)
        self.X_std_ = torch.std(X, dim=0)
        self.X_std_[self.X_std_ == 0] = 1.0
        X_standardized = (X - self.X_mean_) / self.X_std_
        
        # Sort by stop time
        sorted_indices = torch.argsort(durations)
        durations_sorted = durations[sorted_indices]
        events_sorted = events[sorted_indices]
        X_sorted = X_standardized[sorted_indices]
        ids_sorted = ids[sorted_indices]
        starts_sorted = starts[sorted_indices]
        
        # Store sorted data
        self.X_sorted_ = X_sorted
        self.durations_sorted_ = durations_sorted
        self.events_sorted_ = events_sorted
        self.ids_sorted_ = ids_sorted
        self.starts_sorted_ = starts_sorted
        
        # Initialize coefficients
        beta = torch.zeros(n_features, device=self.device, dtype=torch.float64)
        
        # Newton-Raphson optimization
        converged = False
        log_likelihood_history = []
        
        for iteration in range(self.max_iter):
            # Compute derivatives with time-varying risk sets
            log_lik, gradient, hessian = self._compute_time_varying_derivatives(
                beta, X_sorted, durations_sorted, events_sorted, starts_sorted
            )
            
            log_likelihood_history.append(log_lik.item())
            
            # Check convergence
            gradient_norm = torch.norm(gradient)
            if gradient_norm < self.tol:
                converged = True
                break
            
            # Newton-Raphson step
            try:
                delta = torch.linalg.solve(hessian, gradient)
                
                # Line search
                step_size = 1.0
                beta_new = beta - step_size * delta
                log_lik_new, _, _ = self._compute_time_varying_derivatives(
                    beta_new, X_sorted, durations_sorted, events_sorted, starts_sorted
                )
                
                while log_lik_new < log_lik and step_size > 1e-8:
                    step_size *= 0.5
                    beta_new = beta - step_size * delta
                    log_lik_new, _, _ = self._compute_time_varying_derivatives(
                        beta_new, X_sorted, durations_sorted, events_sorted, starts_sorted
                    )
                
                beta = beta_new
                
            except RuntimeError:
                warnings.warn("Hessian is singular, adding regularization")
                regularization = torch.eye(n_features, device=self.device) * 1e-6
                delta = torch.linalg.solve(hessian + regularization, gradient)
                beta = beta - delta
        
        if not converged:
            warnings.warn(f"Optimization did not converge in {self.max_iter} iterations")
        
        # Transform coefficients back
        beta_original = beta / self.X_std_
        
        # Compute final derivatives for standard errors
        final_log_lik, final_gradient, final_hessian = self._compute_time_varying_derivatives(
            beta, X_sorted, durations_sorted, events_sorted, starts_sorted
        )
        
        # Standard errors
        try:
            hessian_inv = torch.linalg.inv(-final_hessian)
            variance_matrix = hessian_inv / (self.X_std_.unsqueeze(1) * self.X_std_.unsqueeze(0))
            standard_errors = torch.sqrt(torch.diag(variance_matrix))
        except RuntimeError:
            warnings.warn("Could not compute standard errors")
            standard_errors = torch.full_like(beta_original, float('nan'))
        
        # Store results
        self.coefficients_ = beta_original.cpu().numpy()
        self.standard_errors_ = standard_errors.cpu().numpy()
        self.log_likelihood_ = final_log_lik.item()
        
        # Note: Baseline hazard estimation for time-varying models is complex
        # Skip for now or use a simplified approach
        
        self._is_fitted = True
        
        # Create result object
        self.result_ = self._create_result_object(converged, iteration + 1)
        
        return self
    
    def _compute_time_varying_derivatives(self,
                                         beta: torch.Tensor,
                                         X: torch.Tensor,
                                         durations: torch.Tensor,
                                         events: torch.Tensor,
                                         starts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute derivatives for time-varying covariates.
        
        The risk set R(t) now includes only intervals where start <= t < stop.
        """
        n_samples, n_features = X.shape
        
        # Compute risk scores
        risk_scores = torch.exp(torch.matmul(X, beta))
        
        # Initialize
        log_likelihood = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        gradient = torch.zeros(n_features, device=self.device, dtype=torch.float64)
        hessian = torch.zeros((n_features, n_features), device=self.device, dtype=torch.float64)
        
        # Find unique event times
        unique_times = torch.unique(durations[events == 1])
        
        for t in unique_times:
            # Risk set: intervals that span time t (start <= t < stop)
            at_risk_mask = (starts <= t) & (durations >= t)
            
            # Event set: events at time t
            event_mask = (durations == t) & (events == 1)
            
            if not torch.any(event_mask):
                continue
            
            d_t = torch.sum(event_mask).item()
            
            # Risk scores and covariates in risk set
            risk_at_risk = risk_scores[at_risk_mask]
            X_at_risk = X[at_risk_mask]
            
            # Covariates for events
            X_events = X[event_mask]
            risk_events = risk_scores[event_mask]
            
            if self.tie_method == 'breslow':
                sum_risk_at_risk = torch.sum(risk_at_risk)
                
                # Log-likelihood
                log_lik_numerator = torch.sum(torch.matmul(X_events, beta))
                log_lik_denominator = d_t * torch.log(sum_risk_at_risk)
                log_likelihood += log_lik_numerator - log_lik_denominator
                
                # Gradient
                weighted_mean_X = torch.sum(X_at_risk * risk_at_risk.unsqueeze(1), dim=0) / sum_risk_at_risk
                gradient += torch.sum(X_events, dim=0) - d_t * weighted_mean_X
                
                # Hessian
                weighted_mean_X_squared = torch.sum(
                    X_at_risk.unsqueeze(2) * X_at_risk.unsqueeze(1) * risk_at_risk.unsqueeze(1).unsqueeze(2),
                    dim=0
                ) / sum_risk_at_risk
                variance_matrix = weighted_mean_X_squared - torch.outer(weighted_mean_X, weighted_mean_X)
                hessian -= d_t * variance_matrix
                
            else:  # efron
                sum_risk_at_risk = torch.sum(risk_at_risk)
                sum_risk_events = torch.sum(risk_events)
                
                # Log-likelihood
                log_lik_numerator = torch.sum(torch.matmul(X_events, beta))
                log_lik_denominator = 0.0
                
                for l in range(d_t):
                    denominator_l = sum_risk_at_risk - (l / d_t) * sum_risk_events
                    log_lik_denominator += torch.log(denominator_l)
                
                log_likelihood += log_lik_numerator - log_lik_denominator
                
                # Gradient
                gradient += torch.sum(X_events, dim=0)
                
                for l in range(d_t):
                    denominator_l = sum_risk_at_risk - (l / d_t) * sum_risk_events
                    
                    weighted_X_risk = torch.sum(X_at_risk * risk_at_risk.unsqueeze(1), dim=0)
                    weighted_X_events = (l / d_t) * torch.sum(X_events * risk_events.unsqueeze(1), dim=0)
                    weighted_mean_X = (weighted_X_risk - weighted_X_events) / denominator_l
                    
                    gradient -= weighted_mean_X
                
                # Hessian
                for l in range(d_t):
                    denominator_l = sum_risk_at_risk - (l / d_t) * sum_risk_events
                    
                    weighted_XX_risk = torch.sum(
                        X_at_risk.unsqueeze(2) * X_at_risk.unsqueeze(1) * risk_at_risk.unsqueeze(1).unsqueeze(2),
                        dim=0
                    )
                    weighted_XX_events = (l / d_t) * torch.sum(
                        X_events.unsqueeze(2) * X_events.unsqueeze(1) * risk_events.unsqueeze(1).unsqueeze(2),
                        dim=0
                    )
                    weighted_mean_XX = (weighted_XX_risk - weighted_XX_events) / denominator_l
                    
                    weighted_X_risk = torch.sum(X_at_risk * risk_at_risk.unsqueeze(1), dim=0)
                    weighted_X_events = (l / d_t) * torch.sum(X_events * risk_events.unsqueeze(1), dim=0)
                    weighted_mean_X = (weighted_X_risk - weighted_X_events) / denominator_l
                    
                    variance_matrix = weighted_mean_XX - torch.outer(weighted_mean_X, weighted_mean_X)
                    hessian -= variance_matrix
        
        return log_likelihood, gradient, hessian
    
    def _create_result_object(self, converged: bool, iterations: int):
        """Create result object."""
        hazard_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / self.standard_errors_
        
        try:
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        except ImportError:
            p_values = np.full_like(z_scores, np.nan)
        
        convergence_info = {
            'converged': converged,
            'iterations': iterations,
            'tolerance': self.tol,
            'tie_method': self.tie_method,
            'time_varying': True
        }
        
        return CoxPHResult(
            coefficients=self.coefficients_,
            standard_errors=self.standard_errors_,
            hazard_ratios=hazard_ratios,
            z_scores=z_scores,
            p_values=p_values,
            concordance_index=0.0,  # Would need to compute separately
            log_likelihood=self.log_likelihood_,
            baseline_hazard=torch.tensor([]),
            baseline_cumulative_hazard=torch.tensor([]),
            baseline_survival=torch.tensor([]),
            timeline=torch.tensor([]),
            convergence_info=convergence_info
        )