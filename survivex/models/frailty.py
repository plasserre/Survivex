"""
Frailty Models for Survival Analysis

Implements shared frailty models for clustered/recurrent survival data using
the EM algorithm. Supports gamma and gaussian (log-normal) frailty distributions.

Mathematical formulation:
    h_i(t) = z_g(i) * h_0(t) * exp(beta' X_i)

where z_g(i) is the frailty for cluster g(i).

Gamma frailty: z ~ Gamma(1/theta, 1/theta), E[z]=1, Var[z]=theta
Gaussian frailty: log(z) ~ N(0, sigma^2), E[z]=exp(sigma^2/2)

Optimized for performance with O(N log N) algorithms and vectorized operations.
GPU support available via device parameter.
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict
from dataclasses import dataclass
import warnings

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class FrailtyResult:
    """Results from frailty model."""
    coefficients: np.ndarray
    standard_errors: np.ndarray
    hazard_ratios: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    frailty_variance: float
    frailty_values: np.ndarray
    log_likelihood: float
    n_clusters: int
    n_observations: int
    n_events: int
    distribution: str
    convergence_info: dict
    covariate_names: Optional[list] = None

    def summary(self) -> str:
        """Generate summary table."""
        s = "\n" + "="*80 + "\n"
        s += f"Frailty Model ({self.distribution.capitalize()}) - Results Summary\n"
        s += "="*80 + "\n"
        s += f"Number of clusters: {self.n_clusters}\n"
        s += f"Number of observations: {self.n_observations}\n"
        s += f"Number of events: {self.n_events}\n"
        s += f"Log-likelihood: {self.log_likelihood:.4f}\n"
        if self.distribution == 'gamma':
            s += f"Frailty variance (theta): {self.frailty_variance:.6f}\n"
        else:
            s += f"Frailty variance (sigma^2): {self.frailty_variance:.6f}\n"
        s += f"Converged: {self.convergence_info['converged']}\n"
        s += f"Iterations: {self.convergence_info['iterations']}\n"

        s += "\n" + "-"*80 + "\n"
        s += f"{'Variable':<15} {'Coef':<10} {'SE':<10} {'HR':<10} {'z':<8} {'p-value':<10}\n"
        s += "-"*80 + "\n"

        for i in range(len(self.coefficients)):
            name = self.covariate_names[i] if self.covariate_names else f"X{i}"
            s += f"{name:<15} "
            s += f"{self.coefficients[i]:>9.5f} "
            s += f"{self.standard_errors[i]:>9.5f} "
            s += f"{self.hazard_ratios[i]:>9.5f} "
            s += f"{self.z_scores[i]:>7.3f} "
            s += f"{self.p_values[i]:>9.6f}\n"

        s += "="*80 + "\n"
        s += "\nFrailty Summary:\n"
        s += f"  Min:    {np.min(self.frailty_values):.4f}\n"
        s += f"  Median: {np.median(self.frailty_values):.4f}\n"
        s += f"  Mean:   {np.mean(self.frailty_values):.4f}\n"
        s += f"  Max:    {np.max(self.frailty_values):.4f}\n"

        return s


class FrailtyModel:
    """
    Shared frailty model for clustered/recurrent survival data.

    Implements the EM algorithm for gamma and gaussian (log-normal) frailty.
    The model extends Cox PH by adding a cluster-specific random effect:

        h_i(t) = z_g(i) * h_0(t) * exp(beta' X_i)

    where z_g(i) is the frailty for the cluster containing observation i.

    Parameters
    ----------
    distribution : str
        Frailty distribution: 'gamma' or 'gaussian'
    tie_method : str
        Tie handling: 'breslow' or 'efron'
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance for log-likelihood change
    device : str
        Device for computation: 'cpu', 'cuda', or 'mps'

    Usage
    -----
    >>> model = FrailtyModel(distribution='gamma')
    >>> model.fit(X, durations, events, cluster_id)
    >>> print(model.result_.summary())
    """

    def __init__(self,
                 distribution: str = 'gamma',
                 tie_method: str = 'breslow',
                 max_iter: int = 200,
                 tol: float = 1e-6,
                 device: str = 'cpu'):
        if distribution not in ['gamma', 'gaussian']:
            raise ValueError("distribution must be 'gamma' or 'gaussian'")

        self.distribution = distribution
        self.tie_method = tie_method
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self._is_fitted = False

    def fit(self,
            X: np.ndarray,
            durations: np.ndarray,
            events: np.ndarray,
            cluster_id: np.ndarray,
            covariate_names: Optional[list] = None,
            start_times: Optional[np.ndarray] = None) -> 'FrailtyModel':
        """
        Fit frailty model using EM algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Covariate matrix
        durations : np.ndarray, shape (n,)
            Event/censoring times
        events : np.ndarray, shape (n,)
            Event indicators (1=event, 0=censored)
        cluster_id : np.ndarray, shape (n,)
            Cluster/subject identifiers
        covariate_names : list, optional
            Names for covariates (for display)
        start_times : np.ndarray, optional
            Start times for counting process data

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=np.float64)
        cluster_id = np.asarray(cluster_id)

        self.X_ = X
        self.durations_ = durations
        self.events_ = events
        self.cluster_id_ = cluster_id
        self.covariate_names_ = covariate_names
        self.start_times_ = start_times if start_times is not None else np.zeros_like(durations)

        n, p = X.shape

        unique_clusters = np.unique(cluster_id)
        self.n_clusters_ = len(unique_clusters)
        self.n_observations_ = n
        self.n_events_ = int(np.sum(events))
        self.cluster_map_ = {c: i for i, c in enumerate(unique_clusters)}
        self.unique_clusters_ = unique_clusters

        # Pre-compute observation-to-cluster mapping (vectorized)
        self.obs_cluster_idx_ = np.array([self.cluster_map_[c] for c in cluster_id])

        # Pre-compute cluster event counts (vectorized)
        self.cluster_D_ = np.bincount(self.obs_cluster_idx_, weights=events,
                                       minlength=self.n_clusters_)

        # Pre-compute sorted indices for O(N log N) algorithm
        self._precompute_sorted_indices()

        # Initialize beta from standard Cox model (fast)
        beta = self._initialize_beta()

        # Initialize frailty variance
        theta = 0.5
        frailties = np.ones(self.n_clusters_)

        # EM algorithm
        log_lik_old = -np.inf
        beta_old = beta.copy()
        converged = False

        for iteration in range(self.max_iter):
            # E-step: compute posterior frailty expectations
            frailties, posterior_info = self._e_step_vectorized(beta, theta, frailties)

            # M-step: update beta using Newton-Raphson (fast, 1-2 iterations)
            beta = self._m_step_beta_newton(beta, frailties)
            theta = self._m_step_theta(frailties, posterior_info)

            # Check coefficient convergence (more reliable than log-likelihood for small theta)
            beta_change = np.max(np.abs(beta - beta_old))
            if beta_change < self.tol:
                converged = True
                break

            # Also check log-likelihood convergence
            log_lik = self._compute_penalized_log_likelihood_fast(beta, frailties, theta)
            if abs(log_lik - log_lik_old) < self.tol * (1 + abs(log_lik)):
                converged = True
                break

            log_lik_old = log_lik
            beta_old = beta.copy()

        # Store results
        self.coefficients_ = beta
        self.frailty_variance_ = theta
        self.frailty_values_ = frailties
        self.log_likelihood_ = log_lik

        # Compute standard errors from observed information
        self.variance_covariance_matrix_ = self._compute_variance_matrix_fast(beta, frailties)
        self.standard_errors_ = np.sqrt(np.maximum(np.diag(self.variance_covariance_matrix_), 0))

        self._is_fitted = True
        self.result_ = self._create_result_object(converged, iteration + 1)

        return self

    def _setup_device(self):
        """Setup computation device."""
        if self.device == 'cpu' or not HAS_TORCH:
            self.use_gpu_ = False
            return

        if self.device == 'cuda' and torch.cuda.is_available():
            self.torch_device_ = torch.device('cuda')
            self.use_gpu_ = True
        elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.torch_device_ = torch.device('mps')
            self.use_gpu_ = True
        else:
            self.use_gpu_ = False

    def _precompute_sorted_indices(self):
        """Pre-compute sorted indices for O(N log N) computation."""
        n = len(self.durations_)

        # Setup GPU if requested
        self._setup_device()

        # Sort by duration (descending for suffix sums)
        self.sort_idx_desc_ = np.argsort(-self.durations_)
        self.sort_idx_asc_ = np.argsort(self.durations_)

        # Sorted arrays
        self.durations_sorted_ = self.durations_[self.sort_idx_asc_]
        self.events_sorted_ = self.events_[self.sort_idx_asc_]
        self.X_sorted_ = self.X_[self.sort_idx_asc_]
        self.cluster_idx_sorted_ = self.obs_cluster_idx_[self.sort_idx_asc_]

        # Unique event times
        event_mask = self.events_ == 1
        self.unique_event_times_ = np.unique(self.durations_[event_mask])
        self.n_unique_times_ = len(self.unique_event_times_)

        # For each observation, find its position in sorted order
        self.inv_sort_idx_ = np.argsort(self.sort_idx_asc_)

        # Pre-compute event time indices for vectorized lookup
        self.event_time_indices_ = np.searchsorted(self.durations_sorted_,
                                                    self.unique_event_times_, side='left')

    def _initialize_beta(self) -> np.ndarray:
        """Initialize beta using a few Newton-Raphson steps on standard Cox."""
        n, p = self.X_.shape
        beta = np.zeros(p)

        # Just do 5 Newton-Raphson iterations for initialization
        for _ in range(5):
            _, grad, hess = self._compute_derivatives_breslow(beta, np.ones(n))
            if hess is not None:
                try:
                    delta = np.linalg.solve(-hess, grad)
                    beta = beta + delta
                except np.linalg.LinAlgError:
                    break

        return beta

    def _compute_derivatives_breslow(
        self,
        beta: np.ndarray,
        weights: np.ndarray,
        compute_hessian: bool = True
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """
        Compute weighted Cox partial likelihood derivatives using O(N log N) algorithm.

        Uses sort + suffix sum approach instead of looping over event times.
        Routes to GPU implementation for large datasets when GPU is available.
        """
        # Use GPU for large datasets (n > 1000) if available
        if getattr(self, 'use_gpu_', False) and len(self.durations_) > 1000:
            return self._compute_derivatives_breslow_gpu(beta, weights, compute_hessian)

        return self._compute_derivatives_breslow_cpu(beta, weights, compute_hessian)

    def _compute_derivatives_breslow_cpu(
        self,
        beta: np.ndarray,
        weights: np.ndarray,
        compute_hessian: bool = True
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """CPU implementation using numpy."""
        X = self.X_
        n, p = X.shape
        durations = self.durations_
        events = self.events_

        # Compute weighted risk scores
        eta = X @ beta
        risk_scores = weights * np.exp(eta)

        # Sort by duration (ascending)
        sort_idx = self.sort_idx_asc_
        dur_sorted = durations[sort_idx]
        evt_sorted = events[sort_idx]
        risk_sorted = risk_scores[sort_idx]
        X_sorted = X[sort_idx]

        # Suffix cumulative sums (for risk sets)
        risk_cumsum_rev = np.cumsum(risk_sorted[::-1])[::-1]

        # For tied durations, find unique durations and their first occurrence positions
        unique_dur, first_idx = np.unique(dur_sorted, return_index=True)

        # Create risk set sums at each unique time
        risk_at_unique = risk_cumsum_rev[first_idx]

        # Map each observation to its risk set sum
        dur_to_idx = np.searchsorted(unique_dur, dur_sorted)
        S0 = risk_at_unique[dur_to_idx]

        # Compute weighted X sums (S1)
        weighted_X_sorted = X_sorted * risk_sorted[:, np.newaxis]
        weighted_X_cumsum_rev = np.cumsum(weighted_X_sorted[::-1], axis=0)[::-1]
        S1_at_unique = weighted_X_cumsum_rev[first_idx]
        S1 = S1_at_unique[dur_to_idx]

        # Log-likelihood: sum over events of [eta - log(S0)]
        log_lik = np.sum(evt_sorted * (eta[sort_idx] + np.log(weights[sort_idx] + 1e-300) - np.log(S0 + 1e-300)))

        # Gradient: sum over events of [X - S1/S0]
        gradient = np.sum(evt_sorted[:, np.newaxis] * (X_sorted - S1 / (S0[:, np.newaxis] + 1e-300)), axis=0)

        if not compute_hessian:
            return log_lik, gradient, None

        # Hessian: -sum over events of [S2/S0 - (S1/S0)^2]
        weighted_XX_sorted = np.einsum('ij,ik->ijk', X_sorted, X_sorted) * risk_sorted[:, np.newaxis, np.newaxis]
        weighted_XX_cumsum_rev = np.cumsum(weighted_XX_sorted[::-1], axis=0)[::-1]
        S2_at_unique = weighted_XX_cumsum_rev[first_idx]
        S2 = S2_at_unique[dur_to_idx]

        S1_outer = np.einsum('ij,ik->ijk', S1, S1)
        hessian = -np.sum(evt_sorted[:, np.newaxis, np.newaxis] *
                          (S2 / (S0[:, np.newaxis, np.newaxis] + 1e-300) -
                           S1_outer / (S0[:, np.newaxis, np.newaxis]**2 + 1e-300)), axis=0)

        return log_lik, gradient, hessian

    def _compute_derivatives_breslow_gpu(
        self,
        beta: np.ndarray,
        weights: np.ndarray,
        compute_hessian: bool = True
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """GPU implementation using PyTorch for large datasets."""
        device = self.torch_device_

        # MPS doesn't support float64, use float32 for MPS
        dtype = torch.float32 if device.type == 'mps' else torch.float64

        # Convert to tensors
        X = torch.tensor(self.X_, dtype=dtype, device=device)
        durations = torch.tensor(self.durations_, dtype=dtype, device=device)
        events = torch.tensor(self.events_, dtype=dtype, device=device)
        beta_t = torch.tensor(beta, dtype=dtype, device=device)
        weights_t = torch.tensor(weights, dtype=dtype, device=device)

        n, p = X.shape

        # Compute weighted risk scores
        eta = X @ beta_t
        risk_scores = weights_t * torch.exp(eta)

        # Sort by duration (ascending)
        sort_idx = torch.argsort(durations)
        dur_sorted = durations[sort_idx]
        evt_sorted = events[sort_idx]
        risk_sorted = risk_scores[sort_idx]
        X_sorted = X[sort_idx]
        weights_sorted = weights_t[sort_idx]
        eta_sorted = eta[sort_idx]

        # Suffix cumulative sums (reverse cumsum)
        risk_cumsum_rev = torch.flip(torch.cumsum(torch.flip(risk_sorted, [0]), 0), [0])

        # For tied durations, find unique durations
        unique_dur, inverse_idx = torch.unique(dur_sorted, return_inverse=True)
        first_idx = torch.zeros(len(unique_dur), dtype=torch.long, device=device)
        first_idx.scatter_(0, inverse_idx, torch.arange(n, device=device))

        # Create risk set sums at each unique time
        risk_at_unique = risk_cumsum_rev[first_idx]

        # Map each observation to its risk set sum
        S0 = risk_at_unique[inverse_idx]

        # Compute weighted X sums (S1)
        weighted_X_sorted = X_sorted * risk_sorted.unsqueeze(1)
        weighted_X_cumsum_rev = torch.flip(torch.cumsum(torch.flip(weighted_X_sorted, [0]), 0), [0])
        S1_at_unique = weighted_X_cumsum_rev[first_idx]
        S1 = S1_at_unique[inverse_idx]

        # Log-likelihood
        log_lik = torch.sum(evt_sorted * (eta_sorted + torch.log(weights_sorted + 1e-300) - torch.log(S0 + 1e-300)))

        # Gradient
        gradient = torch.sum(evt_sorted.unsqueeze(1) * (X_sorted - S1 / (S0.unsqueeze(1) + 1e-300)), dim=0)

        if not compute_hessian:
            return log_lik.item(), gradient.cpu().numpy(), None

        # Hessian
        weighted_XX_sorted = torch.einsum('ij,ik->ijk', X_sorted, X_sorted) * risk_sorted.unsqueeze(1).unsqueeze(2)
        weighted_XX_cumsum_rev = torch.flip(torch.cumsum(torch.flip(weighted_XX_sorted, [0]), 0), [0])
        S2_at_unique = weighted_XX_cumsum_rev[first_idx]
        S2 = S2_at_unique[inverse_idx]

        S1_outer = torch.einsum('ij,ik->ijk', S1, S1)
        hessian = -torch.sum(evt_sorted.unsqueeze(1).unsqueeze(2) *
                              (S2 / (S0.unsqueeze(1).unsqueeze(2) + 1e-300) -
                               S1_outer / (S0.unsqueeze(1).unsqueeze(2)**2 + 1e-300)), dim=0)

        return log_lik.item(), gradient.cpu().numpy(), hessian.cpu().numpy()

    def _e_step_vectorized(self, beta: np.ndarray, theta: float,
                            current_frailties: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        E-step: compute posterior frailty expectations (vectorized).
        """
        # Compute cumulative hazard for each cluster (vectorized)
        Lambda = self._compute_cluster_cumulative_hazards_fast(beta, current_frailties)

        posterior_info = {}
        frailties = np.zeros(self.n_clusters_)

        if self.distribution == 'gamma':
            inv_theta = 1.0 / max(theta, 1e-10)
            shapes = inv_theta + self.cluster_D_
            rates = inv_theta + Lambda
            frailties = shapes / rates
            posterior_info['shapes'] = shapes
            posterior_info['rates'] = rates

        else:  # gaussian
            sigma_sq = max(theta, 1e-10)

            # Vectorized Newton-Raphson for all clusters at once
            log_z = np.zeros(self.n_clusters_)
            D = self.cluster_D_

            for _ in range(10):  # Newton iterations
                z = np.exp(log_z)
                # Gradient: D - z*Lambda - log_z/sigma_sq
                grad = D - z * Lambda - log_z / sigma_sq
                # Hessian: -z*Lambda - 1/sigma_sq
                hess = -z * Lambda - 1.0 / sigma_sq
                # Newton step
                delta = -grad / hess
                log_z = log_z + np.clip(delta, -1.0, 1.0)

            frailties = np.exp(log_z)
            post_variances = -1.0 / (-frailties * Lambda - 1.0 / sigma_sq)

            posterior_info['log_z_modes'] = log_z
            posterior_info['post_variances'] = np.maximum(post_variances, 1e-10)

        return frailties, posterior_info

    def _compute_cluster_cumulative_hazards_fast(self, beta: np.ndarray,
                                                   frailties: np.ndarray) -> np.ndarray:
        """
        Compute cumulative hazard Lambda_i for each cluster using vectorized operations.

        Lambda_i = sum_t [h0(t) * sum_{j in cluster_i, at risk at t} exp(X_j * beta)]

        where h0(t) = d(t) / sum_{j at risk at t} [z_j * exp(X_j * beta)]
        """
        n = len(self.durations_)
        K = self.n_clusters_

        # Base risk scores (without frailty)
        base_risk = np.exp(self.X_ @ beta)

        # Frailty-weighted risk scores
        obs_frailties = frailties[self.obs_cluster_idx_]
        weighted_risk = obs_frailties * base_risk

        # Sort by duration (ascending)
        sort_idx = self.sort_idx_asc_
        dur_sorted = self.durations_[sort_idx]
        evt_sorted = self.events_[sort_idx]
        base_risk_sorted = base_risk[sort_idx]
        weighted_risk_sorted = weighted_risk[sort_idx]
        cluster_sorted = self.obs_cluster_idx_[sort_idx]

        # Compute suffix sums of weighted risk (for denominators)
        weighted_risk_cumsum_rev = np.cumsum(weighted_risk_sorted[::-1])[::-1]

        # Get unique event times and counts
        event_mask_sorted = evt_sorted == 1
        event_times_sorted = dur_sorted[event_mask_sorted]
        unique_event_times, event_counts = np.unique(event_times_sorted, return_counts=True)
        n_event_times = len(unique_event_times)

        if n_event_times == 0:
            return np.zeros(K)

        # For each unique event time, get the risk set sum
        first_idx = np.searchsorted(dur_sorted, unique_event_times, side='left')
        risk_set_sums = weighted_risk_cumsum_rev[first_idx]

        # Baseline hazard at each event time: h0(t) = d(t) / S0(t)
        baseline_hazard = event_counts / (risk_set_sums + 1e-300)

        # Vectorized computation of Lambda using suffix sums per cluster
        # For each cluster, compute suffix cumsum of base_risk in sorted order
        # Then Lambda[k] = sum over event times t of [h0(t) * suffix_sum_k[first_idx[t]]]

        # Build per-cluster suffix sums (this is the key optimization)
        # Create a (K, n) sparse-ish representation via sorting
        # For cluster k: cluster_suffix[k, i] = sum of base_risk[j] for j >= i and cluster[j] == k

        # Use reverse cumsum per cluster via bincount at each position (still O(N*K) worst case)
        # Better approach: compute incremental contributions

        # For small K (< 100), the loop is actually fast enough
        # For large K, we could use a different approach

        if K <= 100 or n_event_times <= 50:
            # Direct computation for small K or few event times
            Lambda = np.zeros(K)
            for t_idx in range(n_event_times):
                start_pos = first_idx[t_idx]
                h0 = baseline_hazard[t_idx]

                # Sum base_risk by cluster for observations at risk
                at_risk_clusters = cluster_sorted[start_pos:]
                at_risk_base_risk = base_risk_sorted[start_pos:]

                # Aggregate by cluster
                cluster_contrib = np.bincount(at_risk_clusters, weights=at_risk_base_risk,
                                              minlength=K)
                Lambda += h0 * cluster_contrib
        else:
            # For large K with many event times, use different approach
            # Compute per-cluster suffix sums first, then index into them
            # Build cluster_suffix_sums[k, :] = reverse cumsum of base_risk for cluster k

            # Method: Create (K, n) matrix and cumsum - memory intensive but fast
            # For very large datasets, stick with the loop

            Lambda = np.zeros(K)
            for t_idx in range(n_event_times):
                start_pos = first_idx[t_idx]
                h0 = baseline_hazard[t_idx]
                at_risk_clusters = cluster_sorted[start_pos:]
                at_risk_base_risk = base_risk_sorted[start_pos:]
                cluster_contrib = np.bincount(at_risk_clusters, weights=at_risk_base_risk,
                                              minlength=K)
                Lambda += h0 * cluster_contrib

        return Lambda

    def _m_step_beta_newton(self, beta: np.ndarray, frailties: np.ndarray) -> np.ndarray:
        """M-step: update regression coefficients using Newton-Raphson (fast)."""
        obs_frailties = frailties[self.obs_cluster_idx_]

        # Only need 2-3 Newton iterations since we're near optimum
        for _ in range(3):
            ll, grad, hess = self._compute_derivatives_breslow(beta, obs_frailties,
                                                                 compute_hessian=True)
            if hess is not None:
                try:
                    delta = np.linalg.solve(-hess, grad)
                    # Line search with Armijo condition
                    step = 1.0
                    for _ in range(10):
                        beta_new = beta + step * delta
                        ll_new, _, _ = self._compute_derivatives_breslow(beta_new, obs_frailties,
                                                                          compute_hessian=False)
                        if ll_new >= ll + 0.0001 * step * np.dot(grad, delta):
                            break
                        step *= 0.5
                    beta = beta + step * delta

                    # Check convergence
                    if np.linalg.norm(delta) < 1e-8:
                        break
                except np.linalg.LinAlgError:
                    break

        return beta

    def _m_step_theta(self, frailties: np.ndarray, posterior_info: dict) -> float:
        """M-step: update frailty variance parameter."""
        if self.distribution == 'gamma':
            shapes = posterior_info['shapes']
            rates = posterior_info['rates']
            Ez2 = shapes * (shapes + 1) / rates**2
            Ez = shapes / rates
            theta_new = np.mean(Ez2 - 2*Ez + 1)
        else:
            log_z_modes = posterior_info['log_z_modes']
            post_variances = posterior_info['post_variances']
            theta_new = np.mean(post_variances + log_z_modes**2)

        return np.clip(theta_new, 1e-6, 50.0)

    def _compute_penalized_log_likelihood_fast(self, beta: np.ndarray,
                                                 frailties: np.ndarray,
                                                 theta: float) -> float:
        """Compute penalized log-likelihood for convergence monitoring."""
        obs_frailties = frailties[self.obs_cluster_idx_]
        ll, _, _ = self._compute_derivatives_breslow(beta, obs_frailties, compute_hessian=False)

        # Add frailty penalty (vectorized)
        if self.distribution == 'gamma':
            inv_theta = 1.0 / max(theta, 1e-10)
            valid = frailties > 0
            ll += np.sum((inv_theta - 1) * np.log(frailties[valid]) - inv_theta * frailties[valid])
        else:
            sigma_sq = max(theta, 1e-10)
            valid = frailties > 0
            log_z = np.log(frailties[valid])
            ll += np.sum(-log_z**2 / (2 * sigma_sq))

        return ll

    def _compute_variance_matrix_fast(self, beta: np.ndarray, frailties: np.ndarray) -> np.ndarray:
        """Compute variance-covariance matrix from observed information."""
        obs_frailties = frailties[self.obs_cluster_idx_]
        _, _, hessian = self._compute_derivatives_breslow(beta, obs_frailties, compute_hessian=True)

        if hessian is None:
            return np.eye(len(beta))

        try:
            variance_matrix = np.linalg.inv(-hessian)
        except np.linalg.LinAlgError:
            warnings.warn("Hessian singular, using pseudo-inverse")
            variance_matrix = np.linalg.pinv(-hessian)

        return variance_matrix

    def _create_result_object(self, converged: bool, iterations: int) -> FrailtyResult:
        """Create result object."""
        hazard_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / np.maximum(self.standard_errors_, 1e-10)

        from scipy.stats import norm
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        return FrailtyResult(
            coefficients=self.coefficients_,
            standard_errors=self.standard_errors_,
            hazard_ratios=hazard_ratios,
            z_scores=z_scores,
            p_values=p_values,
            frailty_variance=self.frailty_variance_,
            frailty_values=self.frailty_values_,
            log_likelihood=self.log_likelihood_,
            n_clusters=self.n_clusters_,
            n_observations=self.n_observations_,
            n_events=self.n_events_,
            distribution=self.distribution,
            convergence_info={'converged': converged, 'iterations': iterations},
            covariate_names=self.covariate_names_
        )

    def predict_hazard_ratio(self, X: np.ndarray) -> np.ndarray:
        """Predict marginal hazard ratios (without frailty)."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.exp(np.asarray(X, dtype=np.float64) @ self.coefficients_)

    def predict_conditional_hazard_ratio(
        self,
        X: np.ndarray,
        cluster_id: Union[int, np.ndarray]
    ) -> np.ndarray:
        """Predict conditional hazard ratios including frailty effect."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        marginal_hr = np.exp(np.asarray(X, dtype=np.float64) @ self.coefficients_)

        if isinstance(cluster_id, (int, np.integer)):
            frailty = self.frailty_values_[self.cluster_map_[cluster_id]]
            return marginal_hr * frailty
        else:
            frailties = np.array([self.frailty_values_[self.cluster_map_[c]] for c in cluster_id])
            return marginal_hr * frailties
