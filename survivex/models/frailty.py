"""
Frailty Models for Survival Analysis

FIXED VERSION - Compatible with survivex.models structure
Save this as: survivex/models/frailty.py
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict
from dataclasses import dataclass
import warnings


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
    
    def summary(self) -> str:
        """Generate summary table."""
        s = "\n" + "="*80 + "\n"
        s += f"Frailty Model ({self.distribution.capitalize()}) - Results Summary\n"
        s += "="*80 + "\n"
        s += f"Number of clusters: {self.n_clusters}\n"
        s += f"Number of observations: {self.n_observations}\n"
        s += f"Number of events: {self.n_events}\n"
        s += f"Log-likelihood: {self.log_likelihood:.4f}\n"
        s += f"Frailty variance (θ or σ²): {self.frailty_variance:.6f}\n"
        s += f"Converged: {self.convergence_info['converged']}\n"
        s += f"Iterations: {self.convergence_info['iterations']}\n"
        
        s += "\n" + "-"*80 + "\n"
        s += f"{'Variable':<12} {'Coef':<10} {'SE':<10} {'HR':<10} {'z':<8} {'p-value':<10}\n"
        s += "-"*80 + "\n"
        
        for i in range(len(self.coefficients)):
            s += f"{'X' + str(i):<12} "
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
    Frailty model for clustered/recurrent survival data.
    
    Usage:
    ------
    from survivex.models.frailty import FrailtyModel
    
    # Gamma frailty
    model = FrailtyModel(distribution='gamma')
    model.fit(X, durations, events, cluster_id)
    
    # Gaussian frailty
    model = FrailtyModel(distribution='gaussian')
    model.fit(X, durations, events, cluster_id)
    """
    
    def __init__(self,
                 distribution: str = 'gamma',
                 tie_method: str = 'efron',
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 alpha: float = 0.05):
        """Initialize frailty model."""
        if distribution not in ['gamma', 'gaussian']:
            raise ValueError("distribution must be 'gamma' or 'gaussian'")
        
        self.distribution = distribution
        self.tie_method = tie_method
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self._is_fitted = False
        
    def fit(self,
            X: np.ndarray,
            durations: np.ndarray,
            events: np.ndarray,
            cluster_id: np.ndarray,
            start_times: Optional[np.ndarray] = None) -> 'FrailtyModel':
        """
        Fit frailty model using EM algorithm.
        
        Parameters:
        -----------
        X : np.ndarray
            Covariate matrix
        durations : np.ndarray
            Event/censoring times
        events : np.ndarray
            Event indicators
        cluster_id : np.ndarray
            Cluster/subject identifiers
        start_times : np.ndarray, optional
            Start times for counting process
        
        Returns:
        --------
        self : FrailtyModel
        """
        # Store data
        self.X_ = X
        self.durations_ = durations
        self.events_ = events
        self.cluster_id_ = cluster_id
        self.start_times_ = start_times if start_times is not None else np.zeros_like(durations)
        
        n, p = X.shape
        
        # Initialize
        unique_clusters = np.unique(cluster_id)
        self.n_clusters_ = len(unique_clusters)
        self.n_observations_ = n
        self.n_events_ = np.sum(events)
        self.cluster_map_ = {c: i for i, c in enumerate(unique_clusters)}
        
        # Initialize parameters
        if self.distribution == 'gamma':
            theta = 0.5  # Frailty variance
        else:
            sigma_sq = 0.5  # Log-frailty variance
        
        frailties = np.ones(self.n_clusters_)
        
        # Initialize with standard Cox model
        from .cox_ph import CoxPHModel
        cox_init = CoxPHModel(tie_method=self.tie_method)
        cox_init.fit(X, durations, events, start_times=start_times)
        beta = cox_init.coefficients_
        
        # EM algorithm
        log_lik_old = -np.inf
        converged = False
        
        for iteration in range(self.max_iter):
            # E-step
            frailties = self._e_step(beta, frailties, 
                                     theta if self.distribution == 'gamma' else sigma_sq)
            
            # M-step
            beta, var_param = self._m_step(beta, frailties)
            
            if self.distribution == 'gamma':
                theta = var_param
            else:
                sigma_sq = var_param
            
            # Check convergence
            log_lik = self._compute_log_likelihood(
                beta, frailties, 
                theta if self.distribution == 'gamma' else sigma_sq
            )
            
            if np.abs(log_lik - log_lik_old) < self.tol:
                converged = True
                break
            
            log_lik_old = log_lik
        
        # Store results
        self.coefficients_ = beta
        self.frailty_variance_ = theta if self.distribution == 'gamma' else sigma_sq
        self.frailty_values_ = frailties
        self.log_likelihood_ = log_lik
        
        # Compute standard errors
        self.variance_covariance_matrix_ = self._compute_variance_matrix(beta, frailties)
        self.standard_errors_ = np.sqrt(np.diag(self.variance_covariance_matrix_))
        
        self._is_fitted = True
        self.result_ = self._create_result_object(converged, iteration + 1)
        
        return self
    
    def _e_step(self, beta: np.ndarray, frailties: np.ndarray, var_param: float) -> np.ndarray:
        """E-step: Compute expected frailties."""
        new_frailties = np.zeros(self.n_clusters_)
        
        for cluster_idx, cluster in enumerate(np.unique(self.cluster_id_)):
            mask = self.cluster_id_ == cluster
            D_i = np.sum(self.events_[mask])
            Lambda_i = self._compute_cumulative_hazard_cluster(beta, mask)
            
            if self.distribution == 'gamma':
                shape = 1.0 / var_param + D_i
                rate = 1.0 / var_param + Lambda_i
                new_frailties[cluster_idx] = shape / rate
            else:  # gaussian
                def neg_log_posterior(log_z):
                    z = np.exp(log_z)
                    ll = D_i * log_z - z * Lambda_i
                    lp = -(log_z + var_param / 2) ** 2 / (2 * var_param)
                    return -(ll + lp)
                
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(neg_log_posterior, bounds=(-5, 5), method='bounded')
                new_frailties[cluster_idx] = np.exp(result.x)
        
        return new_frailties
    
    def _m_step(self, beta: np.ndarray, frailties: np.ndarray) -> Tuple[np.ndarray, float]:
        """M-step: Update parameters."""
        obs_frailties = np.array([frailties[self.cluster_map_[c]] for c in self.cluster_id_])
        beta_new = self._update_beta_weighted(beta, obs_frailties)
        
        if self.distribution == 'gamma':
            theta_new = np.var(frailties)
            theta_new = max(1e-6, min(theta_new, 10.0))
        else:
            log_frailties = np.log(frailties)
            sigma_sq_new = np.var(log_frailties)
            sigma_sq_new = max(1e-6, min(sigma_sq_new, 10.0))
            theta_new = sigma_sq_new
        
        return beta_new, theta_new
    
    def _update_beta_weighted(self, beta: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Update beta using weighted partial likelihood."""
        from scipy.optimize import minimize
        
        def neg_log_likelihood(beta_vec):
            ll, _, _ = self._compute_weighted_partial_likelihood(beta_vec, weights)
            return -ll
        
        def gradient(beta_vec):
            _, grad, _ = self._compute_weighted_partial_likelihood(beta_vec, weights)
            return -grad
        
        def hessian(beta_vec):
            _, _, hess = self._compute_weighted_partial_likelihood(beta_vec, weights)
            return -hess
        
        result = minimize(
            neg_log_likelihood,
            beta,
            method='Newton-CG',
            jac=gradient,
            hess=hessian,
            options={'maxiter': 25}
        )
        
        return result.x
    
    def _compute_weighted_partial_likelihood(
        self,
        beta: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute weighted partial likelihood."""
        X = self.X_
        durations = self.durations_
        events = self.events_
        start_times = self.start_times_
        
        n, p = X.shape
        risk_scores = weights * np.exp(X @ beta)
        
        log_lik = 0.0
        gradient = np.zeros(p)
        hessian = np.zeros((p, p))
        
        unique_times = np.unique(durations[events == 1])
        
        for t in unique_times:
            at_risk = (start_times < t) & (durations >= t)
            at_event = (durations == t) & (events == 1)
            
            if not np.any(at_event):
                continue
            
            n_events = np.sum(at_event)
            risk_at_risk = risk_scores[at_risk]
            X_at_risk = X[at_risk]
            X_events = X[at_event]
            risk_events = risk_scores[at_event]
            
            if self.tie_method == 'efron' and n_events > 1:
                sum_risk_at_risk = np.sum(risk_at_risk)
                sum_risk_events = np.sum(risk_events)
                
                log_lik += np.sum(X_events @ beta)
                
                for l in range(n_events):
                    frac = l / n_events
                    denom = sum_risk_at_risk - frac * sum_risk_events
                    log_lik -= np.log(denom)
                    
                    weighted_X_risk = np.sum(X_at_risk * risk_at_risk[:, np.newaxis], axis=0)
                    weighted_X_events = frac * np.sum(X_events * risk_events[:, np.newaxis], axis=0)
                    weighted_mean_X = (weighted_X_risk - weighted_X_events) / denom
                    
                    gradient -= weighted_mean_X
                    
                    weighted_XX = np.sum(
                        X_at_risk[:, :, np.newaxis] * X_at_risk[:, np.newaxis, :] * 
                        risk_at_risk[:, np.newaxis, np.newaxis], axis=0
                    )
                    weighted_XX_events = frac * np.sum(
                        X_events[:, :, np.newaxis] * X_events[:, np.newaxis, :] * 
                        risk_events[:, np.newaxis, np.newaxis], axis=0
                    )
                    weighted_mean_XX = (weighted_XX - weighted_XX_events) / denom
                    
                    variance = weighted_mean_XX - np.outer(weighted_mean_X, weighted_mean_X)
                    hessian -= variance
                
                gradient += np.sum(X_events, axis=0)
            else:
                sum_risk_at_risk = np.sum(risk_at_risk)
                
                log_lik += np.sum(X_events @ beta) - n_events * np.log(sum_risk_at_risk)
                
                weighted_mean_X = np.sum(X_at_risk * risk_at_risk[:, np.newaxis], axis=0) / sum_risk_at_risk
                gradient += np.sum(X_events, axis=0) - n_events * weighted_mean_X
                
                weighted_XX = np.sum(
                    X_at_risk[:, :, np.newaxis] * X_at_risk[:, np.newaxis, :] * 
                    risk_at_risk[:, np.newaxis, np.newaxis], axis=0
                ) / sum_risk_at_risk
                variance = weighted_XX - np.outer(weighted_mean_X, weighted_mean_X)
                hessian -= n_events * variance
        
        return log_lik, gradient, hessian
    
    def _compute_cumulative_hazard_cluster(self, beta: np.ndarray, mask: np.ndarray) -> float:
        """Compute cumulative hazard for a cluster."""
        X_cluster = self.X_[mask]
        durations_cluster = self.durations_[mask]
        start_times_cluster = self.start_times_[mask]
        
        cumulative_hazard = 0.0
        all_event_times = np.unique(self.durations_[self.events_ == 1])
        
        for t in all_event_times:
            at_risk_in_cluster = (start_times_cluster < t) & (durations_cluster >= t)
            if not np.any(at_risk_in_cluster):
                continue
            
            at_risk_full = (self.start_times_ < t) & (self.durations_ >= t)
            at_event_full = (self.durations_ == t) & (self.events_ == 1)
            
            if not np.any(at_event_full):
                continue
            
            risk_full = np.exp(self.X_[at_risk_full] @ beta)
            sum_risk_full = np.sum(risk_full)
            
            n_events_t = np.sum(at_event_full)
            baseline_hazard_t = n_events_t / sum_risk_full
            
            risk_cluster_at_t = np.exp(X_cluster[at_risk_in_cluster] @ beta)
            cumulative_hazard += baseline_hazard_t * np.sum(risk_cluster_at_t)
        
        return cumulative_hazard
    
    def _compute_log_likelihood(self, beta: np.ndarray, frailties: np.ndarray, var_param: float) -> float:
        """Compute full log-likelihood."""
        obs_frailties = np.array([frailties[self.cluster_map_[c]] for c in self.cluster_id_])
        ll, _, _ = self._compute_weighted_partial_likelihood(beta, obs_frailties)
        
        if self.distribution == 'gamma':
            for z in frailties:
                ll += (1.0 / var_param - 1) * np.log(z) - z / var_param
        else:
            for z in frailties:
                log_z = np.log(z)
                ll += -log_z - (log_z + var_param / 2) ** 2 / (2 * var_param)
        
        return ll
    
    def _compute_variance_matrix(self, beta: np.ndarray, frailties: np.ndarray) -> np.ndarray:
        """Compute variance-covariance matrix."""
        obs_frailties = np.array([frailties[self.cluster_map_[c]] for c in self.cluster_id_])
        _, _, hessian = self._compute_weighted_partial_likelihood(beta, obs_frailties)
        
        try:
            variance_matrix = np.linalg.inv(-hessian)
        except np.linalg.LinAlgError:
            warnings.warn("Hessian singular, using pseudo-inverse")
            variance_matrix = np.linalg.pinv(-hessian)
        
        return variance_matrix
    
    def _create_result_object(self, converged: bool, iterations: int) -> FrailtyResult:
        """Create result object."""
        hazard_ratios = np.exp(self.coefficients_)
        z_scores = self.coefficients_ / self.standard_errors_
        
        try:
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        except ImportError:
            p_values = np.full_like(z_scores, np.nan)
        
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
            convergence_info={'converged': converged, 'iterations': iterations}
        )
    
    def predict_hazard_ratio(self, X: np.ndarray) -> np.ndarray:
        """Predict marginal hazard ratios."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.exp(X @ self.coefficients_)
    
    def predict_conditional_hazard_ratio(
        self, 
        X: np.ndarray, 
        cluster_id: Union[int, np.ndarray]
    ) -> np.ndarray:
        """Predict conditional hazard ratios including frailty."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        marginal_hr = np.exp(X @ self.coefficients_)
        
        if isinstance(cluster_id, (int, np.integer)):
            frailty = self.frailty_values_[self.cluster_map_[cluster_id]]
            return marginal_hr * frailty
        else:
            frailties = np.array([self.frailty_values_[self.cluster_map_[c]] for c in cluster_id])
            return marginal_hr * frailties