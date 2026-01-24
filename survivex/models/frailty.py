"""
Frailty Models for Survival Analysis

Implements shared frailty models for clustered/recurrent survival data using
the EM algorithm. Supports gamma and gaussian (log-normal) frailty distributions.

Mathematical formulation:
    h_i(t) = z_g(i) * h_0(t) * exp(beta' X_i)

where z_g(i) is the frailty for cluster g(i).

Gamma frailty: z ~ Gamma(1/theta, 1/theta), E[z]=1, Var[z]=theta
Gaussian frailty: log(z) ~ N(0, sigma^2), E[z]=exp(sigma^2/2)
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
                 tol: float = 1e-6):
        if distribution not in ['gamma', 'gaussian']:
            raise ValueError("distribution must be 'gamma' or 'gaussian'")

        self.distribution = distribution
        self.tie_method = tie_method
        self.max_iter = max_iter
        self.tol = tol
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

        # Pre-compute cluster masks and event counts
        self.cluster_masks_ = []
        self.cluster_D_ = np.zeros(self.n_clusters_)
        for idx, cluster in enumerate(unique_clusters):
            mask = cluster_id == cluster
            self.cluster_masks_.append(mask)
            self.cluster_D_[idx] = np.sum(events[mask])

        # Pre-compute observation-to-cluster mapping
        self.obs_cluster_idx_ = np.array([self.cluster_map_[c] for c in cluster_id])

        # Pre-compute unique event times and risk sets
        self._precompute_risk_sets()

        # Initialize beta from standard Cox model
        from .cox_ph import CoxPHModel
        cox_init = CoxPHModel(tie_method=self.tie_method)
        cox_init.fit(X, durations, events, start_times=start_times)
        beta = cox_init.coefficients_.copy()

        # Initialize frailty variance
        theta = 0.5  # Initial guess for both gamma and gaussian
        frailties = np.ones(self.n_clusters_)

        # EM algorithm
        log_lik_old = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            # E-step: compute posterior frailty expectations using current frailties
            frailties, posterior_info = self._e_step(beta, theta, frailties)

            # M-step: update beta and theta
            beta = self._m_step_beta(beta, frailties)
            theta = self._m_step_theta(frailties, posterior_info)

            # Compute log-likelihood for convergence check
            log_lik = self._compute_penalized_log_likelihood(beta, frailties, theta)

            if abs(log_lik - log_lik_old) < self.tol * (1 + abs(log_lik)):
                converged = True
                break

            log_lik_old = log_lik

        # Store results
        self.coefficients_ = beta
        self.frailty_variance_ = theta
        self.frailty_values_ = frailties
        self.log_likelihood_ = log_lik

        # Compute standard errors from observed information
        self.variance_covariance_matrix_ = self._compute_variance_matrix(beta, frailties)
        self.standard_errors_ = np.sqrt(np.maximum(np.diag(self.variance_covariance_matrix_), 0))

        self._is_fitted = True
        self.result_ = self._create_result_object(converged, iteration + 1)

        return self

    def _precompute_risk_sets(self):
        """Pre-compute unique event times and at-risk indicators."""
        event_mask = self.events_ == 1
        self.unique_event_times_ = np.sort(np.unique(self.durations_[event_mask]))

        n = len(self.durations_)
        n_times = len(self.unique_event_times_)

        # For each event time, compute at-risk and at-event masks
        self.at_risk_indices_ = []
        self.at_event_indices_ = []
        self.n_events_at_time_ = []

        for t in self.unique_event_times_:
            at_risk = (self.start_times_ < t) & (self.durations_ >= t)
            at_event = (self.durations_ == t) & (self.events_ == 1)
            self.at_risk_indices_.append(np.where(at_risk)[0])
            self.at_event_indices_.append(np.where(at_event)[0])
            self.n_events_at_time_.append(int(np.sum(at_event)))

    def _e_step(self, beta: np.ndarray, theta: float,
                 current_frailties: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        E-step: compute posterior frailty expectations.

        For gamma frailty:
            Posterior is Gamma(shape_i, rate_i) where:
            shape_i = 1/theta + D_i
            rate_i = 1/theta + Lambda_i
            E[z_i|data] = shape_i / rate_i

        For gaussian frailty:
            Posterior mode of log(z_i) found by optimization.
            Posterior variance approximated by curvature at mode.
        """
        # Compute cumulative hazard for each cluster using frailty-weighted baseline
        Lambda = self._compute_cluster_cumulative_hazards(beta, current_frailties)

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
            # log(z) ~ N(0, sigma^2), sigma^2 = theta
            sigma_sq = max(theta, 1e-10)
            log_z_modes = np.zeros(self.n_clusters_)
            post_variances = np.zeros(self.n_clusters_)

            from scipy.optimize import minimize_scalar

            for idx in range(self.n_clusters_):
                D_i = self.cluster_D_[idx]
                Lambda_i = Lambda[idx]

                def neg_log_posterior(log_z):
                    z = np.exp(log_z)
                    # Log-likelihood contribution
                    ll = D_i * log_z - z * Lambda_i
                    # Log-normal prior: log(z) ~ N(0, sigma^2)
                    lp = -log_z**2 / (2 * sigma_sq)
                    return -(ll + lp)

                def neg_log_posterior_deriv2(log_z):
                    """Second derivative of negative log posterior."""
                    z = np.exp(log_z)
                    # d²/d(logz)² of -(D*logz - z*Lambda - logz²/(2σ²))
                    # = z*Lambda + 1/σ²
                    return z * Lambda_i + 1.0 / sigma_sq

                result = minimize_scalar(neg_log_posterior, bounds=(-5, 5), method='bounded')
                log_z_modes[idx] = result.x
                frailties[idx] = np.exp(result.x)
                # Posterior variance from curvature
                post_variances[idx] = 1.0 / neg_log_posterior_deriv2(result.x)

            posterior_info['log_z_modes'] = log_z_modes
            posterior_info['post_variances'] = post_variances

        return frailties, posterior_info

    def _compute_cluster_cumulative_hazards(self, beta: np.ndarray,
                                              frailties: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cumulative hazard Lambda_i for each cluster.

        Uses frailty-weighted baseline hazard if frailties are provided:
        Lambda_i = sum over event times t of [baseline_hazard(t) * sum_{j in cluster_i at risk at t} exp(X_j * beta)]

        where baseline_hazard(t) = d(t) / sum_{j at risk at t} [z_j * exp(X_j * beta)]
        """
        Lambda = np.zeros(self.n_clusters_)
        base_risk_scores = np.exp(self.X_ @ beta)

        # Compute frailty-weighted risk scores
        if frailties is not None:
            obs_frailties = frailties[self.obs_cluster_idx_]
            weighted_risk_scores = obs_frailties * base_risk_scores
        else:
            weighted_risk_scores = base_risk_scores

        for t_idx, t in enumerate(self.unique_event_times_):
            at_risk_idx = self.at_risk_indices_[t_idx]
            at_event_idx = self.at_event_indices_[t_idx]
            n_events_t = self.n_events_at_time_[t_idx]

            if n_events_t == 0 or len(at_risk_idx) == 0:
                continue

            # Breslow baseline hazard with frailty-weighted denominator
            sum_weighted_risk = np.sum(weighted_risk_scores[at_risk_idx])
            if sum_weighted_risk < 1e-300:
                continue
            baseline_hazard_t = n_events_t / sum_weighted_risk

            # Add contribution to each cluster (using base risk scores, not weighted)
            for cluster_idx in range(self.n_clusters_):
                mask = self.cluster_masks_[cluster_idx]
                cluster_at_risk = np.intersect1d(at_risk_idx, np.where(mask)[0])
                if len(cluster_at_risk) > 0:
                    Lambda[cluster_idx] += baseline_hazard_t * np.sum(base_risk_scores[cluster_at_risk])

        return Lambda

    def _m_step_beta(self, beta: np.ndarray, frailties: np.ndarray) -> np.ndarray:
        """M-step: update regression coefficients."""
        obs_frailties = frailties[self.obs_cluster_idx_]

        from scipy.optimize import minimize

        def neg_log_likelihood(beta_vec):
            ll, grad, _ = self._compute_weighted_partial_likelihood(beta_vec, obs_frailties,
                                                                     compute_hessian=False)
            return -ll

        def gradient(beta_vec):
            _, grad, _ = self._compute_weighted_partial_likelihood(beta_vec, obs_frailties,
                                                                    compute_hessian=False)
            return -grad

        result = minimize(
            neg_log_likelihood,
            beta,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 50, 'ftol': 1e-10}
        )

        return result.x

    def _m_step_theta(self, frailties: np.ndarray, posterior_info: dict) -> float:
        """
        M-step: update frailty variance parameter.

        For gamma: uses E[z²|data] to compute unbiased variance estimate.
            theta = (1/K) * sum_i [E[z_i²|data] - 2*E[z_i|data] + 1]
            where E[z²|data] = shape*(shape+1)/rate² for posterior Gamma(shape, rate)

        For gaussian: uses posterior mode and variance.
            sigma² = (1/K) * sum_i [Var_post(log z_i) + (log z_i_mode)²]
        """
        K = self.n_clusters_

        if self.distribution == 'gamma':
            shapes = posterior_info['shapes']
            rates = posterior_info['rates']
            # E[z²|data] = shape*(shape+1)/rate²
            Ez2 = shapes * (shapes + 1) / rates**2
            Ez = shapes / rates
            # Var(z) = E[z²] - (E[z])² marginalizes to theta when model is correct
            # But we want E[(z-1)²] = E[z²] - 2E[z] + 1 as the variance around mean 1
            theta_new = np.mean(Ez2 - 2*Ez + 1)

        else:  # gaussian
            log_z_modes = posterior_info['log_z_modes']
            post_variances = posterior_info['post_variances']
            # sigma² = (1/K) * sum [Var_post(log z) + (E[log z])²]
            theta_new = np.mean(post_variances + log_z_modes**2)

        # Bound theta
        theta_new = max(1e-6, min(theta_new, 50.0))
        return theta_new

    def _compute_weighted_partial_likelihood(
        self,
        beta: np.ndarray,
        weights: np.ndarray,
        compute_hessian: bool = True
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """
        Compute weighted Cox partial likelihood (Breslow).

        The weighted risk score for observation i is: w_i * exp(X_i * beta)
        where w_i is the frailty for observation i's cluster.
        """
        X = self.X_
        n, p = X.shape
        weighted_risk_scores = weights * np.exp(X @ beta)

        log_lik = 0.0
        gradient = np.zeros(p)
        hessian = np.zeros((p, p)) if compute_hessian else None

        for t_idx in range(len(self.unique_event_times_)):
            at_risk_idx = self.at_risk_indices_[t_idx]
            at_event_idx = self.at_event_indices_[t_idx]
            n_events_t = self.n_events_at_time_[t_idx]

            if n_events_t == 0 or len(at_risk_idx) == 0:
                continue

            risk_at_risk = weighted_risk_scores[at_risk_idx]
            X_at_risk = X[at_risk_idx]
            X_events = X[at_event_idx]

            sum_risk = np.sum(risk_at_risk)
            if sum_risk < 1e-300:
                continue

            # Breslow: all tied events share the same denominator
            log_lik += np.sum(X_events @ beta + np.log(weights[at_event_idx] + 1e-300))
            log_lik -= n_events_t * np.log(sum_risk)

            weighted_X = (X_at_risk * risk_at_risk[:, np.newaxis]).sum(axis=0) / sum_risk
            gradient += np.sum(X_events, axis=0) - n_events_t * weighted_X

            if compute_hessian:
                weighted_XX = (X_at_risk[:, :, np.newaxis] * X_at_risk[:, np.newaxis, :] *
                              risk_at_risk[:, np.newaxis, np.newaxis]).sum(axis=0) / sum_risk
                hessian -= n_events_t * (weighted_XX - np.outer(weighted_X, weighted_X))

        return log_lik, gradient, hessian

    def _compute_penalized_log_likelihood(self, beta: np.ndarray, frailties: np.ndarray,
                                           theta: float) -> float:
        """Compute penalized log-likelihood for convergence monitoring."""
        obs_frailties = frailties[self.obs_cluster_idx_]
        ll, _, _ = self._compute_weighted_partial_likelihood(beta, obs_frailties,
                                                              compute_hessian=False)

        # Add frailty penalty
        if self.distribution == 'gamma':
            inv_theta = 1.0 / max(theta, 1e-10)
            for z in frailties:
                if z > 0:
                    ll += (inv_theta - 1) * np.log(z) - inv_theta * z
        else:
            sigma_sq = max(theta, 1e-10)
            for z in frailties:
                if z > 0:
                    log_z = np.log(z)
                    ll += -log_z**2 / (2 * sigma_sq)

        return ll

    def _compute_variance_matrix(self, beta: np.ndarray, frailties: np.ndarray) -> np.ndarray:
        """Compute variance-covariance matrix from observed information."""
        obs_frailties = frailties[self.obs_cluster_idx_]
        _, _, hessian = self._compute_weighted_partial_likelihood(beta, obs_frailties,
                                                                   compute_hessian=True)

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
