"""
Parametric Survival Models
Implemented from scratch following lifelines and R survival package
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma, gammainc, digamma
import pandas as pd
from typing import Optional




class WeibullPHFitter:
    """
    Weibull Proportional Hazards Model
    
    Hazard: h(t|X) = (rho/lambda) * (t/lambda)^(rho-1) * exp(beta' X)
    Survival: S(t|X) = exp(-exp(beta' X) * (t/lambda)^rho)
    """
    
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_ = None
        self.coef_ = None
        self.rho_ = None
        self.lambda_ = None
        self.log_likelihood_ = None
        self.variance_matrix_ = None
        
    def _negative_log_likelihood(self, params: np.ndarray, T: np.ndarray, 
                                  E: np.ndarray, X: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        n, p = X.shape
        log_rho = params[0]
        log_lambda = params[1]
        beta = params[2:]
        
        rho = np.exp(log_rho)
        lambda_ = np.exp(log_lambda)
        eta = X @ beta
        
        log_hazard = log_rho - rho * log_lambda + (rho - 1) * np.log(T) + eta
        log_survival = -np.exp(eta) * np.power(T / lambda_, rho)
        log_lik = np.sum(E * log_hazard + log_survival)
        
        if self.penalizer > 0:
            log_lik -= 0.5 * self.penalizer * np.sum(beta ** 2)
        
        return -log_lik
    
    def _gradient(self, params: np.ndarray, T: np.ndarray, 
                  E: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        n, p = X.shape
        log_rho = params[0]
        log_lambda = params[1]
        beta = params[2:]
        
        rho = np.exp(log_rho)
        lambda_ = np.exp(log_lambda)
        eta = X @ beta
        exp_eta = np.exp(eta)
        
        t_over_lambda = T / lambda_
        t_over_lambda_rho = np.power(t_over_lambda, rho)
        log_t_over_lambda = np.log(t_over_lambda)
        
        d_log_rho = np.sum(E * (1 + log_t_over_lambda * rho) -
                          exp_eta * t_over_lambda_rho * log_t_over_lambda * rho)
        d_log_lambda = np.sum(-E * rho + exp_eta * rho * t_over_lambda_rho)
        d_beta = X.T @ (E - exp_eta * t_over_lambda_rho)
        
        if self.penalizer > 0:
            d_beta -= self.penalizer * beta
        
        grad = np.concatenate([[d_log_rho], [d_log_lambda], d_beta])
        return -grad
    
    def fit(self, T: np.ndarray, E: np.ndarray, X: Optional[np.ndarray] = None) -> 'WeibullPHFitter':
        """Fit model"""
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        n = len(T)
        
        if X is None:
            X = np.ones((n, 1))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = np.column_stack([np.ones(n), X])
        
        p = X.shape[1]
        
        # Initialize
        median_time = np.median(T[E == 1]) if np.any(E) else np.median(T)
        initial_params = np.concatenate([
            [0.0],  # log_rho
            [np.log(max(median_time, 1.0))],  # log_lambda
            np.zeros(p)  # beta
        ])
        
        # Optimize
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(T, E, X),
            method='BFGS',
            jac=self._gradient,
            options={'maxiter': 500}
        )
        
        # Extract parameters
        self.params_ = result.x
        self.rho_ = np.exp(result.x[0])
        self.lambda_ = np.exp(result.x[1])
        self.coef_ = result.x[2:]
        self.log_likelihood_ = -result.fun
        
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if isinstance(result.hess_inv, np.ndarray):
                self.variance_matrix_ = result.hess_inv
            else:
                self.variance_matrix_ = np.array(
                    [result.hess_inv @ np.eye(len(result.x))[i] 
                     for i in range(len(result.x))]
                ).T
        
        return self
    
    def predict_survival_function(self, X: Optional[np.ndarray] = None, 
                                   times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict survival function S(t|X)
        
        For baseline (no covariates), uses the effective lambda that combines
        lambda and intercept to match standard Weibull parameterization.
        """
        if times is None:
            times = np.linspace(0.01, self.lambda_ * 3, 100)
        times = np.asarray(times).flatten()
        
        # Check if this is an intercept-only model
        is_intercept_only = len(self.coef_) == 1
        
        if X is None:
            if is_intercept_only:
                # For baseline: use effective lambda = lambda * exp(-beta_0/rho)
                # This gives S(t) = exp(-(t/lambda_eff)^rho)
                lambda_eff = self.lambda_ * np.exp(-self.coef_[0] / self.rho_)
                S = np.exp(-np.power(times / lambda_eff, self.rho_))
                return S
            else:
                # For models with covariates, X=None means X=0 (no covariate effects)
                X = np.zeros((1, len(self.coef_)))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        eta = X @ self.coef_
        exp_eta = np.exp(eta)[:, np.newaxis]
        t_over_lambda_rho = np.power(times / self.lambda_, self.rho_)[np.newaxis, :]
        S = np.exp(-exp_eta * t_over_lambda_rho)
        
        return S.flatten() if S.shape[0] == 1 else S
    
    def predict_median(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict median survival time"""
        # Check if this is an intercept-only model
        is_intercept_only = len(self.coef_) == 1
        
        if X is None:
            if is_intercept_only:
                # For baseline: use effective lambda
                lambda_eff = self.lambda_ * np.exp(-self.coef_[0] / self.rho_)
                return lambda_eff * np.power(np.log(2), 1 / self.rho_)
            else:
                X = np.zeros((1, len(self.coef_)))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        eta = X @ self.coef_
        median = self.lambda_ * np.power(np.log(2) / np.exp(eta), 1 / self.rho_)
        return median.flatten() if len(median) > 1 else median[0]


class WeibullAFTFitter:
    """
    Weibull Accelerated Failure Time Model
    
    Model: log(T) = mu + beta'X + sigma*epsilon
    where epsilon ~ Gumbel(0,1)
    
    Survival: S(t|X) = exp(-(t/exp(mu + beta'X))^(1/sigma))
    Shape: rho = 1/sigma
    """
    
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_ = None
        self.lambda_params_ = None
        self.rho_ = None
        self.log_likelihood_ = None
        self.variance_matrix_ = None
        
    def _negative_log_likelihood(self, params: np.ndarray, T: np.ndarray, 
                                  E: np.ndarray, X: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        n, p = X.shape
        lambda_params = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)
        
        eta = X @ lambda_params
        log_T = np.log(T)
        residual = (log_T - eta) / sigma
        exp_residual = np.exp(residual)
        
        log_pdf = -log_sigma - log_T + residual - exp_residual
        log_survival = -exp_residual
        log_lik = np.sum(E * log_pdf + (1 - E) * log_survival)
        
        if self.penalizer > 0 and p > 1:
            log_lik -= 0.5 * self.penalizer * np.sum(lambda_params[1:] ** 2)
        
        return -log_lik
    
    def _gradient(self, params: np.ndarray, T: np.ndarray, 
                  E: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        n, p = X.shape
        lambda_params = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)
        
        eta = X @ lambda_params
        log_T = np.log(T)
        residual = (log_T - eta) / sigma
        exp_residual = np.exp(residual)
        
        d_lambda = (1.0 / sigma) * X.T @ (exp_residual - E)
        d_log_sigma = -np.sum(E * (1 + residual) - exp_residual * residual)
        
        if self.penalizer > 0 and p > 1:
            d_lambda[1:] -= self.penalizer * lambda_params[1:]
        
        grad = np.concatenate([d_lambda, [d_log_sigma]])
        return -grad
    
    def fit(self, X, T: np.ndarray, E: np.ndarray) -> 'WeibullAFTFitter':
        """Fit Weibull AFT model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features) or None
            Covariate matrix. If None, fits intercept-only model.
        T : array-like, shape (n_samples,)
            Duration/survival times.
        E : array-like, shape (n_samples,)
            Event indicator (1=event, 0=censored).
        """
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        n = len(T)

        if X is None:
            X = np.ones((n, 1))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = np.column_stack([np.ones(n), X])
        
        p = X.shape[1]
        
        # Initialize
        median_time = np.median(T)
        mu_init = np.log(median_time)
        beta_init = np.zeros(p - 1)
        log_sigma_init = np.log(0.7)
        
        initial_params = np.concatenate([[mu_init], beta_init, [log_sigma_init]])
        
        # Try multiple methods
        methods = ['L-BFGS-B', 'BFGS']
        best_result = None
        best_nll = np.inf
        
        for method in methods:
            try:
                if method == 'L-BFGS-B':
                    bounds = [(None, None)] * (p + 1)
                    bounds[-1] = (-5, 2)
                    result = minimize(
                        fun=self._negative_log_likelihood,
                        x0=initial_params,
                        args=(T, E, X),
                        method=method,
                        jac=self._gradient,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                else:
                    result = minimize(
                        fun=self._negative_log_likelihood,
                        x0=initial_params,
                        args=(T, E, X),
                        method=method,
                        jac=self._gradient,
                        options={'maxiter': 1000}
                    )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
                    
                if result.success:
                    break
            except:
                continue
        
        result = best_result
        
        # Extract parameters
        self.params_ = result.x
        self.lambda_params_ = result.x[:-1]
        sigma = np.exp(result.x[-1])
        self.rho_ = 1.0 / sigma
        self.log_likelihood_ = -result.fun
        
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if isinstance(result.hess_inv, np.ndarray):
                self.variance_matrix_ = result.hess_inv
            else:
                try:
                    self.variance_matrix_ = np.array(
                        [result.hess_inv @ np.eye(len(result.x))[i] 
                         for i in range(len(result.x))]
                    ).T
                except:
                    self.variance_matrix_ = np.eye(len(result.x))
        else:
            self.variance_matrix_ = np.eye(len(result.x))
        
        self._n_covariates = p - 1
        return self
    
    def predict_survival_function(self, X: Optional[np.ndarray] = None, 
                                   times: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict survival function S(t|X)"""
        if times is None:
            times = np.linspace(0.01, np.median(np.exp(self.lambda_params_[0])) * 3, 100)
        times = np.asarray(times).flatten()
        times = np.maximum(times, 1e-10)
        
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        eta_expanded = eta[:, np.newaxis]
        times_expanded = times[np.newaxis, :]
        
        S = np.exp(-np.power(times_expanded * np.exp(-eta_expanded), self.rho_))
        return S.flatten() if S.shape[0] == 1 else S
    
    def predict_median(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict median survival time"""
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        median = np.exp(eta) * np.power(np.log(2), 1.0 / self.rho_)
        return median.flatten() if len(median) > 1 else median[0]
    
    def predict_expectation(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict mean survival time"""
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        lambda_ = np.exp(eta)
        mean = lambda_ * gamma(1.0 + 1.0 / self.rho_)
        return mean.flatten() if len(mean) > 1 else mean[0]


class LogNormalAFTFitter:
    """
    Log-Normal Accelerated Failure Time Model
    
    Model: log(T) = mu + beta'X + sigma*epsilon
    where epsilon ~ Normal(0,1)
    
    For log-normal, there's no simple closed form for hazard in PH,
    so AFT is the natural parameterization.
    """
    
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_ = None
        self.lambda_params_ = None
        self.sigma_ = None
        self.log_likelihood_ = None
        self.variance_matrix_ = None
        
    def _negative_log_likelihood(self, params: np.ndarray, T: np.ndarray, 
                                  E: np.ndarray, X: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        from scipy.stats import norm
        
        n, p = X.shape
        lambda_params = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)
        
        eta = X @ lambda_params
        log_T = np.log(T)
        residual = (log_T - eta) / sigma
        
        # PDF: f(t) = (1/(t*sigma*sqrt(2*pi))) * exp(-residual^2/2)
        log_pdf = -np.log(T) - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * residual ** 2
        
        # Survival: S(t) = 1 - Phi(residual) = Phi(-residual)
        log_survival = norm.logsf(residual)  # log(1 - Phi(residual))
        
        log_lik = np.sum(E * log_pdf + (1 - E) * log_survival)
        
        if self.penalizer > 0 and p > 1:
            log_lik -= 0.5 * self.penalizer * np.sum(lambda_params[1:] ** 2)
        
        return -log_lik
    
    def _gradient(self, params: np.ndarray, T: np.ndarray, 
                  E: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        from scipy.stats import norm
        
        n, p = X.shape
        lambda_params = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)
        
        eta = X @ lambda_params
        log_T = np.log(T)
        residual = (log_T - eta) / sigma
        
        # For censored: phi(residual) / Phi(-residual)
        phi = norm.pdf(residual)
        Phi_neg = norm.cdf(-residual)
        hazard_ratio = phi / (Phi_neg + 1e-10)
        
        # Gradient w.r.t. lambda parameters (mu, beta)
        # ∂LL/∂λ = (1/σ) * Σ[X * (δ * z + (1-δ) * phi(z)/Phi(-z))]
        d_lambda = (1.0 / sigma) * X.T @ (E * residual + (1 - E) * hazard_ratio)
        
        # Gradient w.r.t. log(sigma)
        # ∂LL/∂log(σ) = Σ[δ * (-1 + z^2) + (1-δ) * z * phi(z)/Phi(-z)]
        d_log_sigma = np.sum(E * (-1 + residual ** 2) + (1 - E) * residual * hazard_ratio)
        
        if self.penalizer > 0 and p > 1:
            d_lambda[1:] -= self.penalizer * lambda_params[1:]
        
        grad = np.concatenate([d_lambda, [d_log_sigma]])
        return -grad
    
    def fit(self, T: np.ndarray, E: np.ndarray, X: Optional[np.ndarray] = None) -> 'LogNormalAFTFitter':
        """Fit model"""
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        n = len(T)
        
        if X is None:
            X = np.ones((n, 1))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = np.column_stack([np.ones(n), X])
        
        p = X.shape[1]
        
        # Initialize
        log_T_events = np.log(T[E == 1]) if np.any(E) else np.log(T)
        mu_init = np.mean(log_T_events)
        beta_init = np.zeros(p - 1)
        sigma_init = np.std(log_T_events)
        log_sigma_init = np.log(max(sigma_init, 0.1))
        
        initial_params = np.concatenate([[mu_init], beta_init, [log_sigma_init]])
        
        # Optimize
        methods = ['L-BFGS-B', 'BFGS']
        best_result = None
        best_nll = np.inf
        
        for method in methods:
            try:
                if method == 'L-BFGS-B':
                    bounds = [(None, None)] * (p + 1)
                    bounds[-1] = (-5, 2)
                    result = minimize(
                        fun=self._negative_log_likelihood,
                        x0=initial_params,
                        args=(T, E, X),
                        method=method,
                        jac=self._gradient,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                else:
                    result = minimize(
                        fun=self._negative_log_likelihood,
                        x0=initial_params,
                        args=(T, E, X),
                        method=method,
                        jac=self._gradient,
                        options={'maxiter': 1000}
                    )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
                    
                if result.success:
                    break
            except:
                continue
        
        result = best_result
        
        # Extract parameters
        self.params_ = result.x
        self.lambda_params_ = result.x[:-1]
        self.sigma_ = np.exp(result.x[-1])
        self.log_likelihood_ = -result.fun
        
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if isinstance(result.hess_inv, np.ndarray):
                self.variance_matrix_ = result.hess_inv
            else:
                try:
                    self.variance_matrix_ = np.array(
                        [result.hess_inv @ np.eye(len(result.x))[i] 
                         for i in range(len(result.x))]
                    ).T
                except:
                    self.variance_matrix_ = np.eye(len(result.x))
        else:
            self.variance_matrix_ = np.eye(len(result.x))
        
        self._n_covariates = p - 1
        return self
    
    def predict_survival_function(self, X: Optional[np.ndarray] = None, 
                                   times: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict survival function S(t|X)"""
        from scipy.stats import norm
        
        if times is None:
            times = np.linspace(0.01, np.median(np.exp(self.lambda_params_[0])) * 3, 100)
        times = np.asarray(times).flatten()
        times = np.maximum(times, 1e-10)
        
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        
        # S(t|X) = Phi((mu + beta'X - log(t)) / sigma) = Phi(-(log(t) - mu - beta'X) / sigma)
        eta_expanded = eta[:, np.newaxis]
        log_times = np.log(times)[np.newaxis, :]
        residual = (log_times - eta_expanded) / self.sigma_
        
        S = norm.sf(residual)  # 1 - Phi(residual) = Phi(-residual)
        return S.flatten() if S.shape[0] == 1 else S
    
    def predict_median(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict median survival time"""
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        
        # For log-normal, median = exp(mu + beta'X)
        median = np.exp(eta)
        return median.flatten() if len(median) > 1 else median[0]
    
    def predict_expectation(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict mean survival time"""
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        
        # For log-normal, mean = exp(mu + beta'X + sigma^2/2)
        mean = np.exp(eta + 0.5 * self.sigma_ ** 2)
        return mean.flatten() if len(mean) > 1 else mean[0]


class LogLogisticAFTFitter:
    """
    Log-Logistic Accelerated Failure Time Model
    
    Model: log(T) = mu + beta'X + sigma*epsilon
    where epsilon follows a standard logistic distribution
    
    Survival: S(t|X) = 1 / (1 + (t/exp(mu+beta'X))^(1/sigma))
    """
    
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_ = None
        self.lambda_params_ = None
        self.alpha_ = None  # shape parameter (alpha = 1/sigma)
        self.log_likelihood_ = None
        self.variance_matrix_ = None
        
    def _negative_log_likelihood(self, params: np.ndarray, T: np.ndarray, 
                                  E: np.ndarray, X: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        n, p = X.shape
        lambda_params = params[:-1]
        log_alpha = params[-1]  # log(1/sigma)
        alpha = np.exp(log_alpha)
        
        eta = X @ lambda_params
        log_T = np.log(T)
        z = alpha * (log_T - eta)
        exp_z = np.exp(z)
        
        # PDF: f(t) = (alpha/t) * exp(z) / (1 + exp(z))^2
        log_pdf = log_alpha - log_T + z - 2 * np.log(1 + exp_z)
        
        # Survival: S(t) = 1 / (1 + exp(z))
        log_survival = -np.log(1 + exp_z)
        
        log_lik = np.sum(E * log_pdf + (1 - E) * log_survival)
        
        if self.penalizer > 0 and p > 1:
            log_lik -= 0.5 * self.penalizer * np.sum(lambda_params[1:] ** 2)
        
        return -log_lik
    
    def _gradient(self, params: np.ndarray, T: np.ndarray, 
                  E: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        n, p = X.shape
        lambda_params = params[:-1]
        log_alpha = params[-1]
        alpha = np.exp(log_alpha)
        
        eta = X @ lambda_params
        log_T = np.log(T)
        residual = log_T - eta
        z = alpha * residual
        exp_z = np.exp(z)
        
        # Common term: p = exp(z) / (1 + exp(z))
        p = exp_z / (1 + exp_z)
        
        # Gradient w.r.t. lambda parameters
        # ∂LL/∂λ = alpha * Σ[X * (p - δ*(1-p))]
        #        = alpha * Σ[X * (p - δ + δ*p)]
        #        = alpha * Σ[X * ((1+δ)*p - δ)]
        d_lambda = alpha * X.T @ ((1 + E) * p - E)
        
        # Gradient w.r.t. log(alpha)
        # d(log_pdf)/d(log(α)) = 1 + residual*alpha*(1 - 2*p)
        # d(log_survival)/d(log(α)) = -p*residual*alpha
        d_log_alpha = np.sum(E * (1 + residual * alpha * (1 - 2 * p)) + 
                             (1 - E) * (-p * residual * alpha))
        
        if self.penalizer > 0 and p > 1:
            d_lambda[1:] -= self.penalizer * lambda_params[1:]
        
        grad = np.concatenate([d_lambda, [d_log_alpha]])
        return -grad
    
    def fit(self, T: np.ndarray, E: np.ndarray, X: Optional[np.ndarray] = None) -> 'LogLogisticAFTFitter':
        """Fit model"""
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        n = len(T)
        
        if X is None:
            X = np.ones((n, 1))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = np.column_stack([np.ones(n), X])
        
        p = X.shape[1]
        
        # Initialize
        median_time = np.median(T)
        mu_init = np.log(median_time)
        beta_init = np.zeros(p - 1)
        log_alpha_init = 0.0  # alpha = 1
        
        initial_params = np.concatenate([[mu_init], beta_init, [log_alpha_init]])
        
        # Optimize
        methods = ['L-BFGS-B', 'BFGS']
        best_result = None
        best_nll = np.inf
        
        for method in methods:
            try:
                if method == 'L-BFGS-B':
                    bounds = [(None, None)] * (p + 1)
                    bounds[-1] = (-3, 3)
                    result = minimize(
                        fun=self._negative_log_likelihood,
                        x0=initial_params,
                        args=(T, E, X),
                        method=method,
                        jac=self._gradient,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                else:
                    result = minimize(
                        fun=self._negative_log_likelihood,
                        x0=initial_params,
                        args=(T, E, X),
                        method=method,
                        jac=self._gradient,
                        options={'maxiter': 1000}
                    )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
                    
                if result.success:
                    break
            except:
                continue
        
        result = best_result
        
        # Extract parameters
        self.params_ = result.x
        self.lambda_params_ = result.x[:-1]
        self.alpha_ = np.exp(result.x[-1])
        self.log_likelihood_ = -result.fun
        
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if isinstance(result.hess_inv, np.ndarray):
                self.variance_matrix_ = result.hess_inv
            else:
                try:
                    self.variance_matrix_ = np.array(
                        [result.hess_inv @ np.eye(len(result.x))[i] 
                         for i in range(len(result.x))]
                    ).T
                except:
                    self.variance_matrix_ = np.eye(len(result.x))
        else:
            self.variance_matrix_ = np.eye(len(result.x))
        
        self._n_covariates = p - 1
        return self
    
    def predict_survival_function(self, X: Optional[np.ndarray] = None, 
                                   times: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict survival function S(t|X)"""
        if times is None:
            times = np.linspace(0.01, np.median(np.exp(self.lambda_params_[0])) * 3, 100)
        times = np.asarray(times).flatten()
        times = np.maximum(times, 1e-10)
        
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        
        # S(t|X) = 1 / (1 + (t/exp(eta))^alpha)
        eta_expanded = eta[:, np.newaxis]
        times_expanded = times[np.newaxis, :]
        
        ratio = times_expanded / np.exp(eta_expanded)
        S = 1.0 / (1.0 + np.power(ratio, self.alpha_))
        
        return S.flatten() if S.shape[0] == 1 else S
    
    def predict_median(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict median survival time"""
        if X is None:
            X = np.zeros((1, self._n_covariates))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        
        X = np.column_stack([np.ones(X.shape[0]), X])
        eta = X @ self.lambda_params_
        
        # For log-logistic, median = exp(mu + beta'X)
        median = np.exp(eta)
        return median.flatten() if len(median) > 1 else median[0]
    



"""
Additional Parametric Survival Models
Add these classes to the end of parametric_models.py
"""

class ExponentialFitter:
    """
    Exponential Survival Model (PH parameterization)
    
    Special case of Weibull with rho = 1 (constant hazard)
    
    Hazard: h(t|X) = lambda * exp(beta'X)
    Survival: S(t|X) = exp(-lambda * t * exp(beta'X))
    
    The exponential is memoryless: P(T > s+t | T > s) = P(T > t)
    """
    
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_ = None
        self.coef_ = None
        self.lambda_ = None
        self.log_likelihood_ = None
        self.variance_matrix_ = None
        
    def _negative_log_likelihood(self, params: np.ndarray, T: np.ndarray, 
                                  E: np.ndarray, X: np.ndarray) -> float:
        """Compute negative log-likelihood"""
        n, p = X.shape
        log_lambda = params[0]
        beta = params[1:]
        
        lambda_ = np.exp(log_lambda)
        eta = X @ beta
        
        # For exponential: h(t) = lambda * exp(eta), H(t) = lambda * t * exp(eta)
        log_hazard = log_lambda + eta
        cum_hazard = lambda_ * T * np.exp(eta)
        
        log_lik = np.sum(E * log_hazard - cum_hazard)
        
        if self.penalizer > 0:
            log_lik -= 0.5 * self.penalizer * np.sum(beta ** 2)
        
        return -log_lik
    
    def _gradient(self, params: np.ndarray, T: np.ndarray, 
                  E: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        n, p = X.shape
        log_lambda = params[0]
        beta = params[1:]
        
        lambda_ = np.exp(log_lambda)
        eta = X @ beta
        exp_eta = np.exp(eta)
        
        # Gradient w.r.t. log(lambda)
        d_log_lambda = np.sum(E - lambda_ * T * exp_eta)
        
        # Gradient w.r.t. beta
        d_beta = X.T @ (E - lambda_ * T * exp_eta)
        
        if self.penalizer > 0:
            d_beta -= self.penalizer * beta
        
        grad = np.concatenate([[d_log_lambda], d_beta])
        return -grad
    
    def fit(self, T: np.ndarray, E: np.ndarray, X: Optional[np.ndarray] = None) -> 'ExponentialFitter':
        """Fit model"""
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        n = len(T)
        
        if X is None:
            X = np.ones((n, 1))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = np.column_stack([np.ones(n), X])
        
        p = X.shape[1]
        
        # Initialize: lambda = 1/mean(T[E==1])
        mean_event_time = np.mean(T[E == 1]) if np.any(E) else np.mean(T)
        log_lambda_init = -np.log(mean_event_time)
        beta_init = np.zeros(p)
        
        initial_params = np.concatenate([[log_lambda_init], beta_init])
        
        # Optimize
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(T, E, X),
            method='BFGS',
            jac=self._gradient,
            options={'maxiter': 500}
        )
        
        # Extract parameters
        self.params_ = result.x
        self.lambda_ = np.exp(result.x[0])
        self.coef_ = result.x[1:]
        self.log_likelihood_ = -result.fun
        
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if isinstance(result.hess_inv, np.ndarray):
                self.variance_matrix_ = result.hess_inv
            else:
                self.variance_matrix_ = np.array(
                    [result.hess_inv @ np.eye(len(result.x))[i] 
                     for i in range(len(result.x))]
                ).T
        
        return self
    
    def predict_survival_function(self, X: Optional[np.ndarray] = None, 
                                   times: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict survival function S(t|X)"""
        if times is None:
            times = np.linspace(0.01, 1/self.lambda_ * 5, 100)
        times = np.asarray(times).flatten()
        
        # Check if this is an intercept-only model
        is_intercept_only = len(self.coef_) == 1
        
        if X is None:
            if is_intercept_only:
                # Baseline: S(t) = exp(-lambda * t)
                lambda_eff = self.lambda_ * np.exp(self.coef_[0])
                S = np.exp(-lambda_eff * times)
                return S
            else:
                X = np.zeros((1, len(self.coef_)))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        eta = X @ self.coef_
        exp_eta = np.exp(eta)[:, np.newaxis]
        times_expanded = times[np.newaxis, :]
        
        S = np.exp(-self.lambda_ * times_expanded * exp_eta)
        return S.flatten() if S.shape[0] == 1 else S
    
    def predict_median(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict median survival time"""
        is_intercept_only = len(self.coef_) == 1
        
        if X is None:
            if is_intercept_only:
                lambda_eff = self.lambda_ * np.exp(self.coef_[0])
                return np.log(2) / lambda_eff
            else:
                X = np.zeros((1, len(self.coef_)))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        eta = X @ self.coef_
        median = np.log(2) / (self.lambda_ * np.exp(eta))
        return median.flatten() if len(median) > 1 else median[0]
    
    def predict_expectation(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict mean survival time"""
        is_intercept_only = len(self.coef_) == 1
        
        if X is None:
            if is_intercept_only:
                lambda_eff = self.lambda_ * np.exp(self.coef_[0])
                return 1.0 / lambda_eff
            else:
                X = np.zeros((1, len(self.coef_)))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        eta = X @ self.coef_
        mean = 1.0 / (self.lambda_ * np.exp(eta))
        return mean.flatten() if len(mean) > 1 else mean[0]


