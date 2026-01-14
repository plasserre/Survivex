
"""
Gradient Boosting Survival Analysis Implementation with GPU Support

Mathematical Foundation:
=======================

Gradient boosting builds an additive ensemble model by sequentially fitting
weak learners (typically shallow trees) to the negative gradient of a loss function.

For survival analysis, we use the negative Cox partial likelihood as the loss function.

Cox Partial Likelihood:
-----------------------
For a dataset with n samples, the partial likelihood is:

    L(β) = Π_{i:δ_i=1} [ exp(η_i) / Σ_{j:T_j≥T_i} exp(η_j) ]

Where:
    η_i = f(X_i) is the risk score
    δ_i is the event indicator
    T_i is the observed time
    The product is over all events

Negative Log-Partial Likelihood (Loss):
---------------------------------------
    ℓ(η) = -log L(η) = Σ_{i:δ_i=1} [ -η_i + log(Σ_{j:T_j≥T_i} exp(η_j)) ]

Gradient:
---------
The gradient of the loss with respect to the risk score η_i is:

    ∂ℓ/∂η_i = δ_i - Σ_{j:δ_j=1,T_j≤T_i} [ exp(η_i) / Σ_{k:T_k≥T_j} exp(η_k) ]

Gradient Boosting Algorithm:
----------------------------
1. Initialize: F_0(X) = 0

2. For m = 1 to M:
   a. Compute negative gradients (pseudo-residuals):
      r_i = -∂ℓ/∂F_{m-1}(X_i)
   
   b. Fit a regression tree h_m to r_i using X
   
   c. Update model:
      F_m(X) = F_m-1(X) + ν · h_m(X)
      
      where ν is the learning rate

3. Final model: F_M(X) = risk score

Baseline Hazard Estimation:
---------------------------
After fitting, estimate baseline hazard using Breslow estimator:

    h_0(t_i) = d_i / Σ_{j:T_j≥t_i} exp(F_M(X_j))

References:
-----------
- Ridgeway, G. (1999). "The state of boosting"
- Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine"
- Friedman, J. H. (2002). "Stochastic gradient boosting"
- Hothorn, T., et al. (2006). "Survival ensembles"
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class GradientBoostingResult:
    """Results from Gradient Boosting Survival Analysis."""
    train_score: List[float]
    oob_improvement: Optional[List[float]]
    feature_importances: np.ndarray
    n_estimators: int
    learning_rate: float


class GradientBoostingSurvivalAnalysis:
    """
    Gradient Boosting for Survival Analysis with GPU support.
    
    Builds an additive ensemble by sequentially fitting regression trees
    to negative gradients of the Cox partial likelihood loss.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting iterations
    learning_rate : float, default=0.1
        Learning rate (shrinkage parameter)
    max_depth : int, default=3
        Maximum depth of base learners
    min_samples_split : int, default=10
        Minimum samples required to split
    min_samples_leaf : int, default=5
        Minimum samples in a leaf
    subsample : float, default=1.0
        Fraction of samples per iteration
    max_features : int, str, or None, default=None
        Features per split
    random_state : int, optional
        Random seed
    verbose : int, default=0
        Verbosity level
    device : str, default='cpu'
        Device for computation
    
    Examples:
    ---------
    >>> gb = GradientBoostingSurvivalAnalysis(
    ...     n_estimators=100,
    ...     learning_rate=0.1,
    ...     max_depth=3
    ... )
    >>> gb.fit(X_train, durations_train, events_train)
    >>> risk_scores = gb.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        max_features: Union[int, str, None] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        device: str = 'cpu'
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose
        
        # Set device
        if device == 'mps' and not torch.backends.mps.is_available():
            warnings.warn("MPS device not available, using CPU")
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available, using CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        
        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Attributes set during fitting
        self.estimators_ = []
        self.train_score_ = []
        self.oob_improvement_ = [] if subsample < 1.0 else None
        self.feature_importances_ = None
        self.baseline_hazard_ = None
        self.baseline_cumulative_hazard_ = None
        self.baseline_survival_ = None
        self.timeline_ = None
        self.n_features_ = None
        self._initial_prediction = 0.0
        self._is_fitted = False
    
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        durations: Union[np.ndarray, torch.Tensor],
        events: Union[np.ndarray, torch.Tensor]
    ) -> 'GradientBoostingSurvivalAnalysis':
        """Fit the gradient boosting model."""
        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(durations, torch.Tensor):
            durations = durations.cpu().numpy()
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        if self.verbose > 0:
            print(f"Fitting Gradient Boosting Survival Analysis...")
            print(f"  n_estimators: {self.n_estimators}")
            print(f"  learning_rate: {self.learning_rate}")
            print(f"  max_depth: {self.max_depth}")
        
        # Initialize predictions
        F = np.zeros(n_samples)
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Boosting iterations
        for m in range(self.n_estimators):
            # Compute gradients
            residuals = self._compute_gradients(F, durations, events)
            
            # Subsample if needed
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    n_samples,
                    size=int(self.subsample * n_samples),
                    replace=False
                )
                X_sample = X[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X
                residuals_sample = residuals
                sample_indices = np.arange(n_samples)
            
            # Fit tree
            tree = self._fit_regression_tree(X_sample, residuals_sample)
            
            # Make predictions
            tree_predictions = self._predict_tree(tree, X)
            
            # Update model
            F = F + self.learning_rate * tree_predictions
            
            # Store
            self.estimators_.append(tree)
            
            # Compute loss
            loss = self._compute_loss(F, durations, events)
            self.train_score_.append(loss)
            
            # Update importances
            self.feature_importances_ += tree['feature_importances']
            
            # OOB improvement
            if self.subsample < 1.0:
                oob_indices = np.array([i for i in range(n_samples) if i not in sample_indices])
                if len(oob_indices) > 0:
                    F_oob_old = F[oob_indices] - self.learning_rate * tree_predictions[oob_indices]
                    loss_oob_old = self._compute_loss(F_oob_old, durations[oob_indices], events[oob_indices])
                    loss_oob_new = self._compute_loss(F[oob_indices], durations[oob_indices], events[oob_indices])
                    self.oob_improvement_.append(loss_oob_old - loss_oob_new)
            
            # Verbose
            if self.verbose > 0 and (m + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"  Iteration {m+1}/{self.n_estimators}, Loss: {loss:.4f}")
        
        # Normalize importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        # Estimate baseline hazard
        self._estimate_baseline_hazard(F, durations, events)
        
        self._is_fitted = True
        
        if self.verbose > 0:
            print(f"✅ Gradient Boosting fitted")
        
        return self
    
    def _fit_regression_tree(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit a simple regression tree."""
        n_samples, n_features = X.shape
        
        # Determine features
        if self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            max_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            max_features = max(1, int(np.log2(n_features)))
        else:
            max_features = n_features
        
        tree = self._build_regression_tree_recursive(X, y, depth=0, max_features=max_features)
        return tree
    
    def _build_regression_tree_recursive(
        self, X: np.ndarray, y: np.ndarray, depth: int, max_features: int
    ) -> Dict:
        """Recursively build regression tree."""
        n_samples, n_features = X.shape
        
        node = {
            'n_samples': n_samples,
            'prediction': np.mean(y),
            'feature_importances': np.zeros(self.n_features_)
        }
        
        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            np.all(y == y[0])):
            node['is_leaf'] = True
            return node
        
        # Find best split
        best_split = self._find_best_regression_split(X, y, max_features)
        
        if best_split is None or best_split['improvement'] < 1e-7:
            node['is_leaf'] = True
            return node
        
        # Check min samples
        if (best_split['n_left'] < self.min_samples_leaf or
            best_split['n_right'] < self.min_samples_leaf):
            node['is_leaf'] = True
            return node
        
        # Split
        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        right_mask = ~left_mask
        
        node['feature_importances'][best_split['feature']] = best_split['improvement']
        node['is_leaf'] = False
        node['feature'] = best_split['feature']
        node['threshold'] = best_split['threshold']
        
        # Build children
        node['left'] = self._build_regression_tree_recursive(
            X[left_mask], y[left_mask], depth + 1, max_features
        )
        node['right'] = self._build_regression_tree_recursive(
            X[right_mask], y[right_mask], depth + 1, max_features
        )
        
        node['feature_importances'] += node['left']['feature_importances']
        node['feature_importances'] += node['right']['feature_importances']
        
        return node
    
    def _find_best_regression_split(
        self, X: np.ndarray, y: np.ndarray, max_features: int
    ) -> Optional[Dict]:
        """Find best split for regression."""
        n_samples, n_features = X.shape
        
        if max_features < n_features:
            features = np.random.choice(n_features, max_features, replace=False)
        else:
            features = range(n_features)
        
        best_split = None
        best_improvement = -np.inf
        total_variance = np.var(y) * n_samples
        
        for feature_idx in features:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) < 2:
                continue
            
            if len(unique_values) > 10:
                indices = np.linspace(0, len(unique_values) - 1, 10, dtype=int)
                unique_values = unique_values[indices]
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                left_variance = np.var(y_left) * n_left
                right_variance = np.var(y_right) * n_right
                
                improvement = total_variance - (left_variance + right_variance)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_split = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'improvement': improvement,
                        'n_left': n_left,
                        'n_right': n_right
                    }
        
        return best_split
    
    def _predict_tree(self, tree: Dict, X: np.ndarray) -> np.ndarray:
        """Predict using tree."""
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            predictions[i] = self._traverse_tree(tree, x)
        return predictions
    
    def _traverse_tree(self, node: Dict, x: np.ndarray) -> float:
        """Traverse tree for prediction."""
        if node['is_leaf']:
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(node['left'], x)
        else:
            return self._traverse_tree(node['right'], x)
    
    def _compute_gradients(
        self, risk_scores: np.ndarray, durations: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        """Compute negative gradients of Cox partial likelihood."""
        n = len(risk_scores)
        gradients = np.zeros(n)
        
        order = np.argsort(durations)
        durations_sorted = durations[order]
        events_sorted = events[order]
        risk_scores_sorted = risk_scores[order]
        
        exp_risk = np.exp(risk_scores_sorted)
        event_times = durations_sorted[events_sorted == 1]
        
        for t in event_times:
            at_risk_mask = durations_sorted >= t
            sum_exp_risk = np.sum(exp_risk[at_risk_mask])
            
            if sum_exp_risk > 0:
                gradients[order[at_risk_mask]] += exp_risk[at_risk_mask] / sum_exp_risk
        
        gradients -= events
        return -gradients
    
    def _compute_loss(
        self, risk_scores: np.ndarray, durations: np.ndarray, events: np.ndarray
    ) -> float:
        """Compute negative Cox partial log-likelihood."""
        order = np.argsort(durations)
        durations_sorted = durations[order]
        events_sorted = events[order]
        risk_scores_sorted = risk_scores[order]
        
        log_likelihood = 0.0
        
        for i in range(len(durations_sorted)):
            if events_sorted[i] == 0:
                continue
            
            at_risk = durations_sorted >= durations_sorted[i]
            log_risk_sum = np.log(np.sum(np.exp(risk_scores_sorted[at_risk])))
            log_likelihood += risk_scores_sorted[i] - log_risk_sum
        
        return -log_likelihood
    
    def _estimate_baseline_hazard(
        self, risk_scores: np.ndarray, durations: np.ndarray, events: np.ndarray
    ):
        """Estimate baseline hazard using Breslow estimator."""
        order = np.argsort(durations)
        durations_sorted = durations[order]
        events_sorted = events[order]
        risk_scores_sorted = risk_scores[order]
        
        unique_times = np.unique(durations_sorted[events_sorted == 1])
        
        if len(unique_times) == 0:
            self.timeline_ = torch.tensor([0.0, durations_sorted.max()])
            self.baseline_hazard_ = torch.tensor([0.0, 0.0])
            self.baseline_cumulative_hazard_ = torch.tensor([0.0, 0.0])
            self.baseline_survival_ = torch.tensor([1.0, 1.0])
            return
        
        timeline = np.concatenate([[0.0], unique_times])
        baseline_hazard = np.zeros(len(timeline))
        
        for i, t in enumerate(unique_times):
            d_t = np.sum((durations_sorted == t) & (events_sorted == 1))
            at_risk_mask = durations_sorted >= t
            sum_exp_risk = np.sum(np.exp(risk_scores_sorted[at_risk_mask]))
            
            if sum_exp_risk > 0:
                baseline_hazard[i + 1] = d_t / sum_exp_risk
        
        baseline_cumulative_hazard = np.cumsum(baseline_hazard)
        baseline_survival = np.exp(-baseline_cumulative_hazard)
        
        self.timeline_ = torch.tensor(timeline, dtype=torch.float32, device=self.device)
        self.baseline_hazard_ = torch.tensor(baseline_hazard, dtype=torch.float32, device=self.device)
        self.baseline_cumulative_hazard_ = torch.tensor(
            baseline_cumulative_hazard, dtype=torch.float32, device=self.device
        )
        self.baseline_survival_ = torch.tensor(baseline_survival, dtype=torch.float32, device=self.device)
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict risk scores."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        predictions = np.full(len(X), self._initial_prediction)
        
        for tree in self.estimators_:
            predictions += self.learning_rate * self._predict_tree(tree, X)
        
        return predictions
    
    def predict_survival_function(
        self, X: Union[np.ndarray, torch.Tensor], times: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """Predict survival function."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        risk_scores = self.predict(X)
        
        if times is None:
            baseline_chf = self.baseline_cumulative_hazard_.cpu().numpy()
        else:
            times = np.array(times)
            baseline_chf = np.interp(
                times, self.timeline_.cpu().numpy(), self.baseline_cumulative_hazard_.cpu().numpy()
            )
        
        n_samples = len(risk_scores)
        n_times = len(baseline_chf)
        survival = np.zeros((n_samples, n_times))
        
        for i in range(n_samples):
            individual_chf = baseline_chf * np.exp(risk_scores[i])
            survival[i] = np.exp(-individual_chf)
        
        return survival
    
    def predict_cumulative_hazard(
        self, X: Union[np.ndarray, torch.Tensor], times: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """Predict cumulative hazard function."""
        risk_scores = self.predict(X)
        
        if times is None:
            baseline_chf = self.baseline_cumulative_hazard_.cpu().numpy()
        else:
            times = np.array(times)
            baseline_chf = np.interp(
                times, self.timeline_.cpu().numpy(), self.baseline_cumulative_hazard_.cpu().numpy()
            )
        
        n_samples = len(risk_scores)
        n_times = len(baseline_chf)
        chf = np.zeros((n_samples, n_times))
        
        for i in range(n_samples):
            chf[i] = baseline_chf * np.exp(risk_scores[i])
        
        return chf
    
    def score(
        self, X: Union[np.ndarray, torch.Tensor],
        durations: Union[np.ndarray, torch.Tensor],
        events: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Calculate concordance index."""
        if isinstance(durations, torch.Tensor):
            durations = durations.cpu().numpy()
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        
        risk_scores = self.predict(X)
        return self._concordance_index(durations, events, risk_scores)
    
    def _concordance_index(
        self, durations: np.ndarray, events: np.ndarray, risk_scores: np.ndarray
    ) -> float:
        """Calculate Harrell's concordance index."""
        n = len(durations)
        concordant = 0
        permissible = 0
        
        for i in range(n):
            if events[i] == 0:
                continue
            
            for j in range(n):
                if durations[j] > durations[i]:
                    permissible += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
        
        if permissible == 0:
            return 0.5
        
        return concordant / permissible