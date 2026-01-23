
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
        
        # Pre-compute sort indices for each feature (reused across all trees)
        self._sort_indices = np.empty((n_features, n_samples), dtype=np.intp)
        for f_idx in range(n_features):
            self._sort_indices[f_idx] = np.argsort(X[:, f_idx])

        # Pre-compute duration sort order (used in gradient and loss computation)
        self._duration_order = np.argsort(durations)
        self._durations_sorted = durations[self._duration_order]
        self._events_sorted = events[self._duration_order]
        # Pre-compute unique duration positions for tie handling
        self._unique_dur_vals, self._unique_dur_first = np.unique(
            self._durations_sorted, return_index=True
        )

        # Boosting iterations
        for m in range(self.n_estimators):
            # Compute gradients
            residuals = self._compute_gradients(F, durations, events)

            # Subsample if needed
            if self.subsample < 1.0:
                sub_indices = np.random.choice(
                    n_samples,
                    size=int(self.subsample * n_samples),
                    replace=False
                )
                X_sample = X[sub_indices]
                residuals_sample = residuals[sub_indices]
            else:
                X_sample = X
                residuals_sample = residuals

            # Fit tree (use pre-sorted indices for full data)
            use_presorted = (self.subsample >= 1.0)
            tree = self._fit_regression_tree(X_sample, residuals_sample, use_presorted=use_presorted)

            # Make predictions
            tree_predictions = self._predict_tree(tree, X)
            
            # Update model
            F = F + self.learning_rate * tree_predictions
            
            # Store
            self.estimators_.append(tree)

            # Update importances
            self.feature_importances_ += tree['feature_importances']

            # Compute loss only when needed (verbose or OOB)
            if self.verbose > 0 or self.subsample < 1.0:
                loss = self._compute_loss(F, durations, events)
                self.train_score_.append(loss)

            # OOB improvement
            if self.subsample < 1.0:
                oob_indices = np.array([i for i in range(n_samples) if i not in sub_indices])
                if len(oob_indices) > 0:
                    F_oob_old = F[oob_indices] - self.learning_rate * tree_predictions[oob_indices]
                    loss_oob_old = self._compute_loss(F_oob_old, durations[oob_indices], events[oob_indices])
                    loss_oob_new = self._compute_loss(F[oob_indices], durations[oob_indices], events[oob_indices])
                    self.oob_improvement_.append(loss_oob_old - loss_oob_new)
            
            # Verbose
            if self.verbose > 0 and (m + 1) % max(1, self.n_estimators // 10) == 0:
                if self.train_score_:
                    print(f"  Iteration {m+1}/{self.n_estimators}, Loss: {self.train_score_[-1]:.4f}")
        
        # Clean up pre-computed data (not needed after fit)
        for attr in ('_sort_indices', '_duration_order', '_durations_sorted',
                     '_events_sorted', '_unique_dur_vals', '_unique_dur_first'):
            if hasattr(self, attr):
                delattr(self, attr)

        # Normalize importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        # Estimate baseline hazard
        self._estimate_baseline_hazard(F, durations, events)
        
        self._is_fitted = True
        
        if self.verbose > 0:
            print(f"✅ Gradient Boosting fitted")
        
        return self
    
    def _fit_regression_tree(self, X: np.ndarray, y: np.ndarray, use_presorted=False) -> Dict:
        """Fit a regression tree using index-mask approach to avoid repeated sorting."""
        n_samples, n_features = X.shape

        # Determine max features per split
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

        # Use pre-sorted indices to avoid sorting at every node
        if use_presorted and hasattr(self, '_sort_indices'):
            # Build filtered sort orders for each feature
            # _sort_indices[f] gives indices that sort X[:, f]
            # We can filter these for any subset using a boolean mask
            tree = self._build_tree_with_indices(
                X, y, np.arange(n_samples), depth=0, max_features=max_features
            )
        else:
            tree = self._build_tree_simple(X, y, depth=0, max_features=max_features)
        return tree

    def _build_tree_with_indices(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
        depth: int, max_features: int
    ) -> Dict:
        """Build regression tree using pre-sorted indices (no sorting at any level)."""
        n_samples = len(indices)

        node = {
            'n_samples': n_samples,
            'prediction': y[indices].mean(),
            'feature_importances': np.zeros(self.n_features_)
        }

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split):
            node['is_leaf'] = True
            return node

        y_node = y[indices]
        if np.all(y_node == y_node[0]):
            node['is_leaf'] = True
            return node

        # Find best split using pre-sorted indices
        best_split = self._find_split_with_indices(X, y, indices, max_features)

        if best_split is None or best_split['improvement'] < 1e-7:
            node['is_leaf'] = True
            return node

        # Split indices
        feature_vals = X[indices, best_split['feature']]
        left_mask = feature_vals <= best_split['threshold']
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        node['feature_importances'][best_split['feature']] = best_split['improvement']
        node['is_leaf'] = False
        node['feature'] = best_split['feature']
        node['threshold'] = best_split['threshold']

        node['left'] = self._build_tree_with_indices(
            X, y, left_indices, depth + 1, max_features
        )
        node['right'] = self._build_tree_with_indices(
            X, y, right_indices, depth + 1, max_features
        )

        node['feature_importances'] += node['left']['feature_importances']
        node['feature_importances'] += node['right']['feature_importances']
        return node

    def _find_split_with_indices(
        self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, max_features: int
    ) -> Optional[Dict]:
        """Find best split using pre-sorted indices (O(n) filter instead of O(n log n) sort)."""
        n_samples = len(indices)
        n_features = X.shape[1]

        if max_features < n_features:
            features = np.random.choice(n_features, max_features, replace=False)
        else:
            features = np.arange(n_features)

        y_node = y[indices]
        total_sum = y_node.sum()
        total_sq_sum = (y_node * y_node).sum()
        total_variance = total_sq_sum - total_sum * total_sum / n_samples

        best_split = None
        best_improvement = -np.inf
        MAX_THRESHOLDS = 30

        # Create membership mask for O(1) lookup
        in_node = np.zeros(len(X), dtype=bool)
        in_node[indices] = True

        for feature_idx in features:
            # Filter pre-sorted indices to get sorted order for this node
            full_sort = self._sort_indices[feature_idx]
            # O(n_total) filter - but we only read the boolean mask
            node_sort = full_sort[in_node[full_sort]]

            sorted_vals = X[node_sort, feature_idx]
            sorted_y = y[node_sort]

            # Cumulative sums for variance computation
            cum_sum = np.cumsum(sorted_y)
            cum_sq_sum = np.cumsum(sorted_y * sorted_y)

            # Find split positions
            change_mask = sorted_vals[1:] != sorted_vals[:-1]
            change_positions = np.where(change_mask)[0]

            if len(change_positions) == 0:
                continue

            if len(change_positions) > MAX_THRESHOLDS:
                idx = np.linspace(0, len(change_positions) - 1, MAX_THRESHOLDS, dtype=int)
                change_positions = change_positions[idx]

            n_lefts = change_positions + 1
            n_rights = n_samples - n_lefts
            valid = (n_lefts >= self.min_samples_leaf) & (n_rights >= self.min_samples_leaf)
            if not valid.any():
                continue

            valid_positions = change_positions[valid]
            valid_n_lefts = n_lefts[valid]
            valid_n_rights = n_rights[valid]

            left_sums = cum_sum[valid_positions]
            left_sq_sums = cum_sq_sum[valid_positions]
            left_var_n = left_sq_sums - left_sums * left_sums / valid_n_lefts

            right_sums = total_sum - left_sums
            right_sq_sums = total_sq_sum - left_sq_sums
            right_var_n = right_sq_sums - right_sums * right_sums / valid_n_rights

            improvements = total_variance - (left_var_n + right_var_n)

            best_idx = np.argmax(improvements)
            if improvements[best_idx] > best_improvement:
                best_improvement = improvements[best_idx]
                pos = valid_positions[best_idx]
                best_split = {
                    'feature': feature_idx,
                    'threshold': (sorted_vals[pos] + sorted_vals[pos + 1]) / 2,
                    'improvement': best_improvement,
                    'n_left': int(valid_n_lefts[best_idx]),
                    'n_right': int(valid_n_rights[best_idx])
                }

        return best_split

    def _build_tree_simple(
        self, X: np.ndarray, y: np.ndarray, depth: int, max_features: int
    ) -> Dict:
        """Build regression tree with sorting (fallback for subsampled data)."""
        n_samples, n_features = X.shape

        node = {
            'n_samples': n_samples,
            'prediction': np.mean(y),
            'feature_importances': np.zeros(self.n_features_)
        }

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            np.all(y == y[0])):
            node['is_leaf'] = True
            return node

        best_split = self._find_split_simple(X, y, max_features)

        if best_split is None or best_split['improvement'] < 1e-7:
            node['is_leaf'] = True
            return node

        left_mask = X[:, best_split['feature']] <= best_split['threshold']

        node['feature_importances'][best_split['feature']] = best_split['improvement']
        node['is_leaf'] = False
        node['feature'] = best_split['feature']
        node['threshold'] = best_split['threshold']

        node['left'] = self._build_tree_simple(X[left_mask], y[left_mask], depth + 1, max_features)
        node['right'] = self._build_tree_simple(X[~left_mask], y[~left_mask], depth + 1, max_features)

        node['feature_importances'] += node['left']['feature_importances']
        node['feature_importances'] += node['right']['feature_importances']
        return node

    def _find_split_simple(
        self, X: np.ndarray, y: np.ndarray, max_features: int
    ) -> Optional[Dict]:
        """Find best split with sorting (for subsampled data)."""
        n_samples, n_features = X.shape

        if max_features < n_features:
            features = np.random.choice(n_features, max_features, replace=False)
        else:
            features = np.arange(n_features)

        best_split = None
        best_improvement = -np.inf
        total_sum = y.sum()
        total_sq_sum = (y * y).sum()
        total_variance = total_sq_sum - total_sum * total_sum / n_samples
        MAX_THRESHOLDS = 30

        for feature_idx in features:
            sort_idx = np.argsort(X[:, feature_idx])
            sorted_vals = X[sort_idx, feature_idx]
            sorted_y = y[sort_idx]

            cum_sum = np.cumsum(sorted_y)
            cum_sq_sum = np.cumsum(sorted_y * sorted_y)

            change_mask = sorted_vals[1:] != sorted_vals[:-1]
            change_positions = np.where(change_mask)[0]

            if len(change_positions) == 0:
                continue
            if len(change_positions) > MAX_THRESHOLDS:
                idx = np.linspace(0, len(change_positions) - 1, MAX_THRESHOLDS, dtype=int)
                change_positions = change_positions[idx]

            n_lefts = change_positions + 1
            n_rights = n_samples - n_lefts
            valid = (n_lefts >= self.min_samples_leaf) & (n_rights >= self.min_samples_leaf)
            if not valid.any():
                continue

            valid_positions = change_positions[valid]
            valid_n_lefts = n_lefts[valid]
            valid_n_rights = n_rights[valid]

            left_sums = cum_sum[valid_positions]
            left_sq_sums = cum_sq_sum[valid_positions]
            left_var_n = left_sq_sums - left_sums * left_sums / valid_n_lefts
            right_sums = total_sum - left_sums
            right_sq_sums = total_sq_sum - left_sq_sums
            right_var_n = right_sq_sums - right_sums * right_sums / valid_n_rights
            improvements = total_variance - (left_var_n + right_var_n)

            best_idx = np.argmax(improvements)
            if improvements[best_idx] > best_improvement:
                best_improvement = improvements[best_idx]
                pos = valid_positions[best_idx]
                best_split = {
                    'feature': feature_idx,
                    'threshold': (sorted_vals[pos] + sorted_vals[pos + 1]) / 2,
                    'improvement': best_improvement,
                    'n_left': int(valid_n_lefts[best_idx]),
                    'n_right': int(valid_n_rights[best_idx])
                }

        return best_split
    
    def _predict_tree(self, tree: Dict, X: np.ndarray) -> np.ndarray:
        """Predict using tree - vectorized batch traversal."""
        predictions = np.empty(len(X))
        # Use stack-based traversal with index masks for vectorized prediction
        stack = [(tree, np.arange(len(X)))]
        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue
            if node['is_leaf']:
                predictions[indices] = node['prediction']
            else:
                feature_vals = X[indices, node['feature']]
                left_mask = feature_vals <= node['threshold']
                right_mask = ~left_mask
                stack.append((node['left'], indices[left_mask]))
                stack.append((node['right'], indices[right_mask]))
        return predictions
    
    def _compute_gradients(
        self, risk_scores: np.ndarray, durations: np.ndarray, events: np.ndarray
    ) -> np.ndarray:
        """Compute negative gradients of Cox partial likelihood using pre-sorted order."""
        n = len(risk_scores)

        # Use pre-computed sort order
        order = self._duration_order
        events_sorted = self._events_sorted
        risk_scores_sorted = risk_scores[order]

        # Numerical stability
        rs_max = risk_scores_sorted.max()
        exp_risk = np.exp(risk_scores_sorted - rs_max)

        # Risk set sum: reverse cumsum in ascending order
        risk_set_sums = np.cumsum(exp_risk[::-1])[::-1]

        # Handle ties using pre-computed unique positions
        first_idx = self._unique_dur_first
        n_unique = len(first_idx)
        for idx in range(n_unique):
            start = first_idx[idx]
            end = first_idx[idx + 1] if idx + 1 < n_unique else n
            if end > start + 1:
                risk_set_sums[start:end] = risk_set_sums[start]

        # Event contributions
        event_contributions = np.where(events_sorted == 1, 1.0 / risk_set_sums, 0.0)
        cumulative_event_contrib = np.cumsum(event_contributions)

        # Handle ties for cumulative contribution
        for idx in range(n_unique):
            start = first_idx[idx]
            end = first_idx[idx + 1] if idx + 1 < n_unique else n
            if end > start + 1:
                cumulative_event_contrib[start:end] = cumulative_event_contrib[end - 1]

        # Gradient
        gradients_sorted = exp_risk * cumulative_event_contrib - events_sorted

        # Map back to original order
        gradients = np.empty(n)
        gradients[order] = gradients_sorted
        return -gradients
    
    def _compute_loss(
        self, risk_scores: np.ndarray, durations: np.ndarray, events: np.ndarray
    ) -> float:
        """Compute negative Cox partial log-likelihood using vectorized cumsum."""
        # Sort ascending by duration
        order = np.argsort(durations)
        durations_sorted = durations[order]
        events_sorted = events[order]
        risk_scores_sorted = risk_scores[order]

        # Numerical stability
        rs_max = risk_scores_sorted.max()
        exp_risk = np.exp(risk_scores_sorted - rs_max)

        # Risk set sums: reverse cumsum in ascending order
        risk_set_sums = np.cumsum(exp_risk[::-1])[::-1]

        # Handle ties: first occurrence in ascending order has the correct sum
        n = len(durations_sorted)
        unique_vals, first_idx = np.unique(durations_sorted, return_index=True)
        for idx in range(len(first_idx)):
            start = first_idx[idx]
            end = first_idx[idx + 1] if idx + 1 < len(first_idx) else n
            if end > start + 1:
                risk_set_sums[start:end] = risk_set_sums[start]

        # Log-likelihood: Σ_{i: δ_i=1} [η_i - log(Σ_{j:T_j>=T_i} exp(η_j))]
        # = Σ_{i: δ_i=1} [(η_i - rs_max) - log(risk_set_sum_i)]
        event_mask = events_sorted == 1
        log_likelihood = np.sum(
            (risk_scores_sorted[event_mask] - rs_max) - np.log(risk_set_sums[event_mask])
        )

        return -log_likelihood
    
    def _estimate_baseline_hazard(
        self, risk_scores: np.ndarray, durations: np.ndarray, events: np.ndarray
    ):
        """Estimate baseline hazard using Breslow estimator - vectorized."""
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

        # Vectorized: compute exp(risk) and reverse cumsum for risk set sums
        exp_risk = np.exp(risk_scores_sorted)
        # Risk set sum at each unique time using searchsorted
        # For each unique_time t, at_risk = durations >= t, i.e. indices >= searchsorted(t, 'left')
        positions = np.searchsorted(durations_sorted, unique_times, side='left')
        risk_set_sums = np.array([exp_risk[pos:].sum() for pos in positions])

        # Count events at each unique time using searchsorted on event times
        event_durations = durations_sorted[events_sorted == 1]
        event_counts = np.zeros(len(unique_times))
        idx = np.searchsorted(unique_times, event_durations)
        np.add.at(event_counts, idx, 1)

        # Baseline hazard
        nonzero = risk_set_sums > 0
        baseline_hazard[1:][nonzero] = event_counts[nonzero] / risk_set_sums[nonzero]

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
        """Calculate Harrell's concordance index - vectorized."""
        n = len(durations)
        if n < 2:
            return 0.5

        event_mask = events == 1
        if event_mask.sum() < 1:
            return 0.5

        # Sort by duration
        order = np.argsort(durations)
        dur_sorted = durations[order]
        evt_sorted = events[order]
        risk_sorted = risk_scores[order]

        concordant = 0.0
        discordant = 0.0
        tied_risk = 0.0

        for i in range(n):
            if evt_sorted[i] == 0:
                continue
            # All j where T_j > T_i
            later = dur_sorted > dur_sorted[i]
            if not later.any():
                continue
            later_risks = risk_sorted[later]
            concordant += np.sum(risk_sorted[i] > later_risks)
            discordant += np.sum(risk_sorted[i] < later_risks)
            tied_risk += np.sum(risk_sorted[i] == later_risks)

        permissible = concordant + discordant + tied_risk
        if permissible == 0:
            return 0.5
        return (concordant + 0.5 * tied_risk) / permissible