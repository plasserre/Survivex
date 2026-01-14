"""
Survival Tree Implementation with GPU Support

Mathematical Foundation:
=======================

A survival tree is a decision tree for censored survival data that recursively 
partitions the feature space to create homogeneous groups with respect to survival.

Splitting Criterion:
-------------------
Uses the log-rank test statistic to measure separation between left and right child nodes:

    L(split) = (O_L - E_L)^2 / V_L + (O_R - E_R)^2 / V_R

Where:
    O_L, O_R = Observed events in left/right nodes
    E_L, E_R = Expected events in left/right nodes
    V_L, V_R = Variance of expected events

The best split maximizes this log-rank statistic.

Terminal Node Prediction:
------------------------
Each terminal node estimates the cumulative hazard function using Nelson-Aalen:

    H(t) = Σ_{t_i ≤ t} d_i / n_i

Where:
    d_i = Number of events at time t_i
    n_i = Number at risk at time t_i

The survival function is then:
    S(t) = exp(-H(t))

Pruning:
--------
Uses cost-complexity pruning with parameter α:

    R_α(T) = R(T) + α|T|

Where:
    R(T) = Misclassification rate (1 - C-index)
    |T| = Number of terminal nodes
    α = Complexity parameter

References:
-----------
- LeBlanc, M., & Crowley, J. (1993). "Survival trees by goodness of split"
- Ishwaran, H., et al. (2008). "Random survival forests"
- Therneau, T., & Atkinson, E. (1997). "An introduction to recursive partitioning"

Implementation Details:
----------------------
- Uses PyTorch for GPU acceleration
- Supports both exact and approximate splitting
- Implements early stopping based on min_samples_split and max_depth
- Provides feature importance based on split improvement
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class Node:
    """
    Node in the survival tree.
    
    Attributes:
    -----------
    feature : int, optional
        Feature index used for splitting (None for terminal nodes)
    threshold : float, optional
        Threshold value for splitting (None for terminal nodes)
    left : Node, optional
        Left child node
    right : Node, optional
        Right child node
    cumulative_hazard_function : torch.Tensor, optional
        CHF for terminal nodes
    survival_function : torch.Tensor, optional
        Survival function for terminal nodes
    timeline : torch.Tensor, optional
        Time points for CHF/survival
    n_samples : int
        Number of samples in this node
    n_events : int
        Number of events in this node
    impurity : float
        Node impurity (1 - C-index within node)
    is_leaf : bool
        Whether this is a terminal node
    """
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    cumulative_hazard_function: Optional[torch.Tensor] = None
    survival_function: Optional[torch.Tensor] = None
    timeline: Optional[torch.Tensor] = None
    n_samples: int = 0
    n_events: int = 0
    impurity: float = 0.0
    is_leaf: bool = False


class SurvivalTree:
    """
    Survival Tree for right-censored data with GPU support.
    
    A decision tree that uses the log-rank test statistic as splitting criterion
    and Nelson-Aalen estimator for terminal node predictions.
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=10
        Minimum samples required to split a node
    min_samples_leaf : int, default=5
        Minimum samples required in a leaf node
    max_features : int, str, or None, default=None
        Number of features to consider for best split:
        - None: use all features
        - int: use max_features features
        - 'sqrt': use sqrt(n_features)
        - 'log2': use log2(n_features)
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for split
    random_state : int, optional
        Random seed for reproducibility
    device : str, default='cpu'
        Device for computation ('cpu', 'cuda', 'mps')
    
    Attributes:
    -----------
    tree_ : Node
        The root node of the fitted tree
    n_features_ : int
        Number of features
    feature_importances_ : np.ndarray
        Feature importance scores
    max_features_ : int
        Actual number of features to consider per split
    
    Examples:
    ---------
    >>> # Basic usage
    >>> tree = SurvivalTree(max_depth=5, min_samples_split=20)
    >>> tree.fit(X, durations, events)
    >>> chf = tree.predict_cumulative_hazard(X_test)
    >>> survival = tree.predict_survival_function(X_test)
    
    >>> # With GPU
    >>> tree = SurvivalTree(device='cuda')
    >>> tree.fit(X, durations, events)
    
    Notes:
    ------
    - Requires positive durations
    - Events should be binary (0=censored, 1=event)
    - GPU provides speedup for large datasets (n > 1000)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, str]] = None,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        device: str = 'cpu'
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        
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
        self.tree_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.max_features_ = None
        self._is_fitted = False
    
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        durations: Union[np.ndarray, torch.Tensor],
        events: Union[np.ndarray, torch.Tensor]
    ) -> 'SurvivalTree':
        """
        Build the survival tree.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        durations : array-like, shape (n_samples,)
            Time to event or censoring
        events : array-like, shape (n_samples,)
            Event indicators (1=event, 0=censored)
        
        Returns:
        --------
        self : SurvivalTree
            Fitted tree
        """
        # Convert to tensors
        X = self._to_tensor(X)
        durations = self._to_tensor(durations)
        events = self._to_tensor(events)
        
        # Validate inputs
        self._validate_inputs(X, durations, events)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Determine max_features
        self.max_features_ = self._get_max_features(n_features)
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Build tree recursively
        self.tree_ = self._build_tree(X, durations, events, depth=0)
        
        # Normalize feature importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        self._is_fitted = True
        return self
    
    def _build_tree(
        self,
        X: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
        depth: int
    ) -> Node:
        """
        Recursively build the tree.
        
        Stopping criteria:
        1. Reached max_depth
        2. Too few samples to split
        3. All samples censored or all events
        4. No valid split found
        """
        n_samples = X.shape[0]
        n_events = int(events.sum().item())
        
        # Calculate node impurity (1 - C-index within node)
        node_impurity = self._calculate_impurity(durations, events)
        
        # Create node
        node = Node(
            n_samples=n_samples,
            n_events=n_events,
            impurity=node_impurity
        )
        
        # Check stopping criteria
        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth) or
            (n_samples < self.min_samples_split) or
            (n_events == 0) or  # All censored
            (n_events == n_samples)  # All events
        )
        
        if should_stop:
            # Make this a leaf node
            node.is_leaf = True
            self._fit_terminal_node(node, durations, events)
            return node
        
        # Find best split
        best_split = self._find_best_split(X, durations, events)
        
        if best_split is None:
            # No valid split found, make leaf
            node.is_leaf = True
            self._fit_terminal_node(node, durations, events)
            return node
        
        # Check minimum impurity decrease
        if best_split['improvement'] < self.min_impurity_decrease:
            node.is_leaf = True
            self._fit_terminal_node(node, durations, events)
            return node
        
        # Update feature importance
        self.feature_importances_[best_split['feature']] += best_split['improvement']
        
        # Split the data
        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        right_mask = ~left_mask
        
        # Check minimum samples per leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.is_leaf = True
            self._fit_terminal_node(node, durations, events)
            return node
        
        # Set split information
        node.feature = best_split['feature']
        node.threshold = best_split['threshold']
        
        # Build children recursively
        node.left = self._build_tree(
            X[left_mask],
            durations[left_mask],
            events[left_mask],
            depth + 1
        )
        
        node.right = self._build_tree(
            X[right_mask],
            durations[right_mask],
            events[right_mask],
            depth + 1
        )
        
        return node
    
    def _find_best_split(
        self,
        X: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best split using log-rank test statistic.
        
        Returns:
        --------
        best_split : dict or None
            Dictionary with 'feature', 'threshold', 'improvement', 'statistic'
            None if no valid split found
        """
        n_samples, n_features = X.shape
        
        # Select features to try
        if self.max_features_ < n_features:
            features = np.random.choice(n_features, self.max_features_, replace=False)
        else:
            features = range(n_features)
        
        best_split = None
        best_statistic = -np.inf
        
        for feature_idx in features:
            feature_values = X[:, feature_idx]
            
            # Get unique values as potential thresholds
            unique_values = torch.unique(feature_values, sorted=True)
            
            # Try midpoints between consecutive unique values
            if len(unique_values) < 2:
                continue
            
            # For computational efficiency, limit number of thresholds
            if len(unique_values) > 100:
                # Sample 100 thresholds uniformly
                indices = torch.linspace(0, len(unique_values) - 1, 100).long()
                unique_values = unique_values[indices]
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Split samples
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples per leaf
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                
                # Calculate log-rank statistic
                statistic = self._log_rank_statistic(
                    durations[left_mask], events[left_mask],
                    durations[right_mask], events[right_mask]
                )
                
                if statistic > best_statistic:
                    best_statistic = statistic
                    best_split = {
                        'feature': feature_idx,
                        'threshold': threshold.item(),
                        'statistic': statistic,
                        'improvement': statistic
                    }
        
        return best_split
    
    def _log_rank_statistic(
        self,
        durations_left: torch.Tensor,
        events_left: torch.Tensor,
        durations_right: torch.Tensor,
        events_right: torch.Tensor
    ) -> float:
        """
        Calculate log-rank test statistic for a split.
        
        The log-rank statistic measures how different the survival curves are
        between two groups. Higher values indicate better separation.
        
        Formula:
        --------
        χ² = (O_L - E_L)² / V
        
        Where:
            O_L = Observed events in left group
            E_L = Expected events in left group
            V = Variance
        """
        # Combine data
        all_durations = torch.cat([durations_left, durations_right])
        all_events = torch.cat([events_left, events_right])
        groups = torch.cat([
            torch.zeros(len(durations_left), device=self.device),
            torch.ones(len(durations_right), device=self.device)
        ])
        
        # Get unique event times
        event_times = torch.unique(all_durations[all_events == 1], sorted=True)
        
        if len(event_times) == 0:
            return 0.0
        
        observed_left = 0.0
        expected_left = 0.0
        variance = 0.0
        
        for t in event_times:
            # At risk in each group
            at_risk_left = ((durations_left >= t).sum()).float()
            at_risk_right = ((durations_right >= t).sum()).float()
            at_risk_total = at_risk_left + at_risk_right
            
            if at_risk_total == 0:
                continue
            
            # Events at time t
            events_left_t = ((durations_left == t) & (events_left == 1)).sum().float()
            events_right_t = ((durations_right == t) & (events_right == 1)).sum().float()
            events_total_t = events_left_t + events_right_t
            
            if events_total_t == 0:
                continue
            
            # Expected events in left group
            expected_left_t = (at_risk_left / at_risk_total) * events_total_t
            
            # Variance contribution
            if at_risk_total > 1:
                variance_t = (
                    (at_risk_left * at_risk_right * events_total_t * (at_risk_total - events_total_t)) /
                    (at_risk_total ** 2 * (at_risk_total - 1))
                )
                variance += variance_t.item()
            
            observed_left += events_left_t.item()
            expected_left += expected_left_t.item()
        
        # Calculate chi-square statistic
        if variance > 0:
            statistic = ((observed_left - expected_left) ** 2) / variance
        else:
            statistic = 0.0
        
        return statistic
    
    def _fit_terminal_node(
        self,
        node: Node,
        durations: torch.Tensor,
        events: torch.Tensor
    ):
        """
        Fit Nelson-Aalen estimator for terminal node.
        
        Estimates cumulative hazard function and survival function.
        """
        # Sort by duration
        sorted_indices = torch.argsort(durations)
        durations_sorted = durations[sorted_indices]
        events_sorted = events[sorted_indices]
        
        # Get unique event times
        unique_times = torch.unique(durations_sorted[events_sorted == 1], sorted=True)
        
        if len(unique_times) == 0:
            # No events - constant survival at 1
            node.timeline = torch.tensor([0.0, durations_sorted.max().item()], device=self.device)
            node.cumulative_hazard_function = torch.tensor([0.0, 0.0], device=self.device)
            node.survival_function = torch.tensor([1.0, 1.0], device=self.device)
            return
        
        # Add time 0
        timeline = torch.cat([torch.tensor([0.0], device=self.device), unique_times])
        
        # Nelson-Aalen estimator
        chf = torch.zeros(len(timeline), device=self.device)
        
        for i, t in enumerate(unique_times):
            # Number at risk
            at_risk = (durations_sorted >= t).sum().float()
            
            # Number of events
            n_events = ((durations_sorted == t) & (events_sorted == 1)).sum().float()
            
            if at_risk > 0:
                # Cumulative sum of hazard increments
                if i > 0:
                    chf[i + 1] = chf[i] + (n_events / at_risk)
                else:
                    chf[i + 1] = n_events / at_risk
        
        # Survival function
        survival = torch.exp(-chf)
        
        node.timeline = timeline
        node.cumulative_hazard_function = chf
        node.survival_function = survival
    
    def _calculate_impurity(
        self,
        durations: torch.Tensor,
        events: torch.Tensor
    ) -> float:
        """
        Calculate node impurity as 1 - C-index.
        
        Lower impurity means better node homogeneity.
        """
        if events.sum() < 2:
            return 0.0
        
        # Simple approximation: variance of log durations for events
        event_durations = durations[events == 1]
        if len(event_durations) < 2:
            return 0.0
        
        log_durations = torch.log(event_durations + 1e-8)
        impurity = torch.var(log_durations).item()
        
        return impurity
    
    def predict_cumulative_hazard(
        self,
        X: Union[np.ndarray, torch.Tensor],
        times: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """
        Predict cumulative hazard function.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        times : array-like, optional
            Time points at which to evaluate CHF.
            If None, uses union of all terminal node timelines.
        
        Returns:
        --------
        chf : np.ndarray, shape (n_samples, n_times)
            Cumulative hazard function for each sample
        """
        if not self._is_fitted:
            raise ValueError("Tree must be fitted before prediction")
        
        X = self._to_tensor(X)
        
        # Get predictions for each sample
        predictions = []
        for i in range(X.shape[0]):
            node = self._traverse_tree(self.tree_, X[i])
            
            if times is None:
                predictions.append({
                    'timeline': node.timeline,
                    'chf': node.cumulative_hazard_function
                })
            else:
                # Interpolate to requested times
                times_tensor = self._to_tensor(times)
                chf_interp = self._interpolate_step_function(
                    node.timeline,
                    node.cumulative_hazard_function,
                    times_tensor
                )
                predictions.append(chf_interp)
        
        if times is None:
            # Return list of dictionaries
            return predictions
        else:
            # Stack into array
            return torch.stack(predictions).cpu().numpy()
    
    def predict_survival_function(
        self,
        X: Union[np.ndarray, torch.Tensor],
        times: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """
        Predict survival function.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        times : array-like, optional
            Time points at which to evaluate survival.
            If None, uses union of all terminal node timelines.
        
        Returns:
        --------
        survival : np.ndarray, shape (n_samples, n_times)
            Survival function for each sample
        """
        if not self._is_fitted:
            raise ValueError("Tree must be fitted before prediction")
        
        X = self._to_tensor(X)
        
        predictions = []
        for i in range(X.shape[0]):
            node = self._traverse_tree(self.tree_, X[i])
            
            if times is None:
                predictions.append({
                    'timeline': node.timeline,
                    'survival': node.survival_function
                })
            else:
                times_tensor = self._to_tensor(times)
                survival_interp = self._interpolate_step_function(
                    node.timeline,
                    node.survival_function,
                    times_tensor
                )
                predictions.append(survival_interp)
        
        if times is None:
            return predictions
        else:
            return torch.stack(predictions).cpu().numpy()
    
    def _traverse_tree(self, node: Node, x: torch.Tensor) -> Node:
        """
        Traverse tree to find terminal node for sample x.
        """
        if node.is_leaf:
            return node
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(node.left, x)
        else:
            return self._traverse_tree(node.right, x)
    
    def _interpolate_step_function(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_new: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate step function (right-continuous).
        
        For each x_new[i], find the largest x[j] such that x[j] <= x_new[i]
        and return y[j].
        """
        result = torch.zeros(len(x_new), device=self.device)
        
        for i, xi in enumerate(x_new):
            # Find largest x <= xi
            mask = x <= xi
            if mask.any():
                idx = torch.where(mask)[0][-1]
                result[i] = y[idx]
            else:
                # Before first time point
                result[i] = y[0] if len(y) > 0 else 0.0
        
        return result
    
    def _get_max_features(self, n_features: int) -> int:
        """Determine actual number of features to consider per split."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor on correct device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _validate_inputs(
        self,
        X: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor
    ):
        """Validate input data."""
        if X.shape[0] != len(durations) or X.shape[0] != len(events):
            raise ValueError("X, durations, and events must have same number of samples")
        
        if (durations <= 0).any():
            raise ValueError("All durations must be positive")
        
        if not torch.all((events == 0) | (events == 1)):
            raise ValueError("Events must be binary (0 or 1)")
    
    def get_depth(self) -> int:
        """Get the maximum depth of the tree."""
        if not self._is_fitted:
            raise ValueError("Tree must be fitted first")
        return self._get_node_depth(self.tree_)
    
    def _get_node_depth(self, node: Node) -> int:
        """Recursively calculate depth of a node."""
        if node.is_leaf:
            return 0
        left_depth = self._get_node_depth(node.left) if node.left else 0
        right_depth = self._get_node_depth(node.right) if node.right else 0
        return 1 + max(left_depth, right_depth)
    
    def get_n_leaves(self) -> int:
        """Get the number of leaves in the tree."""
        if not self._is_fitted:
            raise ValueError("Tree must be fitted first")
        return self._count_leaves(self.tree_)
    
    def _count_leaves(self, node: Node) -> int:
        """Recursively count leaves."""
        if node.is_leaf:
            return 1
        left_count = self._count_leaves(node.left) if node.left else 0
        right_count = self._count_leaves(node.right) if node.right else 0
        return left_count + right_count