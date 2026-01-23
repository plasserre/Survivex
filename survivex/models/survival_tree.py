"""
Survival Tree Implementation - Optimized with Vectorized NumPy

Mathematical Foundation:
=======================

A survival tree is a decision tree for censored survival data that recursively
partitions the feature space to create homogeneous groups with respect to survival.

Splitting Criterion:
-------------------
Uses the log-rank test statistic to measure separation between left and right child nodes:

    χ² = (O_L - E_L)² / V

Where:
    O_L = Observed events in left group
    E_L = Expected events in left group (under null hypothesis)
    V = Variance of the statistic

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

References:
-----------
- LeBlanc, M., & Crowley, J. (1993). "Survival trees by goodness of split"
- Ishwaran, H., et al. (2008). "Random survival forests"
- Therneau, T., & Atkinson, E. (1997). "An introduction to recursive partitioning"

Implementation Details:
----------------------
- Uses vectorized numpy for fast CPU computation
- Sorted-scan approach for finding best splits (O(n log n) per feature)
- Efficient Nelson-Aalen using numpy bincount/cumsum
- Feature importance based on split improvement
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class Node:
    """Node in the survival tree."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    cumulative_hazard_function: Optional[np.ndarray] = None
    survival_function: Optional[np.ndarray] = None
    timeline: Optional[np.ndarray] = None
    n_samples: int = 0
    n_events: int = 0
    impurity: float = 0.0
    is_leaf: bool = False


class SurvivalTree:
    """
    Survival Tree for right-censored data.

    A decision tree that uses the log-rank test statistic as splitting criterion
    and Nelson-Aalen estimator for terminal node predictions.

    Uses vectorized numpy for fast computation.

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
        Kept for API compatibility (always uses CPU numpy)
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
        self.device = device  # kept for API compatibility

        if random_state is not None:
            np.random.seed(random_state)

        self.tree_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.max_features_ = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> 'SurvivalTree':
        """
        Build the survival tree.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        durations : array-like, shape (n_samples,)
        events : array-like, shape (n_samples,)
        """
        # Convert to numpy
        X = np.asarray(X, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=np.float64)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.max_features_ = self._get_max_features(n_features)
        self.feature_importances_ = np.zeros(n_features)

        # Pre-sort data by duration for efficient splitting
        sort_idx = np.argsort(durations)
        X_sorted = X[sort_idx]
        durations_sorted = durations[sort_idx]
        events_sorted = events[sort_idx]

        # Build tree
        indices = np.arange(n_samples)
        self.tree_ = self._build_tree(X_sorted, durations_sorted, events_sorted, indices, depth=0)

        # Normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        self._is_fitted = True
        return self

    def _build_tree(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        indices: np.ndarray,
        depth: int
    ) -> Node:
        """Recursively build the tree."""
        n_samples = len(indices)
        X_node = X[indices]
        dur_node = durations[indices]
        evt_node = events[indices]
        n_events = int(evt_node.sum())

        node = Node(n_samples=n_samples, n_events=n_events)

        # Stopping criteria
        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth) or
            (n_samples < self.min_samples_split) or
            (n_events == 0) or
            (n_events == n_samples)
        )

        if should_stop:
            node.is_leaf = True
            self._fit_terminal_node(node, dur_node, evt_node)
            return node

        # Find best split
        best_split = self._find_best_split(X_node, dur_node, evt_node)

        if best_split is None or best_split['statistic'] < self.min_impurity_decrease:
            node.is_leaf = True
            self._fit_terminal_node(node, dur_node, evt_node)
            return node

        # Update feature importance
        self.feature_importances_[best_split['feature']] += best_split['statistic']

        # Split
        feature_vals = X_node[:, best_split['feature']]
        left_mask = feature_vals <= best_split['threshold']
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.is_leaf = True
            self._fit_terminal_node(node, dur_node, evt_node)
            return node

        node.feature = best_split['feature']
        node.threshold = best_split['threshold']

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        node.left = self._build_tree(X, durations, events, left_indices, depth + 1)
        node.right = self._build_tree(X, durations, events, right_indices, depth + 1)

        return node

    def _find_best_split(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Find best split using vectorized log-rank scanning.

        For each feature, sorts data and scans through possible split points,
        computing log-rank statistics efficiently using running sums.
        """
        n_samples, n_features = X.shape

        # Select features to try
        if self.max_features_ < n_features:
            features = np.random.choice(n_features, self.max_features_, replace=False)
        else:
            features = np.arange(n_features)

        best_split = None
        best_statistic = -np.inf

        # Pre-compute global stats for log-rank using a coarse time grid
        # For efficiency, use at most MAX_GRID event times (quantiles)
        MAX_GRID = 30
        event_mask = events == 1
        event_times_all = durations[event_mask]

        if len(event_times_all) == 0:
            return None

        unique_event_times = np.unique(event_times_all)

        # Use coarse grid if too many unique times
        if len(unique_event_times) > MAX_GRID:
            quantiles = np.linspace(0, 100, MAX_GRID + 2)[1:-1]
            grid_times = np.percentile(event_times_all, quantiles)
            grid_times = np.unique(grid_times)
        else:
            grid_times = unique_event_times

        n_grid = len(grid_times)

        # For each grid time, compute: events and at-risk
        # at_risk[k] = number of samples with duration >= grid_times[k]
        sorted_dur = np.sort(durations)
        at_risk = n_samples - np.searchsorted(sorted_dur, grid_times, side='left')
        at_risk = at_risk.astype(np.float64)

        # events at each grid time: sum events in [grid_times[k], grid_times[k+1])
        # Use bins defined by grid_times
        events_at_time = np.zeros(n_grid)
        # Assign each event to its nearest grid time
        event_grid_idx = np.searchsorted(grid_times, event_times_all, side='right') - 1
        event_grid_idx = np.clip(event_grid_idx, 0, n_grid - 1)
        np.add.at(events_at_time, event_grid_idx, 1.0)

        # Assign all samples to grid buckets for at-risk tracking
        # time_inverse[i] = grid bucket index for sample i
        time_inverse = np.searchsorted(grid_times, durations, side='right') - 1
        time_inverse = np.clip(time_inverse, 0, n_grid - 1)
        n_times = n_grid

        for feature_idx in features:
            stat, threshold = self._best_split_for_feature(
                X[:, feature_idx], durations, events,
                grid_times, time_inverse, events_at_time, at_risk, n_times
            )

            if stat > best_statistic:
                best_statistic = stat
                best_split = {
                    'feature': feature_idx,
                    'threshold': threshold,
                    'statistic': stat,
                    'improvement': stat
                }

        return best_split if best_statistic > 0 else None

    def _best_split_for_feature(
        self,
        feature_values: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        unique_times: np.ndarray,
        time_inverse: np.ndarray,
        events_at_time: np.ndarray,
        at_risk_total: np.ndarray,
        n_times: int
    ) -> Tuple[float, float]:
        """
        Find best threshold for a single feature using fully vectorized log-rank.

        All candidate thresholds are evaluated simultaneously using matrix operations.
        """
        n_samples = len(feature_values)

        # Sort by feature value
        sort_idx = np.argsort(feature_values)
        sorted_features = feature_values[sort_idx]
        sorted_time_idx = time_inverse[sort_idx]
        sorted_events = events[sort_idx]

        # Pre-compute valid mask (times with events and at least 2 at risk)
        valid = (events_at_time > 0) & (at_risk_total > 1)
        if not valid.any():
            return -np.inf, 0.0
        valid_idx = np.where(valid)[0]
        n_valid = len(valid_idx)

        events_valid = events_at_time[valid_idx]
        at_risk_valid = at_risk_total[valid_idx]

        # Pre-compute variance denominator
        var_denom = at_risk_valid ** 2 * (at_risk_valid - 1)
        safe_var = var_denom > 0
        if not safe_var.any():
            return -np.inf, 0.0

        # Determine candidate split positions
        max_thresholds = min(80, n_samples - 2 * self.min_samples_leaf)
        if max_thresholds <= 0:
            return -np.inf, 0.0

        # Find positions where feature value changes
        change_mask = sorted_features[1:] != sorted_features[:-1]
        change_positions = np.where(change_mask)[0] + 1  # position of first element in new group

        if len(change_positions) == 0:
            return -np.inf, 0.0

        # Filter by min_samples_leaf
        valid_mask = (change_positions >= self.min_samples_leaf) & \
                     (change_positions <= n_samples - self.min_samples_leaf)
        change_positions = change_positions[valid_mask]

        if len(change_positions) == 0:
            return -np.inf, 0.0

        # Limit number of candidates
        if len(change_positions) > max_thresholds:
            idx = np.linspace(0, len(change_positions) - 1, max_thresholds, dtype=int)
            change_positions = change_positions[idx]

        n_candidates = len(change_positions)

        # Build cumulative count and event matrices
        # count_matrix[i, t] = 1 if sorted sample i maps to time bin t
        # event_matrix[i, t] = event indicator for sorted sample i at time bin t
        count_matrix = np.zeros((n_samples, n_times))
        count_matrix[np.arange(n_samples), sorted_time_idx] = 1.0

        event_matrix = np.zeros((n_samples, n_times))
        event_matrix[np.arange(n_samples), sorted_time_idx] = sorted_events

        # Cumulative sums along sample axis
        # cum_count[p, t] = number of left-group samples at time t for split at position p
        cum_count = np.cumsum(count_matrix, axis=0)
        cum_events = np.cumsum(event_matrix, axis=0)

        # Extract values at candidate positions (position p means left = [0..p-1])
        # Index is p-1 since cumsum[p-1] = sum of [0..p-1]
        left_count_at_time = cum_count[change_positions - 1]  # (n_candidates, n_times)
        left_events_at_time = cum_events[change_positions - 1]  # (n_candidates, n_times)

        # Left at-risk: reverse cumsum over time axis
        # at_risk[t] = sum of counts for all t' >= t
        left_at_risk = np.cumsum(left_count_at_time[:, ::-1], axis=1)[:, ::-1]

        # Extract only valid time indices for log-rank
        left_at_risk_v = left_at_risk[:, valid_idx]  # (n_candidates, n_valid)
        left_events_v = left_events_at_time[:, valid_idx]  # (n_candidates, n_valid)

        # Expected events in left group: (left_at_risk / total_at_risk) * events
        expected_left = (left_at_risk_v / at_risk_valid[np.newaxis, :]) * events_valid[np.newaxis, :]

        # Observed - Expected (summed over time)
        obs_minus_exp = left_events_v.sum(axis=1) - expected_left.sum(axis=1)  # (n_candidates,)

        # Variance for each candidate
        right_at_risk_v = at_risk_valid[np.newaxis, :] - left_at_risk_v
        # Only compute at safe positions
        v_num = (left_at_risk_v[:, safe_var] * right_at_risk_v[:, safe_var] *
                 events_valid[safe_var][np.newaxis, :] *
                 (at_risk_valid[safe_var] - events_valid[safe_var])[np.newaxis, :])
        v_den = var_denom[safe_var][np.newaxis, :]
        variance_sum = (v_num / v_den).sum(axis=1)  # (n_candidates,)

        # Log-rank statistic
        valid_variance = variance_sum > 0
        if not valid_variance.any():
            return -np.inf, 0.0

        stats = np.full(n_candidates, -np.inf)
        stats[valid_variance] = (obs_minus_exp[valid_variance] ** 2) / variance_sum[valid_variance]

        # Best candidate
        best_idx = np.argmax(stats)
        best_stat = stats[best_idx]
        if best_stat <= 0:
            return -np.inf, 0.0

        pos = change_positions[best_idx]
        best_threshold = (sorted_features[pos - 1] + sorted_features[pos]) / 2.0

        return best_stat, best_threshold

    def _fit_terminal_node(
        self,
        node: Node,
        durations: np.ndarray,
        events: np.ndarray
    ):
        """Fit Nelson-Aalen estimator for terminal node (vectorized)."""
        # Get unique event times
        event_mask = events == 1
        event_times = durations[event_mask]

        if len(event_times) == 0:
            node.timeline = np.array([0.0, durations.max() if len(durations) > 0 else 1.0])
            node.cumulative_hazard_function = np.array([0.0, 0.0])
            node.survival_function = np.array([1.0, 1.0])
            return

        unique_times = np.unique(event_times)

        # Sort durations for searchsorted
        sorted_dur = np.sort(durations)
        n = len(durations)

        # At-risk at each unique time: n - number with duration < t
        at_risk = n - np.searchsorted(sorted_dur, unique_times, side='left')

        # Events at each unique time
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        events_count = np.zeros(len(unique_times))
        for t in event_times:
            events_count[time_to_idx[t]] += 1

        # Hazard increments
        hazard_increments = events_count / np.maximum(at_risk, 1).astype(np.float64)

        # Cumulative hazard (prepend 0 for time 0)
        chf = np.concatenate([[0.0], np.cumsum(hazard_increments)])
        timeline = np.concatenate([[0.0], unique_times])

        node.timeline = timeline
        node.cumulative_hazard_function = chf
        node.survival_function = np.exp(-chf)

    def predict_cumulative_hazard(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict cumulative hazard function.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        times : array-like, optional
            Time points for evaluation.

        Returns:
        --------
        chf : np.ndarray, shape (n_samples, n_times)
        """
        if not self._is_fitted:
            raise ValueError("Tree must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)

        if times is not None:
            times = np.asarray(times, dtype=np.float64)
            n_times = len(times)
            result = np.zeros((X.shape[0], n_times))

            for i in range(X.shape[0]):
                node = self._traverse_tree(self.tree_, X[i])
                # Interpolate step function
                result[i] = np.interp(times, node.timeline, node.cumulative_hazard_function,
                                       left=0.0, right=node.cumulative_hazard_function[-1])
            return result
        else:
            predictions = []
            for i in range(X.shape[0]):
                node = self._traverse_tree(self.tree_, X[i])
                predictions.append({
                    'timeline': node.timeline,
                    'chf': node.cumulative_hazard_function
                })
            return predictions

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict survival function.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        times : array-like, optional

        Returns:
        --------
        survival : np.ndarray, shape (n_samples, n_times)
        """
        if not self._is_fitted:
            raise ValueError("Tree must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)

        if times is not None:
            chf = self.predict_cumulative_hazard(X, times)
            return np.exp(-chf)
        else:
            predictions = []
            for i in range(X.shape[0]):
                node = self._traverse_tree(self.tree_, X[i])
                predictions.append({
                    'timeline': node.timeline,
                    'survival': node.survival_function
                })
            return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (sum of CHF)."""
        if not self._is_fitted:
            raise ValueError("Tree must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)
        risk_scores = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            node = self._traverse_tree(self.tree_, X[i])
            risk_scores[i] = node.cumulative_hazard_function[-1] if len(node.cumulative_hazard_function) > 0 else 0.0

        return risk_scores

    def _traverse_tree(self, node: Node, x: np.ndarray) -> Node:
        """Traverse tree to find terminal node for sample x."""
        if node.is_leaf:
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(node.left, x)
        else:
            return self._traverse_tree(node.right, x)

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

    def get_depth(self) -> int:
        """Get the maximum depth of the tree."""
        if not self._is_fitted:
            raise ValueError("Tree must be fitted first")
        return self._get_node_depth(self.tree_)

    def _get_node_depth(self, node: Node) -> int:
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
        if node.is_leaf:
            return 1
        left_count = self._count_leaves(node.left) if node.left else 0
        right_count = self._count_leaves(node.right) if node.right else 0
        return left_count + right_count
