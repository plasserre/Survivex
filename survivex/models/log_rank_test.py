import torch
import numpy as np
from typing import Optional, Union, List, Tuple
import warnings
from dataclasses import dataclass

@dataclass
class LogRankResult:
    """
    Results from a log-rank test.
    
    Attributes:
    -----------
    test_statistic : float
        The chi-square test statistic
    p_value : float
        The p-value from the chi-square distribution
    degrees_of_freedom : int
        Degrees of freedom (number of groups - 1)
    test_name : str
        Name of the test performed
    summary : dict
        Detailed summary statistics for each group
    """
    test_statistic: float
    p_value: float
    degrees_of_freedom: int
    test_name: str = "log-rank"
    summary: dict = None
    
    def __str__(self):
        """String representation of results."""
        s = f"\n{'='*60}\n"
        s += f"Log-Rank Test Results\n"
        s += f"{'='*60}\n"
        s += f"Test statistic: {self.test_statistic:.4f}\n"
        s += f"Degrees of freedom: {self.degrees_of_freedom}\n"
        s += f"p-value: {self.p_value:.6f}\n"
        
        if self.p_value < 0.001:
            s += f"Result: Highly significant (p < 0.001) ***\n"
        elif self.p_value < 0.01:
            s += f"Result: Very significant (p < 0.01) **\n"
        elif self.p_value < 0.05:
            s += f"Result: Significant (p < 0.05) *\n"
        else:
            s += f"Result: Not significant (p ≥ 0.05)\n"
        
        s += f"{'='*60}\n"
        return s


class LogRankTest:
    """
    Log-rank test for comparing survival distributions between groups.
    
    The log-rank test is a hypothesis test to compare the survival distributions 
    of two or more groups. It's a non-parametric test and appropriate when the 
    data are right-censored.
    
    The test statistic follows a chi-square distribution with (k-1) degrees of 
    freedom, where k is the number of groups.
    
    Mathematical formulation:
    ------------------------
    For each unique event time t_i:
    - O_j(t_i) = observed events in group j at time t_i
    - E_j(t_i) = expected events = n_j(t_i) × d(t_i) / n(t_i)
    - V_j(t_i) = variance component
    
    Test statistic: χ² = Σ_j [(O_j - E_j)² / V_j]
    
    References:
    -----------
    - Mantel, N. (1966). Evaluation of survival data and two new rank order statistics
    - Peto, R., & Peto, J. (1972). Asymptotically efficient rank invariant test procedures
    - lifelines.statistics.logrank_test
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Log-Rank test.
        
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
    
    def compare(self, 
                durations_A: Union[np.ndarray, torch.Tensor],
                events_A: Union[np.ndarray, torch.Tensor],
                durations_B: Union[np.ndarray, torch.Tensor],
                events_B: Union[np.ndarray, torch.Tensor],
                weights_A: Optional[Union[np.ndarray, torch.Tensor]] = None,
                weights_B: Optional[Union[np.ndarray, torch.Tensor]] = None) -> LogRankResult:
        """
        Compare survival distributions between two groups.
        
        Parameters:
        -----------
        durations_A : array-like
            Time to event or censoring for group A
        events_A : array-like
            Event indicator for group A (1 if event occurred, 0 if censored)
        durations_B : array-like
            Time to event or censoring for group B
        events_B : array-like
            Event indicator for group B
        weights_A : array-like, optional
            Sample weights for group A
        weights_B : array-like, optional
            Sample weights for group B
            
        Returns:
        --------
        LogRankResult
            Object containing test statistic, p-value, and summary statistics
        """
        # Convert to tensors
        durations_A = self._to_tensor(durations_A)
        events_A = self._to_tensor(events_A)
        durations_B = self._to_tensor(durations_B)
        events_B = self._to_tensor(events_B)
        
        if weights_A is None:
            weights_A = torch.ones_like(durations_A, device=self.device)
        else:
            weights_A = self._to_tensor(weights_A)
            
        if weights_B is None:
            weights_B = torch.ones_like(durations_B, device=self.device)
        else:
            weights_B = self._to_tensor(weights_B)
        
        # Combine all data
        all_durations = torch.cat([durations_A, durations_B])
        all_events = torch.cat([events_A, events_B])
        all_weights = torch.cat([weights_A, weights_B])
        
        # Group indicators (0 for A, 1 for B)
        group_indicators = torch.cat([
            torch.zeros(len(durations_A), device=self.device),
            torch.ones(len(durations_B), device=self.device)
        ])
        
        # Calculate log-rank statistic
        test_stat, observed, expected, variance = self._calculate_logrank_statistic(
            all_durations, all_events, all_weights, group_indicators
        )
        
        # Calculate p-value using chi-square distribution
        try:
            from scipy.stats import chi2
            # Convert tensor to CPU and extract scalar value for scipy
            test_stat_cpu = test_stat.cpu().item() if isinstance(test_stat, torch.Tensor) else test_stat
            p_value = 1 - chi2.cdf(test_stat_cpu, df=1)
        except ImportError:
            warnings.warn("scipy not available, p-value not calculated")
            p_value = None
        
        # Create summary
        summary = {
            'group_A': {
                'n': len(durations_A),
                'events': int(events_A.sum().cpu().item() if isinstance(events_A, torch.Tensor) else events_A.sum()),
                'observed': observed[0].cpu().item() if isinstance(observed, torch.Tensor) else observed[0],
                'expected': expected[0].cpu().item() if isinstance(expected, torch.Tensor) else expected[0]
            },
            'group_B': {
                'n': len(durations_B),
                'events': int(events_B.sum().cpu().item() if isinstance(events_B, torch.Tensor) else events_B.sum()),
                'observed': observed[1].cpu().item() if isinstance(observed, torch.Tensor) else observed[1],
                'expected': expected[1].cpu().item() if isinstance(expected, torch.Tensor) else expected[1]
            }
        }
        
        return LogRankResult(
            test_statistic=test_stat.cpu().item() if isinstance(test_stat, torch.Tensor) else test_stat,
            p_value=p_value,
            degrees_of_freedom=1,
            test_name="log-rank",
            summary=summary
        )
    
    def compare_multiple(self,
                        durations_list: List[Union[np.ndarray, torch.Tensor]],
                        events_list: List[Union[np.ndarray, torch.Tensor]],
                        weights_list: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
                        group_names: Optional[List[str]] = None) -> LogRankResult:
        """
        Compare survival distributions among multiple groups (k ≥ 2).
        
        Parameters:
        -----------
        durations_list : list of array-like
            List of duration arrays for each group
        events_list : list of array-like
            List of event indicator arrays for each group
        weights_list : list of array-like, optional
            List of weight arrays for each group
        group_names : list of str, optional
            Names for each group
            
        Returns:
        --------
        LogRankResult
            Object containing test statistic, p-value, and summary statistics
        """
        n_groups = len(durations_list)
        
        if n_groups < 2:
            raise ValueError("Must have at least 2 groups to compare")
        
        if len(events_list) != n_groups:
            raise ValueError("durations_list and events_list must have same length")
        
        # Convert all to tensors
        durations_tensors = [self._to_tensor(d) for d in durations_list]
        events_tensors = [self._to_tensor(e) for e in events_list]
        
        # Handle weights
        if weights_list is None:
            weights_tensors = [torch.ones_like(d, device=self.device) for d in durations_tensors]
        else:
            weights_tensors = [self._to_tensor(w) for w in weights_list]
        
        # Combine all data
        all_durations = torch.cat(durations_tensors)
        all_events = torch.cat(events_tensors)
        all_weights = torch.cat(weights_tensors)
        
        # Create group indicators
        group_indicators = torch.cat([
            torch.full((len(d),), i, device=self.device, dtype=torch.float32)
            for i, d in enumerate(durations_tensors)
        ])
        
        # Calculate log-rank statistic for multiple groups
        test_stat, observed, expected, variance = self._calculate_logrank_statistic(
            all_durations, all_events, all_weights, group_indicators, n_groups
        )
        
        # Calculate p-value
        try:
            from scipy.stats import chi2
            # Convert tensor to CPU and extract scalar value for scipy
            test_stat_cpu = test_stat.cpu().item() if isinstance(test_stat, torch.Tensor) else test_stat
            p_value = 1 - chi2.cdf(test_stat_cpu, df=n_groups - 1)
        except ImportError:
            warnings.warn("scipy not available, p-value not calculated")
            p_value = None
        
        # Create summary
        if group_names is None:
            group_names = [f"Group_{i}" for i in range(n_groups)]
        
        summary = {}
        for i, name in enumerate(group_names):
            summary[name] = {
                'n': len(durations_tensors[i]),
                'events': int(events_tensors[i].sum().cpu().item() if isinstance(events_tensors[i], torch.Tensor) else events_tensors[i].sum()),
                'observed': observed[i].cpu().item() if isinstance(observed, torch.Tensor) else observed[i],
                'expected': expected[i].cpu().item() if isinstance(expected, torch.Tensor) else expected[i]
            }
        
        return LogRankResult(
            test_statistic=test_stat.cpu().item() if isinstance(test_stat, torch.Tensor) else test_stat,
            p_value=p_value,
            degrees_of_freedom=n_groups - 1,
            test_name="log-rank",
            summary=summary
        )
    
    def _calculate_logrank_statistic(self,
                                     durations: torch.Tensor,
                                     events: torch.Tensor,
                                     weights: torch.Tensor,
                                     groups: torch.Tensor,
                                     n_groups: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the log-rank test statistic.
        
        For 2 groups: Uses standard O-E formula
        For k≥3 groups: Uses proper covariance matrix approach
        
        Returns:
        --------
        tuple of (test_statistic, observed, expected, variance)
        """
        if n_groups is None:
            n_groups = len(torch.unique(groups))
        
        # Get unique event times
        unique_times = torch.unique(durations[events == 1])
        
        # Initialize accumulators for each group
        observed = torch.zeros(n_groups, device=self.device)
        expected = torch.zeros(n_groups, device=self.device)
        
        # For covariance matrix calculation (multivariate case)
        # We'll accumulate covariance at each time point
        if n_groups > 2:
            covariance_matrix = torch.zeros((n_groups - 1, n_groups - 1), device=self.device)
        else:
            variance = torch.zeros(n_groups, device=self.device)
        
        # For each unique event time, calculate O, E, and V
        for t in unique_times:
            # At risk: all subjects with duration >= t
            at_risk_mask = durations >= t
            
            # Events at this time
            event_mask = (durations == t) & (events == 1)
            
            # Total at risk and events at this time
            n_at_risk_total = torch.sum(weights[at_risk_mask])
            n_events_total = torch.sum(weights[event_mask])
            
            if n_at_risk_total == 0 or n_events_total == 0:
                continue
            
            # Calculate O, E for each group at this time point
            n_g = torch.zeros(n_groups, device=self.device)
            o_g = torch.zeros(n_groups, device=self.device)
            
            for g in range(n_groups):
                group_mask = groups == g
                
                # Observed events in this group at this time
                group_event_mask = group_mask & event_mask
                o_g[g] = torch.sum(weights[group_event_mask])
                
                # At risk in this group at this time
                group_at_risk_mask = group_mask & at_risk_mask
                n_g[g] = torch.sum(weights[group_at_risk_mask])
            
            # Expected events for each group
            e_g = (n_g / n_at_risk_total) * n_events_total
            
            # Accumulate observed and expected
            observed += o_g
            expected += e_g
            
            # Calculate variance/covariance components
            if n_groups == 2:
                # Two-group case: simple variance
                if n_at_risk_total > 1:
                    v = (n_g[0] / n_at_risk_total) * \
                        (1 - n_g[0] / n_at_risk_total) * \
                        n_events_total * \
                        (n_at_risk_total - n_events_total) / \
                        (n_at_risk_total - 1)
                    variance[0] += v
            else:
                # Multi-group case: covariance matrix
                # Use first k-1 groups (last group is redundant)
                if n_at_risk_total > 1:
                    for i in range(n_groups - 1):
                        for j in range(n_groups - 1):
                            if i == j:
                                # Variance term
                                cov_ij = (n_g[i] / n_at_risk_total) * \
                                        (1 - n_g[i] / n_at_risk_total) * \
                                        n_events_total * \
                                        (n_at_risk_total - n_events_total) / \
                                        (n_at_risk_total - 1)
                            else:
                                # Covariance term
                                cov_ij = -(n_g[i] / n_at_risk_total) * \
                                        (n_g[j] / n_at_risk_total) * \
                                        n_events_total * \
                                        (n_at_risk_total - n_events_total) / \
                                        (n_at_risk_total - 1)
                            covariance_matrix[i, j] += cov_ij
        
        # Calculate test statistic
        if n_groups == 2:
            # Two-group case: χ² = (O₁ - E₁)² / V₁
            diff = observed[0] - expected[0]
            test_stat = (diff ** 2) / variance[0] if variance[0] > 0 else torch.tensor(0.0, device=self.device)
            variance_out = variance
        else:
            # Multi-group case: χ² = (O-E)^T * V^-1 * (O-E)
            # Use only first k-1 groups
            oe_diff = (observed - expected)[:-1]  # Exclude last group
            
            # Add small value to diagonal for numerical stability
            covariance_matrix += torch.eye(n_groups - 1, device=self.device) * 1e-10
            
            try:
                # Compute test statistic using matrix inverse
                cov_inv = torch.linalg.inv(covariance_matrix)
                test_stat = torch.matmul(torch.matmul(oe_diff.unsqueeze(0), cov_inv), 
                                        oe_diff.unsqueeze(1)).squeeze()
            except:
                # Fallback if matrix is singular
                warnings.warn("Covariance matrix singular, using simplified calculation")
                test_stat = torch.sum((observed - expected) ** 2 / (torch.abs(observed - expected) + 1e-10))
            
            # Return diagonal of covariance as variance for summary
            variance_out = torch.diag(covariance_matrix)
            # Pad with zero for last group
            variance_out = torch.cat([variance_out, torch.tensor([0.0], device=self.device)])
        
        return test_stat, observed, expected, variance_out
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input data to tensor and move to device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.tensor(data, device=self.device, dtype=torch.float32)


# Convenience function for quick two-group comparison
def logrank_test(durations_A: Union[np.ndarray, torch.Tensor],
                 events_A: Union[np.ndarray, torch.Tensor],
                 durations_B: Union[np.ndarray, torch.Tensor],
                 events_B: Union[np.ndarray, torch.Tensor],
                 weights_A: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 weights_B: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 device: Optional[str] = None) -> LogRankResult:
    """
    Convenience function to perform log-rank test between two groups.
    
    Parameters:
    -----------
    durations_A : array-like
        Time to event or censoring for group A
    events_A : array-like
        Event indicator for group A (1 if event occurred, 0 if censored)
    durations_B : array-like
        Time to event or censoring for group B
    events_B : array-like
        Event indicator for group B
    weights_A : array-like, optional
        Sample weights for group A
    weights_B : array-like, optional
        Sample weights for group B
    device : str, optional
        Device to run computations on
        
    Returns:
    --------
    LogRankResult
        Object containing test statistic, p-value, and summary statistics
        
    Example:
    --------
    >>> result = logrank_test(durations_treatment, events_treatment,
    ...                       durations_control, events_control)
    >>> print(result)
    >>> print(f"p-value: {result.p_value:.4f}")
    """
    lr = LogRankTest(device=device)
    return lr.compare(durations_A, events_A, durations_B, events_B, weights_A, weights_B)