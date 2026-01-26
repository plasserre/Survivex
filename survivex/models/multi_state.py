
"""
Multi-State Survival Analysis - Base Data Structures

This module provides the foundation for multi-state survival analysis,
implementing data structures and utilities for working with multi-state models.

References:
- Andersen, P. K., & Keiding, N. (2002). Multi-state models for event history analysis.
- Putter, H., Fiocco, M., & Geskus, R. B. (2007). Tutorial in biostatistics: 
  Competing risks and multi-state models analyzed using the mstate package.
- R survival package and mstate package
"""

import numpy as np
import torch
from typing import Optional, Union, List, Dict, Tuple, Set
from dataclasses import dataclass
import warnings


@dataclass
class TransitionMatrix:
    """
    Represents the transition matrix for a multi-state model.
    
    The transition matrix defines which transitions are possible in the model.
    Entry (i,j) is non-zero if transition from state i to state j is possible.
    
    Attributes:
    -----------
    states : List[str]
        Names of all states in the model
    matrix : np.ndarray
        Transition matrix where entry (i,j) is the transition number if possible, 0 otherwise
    n_states : int
        Number of states
    n_transitions : int
        Number of possible transitions
    """
    states: List[str]
    matrix: np.ndarray
    
    def __post_init__(self):
        self.n_states = len(self.states)
        self.n_transitions = int(np.sum(self.matrix > 0))
        
    def get_transition_number(self, from_state: int, to_state: int) -> int:
        """Get transition number for state pair, or 0 if not possible."""
        return int(self.matrix[from_state, to_state])
    
    def is_absorbing(self, state: int) -> bool:
        """Check if a state is absorbing (no outgoing transitions)."""
        return np.sum(self.matrix[state, :]) == 0
    
    def get_possible_transitions(self, from_state: int) -> List[int]:
        """Get list of states that can be reached from given state."""
        return [j for j in range(self.n_states) if self.matrix[from_state, j] > 0]
    
    def __str__(self) -> str:
        result = f"Multi-State Transition Matrix\n"
        result += f"States: {self.states}\n"
        result += f"Number of states: {self.n_states}\n"
        result += f"Number of transitions: {self.n_transitions}\n\n"
        result += "Transition matrix:\n"
        
        # Header
        result += "     " + "  ".join([f"{s:>8}" for s in self.states]) + "\n"
        result += "     " + "-" * (10 * self.n_states) + "\n"
        
        # Rows
        for i, state in enumerate(self.states):
            result += f"{state:>8} |"
            for j in range(self.n_states):
                trans_num = int(self.matrix[i, j])
                if trans_num > 0:
                    result += f"{trans_num:>8}  "
                else:
                    result += "       .  "
            result += "\n"
        
        return result


def create_competing_risks_matrix(n_competing_events: int, 
                                  state_names: Optional[List[str]] = None) -> TransitionMatrix:
    """
    Create transition matrix for competing risks model.
    
    Competing risks has one initial state and multiple absorbing states.
    
    Parameters:
    -----------
    n_competing_events : int
        Number of competing event types
    state_names : List[str], optional
        Names for the states. If None, uses default names.
    
    Returns:
    --------
    TransitionMatrix
    """
    n_states = n_competing_events + 1  # Initial state + competing events
    
    if state_names is None:
        state_names = ["Initial"] + [f"Event{i+1}" for i in range(n_competing_events)]
    
    matrix = np.zeros((n_states, n_states))
    for i in range(1, n_states):
        matrix[0, i] = i  # Transitions from initial to each competing event
    
    return TransitionMatrix(states=state_names, matrix=matrix)


def create_illness_death_matrix(with_recovery: bool = False,
                                state_names: Optional[List[str]] = None) -> TransitionMatrix:
    """
    Create transition matrix for illness-death model.
    
    States:
    - 0: Healthy/Alive
    - 1: Illness
    - 2: Death
    
    Without recovery: 0 -> 1 -> 2 and 0 -> 2
    With recovery: Also allows 1 -> 0
    
    Parameters:
    -----------
    with_recovery : bool
        Whether to allow recovery (transition from illness back to healthy)
    state_names : List[str], optional
        Names for the states
    
    Returns:
    --------
    TransitionMatrix
    """
    if state_names is None:
        state_names = ["Healthy", "Illness", "Death"]
    
    matrix = np.zeros((3, 3))
    matrix[0, 1] = 1  # Healthy -> Illness
    matrix[0, 2] = 2  # Healthy -> Death
    matrix[1, 2] = 3  # Illness -> Death
    
    if with_recovery:
        matrix[1, 0] = 4  # Illness -> Healthy (recovery)
    
    return TransitionMatrix(states=state_names, matrix=matrix)


def create_progressive_matrix(n_stages: int,
                              allow_death_from_any: bool = True,
                              state_names: Optional[List[str]] = None) -> TransitionMatrix:
    """
    Create transition matrix for progressive disease model.
    
    Progressive model with sequential stages (e.g., disease progression).
    Can only move forward to next stage or to death.
    
    Parameters:
    -----------
    n_stages : int
        Number of disease stages (excluding death)
    allow_death_from_any : bool
        If True, death is possible from any stage
        If False, death only possible from final stage
    state_names : List[str], optional
        Names for the states
    
    Returns:
    --------
    TransitionMatrix
    """
    n_states = n_stages + 1  # Stages + death
    
    if state_names is None:
        state_names = [f"Stage{i+1}" for i in range(n_stages)] + ["Death"]
    
    matrix = np.zeros((n_states, n_states))
    
    trans_num = 1
    # Sequential progression
    for i in range(n_stages - 1):
        matrix[i, i + 1] = trans_num
        trans_num += 1
    
    # Death transitions
    if allow_death_from_any:
        for i in range(n_stages):
            matrix[i, n_states - 1] = trans_num
            trans_num += 1
    else:
        matrix[n_stages - 1, n_states - 1] = trans_num
    
    return TransitionMatrix(states=state_names, matrix=matrix)


@dataclass
class MultiStateData:
    """
    Container for multi-state survival data in long format.
    
    Each row represents a time interval for a subject during which they are
    at risk for specific transitions.
    
    Attributes:
    -----------
    subject_id : np.ndarray
        Subject identifier for each row
    from_state : np.ndarray
        Starting state for this interval
    to_state : np.ndarray
        Ending state (if event occurred) or current state (if censored)
    time_start : np.ndarray
        Start time of the interval
    time_stop : np.ndarray
        Stop time of the interval
    status : np.ndarray
        1 if transition occurred, 0 if censored
    transition_number : np.ndarray
        Transition number (from transition matrix) if event occurred
    covariates : Optional[np.ndarray]
        Covariate matrix (n_rows x n_covariates)
    """
    subject_id: np.ndarray
    from_state: np.ndarray
    to_state: np.ndarray
    time_start: np.ndarray
    time_stop: np.ndarray
    status: np.ndarray
    transition_number: np.ndarray
    covariates: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate data consistency."""
        n = len(self.subject_id)
        assert len(self.from_state) == n
        assert len(self.to_state) == n
        assert len(self.time_start) == n
        assert len(self.time_stop) == n
        assert len(self.status) == n
        assert len(self.transition_number) == n
        
        if self.covariates is not None:
            assert self.covariates.shape[0] == n
    
    @property
    def n_subjects(self) -> int:
        """Number of unique subjects."""
        return len(np.unique(self.subject_id))
    
    @property
    def n_observations(self) -> int:
        """Number of observations (rows)."""
        return len(self.subject_id)
    
    def filter_transition(self, transition_number: int) -> 'MultiStateData':
        """Filter data for a specific transition."""
        mask = self.transition_number == transition_number
        
        return MultiStateData(
            subject_id=self.subject_id[mask],
            from_state=self.from_state[mask],
            to_state=self.to_state[mask],
            time_start=self.time_start[mask],
            time_stop=self.time_stop[mask],
            status=self.status[mask],
            transition_number=self.transition_number[mask],
            covariates=self.covariates[mask] if self.covariates is not None else None
        )


def prepare_multistate_data_simple(durations: np.ndarray,
                                   events: np.ndarray,
                                   trans_matrix: TransitionMatrix,
                                   covariates: Optional[np.ndarray] = None) -> MultiStateData:
    """
    Prepare multi-state data from simple format (for competing risks or simple models).
    
    This is for the simple case where all subjects start in state 0 and can
    transition to any of the other states.
    
    Parameters:
    -----------
    durations : np.ndarray
        Time to event or censoring
    events : np.ndarray
        Event type (0 = censored, 1,2,... = event types corresponding to states 1,2,...)
    trans_matrix : TransitionMatrix
        Transition matrix defining the model
    covariates : np.ndarray, optional
        Covariate matrix
    
    Returns:
    --------
    MultiStateData
    """
    n = len(durations)
    
    # Create one row per subject per possible transition
    rows_per_subject = trans_matrix.n_transitions
    total_rows = n * rows_per_subject
    
    subject_id = np.repeat(np.arange(n), rows_per_subject)
    time_start = np.repeat(np.zeros(n), rows_per_subject)
    time_stop = np.repeat(durations, rows_per_subject)
    
    # For each subject, create rows for all possible transitions from state 0
    from_state = np.zeros(total_rows, dtype=int)
    to_state = np.zeros(total_rows, dtype=int)
    status = np.zeros(total_rows, dtype=int)
    transition_number = np.zeros(total_rows, dtype=int)
    
    trans_list = []
    for j in range(trans_matrix.n_states):
        trans_num = trans_matrix.get_transition_number(0, j)
        if trans_num > 0:
            trans_list.append((j, trans_num))
    
    for i in range(n):
        event_i = int(events[i])
        for idx, (to_state_j, trans_num) in enumerate(trans_list):
            row_idx = i * rows_per_subject + idx
            from_state[row_idx] = 0
            to_state[row_idx] = to_state_j
            transition_number[row_idx] = trans_num
            
            # Status = 1 only for the actual transition that occurred
            if event_i > 0 and to_state_j == event_i:
                status[row_idx] = 1
    
    # Expand covariates if provided
    if covariates is not None:
        covariates = np.repeat(covariates, rows_per_subject, axis=0)
    
    return MultiStateData(
        subject_id=subject_id,
        from_state=from_state,
        to_state=to_state,
        time_start=time_start,
        time_stop=time_stop,
        status=status,
        transition_number=transition_number,
        covariates=covariates
    )


def print_data_summary(data: MultiStateData, trans_matrix: TransitionMatrix):
    """Print summary statistics of multi-state data."""
    print("="*70)
    print("Multi-State Data Summary")
    print("="*70)
    print(f"Number of subjects: {data.n_subjects}")
    print(f"Number of observations: {data.n_observations}")
    print(f"\nTransition Matrix:")
    print(trans_matrix)
    
    print("\nEvent counts by transition:")
    print(f"{'Transition':<30} {'From':<10} {'To':<10} {'Events':<10}")
    print("-"*60)
    
    for from_s in range(trans_matrix.n_states):
        for to_s in range(trans_matrix.n_states):
            trans_num = trans_matrix.get_transition_number(from_s, to_s)
            if trans_num > 0:
                mask = (data.transition_number == trans_num) & (data.status == 1)
                n_events = np.sum(mask)
                trans_name = f"{trans_matrix.states[from_s]} -> {trans_matrix.states[to_s]}"
                print(f"{trans_name:<30} {from_s:<10} {to_s:<10} {n_events:<10}")
    
    print("="*70)




"""
Aalen-Johansen Estimator for General Multi-State Models

This extends beyond competing risks to handle arbitrary multi-state models
with intermediate transient states.

Mathematical Foundation:
------------------------
For a multi-state model with K states, the transition probability matrix P(s,t)
has entries:

P_hj(s,t) = P(X(t) = j | X(s) = h)

The Aalen-Johansen estimator computes this via the product integral:

P(s,t) = ∏_{s < u ≤ t} [I + dA(u)]

where dA(u) is the increment in the Nelson-Aalen transition intensity matrix.

References:
- Aalen, O. O., & Johansen, S. (1978). An empirical transition matrix for 
  non-homogeneous Markov chains based on censored observations.
- Andersen, P. K., Borgan, Ø., Gill, R. D., & Keiding, N. (1993). 
  Statistical models based on counting processes.
- R survival package survfit.coxph with multi-state
"""




@dataclass
class AalenJohansenResult:
    """
    Results from Aalen-Johansen estimation of transition probabilities.
    
    Attributes:
    -----------
    times : np.ndarray
        Time points at which probabilities are estimated
    transition_probs : Dict[Tuple[int,int], np.ndarray]
        Transition probabilities P_hj(0,t) for each (h,j) pair
        Key: (from_state, to_state), Value: probabilities at each time
    state_probs : Dict[int, np.ndarray]
        State occupation probabilities P_j(t) for each state j
        Only computed if initial state distribution is provided
    variance : Optional[Dict]
        Variance estimates (if calculated)
    """
    times: np.ndarray
    transition_probs: Dict[Tuple[int, int], np.ndarray]
    state_probs: Optional[Dict[int, np.ndarray]] = None
    variance: Optional[Dict] = None


class MultiStateAalenJohansen:
    """
    Aalen-Johansen Estimator for General Multi-State Models.
    
    Computes non-parametric estimates of:
    1. Transition probabilities P_hj(s,t)
    2. State occupation probabilities P_j(t)
    
    This is the general formulation that works for any multi-state model,
    including illness-death, progressive disease, and competing risks as
    special cases.
    
    Parameters:
    -----------
    trans_matrix : TransitionMatrix
        Defines the state space and possible transitions
    
    Examples:
    ---------
    >>> # Illness-death model
    >>> trans_matrix = create_illness_death_matrix(with_recovery=False)
    >>> ajest = MultiStateAalenJohansen(trans_matrix)
    >>> result = ajest.fit(ms_data)
    >>> # Get probability of being dead at time 100, starting from healthy
    >>> p_death = result.transition_probs[(0, 2)][times == 100]
    """
    
    def __init__(self, trans_matrix: TransitionMatrix):
        self.trans_matrix = trans_matrix
        self.n_states = trans_matrix.n_states
        
    def fit(self, 
            data: MultiStateData,
            start_state: int = 0,
            variance: bool = False) -> AalenJohansenResult:
        """
        Fit the Aalen-Johansen estimator to multi-state data.
        
        Parameters:
        -----------
        data : MultiStateData
            Multi-state data in long format
        start_state : int
            Initial state (assumes all subjects start here)
        variance : bool
            Whether to compute variance estimates (TODO)
        
        Returns:
        --------
        AalenJohansenResult
        """
        # Get all unique event times
        event_times = data.time_stop[data.status == 1]
        unique_times = np.sort(np.unique(event_times))
        unique_times = np.concatenate([[0], unique_times])  # Add time 0
        n_times = len(unique_times)
        
        # Initialize transition probability matrices
        # P[t][h,j] = P(X(t) = j | X(0) = h)
        P = np.zeros((n_times, self.n_states, self.n_states))
        
        # At time 0, identity matrix (stay in starting state)
        for h in range(self.n_states):
            P[0, h, h] = 1.0
        
        # Compute transition probability matrix at each time
        for t_idx in range(1, n_times):
            t = unique_times[t_idx]
            t_prev = unique_times[t_idx - 1]
            
            # Start with previous transition probabilities
            P[t_idx] = P[t_idx - 1].copy()
            
            # Compute increment in transition intensity matrix at time t
            dA = self._compute_transition_increment(data, t)
            
            # Update using product integral: P(t) = P(t-) * [I + dA(t)]
            # P_hj(t) = sum_k P_hk(t-) * delta_kj  where delta = I + dA
            delta = np.eye(self.n_states) + dA
            P[t_idx] = P[t_idx - 1] @ delta
        
        # Extract transition probabilities for each (from, to) pair
        transition_probs = {}
        for h in range(self.n_states):
            for j in range(self.n_states):
                transition_probs[(h, j)] = P[:, h, j]
        
        # Compute state occupation probabilities (starting from start_state)
        state_probs = {}
        for j in range(self.n_states):
            state_probs[j] = P[:, start_state, j]
        
        return AalenJohansenResult(
            times=unique_times,
            transition_probs=transition_probs,
            state_probs=state_probs,
            variance=None if not variance else {}
        )
    
    def _compute_transition_increment(self, 
                                     data: MultiStateData, 
                                     t: float) -> np.ndarray:
        """
        Compute increment in transition intensity matrix at time t.
        
        dA_hj(t) = (# transitions h→j at t) / (# at risk in state h just before t)
        
        For non-diagonal: dA_hj(t) = d_hj(t) / Y_h(t)
        For diagonal: dA_hh(t) = -sum_{j≠h} dA_hj(t)
        
        Parameters:
        -----------
        data : MultiStateData
        t : float
            Time point
        
        Returns:
        --------
        dA : np.ndarray, shape (n_states, n_states)
            Increment in transition intensity matrix
        """
        dA = np.zeros((self.n_states, self.n_states))
        
        for h in range(self.n_states):
            # Number at risk in state h just before time t
            # These are subjects who: (1) are in state h at this interval,
            # (2) have Tstart < t, (3) have Tstop >= t
            # NOTE: Count unique subjects, not rows (since there may be multiple
            # rows per subject for different possible transitions)
            at_risk_mask = (data.from_state == h) & (data.time_start < t) & (data.time_stop >= t)
            Y_h = len(np.unique(data.subject_id[at_risk_mask]))

            if Y_h == 0:
                continue
            
            # For each possible transition from h
            for j in range(self.n_states):
                if h == j:
                    continue  # Handle diagonal later
                
                trans_num = self.trans_matrix.get_transition_number(h, j)
                if trans_num == 0:
                    continue  # Not a possible transition
                
                # Number of transitions from h to j at exactly time t
                event_mask = (data.from_state == h) & (data.to_state == j) & \
                            (data.time_stop == t) & (data.status == 1)
                d_hj = np.sum(event_mask)
                
                dA[h, j] = d_hj / Y_h
            
            # Diagonal element: ensures rows sum to 0
            dA[h, h] = -np.sum(dA[h, :])
        
        return dA


def plot_state_probabilities(result: AalenJohansenResult, 
                             trans_matrix: TransitionMatrix,
                             title: str = "State Occupation Probabilities"):
    """
    Plot state occupation probabilities over time (stacked area plot).
    
    Parameters:
    -----------
    result : AalenJohansenResult
    trans_matrix : TransitionMatrix
    title : str
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for stacked plot
        times = result.times
        probs = np.zeros((len(times), trans_matrix.n_states))
        
        for j in range(trans_matrix.n_states):
            probs[:, j] = result.state_probs[j]
        
        # Create stacked area plot
        ax.stackplot(times, *[probs[:, j] for j in range(trans_matrix.n_states)],
                    labels=trans_matrix.states, alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")


def plot_transition_probabilities(result: AalenJohansenResult,
                                  trans_matrix: TransitionMatrix,
                                  from_state: int,
                                  title: Optional[str] = None):
    """
    Plot transition probabilities from a specific starting state.
    
    Parameters:
    -----------
    result : AalenJohansenResult
    trans_matrix : TransitionMatrix
    from_state : int
    title : str, optional
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = result.times
        for j in range(trans_matrix.n_states):
            probs = result.transition_probs[(from_state, j)]
            ax.step(times, probs, where='post', 
                   label=f"To {trans_matrix.states[j]}", linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Transition Probability from {trans_matrix.states[from_state]}')
        if title is None:
            title = f'Transition Probabilities from {trans_matrix.states[from_state]}'
        ax.set_title(title)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")


# File: msm_cox.py (FIXED - Handle CoxPHModel attributes correctly)

"""
Transition-Specific Cox Models for Multi-State Survival Analysis

Implements Cox proportional hazards models for each transition in a multi-state model.
Each transition h→j gets its own Cox model with potentially different covariate effects.

Mathematical Foundation:
------------------------
For a multi-state model with transitions 1, 2, ..., T, we fit separate Cox models:

λ_t(u|X) = λ_0,t(u) × exp(β_t' X)

where:
- λ_t(u|X) is the hazard for transition t at time u given covariates X
- λ_0,t(u) is the baseline hazard for transition t
- β_t are the coefficients specific to transition t

This allows different covariates to have different effects on different transitions.

References:
-----------
- Putter, H., Fiocco, M., & Geskus, R. B. (2007). Tutorial in biostatistics: 
  Competing risks and multi-state models. Statistics in Medicine.
- Andersen, P. K., & Keiding, N. (2002). Multi-state models for event history analysis.
- R mstate package: https://cran.r-project.org/web/packages/mstate/
"""

import numpy as np
import torch
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
import warnings

# Import our existing Cox implementation
from .cox_ph import CoxPHModel
from .multi_state import TransitionMatrix, MultiStateData


@dataclass
class TransitionCoxResult:
    """
    Results from Cox model fitted to a specific transition.
    
    Attributes:
    -----------
    transition_number : int
        Which transition this model corresponds to
    from_state : int
        Starting state
    to_state : int
        Ending state
    transition_name : str
        Human-readable transition name
    coefficients : np.ndarray
        Coefficient estimates
    standard_errors : np.ndarray
        Standard errors of coefficients
    hazard_ratios : np.ndarray
        Hazard ratios (exp(coefficients))
    z_scores : np.ndarray
        Z-scores for coefficients
    p_values : np.ndarray
        P-values for coefficients
    concordance_index : float
        C-index
    log_likelihood : float
        Final log-likelihood
    n_events : int
        Number of events for this transition
    n_at_risk : int
        Total number of observations at risk for this transition
    """
    transition_number: int
    from_state: int
    to_state: int
    transition_name: str
    coefficients: np.ndarray
    standard_errors: np.ndarray
    hazard_ratios: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    concordance_index: float
    log_likelihood: float
    n_events: int
    n_at_risk: int


@dataclass
class MultiStateCoxResult:
    """
    Results from fitting transition-specific Cox models.
    
    Attributes:
    -----------
    transition_results : Dict[int, TransitionCoxResult]
        Cox model results for each transition (key = transition_number)
    trans_matrix : TransitionMatrix
        The transition matrix defining the model structure
    covariate_names : Optional[List[str]]
        Names of covariates
    """
    transition_results: Dict[int, TransitionCoxResult]
    trans_matrix: TransitionMatrix
    covariate_names: Optional[List[str]] = None
    
    def summary(self) -> str:
        """Print summary of all transition-specific models."""
        result = "="*80 + "\n"
        result += "Multi-State Cox Model Results\n"
        result += "="*80 + "\n"
        result += f"Model structure: {self.trans_matrix.n_states} states, "
        result += f"{self.trans_matrix.n_transitions} transitions\n\n"
        
        for trans_num in sorted(self.transition_results.keys()):
            tres = self.transition_results[trans_num]
            result += f"\nTransition {trans_num}: {tres.transition_name}\n"
            result += f"  Events: {tres.n_events}, At risk: {tres.n_at_risk}\n"
            result += f"  Log-likelihood: {tres.log_likelihood:.4f}\n"
            result += f"  Concordance: {tres.concordance_index:.4f}\n"
            
            result += f"\n  {'Covariate':<20} {'Coef':<12} {'HR':<12} {'SE':<12} {'z':<10} {'p':<10}\n"
            result += "  " + "-"*76 + "\n"
            
            for i in range(len(tres.coefficients)):
                covar_name = self.covariate_names[i] if self.covariate_names else f"X{i+1}"
                coef = tres.coefficients[i]
                hr = tres.hazard_ratios[i]
                se = tres.standard_errors[i]
                z = tres.z_scores[i]
                p = tres.p_values[i]
                
                result += f"  {covar_name:<20} {coef:>11.4f} {hr:>11.4f} {se:>11.4f} "
                result += f"{z:>9.3f} {p:>9.4f}\n"
        
        result += "\n" + "="*80 + "\n"
        return result


class MultiStateCoxPH:
    """
    Transition-Specific Cox Proportional Hazards Models for Multi-State Analysis.
    
    Fits separate Cox models for each possible transition in a multi-state model.
    This allows different covariates to have different effects on different transitions.
    
    Parameters:
    -----------
    trans_matrix : TransitionMatrix
        Defines the state space and possible transitions
    tie_method : str
        Method for handling tied event times ('breslow' or 'efron')
    alpha : float
        Significance level for confidence intervals
    
    Examples:
    ---------
    >>> # Illness-death model
    >>> trans_matrix = create_illness_death_matrix()
    >>> mscox = MultiStateCoxPH(trans_matrix)
    >>> result = mscox.fit(ms_data)
    >>> print(result.summary())
    """
    
    def __init__(self,
                 trans_matrix: TransitionMatrix,
                 tie_method: str = 'efron',
                 alpha: float = 0.05,
                 max_iter: int = 50,
                 tol: float = 1e-6):
        """
        Initialize Multi-State Cox model.
        
        Parameters:
        -----------
        trans_matrix : TransitionMatrix
            The transition matrix defining possible transitions
        tie_method : str
            'breslow' or 'efron' for handling tied times
        alpha : float
            Significance level
        max_iter : int
            Maximum iterations for Newton-Raphson
        tol : float
            Convergence tolerance
        """
        self.trans_matrix = trans_matrix
        self.tie_method = tie_method
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
        self._is_fitted = False
        self.transition_models_ = {}  # Will store CoxPHModel for each transition
        
    def fit(self,
            data: MultiStateData,
            covariate_names: Optional[List[str]] = None) -> MultiStateCoxResult:
        """
        Fit transition-specific Cox models.
        
        For each transition in the multi-state model, fits a separate Cox PH model
        using only the data relevant to that transition.
        
        Parameters:
        -----------
        data : MultiStateData
            Multi-state data in long format with covariates
        covariate_names : List[str], optional
            Names for the covariates
        
        Returns:
        --------
        MultiStateCoxResult
            Results containing fitted models for all transitions
        """
        if data.covariates is None:
            raise ValueError("Covariates are required for Cox models")
        
        # Validate covariate names
        n_covariates = data.covariates.shape[1]
        if covariate_names is None:
            covariate_names = [f"X{i+1}" for i in range(n_covariates)]
        elif len(covariate_names) != n_covariates:
            raise ValueError(f"Expected {n_covariates} covariate names, got {len(covariate_names)}")
        
        transition_results = {}
        
        # Fit a separate Cox model for each transition
        for trans_num in range(1, self.trans_matrix.n_transitions + 1):
            # Filter data for this transition
            trans_data = data.filter_transition(trans_num)
            
            # Find which states this transition connects
            from_state, to_state = self._get_transition_states(trans_num)
            transition_name = f"{self.trans_matrix.states[from_state]} → {self.trans_matrix.states[to_state]}"
            
            # Count events and at-risk observations
            n_events = np.sum(trans_data.status)
            n_at_risk = len(trans_data.status)
            
            print(f"\nFitting Cox model for Transition {trans_num}: {transition_name}")
            print(f"  Events: {n_events}, At risk: {n_at_risk}")
            
            if n_events == 0:
                warnings.warn(f"No events for transition {trans_num}, skipping")
                continue
            
            # Fit Cox model for this transition
            # Use time_stop - time_start as the duration
            durations = trans_data.time_stop - trans_data.time_start
            events = trans_data.status
            X = trans_data.covariates
            
            cox_model = CoxPHModel(
                tie_method=self.tie_method,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                device='cpu'  # Use CPU for float64 precision
            )
            
            # Fit returns the model object with fitted attributes
            cox_model.fit(X, durations, events)
            
            # Store the model
            self.transition_models_[trans_num] = cox_model
            
            # Extract results from fitted model (attributes have underscores)
            trans_result = TransitionCoxResult(
                transition_number=trans_num,
                from_state=from_state,
                to_state=to_state,
                transition_name=transition_name,
                coefficients=cox_model.coefficients_,
                standard_errors=cox_model.standard_errors_,
                hazard_ratios=np.exp(cox_model.coefficients_),
                z_scores=cox_model.coefficients_ / cox_model.standard_errors_,
                p_values=self._compute_p_values(cox_model.coefficients_, cox_model.standard_errors_),
                concordance_index=cox_model.concordance_index_,
                log_likelihood=float(cox_model.log_likelihood_),
                n_events=int(n_events),
                n_at_risk=int(n_at_risk)
            )
            
            transition_results[trans_num] = trans_result
        
        self._is_fitted = True
        
        return MultiStateCoxResult(
            transition_results=transition_results,
            trans_matrix=self.trans_matrix,
            covariate_names=covariate_names
        )
    
    def _compute_p_values(self, coefficients: np.ndarray, standard_errors: np.ndarray) -> np.ndarray:
        """Compute p-values from z-scores."""
        try:
            from scipy.stats import norm
            z_scores = coefficients / standard_errors
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
            return p_values
        except ImportError:
            return np.full_like(coefficients, np.nan)
    
    def _get_transition_states(self, trans_num: int) -> Tuple[int, int]:
        """Find which states a transition number connects."""
        for i in range(self.trans_matrix.n_states):
            for j in range(self.trans_matrix.n_states):
                if self.trans_matrix.get_transition_number(i, j) == trans_num:
                    return (i, j)
        raise ValueError(f"Transition number {trans_num} not found in transition matrix")