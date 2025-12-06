"""
SurviveX: GPU-Accelerated Survival Analysis Library

A comprehensive survival analysis library with PyTorch backend for GPU acceleration.
"""

__version__ = "0.1.0"
__author__ = "Tanin Zeraati"

# Import main classes from models subdirectory
from survivex.models.kaplan_meier import KaplanMeier
from survivex.models.nelson_aalen import NelsonAalen
from survivex.models.log_rank_test import LogRankTest
from survivex.models.cox_ph import CoxPH
from survivex.models.competing_risk import AalenJohansen, FineGray, GraysTest
from survivex.models.multi_state import MultiState
from survivex.models.parametric_models import (
    WeibullModel, 
    ExponentialModel, 
    LogNormalModel, 
    LogLogisticModel
)

# Define what's available when doing "from survivex import *"
__all__ = [
    # Non-parametric
    'KaplanMeier',
    'NelsonAalen',
    'LogRankTest',
    
    # Semi-parametric
    'CoxPH',
    
    # Competing risks
    'AalenJohansen',
    'FineGray',
    'GraysTest',
    
    # Multi-state
    'MultiState',
    
    # Parametric
    'WeibullModel',
    'ExponentialModel',
    'LogNormalModel',
    'LogLogisticModel',
]