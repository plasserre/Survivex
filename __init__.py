"""
SurviveX: GPU-Accelerated Survival Analysis Library

A comprehensive survival analysis library with PyTorch backend for GPU acceleration.
Developed as part of research on regulatory application lifecycle prediction using
hybrid process mining and survival analysis frameworks.
"""

__version__ = "0.1.0"
__author__ = "Tanin Zeraati"
import os
import sys
project_root = os.path.dirname('survivex/core')
sys.path.insert(0, project_root)

# Import main classes
from .kaplan_meier import KaplanMeier
from .nelson_aalen import NelsonAalen
from .log_rank_test import LogRankTest
from .cox_ph import CoxPH
from .competing_risk import AalenJohansen, FineGray, GraysTest
from .multi_state import MultiState
from .parametric_models import (
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