"""
SurviveX: GPU-Accelerated Survival Analysis Library

A comprehensive survival analysis library with PyTorch backend for GPU acceleration.
Developed for regulatory application lifecycle prediction using hybrid process mining
and survival analysis frameworks.
"""

__version__ = "0.1.0"
__author__ = "Tanin Zeraati"

# Import main classes from models subdirectory
from survivex.models.kaplan_meier import KaplanMeierEstimator, KaplanMeierEstimatorWith100Points
from survivex.models.nelson_aalen import NelsonAalenEstimator
from survivex.models.log_rank_test import LogRankTest, LogRankResult
from survivex.models.cox_ph import CoxPHModel, CoxPHResult
from survivex.models.competing_risk import (
    AalenJohansenFitter,
    FineGrayModel,
    CIFResult,
    FineGrayResult
)
from survivex.models.multi_state import (
    MultiStateAalenJohansen,
    MultiStateCoxPH,
    TransitionMatrix,
    MultiStateData,
    AalenJohansenResult,
    TransitionCoxResult,
    MultiStateCoxResult
)
from survivex.models.parametric_models import (
    WeibullPHFitter,
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
    ExponentialFitter
)
from survivex.models.frailty import FrailtyModel, FrailtyResult

# Data utilities
from survivex.core.data import SurvivalData
from survivex.datasets.loaders import load_survival_dataset
from survivex.datasets.validators import validate_survival_data

# Define what's available when doing "from survivex import *"
__all__ = [
    # Non-parametric
    'KaplanMeierEstimator',
    'KaplanMeierEstimatorWith100Points',
    'NelsonAalenEstimator',
    'LogRankTest',
    'LogRankResult',
    
    # Semi-parametric
    'CoxPHModel',
    'CoxPHResult',
    
    # Competing risks
    'AalenJohansenFitter',
    'FineGrayModel',
    'CIFResult',
    'FineGrayResult',
    
    # Multi-state
    'MultiStateAalenJohansen',
    'MultiStateCoxPH',
    'TransitionMatrix',
    'MultiStateData',
    'AalenJohansenResult',
    'TransitionCoxResult',
    'MultiStateCoxResult',
    
    # Parametric
    'WeibullPHFitter',
    'WeibullAFTFitter',
    'LogNormalAFTFitter',
    'LogLogisticAFTFitter',
    'ExponentialFitter',

    # Frailty
    'FrailtyModel',
    'FrailtyResult',
]