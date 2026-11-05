"""
Federated GLM - A library for federated learning with Generalized Linear Models
"""

__version__ = "0.3.0"
__author__ = "Mohammad Amini"
__email__ = "m.amini@ufl.edu"

from .core import FederatedGLM
from .federation import FederatedLearningManager
from .evaluation import ModelEvaluator
from .utils import DataGenerator

from .personalized import PersonalizedFederatedGLM

__all__ = [
    'FederatedGLM',
    'FederatedLearningManager', 
    'ModelEvaluator',
    'DataGenerator',
    'PersonalizedFederatedGLM'
]