"""
Federated GLM - A library for federated learning with Generalized Linear Models
"""

__version__ = "0.1.0"
__author__ = "Mohammad Amini"
__email__ = "m.amini@ufl.edu"

from .core import FederatedGLM
from .federation import FederatedLearningManager
from .evaluation import ModelEvaluator
from .utils import DataGenerator

__all__ = [
    'FederatedGLM',
    'FederatedLearningManager', 
    'ModelEvaluator',
    'DataGenerator'
]