"""
Federated GLM - A library for federated learning with Generalized Linear Models
"""

__version__ = "0.2.0"  # Bump version for new features
__author__ = "Mohammad Amini"
__email__ = "m.amini@ufl.edu"

# Existing imports
from .core import FederatedGLM
from .federation import FederatedLearningManager
from .evaluation import ModelEvaluator
from .utils import DataGenerator

# NEW: Personalized federated learning
from .personalized import PersonalizedFederatedGLM

__all__ = [
    'FederatedGLM',
    'FederatedLearningManager', 
    'ModelEvaluator',
    'DataGenerator',
    'PersonalizedFederatedGLM'  # NEW
]