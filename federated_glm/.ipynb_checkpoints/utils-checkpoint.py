# federated_glm/utils.py
import numpy as np
import statsmodels.api as sm
from typing import Tuple, List, Any

class DataGenerator:
    """Utility class for generating synthetic data"""
    
    @staticmethod
    def generate_glm_data(family: str, n=300, p=5, seed=42, 
                         noise_level=0.1) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Generate synthetic data for GLM families"""
        np.random.seed(seed)
        X = np.random.randn(n, p)
        beta = np.random.randn(p)
        lin_pred = X @ beta
        
        if family == "gaussian":
            y = lin_pred + np.random.randn(n) * noise_level
            fam = sm.families.Gaussian()
        elif family == "poisson":
            y = np.random.poisson(np.exp(lin_pred))
            fam = sm.families.Poisson()
        elif family == "binomial":
            prob = 1 / (1 + np.exp(-lin_pred))
            y = np.random.binomial(1, prob)
            fam = sm.families.Binomial()
        else:
            raise ValueError(f"Unsupported family: {family}")
        
        return sm.add_constant(X), y, fam
    
    @staticmethod
    def partition_data(X, y, n_clients=3, method='random', 
                      seed=42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition data among clients"""
        np.random.seed(seed)
        n_samples = len(y)
        
        if method == 'random':
            indices = np.random.permutation(n_samples)
            splits = np.array_split(indices, n_clients)
            return [(X[split], y[split]) for split in splits]
        
        elif method == 'sequential':
            splits = np.array_split(range(n_samples), n_clients)
            return [(X[split], y[split]) for split in splits]
        
        else:
            raise ValueError("Method must be 'random' or 'sequential'")