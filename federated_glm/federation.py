# federated_glm/federation.py
import numpy as np
from typing import List, Tuple, Dict, Any
from .core import FederatedGLM

class FederatedLearningManager:
    """Manages the federated learning process"""
    
    def __init__(self, aggregation_method='average'):
        self.aggregation_method = aggregation_method
        self.global_model = None
        self.history = []
    
    def fit(self, client_data: List[Tuple], family, n_rounds=10, 
            alpha=0.01, L1_wt=0.5, rho=0.1, method="elastic_net", 
            convergence_tol=1e-6, verbose=False):
        """
        Perform federated learning
        
        Parameters:
        -----------
        client_data : List of (X, y) tuples for each client
        family : statsmodels family object
        n_rounds : int, number of federation rounds
        """
        n_features = client_data[0][0].shape[1]
        self.global_model = np.zeros(n_features)
        
        for round_num in range(n_rounds):
            local_models = []
            
            # Train local models
            for X_client, y_client in client_data:
                model = FederatedGLM(y_client, X_client, family=family)
                result = model.fit_proximal(
                    alpha=alpha, L1_wt=L1_wt, rho=rho,
                    prox_center=self.global_model, method=method,
                    max_steps=1, verbose=verbose
                )
                local_models.append(result.params)
            
            # Aggregate models
            prev_global = self.global_model.copy()
            self.global_model = self._aggregate_models(local_models, client_data)
            
            # Check convergence
            change = np.linalg.norm(self.global_model - prev_global)
            self.history.append({
                'round': round_num,
                'global_model': self.global_model.copy(),
                'change': change
            })
            
            if verbose:
                print(f"Round {round_num}: Change = {change:.6f}")
                
            if change < convergence_tol:
                if verbose:
                    print(f"Converged after {round_num + 1} rounds")
                break
        
        return self
    
    def _aggregate_models(self, local_models: List[np.ndarray], 
                         client_data: List[Tuple]) -> np.ndarray:
        """Aggregate local models into global model"""
        if self.aggregation_method == 'average':
            return np.mean(local_models, axis=0)
        elif self.aggregation_method == 'weighted':
            weights = [len(data[1]) for data in client_data]  # Weight by sample size
            weights = np.array(weights) / sum(weights)
            return np.average(local_models, weights=weights, axis=0)
        else:
            raise ValueError("Aggregation method must be 'average' or 'weighted'")
    
    def predict(self, X, family):
        """Make predictions using the global model"""
        if self.global_model is None:
            raise RuntimeError("Model must be fitted before prediction")
        return family.link.inverse(X @ self.global_model)