import numpy as np
from typing import Dict, List, Tuple, Optional
from .core import FederatedGLM

class PersonalizedFederatedGLM:
    """
    Personalized Federated Learning for GLMs
    
    Supports multiple personalization strategies:
    - pFedMe: Personalized Federated Meta-Learning
    - Per-FedAvg: Personalized FedAvg with fine-tuning
    - Local Adaptation: Interpolation between global and local models
    """
    
    def __init__(self, method: str = 'pfedme', lambda_reg: float = 0.1):
        """
        Initialize personalized federated learning
        
        Parameters:
        -----------
        method : str
            Personalization method ('pfedme', 'perfedavg', 'local_adaptation')
        lambda_reg : float
            Regularization strength for personalization
        """
        if method not in ['pfedme', 'perfedavg', 'local_adaptation']:
            raise ValueError("Method must be one of: 'pfedme', 'perfedavg', 'local_adaptation'")
            
        self.method = method
        self.lambda_reg = lambda_reg
        self.global_model = None
        self.client_models = {}
        self.history = []
    
    def fit(self, client_data: List[Tuple], family, n_rounds: int = 20, 
            alpha: float = 0.01, rho: float = 0.1, personalization_steps: int = 3,
            verbose: bool = False):
        """
        Train personalized federated models
        
        Parameters:
        -----------
        client_data : List[Tuple]
            List of (X, y) tuples for each client
        family : statsmodels.Family
            GLM family (Gaussian, Binomial, etc.)
        n_rounds : int
            Number of federated learning rounds
        alpha : float
            Regularization strength
        rho : float
            Proximal regularization strength
        personalization_steps : int
            Number of local personalization steps
        verbose : bool
            Print training progress
        """
        n_features = client_data[0][0].shape[1]
        self.global_model = np.zeros(n_features)
        
        # Initialize client models
        for i in range(len(client_data)):
            self.client_models[i] = np.zeros(n_features)
        
        for round_num in range(n_rounds):
            if verbose:
                print(f"Round {round_num + 1}/{n_rounds}")
            
            if self.method == 'pfedme':
                self._pfedme_round(client_data, family, alpha, rho, personalization_steps)
            elif self.method == 'perfedavg':
                self._perfedavg_round(client_data, family, alpha, rho, personalization_steps)
            elif self.method == 'local_adaptation':
                self._local_adaptation_round(client_data, family, alpha, rho)
            
            # Track convergence
            round_info = {
                'round': round_num,
                'global_model': self.global_model.copy(),
                'client_models': {k: v.copy() for k, v in self.client_models.items()}
            }
            self.history.append(round_info)
        
        return self
    
    def _pfedme_round(self, client_data, family, alpha, rho, K_steps):
        """pFedMe: Personalized Federated Meta-Learning round"""
        local_updates = []
        
        for i, (X_client, y_client) in enumerate(client_data):
            theta_i = self.global_model.copy()
            
            # Local personalization steps
            for k in range(K_steps):
                local_model = FederatedGLM(y_client, X_client, family=family)
                local_result = local_model.fit_proximal(
                    alpha=alpha,
                    rho=self.lambda_reg,
                    prox_center=self.global_model,
                    method="elastic_net",
                    max_steps=1
                )
                theta_i = local_result.params
            
            self.client_models[i] = theta_i
            local_updates.append(theta_i - self.global_model)
        
        # Update global model
        avg_update = np.mean(local_updates, axis=0)
        self.global_model += 0.1 * avg_update
    
    def _perfedavg_round(self, client_data, family, alpha, rho, K_steps):
        """Per-FedAvg: Standard FedAvg + local fine-tuning"""
        # Step 1: Standard federated averaging
        local_models = []
        
        for i, (X_client, y_client) in enumerate(client_data):
            local_model = FederatedGLM(y_client, X_client, family=family)
            local_result = local_model.fit_proximal(
                alpha=alpha, rho=rho, prox_center=self.global_model,
                method="elastic_net", max_steps=1
            )
            local_models.append(local_result.params)
        
        # Update global model
        self.global_model = np.mean(local_models, axis=0)
        
        # Step 2: Local fine-tuning
        for i, (X_client, y_client) in enumerate(client_data):
            theta_i = self.global_model.copy()
            
            for k in range(K_steps):
                local_model = FederatedGLM(y_client, X_client, family=family)
                local_result = local_model.fit_proximal(
                    alpha=alpha * 0.1, rho=0.01, prox_center=theta_i,
                    method="elastic_net", max_steps=1
                )
                theta_i = local_result.params
            
            self.client_models[i] = theta_i
    
    def _local_adaptation_round(self, client_data, family, alpha, rho):
        """Local Adaptation: Interpolation between global and local models"""
        local_models = []
        
        for i, (X_client, y_client) in enumerate(client_data):
            local_model = FederatedGLM(y_client, X_client, family=family)
            local_result = local_model.fit_proximal(
                alpha=alpha, rho=rho, prox_center=self.global_model,
                method="elastic_net", max_steps=1
            )
            local_models.append(local_result.params)
        
        # Update global model
        self.global_model = np.mean(local_models, axis=0)
        
        # Personalized models as interpolation
        for i, local_model in enumerate(local_models):
            self.client_models[i] = 0.7 * self.global_model + 0.3 * local_model
    
    def predict(self, X, family, client_id: Optional[int] = None):
        """
        Make predictions using global or personalized model
        
        Parameters:
        -----------
        X : array-like
            Input features
        family : statsmodels.Family
            GLM family
        client_id : int, optional
            If provided, use personalized model for this client
            
        Returns:
        --------
        array : Predictions
        """
        if client_id is not None and client_id in self.client_models:
            # Use personalized model
            return family.link.inverse(X @ self.client_models[client_id])
        else:
            # Use global model
            return family.link.inverse(X @ self.global_model)
    
    def get_client_model(self, client_id: int):
        """Get personalized model for specific client"""
        if client_id in self.client_models:
            return self.client_models[client_id]
        else:
            raise ValueError(f"Client {client_id} not found")
    
    def get_global_model(self):
        """Get global model parameters"""
        return self.global_model