# federated_glm/core.py
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.base.elastic_net import RegularizedResults, RegularizedResultsWrapper
from statsmodels.genmod.generalized_linear_model import GLM

class FederatedGLM(GLM):
    """Federated GLM with proximal operator support"""
    
    def __init__(self, endog, exog, family=None, **kwargs):
        super().__init__(endog, exog, family, **kwargs)
        
    def fit_proximal(self, alpha=0.1, L1_wt=1.0, rho=0.0, prox_center=None,
                    start_params=None, maxiter=100, max_steps=None,
                    tol=1e-6, method="elastic_net", verbose=False):
        """
        Fit GLM with proximal penalty for federated learning
        
        Parameters:
        -----------
        alpha : float, penalty strength
        L1_wt : float, elastic net mixing parameter (0=ridge, 1=lasso)
        rho : float, proximal penalty strength
        prox_center : array, proximal center point (global model)
        method : str, regularization method ('ordinary', 'lasso', 'elastic_net')
        """
        if start_params is None:
            start_params = self.fit().params
        if prox_center is None:
            prox_center = np.zeros_like(start_params)
            
        nobs = self.endog.shape[0]
        epsilon = 1e-4
        
        def objective(beta):
            loglike_val = self.loglike(beta) / nobs
            penalty = self._compute_penalty(beta, alpha, L1_wt, method)
            prox_penalty = 0.5 * rho * np.sum((beta - prox_center) ** 2)
            return -loglike_val + penalty + prox_penalty
            
        def gradient(beta):
            grad_loglike = self.score(beta) / nobs
            grad_penalty = self._compute_penalty_grad(beta, alpha, L1_wt, method, epsilon)
            grad_prox = rho * (beta - prox_center)
            return -grad_loglike + grad_penalty + grad_prox
            
        result = minimize(objective, start_params, jac=gradient, method="L-BFGS-B",
                         options={"maxiter": maxiter, "gtol": tol, "disp": verbose})
        
        res = RegularizedResults(self, result.x)
        return RegularizedResultsWrapper(res)
    
    def _compute_penalty(self, beta, alpha, L1_wt, method):
        if method == "ordinary":
            return 0.0
        elif method == "lasso":
            return alpha * np.sum(np.abs(beta))
        elif method == "elastic_net":
            return alpha * (L1_wt * np.sum(np.abs(beta)) + 
                           (1 - L1_wt) * 0.5 * np.sum(beta**2))
        else:
            raise ValueError("Method must be 'ordinary', 'lasso', or 'elastic_net'")
    
    def _compute_penalty_grad(self, beta, alpha, L1_wt, method, epsilon):
        if method == "ordinary":
            return np.zeros_like(beta)
        elif method == "lasso":
            return alpha * beta / np.sqrt(beta**2 + epsilon)
        elif method == "elastic_net":
            return alpha * (L1_wt * beta / np.sqrt(beta**2 + epsilon) + 
                           (1 - L1_wt) * beta)