# federated_glm/evaluation.py
import numpy as np
from scipy.special import xlogy
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, log_loss
from typing import Dict, Any

class ModelEvaluator:
    """Comprehensive model evaluation for federated GLM"""
    
    @staticmethod
    def evaluate(y_true, y_pred, family_name: str, y_prob=None) -> Dict[str, Any]:
        """
        Comprehensive evaluation metrics
        
        Parameters:
        -----------
        y_true : array, true values
        y_pred : array, predicted values
        family_name : str, GLM family name
        y_prob : array, predicted probabilities (for classification)
        """
        metrics = {}
        
        # Common metrics
        metrics['r2_score'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Family-specific metrics
        if family_name == "binomial":
            y_pred_binary = (y_pred >= 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(y_true.astype(int), y_pred_binary)
            
            if y_prob is not None:
                metrics['log_loss'] = log_loss(y_true.astype(int), y_prob)
                metrics['auc_roc'] = ModelEvaluator._compute_auc(y_true.astype(int), y_prob)
        
        elif family_name == "poisson":
            # Poisson-specific metrics
            metrics['poisson_deviance'] = ModelEvaluator._poisson_deviance(y_true, y_pred)
            
        return metrics
    
    @staticmethod
    def _compute_auc(y_true, y_prob):
        """Compute AUC-ROC score"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_prob)
        except ImportError:
            return None
    
    @staticmethod
    def _poisson_deviance(y_true, y_pred):
        """Poisson deviance, using 0 * log(0) = 0 for zero counts"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-10, None)  # Avoid log(0)
        return 2 * np.sum(xlogy(y_true, y_true / y_pred) - (y_true - y_pred))
    
    @staticmethod
    def compare_methods(results: Dict[str, Dict]) -> Dict:
        """Compare different training methods"""
        comparison = {}
        
        for metric in ['r2_score', 'mse', 'accuracy']:
            if metric in next(iter(results.values())):
                comparison[metric] = {
                    method: results[method][metric]
                    for method in results
                    if results[method].get(metric) is not None
                }
        
        return comparison