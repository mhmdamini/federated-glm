# federated_glm/federation.py
import numpy as np
from typing import List, Tuple
from .core import FederatedGLM

AGGREGATIONS = ("average", "weighted")


class FederatedLearningManager:
    """Federated training of one global GLM (FedProx-style local updates).

    Each round, every client solves its penalized local problem with a
    proximal term ``rho / 2 * ||beta - global||^2`` (see
    ``FederatedGLM.fit_proximal``), starting from the current global model.
    The server then averages the local models, either equally
    (``"average"``) or by client sample size (``"weighted"``). Only model
    parameters leave the clients.
    """

    def __init__(self, aggregation_method='average'):
        if aggregation_method not in AGGREGATIONS:
            raise ValueError("Aggregation method must be 'average' or 'weighted'")
        self.aggregation_method = aggregation_method
        self.global_model = None
        self.family = None
        self.history = []

    def fit(self, client_data: List[Tuple], family, n_rounds=10,
            alpha=0.01, L1_wt=0.5, rho=0.1, method="elastic_net",
            convergence_tol=1e-6, verbose=False):
        """
        Perform federated learning.

        Parameters
        ----------
        client_data : list of (X, y) tuples, one per client
        family : statsmodels family object
        n_rounds : int, maximum number of federation rounds
        alpha, L1_wt, method : local penalty (see ``FederatedGLM.fit_proximal``)
        rho : float, strength of the proximal pull toward the global model
        convergence_tol : float, stop when the global model moves less than this
        verbose : bool, print the change in the global model each round
        """
        n_features = client_data[0][0].shape[1]
        self.family = family
        self.global_model = np.zeros(n_features)
        self.history = []

        for round_num in range(n_rounds):
            local_models = []
            for X_client, y_client in client_data:
                model = FederatedGLM(y_client, X_client, family=family)
                result = model.fit_proximal(
                    alpha=alpha, L1_wt=L1_wt, rho=rho,
                    prox_center=self.global_model, method=method,
                )
                local_models.append(result.params)

            prev_global = self.global_model.copy()
            self.global_model = self._aggregate_models(local_models, client_data)

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
        """Aggregate local models into the global model."""
        if self.aggregation_method == 'average':
            return np.mean(local_models, axis=0)
        weights = np.array([len(data[1]) for data in client_data], dtype=float)
        return np.average(local_models, weights=weights / weights.sum(), axis=0)

    def predict(self, X, family=None):
        """Predict the mean response with the global model.

        ``family`` defaults to the one passed to ``fit``.
        """
        if self.global_model is None:
            raise RuntimeError("Model must be fitted before prediction")
        family = family or self.family
        return family.link.inverse(X @ self.global_model)
