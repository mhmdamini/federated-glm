# federated_glm/utils.py
import numpy as np
import statsmodels.api as sm
from typing import Tuple, List, Any

FAMILIES = {
    "gaussian": sm.families.Gaussian,
    "poisson": sm.families.Poisson,
    "binomial": sm.families.Binomial,
}


def _response(family: str, lin_pred, noise_level, rng):
    if family == "gaussian":
        return lin_pred + rng.randn(len(lin_pred)) * noise_level
    if family == "poisson":
        return rng.poisson(np.exp(lin_pred))
    return rng.binomial(1, 1 / (1 + np.exp(-lin_pred)))


class DataGenerator:
    """Synthetic data for GLM experiments.

    Every method draws from its own seeded generator, so calls are
    reproducible and leave NumPy's global random state untouched.
    """

    @staticmethod
    def generate_glm_data(family: str, n=300, p=5, seed=42,
                          noise_level=0.1) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Generate one data set ``(X, y, family)``; ``X`` includes a constant column."""
        if family not in FAMILIES:
            raise ValueError(f"Unsupported family: {family}")
        rng = np.random.RandomState(seed)
        X = rng.randn(n, p)
        beta = rng.randn(p)
        y = _response(family, X @ beta, noise_level, rng)
        return sm.add_constant(X), y, FAMILIES[family]()

    @staticmethod
    def generate_federated_data(family: str, n_clients=5, n_per_client=100, p=5,
                                heterogeneity=0.5, test_size=0.25, seed=42,
                                noise_level=0.5):
        """
        Generate clients whose true coefficients differ.

        Client i uses ``beta_i = beta + heterogeneity * N(0, I)``, so
        ``heterogeneity=0`` gives identically distributed clients.

        Returns
        -------
        train, test : lists of (X, y) tuples, one per client
        family : statsmodels family object
        """
        if family not in FAMILIES:
            raise ValueError(f"Unsupported family: {family}")
        rng = np.random.RandomState(seed)
        # Keep the linear predictor moderate so Poisson counts stay reasonable.
        scale = 1.0 if family == "gaussian" else 1.0 / np.sqrt(p + 1)
        beta = rng.randn(p + 1) * scale
        n_test = int(round(n_per_client * test_size))
        train, test = [], []
        for _ in range(n_clients):
            beta_i = beta + heterogeneity * scale * rng.randn(p + 1)
            X = sm.add_constant(rng.randn(n_per_client, p), has_constant="add")
            y = _response(family, X @ beta_i, noise_level, rng)
            train.append((X[n_test:], y[n_test:]))
            test.append((X[:n_test], y[:n_test]))
        return train, test, FAMILIES[family]()

    @staticmethod
    def partition_data(X, y, n_clients=3, method='random',
                       seed=42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition one data set among clients ('random' or 'sequential')."""
        n_samples = len(y)
        if method == 'random':
            indices = np.random.RandomState(seed).permutation(n_samples)
        elif method == 'sequential':
            indices = np.arange(n_samples)
        else:
            raise ValueError("Method must be 'random' or 'sequential'")
        return [(X[split], y[split]) for split in np.array_split(indices, n_clients)]
