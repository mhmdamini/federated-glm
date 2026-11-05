"""
Example: personalized federated learning on heterogeneous clients.

Each client has its own true coefficients (``heterogeneity`` controls how far
apart they are), so a single global model fits nobody well. Every model is
scored on each client's own held-out data.
"""
import numpy as np
from federated_glm import (FederatedGLM, FederatedLearningManager,
                           PersonalizedFederatedGLM, DataGenerator)

N_ROUNDS = 20


def client_mse(predict, test):
    """Average over clients of the test MSE on that client's own data."""
    return np.mean([np.mean((y - predict(X, i)) ** 2) for i, (X, y) in enumerate(test)])


for heterogeneity in [0.0, 1.5]:
    train, test, family = DataGenerator.generate_federated_data(
        "gaussian", n_clients=5, n_per_client=120, p=5,
        heterogeneity=heterogeneity, seed=3)

    results = {}

    # One global model for everybody.
    fed = FederatedLearningManager().fit(train, family, n_rounds=N_ROUNDS)
    results["Federated (one global model)"] = client_mse(lambda X, i: fed.predict(X), test)

    # Each client alone, no collaboration.
    local = [FederatedGLM(y, X, family=family).fit_proximal(alpha=0.01, L1_wt=0.5).params
             for X, y in train]
    results["Local only"] = client_mse(lambda X, i: family.link.inverse(X @ local[i]), test)

    # Personalized federated learning.
    for method in ["pfedme", "perfedavg", "local_adaptation"]:
        model = PersonalizedFederatedGLM(method=method).fit(train, family, n_rounds=N_ROUNDS)
        results[f"Personalized ({method})"] = client_mse(
            lambda X, i: model.predict(X, client_id=i), test)

    print(f"\nHeterogeneity = {heterogeneity}  (average test MSE on each client's own data)")
    print("-" * 52)
    for name, mse in results.items():
        print(f"{name:<36} {mse:8.3f}")
