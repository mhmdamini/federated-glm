"""
Simple Example: Quick Start with Federated GLM
"""

from federated_glm import FederatedLearningManager, DataGenerator, ModelEvaluator

# 1. Generate data
X, y, family = DataGenerator.generate_glm_data("gaussian", n=200, p=3)

# 2. Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# 3. Partition among clients
client_data = DataGenerator.partition_data(X_train, y_train, n_clients=3)

# 4. Train federated model
manager = FederatedLearningManager()
manager.fit(client_data, family, n_rounds=10)

# 5. Make predictions
y_pred = manager.predict(X_test, family)

# 6. Evaluate
metrics = ModelEvaluator.evaluate(y_test, y_pred, "gaussian")

print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print("Success! 🎉")