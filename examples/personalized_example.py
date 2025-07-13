"""
Example: Personalized Federated Learning with federated_glm
"""
from federated_glm import PersonalizedFederatedGLM, FederatedLearningManager, DataGenerator, ModelEvaluator
import numpy as np

# 1. Generate data with heterogeneity
X, y, family = DataGenerator.generate_glm_data("gaussian", n=200, p=3)

# 2. Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# 3. Partition among clients
client_data = DataGenerator.partition_data(X_train, y_train, n_clients=3)

# 4A. Train STANDARD federated model
print("=" * 50)
print("STANDARD FEDERATED LEARNING")
print("=" * 50)

manager = FederatedLearningManager()
manager.fit(client_data, family, n_rounds=10)

# Make global predictions
y_pred_global = manager.predict(X_test, family)
metrics_global = ModelEvaluator.evaluate(y_test, y_pred_global, "gaussian")

print(f"Global Model R² Score: {metrics_global['r2_score']:.4f}")
print(f"Global Model RMSE: {metrics_global['rmse']:.4f}")

# 4B. Train PERSONALIZED federated models
print("\n" + "=" * 50)
print("PERSONALIZED FEDERATED LEARNING")
print("=" * 50)

methods = ['pfedme', 'perfedavg', 'local_adaptation']

for method in methods:
    print(f"\n--- {method.upper()} ---")
    
    # Train personalized model
    personalized_manager = PersonalizedFederatedGLM(method=method, lambda_reg=0.1)
    personalized_manager.fit(client_data, family, n_rounds=10, verbose=False)
    
    # Method 1: Global personalized prediction (average across all clients)
    y_pred_personalized = personalized_manager.predict(X_test, family)
    metrics_personalized = ModelEvaluator.evaluate(y_test, y_pred_personalized, "gaussian")
    
    print(f"Global Personalized R² Score: {metrics_personalized['r2_score']:.4f}")
    print(f"Global Personalized RMSE: {metrics_personalized['rmse']:.4f}")
    print(f"Improvement over Standard FL: {metrics_personalized['r2_score'] - metrics_global['r2_score']:+.4f}")
    
    # Method 2: Client-specific predictions (if you know which client the test data belongs to)
    print("Client-specific predictions:")
    for client_id in range(3):
        y_pred_client = personalized_manager.predict(X_test, family, client_id=client_id)
        metrics_client = ModelEvaluator.evaluate(y_test, y_pred_client, "gaussian")
        print(f"  Client {client_id} R²: {metrics_client['r2_score']:.4f}")

# 5. Compare with Purely Local Training
print("\n" + "=" * 50)
print("PURELY LOCAL TRAINING (for comparison)")
print("=" * 50)

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import statsmodels.api as sm

local_predictions = []
for i, (X_client, y_client) in enumerate(client_data):
    # Train local model using sklearn (no federation)
    X_client_sklearn = X_client[:, 1:]  # Remove constant for sklearn
    ridge_model = Ridge(alpha=0.01)
    ridge_model.fit(X_client_sklearn, y_client)
    
    # Predict on test set
    X_test_sklearn = X_test[:, 1:]
    y_pred_local = ridge_model.predict(X_test_sklearn)
    local_predictions.append(y_pred_local)
    
    local_r2 = r2_score(y_test, y_pred_local)
    print(f"Client {i} Local R²: {local_r2:.4f}")

# Average local predictions
y_pred_local_avg = np.mean(local_predictions, axis=0)
metrics_local_avg = ModelEvaluator.evaluate(y_test, y_pred_local_avg, "gaussian")

print(f"Average Local R² Score: {metrics_local_avg['r2_score']:.4f}")
print(f"Average Local RMSE: {metrics_local_avg['rmse']:.4f}")

# 6. Summary Comparison
print("\n" + "=" * 60)
print("FINAL COMPARISON SUMMARY")
print("=" * 60)

results = [
    ("Standard Federated Learning", metrics_global['r2_score'], metrics_global['rmse']),
    ("Personalized FL (pFedMe)", 
     PersonalizedFederatedGLM(method='pfedme').fit(client_data, family, n_rounds=10).predict(X_test, family),
     ""), # Will compute separately
    ("Purely Local (Average)", metrics_local_avg['r2_score'], metrics_local_avg['rmse'])
]

# Quick re-evaluation for clean summary
pfedme = PersonalizedFederatedGLM(method='pfedme', lambda_reg=0.1)
pfedme.fit(client_data, family, n_rounds=10, verbose=False)
y_pred_pfedme = pfedme.predict(X_test, family)
metrics_pfedme = ModelEvaluator.evaluate(y_test, y_pred_pfedme, "gaussian")

print(f"{'Method':<30} {'R² Score':<10} {'RMSE':<10} {'vs Standard FL':<15}")
print("-" * 70)
print(f"{'Standard FL':<30} {metrics_global['r2_score']:<10.4f} {metrics_global['rmse']:<10.4f} {'baseline':<15}")
print(f"{'Personalized FL (pFedMe)':<30} {metrics_pfedme['r2_score']:<10.4f} {metrics_pfedme['rmse']:<10.4f} {metrics_pfedme['r2_score'] - metrics_global['r2_score']:+.4f}")
print(f"{'Purely Local (Average)':<30} {metrics_local_avg['r2_score']:<10.4f} {metrics_local_avg['rmse']:<10.4f} {metrics_local_avg['r2_score'] - metrics_global['r2_score']:+.4f}")

print("\nSuccess! 🎉")
print("\nKey Insights:")
print("- Personalized FL adapts to client heterogeneity")
print("- Can provide both global and client-specific predictions")
print("- Balances collaboration benefits with personalization")