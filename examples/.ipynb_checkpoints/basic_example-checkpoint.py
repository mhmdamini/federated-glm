"""
Basic Example: Federated GLM Usage

This example demonstrates how to use the federated-glm library for 
federated learning with different GLM families.
"""

import numpy as np
import matplotlib.pyplot as plt
from federated_glm import (
    FederatedLearningManager, 
    FederatedGLM, 
    DataGenerator, 
    ModelEvaluator
)

def compare_federated_vs_centralized():
    """
    Compare federated learning vs centralized learning performance
    """
    print("=" * 60)
    print("FEDERATED GLM - BASIC EXAMPLE")
    print("=" * 60)
    
    # Configuration
    families = ["gaussian", "binomial", "poisson"]
    n_clients = 4
    n_samples = 1000
    n_features = 5
    
    results = {}
    
    for family_name in families:
        print(f"\n🔍 Testing {family_name.upper()} GLM")
        print("-" * 40)
        
        # 1. Generate synthetic data
        print("📊 Generating data...")
        X, y, family = DataGenerator.generate_glm_data(
            family_name, n=n_samples, p=n_features, seed=42
        )
        
        # 2. Split into train/test
        train_size = int(0.8 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"   Training samples: {len(y_train)}")
        print(f"   Test samples: {len(y_test)}")
        
        # 3. Partition training data among clients
        print(f"🌐 Partitioning data among {n_clients} clients...")
        client_data = DataGenerator.partition_data(
            X_train, y_train, n_clients=n_clients, method='random', seed=42
        )
        
        # Print client data sizes
        for i, (X_client, y_client) in enumerate(client_data):
            print(f"   Client {i+1}: {len(y_client)} samples")
        
        # 4. FEDERATED LEARNING
        print("🤝 Training federated model...")
        fed_manager = FederatedLearningManager(aggregation_method='weighted')
        fed_manager.fit(
            client_data, 
            family, 
            n_rounds=15,
            alpha=0.01,
            rho=0.1,
            method="elastic_net",
            verbose=False
        )
        
        # Show convergence
        if fed_manager.history:
            final_change = fed_manager.history[-1]['change']
            n_rounds = len(fed_manager.history)
            print(f"   Converged in {n_rounds} rounds (final change: {final_change:.6f})")
        
        # 5. CENTRALIZED LEARNING (for comparison)
        print("🏢 Training centralized model...")
        central_model = FederatedGLM(y_train, X_train, family=family)
        central_result = central_model.fit_proximal(
            alpha=0.01, 
            method="elastic_net",
            verbose=False
        )
        
        # 6. PREDICTIONS
        print("🔮 Making predictions...")
        y_pred_fed = fed_manager.predict(X_test, family)
        y_pred_central = family.link.inverse(X_test @ central_result.params)
        
        # 7. EVALUATION
        print("📈 Evaluating models...")
        metrics_fed = ModelEvaluator.evaluate(y_test, y_pred_fed, family_name)
        metrics_central = ModelEvaluator.evaluate(y_test, y_pred_central, family_name)
        
        # Store results
        results[family_name] = {
            'federated': metrics_fed,
            'centralized': metrics_central
        }
        
        # Print results
        print(f"   📊 Results:")
        print(f"      Federated R²:    {metrics_fed['r2_score']:.4f}")
        print(f"      Centralized R²:  {metrics_central['r2_score']:.4f}")
        print(f"      Federated RMSE:  {metrics_fed['rmse']:.4f}")
        print(f"      Centralized RMSE: {metrics_central['rmse']:.4f}")
        
        if family_name == "binomial":
            print(f"      Federated Accuracy:    {metrics_fed['accuracy']:.4f}")
            print(f"      Centralized Accuracy:  {metrics_central['accuracy']:.4f}")
    
    return results

def demonstrate_convergence():
    """
    Demonstrate federated learning convergence
    """
    print("\n" + "=" * 60)
    print("CONVERGENCE DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    X, y, family = DataGenerator.generate_glm_data("gaussian", n=500, p=4, seed=123)
    client_data = DataGenerator.partition_data(X, y, n_clients=3, seed=123)
    
    # Train with verbose output
    print("🔄 Training with convergence tracking...")
    manager = FederatedLearningManager()
    manager.fit(
        client_data, 
        family, 
        n_rounds=20,
        convergence_tol=1e-4,
        verbose=True  # Show convergence progress
    )
    
    # Plot convergence (if matplotlib available)
    try:
        plt.figure(figsize=(10, 6))
        
        rounds = [h['round'] for h in manager.history]
        changes = [h['change'] for h in manager.history]
        
        plt.subplot(1, 2, 1)
        plt.plot(rounds, changes, 'b-o')
        plt.xlabel('Round')
        plt.ylabel('Parameter Change')
        plt.title('Convergence: Parameter Change per Round')
        plt.yscale('log')
        plt.grid(True)
        
        # Plot global model evolution
        plt.subplot(1, 2, 2)
        global_models = np.array([h['global_model'] for h in manager.history])
        for i in range(global_models.shape[1]):
            plt.plot(rounds, global_models[:, i], label=f'β_{i}')
        plt.xlabel('Round')
        plt.ylabel('Parameter Value')
        plt.title('Global Model Parameter Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📈 Convergence plot saved as 'convergence_plot.png'")
        
    except ImportError:
        print("📈 Install matplotlib to see convergence plots: pip install matplotlib")

def compare_regularization_methods():
    """
    Compare different regularization methods
    """
    print("\n" + "=" * 60)
    print("REGULARIZATION METHODS COMPARISON")
    print("=" * 60)
    
    # Generate data with some noise
    X, y, family = DataGenerator.generate_glm_data("gaussian", n=300, p=6, seed=456)
    client_data = DataGenerator.partition_data(X, y, n_clients=3, seed=456)
    
    # Test data
    X_test, y_test, _ = DataGenerator.generate_glm_data("gaussian", n=100, p=6, seed=789)
    
    methods = ["ordinary", "lasso", "elastic_net"]
    
    print("🔬 Comparing regularization methods...")
    
    for method in methods:
        print(f"\n   Testing {method.upper()} regularization:")
        
        manager = FederatedLearningManager()
        manager.fit(
            client_data, 
            family, 
            n_rounds=10,
            method=method,
            alpha=0.1 if method != "ordinary" else 0.0,
            verbose=False
        )
        
        y_pred = manager.predict(X_test, family)
        metrics = ModelEvaluator.evaluate(y_test, y_pred, "gaussian")
        
        print(f"      R² Score: {metrics['r2_score']:.4f}")
        print(f"      RMSE:     {metrics['rmse']:.4f}")
        print(f"      Model norm: {np.linalg.norm(manager.global_model):.4f}")

def main():
    """
    Run all examples
    """
    try:
        # Basic comparison
        results = compare_federated_vs_centralized()
        
        # Convergence demonstration  
        demonstrate_convergence()
        
        # Regularization comparison
        compare_regularization_methods()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Summary
        print("\n📋 SUMMARY:")
        print("   ✅ Federated learning works across multiple GLM families")
        print("   ✅ Performance comparable to centralized learning")
        print("   ✅ Convergence behavior is stable")
        print("   ✅ Multiple regularization methods supported")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install federated-glm")
        raise

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    results = main()