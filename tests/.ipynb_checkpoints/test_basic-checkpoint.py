# tests/test_basic.py
import pytest
import numpy as np
import statsmodels.api as sm
from federated_glm import FederatedGLM, FederatedLearningManager, ModelEvaluator, DataGenerator

class TestDataGenerator:
    """Test the DataGenerator utility class"""
    
    def test_generate_gaussian_data(self):
        X, y, family = DataGenerator.generate_glm_data("gaussian", n=100, p=3, seed=42)
        assert X.shape == (100, 4)  # 3 features + constant
        assert y.shape == (100,)
        assert isinstance(family, sm.families.Gaussian)
    
    def test_generate_poisson_data(self):
        X, y, family = DataGenerator.generate_glm_data("poisson", n=50, p=2, seed=42)
        assert X.shape == (50, 3)  # 2 features + constant
        assert y.shape == (50,)
        assert isinstance(family, sm.families.Poisson)
    
    def test_generate_binomial_data(self):
        X, y, family = DataGenerator.generate_glm_data("binomial", n=80, p=2, seed=42)
        assert X.shape == (80, 3)
        assert y.shape == (80,)
        assert isinstance(family, sm.families.Binomial)
        assert all(val in [0, 1] for val in y)  # Binary values
    
    def test_unsupported_family(self):
        with pytest.raises(ValueError):
            DataGenerator.generate_glm_data("unsupported", n=50, p=2)
    
    def test_partition_data_random(self):
        X, y, _ = DataGenerator.generate_glm_data("gaussian", n=100, p=3, seed=42)
        client_data = DataGenerator.partition_data(X, y, n_clients=3, method='random', seed=42)
        
        assert len(client_data) == 3
        total_samples = sum(len(data[1]) for data in client_data)
        assert total_samples == 100
        
        # Check shapes
        for X_client, y_client in client_data:
            assert X_client.shape[1] == 4  # Same number of features
            assert len(X_client) == len(y_client)
    
    def test_partition_data_sequential(self):
        X, y, _ = DataGenerator.generate_glm_data("gaussian", n=99, p=2, seed=42)
        client_data = DataGenerator.partition_data(X, y, n_clients=3, method='sequential')
        
        assert len(client_data) == 3
        total_samples = sum(len(data[1]) for data in client_data)
        assert total_samples == 99

class TestFederatedGLM:
    """Test the FederatedGLM core class"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X, self.y, self.family = DataGenerator.generate_glm_data("gaussian", n=100, p=3, seed=42)
    
    def test_initialization(self):
        model = FederatedGLM(self.y, self.X, family=self.family)
        assert model.endog.shape == (100,)
        assert model.exog.shape == (100, 4)
    
    def test_fit_proximal_ordinary(self):
        model = FederatedGLM(self.y, self.X, family=self.family)
        result = model.fit_proximal(method="ordinary", alpha=0.0, rho=0.0)
        assert result.params.shape == (4,)  # 3 features + constant
        assert hasattr(result, 'params')
    
    def test_fit_proximal_lasso(self):
        model = FederatedGLM(self.y, self.X, family=self.family)
        result = model.fit_proximal(method="lasso", alpha=0.1, rho=0.1)
        assert result.params.shape == (4,)
    
    def test_fit_proximal_elastic_net(self):
        model = FederatedGLM(self.y, self.X, family=self.family)
        result = model.fit_proximal(method="elastic_net", alpha=0.1, L1_wt=0.5, rho=0.1)
        assert result.params.shape == (4,)
    
    def test_invalid_method(self):
        model = FederatedGLM(self.y, self.X, family=self.family)
        with pytest.raises(ValueError):
            model.fit_proximal(method="invalid")

class TestFederatedLearningManager:
    """Test the FederatedLearningManager class"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        X, y, self.family = DataGenerator.generate_glm_data("gaussian", n=150, p=3, seed=42)
        self.client_data = DataGenerator.partition_data(X, y, n_clients=3, seed=42)
    
    def test_initialization(self):
        manager = FederatedLearningManager()
        assert manager.aggregation_method == 'average'
        assert manager.global_model is None
        assert manager.history == []
    
    def test_initialization_weighted(self):
        manager = FederatedLearningManager(aggregation_method='weighted')
        assert manager.aggregation_method == 'weighted'
    
    def test_fit_basic(self):
        manager = FederatedLearningManager()
        manager.fit(self.client_data, self.family, n_rounds=3, verbose=False)
        
        assert manager.global_model is not None
        assert manager.global_model.shape == (4,)  # 3 features + constant
        assert len(manager.history) <= 3  # Might converge early
    
    def test_predict_before_fit(self):
        manager = FederatedLearningManager()
        X_test = np.random.randn(10, 4)
        
        with pytest.raises(RuntimeError):
            manager.predict(X_test, self.family)
    
    def test_predict_after_fit(self):
        manager = FederatedLearningManager()
        manager.fit(self.client_data, self.family, n_rounds=2)
        
        X_test = np.random.randn(10, 4)
        predictions = manager.predict(X_test, self.family)
        assert predictions.shape == (10,)
    
    def test_invalid_aggregation_method(self):
        with pytest.raises(ValueError):
            manager = FederatedLearningManager(aggregation_method='invalid')
            manager.fit(self.client_data, self.family, n_rounds=1)

class TestModelEvaluator:
    """Test the ModelEvaluator class"""
    
    def test_evaluate_gaussian(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred, "gaussian")
        
        assert 'r2_score' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert metrics['r2_score'] > 0.9  # Should be high for close predictions
    
    def test_evaluate_binomial(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.9])
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred, "binomial")
        
        assert 'accuracy' in metrics
        assert 'r2_score' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect classification
    
    def test_evaluate_poisson(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred, "poisson")
        
        assert 'poisson_deviance' in metrics
        assert 'r2_score' in metrics

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_workflow_gaussian(self):
        """Test complete workflow for Gaussian GLM"""
        # Generate data
        X, y, family = DataGenerator.generate_glm_data("gaussian", n=200, p=3, seed=42)
        
        # Split into train/test
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        # Partition training data
        client_data = DataGenerator.partition_data(X_train, y_train, n_clients=3)
        
        # Train federated model
        manager = FederatedLearningManager()
        manager.fit(client_data, family, n_rounds=5)
        
        # Make predictions
        y_pred = manager.predict(X_test, family)
        
        # Evaluate
        metrics = ModelEvaluator.evaluate(y_test, y_pred, "gaussian")
        
        # Basic sanity checks
        assert len(y_pred) == len(y_test)
        assert 'r2_score' in metrics
        assert metrics['r2_score'] > -1  # R² should be reasonable
    
    def test_full_workflow_binomial(self):
        """Test complete workflow for Binomial GLM"""
        # Generate data  
        X, y, family = DataGenerator.generate_glm_data("binomial", n=200, p=3, seed=42)
        
        # Split into train/test
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        # Partition training data
        client_data = DataGenerator.partition_data(X_train, y_train, n_clients=3)
        
        # Train federated model
        manager = FederatedLearningManager()
        manager.fit(client_data, family, n_rounds=5)
        
        # Make predictions
        y_pred = manager.predict(X_test, family)
        
        # Evaluate
        metrics = ModelEvaluator.evaluate(y_test, y_pred, "binomial")
        
        # Basic sanity checks
        assert len(y_pred) == len(y_test)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1