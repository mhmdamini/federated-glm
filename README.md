````markdown
# Federated GLM

[![PyPI - Version](https://img.shields.io/pypi/v/federated-glm.svg)](https://pypi.org/project/federated-glm/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/federated-glm.svg)](https://pypi.org/project/federated-glm/)

**Federated GLM** is a Python package for simulating **federated learning** with **generalized linear models (GLMs)**. It supports Gaussian, Binomial, and Poisson families, and allows flexible experimentation with elastic net regularization, client data partitioning, and convergence diagnostics.

---

## 🔧 Features

- Federated learning framework with `average` or `weighted` aggregation
- Supports **Gaussian**, **Binomial**, and **Poisson** GLM families
- Elastic Net, Lasso, and Ridge regularization (proximal updates)
- Utilities for synthetic data generation and partitioning across clients
- Model evaluation with R², RMSE, accuracy, Poisson deviance, etc.
- Examples for comparing centralized and federated learning
- Simple API and complete test coverage

---

## 📦 Installation

```bash
pip install git+https://github.com/mhmdamini/federated-glm.git
````

To install with development or example dependencies:

```bash
pip install "federated-glm[dev]"
pip install "federated-glm[examples]"
```

---

## 🛠 Quick Start

Here’s a minimal example using Gaussian regression:

```python
from federated_glm import PersonalizedFederatedGLM, FederatedLearningManager, DataGenerator, ModelEvaluator

# Generate synthetic data
X, y, family = DataGenerator.generate_glm_data("gaussian", n=200, p=3)

# Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Partition data across clients
client_data = DataGenerator.partition_data(X_train, y_train, n_clients=3)

# Train a federated model
manager = FederatedLearningManager()
manager.fit(client_data, family, n_rounds=10)

# Predict and evaluate
y_pred = manager.predict(X_test, family)
metrics = ModelEvaluator.evaluate(y_test, y_pred, "gaussian")

print("R² Score:", metrics["r2_score"])
print("RMSE:", metrics["rmse"])

```

For personalized federated learning, it will be:

```python
pfed = PersonalizedFederatedGLM(method='pfedme')
pfed.fit(client_data, family, n_rounds=20)
```
Where method can be 'pfedme', 'perfedavg', and 'local_adaptation'.

---

## 📁 Project Structure

```
federated-glm/
├── federated_glm/             # Core package
│   ├── core.py                # Federated GLM base class with proximal optimization
│   ├── federation.py          # Federated learning manager
│   ├── evaluation.py          # Model evaluation metrics
│   ├── utils.py               # Data generation & partitioning utilities
│   └── __init__.py
│   └── personalized.py.       # personalized federated learning
├── examples/                  # Usage examples
│   └── basic_example.py
│   └── simple_example.py
│   └── personalized_example.py
├── tests/                     # Unit and integration tests
│   └── test_basic.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## 📈 Examples

Run the full demo script:

```bash
python examples/basic_example.py
```

This includes:

* Federated and personalized federated learning vs centralized performance comparison
* Convergence visualization
* Comparison of regularization strategies (ordinary, lasso, elastic net)

---

## ✅ Supported GLM Families

| Family   | Link Function | Use Case                         |
| -------- | ------------- | -------------------------------- |
| Gaussian | Identity      | Regression on continuous targets |
| Binomial | Logit         | Binary classification            |
| Poisson  | Log           | Count data modeling              |

---

## 🧪 Testing

To run tests:

```bash
pip install "federated-glm[dev]"
pytest tests/
```

---

## 📚 Documentation

Documentation is available in the [GitHub README](https://github.com/mhmdamini/federated-glm#readme).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Mohammad Amini**
Ph.D. Student at University of Florida
[m.amini@ufl.edu](mailto:m.amini@ufl.edu)
[GitHub: @mhmdamini](https://github.com/mhmdamini)

---

## 🙌 Acknowledgements

* Built on [StatsModels](https://www.statsmodels.org/) and [Scikit-learn](https://scikit-learn.org/)
* Inspired by research in federated learning, GLMs, and distributed optimization

---

## 📬 Contributing & Issues

Please open issues or submit pull requests via the [GitHub repository](https://github.com/mhmdamini/federated-glm).

We welcome contributions to:

* Support more GLM families (e.g., Negative Binomial, Gamma)
* Extend to real-world datasets
* Add differential privacy or secure aggregation

---