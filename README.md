# Federated GLM

[![tests](https://github.com/mhmdamini/federated-glm/actions/workflows/tests.yml/badge.svg)](https://github.com/mhmdamini/federated-glm/actions/workflows/tests.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/federated-glm.svg)](https://pypi.org/project/federated-glm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/federated-glm.svg)](https://pypi.org/project/federated-glm/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Federated GLM** fits generalized linear models (Gaussian, binomial and
Poisson) on data split across clients that never share raw data. It provides
federated training of one global model (FedProx-style), three personalized
methods (pFedMe, Per-FedAvg and local adaptation), elastic-net
regularization, and tools to simulate heterogeneous clients.

## Installation

```bash
pip install federated-glm
```

With the extras for development or for the plotting examples:

```bash
pip install "federated-glm[dev]"
pip install "federated-glm[examples]"
```

## Quick start

```python
from federated_glm import DataGenerator, FederatedLearningManager, ModelEvaluator

# Five clients; their true coefficients differ by `heterogeneity`.
train, test, family = DataGenerator.generate_federated_data(
    "gaussian", n_clients=5, n_per_client=120, p=5, heterogeneity=0.0)

manager = FederatedLearningManager().fit(train, family, n_rounds=10)

X_test, y_test = test[0]
metrics = ModelEvaluator.evaluate(y_test, manager.predict(X_test), "gaussian")
print("R²:", metrics["r2_score"], " RMSE:", metrics["rmse"])
```

Personalized models, one per client:

```python
from federated_glm import PersonalizedFederatedGLM

train, test, family = DataGenerator.generate_federated_data(
    "gaussian", n_clients=5, heterogeneity=1.5)

model = PersonalizedFederatedGLM(method="pfedme").fit(train, family, n_rounds=20)
y_pred = model.predict(test[2][0], client_id=2)   # client 2's own model
```

To use your own data, pass a list of `(X, y)` tuples, one per client, where
`X` includes a constant column (for example from `statsmodels.api.add_constant`),
and a family such as `statsmodels.api.families.Binomial()`.

## Method

### Local problem

Client *i* holds data `(X_i, y_i)` with `n_i` rows. Every local fit solves

$$
\min_{\beta}\; \frac{1}{n_i}\,\mathrm{NLL}_i(\beta)
\;+\; \alpha\Big[\,\gamma\,\lVert\beta\rVert_1 + \tfrac{1-\gamma}{2}\lVert\beta\rVert_2^2\Big]
\;+\; \tfrac{\rho}{2}\,\lVert\beta-\beta_{\text{center}}\rVert_2^2 ,
$$

where γ is `L1_wt`.

- **NLL** is the GLM negative log-likelihood with the dispersion fixed at 1.
  For the Gaussian family it is half the squared error, so the elastic-net
  solution matches scikit-learn's `ElasticNet` exactly.
- The **intercept** is not penalized unless you pass `penalize_intercept=True`.
- The problem is solved by **accelerated proximal gradient** (FISTA) with
  backtracking and adaptive restart, so the L1 term sets coefficients exactly
  to zero.

`FederatedGLM.fit_proximal` solves this problem for one client. `method`
selects `"ordinary"` (no penalty), `"lasso"` or `"elastic_net"`.

### Federated training: `FederatedLearningManager`

Each round, every client solves the local problem with `β_center` equal to the
current global model. This is the FedProx update, and `ρ` controls how far
clients may move. The server then averages the local models, either equally
(`"average"`) or weighted by sample size (`"weighted"`). Only parameters leave
the clients.

### Personalized training: `PersonalizedFederatedGLM`

`w` is the global model and `R` is `personalization_steps`.

| `method` | Training | Personalized model of client *i* |
| --- | --- | --- |
| `"pfedme"` ([Dinh et al., 2020](https://arxiv.org/abs/2006.08848)) | Each client repeats R times: solve θᵢ = argmin fᵢ(θ) + λ/2 ‖θ − wᵢ‖², then wᵢ ← wᵢ − η λ (wᵢ − θᵢ). The server sets w ← (1 − β) w + β · mean(wᵢ). | θᵢ |
| `"perfedavg"` ([Fallah et al., 2020](https://arxiv.org/abs/2002.07948)) | Each client takes R MAML steps on fᵢ(w − a∇fᵢ(w)) using the exact Hessian. The server averages. | w − a∇fᵢ(w) |
| `"local_adaptation"` | FedProx training of w. | (1 − mix) w + mix · localᵢ |

Parameters: `lambda_reg` = λ, `local_lr` = η, `server_lr` = β,
`inner_lr` / `outer_lr` = a / b, and `mix`. Per-FedAvg needs a smooth
objective, so it uses a ridge penalty of strength `alpha`. It halves its steps
when a step would increase the loss, which keeps Poisson models stable.

**Which method to use.** When clients are alike (`heterogeneity=0`), the
single global model is as good as any personalized one. When they differ,
personalized models help a lot. Here is the output of
[`examples/personalized_example.py`](examples/personalized_example.py)
(average test MSE on each client's own data, 5 Gaussian clients):

| Method | heterogeneity = 0 | heterogeneity = 1.5 |
| --- | ---: | ---: |
| Federated (one global model) | 0.256 | 5.034 |
| Local only | 0.296 | 0.294 |
| pFedMe | 0.280 | 0.789 |
| Per-FedAvg | 0.263 | 2.500 |
| Local adaptation | 0.264 | 2.837 |

## API

| Object | What it does |
| --- | --- |
| `FederatedGLM(y, X, family)` | A statsmodels `GLM` with `fit_proximal(alpha, L1_wt, rho, prox_center, method, ...)`, plus `smooth_loss`, `smooth_grad` and `fisher_information`. |
| `FederatedLearningManager(aggregation_method)` | `fit(client_data, family, n_rounds, alpha, L1_wt, rho, method)`, then `predict(X)`. Each round is recorded in `history`. |
| `PersonalizedFederatedGLM(method, ...)` | `fit(client_data, family, n_rounds, alpha, rho, personalization_steps)`, then `predict(X, client_id=i)`, `get_client_model(i)` and `get_global_model()`. |
| `DataGenerator` | `generate_glm_data` (one data set), `generate_federated_data` (heterogeneous clients with train/test splits) and `partition_data`. Every call is reproducible and leaves NumPy's global random state untouched. |
| `ModelEvaluator.evaluate(y, y_pred, family_name)` | R², MSE, RMSE and MAE. Adds accuracy for binomial models and deviance for Poisson models. |

## Examples

```bash
python examples/simple_example.py         # quick start
python examples/personalized_example.py   # global vs local vs personalized
python examples/basic_example.py          # all families, convergence plot, penalties
```

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=federated_glm tests
```

The tests check the solver against scikit-learn and statsmodels, and cover all
three personalized methods. Coverage is 93%.

## Citing

If you use this package, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff). GitHub's **"Cite this repository"** button
gives BibTeX and APA formats.

## Author

**Mohammad Amini**, Ph.D. in Industrial and Systems Engineering, University of Florida.
[mhmdamini.github.io](https://mhmdamini.github.io) · [m.amini@ufl.edu](mailto:m.amini@ufl.edu)

## Contributing

Issues and pull requests are welcome. Possible directions:

- more families (negative binomial, gamma)
- secure aggregation or differential privacy
- benchmarks on real federated data sets

## License

[MIT](LICENSE)
