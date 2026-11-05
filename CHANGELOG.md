# Changelog

## 0.3.0

### Fixed
- `fit_proximal` now solves the stated elastic-net problem by accelerated
  proximal gradient (FISTA). The L1 penalty gives exact zeros; before, it was
  smoothed and never did. Results match scikit-learn's `Lasso` and `ElasticNet`
  for the Gaussian family.
- The intercept is no longer penalized (pass `penalize_intercept=True` for the
  old behavior).
- `pfedme` now runs the pFedMe algorithm: personalized subproblems, local
  steps, server mixing. Before, its local steps all solved the same problem,
  so `personalization_steps` had no effect.
- `perfedavg` now runs Per-FedAvg (MAML steps with the exact Hessian).
  Before, it was FedAvg followed by fine-tuning.
- Poisson deviance no longer returns NaN when the data contain zero counts.
- `DataGenerator` no longer reseeds NumPy's global random generator. It
  produces the same data for a given seed as before.
- `examples/basic_example.py` now tests the regularization comparison on data
  from the same model it trained on.

### Added
- `DataGenerator.generate_federated_data` for heterogeneous clients with
  train/test splits.
- `predict()` remembers the family passed to `fit()`.
- `PersonalizedFederatedGLM` options: `local_lr`, `server_lr`, `inner_lr`,
  `outer_lr` and `mix`.
- `FederatedGLM.smooth_loss`, `smooth_grad` and `fisher_information`.
- Results report `n_iter` and `converged`.
- Tests against scikit-learn and statsmodels (coverage 66% → 93%) and GitHub
  Actions CI.

### Changed
- `PersonalizedFederatedGLM` defaults to `lambda_reg=1.0` (was 0.1).
- `predict(..., client_id=i)` raises `ValueError` for an unknown client
  instead of silently using the global model.
- An invalid `aggregation_method` raises when the manager is created.
- `max_steps` is deprecated and ignored; use `maxiter`.
- Requires Python 3.10 or newer (tested on 3.10, 3.12 and 3.13).
