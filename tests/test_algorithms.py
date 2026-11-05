# tests/test_algorithms.py
import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Lasso

from federated_glm import (DataGenerator, FederatedGLM, FederatedLearningManager,
                           ModelEvaluator, PersonalizedFederatedGLM)


def sparse_regression(n=300, p=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2 * X[:, 0] - X[:, 1] + 0.5 + 0.3 * rng.randn(n)
    return X, y


class TestSolver:
    """fit_proximal solves the stated objective exactly."""

    def test_lasso_matches_sklearn_and_gives_exact_zeros(self):
        X, y = sparse_regression()
        res = FederatedGLM(y, sm.add_constant(X), family=sm.families.Gaussian()).fit_proximal(
            alpha=0.1, method="lasso", tol=1e-10, maxiter=5000)
        sk = Lasso(alpha=0.1, tol=1e-12, max_iter=100000).fit(X, y)
        np.testing.assert_allclose(res.params, np.r_[sk.intercept_, sk.coef_], atol=1e-7)
        assert np.sum(res.params[3:] == 0) == 8  # the noise features
        assert res.converged

    def test_elastic_net_matches_sklearn(self):
        X, y = sparse_regression()
        res = FederatedGLM(y, sm.add_constant(X), family=sm.families.Gaussian()).fit_proximal(
            alpha=0.1, L1_wt=0.3, tol=1e-10, maxiter=5000)
        sk = ElasticNet(alpha=0.1, l1_ratio=0.3, tol=1e-12, max_iter=100000).fit(X, y)
        np.testing.assert_allclose(res.params, np.r_[sk.intercept_, sk.coef_], atol=1e-7)

    @pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
    def test_ordinary_matches_statsmodels(self, family):
        X, y, fam = DataGenerator.generate_glm_data(family, n=400, p=3, seed=1)
        res = FederatedGLM(y, X, family=fam).fit_proximal(method="ordinary", tol=1e-10, maxiter=5000)
        np.testing.assert_allclose(res.params, sm.GLM(y, X, family=fam).fit().params, atol=1e-6)

    def test_intercept_not_penalized(self):
        X, y = sparse_regression()
        res = FederatedGLM(y, sm.add_constant(X), family=sm.families.Gaussian()).fit_proximal(
            alpha=100.0, method="lasso")
        assert np.all(res.params[1:] == 0)
        assert res.params[0] == pytest.approx(y.mean(), abs=1e-6)

    def test_penalize_intercept_option(self):
        X, y = sparse_regression()
        res = FederatedGLM(y, sm.add_constant(X), family=sm.families.Gaussian()).fit_proximal(
            alpha=100.0, method="lasso", penalize_intercept=True)
        assert np.all(res.params == 0)

    def test_large_rho_pulls_to_center(self):
        X, y = sparse_regression()
        center = np.linspace(-1, 1, 11)
        res = FederatedGLM(y, sm.add_constant(X), family=sm.families.Gaussian()).fit_proximal(
            alpha=0.0, method="ordinary", rho=1e6, prox_center=center)
        np.testing.assert_allclose(res.params, center, atol=1e-4)

    def test_max_steps_is_deprecated(self):
        X, y = sparse_regression(n=50, p=2)
        model = FederatedGLM(y, sm.add_constant(X), family=sm.families.Gaussian())
        with pytest.warns(DeprecationWarning):
            model.fit_proximal(max_steps=1)


class TestFederation:

    def test_predict_uses_fitted_family(self):
        train, test, fam = DataGenerator.generate_federated_data("binomial", n_clients=3, seed=0)
        manager = FederatedLearningManager().fit(train, fam, n_rounds=3)
        X = test[0][0]
        np.testing.assert_allclose(manager.predict(X), manager.predict(X, fam))
        assert np.all((manager.predict(X) > 0) & (manager.predict(X) < 1))

    def test_invalid_aggregation_fails_at_construction(self):
        with pytest.raises(ValueError):
            FederatedLearningManager(aggregation_method="median")

    def test_federated_close_to_centralized_on_iid_data(self):
        train, _, fam = DataGenerator.generate_federated_data("gaussian", n_clients=4, heterogeneity=0.0, seed=2)
        manager = FederatedLearningManager().fit(train, fam, n_rounds=50, alpha=0.0, method="ordinary")
        X = np.vstack([c[0] for c in train])
        y = np.concatenate([c[1] for c in train])
        central = sm.GLM(y, X, family=fam).fit().params
        assert np.linalg.norm(manager.global_model - central) < 0.05


def client_mse(predict, test):
    return np.mean([np.mean((y - predict(X, i)) ** 2) for i, (X, y) in enumerate(test)])


@pytest.fixture(scope="module")
def heterogeneous():
    return DataGenerator.generate_federated_data("gaussian", n_clients=5, n_per_client=120,
                                                 heterogeneity=1.5, seed=3)


class TestPersonalized:

    @pytest.mark.parametrize("method", ["pfedme", "perfedavg", "local_adaptation"])
    def test_personalized_beats_global_on_heterogeneous_clients(self, heterogeneous, method):
        train, test, fam = heterogeneous
        global_mse = client_mse(lambda X, i: FederatedLearningManager().fit(train, fam, n_rounds=20).predict(X), test)
        model = PersonalizedFederatedGLM(method).fit(train, fam, n_rounds=20)
        assert client_mse(lambda X, i: model.predict(X, client_id=i), test) < 0.8 * global_mse

    @pytest.mark.parametrize("method", ["pfedme", "perfedavg", "local_adaptation"])
    def test_poisson_runs_and_stays_finite(self, method):
        train, _, fam = DataGenerator.generate_federated_data("poisson", n_clients=4, heterogeneity=1.5, seed=3)
        model = PersonalizedFederatedGLM(method).fit(train, fam, n_rounds=10)
        assert np.all(np.isfinite(model.global_model))
        assert all(np.all(np.isfinite(m)) for m in model.client_models.values())

    def test_pfedme_personalization_steps_matter(self, heterogeneous):
        train, _, fam = heterogeneous
        one = PersonalizedFederatedGLM("pfedme").fit(train, fam, n_rounds=3, personalization_steps=1)
        five = PersonalizedFederatedGLM("pfedme").fit(train, fam, n_rounds=3, personalization_steps=5)
        assert not np.allclose(one.global_model, five.global_model)

    def test_pfedme_lambda_controls_personalization(self, heterogeneous):
        train, _, fam = heterogeneous

        def spread(lam):
            m = PersonalizedFederatedGLM("pfedme", lambda_reg=lam).fit(train, fam, n_rounds=10)
            return np.mean([np.linalg.norm(t - m.global_model) for t in m.client_models.values()])

        assert spread(10.0) < spread(0.5)

    def test_local_adaptation_rho_matters(self, heterogeneous):
        train, _, fam = heterogeneous
        a = PersonalizedFederatedGLM("local_adaptation").fit(train, fam, n_rounds=3, rho=0.01)
        b = PersonalizedFederatedGLM("local_adaptation").fit(train, fam, n_rounds=3, rho=10.0)
        assert not np.allclose(a.client_models[0], b.client_models[0])

    def test_unknown_client_raises(self, heterogeneous):
        train, test, fam = heterogeneous
        model = PersonalizedFederatedGLM("local_adaptation").fit(train, fam, n_rounds=2)
        with pytest.raises(ValueError):
            model.predict(test[0][0], client_id=99)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            PersonalizedFederatedGLM("fedavg")


class TestUtilitiesAndMetrics:

    def test_poisson_deviance_with_zero_counts(self):
        y = np.array([0, 0, 1, 3, 0, 2])
        mu = np.array([0.5, 0.2, 1.1, 2.5, 0.1, 2.0])
        dev = ModelEvaluator.evaluate(y, mu, "poisson")["poisson_deviance"]
        assert dev == pytest.approx(sm.families.Poisson().deviance(y, mu))

    def test_generators_do_not_touch_global_random_state(self):
        np.random.seed(123)
        expected = np.random.rand()
        np.random.seed(123)
        DataGenerator.generate_glm_data("gaussian", seed=7)
        DataGenerator.generate_federated_data("gaussian", seed=7)
        X, y, _ = DataGenerator.generate_glm_data("gaussian", n=30, p=2)
        DataGenerator.partition_data(X, y, seed=7)
        assert np.random.rand() == expected

    def test_generators_are_reproducible(self):
        a = DataGenerator.generate_federated_data("poisson", seed=5)
        b = DataGenerator.generate_federated_data("poisson", seed=5)
        np.testing.assert_array_equal(a[0][2][1], b[0][2][1])

    def test_heterogeneity_zero_gives_shared_coefficients(self):
        train, _, fam = DataGenerator.generate_federated_data("gaussian", n_clients=2, n_per_client=4000,
                                                              heterogeneity=0.0, seed=1)
        fits = [sm.GLM(y, X, family=fam).fit().params for X, y in train]
        np.testing.assert_allclose(fits[0], fits[1], atol=0.1)
