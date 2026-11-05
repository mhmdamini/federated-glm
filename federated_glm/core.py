# federated_glm/core.py
import warnings

import numpy as np
from statsmodels.base.elastic_net import RegularizedResults, RegularizedResultsWrapper
from statsmodels.genmod.generalized_linear_model import GLM

METHODS = ("ordinary", "lasso", "elastic_net")


class FederatedGLM(GLM):
    """GLM with an elastic-net penalty and a proximal term, for federated learning.

    ``fit_proximal`` minimizes

        (1/n) NLL(beta)
          + alpha * [L1_wt * ||beta||_1 + (1 - L1_wt) / 2 * ||beta||_2^2]
          + rho / 2 * ||beta - prox_center||_2^2

    where NLL is the negative log-likelihood with the dispersion fixed at 1
    (half the squared error for the Gaussian family). A constant column of the
    design (the intercept) is not penalized unless ``penalize_intercept=True``.
    The problem is solved by accelerated proximal gradient (FISTA) with
    backtracking, so the L1 term produces exact zeros.
    """

    def __init__(self, endog, exog, family=None, **kwargs):
        super().__init__(endog, exog, family, **kwargs)
        # Columns that are constant over the sample are treated as intercepts.
        self._intercept = np.ptp(self.exog, axis=0) == 0

    # ---- smooth part of the objective -------------------------------------

    def smooth_loss(self, beta):
        """Average negative log-likelihood with dispersion 1."""
        # Trial points of the line search may overflow; they score as +inf.
        with np.errstate(all="ignore"):
            mu = self.family.fitted(self.exog @ beta)
            value = -self.family.loglike(self.endog, mu, scale=1.0) / len(self.endog)
        return value if np.isfinite(value) else np.inf

    def smooth_grad(self, beta):
        """Gradient of ``smooth_loss``."""
        mu = self.family.fitted(self.exog @ beta)
        w = (self.endog - mu) / (self.family.variance(mu) * self.family.link.deriv(mu))
        return -self.exog.T @ w / len(self.endog)

    def fisher_information(self, beta):
        """Expected Hessian of ``smooth_loss`` (exact for canonical links)."""
        mu = self.family.fitted(self.exog @ beta)
        w = 1.0 / (self.family.variance(mu) * self.family.link.deriv(mu) ** 2)
        return (self.exog * w[:, None]).T @ self.exog / len(self.endog)

    def penalty_mask(self, penalize_intercept=False):
        """Boolean mask of the coefficients that are penalized."""
        if penalize_intercept:
            return np.ones(self.exog.shape[1], dtype=bool)
        return ~self._intercept

    # ---- fitting ------------------------------------------------------------

    def fit_proximal(self, alpha=0.1, L1_wt=1.0, rho=0.0, prox_center=None,
                     start_params=None, maxiter=1000, max_steps=None,
                     tol=1e-6, method="elastic_net", penalize_intercept=False,
                     verbose=False):
        """
        Fit the GLM with an elastic-net penalty and a proximal term.

        Parameters
        ----------
        alpha : float
            Penalty strength.
        L1_wt : float
            Elastic-net mixing parameter (0 = ridge, 1 = lasso). Used only
            when ``method="elastic_net"``.
        rho : float
            Strength of the proximal term pulling ``beta`` toward
            ``prox_center`` (the FedProx term in federated training).
        prox_center : array, optional
            Center of the proximal term, usually the current global model.
            Defaults to zeros.
        start_params : array, optional
            Starting point. Defaults to ``prox_center``.
        maxiter : int
            Maximum number of proximal-gradient iterations.
        max_steps : None
            Deprecated and ignored; use ``maxiter``.
        tol : float
            Stop when the relative change in ``beta`` falls below ``tol``.
        method : {"ordinary", "lasso", "elastic_net"}
            "ordinary" ignores ``alpha``; "lasso" uses ``alpha * ||beta||_1``.
        penalize_intercept : bool
            Also penalize constant columns of the design.
        verbose : bool
            Print the objective at each iteration.

        Returns
        -------
        RegularizedResultsWrapper
            ``params`` holds the solution; ``n_iter`` and ``converged`` report
            the solver's progress.
        """
        if method not in METHODS:
            raise ValueError("Method must be 'ordinary', 'lasso', or 'elastic_net'")
        if max_steps is not None:
            warnings.warn("max_steps is ignored and will be removed; use maxiter",
                          DeprecationWarning, stacklevel=2)

        p = self.exog.shape[1]
        center = np.zeros(p) if prox_center is None else np.asarray(prox_center, dtype=float)
        if method == "ordinary":
            l1, l2 = 0.0, 0.0
        elif method == "lasso":
            l1, l2 = alpha, 0.0
        else:
            l1, l2 = alpha * L1_wt, alpha * (1.0 - L1_wt)
        mask = self.penalty_mask(penalize_intercept)

        def f(b):
            return self.smooth_loss(b) + 0.5 * l2 * np.sum(b[mask] ** 2) + 0.5 * rho * np.sum((b - center) ** 2)

        def grad(b):
            g = self.smooth_grad(b) + rho * (b - center)
            g[mask] += l2 * b[mask]
            return g

        def prox(v, step):
            out = v.copy()
            out[mask] = np.sign(v[mask]) * np.maximum(np.abs(v[mask]) - step * l1, 0.0)
            return out

        x = center.copy() if start_params is None else np.asarray(start_params, dtype=float).copy()
        y, t, L = x.copy(), 1.0, 1.0
        converged, it = False, 0
        for it in range(1, maxiter + 1):
            fy, gy = f(y), grad(y)
            while True:
                x_new = prox(y - gy / L, 1.0 / L)
                d = x_new - y
                if f(x_new) <= fy + gy @ d + 0.5 * L * (d @ d) + 1e-12 * abs(fy) or L > 1e12:
                    break
                L *= 2.0
            # Restart the momentum when it points uphill (adaptive restart).
            if gy @ (x_new - x) > 0:
                t = 1.0
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            step = np.linalg.norm(x_new - x)
            x, t = x_new, t_new
            if verbose:
                print(f"iter {it}: objective {f(x) + l1 * np.sum(np.abs(x[mask])):.6f}")
            if step <= tol * (1.0 + np.linalg.norm(x)):
                converged = True
                break
            L *= 0.9  # let the step size grow again

        res = RegularizedResults(self, x)
        res.n_iter = it
        res.converged = converged
        return RegularizedResultsWrapper(res)
