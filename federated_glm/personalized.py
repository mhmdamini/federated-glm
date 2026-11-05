import numpy as np
from typing import List, Tuple, Optional
from .core import FederatedGLM

METHODS = ('pfedme', 'perfedavg', 'local_adaptation')


class PersonalizedFederatedGLM:
    """
    Personalized federated learning for GLMs.

    Every method keeps a global model ``w`` and one personalized model per
    client. Below, ``f_i`` is client i's average negative log-likelihood plus
    its penalty, and ``R`` is ``personalization_steps``.

    ``'pfedme'`` -- pFedMe (Dinh, Tran and Nguyen, 2020). Each client starts
        from ``w`` and repeats R times: solve the personalized problem
        ``theta_i = argmin f_i(theta) + lambda_reg / 2 * ||theta - w_i||^2``
        exactly, then step ``w_i <- w_i - local_lr * lambda_reg * (w_i - theta_i)``.
        The server sets ``w <- (1 - server_lr) * w + server_lr * mean(w_i)``.
        The personalized model is ``theta_i``.

    ``'perfedavg'`` -- Per-FedAvg (Fallah, Mokhtari and Ozdaglar, 2020), the
        Hessian version. Each client starts from ``w`` and repeats R times:
        ``tmp = w_i - inner_lr * grad f_i(w_i)``, then
        ``w_i <- w_i - outer_lr * (I - inner_lr * H_i(w_i)) grad f_i(tmp)``,
        with ``H_i`` the Fisher information. The server averages the ``w_i``.
        The personalized model is one adaptation step from the global model,
        ``w - inner_lr * grad f_i(w)``. Here ``f_i`` uses a ridge penalty of
        strength ``alpha`` because the method needs a smooth objective. As a
        safeguard for losses that are not globally smooth (Poisson), the
        inner step is halved until it does not increase ``f_i`` and the outer
        step until it does not increase the Per-FedAvg objective.

    ``'local_adaptation'`` -- FedProx training of ``w`` (local fits with a
        proximal term of strength ``rho``), then each personalized model mixes
        the global and the client's local model:
        ``(1 - mix) * w + mix * local_i``.
    """

    def __init__(self, method: str = 'pfedme', lambda_reg: float = 1.0,
                 local_lr: Optional[float] = None, server_lr: float = 1.0,
                 inner_lr: Optional[float] = None, outer_lr: Optional[float] = None,
                 mix: float = 0.3):
        """
        Parameters
        ----------
        method : {'pfedme', 'perfedavg', 'local_adaptation'}
        lambda_reg : float
            pFedMe: weight of the proximal term tying ``theta_i`` to ``w_i``.
            Larger values give less personalization.
        local_lr : float, optional
            pFedMe local step size. Defaults to ``0.5 / lambda_reg``, so each
            step moves ``w_i`` halfway to ``theta_i``.
        server_lr : float
            pFedMe server mixing weight (``beta`` in the paper).
        inner_lr, outer_lr : float, optional
            Per-FedAvg step sizes. Default to ``0.5 / L``, where ``L`` is the
            largest eigenvalue of the client's Hessian at the current point.
        mix : float
            local_adaptation: weight of the local model in the personalized model.
        """
        if method not in METHODS:
            raise ValueError("Method must be one of: 'pfedme', 'perfedavg', 'local_adaptation'")
        self.method = method
        self.lambda_reg = lambda_reg
        self.local_lr = local_lr
        self.server_lr = server_lr
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.mix = mix
        self.global_model = None
        self.family = None
        self.client_models = {}
        self.history = []

    def fit(self, client_data: List[Tuple], family, n_rounds: int = 20,
            alpha: float = 0.01, rho: float = 0.1, personalization_steps: int = 3,
            L1_wt: float = 0.5, verbose: bool = False):
        """
        Train the global and personalized models.

        Parameters
        ----------
        client_data : list of (X, y) tuples, one per client
        family : statsmodels family object
        n_rounds : int, number of federated rounds
        alpha : float, penalty strength of each client's objective
        rho : float, proximal strength of the local fits in 'local_adaptation'
            (pFedMe uses ``lambda_reg`` instead; Per-FedAvg has no proximal term)
        personalization_steps : int, local steps per round (R)
        L1_wt : float, elastic-net mixing for 'pfedme' and 'local_adaptation'
        verbose : bool, print the change in the global model each round
        """
        n_features = client_data[0][0].shape[1]
        self.family = family
        self.global_model = np.zeros(n_features)
        self.client_models = {i: np.zeros(n_features) for i in range(len(client_data))}
        self.history = []
        models = [FederatedGLM(y, X, family=family) for X, y in client_data]

        for round_num in range(n_rounds):
            prev = self.global_model.copy()
            if self.method == 'pfedme':
                self._pfedme_round(models, alpha, L1_wt, personalization_steps)
            elif self.method == 'perfedavg':
                self._perfedavg_round(models, alpha, personalization_steps)
            else:
                self._local_adaptation_round(models, alpha, L1_wt, rho)

            change = np.linalg.norm(self.global_model - prev)
            self.history.append({
                'round': round_num,
                'global_model': self.global_model.copy(),
                'client_models': {k: v.copy() for k, v in self.client_models.items()},
                'change': change,
            })
            if verbose:
                print(f"Round {round_num + 1}/{n_rounds}: change = {change:.6f}")

        return self

    def _pfedme_round(self, models, alpha, L1_wt, R):
        local_lr = self.local_lr if self.local_lr is not None else 0.5 / self.lambda_reg
        local_w = []
        for i, model in enumerate(models):
            w_i = self.global_model.copy()
            theta = self.client_models[i]
            for _ in range(R):
                theta = model.fit_proximal(alpha=alpha, L1_wt=L1_wt, rho=self.lambda_reg,
                                           prox_center=w_i, start_params=theta).params
                w_i = w_i - local_lr * self.lambda_reg * (w_i - theta)
            self.client_models[i] = theta
            local_w.append(w_i)
        self.global_model = ((1 - self.server_lr) * self.global_model
                             + self.server_lr * np.mean(local_w, axis=0))

    @staticmethod
    def _smooth(model, alpha):
        """Value, gradient and Hessian of the client's ridge-penalized objective."""
        mask = model.penalty_mask().astype(float)
        value = lambda b: model.smooth_loss(b) + 0.5 * alpha * np.sum(mask * b * b)
        grad = lambda b: model.smooth_grad(b) + alpha * mask * b
        hess = lambda b: model.fisher_information(b) + alpha * np.diag(mask)
        return value, grad, hess

    def _step_sizes(self, H):
        L = max(np.linalg.eigvalsh(H)[-1], 1e-12)
        inner = self.inner_lr if self.inner_lr is not None else 0.5 / L
        outer = self.outer_lr if self.outer_lr is not None else 0.5 / L
        return inner, outer

    @staticmethod
    def _inner_step(value, w, g, a):
        """Halve the adaptation step until it does not increase the client loss."""
        current = value(w)
        while a > 1e-10 and not value(w - a * g) <= current:
            a /= 2
        return a

    def _perfedavg_round(self, models, alpha, R):
        local_w = []
        for model in models:
            value, grad, hess = self._smooth(model, alpha)
            w_i = self.global_model.copy()
            for _ in range(R):
                H = hess(w_i)
                a, b = self._step_sizes(H)
                g = grad(w_i)
                a = self._inner_step(value, w_i, g, a)
                meta = lambda w: value(w - a * grad(w))  # the Per-FedAvg objective
                direction = (np.eye(len(w_i)) - a * H) @ grad(w_i - a * g)
                # Halve the outer step until the Per-FedAvg objective does not
                # increase; losses like the Poisson one are not globally smooth.
                current = meta(w_i)
                while b > 1e-10 and not meta(w_i - b * direction) <= current:
                    b /= 2
                w_i = w_i - b * direction
            local_w.append(w_i)
        self.global_model = np.mean(local_w, axis=0)
        # Personalize: one gradient step from the new global model.
        for i, model in enumerate(models):
            value, grad, hess = self._smooth(model, alpha)
            a, _ = self._step_sizes(hess(self.global_model))
            g = grad(self.global_model)
            a = self._inner_step(value, self.global_model, g, a)
            self.client_models[i] = self.global_model - a * g

    def _local_adaptation_round(self, models, alpha, L1_wt, rho):
        local_models = [
            model.fit_proximal(alpha=alpha, L1_wt=L1_wt, rho=rho,
                               prox_center=self.global_model).params
            for model in models
        ]
        self.global_model = np.mean(local_models, axis=0)
        for i, local in enumerate(local_models):
            self.client_models[i] = (1 - self.mix) * self.global_model + self.mix * local

    def predict(self, X, family=None, client_id: Optional[int] = None):
        """
        Predict the mean response.

        Parameters
        ----------
        X : array-like, input features
        family : statsmodels family, optional; defaults to the one used in ``fit``
        client_id : int, optional; use this client's personalized model
        """
        family = family or self.family
        if client_id is not None:
            return family.link.inverse(X @ self.get_client_model(client_id))
        return family.link.inverse(X @ self.global_model)

    def get_client_model(self, client_id: int):
        """Personalized model of one client."""
        if client_id in self.client_models:
            return self.client_models[client_id]
        raise ValueError(f"Client {client_id} not found")

    def get_global_model(self):
        """Global model parameters."""
        return self.global_model
