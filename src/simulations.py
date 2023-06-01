from abc import ABC, abstractmethod

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import norm
from sklearn.utils import check_random_state

from ._linalg import gram_schmidt_orthonormalization


def step(x, n):
    y = x.copy()
    y[x <= 1 / n] = n / 2 * x[x <= 1 / n]
    y[(x > 1 / n) & (x < (n - 1) / n)] = 1 / 2
    y[x >= (n - 1) / n] = n / 2 * (x[x >= (n - 1) / n] - 1) + 1
    return y


def slope(x, n):
    y = x.copy()
    a1 = (n - 2) / 2
    b1 = 0
    a2 = 2 / (n - 2)
    b2 = (1 - a2) / 2
    a3 = n / 2 - 1
    b3 = 1 - a3
    y[x <= 1 / n] = a1 * x[x <= 1 / n] + b1
    y[(x > 1 / n) & (x < (n - 1) / n)] = a2 * x[(x > 1 / n) & (x < (n - 1) / n)] + b2
    y[x >= (n - 1) / n] = a3 * x[x >= (n - 1) / n] + b3

    return y


class BaseExample(ABC):
    @abstractmethod
    def generate_X_y(self, n, random_state=0):
        pass

    @abstractmethod
    def f(X):
        pass

    @abstractmethod
    def f_star(X):
        pass

    def _base_emp(self, f, N=100000, random_state=0):
        """Base code for empirical estimation."""
        rng = check_random_state(random_state)
        dist = self.dist()
        X = dist.rvs(size=N, random_state=rng)
        return np.mean(f(X))

    def GL_emp(self, N=100000, random_state=0):
        """Empirical estimation of grouping loss."""

        def d(x):
            # GL = MSE(C, Q) with brier score as proper scoring rule
            return np.square(self.C(x) - self.Q(x))

        return self._base_emp(d, N, random_state)

    def repr(self, kwargs) -> str:
        return (
            self.__class__.__name__
            + "("
            + ",".join([f"{k}={v}" for k, v in kwargs.items()])
            + ")"
        )

    def __repr__(self) -> str:
        return self.repr({})


class SigmoidExample(BaseExample):
    def __init__(self, w, w_perp, bayes_opt=False, delta_width=3, lbd=10):
        self.w = np.squeeze(w)
        self.w_perp = np.squeeze(w_perp)
        self.bayes_opt = bayes_opt
        self.delta_width = delta_width
        self.lbd = lbd

        d = self.w.shape[0]
        P = np.eye(d)
        P[:, 0] = self.w
        P[:, 1] = self.w_perp
        P = gram_schmidt_orthonormalization(P)
        D = np.eye(d)
        D[0, 0] = lbd
        # Create PSD cov from PDP^-1 decomposition
        self.cov = P @ D @ np.linalg.inv(P)
        self.mean = np.zeros_like(self.w)

    def dist(self):
        return scipy.stats.multivariate_normal(mean=self.mean, cov=self.cov)

    def C(self, X):
        return self.S(X)

    def phi(self, dot):
        return expit(dot)

    def S(self, X):
        dot = np.dot(X, self.w)
        return self.phi(dot)

    def Q(self, X):
        return self.S(X) + self.delta(X)

    def generate_X_y(self, n, random_state=0):
        rng = check_random_state(random_state)
        X = rng.multivariate_normal(self.mean, self.cov, size=n)
        y = rng.binomial(1, self.f_star(X))
        return X, y

    def f_1d(self, dot):
        return expit(dot)

    def f(self, X):
        dot = np.dot(X, self.w)
        return self.f_1d(dot)

    def psi(self, dot_perp):
        if self.delta_width is not None:
            y = 2 * expit(self.delta_width * dot_perp) - 1
            return np.sign(y) * np.abs(y)
        else:
            y = np.array(dot_perp > 0).astype(float) - np.array(dot_perp < 0).astype(
                float
            )
            return y

    def delta_max(self, dot):
        _delta_max = np.minimum(1 - self.f_1d(dot), self.f_1d(dot))
        if self.bayes_opt:
            _delta_max = np.minimum(_delta_max, np.abs(self.f_1d(dot) - 1 / 2))
        return _delta_max

    def _delta(self, dot, dot_perp):
        _delta_max = self.delta_max(dot)
        return np.multiply(self.psi(dot_perp), _delta_max)

    def delta(self, X):
        dot_perp = np.dot(X, self.w_perp)
        dot = np.dot(X, self.w)
        return self._delta(dot, dot_perp)

    def f_star(self, X):
        return self.f(X) + self.delta(X)

    def __repr__(self) -> str:
        return self.repr(
            {
                "width": self.delta_width,
                "lbd": self.lbd,
            }
        )


class CustomUniform(BaseExample):
    def __init__(self, name="poly", dist="uniform", half=False, alpha=1):
        self.name = name
        self.dist = dist
        self.half = half
        self.alpha = alpha
        self.x_min = -2
        self.x_max = 2

    def h(self, x):
        if self.name == "sin":
            return self.alpha / np.pi * np.sin(np.pi * x) + x

        if self.name == "sin2":
            return self.alpha / (2 * np.pi) * np.sin(2 * np.pi * x) + x

        if self.name == "poly":
            return -np.square(x) + 2 * x

        if self.name == "2x":
            return np.clip(2 * x, 0, 1)

        if self.name == "step4":
            return step(x, n=4)

        if self.name == "slope":
            p = x.copy()
            p[x <= 1 / 5] = 2 * x[x <= 1 / 5]
            p[(x > 1 / 5) & (x < 4 / 5)] = 1 / 3 * x[(x > 1 / 5) & (x < 4 / 5)] + 1 / 3
            p[x >= 4 / 5] = 2 * x[x >= 4 / 5] - 1
            return p

        if self.name == "slope10":
            return slope(x, n=10)

    def g(self, x):
        if self.half:
            return self.h(x)
        else:
            return 2 * x - self.h(x)

    def f(self, x):
        return 1 - np.exp(-np.square(0.9 * x))

    def f_star(self, x):
        p = self.h(self.f(x))
        p[x < 0] = self.g(self.f(x[x < 0]))
        return p

    def p(self, x):
        if self.dist == "gaussian":
            return norm.pdf(x)

        raise ValueError(f"Unknown {self.dist}")

    def generate_X_y(self, n, random_state=0):
        rng = check_random_state(random_state)
        if self.dist == "uniform":
            X = rng.uniform(self.x_min, self.x_max, size=n)

        elif self.dist == "gaussian":
            X = rng.normal(size=n)

        else:
            ValueError(f'Unsupported dist "{self.dist}"')
        y = rng.binomial(1, self.f_star(X))
        return X, y

    def analytical_gl(self):
        def diff(x):
            s = self.f(x)
            a = np.square(self.h(s) - s)
            b = np.square(self.g(s) - s)
            return 0.5 * (a + b)

        if self.dist == "uniform":
            dist = scipy.stats.uniform(
                loc=self.x_min, scale=(self.x_max - self.x_min)
            ).pdf

        elif self.dist == "gaussian":
            dist = scipy.stats.norm().pdf

        else:
            ValueError(f'Unsupported dist "{self.dist}"')

        GL = scipy.integrate.quad(lambda x: diff(x) * dist(x), -10, 10)

        return GL


class CustomUnconstrained(BaseExample):
    def __init__(self, name="poly", x_min=-1, x_max=1):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max

    def h(self, x):
        name = self.name

        if name == "square10":
            return np.power(x, 1 / 10)

        if name == "step10":
            return step(x, n=10)

        if name == "step100":
            return step(x, n=100)

        if name == "slope100":
            return slope(x, n=100)

        if name == "constant":
            q = np.full_like(x, 0.5)
            q[x == 0] = 0
            q[x == 1] = 1
            return q

        if name == "z":
            p = np.ones_like(x)
            p[x == 0] = 0
            return p

    def g(self, x):
        y = np.zeros_like(x)
        y[self.h(x) < x] = 1
        return y

    def f(self, x):
        return 1 - np.exp(-np.square(x))

    def f_star(self, x):
        p = self.h(self.f(x))
        p[x < 0] = self.g(self.f(x[x < 0]))
        return p

    def p(self, x):
        w_pos = np.divide(1, np.absolute(self.h(self.f(x)) - self.f(x)))
        w_neg = np.divide(1, np.absolute(self.g(self.f(x)) - self.f(x)))

        w_pos = np.nan_to_num(w_pos)
        w_neg = np.nan_to_num(w_neg)

        w_tot = w_pos + w_neg

        p = np.divide(w_pos, w_tot)
        p[x < 0] = np.divide(w_neg, w_tot)[x < 0]

        return p

    def generate_X_y(self, n, random_state=0):
        rng = check_random_state(random_state)
        XX1 = np.linspace(self.x_min, self.x_max, 1000)
        pp = self.p(XX1)
        pp_norm = pp / np.sum(pp)
        X = rng.choice(XX1, p=pp_norm, size=n)
        y = rng.binomial(1, self.f_star(X))
        return X, y
