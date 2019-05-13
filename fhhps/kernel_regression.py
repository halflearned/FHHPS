import logging
from typing import *

import numpy as np
from numba import njit
from sklearn.linear_model import Ridge
from sklearn.utils import check_X_y


class KernelRegression(Ridge):

    def __init__(self, alpha=1e-6, fit_intercept=True, normalize=True,
                 copy_X=True, max_iter=None, tol=0.001, solver="auto",
                 random_state=None, kernel="gaussian"):
        self.kernel = kernel
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state)

    @property
    def coefficients(self):
        if len(self.intercept_) == 1:
            return np.hstack([self.intercept_, self.coef_.flatten()])
        else:
            return np.hstack([self.intercept_.reshape(-1, 1), self.coef_])

    def get_weights(self, cond, bw):
        return kernel_density(cond, bw=bw)

    def fit_predict_local(self, X, y, bw: Union[float, str] = "scott"):
        X, y = check_X_y(X, y, ensure_2d=True, multi_output=True, ensure_min_samples=10,
                         y_numeric=True)
        n = len(X)
        yhat = np.empty_like(y, dtype=np.float64)
        if isinstance(bw, str):
            bw = bandwidth_selector(X, method=bw)
        for i in range(n):
            if i % (n // 10) == 0:
                logging.info("KernelRegression.fit_predict_local[{i}]".format(i=i))
            X_in = np.vstack([X[:i], X[i + 1:]])
            y_in = np.vstack([y[:i], y[i + 1:]])
            X_out = X[[i]]
            wts = self.get_weights(cond=X_in - X_out, bw=bw)
            yhat[i] = super().fit(X_in, y_in, sample_weight=wts).predict(X_out)
        return yhat


@njit()
def kernel_density(X: np.ndarray, bw: float):
    p = 1 / (2 * np.pi * bw) * np.exp(-X ** 2 / (2 * bw ** 2))
    k = X.shape[1]
    out = p[:, 0]
    for i in range(1, k):
        out *= p[:, i]
    return out


@njit()
def bandwidth_selector(X, method="scott"):
    """
    A = minimum of std(X, ddof=1) and normalized IQR(X)
    C = depends on method

    References
    ----------
    Silverman (1986) p.47
    Scott, D.W. (1992) Multivariate Density Estimation: Theory, Practice, and
        Visualization.
    """
    if method == "scott":
        C = 1.059
    elif method == "silverman":
        C = 0.9
    else:
        C = 1

    k = X.shape[1]
    n = X.shape[0]
    bw = np.empty(shape=k, dtype=np.float64)
    for i in range(k):
        x = X[:, i]
        A1 = np.diff(np.percentile(x, q=[25, 75])) / 1.349
        A2 = np.std(x)
        A = np.minimum(A1, A2).item()
        bw[i] = C * A * n ** (-0.2)
    return bw
