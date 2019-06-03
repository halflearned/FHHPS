import logging

import numpy as np
from numba import njit
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y


class KernelRegression(Ridge):

    def __init__(self, alpha=1e-6, fit_intercept=True, normalize=True,
                 copy_X=True, max_iter=None, tol=0.001, solver="auto",
                 random_state=None, kernel="gaussian",
                 num_neighbors=None,
                 bw_selection_method=None):
        self.kernel = kernel
        if kernel == "neighbors":
            assert bw_selection_method is None
            self._nn = NearestNeighbors(num_neighbors)
            self.bw_selection_method = None
        else:
            assert num_neighbors is None
            self._nn = None
            self.bw_selection_method = None

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

    def get_weights(self, x_in, x_out, param=None):
        if self.kernel == "gaussian":
            return gaussian_kernel(x_in - x_out, bw=param)
        elif self.kernel == "uniform":
            return uniform_kernel(x_in - x_out, bw=param)
        elif self.kernel == "knn":
            return self._nn.kneighbors(x_out, param, return_distance=False)[0, 1:]

    def fit_predict_local(self, X, y, bw=None):
        X, y = check_X_y(X, y,
                         ensure_2d=True,
                         multi_output=True,
                         ensure_min_samples=10,
                         y_numeric=True)
        if self.kernel == "neighbor" and bw is not None:
            raise ValueError("Knn kernel does not accept bw.")
        else:
            self._nn.fit(X)

        n = len(X)
        yhat = np.empty_like(y, dtype=np.float64)
        for i in range(n):
            if i % (n // 10) == 0:
                logging.info("KernelRegression.fit_predict_local[{i}]".format(i=i))
            X_in = np.vstack([X[:i], X[i + 1:]])
            y_in = np.vstack([y[:i], y[i + 1:]])
            X_out = X[[i]]
            wts = self.get_weights(x_in=X_in, x_out=X_out, param=bw)
            yhat[i] = super().fit(X_in, y_in, sample_weight=wts).predict(X_out)
        return yhat


@njit()
def gaussian_kernel(X: np.ndarray, bw: float):
    p = 1 / (2 * np.pi * bw) * np.exp(-X ** 2 / (2 * bw ** 2))
    k = X.shape[1]
    out = p[:, 0]
    for i in range(1, k):
        out *= p[:, i]
    return out


def uniform_kernel(X: np.ndarray, bw: float):
    return (np.abs(X) < bw).astype(float)
