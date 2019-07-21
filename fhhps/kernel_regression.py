import logging

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


class KernelRegression(Ridge):

    def __init__(self, kernel,
                 alpha=1e-6, fit_intercept=True, normalize=True,
                 copy_X=True, max_iter=None, tol=0.001, solver="auto",
                 random_state=None):

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

    def get_weights(self, x_in, x_out, param=None):

        if self.kernel == "gaussian":
            wts = gaussian_kernel(x_in - x_out, bw=param)

        elif self.kernel == "uniform":
            wts = uniform_kernel(x_in - x_out, bw=param)

        elif self.kernel == "neighbor":
            wts = knn_kernel(x_in, x_out, bw=param)

        else:
            raise ValueError(f"Unknown kernel {self.kernel}")

        wts_sum = wts.sum()
        wts /= wts_sum
        return wts

    def fit_predict_local(self, X, y, bw=None):

        n, p = X.shape
        small_bw_pts = 0
        yhat = np.full_like(y, fill_value=np.nan, dtype=np.float64)
        for i in range(n):

            if i % (n // 10) == 0:
                logging.info("KernelRegression.fit_predict_local[{i}]".format(i=i))
            X_train = np.vstack([X[:i], X[i + 1:]])
            y_train = np.vstack([y[:i], y[i + 1:]])
            X_eval = X[[i]]

            wts = self.get_weights(x_in=X_train, x_out=X_eval, param=bw)
            valid = wts > 1e-16
            if np.any(valid):
                model = super().fit(X_train[valid] - X_eval, y_train[valid],
                                    sample_weight=wts[valid])
                yhat[i] = model.intercept_
            else:
                logging.warning("Could not predict for observation {}.".format(i))
                yhat[i] = np.nan

            if np.sum(valid) < p:
                small_bw_pts += 1

        if small_bw_pts > 0:
            logging.warning("The bandwidth was too small for {} points.".format(small_bw_pts))

        return yhat


def gaussian_kernel(a: np.ndarray, bw: float):
    H = np.diag(bw * np.var(a, 0))
    k = multivariate_normal(cov=H).pdf(a)
    return k


def uniform_kernel(a: np.ndarray, bw: float):
    H = bw * np.std(a, 0)
    k = np.all(np.abs(a) < H, axis=1).astype(float)
    return k


def knn_kernel(a, b, bw):
    # The 'bandwidth' for a knn problem is
    # the usual kernel bandwidth times the number of obs
    n = len(a)
    knn_bw = max(min(int(n * bw), n - 1), 1)
    knn = NearestNeighbors(n_neighbors=knn_bw).fit(a)
    _, idx = knn.kneighbors(b)
    k = np.zeros(n, dtype=float)
    k[idx.flatten()] = 1.
    return k
