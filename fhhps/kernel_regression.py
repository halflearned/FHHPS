import logging

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


class KernelRegression(Ridge):

    def __init__(self, alpha=1e-6, fit_intercept=True, normalize=True,
                 copy_X=True, max_iter=None, tol=0.001, solver="auto",
                 random_state=None, kernel="gaussian",
                 num_neighbors=None):

        self.kernel = kernel

        if kernel == "knn":
            self._nn = NearestNeighbors(num_neighbors + 1)
            self.num_neighbors = num_neighbors
        else:
            assert num_neighbors is None
            self._nn = None

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
            wts = np.zeros(shape=len(x_in))
            j, *idx = self._nn.kneighbors(x_out, return_distance=False)[0]
            for i in idx:
                if i < j:
                    wts[i] = 1
                else:
                    wts[i - 1] = 1
            return wts

        else:
            raise ValueError(f"Unknown kernel {self.kernel}")

    def fit_predict_local(self, X, y, bw=None):

        # X, y = check_X_y(X, y,
        #                  ensure_2d=True,
        #                  multi_output=True,
        #                  ensure_min_samples=10,
        #                  y_numeric=True,
        #                  force_all_finite=)

        if self.kernel == "knn":
            if bw is not None:
                raise TypeError("When kernel is knn, bw must be None")
            self._nn.fit(X)

        n, p = X.shape
        invalid_pts = 0
        yhat = np.full_like(y, fill_value=np.nan, dtype=np.float64)
        for i in range(n):
            if np.any(np.isnan(y[i])):
                continue

            if i % (n // 10) == 0:
                logging.info("KernelRegression.fit_predict_local[{i}]".format(i=i))
            X_in = np.vstack([X[:i], X[i + 1:]])
            y_in = np.vstack([y[:i], y[i + 1:]])

            valid = np.isfinite(X_in).all(1) & np.isfinite(y_in).all(1)
            X_in = X_in[valid]
            y_in = y_in[valid]

            X_out = X[[i]]
            wts = self.get_weights(x_in=X_in, x_out=X_out, param=bw)
            valid = ~np.isclose(wts, 0)
            if np.sum(valid) > p:
                model = super().fit(X_in[valid] - X_out, y_in[valid], sample_weight=wts[valid])
                yhat[i] = model.intercept_
            else:
                invalid_pts += 1

        if invalid_pts > 0:
            logging.warning("Number of invalid points: {}".format(invalid_pts))

        return yhat


def gaussian_kernel(a: np.ndarray, bw: float):
    return norm(scale=bw).pdf(a).prod(axis=1)


def uniform_kernel(a: np.ndarray, bw: float):
    return np.all(np.abs(a) < bw, axis=1).astype(float)


if __name__ == "__main__":
    n, p = 1000, 4
    X = np.random.normal(np.pi, np.pi, size=(n, p))
    mu = np.sin(X[:, 0:1])
    y = mu + 0.001 * np.random.normal(size=(n, 1))
    y[50] = np.nan
    reg = KernelRegression(kernel="uniform")
    yhat = reg.fit_predict_local(X, y, bw=1 * np.std(X))

    # # reg = KernelRegression(kernel="knn", num_neighbors=10)
    # # yhat = reg.fit_predict_local(X, y, bw=None)
    # plt.scatter(X[:, 0], mu.flatten())
    # plt.scatter(X[:, 0], yhat.flatten())
    # plt.show()
