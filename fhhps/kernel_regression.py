import logging

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge


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

    def get_weights(self, x_in, x_out, param=None):

        if self.kernel == "gaussian":
            wts = gaussian_kernel(x_in - x_out, bw=param)

        elif self.kernel == "uniform":
            wts = uniform_kernel(x_in - x_out, bw=param)

        else:
            raise ValueError(f"Unknown kernel {self.kernel}")

        wts = wts / wts.sum()
        return wts

    def fit_predict_local(self, X, y, bw=None):

        n, p = X.shape
        # invalid_pts = 0
        yhat = np.full_like(y, fill_value=np.nan, dtype=np.float64)
        for i in range(n):

            if i % (n // 10) == 0:
                logging.info("KernelRegression.fit_predict_local[{i}]".format(i=i))
            X_train = np.vstack([X[:i], X[i + 1:]])
            y_train = np.vstack([y[:i], y[i + 1:]])
            X_eval = X[[i]]

            wts = self.get_weights(x_in=X_train, x_out=X_eval, param=bw)
            valid = ~np.isclose(wts, 0)
            # if np.sum(valid) > p:
            model = super().fit(X_train[valid] - X_eval, y_train[valid], sample_weight=wts[valid])
            yhat[i] = model.intercept_
            # else:
            #    invalid_pts += 1

        # if invalid_pts > 0:
        #     logging.warning("Number of invalid points: {}".format(invalid_pts))

        return yhat


def gaussian_kernel(a: np.ndarray, bw: float):
    H = np.diag(bw * np.var(a, 0))
    k = multivariate_normal(cov=H).pdf(a)
    return k


def uniform_kernel(a: np.ndarray, bw: float):
    H = bw * np.std(a, 0)
    k = np.all(np.abs(a) < H, axis=1).astype(float)
    return k


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
