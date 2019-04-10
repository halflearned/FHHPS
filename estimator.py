import numpy as np
from utils import *

from kernel_regression import KernelRegression


class FHHPSEstimator:

    def __init__(self):
        pass

    def add_data(self, X, Z, Y):
        self.n, self.T = X.shape
        self.num_coef = 3
        self.X = X
        self.Z = Z
        self.Y = Y
        self.XZ = np.hstack([X, Z])

    def fit(self, X, Z, Y):
        self.add_data(X, Z, Y)
        self.fit_shock_first_moments()
        self.fit_output_cond_first_moments()
        self.fit_coefficient_first_moments()
        self.fit_shock_second_moments()
        self.fit_output_cond_second_moments()
        self.fit_coefficient_second_moments()

    """ Output moments """

    def fit_output_cond_first_moments(self):
        self.output_cond_first_moments = KernelRegression().fit_predict_local(self.XZ, self.Y)

    def fit_output_cond_second_moments(self):
        output_resid = (self.Y - self.output_cond_first_moments)
        output_resid_sq = output_resid ** 2
        output_resid_cross = np.vstack([output_resid[:, 1] * output_resid[:, 0],
                                        output_resid[:, 2] * output_resid[:, 0],
                                        output_resid[:, 2] * output_resid[:, 1]]).T
        cond_variances = KernelRegression().fit_predict_local(self.XZ, output_resid_sq)
        cond_covariances = KernelRegression().fit_predict_local(self.XZ, output_resid_cross)
        self.output_cond_second_moments = np.hstack([cond_variances, cond_covariances])

    """ Shock moments """

    def fit_shock_first_moments(self):
        """
        Populates attribute shock_first_moments with estimates of:
            0, E[U2], E[U3]
            0, E[V2], E[V3]
            0, E[W2], E[W3]
        """
        self.shock_first_moments = np.zeros(shape=(self.T, self.num_coef))
        for t in range(1, self.T):
            self.shock_first_moments[:, t] = get_shock_first_moments(self.X, self.Z, self.Y, t=t)
        self.cum_shock_first_moments = self.shock_first_moments.cumsum(axis=1)

    def fit_shock_second_moments(self):
        """
        Populates attribute shock_second_moments with estimates of:
            0,  Var[U2],       Var[U3]
            0,  Var[V2],       Var[V3]
            0,  Var[W2],       Var[W3]
            0,  Cov[U2, V2],   Cov[U3, V3]
            0,  Cov[U2, W2],   Cov[U3, W3]
            0,  Cov[V2, W2],   Cov[V3, W3]
        """
        self.shock_second_moments = np.zeros(shape=(6, 3))
        for t in range(1, self.T):
            self.shock_second_moments[:, t] = get_shock_second_moments(self.X, self.Z, self.Y, t=t)

    """ Random coefficients """

    def fit_coefficient_first_moments(self):
        self.coefficient_cond_first_moments = np.empty(shape=(self.n, self.T))

        # Construct E[Y|X,Z] minus shocks
        cond_mean_output_minus_shocks = self.output_cond_first_moments.copy()
        cond_mean_output_minus_shocks -= self.cum_shock_first_moments[0]  # Subtracting sum[t] E[Ut]
        cond_mean_output_minus_shocks -= (X * self.cum_shock_first_moments[1])  # Subtracting sum[t] E[Vt]*X[s]
        cond_mean_output_minus_shocks -= (Z * self.cum_shock_first_moments[2])  # Subtracting sum[t] E[Wt]*Z[s]

        # Compute conditional first moments
        for i in range(self.n):
            self.coefficient_cond_first_moments[i] = \
                gamma_inv(X[i], Z[i]) @ cond_mean_output_minus_shocks[i]

        # Average out to get unconditional moments
        self.coefficient_first_moments = self.coefficient_cond_first_moments.mean(0)

    def fit_coefficient_second_moments(self):
        self.coefficient_cond_second_moments = np.empty(shape=(self.n, 6))

        # Construct Var[Y|X,Z] minus shocks
        cond_var_output = self.output_cond_second_moments.copy()
        second_shock_terms = self.get_second_shock_terms()
        cond_var_output_minus_shock_terms = cond_var_output - second_shock_terms

        # Compute conditional second moments of random coefficients
        for i in range(self.n):
            self.coefficient_cond_second_moments[i] = \
                gamma_inv2(X[i], Z[i]) @ cond_var_output_minus_shock_terms[i]

        # Use EVVE and ECCE formulas to get unconditional moments
        variances = \
            self.coefficient_cond_second_moments[:, :3].mean(0) + self.coefficient_cond_first_moments.var(0)
        covariances = \
            self.coefficient_cond_second_moments[:, 3:].mean(0) + \
            np.cov(self.coefficient_cond_first_moments.T)[[0, 0, 1], [1, 2, 2]]
        self.coefficient_second_moments = np.hstack([variances, covariances])

    """ Utils """

    def get_second_shock_terms(self):
        def matrix(xi, xj, zi, zj):
            Xi = self.X[:, xi, None]
            Xj = self.X[:, xj, None]
            Zi = self.Z[:, zi, None]
            Zj = self.Z[:, zj, None]
            return np.hstack([np.ones_like(Xi), Xi * Xj, Zi * Zj, Xi + Xj, Zi * Zj, Xi * Zj + Xj * Zi])

        shock_terms = np.zeros((self.n, 6))
        shock_terms[:, 1] = (matrix(1, 1, 1, 1) @ self.shock_second_moments[:, 1, None]).flatten()
        shock_terms[:, 2] = (matrix(1, 1, 1, 1) @ self.shock_second_moments[:, 1, None] +
                             matrix(2, 2, 2, 2) @ self.shock_second_moments[:, 2, None]).flatten()
        shock_terms[:, 5] = (matrix(1, 2, 1, 2) @ self.shock_second_moments[:, 1, None]).flatten()
        return shock_terms


def gamma_inv(x, z):
    g = np.vstack([np.ones_like(x), x, z]).T
    return np.linalg.inv(g)


def gamma_inv2(x, z):
    f = lambda i, j: [1, x[i] * x[j], z[i] * z[j], x[i] + x[j], z[i] + z[j], x[i] * z[j] + x[j] * z[i]]
    g = np.array([f(0, 0), f(1, 1), f(2, 2), f(0, 1), f(0, 2), f(1, 2)])
    return np.linalg.inv(g)


def get_shock_first_moments(X, Z, Y, t):
    """
    Creates a 3-vector of shock means
    [E[Ut], E[Vt], E[Wt]]
    """
    n, _ = X.shape
    DYt = difference(Y, t=t)
    DXZ = np.hstack(difference(X, Z, t=t))
    XZt = np.hstack(extract(X, Z, t=t))
    kern = KernelRegression()
    w = kern.get_weights(DXZ)
    moments = kern.fit(XZt, DYt, sample_weight=w).coefficients
    return moments


def get_shock_second_moments(X, Z, Y, t):
    """
    Creates 6-vector of shock variances and covariances
    [Var[Ut], Var[Vt], Var[Wt], Cov[Ut, Vt], Cov[Ut, Wt], Cov[Vt, Wt]]
    """
    n, _ = X.shape
    DYt = difference(Y, t=t)
    DXZ = np.hstack(difference(X, Z, t=t))
    XZt = np.hstack(extract(X ** 2, Z ** 2, 2 * X, 2 * Z, X * Z, t=t))
    kern = KernelRegression()
    w = kern.get_weights(DXZ)
    moments = kern.fit(XZt, DYt, sample_weight=w).coefficients
    return moments





if __name__ == "__main__":
    n = 500000
    X, Z, Y = fake_data(n)
    s = get_shock_first_moments(X, Z, Y, 2)
    #est = FHHPSEstimator()
    #est.fit(X, Z, Y)
