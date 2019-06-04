import logging

from numpy.linalg import det as det

from fhhps.kernel_regression import KernelRegression, uniform_kernel
from fhhps.utils import *


class FHHPSEstimator:

    def __init__(self,
                 shock_const: float = 5.0,
                 shock_alpha: float = 0.2,
                 coef_const: float = 0.1,
                 coef_alpha: float = 0.5,
                 censor1_const: float = 1.,
                 censor2_const: float = 1.,
                 censor1_alpha: float = 0.25,
                 censor2_alpha: float = 0.125):
        self.shock_const = shock_const
        self.shock_alpha = shock_alpha
        self.coef_const = coef_const
        self.coef_alpha = coef_alpha
        self.censor1_const = censor1_const
        self.censor2_const = censor2_const
        self.censor1_alpha = censor1_alpha
        self.censor2_alpha = censor2_alpha

    def add_data(self, X, Z, Y):
        self.n, self.T = X.shape
        self.num_coef = 3
        self.X = np.array(X)
        self.Z = np.array(Z)
        self.Y = np.array(Y)
        self.XZ = np.hstack([X, Z])
        self.shock_bw = self.shock_const * self.n ** (-self.shock_alpha)
        self.coef_bw = self.coef_const * self.n ** (-self.coef_alpha)
        self.censor1_bw = self.censor1_const * self.n ** (-self.censor1_alpha)
        self.censor2_bw = self.censor2_const * self.n ** (-self.censor2_alpha)

    def fit(self, X, Z, Y):
        self.add_data(X, Z, Y)
        self.fit_shock_means()
        self.fit_output_cond_means()
        self.fit_coefficient_means()
        self.fit_shock_second_moments()
        self.fit_output_cond_cov()
        self.fit_coefficient_second_moments()

    @property
    def shock_means(self):
        return pd.DataFrame(data=self._shock_means,
                            index=["E[Ut]", "E[Vt]", "E[Wt]"])

    @property
    def shock_cov(self):
        return pd.DataFrame(data=self._shock_cov,
                            index=["Var[Ut]", "Var[Vt]", "Var[Wt]",
                                   "Cov[Ut, Vt]", "Cov[Ut, Wt]", "Cov[Vt, Wt]"])

    @property
    def coefficient_means(self):
        return pd.Series(self._coefficient_means,
                         index=["E[A1]", "E[B1]", "E[C1]"])

    @property
    def coefficient_cov(self):
        return pd.Series(self._coefficient_cov,
                         index=["Var[A1]", "Var[B1]", "Var[C1]",
                                "Cov[A1, B1]", "Cov[A1, C1]", "Cov[B1, C1]"])

    """ Shock moments """

    def fit_shock_means(self):
        """
        Populates attribute _shock_means with estimates of:
            0, E[U2], E[U3]
            0, E[V2], E[V3]
            0, E[W2], E[W3]
        """
        logging.info("--Fitting shock means--")
        self._shock_means = np.zeros(shape=(self.T, self.num_coef))
        for t in range(1, self.T):
            self._shock_means[:, t] = \
                get_shock_means(self.X, self.Z, self.Y, t=t, bw=self.shock_bw)

    def fit_shock_second_moments(self):
        """
        Populates attribute _shock_cov with estimates of:
            0,  Var[U2],       Var[U3]
            0,  Var[V2],       Var[V3]
            0,  Var[W2],       Var[W3]
            0,  Cov[U2, V2],   Cov[U3, V3]
            0,  Cov[U2, W2],   Cov[U3, W3]
            0,  Cov[V2, W2],   Cov[V3, W3]
        """
        logging.info("--Fitting shock second moments--")
        self._shock_second_moments = np.zeros(shape=(6, 3))
        for t in range(1, self.T):
            self._shock_second_moments[:, t] = \
                get_shock_second_moments(self.X, self.Z, self.Y, t=t, bw=self.shock_bw)

        self._shock_cov = np.zeros((6, 3))
        for t in range(self.T):
            self._shock_cov[:, t] = center_shock_second_moments(
                self._shock_means[:, t],
                self._shock_second_moments[:, t])

    """ Output moments """

    def fit_output_cond_means(self):
        logging.info("--Fitting output conditional means--")
        self.output_cond_mean = KernelRegression().fit_predict_local(
            self.XZ, self.Y, bw=self.coef_bw)

    def fit_output_cond_cov(self):
        logging.info("--Fitting output conditional second moments--")

        resid = (self.Y - self.output_cond_mean)
        Y_centered = np.empty((self.n, 6))
        Y_centered[:, :3] = resid ** 2  # Var[Yt|X]
        Y_centered[:, 3:] = np.column_stack([
            resid[:, 0] * resid[:, 1],  # Cov[Y1, Y2|X]
            resid[:, 0] * resid[:, 2],  # Cov[Y1, Y3|X]
            resid[:, 1] * resid[:, 2]])  # Cov[Y2, Y3|X]

        # Estimate Var[Yt|X] and Cov[Yt, Ys|X]
        self.output_cond_var = KernelRegression().fit_predict_local(
            self.XZ, Y_centered, bw=self.coef_bw)

    """ Random coefficients """

    def fit_coefficient_means(self):
        logging.info("--Fitting coefficient means--")
        self.coefficient_cond_means = np.empty(shape=(self.n, self.T))
        self.valid1 = np.zeros(self.n, dtype=bool)

        # Construct E[Y|X,Z] minus excess terms
        excess_terms = self.get_mean_excess_terms()
        output_cond_mean_clean = self.output_cond_mean - excess_terms

        # Compute conditional first moments
        for i in range(self.n):
            self.valid1[i] = np.abs(det(gamma1(self.X[i], self.Z[i]))) > self.censor1_bw
            self.coefficient_cond_means[i] = \
                gamma_inv(self.X[i], self.Z[i]) @ output_cond_mean_clean[i]

        # Average out to get unconditional moments
        self._coefficient_means = self.coefficient_cond_means[self.valid1].mean(0)

    def fit_coefficient_second_moments(self):
        logging.info("--Fitting coefficient second moments--")
        self.coefficient_cond_var = np.empty(shape=(self.n, 6))
        self.valid2 = np.zeros(self.n, dtype=bool)

        # Construct Var[Y|X,Z] minus excess terms
        excess_terms = self.get_cov_excess_terms()
        output_cond_var_clean = self.output_cond_var - excess_terms

        # Compute conditional second moments of random coefficients
        for i in range(self.n):
            self.valid2[i] = np.abs(det(gamma2(self.X[i], self.Z[i]))) > self.censor2_bw
            self.coefficient_cond_var[i] = \
                gamma2_inv(self.X[i], self.Z[i]) @ output_cond_var_clean[i]

        # Use EVVE and ECCE formulas to get unconditional moments
        ev = self.coefficient_cond_var[self.valid2, :3].mean(0)
        ve = self.coefficient_cond_means[self.valid2].var(0)
        variances = ev + ve

        ec = self.coefficient_cond_var[self.valid2, 3:].mean(0)
        ce = np.cov(self.coefficient_cond_means[self.valid2].T)[[0, 0, 1], [1, 2, 2]]
        covariances = ec + ce
        self._coefficient_cov = np.hstack([variances, covariances])

    """ Utils """

    def get_mean_excess_terms(self):
        """
        The 'excess' terms are those that need to be subtracted from E[Y|X]
            right before computing the random coefficients.
        For the first moments, the 'excess' terms are:

        [0,
        E[U2] + E[V2]*X2 + E[W2]*Z2,
        (E[U2] + E[U3]) + (E[V2] + E[V3])*X3 + (E[W2] + E[W3])*Z3]
        """
        excess_terms = np.zeros((self.n, 3))
        cum_shock_means = self._shock_means.cumsum(axis=1)
        for t in range(1, self.T):
            XZt = np.column_stack([np.zeros(self.n), self.X[:, t], self.Z[:, t]])
            excess_terms[:, t] += XZt @ cum_shock_means[:, t]
        return excess_terms

    def get_cov_excess_terms(self):
        """
        The 'excess' terms are those that need to be subtracted from E[Y^2|X]
            right before computing the random coefficients.
        For the second moments, the 'excess' terms look like:
        """
        def matrix(i, j):
            Xi = self.X[:, i]
            Xj = self.X[:, j]
            Zi = self.Z[:, i]
            Zj = self.Z[:, j]
            return np.column_stack(
                [np.ones_like(Xi), Xi * Xj, Zi * Zj, Xi + Xj, Zi + Zj, Xi * Zj + Xj * Zi])

        excess_terms = np.zeros((self.n, 6))
        excess_terms[:, 1] = matrix(1, 1) @ self._shock_cov[:, 1]
        excess_terms[:, 2] = matrix(1, 1) @ self._shock_cov[:, 1] + \
                             matrix(2, 2) @ self._shock_cov[:, 2]
        excess_terms[:, 5] = matrix(1, 2) @ self._shock_cov[:, 1]
        return excess_terms


def gamma1(x, z):
    """
    This is now called matrix M3 in the paper
    """
    return np.column_stack([np.ones_like(x), x, z])


def gamma_inv(x, z):
    return np.linalg.inv(gamma1(x, z))


def gamma2(x, z):
    """
    This is now called matrix M6 in the paper
    """
    f = lambda i, j: [1,
                      x[i] * x[j],
                      z[i] * z[j],
                      x[i] + x[j],
                      z[i] + z[j],
                      x[i] * z[j] + x[j] * z[i]]
    g = np.array([f(0, 0), f(1, 1), f(2, 2), f(0, 1), f(0, 2), f(1, 2)])
    return g


def gamma2_inv(x, z):
    return np.linalg.inv(gamma2(x, z))


def get_shock_means(X, Z, Y, t: int, bw: float):
    """
    Creates a 3-vector of shock means
    [E[Ut], E[Vt], E[Wt]]
    """
    n, _ = X.shape
    DYt = difference(Y, t=t)
    DXZt = np.hstack(difference(X, Z, t=t))
    XZt = np.hstack(extract(X, Z, t=t))
    kern = KernelRegression()
    # wts = kern.get_weights(DXZt, param=bw)
    wts = uniform_kernel(DXZt, bw)
    moments = kern.fit(XZt, DYt, sample_weight=wts).coefficients
    return moments


def get_shock_second_moments(X, Z, Y, t: int, bw: float):
    """
    Creates 6-vector of shock second moments
    [E[Ut^2], E[Vt^2], E[Wt^2], E[Ut*Vt], E[Ut*Wt], E[Vt*Wt]]
    """
    n, _ = X.shape
    DYt = difference(Y, t=t)
    DXZt = np.hstack(difference(X, Z, t=t))
    XZt = np.hstack(extract(X ** 2, Z ** 2, 2 * X, 2 * Z, 2 * X * Z, t=t))
    kern = KernelRegression()
    # wts = kern.get_weights(DXZt, bw)
    wts = uniform_kernel(DXZt, bw)
    moments = kern.fit(XZt, DYt ** 2, sample_weight=wts).coefficients
    return moments


def center_shock_second_moments(m1, m2):
    """
    Creates 6-vector of shock variances and covariances
    [Var[Ut], Var[Vt], Var[Wt], Cov[Ut, Vt], Cov[Ut, Wt], Cov[Vt, Wt]]
    """
    VarUt = m2[0] - m1[0] ** 2
    VarVt = m2[1] - m1[1] ** 2
    VarWt = m2[2] - m1[2] ** 2
    CovUVt = m2[3] - m1[0] * m1[1]
    CovUWt = m2[4] - m1[0] * m1[1]
    CovVWt = m2[5] - m1[1] * m1[2]
    return np.array([VarUt, VarVt, VarWt, CovUVt, CovUWt, CovVWt])

