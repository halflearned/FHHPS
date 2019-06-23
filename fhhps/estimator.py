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
        self.censor1_thres = self.censor1_const * self.n ** (-self.censor1_alpha)
        self.censor2_thres = self.censor2_const * self.n ** (-self.censor2_alpha)

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
            self._shock_means[:, t] = get_shock_means(self.X, self.Z, self.Y, t=t, bw=self.shock_bw)

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
        """
        Estimates

        Output is a matrix whose ith row is

        Var(Y[0]|I)
        Var(Y[1]|I)
        Var(Y[2]|I)
        Cov(Y[0], Y[1]|I)
        Cov(Y[0], Y[2]|I)
        Cov(Y[1], Y[2]|I)

        where I = {X[1],Z[1],X[2],Z[2],X[3],Z[3]}.

        """
        logging.info("--Fitting output conditional second moments--")

        resid = (self.Y - self.output_cond_mean)
        Y_centered = np.empty((self.n, 6))
        Y_centered[:, :3] = resid ** 2  # Var[Yt|I]
        Y_centered[:, 3:] = np.column_stack([
            resid[:, 0] * resid[:, 1],  # Cov[Y1, Y2|I]
            resid[:, 0] * resid[:, 2],  # Cov[Y1, Y3|I]
            resid[:, 1] * resid[:, 2],  # Cov[Y2, Y3|I]
        ])
        self.output_cond_var = KernelRegression().fit_predict_local(
            self.XZ, Y_centered, bw=self.coef_bw)

    """ Random coefficients """

    def fit_coefficient_means(self):
        logging.info("--Fitting coefficient means--")
        self.coefficient_cond_means = np.empty(shape=(self.n, self.T))
        self.valid1 = np.zeros(self.n, dtype=bool)

        # Construct E[Y|X,Z] minus excess terms
        excess_terms = self.get_mean_excess_terms(self.X, self.Z, self._shock_means)
        output_cond_mean_clean = self.output_cond_mean - excess_terms

        # Compute conditional first moments
        for i in range(self.n):
            if not np.all(np.isfinite(output_cond_mean_clean[i])):
                continue
            self.valid1[i] = np.abs(det(m3(self.X[i], self.Z[i]))) > self.censor1_thres
            self.coefficient_cond_means[i] = \
                m3_inv(self.X[i], self.Z[i]) @ output_cond_mean_clean[i]

        # Average out to get unconditional moments
        self._coefficient_means = self.coefficient_cond_means[self.valid1].mean(0)

    def fit_coefficient_second_moments(self):
        logging.info("--Fitting coefficient second moments--")
        self.coefficient_cond_var = np.empty(shape=(self.n, 6))
        self.valid2 = np.zeros(self.n, dtype=bool)

        # Construct Var[Y|X,Z] minus excess terms
        excess_terms = get_cov_excess_terms(self.X, self.Z, self._shock_cov)
        output_cond_var_clean = self.output_cond_var - excess_terms

        # Compute conditional second moments of random coefficients
        for i in range(self.n):
            if not np.all(np.isfinite(output_cond_var_clean[i])):
                continue
            self.valid2[i] = np.abs(det(m6(self.X[i], self.Z[i]))) > self.censor2_thres
            self.coefficient_cond_var[i] = \
                m6_inv(self.X[i], self.Z[i]) @ output_cond_var_clean[i]

        # Use EVVE and ECCE formulas to get unconditional moments
        ev = self.coefficient_cond_var[self.valid2, :3].mean(0)
        ve = self.coefficient_cond_means[self.valid2].var(0)
        variances = ev + ve

        ec = self.coefficient_cond_var[self.valid2, 3:].mean(0)
        ce = np.cov(self.coefficient_cond_means[self.valid2].T)[[0, 0, 1], [1, 2, 2]]
        covariances = ec + ce
        self._coefficient_cov = np.hstack([variances, covariances])


""" Utils """


@njit()
def get_coefficient_cond_means(X, Z, output_cond_means, shock_means):
    """
    Fit random coefficient conditional means
    """
    # Construct conditional output minus excess terms:
    # E[Y|I] - E
    n, T = X.shape
    excess_terms = get_mean_excess_terms(X, Z, shock_means)
    output_cond_mean_clean = output_cond_means - excess_terms

    # Compute conditional first moments of random coefficients:
    # E[A,B|I] = Gamma^{-1} * (E[Y|I] - E)
    coefficient_cond_means = np.empty(shape=(n, T))
    for i in range(n):
        coefficient_cond_means[i] = m3_inv(X[i], Z[i]) @ output_cond_mean_clean[i]

    return coefficient_cond_means


@njit()
def get_coefficient_cond_cov(X, Z, output_cond_var, shock_cov):
    """
    Fit random coefficient conditional variances and covariances
    """
    coefficient_cond_var = np.empty(shape=(n, 6))

    # Construct Var[Y|X,Z] minus excess terms
    excess_terms = get_cov_excess_terms(X, Z, shock_cov)
    output_cond_var_clean = output_cond_var - excess_terms

    # Compute conditional second moments of random coefficients
    for i in range(n):
        coefficient_cond_var[i] = m6_inv(X[i], Z[i]) @ output_cond_var_clean[i]

    return coefficient_cond_var


def get_unconditional_first_moments(coef_cond_means, valid=None):
    return coef_cond_means[valid].mean(0)


def get_unconditional_second_moments(coef_cond_means, coef_cond_cov, valid=None):
    """
    Use ANOVA-type formulas to get unconditional variances and covariances
    """
    if valid is None:
        valid = np.ones(len(coef_cond_means), dtype=bool)

    ev = coef_cond_cov[valid, :3].mean(0)
    ve = coef_cond_means[valid].var(0)
    variances = ev + ve

    ec = coef_cond_cov[valid, 3:].mean(0)
    ce = np.cov(coef_cond_means[valid].T)[[0, 0, 1], [1, 2, 2]]
    covariances = ec + ce

    coefficient_cov = np.hstack([variances, covariances])
    return coefficient_cov


@njit()
def get_valid_cond_means(X, Z, censor_threshold):
    n = len(X)
    valid = np.zeros(n, np.bool_)
    for i in range(n):
        valid[i] = np.abs(det(m3(X[i], Z[i]))) > censor_threshold
    return valid


@njit()
def get_valid_cond_cov(X, Z, censor_threshold):
    n = len(X)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        valid[i] = np.abs(det(m6(X[i], Z[i]))) > censor_threshold
    return valid


@njit()
def get_mean_excess_terms(X, Z, shock_means):
    """
    The 'excess' terms are those that need to be subtracted from E[Y|X]
        right before computing the random coefficients.
    For the first moments, the 'excess' terms are:

    E1 = 0
    E2 = E[U2] + E[V2]*X2 + E[W2]*Z2
    E3 = (E[U2] + E[U3]) + (E[V2] + E[V3])*X3 + (E[W2] + E[W3])*Z3
    """
    n = len(X)
    EU2, EV2, EW2 = shock_means[:, 1]
    EU3, EV3, EW3 = shock_means[:, 2]
    X2, X3 = X[:, 1], X[:, 2]
    Z2, Z3 = Z[:, 1], Z[:, 2]

    E1 = np.zeros(n)
    E2 = EU2 + EV2 * X2 + EW2 * Z2
    E3 = (EU2 + EU3) + (EV2 + EV3) * X3 + (EW2 + EW3) * Z3

    excess_terms = np.column_stack((E1, E2, E3))
    return excess_terms


@njit()
def get_cov_excess_terms(X, Z, shock_cov):
    """
    Creates a matrix of excess terms for the second moments.
    See supplementary material.
    """
    n = len(X)
    X2, X3 = X[:, 1], X[:, 2]
    Z2, Z3 = Z[:, 1], Z[:, 2]
    VarU2, VarV2, VarW2, CovU2V2, CovU2W2, CovV2W2 = shock_cov[:, 1]
    VarU3, VarV3, VarW3, CovU3V3, CovU3W3, CovV3W3 = shock_cov[:, 2]

    E1 = np.zeros(n)

    E2 = VarU2 + VarV2 * X2 ** 2 + VarW2 * Z2 ** 2 + \
         2 * CovU2V2 * X2 + 2 * CovU2W2 * Z2 + 2 * CovV2W2 * X2 * Z2

    E3 = VarU2 + VarV2 * X2 ** 2 + VarW2 * Z2 ** 2 \
         + 2 * CovU2V2 * X2 + 2 * CovU2W2 * Z2 + 2 * CovV2W2 * X2 * Z2 \
         + VarU3 + VarV3 * X3 ** 2 + VarW3 * Z3 ** 2 \
         + 2 * CovU3V3 * X3 + 2 * CovU3W3 * Z3 + 2 * CovV3W3 * X3 * Z3

    E4 = np.zeros(n)

    E5 = np.zeros(n)

    E6 = VarU2 + VarV2 * X2 * X3 + VarW2 * Z2 * Z3 \
         + CovU2V2 * (X2 + X3) + CovU2W2 * (Z2 + Z3) \
         + CovV2W2 * (X2 * Z3 + Z2 * X3)

    excess_terms = np.column_stack((E1, E2, E3, E4, E5, E6))
    return excess_terms


@njit()
def m3(x, z):
    """
    This is now called matrix M3 in the paper
    """
    return np.column_stack((np.ones_like(x), x, z))


@njit()
def m3_inv(x, z):
    return np.linalg.inv(m3(x, z))


@njit()
def m6(x, z):
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


def m6_inv(x, z):
    return np.linalg.inv(m6(x, z))


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulation config
    n = 2000
    fake = generate_data(n=n)
    data = fake["df"]

    # Algorithm configuration
    est = FHHPSEstimator(shock_const=5.0,
                         shock_alpha=0.2,
                         coef_const=20.,
                         coef_alpha=0.5,
                         censor1_const=3.0,
                         censor2_const=3.0)
    est.add_data(X=data[["X1", "X2", "X3"]],
                 Z=data[["Z1", "Z2", "Z3"]],
                 Y=data[["Y1", "Y2", "Y3"]])

    # Computing all objects
    t1 = time()
    est.fit_shock_means()
    est.fit_shock_second_moments()
    est.fit_output_cond_means()
    est.fit_coefficient_means()
    est.fit_output_cond_cov()
    est.fit_coefficient_second_moments()
    t2 = time()
    print(f"Fitting took {t2 - t1} seconds")

    print("SHOCKS")
    print("Means:")
    print(est.shock_means)
    print("Covariances:")
    print(est.shock_cov)

    print("RANDOM COEFFICIENTS")
    print("Means:")
    print(est.coefficient_means)
    print("Covariances:")
    print(est.coefficient_cov)
