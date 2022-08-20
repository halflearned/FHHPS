import logging

import numpy as np
import pandas as pd
#from numba import njit
from time import time
from collections import OrderedDict as ODict
from numpy.linalg import det as det
from sklearn.preprocessing import PolynomialFeatures

from fhhps.kernel_regression import KernelRegression, gaussian_kernel, knn_kernel
from fhhps.utils import *



def fhhps(X: np.ndarray,
          Z: np.ndarray,
          Y: np.ndarray,

          # Parameters
          kernel1: str,
          kernel2: str,
          shock_bw1_const: float,
          shock_bw2_const: float,
          output_bw1_const_step1: float,
          output_bw1_const_step2: float,
          output_bw2_const: float,
          censor1_const: float,
          censor2_const: float,
          poly: int = 1,

          # Asymptotically optimal exponents
          shock_bw1_alpha: float = 1 / 6,
          shock_bw2_alpha: float = 1 / 6,
          output_bw1_alpha: float = 1 / 10,
          output_bw2_alpha: float = 1 / 10,
          censor1_alpha: float = 1 / 5,
          censor2_alpha: float = 1 / 5):

    """
    Parameters
    ----------
    X, Z, Y: np.ndarray
        Data. Each array should be (n, 3).

    kernel1: str ["gaussian", "neighbor"]
        Kernel to use when estimating shocks.

    kernel2: str ["gaussian", "neighbor"]
        Kernel to use when estimating dependent variable conditional means.

    shock_bw{1,2}_const, shock_bw{1,2}_alpha: float
        Parameters used to compute the bandwidth for shock moment estimation.
        shock_bw1 = shock_bw1_const * n ^ (-shock_bw1_alpha)  [First moments]
        shock_bw2 = shock_bw2_const * n ^ (-shock_bw2_alpha)  [Second moments]

    output_bw{1,2}_const, output_bw{1,2}_alpha: float
        Parameters used to compute bandwidth for Y-conditional moment estimation.
        Note that conditional *means* are used *twice*.
        First, during the computation of random coefficient means.
        In this step we simply need:
            E[Y|X,Z]
            -------- (1)
        Second, during the computation of random coefficient second moments.
        In the latter step we need to compute:
            Var[Y|X,Z] = E[(Y - E[Y|X,Z])^2|X,Z]
                                -------- (2)
        This inner conditional expectation can use a different bandwidth.

        Parameter (1) is computed using
        output_bw1_step1 = output_bw1_const_step1 * n ** (-output_bw1_alpha)

        Parameter (2) is computed using
        output_bw1_step2 = output_bw1_const_step2 * n ** (-output_bw1_alpha)

        Finally the outer expectation is (2) is computed using bandwidth:
        output_bw2 = output_bw2_const * n ** (-output_bw2_alpha)

    censor{1,2}_const, censor{1,2}_alpha: float
        Parameters used to compute "bandwidth" for censoring.
        censor1_bw = censor1_const * n ** (-censor1_alpha)
        censor2_bw = censor2_const * n ** (-censor2_alpha)

    poly: int
        Degree of local polynomial regression

    Returns
    -------
    results: pd.DataFrame
        A dataframe in long format with estimates.
    """

    t1 = time()
    n = len(X)

    # Compute parameters
    output_bw1_step1 = output_bw1_const_step1 * n ** (-output_bw1_alpha)
    output_bw1_step2 = output_bw1_const_step2 * n ** (-output_bw1_alpha)
    output_bw2 = output_bw2_const * n ** (-output_bw2_alpha)

    shock_bw1 = shock_bw1_const * n ** (-shock_bw1_alpha)
    shock_bw2 = shock_bw2_const * n ** (-shock_bw2_alpha)

    censor1_bw = censor1_const * n ** (-censor1_alpha)
    censor2_bw = censor2_const * n ** (-censor2_alpha)

    # Fit shock moments
    shock_means = fit_shock_means(X, Z, Y, bw=shock_bw1, kernel=kernel1)
    shock_cov = fit_shock_cov(X, Z, Y, shock_means, bw=shock_bw2, kernel=kernel1)

    # Estimate conditional means
    output_cond_means_step1 = fit_output_cond_means(
        X, Z, Y, bw=output_bw1_step1, kernel=kernel2)
    output_cond_means_step2 = fit_output_cond_means(
        X, Z, Y, bw=output_bw1_step2, kernel=kernel2)

    # Estimate conditional second moments
    output_cond_cov = fit_output_cond_cov(
        X, Z, Y, output_cond_means_step2,
        bw=output_bw2, kernel=kernel2, poly=poly)

    # Valid (non-censored) indices
    mean_valid = get_valid_cond_means(X, Z, censor1_bw)
    cov_valid = get_valid_cond_cov(X, Z, censor2_bw)

    # Average over valid indices
    coef_cond_means_step1 = get_coef_cond_means(X, Z, output_cond_means_step1, shock_means)
    mean_estimate = get_coef_means(coef_cond_means_step1, mean_valid)

    coef_cond_means_step2 = get_coef_cond_means(X, Z, output_cond_means_step2, shock_means)
    coef_cond_cov = get_coef_cond_cov(X, Z, output_cond_cov, shock_cov)
    cov_estimate = get_coef_cov(coef_cond_means_step2, coef_cond_cov, cov_valid)

    t2 = time()
    config = ODict(**{"n": n,
                      "kernel1": kernel1,
                      "kernel2": kernel2,
                      "output_bw1_const_step1": output_bw1_const_step1,
                      "output_bw1_const_step2": output_bw1_const_step2,
                      "output_bw2_const": output_bw2_const,
                      "output_bw1_alpha": output_bw1_alpha,
                      "output_bw2_alpha": output_bw2_alpha,
                      "shock_bw1_const": shock_bw1_const,
                      "shock_bw2_const": shock_bw2_const,
                      "shock_bw1_alpha": shock_bw1_alpha,
                      "shock_bw2_alpha": shock_bw2_alpha,
                      "censor1_const": censor1_const,
                      "censor2_const": censor2_const,
                      "mean_valid": np.mean(mean_valid),
                      "cov_valid": np.mean(cov_valid),
                      "time": t2 - t1
                      })

    mean_names = ["EA", "EB", "EC"]
    cov_names = ["VarA", "VarB", "VarC", "CovAB", "CovAC", "CovBC"]

    config.update(zip(mean_names, mean_estimate))
    config.update(zip(cov_names, cov_estimate))

    result = pd.DataFrame(config, index=[abs(hash(str(config)))])
    return result


def fit_output_cond_means(X, Z, Y, bw: float, kernel: str):
    logging.info("--Fitting output conditional means--")
    XZ = np.column_stack([X, Z])
    cond_means = KernelRegression(kernel=kernel).fit_predict_local(XZ, Y, bw=bw)
    return cond_means


def fit_output_cond_cov(X, Z, Y, output_cond_means, bw: float, kernel: str, poly=1):
    n = len(X)
    XZ = np.column_stack([X, Z])
    resid = (Y - output_cond_means)
    Yt = np.empty((n, 6))
    Yt[:, :3] = resid ** 2  # Var[Y1|I], Var[Y2|I], Var[Y3|I]
    Yt[:, 3:] = np.column_stack([
        resid[:, 0] * resid[:, 1],  # Cov[Y1, Y2|I]
        resid[:, 0] * resid[:, 2],  # Cov[Y1, Y3|I]
        resid[:, 1] * resid[:, 2],  # Cov[Y2, Y3|I]
    ])
    XZp = PolynomialFeatures(degree=poly, include_bias=False).fit_transform(XZ)
    cond_cov = KernelRegression(kernel=kernel).fit_predict_local(XZp, Yt, bw=bw)
    return cond_cov


def fit_shock_means(X, Z, Y, bw: float, kernel: str):
    """
    Creates a 3-vector of shock means
    [E[Ut], E[Vt], E[Wt]]
    """
    shock_means = np.zeros((3, 3))
    n, _ = X.shape
    kreg = KernelRegression(kernel=kernel)
    for t in [1, 2]:
        DYt = difference(Y, t=t)
        DXZt = np.hstack(difference(X, Z, t=t))
        XZt = np.hstack(extract(X, Z, t=t))
        if kernel == "gaussian":
            wts = gaussian_kernel(DXZt, bw)
        elif kernel == "neighbor":
            wts = knn_kernel(DXZt, np.zeros_like(DXZt[[0]]), bw)
            wts /= wts.sum()
        else:
            raise ValueError(f"Cannot understand kernel {kernel}")
        shock_means[:, t] = kreg.fit(XZt, DYt, sample_weight=wts).coefficients
    return shock_means


def fit_shock_cov(X, Z, Y, shock_means, bw: float, kernel: str):
    shock_sec_mom = fit_shock_second_moments(X, Z, Y, bw, kernel)
    shock_cov = get_centered_shock_second_moments(shock_means, shock_sec_mom)
    return shock_cov


def fit_shock_second_moments(X, Z, Y, bw: float, kernel: str):
    """
    Creates 6-vector of shock second moments
    [E[Ut^2], E[Vt^2], E[Wt^2], E[Ut*Vt], E[Ut*Wt], E[Vt*Wt]]
    """
    n, _ = X.shape
    shock_sec_mom = np.zeros((6, 3))
    kreg = KernelRegression(kernel=kernel)
    for t in [1, 2]:
        DYt = difference(Y, t=t)
        DXZt = np.hstack(difference(X, Z, t=t))
        XZt = np.hstack(extract(X ** 2, Z ** 2, 2 * X, 2 * Z, 2 * X * Z, t=t))
        if kernel == "gaussian":
            wts = gaussian_kernel(DXZt, bw)
        elif kernel == "neighbor":
            wts = knn_kernel(DXZt, np.zeros_like(DXZt[[0]]), bw)
            wts /= wts.sum()
        else:
            raise ValueError(f"Cannot understand kernel {kernel}")
        shock_sec_mom[:, t] = kreg.fit(XZt, DYt ** 2, sample_weight=wts).coefficients
    return shock_sec_mom


def get_coef_cond_means(X, Z, output_cond_means, shock_means):
    """
    Fit random coefficient conditional means
    """
    # Construct conditional output minus excess terms:
    # E[Y|I] - E
    n, T = X.shape
    excess_terms = get_mean_excess_terms(X, Z, shock_means)
    output_cond_mean_clean = output_cond_means - excess_terms

    # Compute conditional first moments of random coefficients:
    # E[A,B|I] = M_{3}^{-1} * (E[Y|I] - E)
    coefficient_cond_means = np.full(shape=(n, T), fill_value=np.nan)
    for i in range(n):
        try:
            coefficient_cond_means[i] = m3_inv(X[i], Z[i]) @ output_cond_mean_clean[i]
        except np.linalg.LinAlgError:
            print(f"When computing conditional means, could not invert observation {i}")

    return coefficient_cond_means


def get_coef_cond_cov(X, Z, output_cond_cov, shock_cov):
    """
    Fit random coefficient conditional variances and covariances
    """
    # Construct Var[Y|X,Z] minus excess terms
    excess_terms = get_cov_excess_terms(X, Z, shock_cov)
    output_cond_cov_clean = output_cond_cov - excess_terms

    # Compute conditional second moments of random coefficients
    coefficient_cond_cov = np.empty(shape=(len(X), 6))
    for i in range(len(X)):
        try:
            coefficient_cond_cov[i] = m6_inv(X[i], Z[i]) @ output_cond_cov_clean[i]
        except np.linalg.LinAlgError:
            print(f"When computing conditional variances, could not invert observation {i}")

    return coefficient_cond_cov


#
def get_means_censor_threshold(n, const):
    return const * n ** (-1 / 4)


#
def get_cov_censor_threshold(n, const):
    return const * n ** (-1 / 8)


#
def get_valid_cond_means(X, Z, const):
    n = len(X)
    thres = get_means_censor_threshold(n, const)
    valid = np.zeros(n, np.bool_)
    for i in range(n):
        valid[i] = np.abs(det(m3(X[i], Z[i]))) > thres
    return valid


#
def get_valid_cond_cov(X, Z, const):
    n = len(X)
    thres = get_means_censor_threshold(n, const)
    valid = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        valid[i] = np.abs(det(m6(X[i], Z[i]))) > thres
    return valid


#
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


#
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

    E3 = VarU2 + VarV2 * X3 ** 2 + VarW2 * Z3 ** 2 \
         + 2 * CovU2V2 * X3 + 2 * CovU2W2 * Z3 + 2 * CovV2W2 * X3 * Z3 \
         + VarU3 + VarV3 * X3 ** 2 + VarW3 * Z3 ** 2 \
         + 2 * CovU3V3 * X3 + 2 * CovU3W3 * Z3 + 2 * CovV3W3 * X3 * Z3

    E4 = np.zeros(n)

    E5 = np.zeros(n)

    E6 = VarU2 + VarV2 * X2 * X3 + VarW2 * Z2 * Z3 \
         + CovU2V2 * (X2 + X3) + CovU2W2 * (Z2 + Z3) \
         + CovV2W2 * (X2 * Z3 + Z2 * X3)

    excess_terms = np.column_stack((E1, E2, E3, E4, E5, E6))
    return excess_terms


#
def m3(x, z):
    """
    This is now called matrix M3 in the paper
    """
    return np.column_stack((np.ones_like(x, dtype=np.float64), x, z))


#
def m3_inv(x, z):
    return np.linalg.inv(m3(x, z))


#
def m6(x, z):
    """
    Computes matrix M_6 in paper.
    """

    def f(i, j):
        return [1,
                x[i] * x[j],
                z[i] * z[j],
                x[i] + x[j],
                z[i] + z[j],
                x[i] * z[j] + x[j] * z[i]]

    g = np.array([f(0, 0), f(1, 1), f(2, 2), f(0, 1), f(0, 2), f(1, 2)])
    return g


#
def m6_inv(x, z):
    """
    Inverse of matrix m6 in paper.
    """
    return np.linalg.inv(m6(x, z))


def get_coef_means(coef_cond_means, valid):
    """
    Computes means, removing 'invalid' entries.
    """
    return coef_cond_means[valid].mean(0)


def get_coef_cov(coef_cond_means, coef_cond_cov, valid):
    """
    Use ANOVA-type formulas to get unconditional variances and covariances
    Var[A] = Var[E[A|X, Z]] + E[V[A|X, Z]]
    """
    ev = coef_cond_cov[valid, :3].mean(0)
    ve = coef_cond_means[valid].var(0)
    variances = ev + ve

    ec = coef_cond_cov[valid, 3:].mean(0)
    ce = np.cov(coef_cond_means[valid].T)[[0, 0, 1], [1, 2, 2]]
    covariances = ec + ce

    coefficient_cov = np.hstack([variances, covariances])
    return coefficient_cov


def get_centered_shock_second_moments(m1, m2):
    """
    Creates 6-vector of shock variances and covariances from uncentered moments.
    [Var[Ut], Var[Vt], Var[Wt], Cov[Ut, Vt], Cov[Ut, Wt], Cov[Vt, Wt]]
    """
    VarUt = m2[0] - m1[0] ** 2
    VarVt = m2[1] - m1[1] ** 2
    VarWt = m2[2] - m1[2] ** 2
    CovUVt = m2[3] - m1[0] * m1[1]
    CovUWt = m2[4] - m1[0] * m1[1]
    CovVWt = m2[5] - m1[1] * m1[2]
    return np.array([VarUt, VarVt, VarWt, CovUVt, CovUWt, CovVWt])
