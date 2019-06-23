import numpy.testing as nt

from fhhps.estimator import *


def test_shock_means():
    n = 5000000
    fake = generate_data(n)
    data = fake["df"]
    truth1 = fake["means"][["U2", "V2", "W2"]]
    truth2 = fake["means"][["U3", "V3", "W3"]]
    X = data[["X1", "X2", "X3"]].values
    Z = data[["Z1", "Z2", "Z3"]].values
    Y = data[["Y1", "Y2", "Y3"]].values

    bw = 5 * n ** (-1 / 5)  # '5' is optimal constant for this (n, alpha)
    m1 = fit_shock_means(X, Z, Y, t=1, bw=bw)
    m2 = fit_shock_means(X, Z, Y, t=2, bw=bw)
    nt.assert_array_almost_equal(m1, truth1, decimal=2)
    nt.assert_array_almost_equal(m2, truth2, decimal=2)


def test_shock_second_moments():
    n = 5000000
    fake = generate_data(n)
    data = fake["df"]
    truth1 = np.hstack([fake["variances"][["U2", "V2", "W2"]],
                        fake["cov"].loc["U2", "V2"],
                        fake["cov"].loc["U2", "W2"],
                        fake["cov"].loc["V2", "W2"]])
    truth2 = np.hstack([fake["variances"][["U3", "V3", "W3"]],
                        fake["cov"].loc["U3", "V3"],
                        fake["cov"].loc["U3", "W3"],
                        fake["cov"].loc["V3", "W3"]])
    X = data[["X1", "X2", "X3"]].values
    Z = data[["Z1", "Z2", "Z3"]].values
    Y = data[["Y1", "Y2", "Y3"]].values

    bw = 5. * n ** (-1 / 5)  # '5' is optimal constant for this (n, alpha)
    m1 = fit_shock_means(X, Z, Y, t=1, bw=bw)
    m2 = fit_shock_means(X, Z, Y, t=2, bw=bw)
    s1 = fit_shock_second_moments(X, Z, Y, 1, bw=bw)
    s2 = fit_shock_second_moments(X, Z, Y, 2, bw=bw)
    v1 = get_centered_shock_second_moments(m1, s1)
    v2 = get_centered_shock_second_moments(m2, s2)

    nt.assert_array_almost_equal(v1, truth1, decimal=2)
    nt.assert_array_almost_equal(v2, truth2, decimal=2)
