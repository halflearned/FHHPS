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


def test_fit_output_cond_means():
    """
    Checks that we are able to estimate first conditional
    moments of the output at a reasonable level: avg error is not
    statistically different from zero at alpha=0.01
    """
    np.random.seed(1234)
    coef_bw = .05
    errors = []
    for i in range(40):
        fake = generate_data(1000)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        truth = get_true_output_cond_means(fake)
        estimate = fit_output_cond_means(X, Z, Y, bw=coef_bw)
        error = estimate - truth
        errors.append(error.mean(0))

    stats = pd.DataFrame(errors).agg(["mean", "sem"])
    unbiased = np.abs(stats.loc["mean"]) / stats.loc["sem"] < 2.326
    assert np.all(unbiased)


def test_fit_output_cond_cov_with_oracle_means():
    """
    Checks that if we pass it the **true** conditional means, then
    we are able to estimate second conditional
    moments of the output at a reasonable level: avg error is not
    statistically different from zero at alpha=0.01
    """
    np.random.seed(1234)
    coef_bw = .75
    errors = []
    for i in range(40):
        fake = generate_data(1000)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        output_cond_means = get_true_output_cond_means(fake)
        estimate = fit_output_cond_cov(X, Z, Y, output_cond_means, bw=coef_bw)
        truth = get_true_output_cond_cov(fake)
        error = estimate - truth
        errors.append(error.mean(0))

    stats = pd.DataFrame(errors).agg(["mean", "sem"])
    unbiased = np.abs(stats.loc["mean"]) / stats.loc["sem"] < 2.326
    assert np.all(unbiased)


def test_fit_output_cond_cov_with_estimated_means():
    """
    Checks that if we pass it the **estimated** conditional means, then
    we are able to estimate second conditional
    moments of the output at a reasonable level: avg error is not
    statistically different from zero at alpha=0.01
    """
    np.random.seed(1234)
    errors = []
    for i in range(40):
        fake = generate_data(1000)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        output_cond_means = fit_output_cond_means(X, Z, Y, bw=.75)
        estimate = fit_output_cond_cov(X, Z, Y, output_cond_means, bw=.5)
        truth = get_true_output_cond_cov(fake)
        error = estimate - truth
        errors.append(error.mean(0))
        print(error.mean(0))

    stats = pd.DataFrame(errors).agg(["mean", "sem"])
    unbiased = np.abs(stats.loc["mean"]) / stats.loc["sem"] < 2.326
    assert np.all(unbiased)
